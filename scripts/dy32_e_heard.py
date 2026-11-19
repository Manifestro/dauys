"""DY-32 track E: does GSLM2 use the content of what it heard? Likelihood test on held-out DuplexChat (as track D).

Windows: one per piece >= 25 s of the held-out shards (seeded); own = 10 s right after a 15 s heard part.
Heard conditions: a = true (just before own), b = same episode (pieces.jsonl ep), chunk >= 60 s (episode time) away from
the whole window, c = another episode. Heard 15 s and 5 s (last 5 s of each 15 s chunk, encoded on its own).
Heard and own audio are encoded SEPARATELY with SpidR 2.0 and cross-stream deduplicated separately (generation path).
Loss on the true own units: (G) Depformer generation CE = mean over the 6 streams of the teacher-forced CE of each
stream given context and the previously generated streams in the model's generation order (the sampler's likelihood,
the `gen_loss` term); (H) per-stream head CE on the backbone output (mean over 6 streams; the multi-prediction term).
Nats per deduplicated position. Positions mapped to time through the kept full-rate frame index (50 Hz).

    python scripts/dy32_e_heard.py --shards s00197,lbig-00550 --lms A,B --n 600 --out heard.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from dy32_gslm2 import OFFSET, SR, SRC, Encoder, dedup, held_out, load_lm, sh

BUCKETS = [("0-1s", 0, 1), ("1-3s", 1, 3), ("3-10s", 3, 10), ("0-10s", 0, 10)]


def kept(codes: np.ndarray):
    keep = np.ones(codes.shape[1], bool)
    keep[1:] = (codes[:, 1:] != codes[:, :-1]).any(0)
    return codes[:, keep], np.nonzero(keep)[0] / 50.0


def build(shards, n, work: Path, seed=1234):
    audio, meta = {}, []
    for s in shards:
        assert held_out(s), s
        d = work / s
        d.mkdir(parents=True, exist_ok=True)
        for fn in ("index.npy", "segs.bin", "pieces.jsonl"):
            if not (d / fn).exists():
                sh(f"aws s3 cp {SRC}/{s}/{fn} {d}/{fn} --only-show-errors")
        idx = np.load(d / "index.npy")
        segs = np.memmap(d / "segs.bin", dtype=np.int16, mode="r")
        pj = [json.loads(l) for l in (d / "pieces.jsonl").read_text().splitlines()]
        assert len(pj) == len(idx), (s, len(pj), len(idx))
        for k, ((st, ln), p) in enumerate(zip(idx, pj)):
            audio[(s, k)] = (segs, int(st), int(ln))
            meta.append({"key": (s, k), "ep": p["ep"], "start": float(p["start"]), "len": ln / SR})
    rng = np.random.default_rng(seed)
    long = [m for m in meta if m["len"] >= 25]
    rng.shuffle(long)
    by_ep = {}
    for m in meta:
        if m["len"] >= 15:
            by_ep.setdefault(m["ep"], []).append(m)
    eps = list(by_ep)
    wins = []
    for m in long:
        t = float(rng.uniform(15, m["len"] - 10))  # own start inside the piece (s)
        w0, w1 = m["start"] + t - 15, m["start"] + t + 10  # window in episode time
        far = [o for o in by_ep[m["ep"]] if o["start"] + o["len"] <= w0 - 60 or o["start"] >= w1 + 60]
        if not far:
            continue
        ob = far[rng.integers(len(far))]
        tb = float(rng.uniform(0, ob["len"] - 15))
        if ob["start"] + tb + 15 > w0 - 60 and ob["start"] + tb < w1 + 60:
            continue
        others = [e for e in eps if e != m["ep"]]
        oc_ep = others[rng.integers(len(others))]
        oc = by_ep[oc_ep][rng.integers(len(by_ep[oc_ep]))]
        tc = float(rng.uniform(0, oc["len"] - 15))
        wins.append({"piece": m["key"], "t": t, "b": (ob["key"], tb), "c": (oc["key"], tc), "ep": m["ep"]})
        if len(wins) >= n:
            break

    def cut(key, t0, dur):
        segs, st, ln = audio[key]
        a = int(round(t0 * SR))
        return np.asarray(segs[st + a : st + a + int(dur * SR)], dtype=np.float32) / 32768.0

    chunks = []
    for w in wins:
        own = cut(w["piece"], w["t"], 10)
        ha = cut(w["piece"], w["t"] - 15, 15)
        hb = cut(w["b"][0], w["b"][1], 15)
        hc = cut(w["c"][0], w["c"][1], 15)
        chunks.append({"own": own, "a15": ha, "b15": hb, "c15": hc, "a5": ha[-5 * SR :], "b5": hb[-5 * SR :],
                       "c5": hc[-5 * SR :]})
    return wins, chunks


@torch.no_grad()
def score(loaded, seqs, own_starts, device, micro):
    """seqs: list of (6, L) LM ids. Returns per sequence (G, H) CE arrays over the own positions."""
    from fairseq2.nn import BatchLayout

    m = loaded.model
    out = []
    order = sorted(range(len(seqs)), key=lambda i: seqs[i].shape[1])
    res = [None] * len(seqs)
    for b in range(0, len(order), micro):
        ids = order[b : b + micro]
        L = max(seqs[i].shape[1] for i in ids)
        x = np.full((len(ids), 6, L), 1, np.int64)
        for j, i in enumerate(ids):
            x[j, :, : seqs[i].shape[1]] = seqs[i]
        x = torch.from_numpy(x).to(device)
        B = len(ids)
        flat = x.reshape(B * 6, L)
        lay = BatchLayout(shape=flat.shape, seq_lens=[L] * (B * 6), packed=False, device=device)
        h = m.forward(flat, lay, return_decoder_outputs=True)  # (B, L, D)
        tg = x[:, :, 1:].permute(1, 0, 2).contiguous()  # (6, B, L-1)
        lay2 = BatchLayout(shape=(B, L - 1), seq_lens=[L - 1] * B, packed=False, device=device)
        dl = m.forward_depformer(h[:, :-1], tg, lay2)
        G = torch.stack([torch.nn.functional.cross_entropy(dl[s].float().transpose(1, 2), tg[s], reduction="none")
                         for s in range(6)]).mean(0)  # (B, L-1)
        H = torch.stack([torch.nn.functional.cross_entropy(m.final_proj[s](h[:, :-1]).float().transpose(1, 2), tg[s],
                                                           reduction="none") for s in range(6)]).mean(0)
        G, H = G.cpu().numpy(), H.cpu().numpy()
        for j, i in enumerate(ids):
            o, n = own_starts[i], seqs[i].shape[1]
            res[i] = (G[j, o - 1 : n - 1], H[j, o - 1 : n - 1])
    return res


def ci(x):
    x = np.asarray(x, float)
    return float(x.mean()), float(1.96 * x.std(ddof=1) / np.sqrt(len(x)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shards", default="s00197,lbig-00550")
    p.add_argument("--lms", required=True)
    p.add_argument("--names", required=True)
    p.add_argument("--n", default="600", help="windows per LM, comma list")
    p.add_argument("--micro", default="16", help="comma list")
    p.add_argument("--work", default="/opt/dlami/nvme/heard")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    device = torch.device("cuda")
    ns = [int(v) for v in a.n.split(",")]
    wins, chunks = build(a.shards.split(","), max(ns), Path(a.work))
    print(json.dumps({"windows": len(wins), "episodes": len({w["ep"] for w in wins})}), flush=True)
    enc = Encoder("/opt/dlami/nvme/ckpt/spidr-2.0/final.pt", device)
    units = []
    for i in range(0, len(chunks), 16):
        part = chunks[i : i + 16]
        u = {}
        for key in part[0]:
            u[key] = enc.codes([c[key] for c in part])
        for j in range(len(part)):
            units.append({k: kept(u[k][j].astype(np.int64)) for k in u})
    del enc
    torch.cuda.empty_cache()
    results = {}
    for lm, name, n, micro in zip(a.lms.split(","), a.names.split(","), ns, [int(v) for v in a.micro.split(",")]):
        loaded = load_lm(lm, device)
        loaded.model.eval()
        seqs, starts, tags = [], [], []
        for wi in range(min(n, len(units))):
            own, t_own = units[wi]["own"]
            for hl in ("15", "5"):
                for c in "abc":
                    hd, _ = units[wi][c + hl]
                    seqs.append(np.concatenate([hd, own], 1) + OFFSET)
                    starts.append(hd.shape[1])
                    tags.append((wi, hl, c))
        sc = score(loaded, seqs, starts, device, micro)
        rows = {}
        for (wi, hl, c), (g, h) in zip(tags, sc):
            t_own = units[wi]["own"][1]
            for bname, lo, hi in BUCKETS:
                msk = (t_own >= lo) & (t_own < hi)
                for loss, arr in (("G", g), ("H", h)):
                    rows.setdefault((hl, bname, loss), {}).setdefault(c, {})[wi] = float(arr[msk].mean()) if msk.any() else np.nan
        tab = []
        for (hl, bname, loss), d in rows.items():
            w = [i for i in d["a"] if not any(np.isnan(d[c][i]) for c in "abc")]
            A, Bv, C = (np.array([d[c][i] for i in w]) for c in "abc")
            tab.append({"heard_s": int(hl), "bucket": bname, "loss": loss, "n": len(w), "a": A.mean(), "b": Bv.mean(),
                        "c": C.mean(), "b_a": ci(Bv - A), "c_a": ci(C - A), "share_a_lt_b": ci(A < Bv),
                        "share_a_lt_c": ci(A < C)})
        results[name] = {"lm": lm, "windows": min(n, len(units)), "table": tab}
        print(json.dumps({"model": name, "done": True}), flush=True)
        del loaded
        torch.cuda.empty_cache()
        Path(a.out).write_text(json.dumps(results, indent=1, default=float))
    Path(a.out).write_text(json.dumps(results, indent=1, default=float))
    print("HEARD DONE", flush=True)


if __name__ == "__main__":
    main()
