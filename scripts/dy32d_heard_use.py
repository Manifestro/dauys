"""DY-32 track D: does the model use the CONTENT of what it heard? Likelihood test, no training.

Held-out shards only (dy21_train.held_out). For each window: own = 10 s of a piece, heard = H s of audio under three
conditions, cross-entropy of the TRUE own units under each:
  a  true heard: the H s right before the own part (same piece);
  b  same episode, other time: H s from the same episode (pieces.jsonl "ep", episode times start/end) whose interval
     is at least 60 s away from the whole [heard, own] window (same voices, different content);
  c  another episode (different recording) of the held-out shards.
WavLM features are computed separately on the heard audio and on the own audio in every condition (so a, b, c are
processed identically); own unit ids come from the own audio alone. Both heard modes (0 continuous, 1 discrete),
heard lengths 15 s and 5 s (the last 5 s of the same chunks). CE per own frame is averaged in buckets 0-1 s, 1-3 s,
3-10 s. Reported: mean CE per condition, share of windows with CE(a) < CE(b) and CE(a) < CE(c) (95 % Wilson CI),
mean paired gaps b-a and c-a (95 % CI), and the cont-vs-disc paired difference of the b-a gap.

    python scripts/dy32d_heard_use.py --held /opt/dlami/nvme/heldaudio --run /data/dy32d/run \
        --centroids /data/dy32d/acb/km-corpus-K2000.npy --n 1200 --out /data/dy32d/heard_use.json
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from dy21_train import held_out
from dy32d import Codebook, load_wavlm, n_samples, wavlm_feats
from dauys.dy32_heard import SR, HeardConfig, HeardLM

OWN = 500  # frames (10 s)
GAP_S = 60.0
BUCKETS = {"0-1s": (0, 50), "1-3s": (50, 150), "3-10s": (150, 500)}


def wilson(k: int, n: int) -> list[float]:
    if n == 0:
        return [float("nan")] * 2
    p, z = k / n, 1.96
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 4), round(c + h, 4)]


def mean_ci(x: np.ndarray) -> dict:
    m, s = float(x.mean()), float(x.std(ddof=1) / math.sqrt(len(x)))
    return {"mean": round(m, 4), "ci95": [round(m - 1.96 * s, 4), round(m + 1.96 * s, 4)]}


def build_windows(held: str, n: int, seed: int, Hmax: int):
    """The test windows (deterministic in seed): returns pieces, wins, by_ep, shards, hs, os_."""
    rng = np.random.default_rng(seed)
    hs = n_samples(Hmax * 50)
    os_ = n_samples(OWN)
    # pieces of held-out shards with episode times
    pieces = []  # (segs, start_sample, len, ep, ep_t0)
    for d in sorted(Path(held).iterdir()):
        if not held_out(d.name) or not (d / "pieces.jsonl").exists():
            continue
        segs = np.memmap(d / "segs.bin", dtype=np.int16, mode="r")
        idx = np.load(d / "index.npy")
        meta = [json.loads(l) for l in open(d / "pieces.jsonl")]
        assert len(meta) == len(idx)
        for (st, ln), m in zip(idx, meta):
            pieces.append((segs, int(st), int(ln), f"{d.name}/{m['ep']}", float(m["start"])))
    by_ep: dict[str, list[int]] = {}
    for i, pc in enumerate(pieces):
        by_ep.setdefault(pc[3], []).append(i)
    shards = sorted({pc[3].split("/")[0] for pc in pieces})

    # windows: a piece long enough for heard + own; (b) a chunk of the same episode >= 60 s away; (c) other episode
    wins = []
    tries = 0
    while len(wins) < n and tries < 100000:
        tries += 1
        i = int(rng.integers(len(pieces)))
        _, _, ln, ep, t0 = pieces[i]
        if ln < hs + os_:
            continue
        off = int(rng.integers(0, ln - hs - os_ + 1))
        w0, w1 = t0 + off / SR, t0 + (off + hs + os_) / SR  # episode time of the whole window
        cand = []
        for j in by_ep[ep]:
            _, _, lj, _, tj = pieces[j]
            if lj < hs:
                continue
            # chunk start s (seconds in the episode) with [s, s + H] at least GAP_S away from [w0, w1]
            lo, hi = tj, tj + (lj - hs) / SR
            if hi >= w1 + GAP_S:
                cand.append((j, max(lo, w1 + GAP_S), hi))
            if lo + Hmax <= w0 - GAP_S:
                cand.append((j, lo, min(hi, w0 - GAP_S - Hmax)))
        cand = [c for c in cand if c[2] >= c[1]]
        if not cand:
            continue
        j, lo, hi = cand[int(rng.integers(len(cand)))]
        offb = int(round((rng.uniform(lo, hi) - pieces[j][4]) * SR))
        offb = min(max(offb, 0), pieces[j][2] - hs)
        others = [k for k in range(len(pieces)) if pieces[k][3] != ep and pieces[k][2] >= hs]
        k = others[int(rng.integers(len(others)))]
        offc = int(rng.integers(0, pieces[k][2] - hs + 1))
        wins.append((i, off, j, offb, k, offc))
    print(json.dumps({"windows": len(wins), "tries": tries, "pieces": len(pieces), "episodes": len(by_ep),
                      "shards": shards}), flush=True)

    return pieces, wins, by_ep, shards, hs, os_


def read_chunk(pieces, i: int, off: int, n: int) -> np.ndarray:
    segs, st, ln, _, _ = pieces[i]
    return np.asarray(segs[st + off : st + off + n], np.float32) / 32768.0


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--held", required=True)
    p.add_argument("--run", required=True)
    p.add_argument("--centroids", required=True)
    p.add_argument("--n", type=int, default=1200)
    p.add_argument("--heard", default="15,5")
    p.add_argument("--batch", type=int, default=24)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    device = torch.device("cuda")
    Hmax = max(int(x) for x in a.heard.split(","))

    pieces, wins, by_ep, shards, hs, os_ = build_windows(a.held, a.n, a.seed, Hmax)

    def chunk(i, off, n):
        return read_chunk(pieces, i, off, n)

    ck = torch.load(Path(a.run) / "best.pt", map_location="cpu")
    model = HeardLM(HeardConfig(**ck["cfg"])).to(device).eval()
    model.load_state_dict(ck["model"])
    cb = Codebook(a.centroids, "cos", device)
    wavlm = load_wavlm(device)

    @torch.no_grad()
    def ce_own(hf: torch.Tensor, hid: torch.Tensor, oid: torch.Tensor, mode: int) -> np.ndarray:
        """heard feats [B, P, D], heard ids [B, P], own ids [B, OWN] -> CE per own frame [B, OWN]."""
        B, P = hid.shape
        ids = torch.cat([hid, oid], 1)
        feats = torch.cat([hf, torch.zeros(B, OWN, hf.shape[2], device=device, dtype=hf.dtype)], 1)
        T = P + OWN
        x_ids, x_f = ids[:, : T - 1], feats[:, : T - 1]
        heard = (torch.arange(T - 1, device=device) < P)[None].expand(B, -1)
        m = torch.full((B,), mode, device=device, dtype=torch.long)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h = model.states(model.embed(x_ids, x_f, heard, m))
        lg = model.logits(h[:, P - 1 :])
        return F.cross_entropy(lg.flatten(0, 1), oid.flatten(), reduction="none").view(B, OWN).cpu().numpy()

    Hs = [int(x) for x in a.heard.split(",")]
    res = {(H, mode, c): [] for H in Hs for mode in (0, 1) for c in "abc"}
    for b0 in range(0, len(wins), a.batch):
        W = wins[b0 : b0 + a.batch]
        own = np.stack([chunk(i, off + hs, os_) for i, off, *_ in W])
        ha = np.stack([chunk(i, off, hs) for i, off, *_ in W])
        hb = np.stack([chunk(j, offb, hs) for _, _, j, offb, _, _ in W])
        hc = np.stack([chunk(k, offc, hs) for *_, k, offc in W])
        oid = cb.ids(wavlm_feats(wavlm, torch.from_numpy(own).to(device)))
        for c, aud in (("a", ha), ("b", hb), ("c", hc)):
            f = wavlm_feats(wavlm, torch.from_numpy(aud).to(device))  # Hmax s
            hid = cb.ids(f)
            for H in Hs:
                fh, ih = f[:, -H * 50 :], hid[:, -H * 50 :]
                if H != Hmax:  # shorter heard part: WavLM on the last H s alone, as for the full length
                    fh = wavlm_feats(wavlm, torch.from_numpy(aud[:, -n_samples(H * 50) :]).to(device))
                    ih = cb.ids(fh)
                for mode in (0, 1):
                    res[(H, mode, c)].append(ce_own(fh, ih, oid, mode))
        print(json.dumps({"done": b0 + len(W)}), flush=True)
    res = {k: np.concatenate(v) for k, v in res.items()}

    out = {"windows": len(wins), "episodes": len(by_ep), "held_out_shards": shards, "checkpoint_step": ck["step"],
           "codebook": a.centroids, "own_s": 10, "gap_b_s": GAP_S, "seed": a.seed,
           "notes": "WavLM (fp16) run separately on each heard chunk and on the own chunk in every condition; own ids "
                    "from the own audio alone. (b) = same episode (pieces.jsonl ep), chunk >= 60 s away from the whole "
                    "heard+own window; (c) = a different episode of the held-out shards (may be the same show).",
           "results": {}}
    for H in Hs:
        for mode, mn in ((0, "cont"), (1, "disc")):
            r = {}
            for bk, (s, e) in BUCKETS.items():
                ce = {c: res[(H, mode, c)][:, s:e].mean(1) for c in "abc"}
                n = len(ce["a"])
                r[bk] = {"ce_a": round(float(ce["a"].mean()), 4), "ce_b": round(float(ce["b"].mean()), 4),
                         "ce_c": round(float(ce["c"].mean()), 4),
                         "share_a_lt_b": round(float((ce["a"] < ce["b"]).mean()), 4),
                         "share_a_lt_b_ci95": wilson(int((ce["a"] < ce["b"]).sum()), n),
                         "share_a_lt_c": round(float((ce["a"] < ce["c"]).mean()), 4),
                         "share_a_lt_c_ci95": wilson(int((ce["a"] < ce["c"]).sum()), n),
                         "gap_b_minus_a": mean_ci(ce["b"] - ce["a"]), "gap_c_minus_a": mean_ci(ce["c"] - ce["a"])}
            out["results"][f"heard{H}s_{mn}"] = r
        cmp = {}
        for bk, (s, e) in BUCKETS.items():
            g = {m: res[(H, m, "b")][:, s:e].mean(1) - res[(H, m, "a")][:, s:e].mean(1) for m in (0, 1)}
            gc = {m: res[(H, m, "c")][:, s:e].mean(1) - res[(H, m, "a")][:, s:e].mean(1) for m in (0, 1)}
            cmp[bk] = {"b_minus_a_cont_minus_disc": mean_ci(g[0] - g[1]), "c_minus_a_cont_minus_disc": mean_ci(gc[0] - gc[1])}
        out["results"][f"heard{H}s_cont_vs_disc"] = cmp
    Path(a.out).write_text(json.dumps(out, indent=2))

    lines = ["# DY-32 track D: does the model use the content of what it heard? (likelihood, held-out)", "",
             f"best.pt step {ck['step']}; {len(wins)} windows from {len(by_ep)} episodes of held-out shards "
             f"{', '.join(shards)}; own = true next 10 s; CE (nats per 50 Hz unit, K=2000) of the true own units.",
             "a = true heard part; b = same episode, >= 60 s away; c = another episode. Shares: windows with CE(a) lower.", "",
             "| heard | mode | bucket | CE a | CE b | CE c | b-a gap [95% CI] | share a<b [CI] | share a<c [CI] |",
             "|---|---|---|---|---|---|---|---|---|"]
    for H in Hs:
        for mn in ("cont", "disc"):
            for bk, v in out["results"][f"heard{H}s_{mn}"].items():
                g = v["gap_b_minus_a"]
                lines.append(f"| {H} s | {mn} | {bk} | {v['ce_a']:.3f} | {v['ce_b']:.3f} | {v['ce_c']:.3f} | "
                             f"{g['mean']:+.3f} [{g['ci95'][0]:+.3f}, {g['ci95'][1]:+.3f}] | {v['share_a_lt_b']:.3f} "
                             f"[{v['share_a_lt_b_ci95'][0]:.3f}, {v['share_a_lt_b_ci95'][1]:.3f}] | {v['share_a_lt_c']:.3f} "
                             f"[{v['share_a_lt_c_ci95'][0]:.3f}, {v['share_a_lt_c_ci95'][1]:.3f}] |")
    lines += ["", "Cont minus disc, paired per window, of the b-a gap (positive = continuous hearing gains more from the "
              "true content):", "", "| heard | bucket | (b-a)cont - (b-a)disc [95% CI] | (c-a)cont - (c-a)disc [95% CI] |",
              "|---|---|---|---|"]
    for H in Hs:
        for bk, v in out["results"][f"heard{H}s_cont_vs_disc"].items():
            x, y = v["b_minus_a_cont_minus_disc"], v["c_minus_a_cont_minus_disc"]
            lines.append(f"| {H} s | {bk} | {x['mean']:+.4f} [{x['ci95'][0]:+.4f}, {x['ci95'][1]:+.4f}] | "
                         f"{y['mean']:+.4f} [{y['ci95'][0]:+.4f}, {y['ci95'][1]:+.4f}] |")
    lines += ["", "30 s heard was not run: 30 s + 10 s = 2000 frames exceeds the 1280-frame training context.",
              "", out["notes"]]
    Path(a.out).with_suffix(".md").write_text("\n".join(lines) + "\n")
    print("HEARD USE DONE", flush=True)


if __name__ == "__main__":
    main()
