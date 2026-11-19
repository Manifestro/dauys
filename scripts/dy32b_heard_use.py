"""DY-32 track B: likelihood test of heard content on the UnitLM (no training).

Held-out shards only. Window = heard H s + own 10 s of one piece. Heard part under three conditions:
  a  true: the H s right before the own part;  b  same episode (pieces.jsonl "ep"), a chunk whose interval is >= 60 s
  away from the whole window;  c  a different episode of the held-out shards.
WavLM runs separately on each heard chunk (each length H separately: the last H s of the chunk) and on the own chunk,
as at generation. Sequence = EOS + heard codes + own codes; CE of the true own codes (499 frames) per position,
averaged in buckets 0-1 s, 1-3 s, 3-10 s. Also: share of frames, mean frame energy for given codes on own audio.

    python scripts/dy32b_heard_use.py --held /opt/dlami/nvme/held --ckpt /data/b/run/best.pt \
        --cb /data/b/cbA/km-corpus-K2000.npy --n 1000 --out /data/b/heard_use.json
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
from dy32b_channel import SR, Channel, Codebook

from dauys.dy21_lm import LMConfig, UnitLM

OWN_S, GAP_S = 10, 60.0
BUCKETS = {"0-1s": (0, 50), "1-3s": (50, 150), "3-10s": (150, 499)}


def wilson(k: int, n: int) -> list[float]:
    p, z = k / n, 1.96
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 4), round(c + h, 4)]


def mean_ci(x: np.ndarray) -> dict:
    m, s = float(x.mean()), float(x.std(ddof=1) / math.sqrt(len(x)))
    return {"mean": round(m, 4), "ci95": [round(m - 1.96 * s, 4), round(m + 1.96 * s, 4)]}


def windows(held: str, n: int, seed: int, hmax: int):
    rng = np.random.default_rng(seed)
    hs, os_ = hmax * SR, OWN_S * SR
    pieces = []
    for d in sorted(Path(held).iterdir()):
        if not held_out(d.name):
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
    long_ = [k for k in range(len(pieces)) if pieces[k][2] >= hs]
    wins, tries = [], 0
    while len(wins) < n and tries < 200000:
        tries += 1
        i = int(rng.integers(len(pieces)))
        _, _, ln, ep, t0 = pieces[i]
        if ln < hs + os_:
            continue
        off = int(rng.integers(0, ln - hs - os_ + 1))
        w0, w1 = t0 + off / SR, t0 + (off + hs + os_) / SR
        cand = []
        for j in by_ep[ep]:
            _, _, lj, _, tj = pieces[j]
            if lj < hs:
                continue
            lo, hi = tj, tj + (lj - hs) / SR  # chunk start times in the episode
            a, b = max(lo, w1 + GAP_S), hi
            if b >= a:
                cand.append((j, a, b))
            a, b = lo, min(hi, w0 - GAP_S - hmax)
            if b >= a:
                cand.append((j, a, b))
        if not cand:
            continue
        j, lo, hi = cand[int(rng.integers(len(cand)))]
        offb = min(max(int(round((rng.uniform(lo, hi) - pieces[j][4]) * SR)), 0), pieces[j][2] - hs)
        others = [k for k in long_ if pieces[k][3] != ep]
        k = others[int(rng.integers(len(others)))]
        offc = int(rng.integers(0, pieces[k][2] - hs + 1))
        wins.append((i, off, j, offb, k, offc))
    info = {"windows": len(wins), "tries": tries, "pieces": len(pieces), "episodes": len(by_ep)}
    print(json.dumps(info), flush=True)
    return pieces, wins, info


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--held", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--cb", required=True)
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--heard", default="30,15,5")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--codes", default="522,597,866")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    dev = torch.device("cuda")
    Hs = [int(x) for x in a.heard.split(",")]
    hmax = max(Hs)
    pieces, wins, winfo = windows(a.held, a.n, a.seed, hmax)

    def chunk(i, off, n):
        segs, st, _, _, _ = pieces[i]
        return np.asarray(segs[st + off : st + off + n], np.float32) / 32768.0

    ch, cb = Channel(), Codebook(a.cb)
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cfg = LMConfig(**ck["cfg"])
    m = UnitLM(cfg).to(dev).eval()
    m.load_state_dict(ck["model"])
    eos = cfg.vocab - 1

    @torch.no_grad()
    def ce(heard: list[torch.Tensor], own: list[torch.Tensor]) -> np.ndarray:
        x = torch.stack([torch.cat([torch.tensor([eos], device=dev), h, o]) for h, o in zip(heard, own)]).long()
        P = 1 + len(heard[0])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h = m.states(x[:, :-1])
        lg = m.logits(h[:, P - 1 :])
        tgt = x[:, P:]
        return F.cross_entropy(lg.flatten(0, 1), tgt.flatten(), reduction="none").view(tgt.shape).cpu().numpy()

    res = {(H, c): [] for H in Hs for c in "abc"}
    codes = [int(c) for c in a.codes.split(",")]
    cnt, en_sum, n_frames, en_all = np.zeros(cb.K), np.zeros(cb.K), 0, 0.0
    hs, os_ = hmax * SR, OWN_S * SR
    for b0 in range(0, len(wins), a.batch):
        W = wins[b0 : b0 + a.batch]
        own_a = [chunk(i, off + hs, os_) for i, off, *_ in W]
        oid = [cb.encode(f) for f in ch.feats(own_a)]
        for w, u in zip(own_a, oid):  # frame energy (dB) of the 400-sample frame behind each code
            u = u.cpu().numpy()
            fr = np.lib.stride_tricks.sliding_window_view(w, 400)[:: 320][: len(u)]
            e = 10 * np.log10((fr ** 2).mean(1) + 1e-10)
            np.add.at(cnt, u, 1)
            np.add.at(en_sum, u, e)
            n_frames += len(u)
            en_all += e.sum()
        heard = {"a": [chunk(i, off, hs) for i, off, *_ in W],
                 "b": [chunk(j, offb, hs) for _, _, j, offb, _, _ in W],
                 "c": [chunk(k, offc, hs) for *_, k, offc in W]}
        for c, aud in heard.items():
            for H in Hs:
                hid = [cb.encode(f) for f in ch.feats([x[-H * SR :] for x in aud])]
                res[(H, c)].append(ce(hid, oid))
        print(json.dumps({"done": b0 + len(W)}), flush=True)
    res = {k: np.concatenate(v) for k, v in res.items()}
    out = {**winfo, "checkpoint_step": ck["step"], "val": ck["val"], "codebook": a.cb, "own_frames": 499, "seed": a.seed,
           "notes": "sequence = EOS + heard codes + own codes; WavLM separately on each heard chunk (last H s alone) and "
                    "on the own chunk; heard 30 s encoded in one WavLM pass (training used 20 s windows). (b) same "
                    "episode >= 60 s from the window; (c) another episode of the held-out shards.",
           "results": {}}
    for H in Hs:
        r = {}
        for bk, (s, e) in BUCKETS.items():
            x = {c: res[(H, c)][:, s:e].mean(1) for c in "abc"}
            n = len(x["a"])
            r[bk] = {"ce_a": round(float(x["a"].mean()), 4), "ce_b": round(float(x["b"].mean()), 4),
                     "ce_c": round(float(x["c"].mean()), 4),
                     "share_a_lt_b": round(float((x["a"] < x["b"]).mean()), 4),
                     "share_a_lt_b_ci95": wilson(int((x["a"] < x["b"]).sum()), n),
                     "share_a_lt_c": round(float((x["a"] < x["c"]).mean()), 4),
                     "share_a_lt_c_ci95": wilson(int((x["a"] < x["c"]).sum()), n),
                     "gap_b_minus_a": mean_ci(x["b"] - x["a"]), "gap_c_minus_a": mean_ci(x["c"] - x["a"])}
        out["results"][f"heard{H}s"] = r
    out["gap30_minus_gap5_3-10s"] = mean_ci((res[(30, "b")] - res[(30, "a")])[:, 150:].mean(1)
                                            - (res[(5, "b")] - res[(5, "a")])[:, 150:].mean(1)) if 30 in Hs and 5 in Hs else None
    out["codes"] = {"frames": n_frames, "mean_frame_db_all": round(en_all / n_frames, 2),
                    **{str(c): {"share": round(cnt[c] / n_frames, 5),
                                "mean_frame_db": round(en_sum[c] / max(cnt[c], 1), 2)} for c in codes},
                    "quietest_10_codes_by_mean_db": [[int(c), round(en_sum[c] / cnt[c], 1), round(cnt[c] / n_frames, 4)]
                                                    for c in np.argsort(np.where(cnt > 50, en_sum / np.maximum(cnt, 1), 1e9))[:10]]}
    Path(a.out).write_text(json.dumps(out, indent=1, default=float))
    lines = ["# DY-32 track B: likelihood test of heard content (UnitLM 409M, corpus K2000 50 Hz, step "
             f"{ck['step']})", "", f"windows {len(wins)}, episodes {winfo['episodes']}, held-out shards la-00030, "
             "la-00071, la-00109. CE of the true own codes (nats/frame).", "",
             "| heard | bucket | CE a | CE b | CE c | share a<b (95% CI) | share a<c (95% CI) | b-a (95% CI) |",
             "|---|---|---|---|---|---|---|---|"]
    for H in Hs:
        for bk, v in out["results"][f"heard{H}s"].items():
            lines.append(f"| {H} s | {bk} | {v['ce_a']} | {v['ce_b']} | {v['ce_c']} | {v['share_a_lt_b']} "
                         f"{v['share_a_lt_b_ci95']} | {v['share_a_lt_c']} {v['share_a_lt_c_ci95']} | "
                         f"{v['gap_b_minus_a']['mean']} {v['gap_b_minus_a']['ci95']} |")
    lines += ["", f"(b-a at 30 s) - (b-a at 5 s), 3-10 s bucket: {out['gap30_minus_gap5_3-10s']}", "",
              f"Codes on own audio: {json.dumps(out['codes'], default=float)}", "", out["notes"]]
    Path(a.out).with_suffix(".md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
