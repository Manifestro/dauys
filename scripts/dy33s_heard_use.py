"""DY-33 track S: likelihood test of heard content on the multi-stream LM over slow units (no training).

Adapted from dy32c_heard_use.py. Differences: the "pieces" windows are drawn from are RUNS (consecutive pieces of one
episode, contiguous in time and in segs.bin) so heard parts up to 120 s fit; audio -> layer-6 features -> frozen slow
codec (dy33s_gen.encode), heard and own encoded separately; 12.5 steps/s, own 10 s = 125 steps; buckets in steps
0-12 (0-1 s), 12-37 (1-3 s), 37-125 (3-10 s); (b-a)/a in percent is reported too.

Original description (DY-32 track C):

Same protocol as track B's dy32b_heard_use.py (window selection copied: held-out shards, pieces.jsonl episodes, own =
next 10 s, heard H s under a = true preceding, b = same episode >= 60 s away, c = other episode). Heard and own audio
are encoded separately (RVQUnits, as at generation). Sequence = EOS frame + heard frames + own frames in the delay
layout; CE of the true own codes per level (level k of own frame f sits at position P + f + k), buckets 0-1 s,
1-3 s, 3-10 s; also the mean over levels.

    python scripts/dy32c_heard_use.py --held /opt/dlami/nvme/held --ckpt /data/dy32c/run/best.pt --out /data/dy32c/heard_use.json
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
from dy32c_units import SR, RVQUnits
from dy33s_codec import load_codec
from dy33s_gen import encode

from dauys.dy32c_lm import MSConfig, MultiStreamLM, delay

OWN_S, GAP_S = 10, 60.0
BUCKETS = {"0-1s": (0, 12), "1-3s": (12, 37), "3-10s": (37, 125)}
OWN_STEPS = 125


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
    """Copied from track B (dy32b_heard_use.windows) so both tracks draw windows the same way."""
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
        cur = None
        for (st, ln), m in zip(idx, meta):
            st, ln = int(st), int(ln)
            if (cur and cur[3] == f"{d.name}/{m['ep']}" and cur[1] + cur[2] == st
                    and abs(float(m["start"]) - cur[5]) <= 0.05):
                cur = (segs, cur[1], cur[2] + ln, cur[3], cur[4], float(m["end"]))
                continue
            if cur:
                pieces.append(cur[:5])
            cur = (segs, st, ln, f"{d.name}/{m['ep']}", float(m["start"]), float(m["end"]))
        if cur:
            pieces.append(cur[:5])
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
            lo, hi = tj, tj + (lj - hs) / SR
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
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--heard", default="120,60,30,15,5")
    p.add_argument("--codec", required=True)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    dev = torch.device("cuda")
    Hs = [int(x) for x in a.heard.split(",")]
    hmax = max(Hs)
    pieces, wins, winfo = windows(a.held, a.n, a.seed, hmax)

    def chunk(i, off, n):
        segs, st, _, _, _ = pieces[i]
        return np.asarray(segs[st + off : st + off + n], np.int16)

    u = RVQUnits(dev)
    codec, cck = load_codec(Path(a.codec), dev)
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cfg = MSConfig(**ck["cfg"])
    order = ck["order"]
    m = MultiStreamLM(cfg).to(dev).eval()
    m.load_state_dict(ck["model"])
    S, K, PAD = cfg.streams, cfg.k, cfg.k + 1

    @torch.no_grad()
    def ce(heard: list[np.ndarray], own: list[np.ndarray]) -> np.ndarray:
        """-> [B, n_own, S] CE of the true own codes."""
        n_own = min(len(o) for o in own)
        n_h = min(len(h) for h in heard)
        seqs = [np.r_[np.full((1, S), K), h[-n_h:][:, order], o[:n_own][:, order]] for h, o in zip(heard, own)]
        d = torch.from_numpy(np.stack([delay(s.astype(np.int64), PAD) for s in seqs])).to(dev)
        P = 1 + n_h
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h = m.states(d[:, :-1])
        lg = m.logits(h)  # [B, T-1, S, V]; position t-1 predicts position t
        out = []
        for k in range(S):
            pos = torch.arange(P + k, P + k + n_own, device=dev)
            l = lg[:, pos - 1, k]
            t = d[:, pos, k]
            out.append(F.cross_entropy(l.flatten(0, 1), t.flatten(), reduction="none").view(t.shape))
        return torch.stack(out, -1).cpu().numpy()

    res = {(H, c): [] for H in Hs for c in "abc"}
    hs, os_ = hmax * SR, OWN_S * SR
    for b0 in range(0, len(wins), a.batch):
        W = wins[b0 : b0 + a.batch]
        own = encode(u, codec, [chunk(i, off + hs, os_).astype(np.float32) / 32768.0 for i, off, *_ in W], dev)
        src = {"a": [(i, off) for i, off, *_ in W], "b": [(j, ob) for _, _, j, ob, _, _ in W],
               "c": [(k, oc) for *_, k, oc in W]}
        for H in Hs:
            for c, lst in src.items():
                heard = encode(u, codec, [chunk(i, o + hs - H * SR, H * SR).astype(np.float32) / 32768.0 for i, o in lst], dev)
                res[(H, c)].append(ce(heard, own))
        if b0 % 160 == 0:
            print(json.dumps({"done": b0 + len(W)}), flush=True)
    res = {k: np.concatenate([x[:, :OWN_STEPS] for x in v]) for k, v in res.items()}  # [N, 125, S]

    names = [f"L{k + 1}" for k in range(S)] + ["mean"]
    out = {"info": winfo, "step": ck["step"], "val": ck["val"], "codec_step": cck["step"], "table": []}
    for H in Hs:
        for bk, (lo, hi) in BUCKETS.items():
            for li, nm in enumerate(names):
                sel = (lambda x: x[:, lo:hi].mean(-1).mean(-1)) if nm == "mean" else (lambda x, li=li: x[:, lo:hi, li].mean(-1))
                A, B, C = (sel(res[(H, c)]) for c in "abc")
                n = len(A)
                out["table"].append({"heard_s": H, "bucket": bk, "level": nm, "ce_a": round(float(A.mean()), 4),
                                     "ce_b": round(float(B.mean()), 4), "ce_c": round(float(C.mean()), 4),
                                     "b_minus_a": mean_ci(B - A), "c_minus_a": mean_ci(C - A),
                                     "b_minus_a_rel_pct": round(100 * float((B - A).mean() / A.mean()), 3),
                                     "c_minus_a_rel_pct": round(100 * float((C - A).mean() / A.mean()), 3),
                                     "share_a_lt_b": round(float((A < B).mean()), 4), "share_a_lt_b_ci": wilson(int((A < B).sum()), n),
                                     "share_a_lt_c": round(float((A < C).mean()), 4), "share_a_lt_c_ci": wilson(int((A < C).sum()), n)})
    Path(a.out).write_text(json.dumps(out, indent=1))
    lines = [f"# DY-33 track S: likelihood test of heard content (multi-stream LM 411M on slow units 12.5 Hz x 4 x 1024, step {ck['step']})", "",
             f"windows {winfo['windows']}, episodes {winfo['episodes']}, held-out shards la-00030, la-00071 (windows inside runs of contiguous audio). CE of the true own codes "
             "(nats/step code) per level (L1 = coarsest) and mean over levels. Heard and own encoded separately; sequence = EOS step + "
             "heard + own in the delay layout. (a) true preceding, (b) same episode >= 60 s away, (c) another episode.", "",
             "| heard | bucket | level | CE a | CE b | CE c | b-a (95% CI) | (b-a)/a % | c-a (95% CI) | (c-a)/a % | share a<b (CI) | share a<c (CI) |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in out["table"]:
        lines.append(f"| {r['heard_s']} s | {r['bucket']} | {r['level']} | {r['ce_a']} | {r['ce_b']} | {r['ce_c']} | "
                     f"{r['b_minus_a']['mean']} {r['b_minus_a']['ci95']} | {r['b_minus_a_rel_pct']} | {r['c_minus_a']['mean']} {r['c_minus_a']['ci95']} | {r['c_minus_a_rel_pct']} | "
                     f"{r['share_a_lt_b']} {r['share_a_lt_b_ci']} | {r['share_a_lt_c']} {r['share_a_lt_c_ci']} |")
    Path(a.out).with_suffix(".md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
