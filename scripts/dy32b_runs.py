"""DY-32 track B: run lengths of unit sequences (why low temperature collapses into a held sound).

Real = real_XX.wav of the pack through the same tokeniser (the true next 10 s); generated = the "codes" of each
gen folder's info.json. Per set: mean run length (frames), median of per-piece max run, share of frames inside runs
>= 25 frames (0.5 s), distinct codes per 10 s, and the codes that fill most of the long runs.

    python scripts/dy32b_runs.py --cb /data/b/cbA/km-corpus-K2000.npy --pack /data/b/pack --gen b=/data/b/gen_b t06=...
"""

import argparse
import glob
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent))

LONG = 25


def runs(u: np.ndarray) -> list[tuple[int, int]]:
    out, i = [], 0
    while i < len(u):
        j = i
        while j < len(u) and u[j] == u[i]:
            j += 1
        out.append((int(u[i]), j - i))
        i = j
    return out


def stats(seqs: list[np.ndarray]) -> dict:
    rl = [runs(u) for u in seqs]
    allr = [n for r in rl for _, n in r]
    tot = sum(len(u) for u in seqs)
    long_codes = Counter()
    for r in rl:
        for c, n in r:
            if n >= LONG:
                long_codes[c] += n
    return {"pieces": len(seqs), "mean_run": round(float(np.mean(allr)), 2),
            "median_max_run": float(np.median([max(n for _, n in r) for r in rl])),
            "share_in_runs_ge_0.5s": round(sum(long_codes.values()) / tot, 3),
            "distinct_codes_per_piece": round(float(np.mean([len(set(u.tolist())) for u in seqs])), 1),
            "top_long_run_codes": long_codes.most_common(5)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cb", required=True)
    p.add_argument("--pack", required=True)
    p.add_argument("--gen", nargs="+", default=[])
    p.add_argument("--out", default="")
    a = p.parse_args()
    from dy32b_channel import Channel, Codebook

    ch, cb = Channel(), Codebook(a.cb)
    wavs = [sf.read(f, dtype="float32")[0] for f in sorted(glob.glob(f"{a.pack}/real_*.wav"))]
    res = {"real": stats([cb.encode(f).cpu().numpy() for f in ch.feats(wavs)])}
    for g in a.gen:
        name, d = g.split("=")
        info = json.loads(Path(d, "info.json").read_text())
        res[name] = {"step": info["step"], "temp": info["sampling"]["temperature"], "top_p": info["sampling"]["top_p"],
                     **stats([np.asarray(v) for v in info["codes"].values()])}
    for k, v in res.items():
        print("RUNS", k, json.dumps(v), flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
