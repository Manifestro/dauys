"""DY-14 step 1: Sylber segments, features, bleed flag and prosody for every Open Yap track.

Reads ``ROOT/conv_*/{a,b}.npy`` (16 kHz int16 headset tracks) of the conversations in the
split file and writes ``OUT/<conv>.npz`` with, per track ``a`` and ``b``: segments (50 Hz
frames), features (float16), keep, energy_db, f0_st, voiced and frames. The GPU runs the
model while a process pool segments the previous chunks. A conversation already written
is skipped, so the step resumes.

    python scripts/dy14_features.py --root /data/oy16 --split split_round1.json --out /data/dy14/seg
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from dauys.dy14_units import (
    CHUNK_S,
    HOP,
    SR,
    build_track,
    get_segment,
    hidden_states,
    load_sylber,
    segment_f0,
)

CHUNK_FRAMES = CHUNK_S * SR // HOP


def segments_of(states: np.ndarray) -> np.ndarray:
    """Sylber segments chunk by chunk (in process: a pool spent its time pickling)."""
    parts = [
        get_segment(states[i : i + CHUNK_FRAMES]) + i
        for i in range(0, len(states), CHUNK_FRAMES)
    ]
    return np.concatenate(parts) if parts else np.zeros((0, 2), np.int64)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--split", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--parts", default="train,val,test")
    p.add_argument(
        "--limit", type=int, default=0, help="first N conversations per part (dry run)"
    )
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--shards", type=int, default=4, help="extractors sharing the GPU")
    p.add_argument("--shard", type=int, default=-1)
    args = p.parse_args()

    if args.shards > 1 and args.shard < 0:
        # several extractors keep the GPU busy while each one's pool segments on the CPU
        cmd = [
            sys.executable,
            *sys.argv,
            "--workers",
            str(max(args.workers // args.shards, 2)),
        ]
        procs = [
            subprocess.Popen([*cmd, "--shard", str(i)]) for i in range(args.shards)
        ]
        codes = [proc.wait() for proc in procs]
        if any(codes):
            raise SystemExit(f"feature shards failed: {codes}")
        print("FEATURES DONE all shards", flush=True)
        return

    split = json.loads(Path(args.split).read_text())
    convs = []
    for part in args.parts.split(","):
        names = split[part][: args.limit or None]
        convs += names
    if args.shard >= 0:
        convs = convs[args.shard :: args.shards]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_sylber(device)
    started, audio_s = time.time(), 0.0
    for n, conv in enumerate(convs):
        dest = out / f"{conv}.npz"
        if dest.exists():
            continue
        a = np.load(Path(args.root) / conv / "a.npy", mmap_mode="r")
        b = np.load(Path(args.root) / conv / "b.npy", mmap_mode="r")
        a, b = np.asarray(a), np.asarray(b)
        saved = {}
        for name, own, other in (("a", a, b), ("b", b, a)):
            states = hidden_states(model, own)
            seg = segments_of(states)
            f0, voiced = segment_f0(own, seg, device)
            tr = build_track(states, seg, own, other, f0, voiced)
            for key, value in vars(tr).items():
                saved[f"{name}_{key}"] = np.asarray(value)
        tmp = dest.with_suffix(".tmp.npz")
        np.savez(tmp, **saved)
        tmp.rename(dest)
        audio_s += (len(a) + len(b)) / SR
        rate = audio_s / max(time.time() - started, 1e-6)
        kept = saved["a_keep"].mean() if len(saved["a_keep"]) else 0.0
        print(
            json.dumps(
                {
                    "conv": conv,
                    "done": n + 1,
                    "of": len(convs),
                    "track_hours": round(audio_s / 3600, 2),
                    "x_realtime": round(rate, 1),
                    "units_a": int(saved["a_keep"].sum()),
                    "kept_a": round(float(kept), 3),
                }
            ),
            flush=True,
        )
    print("FEATURES DONE", len(convs), flush=True)


if __name__ == "__main__":
    main()
