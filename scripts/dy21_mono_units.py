"""DY-21 data: DuplexChat podcast audio (one mono stream, both people) to DY-14 units on the 10 Hz grid.

Source: S3 ``outputs/duplexchat/en-16k/<shard>/`` with ``segs.bin`` (int16 16 kHz, pieces concatenated),
``index.npy`` (start, length) and ``pieces.jsonl``. Per piece: Sylber layer-9 states, segments, F0,
DY-14 track with no other channel (nothing is dropped as bleed), units by nearest DY-14 centroid,
the 10 Hz grid (unit / continue / silence) and prosody. Per shard one file
``<shard>.npz``: ``tok`` int16 (all pieces concatenated), ``pros`` float16 [T, 3], ``index`` int64 [n, 2]
(start, frames), uploaded to ``--dst`` and then deleted locally. Shards already in ``--dst`` are skipped.
Process ``--part k/n`` takes every n-th shard.

    python scripts/dy21_mono_units.py --centroids /data/dy14/centroids.pt --part 0/8 --work /opt/dlami/nvme/w0
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from dy14_features import segments_of
from dy14_grid import nearest

from dauys.dy14_units import SR, build_track, grid_track, hidden_states, load_sylber, segment_f0

BUCKET = "s3://manifestro-dataset-876108973178-us-east-1-an/outputs"
SRC = f"{BUCKET}/duplexchat/en-16k"
DST = f"{BUCKET}/dauys/dc-mono-units"
K = 8192


def sh(cmd: str) -> str:
    return subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True).stdout


def shards(prefix: str) -> list[str]:
    out = sh(f"aws s3 ls {prefix}/")
    return sorted(line.split()[-1].rstrip("/") for line in out.splitlines() if line.strip().startswith("PRE"))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--centroids", required=True)
    p.add_argument("--part", required=True, help="k/n")
    p.add_argument("--work", required=True)
    p.add_argument("--src", default=SRC)
    p.add_argument("--dst", default=DST)
    p.add_argument("--max-shards", type=int, default=0, help="dry run: stop after this many shards")
    args = p.parse_args()
    k, n = map(int, args.part.split("/"))
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    model = load_sylber(device)
    cent = torch.load(args.centroids).to(device)
    done = {line.split()[-1].removesuffix(".npz") for line in sh(f"aws s3 ls {args.dst}/ || true").splitlines()
            if line.strip().endswith(".npz")}
    mine = [s for s in shards(args.src) if s not in done][k::n]
    if args.max_shards:
        mine = mine[: args.max_shards]
    t0, audio_s = time.time(), 0.0
    for j, s in enumerate(mine):
        d = work / s
        d.mkdir(exist_ok=True)
        for fn in ("index.npy", "segs.bin"):
            sh(f"aws s3 cp {args.src}/{s}/{fn} {d}/{fn} --only-show-errors")
        idx = np.load(d / "index.npy")
        segs = np.fromfile(d / "segs.bin", dtype=np.int16)
        toks, pros, index, pos = [], [], [], 0
        for st, ln in idx:
            wav = segs[st : st + ln]
            if ln < 5 * SR:
                continue
            states = hidden_states(model, wav)
            seg = segments_of(states)
            f0, voiced = segment_f0(wav, seg, device)
            tr = build_track(states, seg, wav, np.zeros_like(wav), f0, voiced)
            f = torch.from_numpy(tr.features.astype(np.float32)).to(device)
            units = nearest(F.normalize(f, dim=1), cent).cpu().numpy() if len(f) else np.zeros(0, np.int64)
            tok, pr, _ = grid_track(tr.segments, tr.keep, units, tr.energy_db, tr.f0_st, tr.voiced,
                                    int(len(states)), K)
            toks.append(tok)
            pros.append(pr.astype(np.float16))
            index.append((pos, len(tok)))
            pos += len(tok)
            audio_s += ln / SR
        out = work / f"{s}.npz"
        np.savez(out, tok=np.concatenate(toks) if toks else np.zeros(0, np.int16),
                 pros=np.concatenate(pros) if pros else np.zeros((0, 3), np.float16),
                 index=np.asarray(index, np.int64).reshape(-1, 2))
        sh(f"aws s3 cp {out} {args.dst}/{s}.npz --only-show-errors")
        sh(f"rm -rf {d} {out}")
        el = time.time() - t0
        print(json.dumps({"part": args.part, "shard": s, "done": j + 1, "of": len(mine), "hours": round(audio_s / 3600, 2),
                          "x_realtime": round(audio_s / el, 1)}), flush=True)
    print(f"MONO UNITS DONE part {args.part}", flush=True)


if __name__ == "__main__":
    main()
