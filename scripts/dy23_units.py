"""DY-23 data: as DY-21 (DuplexChat podcast audio to DY-14 units on the 10 Hz grid), plus voice-invariant
product codes per unit: the content code z of each syllable (units_vinv.py coder, w = 1) split into 8 groups
of 32 dims, each the nearest of 256 group codes (sWUGGY-voice ABX 86.4 % vs 69.9 % for the DY-14 unit).
Extra array per shard: ``codes`` uint8 [T, 8], the 8 codes at the frame where a unit starts, else 0.


Source: S3 ``outputs/duplexchat/en-16k/<shard>/`` with ``segs.bin`` (int16 16 kHz, pieces concatenated),
``index.npy`` (start, length) and ``pieces.jsonl``. Per piece: Sylber layer-9 states, segments, F0,
DY-14 track with no other channel (nothing is dropped as bleed), units by nearest DY-14 centroid,
the 10 Hz grid (unit / continue / silence) and prosody. Per shard one file
``<shard>.npz``: ``tok`` int16 (all pieces concatenated), ``pros`` float16 [T, 3], ``index`` int64 [n, 2]
(start, frames), uploaded to ``--dst`` and then deleted locally. Shards already in ``--dst`` are skipped.
Process ``--part k/n`` takes every n-th shard.

    python scripts/dy23_units.py --centroids /data/dy14/centroids.pt --part 0/8 --work /opt/dlami/nvme/w0
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
from units_vinv import Coder

from dauys.dy14_units import GRID, SR, build_track, grid_track, hidden_states, load_sylber, segment_f0

BUCKET = "s3://manifestro-dataset-876108973178-us-east-1-an/outputs"
SRC = f"{BUCKET}/duplexchat/en-16k"
DST = f"{BUCKET}/dauys/dy23-units"
CODER = f"{BUCKET}/dauys/dy23/vinv_inv1.0.pt"
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
    p.add_argument("--sub", default="0/1", help="j/m: only every m-th shard of this part, from j")
    p.add_argument("--max-shards", type=int, default=0, help="dry run: stop after this many shards")
    p.add_argument("--reverse", action="store_true", help="take the shards from the end (a helper on another instance)")
    args = p.parse_args()
    k, n = map(int, args.part.split("/"))
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    model = load_sylber(device)
    cent = torch.load(args.centroids).to(device)
    sh(f"aws s3 cp {CODER} {work}/coder.pt --only-show-errors")
    ck = torch.load(work / "coder.pt", map_location="cpu")
    coder = Coder().to(device).eval()
    coder.load_state_dict(ck["coder"])
    pq = [b.to(device) for b in ck["pq"]["8x256"]]
    done = {line.split()[-1].removesuffix(".npz") for line in sh(f"aws s3 ls {args.dst}/ || true").splitlines()
            if line.strip().endswith(".npz")}
    j, m = map(int, args.sub.split("/"))
    mine = [s for s in shards(args.src)[k::n][j::m] if s not in done]  # fixed assignment, so instances never overlap
    if args.max_shards:
        mine = mine[: args.max_shards]
    if args.reverse:
        mine = mine[::-1]
    t0, audio_s = time.time(), 0.0
    for j, s in enumerate(mine):
        if sh(f"aws s3 ls {args.dst}/{s}.npz || true").strip():
            continue  # done or taken by another worker (forward and reverse workers meet in the middle)
        claim = f"{args.dst}/claims/{s}"
        if sh(f"aws s3 ls {claim} || true").strip():
            continue
        sh(f"echo {args.part} | aws s3 cp - {claim} --only-show-errors")
        d = work / s
        d.mkdir(exist_ok=True)
        for fn in ("index.npy", "segs.bin"):
            sh(f"aws s3 cp {args.src}/{s}/{fn} {d}/{fn} --only-show-errors")
        idx = np.load(d / "index.npy")
        segs = np.fromfile(d / "segs.bin", dtype=np.int16)
        toks, pros, codes, index, pos = [], [], [], [], 0
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
            cd = np.zeros((len(tok), 8), np.uint8)
            if len(f):
                with torch.no_grad():
                    z = coder.z(F.normalize(f, dim=1))
                    c = torch.stack([torch.cdist(p.contiguous(), b).argmin(1) for p, b in zip(z.chunk(8, 1), pq)], 1)
                c = c.cpu().numpy().astype(np.uint8)
                kept = np.flatnonzero(tr.keep)
                taken = np.zeros(len(tok), bool)
                for j, s0 in enumerate(tr.segments[kept][:, 0] // GRID):  # first segment in a frame wins, as grid_track
                    if not taken[s0]:
                        taken[s0] = True
                        cd[s0] = c[kept[j]]
            codes.append(cd)
            toks.append(tok)
            pros.append(pr.astype(np.float16))
            index.append((pos, len(tok)))
            pos += len(tok)
            audio_s += ln / SR
        out = work / f"{s}.npz"
        np.savez(out, tok=np.concatenate(toks) if toks else np.zeros(0, np.int16),
                 pros=np.concatenate(pros) if pros else np.zeros((0, 3), np.float16),
                 index=np.asarray(index, np.int64).reshape(-1, 2),
                 codes=np.concatenate(codes) if codes else np.zeros((0, 8), np.uint8))
        sh(f"aws s3 cp {out} {args.dst}/{s}.npz --only-show-errors")
        sh(f"rm -rf {d} {out}")
        el = time.time() - t0
        print(json.dumps({"part": args.part, "shard": s, "done": j + 1, "of": len(mine), "hours": round(audio_s / 3600, 2),
                          "x_realtime": round(audio_s / el, 1)}), flush=True)
    print(f"DY23 UNITS DONE part {args.part}", flush=True)


if __name__ == "__main__":
    main()
