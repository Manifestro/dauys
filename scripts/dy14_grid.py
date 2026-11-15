"""DY-14 step 2: k-means over Sylber features of train, units for all tracks, the 10 Hz grid.

k-means: ``--sample`` kept train segments, L2-normalised, K centroids, 20 Lloyd steps on
the GPU, seed 0; empty clusters are refilled from random points. Every kept segment of
every part gets its nearest centroid. Writes ``OUT/centroids.pt``, ``OUT/grid/<conv>.npz``
(``tok_a``, ``tok_b`` int16, ``pros_a``, ``pros_b`` float32 [T, 3]) and ``OUT/stats.json``
(units per second of speech, collisions, bleed share, per part).

    python scripts/dy14_grid.py --seg /data/dy14/seg --split split_round1.json --out /data/dy14
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from dauys.dy14_units import grid_track


@torch.no_grad()
def kmeans(x: torch.Tensor, k: int, steps: int, seed: int) -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(seed)
    c = x[torch.randperm(len(x), generator=g)[:k].to(x.device)].clone()
    for step in range(steps):
        assign = nearest(x, c)
        sums = torch.zeros_like(c).index_add_(0, assign, x)
        counts = torch.bincount(assign, minlength=k).float()
        empty = counts == 0
        c = sums / counts.clamp(min=1)[:, None]
        if empty.any():
            refill = torch.randperm(len(x), generator=g)[: int(empty.sum())].to(
                x.device
            )
            c[empty] = x[refill]
        c = F.normalize(c, dim=1)
        print(json.dumps({"kmeans_step": step, "empty": int(empty.sum())}), flush=True)
    return c


@torch.no_grad()
def nearest(x: torch.Tensor, c: torch.Tensor, chunk: int = 65536) -> torch.Tensor:
    out = torch.empty(len(x), dtype=torch.long, device=x.device)
    for i in range(0, len(x), chunk):
        out[i : i + chunk] = (x[i : i + chunk] @ c.T).argmax(1)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seg", required=True)
    p.add_argument("--split", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--k", type=int, default=8192)
    p.add_argument("--sample", type=int, default=2_000_000)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()

    split = json.loads(Path(args.split).read_text())
    parts = {
        name: split[name][: args.limit or None] for name in ("train", "val", "test")
    }
    seg = Path(args.seg)
    out = Path(args.out)
    (out / "grid").mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    centroids_path = out / "centroids.pt"
    if centroids_path.exists():
        centroids = torch.load(centroids_path).to(device)
    else:
        feats = []
        for conv in parts["train"]:
            z = np.load(seg / f"{conv}.npz")
            for t in "ab":
                feats.append(z[f"{t}_features"][z[f"{t}_keep"]])
        allf = np.concatenate(feats)
        rng = np.random.default_rng(0)
        pick = rng.choice(len(allf), size=min(args.sample, len(allf)), replace=False)
        x = F.normalize(
            torch.from_numpy(allf[pick].astype(np.float32)).to(device), dim=1
        )
        print(
            json.dumps({"train_kept_segments": len(allf), "kmeans_points": len(x)}),
            flush=True,
        )
        centroids = kmeans(x, args.k, args.steps, seed=0)
        torch.save(centroids.cpu(), centroids_path)
        del allf, feats, x

    stats = {}
    for part, convs in parts.items():
        tot = {
            "units": 0,
            "collisions": 0,
            "active_s": 0.0,
            "bleed_dropped": 0,
            "segments": 0,
        }
        for conv in convs:
            z = np.load(seg / f"{conv}.npz")
            saved = {}
            for t in "ab":
                f = torch.from_numpy(z[f"{t}_features"].astype(np.float32)).to(device)
                units = (
                    nearest(F.normalize(f, dim=1), centroids).cpu().numpy()
                    if len(f)
                    else np.zeros(0, int)
                )
                tok, pros, st = grid_track(
                    z[f"{t}_segments"],
                    z[f"{t}_keep"],
                    units,
                    z[f"{t}_energy_db"],
                    z[f"{t}_f0_st"],
                    z[f"{t}_voiced"],
                    int(z[f"{t}_frames"]),
                    args.k,
                )
                saved[f"tok_{t}"], saved[f"pros_{t}"] = tok, pros
                for key in st:
                    tot[key] += st[key]
                tot["segments"] += len(z[f"{t}_keep"])
            np.savez(out / "grid" / f"{conv}.npz", **saved)
        tot["units_per_s_speech"] = tot["units"] / max(tot["active_s"], 1e-6)
        tot["collision_share"] = tot["collisions"] / max(tot["units"], 1)
        tot["bleed_share"] = tot["bleed_dropped"] / max(tot["segments"], 1)
        stats[part] = tot
        print(json.dumps({part: tot}), flush=True)
    (out / "stats.json").write_text(json.dumps(stats, indent=2))
    print("GRID DONE", flush=True)


if __name__ == "__main__":
    main()
