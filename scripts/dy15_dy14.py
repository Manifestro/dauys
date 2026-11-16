"""DY-15: DY-14's two seeds on exactly the anchors DY-15 gives Moshi (``items.json``).

    python scripts/dy15_dy14.py --grid /data/dy14/grid --anchors test_anchors.json \
        --items /data/dy15/items.json --runs /data/dy14/runs/s0,/data/dy14/runs/s1 --out /data/dy15/dy14_subset.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from dauys.dy14_data import load_convs
from dauys.dy14_eval import bootstrap, correct_per_anchor
from dauys.dy14_model import Tower, TowerConfig


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--grid", required=True)
    p.add_argument("--anchors", required=True)
    p.add_argument("--items", required=True)
    p.add_argument("--runs", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--k", type=int, default=8192)
    args = p.parse_args()
    names = json.loads(Path(args.anchors).read_text())["names"]
    items = json.loads(Path(args.items).read_text())
    convs = load_convs(args.grid, names, args.k)
    assert [c.name for c in convs] == names, (
        "grid conversations differ from the anchors'"
    )
    sub = {"items": items}
    conv = np.array([it["conv"] for it in items])
    device = torch.device("cuda")
    res = {}
    for run in args.runs.split(","):
        ck = torch.load(Path(run) / "best.pt", map_location="cpu", weights_only=False)
        model = Tower(TowerConfig(**ck["cfg"])).to(device)
        model.load_state_dict(ck["model"])
        real = correct_per_anchor(model, convs, sub, swap=False, device=device).astype(
            float
        )
        swap = correct_per_anchor(model, convs, sub, swap=True, device=device).astype(
            float
        )
        r = {"n": len(items)}
        r["acc_onset"], r["acc_onset_lo"], r["acc_onset_hi"] = bootstrap(real, conv)
        r["acc_onset_swap"], _, _ = bootstrap(swap, conv)
        r["delta_onset"], r["delta_onset_lo"], r["delta_onset_hi"] = bootstrap(
            real - swap, conv
        )
        res[Path(run).name] = r
        print(json.dumps({Path(run).name: r}), flush=True)
    Path(args.out).write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
