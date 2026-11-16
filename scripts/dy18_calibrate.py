"""DY-18: calibrate the onset threshold of the DY-17 rule on val, never on test.

On val control moments (own channel silent for 1 s, partner spoke in the last 3 s, own channel
really silent for the next 3 s) the rule never fires (threshold = inf), so the generated own channel
stays silent exactly like the real one and the onset score at each of the first 30 frames is the one
the rule would see. The threshold is the 90th percentile of each moment's maximum score over those
30 frames: by construction the rule starts within 3 s at 10% of val control moments. For reference,
the same is reported for val reply moments (scores up to the real onset; after it the paths diverge).

    python scripts/dy18_calibrate.py --grid /data/dy14/grid --split split.json --model /data/dy16/runs/f0/best.pt \
        --out /data/dy18/thr_f0.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from dy16_generate import moments
from dy17_generate import control_moments, run_all

from dauys.dy14_data import load_convs
from dauys.dy19_model import load_speaker

FALSE_START = 0.10


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--grid", required=True)
    p.add_argument("--split", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=400)
    p.add_argument("--k", type=int, default=8192)
    args = p.parse_args()
    device = torch.device("cuda")
    split = json.loads(Path(args.split).read_text())
    convs = load_convs(args.grid, split["val"], args.k)
    ctrl = control_moments(convs, args.n)
    reply = moments(convs, args.n)
    model = load_speaker(args.model, device)
    _, sc = run_all(model, convs, ctrl, device, False, float("inf"))
    _, sr = run_all(model, convs, reply, device, False, float("inf"))
    cmax = sc[:, :30].max(1)
    thr = float(np.quantile(cmax, 1 - FALSE_START))
    onset = []
    for it in reply:
        u = np.flatnonzero(convs[it["conv"]].unit[it["ch"], it["t"] + 1 : it["t"] + 31])
        onset.append(int(u[0]) if len(u) else 30)
    rmax = np.array([sr[i, : max(o, 1)].max() for i, o in enumerate(onset)])
    res = {
        "model": args.model,
        "val_convs": len(convs),
        "n_control": len(ctrl),
        "n_reply": len(reply),
        "false_start": FALSE_START,
        "threshold": thr,
        "val_control_fire_3s": float((cmax > thr).mean()),
        "val_reply_fire_before_real_onset": float((rmax > thr).mean()),
        "val_control_max_quantiles": [float(q) for q in np.quantile(cmax, [0.5, 0.9, 0.99])],
    }
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
