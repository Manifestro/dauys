"""DY-16 code checks before the real run (on the instance).

1. leak: generation moments only in test conversations; the judge checkpoint is not the init.
2. known answer: on DY-14's synthetic echo data (channel 1 repeats channel 0's burst after
   a pause) a small Speaker generates channel 1 after a burst; at least half of the
   burst's units must reappear in the first 3 s it generates.
3. one batch: speech-head CE on one fixed batch falls below 0.1.

    python scripts/dy16_checklist.py --moments /data/dy16/moments.json --anchors test_anchors.json --split split.json \
        --init /data/dy14/runs/s1/best.pt --judge /data/dy14/runs/s0/best.pt
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from dy14_checklist import K, synthetic
from dy16_generate import generate

from dauys.dy14_data import CTX, TrainSampler
from dauys.dy14_model import TowerConfig
from dauys.dy16_model import Speaker


def check(name: str, ok: bool, info: dict) -> bool:
    print(f"CHECK {name} {'PASS' if ok else 'FAIL'} {json.dumps(info)}", flush=True)
    return ok


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--moments", required=True)
    p.add_argument("--anchors", required=True)
    p.add_argument("--split", required=True)
    p.add_argument("--init", required=True)
    p.add_argument("--judge", required=True)
    args = p.parse_args()
    device = torch.device("cuda")
    ok = True

    names = json.loads(Path(args.anchors).read_text())["names"]
    test = set(json.loads(Path(args.split).read_text())["test"])
    items = json.loads(Path(args.moments).read_text())
    in_test = all(names[it["conv"]] in test for it in items)
    ok &= check(
        "leak",
        in_test and Path(args.init).resolve() != Path(args.judge).resolve(),
        {"moments": len(items), "init": args.init, "judge": args.judge},
    )

    torch.manual_seed(0)
    train, held = synthetic(40, 3000, 1), synthetic(4, 3000, 2)
    cfg = TowerConfig(
        k=K, d=128, layers=2, heads=4, ffn=256, latent=64, target_d=64, target_layers=1
    )
    model = Speaker(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    sampler = TrainSampler(train, K, 16, 8, seed=0)
    for _ in range(2000):
        loss, _ = model.loss({k: v.to(device) for k, v in sampler().items()})
        opt.zero_grad()
        loss.backward()
        opt.step()
    moments, bursts = [], []
    for ci, c in enumerate(held):
        starts = np.flatnonzero(c.tok[0] < K)
        for s in starts[np.r_[True, np.diff(starts) > 4]]:
            t = int(s) + 16  # burst over, the echo starts at s + 30
            if CTX <= t < c.length - 130 and len(moments) < 40:
                moments.append({"conv": ci, "ch": 1, "t": t})
                bursts.append(c.tok[0, s : s + 16][c.tok[0, s : s + 16] < K])
    gens = generate(model, held, moments, device, 1.0, 0.9, seed=0)
    hit = np.mean([np.isin(b, g[:30]).mean() for b, g in zip(bursts, gens)])
    ok &= check(
        "known_answer",
        hit >= 0.5,
        {"moments": len(moments), "burst_units_reproduced": round(float(hit), 3)},
    )

    model = Speaker(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    batch = {k: v.to(device) for k, v in TrainSampler(train, K, 8, 8, seed=5)().items()}
    for _ in range(300):
        loss, logs = model.loss(batch)
        opt.zero_grad()
        loss.backward()
        opt.step()
    ok &= check(
        "one_batch", logs["speech"] < 0.1, {"speech_ce_last": round(logs["speech"], 4)}
    )
    print("CHECKLIST", "PASS" if ok else "FAIL", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
