"""DY-19 code checks before the real run (on the instance).

1. known answer: on DY-14's synthetic echo data (channel 1 repeats channel 0's burst 1 s after it
   starts) the onset of channel 1 is fully predictable. A small Onsetter trained 1500 steps must
   reach onset AUC >= 0.95 on held-out channel-1 silent frames.
2. one batch: onset BCE on one fixed batch falls below 0.05.
3. init: the DY-16 f1 checkpoint loads into Onsetter with only the onset head missing.

    python scripts/dy19_checklist.py --init /data/dy16/runs/f1/best.pt
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from dy14_checklist import K, check, synthetic

from dauys.dy14_data import CTX, WHO_AHEAD, TrainSampler, sequence
from dauys.dy14_model import TowerConfig
from dauys.dy19_model import Onsetter


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--init", required=True)
    args = p.parse_args()
    device = torch.device("cuda")
    ok = True

    ck = torch.load(args.init, map_location="cpu", weights_only=False)
    m = Onsetter(TowerConfig(**ck["cfg"]))
    missing, unexpected = m.load_state_dict(ck["model"], strict=False)
    ok &= check("init", set(missing) == {"onset.weight", "onset.bias"} and not unexpected,
                {"missing": sorted(missing), "unexpected": sorted(unexpected)})
    del m

    torch.manual_seed(0)
    train, held = synthetic(40, 3000, 1), synthetic(4, 3000, 2)
    cfg = TowerConfig(k=K, d=128, layers=2, heads=4, ffn=256, latent=64, target_d=64, target_layers=1)
    model = Onsetter(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    sampler = TrainSampler(train, K, 16, 8, seed=0)
    for _ in range(1500):
        loss, _ = model.loss({k: v.to(device) for k, v in sampler().items()})
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()
    pos, neg = [], []
    with torch.no_grad():
        for c in held:
            for s in range(0, c.length - CTX - WHO_AHEAD, CTX):
                x = {k: v[None].to(device) for k, v in sequence(c, 1, s).items()}
                sc = model.onset_logits(model.states(x["tok_self"], x["tok_partner"], x["pros"]))[0].cpu().numpy()
                for i in range(CTX):
                    t = s + i
                    if c.active[1, t]:
                        continue
                    (pos if c.active[1, t + 1 : t + 1 + WHO_AHEAD].any() else neg).append(sc[i])
    pos, neg = np.array(pos), np.array(neg)
    auc = float((pos[:, None] > neg[None, :]).mean())
    ok &= check("known_answer", auc >= 0.95, {"pos": len(pos), "neg": len(neg), "onset_auc": round(auc, 4)})

    model = Onsetter(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    batch = {k: v.to(device) for k, v in TrainSampler(train, K, 8, 8, seed=5)().items()}
    for _ in range(300):
        loss, logs = model.loss(batch)
        opt.zero_grad()
        loss.backward()
        opt.step()
    ok &= check("one_batch", logs["onset"] < 0.05, {"onset_bce_last": round(logs["onset"], 4)})
    print("CHECKLIST", "PASS" if ok else "FAIL", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
