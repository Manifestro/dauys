"""DY-21 code checks before the real run (on the instance).

1. leak: held-out shards (sha1 % 50 == 0) are not among the training files.
2. known answer: a small UnitLM on a stream that repeats one 50-unit pattern learns it: after 600 steps
   the next-unit accuracy on the stream is >= 0.95 and generation from 20 pattern units continues it.
3. one batch: CE on one fixed real batch falls below 0.5 in 300 steps; the JEPA loss falls too.

    python scripts/dy21_checklist.py --units /opt/dlami/nvme/units
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from dy14_checklist import check
from dy21_train import K, held_out, load_stream

from dauys.dy16_model import sample_next
from dauys.dy21_lm import LMConfig, UnitLM, ema_update, loss, make_target


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--units", required=True)
    args = p.parse_args()
    device = torch.device("cuda")
    ok = True
    files = sorted(Path(args.units).glob("*.npz"))
    tr = [f.stem for f in files if not held_out(f.stem)]
    va = [f.stem for f in files if held_out(f.stem)]
    ok &= check("leak", len(va) > 0 and not set(tr) & set(va), {"train": len(tr), "held_out": len(va)})

    torch.manual_seed(0)
    cfg = LMConfig(d=128, layers=2, heads=4, ffn=512, ctx=128)
    rng = np.random.default_rng(0)
    pattern = rng.integers(0, K, 50)
    stream = np.tile(pattern, 400)
    model = UnitLM(cfg).to(device)
    tgt = make_target(model)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    for _ in range(600):
        s = rng.integers(0, len(stream) - 130, 32)
        ids = torch.from_numpy(np.stack([stream[i : i + 129] for i in s])).to(device)
        total, logs = loss(model, tgt, ids)
        opt.zero_grad()
        total.backward()
        opt.step()
        ema_update(tgt, model, 0.99)
    model.eval()
    with torch.no_grad():
        ids = torch.from_numpy(stream[:129][None]).to(device)
        acc = float((model.logits(model.states(ids[:, :-1])).argmax(-1) == ids[:, 1:]).float()[:, 50:].mean())
        g = torch.Generator(device=device).manual_seed(0)
        cur = torch.from_numpy(pattern[:20][None]).to(device)
        for _ in range(30):
            cur = torch.cat([cur, sample_next(model.logits(model.states(cur)[:, -1]), 0.8, 0.95, g)[:, None]], 1)
        cont = float((cur[0, 20:].cpu().numpy() == pattern[20:50]).mean())
    ok &= check("known_answer", acc >= 0.95 and cont >= 0.9, {"next_acc": round(acc, 3), "continuation_match": round(cont, 3)})

    real = load_stream(files[:2])
    model = UnitLM(cfg).to(device)
    tgt = make_target(model)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    s = rng.integers(0, len(real) - 130, 8)
    ids = torch.from_numpy(np.stack([real[i : i + 129] for i in s]).astype(np.int64)).to(device)
    first = None
    for _ in range(300):
        total, logs = loss(model, tgt, ids)
        first = first or dict(logs)
        opt.zero_grad()
        total.backward()
        opt.step()
        ema_update(tgt, model, 0.99)
    ok &= check("one_batch", logs["ce"] < 0.5 and logs["jepa"] < first["jepa"],
                {"ce_last": round(logs["ce"], 4), "jepa_first": round(first["jepa"], 4), "jepa_last": round(logs["jepa"], 4)})
    print("CHECKLIST", "PASS" if ok else "FAIL", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
