"""DY-26 checklist before the real runs (results go to the Airon experiment):

1. leak     — for both alphabets: the same shard names, held-out shards (sha1 % 50 == 0) never in training;
              benchmark clips (ZeroSpeech sLM21 dev) are a different corpus from DuplexChat.
2. known    — a small LM learns a repeating pattern (period 37, random phase): next-token accuracy >= 0.95.
3. overfit  — the DY-26 model (512,8,8,2048,1024) on one fixed batch of real tokens: CE < 0.5 within 300 steps.

    python scripts/dy26_checklist.py --phon /data/dy26/phon --dy14 /data/dy26/dy14 --out /data/dy26/checklist.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from dy21_train import held_out, load_stream

from dauys.dy21_lm import LMConfig, UnitLM


def fit(cfg: LMConfig, batch: torch.Tensor, steps: int, lr: float, skip: int = 0) -> tuple[float, float]:
    torch.manual_seed(0)
    m = UnitLM(cfg).cuda()
    opt = torch.optim.AdamW(m.parameters(), lr=lr)
    x = batch.cuda()
    for _ in range(steps):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg = m.logits(m.states(x[:, :-1]))
        ce = F.cross_entropy(lg.float().flatten(0, 1), x[:, 1:].flatten())
        opt.zero_grad()
        ce.backward()
        opt.step()
    acc = (lg.argmax(-1) == x[:, 1:])[:, skip:].float().mean().item()
    return ce.item(), acc


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--phon", required=True)
    p.add_argument("--dy14", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    res = {}

    a = sorted(f.stem for f in Path(args.phon).glob("*.npz"))
    b = sorted(f.stem for f in Path(args.dy14).glob("*.npz"))
    held = [s for s in a if held_out(s)]
    res["leak"] = {"same_shards": a == b, "shards": len(a), "held_out": held,
                   "train_has_held": any(held_out(s) for s in a if s not in held), "pass": a == b and len(held) > 0}
    print("LEAK", json.dumps(res["leak"]), flush=True)

    rng = np.random.default_rng(0)
    period = rng.integers(0, 64, 37)
    seq = np.stack([np.tile(period, 300)[o : o + 257] for o in rng.integers(0, 37, 32)])
    ce, acc = fit(LMConfig(vocab=8193, d=256, layers=4, heads=4, ffn=1024, ctx=256), torch.from_numpy(seq).long(), 300, 1e-3, skip=37)
    res["known"] = {"ce": round(ce, 4), "acc_after_first_period": round(acc, 4), "pass": acc >= 0.95}
    print("KNOWN", json.dumps(res["known"]), flush=True)

    files = [f for f in sorted(Path(args.phon).glob("*.npz")) if not held_out(f.stem)][:1]
    s = load_stream(files)
    st = rng.integers(0, len(s) - 1026, 16)
    batch = torch.from_numpy(np.stack([s[i : i + 1025] for i in st]).astype(np.int64))
    ce, acc = fit(LMConfig(d=512, layers=8, heads=8, ffn=2048, ctx=1024), batch, 300, 3e-4)
    res["overfit"] = {"ce": round(ce, 4), "acc": round(acc, 4), "pass": ce < 0.5}
    print("OVERFIT", json.dumps(res["overfit"]), flush=True)

    Path(args.out).write_text(json.dumps(res, indent=2))
    print("CHECKLIST", "PASS" if all(v["pass"] for v in res.values()) else "FAIL", flush=True)


if __name__ == "__main__":
    main()
