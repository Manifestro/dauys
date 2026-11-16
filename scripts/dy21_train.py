"""DY-21 training: unit LM + JEPA on DuplexChat mono units (S3 outputs/dauys/dc-mono-units).

``--units DIR`` holds ``<shard>.npz`` files (tok, pros, index). Units of every piece, with "continue" and
"silence" removed, are joined with EOS. Held out: shards whose sha1(name) % 50 == 0 (about 2 %), never
trained on. Batches are random windows of ctx + 1 tokens. Validation every ``--val-every`` steps: CE on
fixed held-out windows; the best checkpoint by it is kept. Logged to Airon.

    python scripts/dy21_train.py --units /opt/dlami/nvme/units --out /data/dy21/run --steps 16000
"""

import argparse
import hashlib
import json
import math
import os
import time
import uuid
from pathlib import Path

import airon
import numpy as np
import torch

from dauys.dy21_lm import LMConfig, UnitLM, ema_update, loss, make_target

K = 8192
EOS = K
EXPERIMENT = "DY-21: языковая модель слогов на ~20 тыс. ч моно-подкастов (300–500M + JEPA) — появляются ли слова"


def held_out(name: str) -> bool:
    return int(hashlib.sha1(name.encode()).hexdigest(), 16) % 50 == 0


def load_stream(files: list[Path]) -> np.ndarray:
    parts = []
    for f in files:
        z = np.load(f)
        tok, index = z["tok"], z["index"]
        for st, n in index:
            u = tok[st : st + n]
            u = u[u < K]
            if len(u) >= 16:
                parts.append(u.astype(np.int16))
                parts.append(np.array([EOS], np.int16))
    return np.concatenate(parts)


def lr_at(step: int, steps: int, peak: float, warmup: int) -> float:
    if step < warmup:
        return peak * (step + 1) / warmup
    return peak * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(steps - warmup, 1))))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--units", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--steps", type=int, default=16000)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup", type=int, default=1000)
    p.add_argument("--val-every", type=int, default=1000)
    p.add_argument("--ema", type=float, default=0.999)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--small", action="store_true", help="dry run: a tiny model")
    p.add_argument("--limit", type=int, default=0, help="dry run: use only this many shard files")
    p.add_argument("--tracking", default="on", choices=["on", "off"])
    args = p.parse_args()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")

    files = sorted(Path(args.units).glob("*.npz"))[: args.limit or None]
    tr_files = [f for f in files if not held_out(f.stem)]
    va_files = [f for f in files if held_out(f.stem)] or files[-1:]
    train, val = load_stream(tr_files), load_stream(va_files)
    cfg = LMConfig(d=256, layers=4, heads=4, ffn=1024, ctx=256) if args.small else LMConfig()
    T = cfg.ctx
    vrng = np.random.default_rng(1234)
    vstarts = vrng.integers(0, len(val) - T - 1, size=256)
    vbatch = torch.from_numpy(np.stack([val[s : s + T + 1] for s in vstarts]).astype(np.int64))

    model = UnitLM(cfg).to(device)
    tgt = make_target(model)
    decay = [q for n, q in model.named_parameters() if q.dim() >= 2 and "emb" not in n]
    other = [q for n, q in model.named_parameters() if not (q.dim() >= 2 and "emb" not in n)]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1}, {"params": other, "weight_decay": 0.0}],
                            lr=args.lr, betas=(0.9, 0.95), fused=True)
    meta = {
        "run_id": str(uuid.uuid4()),
        "params": model.parameters_count(),
        "config": vars(cfg),
        "args": vars(args),
        "train_files": len(tr_files),
        "val_files": len(va_files),
        "train_tokens": int(len(train)),
        "val_tokens": int(len(val)),
        "val_files_names": [f.stem for f in va_files],
        "git": os.environ.get("DY21_COMMIT", ""),
    }
    (out / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps({k: v for k, v in meta.items() if k != "val_files_names"}), flush=True)
    run = airon.init(project="dauys", name=f"dy21-s{args.seed}", config=meta, experiment=EXPERIMENT,
                     tags=["dy21", f"seed-{args.seed}"], total_steps=args.steps, run_id=meta["run_id"],
                     mode="disabled" if args.tracking == "off" else None)

    @torch.no_grad()
    def validate() -> dict:
        model.eval()
        tot, n = 0.0, 0
        for i in range(0, len(vbatch), 32):
            ids = vbatch[i : i + 32].to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                h = model.states(ids[:, :-1])
            tot += torch.nn.functional.cross_entropy(model.logits(h).flatten(0, 1), ids[:, 1:].flatten(),
                                                     reduction="sum").item()
            n += ids[:, 1:].numel()
        model.train()
        return {"ce": tot / n, "ppl": math.exp(tot / n)}

    best, t0 = math.inf, time.time()
    for step in range(args.steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, args.steps, args.lr, args.warmup)
        starts = rng.integers(0, len(train) - T - 1, size=args.batch)
        ids = torch.from_numpy(np.stack([train[s : s + T + 1] for s in starts]).astype(np.int64)).to(device,
                                                                                                    non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            total, logs = loss(model, tgt, ids)
        opt.zero_grad(set_to_none=True)
        total.backward()
        logs["grad_norm"] = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        opt.step()
        ema_update(tgt, model, args.ema)
        if (step + 1) % 50 == 0:
            logs["lr"] = opt.param_groups[0]["lr"]
            logs["steps_per_s"] = 50 / (time.time() - t0)
            t0 = time.time()
            run.log({f"train/{k}": v for k, v in logs.items()}, step=step + 1)
            print(json.dumps({"step": step + 1, **{k: round(v, 4) for k, v in logs.items()}}), flush=True)
        if (step + 1) % args.val_every == 0 or step + 1 == args.steps:
            res = validate()
            run.log({f"val/{k}": v for k, v in res.items()}, step=step + 1)
            print(json.dumps({"step": step + 1, "val": res}), flush=True)
            if res["ce"] < best:
                best = res["ce"]
                torch.save({"model": model.state_dict(), "cfg": vars(cfg), "step": step + 1, "val": res},
                           out / "best.pt")
    (out / "DONE").write_text(f"{best}\n")
    run.finish()
    print("TRAIN DONE", best, flush=True)


if __name__ == "__main__":
    main()
