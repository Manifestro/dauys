"""DY-19 fine-tuning: DY-16 f1 plus an onset head, all DY-16 losses kept.

Every ``--val-every`` steps: onset AUC (silent frames of fixed val batches), speech-head CE and
val acc_all of the listening head; the best checkpoint is the one with the highest onset AUC.

    python scripts/dy19_train.py --grid /data/dy14/grid --split split.json --init /data/dy16/runs/f1/best.pt \
        --out /data/dy19/runs/o0 --seed 0
"""

import argparse
import json
import math
import os
import time
import uuid
from pathlib import Path

import airon
import numpy as np
import torch
import torch.nn.functional as F

from dauys.dy14_data import TrainSampler, TrainStream, build_eval_anchors, load_convs
from dauys.dy14_eval import correct_per_anchor, summarize
from dauys.dy14_model import TowerConfig
from dauys.dy16_model import Speaker
from dauys.dy19_model import Onsetter, onset_target

EXPERIMENT = "DY-19: голова вступления, обученная отдельной целью, порог на val (109M)"


def lr_at(step: int, steps: int, peak: float, warmup: int) -> float:
    if step < warmup:
        return peak * (step + 1) / warmup
    return (
        peak * 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(steps - warmup, 1)))
    )


@torch.no_grad()
def speech_ce(model: Speaker, batches: list[dict], device) -> float:
    model.eval()
    tot, n = 0.0, 0
    for b in batches:
        x = {k: v.to(device) for k, v in b.items()}
        with torch.autocast(
            device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            h = model.states(x["tok_self"], x["tok_partner"], x["pros"])
        logits = model.speech_logits(h[:, :-1])
        tot += F.cross_entropy(
            logits.flatten(0, 1), x["tok_self"][:, 1:].flatten(), reduction="sum"
        ).item()
        n += x["tok_self"][:, 1:].numel()
    model.train()
    return tot / n


@torch.no_grad()
def onset_auc(model: Onsetter, batches: list[dict], device) -> float:
    model.eval()
    pos, neg = [], []
    for b in batches:
        x = {k: v.to(device) for k, v in b.items()}
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            h = model.states(x["tok_self"], x["tok_partner"], x["pros"])
        s = model.onset_logits(h)
        silent, target = onset_target(x, model.cfg.k)
        pos.append(s[silent & (target > 0)].cpu())
        neg.append(s[silent & (target == 0)].cpu())
    model.train()
    pos, neg = torch.cat(pos), torch.cat(neg)
    ranks = torch.cat([pos, neg]).argsort().argsort().double() + 1
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--grid", required=True)
    p.add_argument("--split", required=True)
    p.add_argument("--init", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--val-every", type=int, default=500)
    p.add_argument("--k", type=int, default=8192)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--tracking", default="on", choices=["on", "off"])
    args = p.parse_args()

    torch.manual_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    split = json.loads(Path(args.split).read_text())
    train = load_convs(args.grid, split["train"][: args.limit or None], args.k)
    val = load_convs(args.grid, split["val"][: args.limit or None], args.k)
    val_anchors = build_eval_anchors(val, seed=1234, per_window=3)
    val_batches = [TrainSampler(val, args.k, 32, 16, seed=999)() for _ in range(10)]

    ck = torch.load(args.init, map_location="cpu", weights_only=False)
    cfg = TowerConfig(**ck["cfg"])
    model = Onsetter(cfg).to(device)
    missing, unexpected = model.load_state_dict(ck["model"], strict=False)
    assert set(missing) == {"onset.weight", "onset.bias"} and not unexpected, (
        missing,
        unexpected,
    )
    decay = [q for n, q in model.named_parameters() if q.dim() >= 2 and "emb" not in n]
    other = [
        q for n, q in model.named_parameters() if not (q.dim() >= 2 and "emb" not in n)
    ]
    opt = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": 0.1},
            {"params": other, "weight_decay": 0.0},
        ],
        lr=args.lr,
        betas=(0.9, 0.95),
    )
    meta = {
        "run_id": str(uuid.uuid4()),
        "seed": args.seed,
        "init": args.init,
        "init_step": ck["step"],
        "params": model.parameters_count(),
        "config": vars(cfg),
        "args": vars(args),
        "git": os.environ.get("DY19_COMMIT", ""),
    }
    (out / "metadata.json").write_text(json.dumps(meta, indent=2))
    run = airon.init(
        project="dauys",
        name=f"dy19-o{args.seed}",
        config=meta,
        experiment=EXPERIMENT,
        tags=["dy19", f"seed-{args.seed}"],
        total_steps=args.steps,
        run_id=meta["run_id"],
        mode="disabled" if args.tracking == "off" else None,
    )
    stream = TrainStream(train, args.k, args.batch, 16, seed=args.seed + 100)
    loader = iter(
        torch.utils.data.DataLoader(
            stream,
            batch_size=None,
            num_workers=args.workers,
            pin_memory=True,
            prefetch_factor=4,
            persistent_workers=True,
        )
    )
    first = onset_auc(model, val_batches, device)
    print(
        json.dumps({"params": meta["params"], "val_onset_auc_start": first}), flush=True
    )
    best = -math.inf
    t0 = time.time()
    model.train()
    for step in range(args.steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, args.steps, args.lr, args.warmup)
        batch = next(loader)
        if len(batch["anchor_row"]) == 0:
            continue
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        with torch.autocast(
            device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            loss, logs = model.loss(batch)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        logs["grad_norm"] = float(
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        )
        opt.step()
        if (step + 1) % 50 == 0:
            logs["lr"] = opt.param_groups[0]["lr"]
            logs["steps_per_s"] = 50 / (time.time() - t0)
            t0 = time.time()
            run.log({f"train/{k}": v for k, v in logs.items()}, step=step + 1)
            print(
                json.dumps(
                    {"step": step + 1, **{k: round(v, 4) for k, v in logs.items()}}
                ),
                flush=True,
            )
        if (step + 1) % args.val_every == 0 or step + 1 == args.steps:
            ce = speech_ce(model, val_batches, device)
            real = correct_per_anchor(
                model, val, val_anchors, swap=False, device=device
            )
            swapped = correct_per_anchor(
                model, val, val_anchors, swap=True, device=device
            )
            res = summarize(real, swapped, val_anchors)
            res["speech_ce"] = ce
            res["onset_auc"] = auc = onset_auc(model, val_batches, device)
            run.log({f"val/{k}": v for k, v in res.items()}, step=step + 1)
            print(json.dumps({"step": step + 1, "val": res}), flush=True)
            if auc > best:
                best = auc
                torch.save(
                    {
                        "model": model.state_dict(),
                        "step": step + 1,
                        "val": res,
                        "cfg": vars(cfg),
                    },
                    out / "best.pt",
                )
    (out / "DONE").write_text(f"{best}\n")
    run.finish()
    print("TRAIN DONE", best, np.round(first, 4), flush=True)


if __name__ == "__main__":
    main()
