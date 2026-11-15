"""DY-14 training: one seed of the tower on Open Yap train, validation on val anchors.

Every ``--val-every`` steps: val acc_all, acc_onset with the real and the swapped self
channel, and their difference; the best checkpoint by val acc_all is kept. Stop rule of
the experiment: at step 5000, if val acc_all < 0.08 for this seed and the other seed
(read from ``--peer``, waited for up to 30 min), the run stops and says so.

    python scripts/dy14_train.py --grid /data/dy14/grid --split split.json --out /data/dy14/runs/s0 --seed 0
"""

import argparse
import json
import math
import os
import time
import uuid
from pathlib import Path

import airon
import torch

from dauys.dy14_data import TrainStream, build_eval_anchors, load_convs
from dauys.dy14_eval import correct_per_anchor, summarize
from dauys.dy14_model import Tower, TowerConfig

STOP_STEP, STOP_ACC = 5000, 0.08
EXPERIMENT = "Слушание на Open Yap: единицы Sylber + латент партнёра (одна конфигурация, башня ~100M)"


def lr_at(step: int, steps: int, peak: float, warmup: int) -> float:
    if step < warmup:
        return peak * (step + 1) / warmup
    return (
        peak * 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(steps - warmup, 1)))
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--grid", required=True)
    p.add_argument("--split", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--anchors", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup", type=int, default=1000)
    p.add_argument("--val-every", type=int, default=1000)
    p.add_argument("--k", type=int, default=8192)
    p.add_argument(
        "--peer", default="", help="status file of the other seed for the stop rule"
    )
    p.add_argument("--limit", type=int, default=0)
    p.add_argument(
        "--workers", type=int, default=6, help="CPU workers building batches"
    )
    p.add_argument("--tracking", default="on", choices=["on", "off"])
    args = p.parse_args()

    torch.manual_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    split = json.loads(Path(args.split).read_text())
    train = load_convs(args.grid, split["train"][: args.limit or None], args.k)
    val = load_convs(args.grid, split["val"][: args.limit or None], args.k)
    val_anchors = build_eval_anchors(val, seed=1234, per_window=10)
    (out / "val_anchors.json").write_text(json.dumps(val_anchors))

    cfg = TowerConfig(k=args.k)
    model = Tower(cfg).to(device)
    decay = [
        p_ for n, p_ in model.named_parameters() if p_.dim() >= 2 and "emb" not in n
    ]
    other = [
        p_
        for n, p_ in model.named_parameters()
        if not (p_.dim() >= 2 and "emb" not in n)
    ]
    opt = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": 0.1},
            {"params": other, "weight_decay": 0.0},
        ],
        lr=args.lr,
        betas=(0.9, 0.95),
    )

    meta_path = out / "metadata.json"
    start, best = 0, -1.0
    if (out / "last.pt").exists():
        state = torch.load(out / "last.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        start, best = state["step"], state["best"]
        meta = json.loads(meta_path.read_text())
    else:
        meta = {
            "run_id": str(uuid.uuid4()),
            "seed": args.seed,
            "params": model.parameters_count(),
            "config": vars(cfg),
            "args": vars(args),
            "train_convs": len(train),
            "val_convs": len(val),
            "val_anchors_sha256": val_anchors["sha256"],
            "git": os.environ.get("DY14_COMMIT", ""),
        }
        meta_path.write_text(json.dumps(meta, indent=2))
    print(
        json.dumps(
            {
                "params": meta["params"],
                "train_convs": len(train),
                "val_convs": len(val),
                "val_anchors": len(val_anchors["items"]),
                "start": start,
            }
        ),
        flush=True,
    )

    run = airon.init(
        project="dauys",
        name=f"dy14-s{args.seed}",
        config=meta,
        experiment=EXPERIMENT,
        tags=["dy14", f"seed-{args.seed}"],
        total_steps=args.steps,
        run_id=meta["run_id"],
        resume="allow",
        mode="disabled" if args.tracking == "off" else None,
    )
    # batches are built on CPU workers; a resumed run draws new batches (seed + step)
    stream = TrainStream(
        train, args.k, args.batch, args.anchors, seed=args.seed + start
    )
    loader = iter(
        torch.utils.data.DataLoader(
            stream,
            batch_size=None,
            num_workers=args.workers,
            pin_memory=device.type == "cuda",
            prefetch_factor=4 if args.workers else None,
            persistent_workers=bool(args.workers),
        )
    )
    model.train()
    t0 = time.time()
    for step in range(start, args.steps):
        for group in opt.param_groups:
            group["lr"] = lr_at(step, args.steps, args.lr, args.warmup)
        batch = next(loader)
        if len(batch["anchor_row"]) == 0:
            continue
        batch = {key: v.to(device, non_blocking=True) for key, v in batch.items()}
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
            run.log({f"train/{k_}": v for k_, v in logs.items()}, step=step + 1)
            print(
                json.dumps(
                    {"step": step + 1, **{k_: round(v, 4) for k_, v in logs.items()}}
                ),
                flush=True,
            )
        if (step + 1) % args.val_every == 0 or step + 1 == args.steps:
            real = correct_per_anchor(
                model, val, val_anchors, swap=False, device=device
            )
            swapped = correct_per_anchor(
                model, val, val_anchors, swap=True, device=device
            )
            res = summarize(real, swapped, val_anchors)
            run.log({f"val/{k_}": v for k_, v in res.items()}, step=step + 1)
            print(json.dumps({"step": step + 1, "val": res}), flush=True)
            state = {
                "model": model.state_dict(),
                "opt": opt.state_dict(),
                "step": step + 1,
                "best": max(best, res["acc_all"]),
                "cfg": vars(cfg),
            }
            if res["acc_all"] > best:
                best = res["acc_all"]
                torch.save(
                    {
                        "model": model.state_dict(),
                        "step": step + 1,
                        "val": res,
                        "cfg": vars(cfg),
                    },
                    out / "best.pt",
                )
            torch.save(state, out / "last.tmp")
            os.replace(out / "last.tmp", out / "last.pt")
            if step + 1 == STOP_STEP:
                (out / "status_5000.json").write_text(
                    json.dumps({"acc_all": res["acc_all"]})
                )
                if res["acc_all"] < STOP_ACC and stop_with_peer(args.peer):
                    print(
                        f"STOP RULE: val acc_all {res['acc_all']:.4f} < {STOP_ACC} for both seeds at {STOP_STEP}",
                        flush=True,
                    )
                    (out / "STOPPED").write_text("stop rule at 5000\n")
                    run.finish()
                    return
    (out / "DONE").write_text(f"{best}\n")
    run.finish()
    print("TRAIN DONE", best, flush=True)


def stop_with_peer(peer: str) -> bool:
    if not peer:
        return True
    path = Path(peer)
    for _ in range(180):
        if path.exists():
            return json.loads(path.read_text())["acc_all"] < STOP_ACC
        time.sleep(10)
    print("peer status not found in 30 min; not stopping", flush=True)
    return False


if __name__ == "__main__":
    main()
