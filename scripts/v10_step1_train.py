"""Train one arm of the V10 step 1 experiment.

    python scripts/v10_step1_train.py --config C.yaml --arm B --seed 0 \
        --corpus PACKED --epochs 3 --output RUN

All arms of a seed share their chunks, order, step count and initial weights;
only the packed ``m`` file differs. The m-only arm also drops ``s`` and CE(s).
Checkpoints resume from ``latest.pt``.
"""

import argparse
import dataclasses
import json
import math
import random
import subprocess
import time
import uuid
from pathlib import Path

import airon
import torch
from torch.utils.data import DataLoader

from dauys.v02_data import sha256
from dauys.v02_runner import _durable_save, _truncate_metrics_to_step
from dauys.v02_wandb import start_private_run
from dauys.v10_step1 import TRAIN_ARMS, V10Step1Config, V10Step1Model
from dauys.v10_step1_data import (
    PackedCorpus,
    V10BatchSource,
    steps_for_epochs,
)


def _git() -> tuple[str, bool]:
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    dirty = bool(
        subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=all"], text=True
        ).strip()
    )
    return sha, dirty


def learning_rate(
    step: int, *, steps: int, peak: float, warmup: int, floor: float = 0.1
) -> float:
    """Linear warmup, then cosine decay to ``floor`` of the peak."""
    if step < warmup:
        return peak * (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, steps - warmup)
    return peak * (floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * progress)))


def optimize_step(
    model: V10Step1Model,
    batch: dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    *,
    max_grad_norm: float = 1.0,
    use_s: bool = True,
):
    """CE(m) + CE(s) on one shared batch; the arms differ only in ``batch["m"]``."""
    optimizer.zero_grad(set_to_none=True)
    device = next(model.parameters()).device
    with torch.autocast(
        device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
    ):
        output = model(batch["m"], batch["s"])
        losses = model.loss(
            output, batch["m"], batch["s"], batch["lengths"], use_s=use_s
        )
    losses.total.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()
    return losses


def _save(
    path: Path,
    model: V10Step1Model,
    optimizer: torch.optim.Optimizer,
    *,
    step: int,
    config: V10Step1Config,
    arm: str,
    seed: int,
) -> None:
    _durable_save(
        path,
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "config": dataclasses.asdict(config),
            "arm": arm,
            "seed": seed,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--arm", choices=TRAIN_ARMS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--steps", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunk-frames", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--checkpoint-interval", type=int, default=1_000)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if (args.steps > 0) == (args.epochs > 0):
        raise ValueError("Give exactly one of --steps and --epochs")
    for value in (args.chunk_frames, args.batch_size, args.checkpoint_interval):
        if value < 1:
            raise ValueError("Chunk, batch and checkpoint sizes must be positive")
    if args.warmup_steps < 0 or args.learning_rate <= 0:
        raise ValueError("Invalid learning-rate settings")
    config = V10Step1Config.from_yaml(args.config)
    corpus = PackedCorpus.open(args.corpus, config, args.arm)
    steps = (
        args.steps
        if args.steps > 0
        else steps_for_epochs(
            corpus.frames,
            chunk_frames=args.chunk_frames,
            batch_size=args.batch_size,
            epochs=args.epochs,
        )
    )
    root = args.output
    start_step = 0
    if args.resume:
        if not (root / "latest.pt").is_file():
            raise ValueError("Resume needs an existing latest.pt")
        payload = torch.load(root / "latest.pt", map_location="cpu", weights_only=True)
        previous = V10Step1Config(**payload["config"])
        if previous != config:
            raise ValueError("The checkpoint config does not match --config")
        if (payload["arm"], payload["seed"]) != (args.arm, args.seed):
            raise ValueError("The checkpoint belongs to another arm or seed")
        start_step = int(payload["step"])
        _truncate_metrics_to_step(root / "metrics.jsonl", start_step)
    elif (root / "metrics.jsonl").exists():
        raise FileExistsError(f"{root} already holds a run; use --resume")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Nothing in construction may depend on the arm, so one seed gives one start.
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    model = V10Step1Model(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    if start_step:
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
    print(
        json.dumps(
            {
                "config": config.name,
                "arm": args.arm,
                "seed": args.seed,
                "trainable_parameters": model.trainable_parameters(),
                "steps": steps,
                "start_step": start_step,
                "device": str(device),
            }
        ),
        flush=True,
    )
    raw = root / "metadata.json"
    if args.resume:
        metadata = json.loads(raw.read_text(encoding="utf-8"))
    else:
        root.mkdir(parents=True, exist_ok=True)
        git_sha, dirty = _git()
        metadata = {
            "config": dataclasses.asdict(config),
            "arm": args.arm,
            "seed": args.seed,
            "trainable_parameters": model.trainable_parameters(),
            "corpus": str(args.corpus),
            "corpus_meta_sha256": sha256(args.corpus / "meta.json"),
            "chunk_frames": args.chunk_frames,
            "batch_size": args.batch_size,
            "steps": steps,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "git": git_sha,
            "dirty": dirty,
            "airon_run_id": str(uuid.uuid4()),
        }
        raw.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    corpus_meta = json.loads((args.corpus / "meta.json").read_text(encoding="utf-8"))
    tracked = airon.init(
        project="dauys",
        name=f"v10-step1-{corpus_meta['tier']}-{args.arm}-s{args.seed}",
        config={**metadata, "tier": corpus_meta["tier"]},
        experiment="v10-step1",
        tags=["step-1", str(corpus_meta["tier"]), args.arm, f"seed-{args.seed}"],
        total_steps=steps,
        total_epochs=args.epochs or None,
        run_id=metadata["airon_run_id"],
        resume="allow" if args.resume else "never",
        mode="disabled" if args.smoke else None,
    )
    run = (
        start_private_run(
            "dauys",
            f"v10-step1-{args.arm}-s{args.seed}",
            {"config": dataclasses.asdict(config), "arm": args.arm, "seed": args.seed},
        )
        if args.wandb and not args.smoke
        else None
    )
    source = V10BatchSource(
        corpus,
        config,
        chunk_frames=args.chunk_frames,
        steps=steps,
        batch_size=args.batch_size,
        seed=args.seed,
        start=start_step,
    )
    loader = DataLoader(
        source, batch_size=None, num_workers=args.workers, pin_memory=True
    )
    model.train()
    mode = "a" if args.resume else "w"
    frames_seen = 0
    started = time.time()
    with (root / "metrics.jsonl").open(mode, encoding="utf-8") as stream:
        for step, batch in enumerate(loader, start=start_step + 1):
            batch = {key: value.to(device) for key, value in batch.items()}
            rate = learning_rate(
                step - 1,
                steps=steps,
                peak=args.learning_rate,
                warmup=args.warmup_steps,
            )
            for group in optimizer.param_groups:
                group["lr"] = rate
            losses = optimize_step(
                model,
                batch,
                optimizer,
                max_grad_norm=args.max_grad_norm,
                use_s=args.arm != "m-only",
            )
            frames_seen += int(batch["lengths"].sum())
            if step % args.log_interval == 0 or step == steps:
                elapsed = max(1e-9, time.time() - started)
                log = {
                    "step": float(step),
                    "train/loss": float(losses.total.detach()),
                    "train/ce_m": float(losses.m.detach()),
                    "train/ce_m_unit": float(losses.m_unit),
                    "train/ce_m_sil": float(losses.m_sil),
                    "train/ce_m_pad": float(losses.m_pad),
                    "train/lr": rate,
                    "train/frames_per_second": frames_seen / elapsed,
                }
                if args.arm != "m-only":
                    log["train/ce_s"] = float(losses.s.detach())
                stream.write(json.dumps(log) + "\n")
                stream.flush()
                print(json.dumps(log), flush=True)
                if run is not None:
                    run.log(log, step=step)
                tracked.log(
                    {key: value for key, value in log.items() if key != "step"},
                    step=step,
                    epoch=args.epochs * step / steps if args.epochs else None,
                    samples=args.batch_size,
                    tokens=int(batch["lengths"].sum()),
                )
                frames_seen = 0
                started = time.time()
            if step % args.checkpoint_interval == 0 or step == steps:
                _save(
                    root / "latest.pt",
                    model,
                    optimizer,
                    step=step,
                    config=config,
                    arm=args.arm,
                    seed=args.seed,
                )
    tracked.finish()
    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
