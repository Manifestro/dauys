"""Stateful Stage 1 training with Mamba carry across chunks (TBPTT)."""

import argparse
import json
import math
import random
import sys
from dataclasses import asdict
from pathlib import Path

import torch

from dauys.config import ModelConfig
from dauys.model import DauysHybrid
from dauys.v02_data import sha256
from dauys.v02_runner import (
    _durable_save,
    load_initial_weights,
    optimizer_groups,
)
from dauys.v02_stage1 import Stage1Weights
from dauys.v02_stateful import (
    _batch_to,
    chunk_loss,
    chunk_slice,
    load_streams,
    stack_streams,
    stateful_validation,
)

sys.path.insert(0, str(Path(__file__).parent))
from v02_stage1_train import _git


def compose_batch(
    train: list[dict],
    mixes: list[tuple[list[dict], int]],
    batch: int,
    rng: random.Random,
) -> list[dict]:
    """Take main streams plus fixed-size samples from each mix set."""
    mix_rows = sum(rows for _, rows in mixes)
    main_rows = batch - mix_rows
    if main_rows < 1:
        raise ValueError("The batch needs at least one main stream")
    lengths = {stream["user"].shape[-1] for stream in train}
    for streams, rows in mixes:
        if rows < 1:
            raise ValueError("Each mix needs a positive stream count")
        lengths |= {stream["user"].shape[-1] for stream in streams}
    if len(lengths) != 1:
        raise ValueError("All main and mixed streams must have the same length")
    selected = list(rng.sample(train, main_rows))
    for streams, rows in mixes:
        selected.extend(rng.sample(streams, rows))
    return selected


def train_chunk(
    model: DauysHybrid,
    optimizer,
    batch: dict[str, torch.Tensor],
    *,
    chunk: int,
    index: int,
    stateful: bool,
    states: dict | None,
    label_smoothing: float,
    scheduler=None,
    max_grad_norm: float = 1.0,
) -> dict:
    user, target, user_activity, activity, initial = chunk_slice(batch, chunk, index)
    mamba_states = states if stateful else None
    output = model(
        user,
        target,
        initial,
        mamba_states=mamba_states,
    )
    loss, cb0, depth, activity_loss = chunk_loss(
        model, output, target, activity, label_smoothing
    )
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    trainable = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    torch.nn.utils.clip_grad_norm_(trainable, max_grad_norm)
    optimizer.step()
    if scheduler is not None:
        scheduler.step()
    return {
        "chunk": index,
        "loss": float(loss.detach()),
        "cb0": float(cb0.detach()),
        "depth": float(depth.detach()),
        "activity": float(activity_loss.detach()),
        "user_activity": float(user_activity.sum()),
    }


def run_batch(
    model: DauysHybrid,
    optimizer,
    batch: dict[str, torch.Tensor],
    *,
    chunk: int,
    stateful: bool,
    label_smoothing: float = Stage1Weights().label_smoothing,
    scheduler=None,
    max_grad_norm: float = 1.0,
    max_chunks: int | None = None,
    step_callback=None,
) -> list[dict]:
    """Run one stream batch, stepping the optimizer after every chunk."""
    length = batch["user"].shape[-1]
    if length % chunk:
        raise ValueError("Stream length must be divisible by the chunk size")
    count = length // chunk
    if max_chunks is not None:
        count = min(count, max_chunks)
    states = {} if stateful else None
    logs = []
    for index in range(count):
        log = train_chunk(
            model,
            optimizer,
            batch,
            chunk=chunk,
            index=index,
            stateful=stateful,
            states=states,
            label_smoothing=label_smoothing,
            scheduler=scheduler,
            max_grad_norm=max_grad_norm,
        )
        logs.append(log)
        if step_callback is not None:
            step_callback(index, log)
    return logs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--heldout", type=Path, required=True)
    parser.add_argument("--init-checkpoint", type=Path)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument(
        "--mix", nargs="*", action="append", default=[], metavar="PATH:ROWS"
    )
    parser.add_argument("--stateful", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-interval", type=int, default=1000)
    parser.add_argument("--chunk", type=int, default=256)
    parser.add_argument("--require-clean", action="store_true")
    args = parser.parse_args()
    if min(args.steps, args.batch, args.checkpoint_interval, args.chunk) < 1:
        raise ValueError("Steps, batch, interval and chunk must be positive")
    config = ModelConfig.from_yaml(args.config)
    train = load_streams(args.data)
    heldout = load_streams(args.heldout)
    mix_specs = []
    for group in args.mix:
        for spec in group:
            path, rows = spec.rsplit(":", 1)
            count = int(rows)
            if count < 1:
                raise ValueError("Each mix needs a positive stream count")
            mix_specs.append((Path(path), count))
    mix_sets = [(load_streams(path), rows) for path, rows in mix_specs]
    if args.batch - sum(rows for _, rows in mix_sets) < 1:
        raise ValueError("The batch needs at least one main stream")
    git_sha, dirty = _git()
    if dirty and args.require_clean:
        raise ValueError("This run requires a clean checkout")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DauysHybrid(config)
    if args.init_checkpoint is not None:
        load_initial_weights(model, args.init_checkpoint, from_v01=False)
    model = model.to(device)
    optimizer = torch.optim.AdamW(optimizer_groups(model, 0.05), lr=args.lr)
    warmup = max(1, math.ceil(args.steps * 0.03))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: (
            (step + 1) / warmup
            if step < warmup
            else 0.5
            * (1 + math.cos(math.pi * (step - warmup) / max(1, args.steps - warmup)))
        ),
    )
    root = args.output
    root.mkdir(parents=True, exist_ok=True)
    if (root / "metrics.jsonl").exists():
        raise FileExistsError("Run directory already contains a run")
    metadata = {
        "plan": {
            "steps": args.steps,
            "lr": args.lr,
            "batch": args.batch,
            "stateful": args.stateful,
            "chunk": args.chunk,
            "checkpoint_interval": args.checkpoint_interval,
            "seed": args.seed,
            "mix": [{"path": str(path), "rows": rows} for path, rows in mix_specs],
        },
        "model": asdict(config),
        "provenance": {
            "git_sha": git_sha,
            "git_dirty": str(dirty).lower(),
            "config_sha256": sha256(args.config),
            "data_sha256": sha256(args.data),
            "heldout_sha256": sha256(args.heldout),
            "mix_sha256": [
                {"path": str(path), "sha256": sha256(path)} for path, _ in mix_specs
            ],
            "init_checkpoint": str(args.init_checkpoint or "none"),
        },
        "streams": len(train),
        "heldout_streams": len(heldout),
    }
    (root / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    label_smoothing = Stage1Weights().label_smoothing
    rng = random.Random(args.seed)
    step = 0
    best = float("inf")
    best_step = 0
    best_metrics: dict[str, float] | None = None
    validation_metrics: dict[str, float] = {}
    with (root / "metrics.jsonl").open("w", encoding="utf-8") as stream:
        while step < args.steps:
            batch = _batch_to(
                stack_streams(compose_batch(train, mix_sets, args.batch, rng)),
                device,
            )
            logs = []

            def after(index, log):
                nonlocal step, best, best_step, best_metrics, validation_metrics
                step += 1
                if step % args.checkpoint_interval == 0 or step == args.steps:
                    model.eval()
                    metrics = stateful_validation(
                        model,
                        heldout,
                        chunk=args.chunk,
                        batch_size=args.batch,
                    )
                    model.train()
                    chosen = (
                        metrics["eval/carry_cb0_mean"]
                        if args.stateful
                        else metrics["eval/reset_cb0_mean"]
                    )
                    if chosen < best:
                        best = chosen
                        best_step = step
                        best_metrics = metrics
                        _durable_save(
                            root / "best.pt",
                            {
                                "model": model.state_dict(),
                                "step": step,
                                "metadata": metadata,
                                "metrics": metrics,
                            },
                        )
                    _durable_save(
                        root / "latest.pt",
                        {
                            "model": model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "scheduler": scheduler.state_dict(),
                            "step": step,
                            "metadata": metadata,
                            "metrics": metrics,
                            "best": best,
                            "best_step": best_step,
                            "torch_rng": torch.get_rng_state(),
                        },
                    )
                    validation_metrics = metrics

            stream_log: dict[str, float] = {}
            logs = run_batch(
                model,
                optimizer,
                batch,
                chunk=args.chunk,
                stateful=args.stateful,
                label_smoothing=label_smoothing,
                scheduler=scheduler,
                max_chunks=args.steps - step,
                step_callback=after,
            )
            if logs:
                stream_log.update(validation_metrics)
                stream_log.update(
                    {
                        "step": float(step),
                        "train/loss": sum(log["loss"] for log in logs) / len(logs),
                        "train/cb0": sum(log["cb0"] for log in logs) / len(logs),
                        "train/depth": sum(log["depth"] for log in logs) / len(logs),
                        "train/activity": sum(log["activity"] for log in logs)
                        / len(logs),
                        "train/lr": optimizer.param_groups[0]["lr"],
                        **{f"train/cb0_k{log['chunk']}": log["cb0"] for log in logs},
                    }
                )
                stream.write(json.dumps(stream_log) + "\n")
                stream.flush()
            elif step == 0:
                raise ValueError("No optimizer step could be taken")
    summary = {
        "steps": step,
        "best": best,
        "best_step": best_step,
        "best_metrics": best_metrics,
        "checkpoint": str(root / "best.pt"),
    }
    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
