"""Stateful Stage 1 training with Mamba carry across chunks (TBPTT)."""

import argparse
import json
import math
import random
import sys
from dataclasses import asdict
from pathlib import Path

import torch
from torch.nn import functional as F

from dauys.config import ModelConfig
from dauys.model import DauysHybrid
from dauys.v02_data import sha256
from dauys.v02_runner import (
    _durable_save,
    load_initial_weights,
    optimizer_groups,
)
from dauys.v02_stage1 import Stage1Weights
from dauys.v02_training import masked_mean

sys.path.insert(0, str(Path(__file__).parent))
from v02_stage1_train import _git


def load_streams(path: str | Path) -> list[dict]:
    streams = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(streams, list) or not streams:
        raise ValueError("A stateful split must be a non-empty list of streams")
    return streams


def stack_streams(streams: list[dict]) -> dict[str, torch.Tensor]:
    return {
        key: torch.stack([stream[key] for stream in streams])
        for key in ("user", "target", "user_activity", "activity")
    }


def _batch_to(batch: dict[str, torch.Tensor], device: torch.device) -> dict:
    return {key: value.to(device) for key, value in batch.items()}


def chunk_slice(
    batch: dict[str, torch.Tensor], chunk: int, index: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    start = index * chunk
    end = start + chunk
    initial = (
        batch["target"][:, :, 0] if index == 0 else batch["target"][:, :, start - 1]
    )
    return (
        batch["user"][:, :, start:end],
        batch["target"][:, :, start:end],
        batch["user_activity"][:, start:end],
        batch["activity"][:, start:end],
        initial,
    )


def chunk_loss(
    model: DauysHybrid,
    output,
    target: torch.Tensor,
    activity: torch.Tensor,
    label_smoothing: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, frames = target.shape[0], target.shape[-1]
    vocab = model.config.vocab_size
    speech = activity.bool()
    cb0_values = F.cross_entropy(
        output.cb0_logits.reshape(-1, vocab),
        target[:, 0].reshape(-1),
        reduction="none",
        label_smoothing=label_smoothing,
    ).reshape(batch, frames)
    assert output.depth_logits is not None
    depth_values = F.cross_entropy(
        output.depth_logits.reshape(-1, vocab),
        target[:, 1:].transpose(1, 2).reshape(-1),
        reduction="none",
    ).reshape(batch, frames, 7)
    activity_values = F.cross_entropy(
        output.activity_logits.reshape(-1, 2),
        activity.reshape(-1).long(),
        reduction="none",
    ).reshape(batch, frames)
    cb0 = masked_mean(cb0_values, speech)
    depth = masked_mean(depth_values.mean(-1), speech)
    activity_loss = masked_mean(
        activity_values, torch.ones_like(activity, dtype=torch.bool)
    )
    loss = cb0 + depth + 0.2 * activity_loss
    return loss, cb0, depth, activity_loss


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


@torch.inference_mode()
def chunk_nlls(
    model: DauysHybrid,
    streams: list[dict],
    *,
    chunk: int,
    batch_size: int = 16,
    stateful: bool,
) -> dict[int, float]:
    device = next(model.parameters()).device
    chunks = streams[0]["user"].shape[-1] // chunk
    totals = torch.zeros(chunks, dtype=torch.float64)
    counts = torch.zeros(chunks, dtype=torch.float64)
    vocab = model.config.vocab_size
    for start in range(0, len(streams), batch_size):
        batch = _batch_to(stack_streams(streams[start : start + batch_size]), device)
        states = {} if stateful else None
        for index in range(chunks):
            user, target, _, activity, initial = chunk_slice(batch, chunk, index)
            output = model(
                user,
                target,
                initial,
                mamba_states=states if stateful else None,
            )
            speech = activity.bool()
            values = F.cross_entropy(
                output.cb0_logits.reshape(-1, vocab),
                target[:, 0].reshape(-1),
                reduction="none",
            ).reshape(target.shape[0], target.shape[-1])
            totals[index] += float((values * speech).sum())
            counts[index] += float(speech.sum())
    return {
        index: float(totals[index] / counts[index].clamp_min(1))
        for index in range(chunks)
    }


def _validation(
    model: DauysHybrid, heldout: list[dict], *, chunk: int, batch_size: int
) -> dict:
    carry = chunk_nlls(
        model, heldout, chunk=chunk, batch_size=batch_size, stateful=True
    )
    reset = chunk_nlls(
        model, heldout, chunk=chunk, batch_size=batch_size, stateful=False
    )
    metrics = {f"eval/carry_cb0_k{index}": value for index, value in carry.items()} | {
        f"eval/reset_cb0_k{index}": value for index, value in reset.items()
    }
    metrics["eval/carry_cb0_mean"] = sum(carry.values()) / len(carry)
    metrics["eval/reset_cb0_mean"] = sum(reset.values()) / len(reset)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--heldout", type=Path, required=True)
    parser.add_argument("--init-checkpoint", type=Path)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch", type=int, default=16)
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
        },
        "model": asdict(config),
        "provenance": {
            "git_sha": git_sha,
            "git_dirty": str(dirty).lower(),
            "config_sha256": sha256(args.config),
            "data_sha256": sha256(args.data),
            "heldout_sha256": sha256(args.heldout),
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
            indices = rng.sample(range(len(train)), min(args.batch, len(train)))
            batch = _batch_to(stack_streams([train[i] for i in indices]), device)
            logs = []

            def after(index, log):
                nonlocal step, best, best_step, best_metrics, validation_metrics
                step += 1
                if step % args.checkpoint_interval == 0 or step == args.steps:
                    model.eval()
                    metrics = _validation(
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
