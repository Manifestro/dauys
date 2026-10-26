"""Train concurrent audio streams on multiple disjoint dialogue excerpts."""

import argparse
import hashlib
import json
import platform
import random
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
from datasets import Audio, load_dataset
from overfit import DATASET, REVISION, _pcm

from dauys.checkpoint import load_research_checkpoint
from dauys.codec import decode, encode, load_mimi
from dauys.config import ModelConfig
from dauys.loss import training_loss
from dauys.model import DauysHybrid


@dataclass
class Window:
    conversation_id: str
    start_frame: int
    user: torch.Tensor
    target: torch.Tensor
    activity: torch.Tensor
    user_pcm: np.ndarray | None = None
    target_pcm: np.ndarray | None = None

    @property
    def frames(self) -> int:
        return self.user.shape[-1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=("e004", "e005"), required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--mimi-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--eval-every", type=int, default=32)
    parser.add_argument("--frames", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--resume", type=Path)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _spans(frames: int, width: int) -> list[tuple[int, int]]:
    spans = [(start, min(start + width, frames)) for start in range(0, frames, width)]
    if len(spans) > 1 and spans[-1][1] - spans[-1][0] < 8:
        spans[-2] = (spans[-2][0], spans[-1][1])
        spans.pop()
    return spans


def _load_windows(
    selected: list[dict[str, Any]],
    split: str,
    codec: Any,
    width: int,
) -> list[Window]:
    requests = {item["id"]: item["frames"] for item in selected}
    dataset = load_dataset(DATASET, split=split, revision=REVISION, streaming=True)
    for column in ("channel_a", "channel_b"):
        if isinstance(dataset.features[column], Audio):
            dataset = dataset.cast_column(column, Audio(decode=False))
    windows = []
    seen = set()
    for row in dataset:
        identifier = row["id"]
        if identifier not in requests:
            continue
        frames = requests[identifier]
        user_pcm, target_pcm = _pcm(row["channel_a"]), _pcm(row["channel_b"])
        if min(len(user_pcm), len(target_pcm)) < frames * 1920:
            raise ValueError(f"Conversation {identifier} is shorter than its selection")
        for start, end in _spans(frames, width):
            first, last = start * 1920, end * 1920
            user_piece = user_pcm[first:last].copy()
            target_piece = target_pcm[first:last].copy()
            with torch.inference_mode():
                user = encode(codec, torch.from_numpy(user_piece)[None, None].cuda())[
                    0
                ].cpu()
                target = encode(
                    codec, torch.from_numpy(target_piece)[None, None].cuda()
                )[0].cpu()
            if user.shape != target.shape or user.shape[-1] != end - start:
                raise ValueError(f"Codec alignment failed for {identifier}")
            block = target_piece.reshape(end - start, 1920)
            rms = np.sqrt(np.mean(np.square(block), axis=1))
            windows.append(
                Window(
                    conversation_id=identifier,
                    start_frame=start,
                    user=user,
                    target=target,
                    activity=torch.from_numpy((rms > 1e-4).astype(np.int64)),
                    user_pcm=user_piece,
                    target_pcm=target_piece,
                )
            )
        seen.add(identifier)
        if len(seen) == len(requests):
            break
    if seen != set(requests):
        raise ValueError(
            f"Missing {split} conversations: {sorted(set(requests) - seen)}"
        )
    return windows


def _on_device(window: Window) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        window.user[None].cuda(),
        window.target[None].cuda(),
        window.activity[None].cuda(),
    )


def _rollout(
    model: DauysHybrid, user: torch.Tensor, silence: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    stream = model.new_stream(silence)
    codes = []
    activity = []
    for frame in range(user.shape[-1]):
        generated, activity_logits = model.generate_step(user[:, :, frame], stream)
        codes.append(generated)
        activity.append(activity_logits.argmax(dim=-1))
    return torch.stack(codes, dim=-1), torch.stack(activity, dim=-1)


def _nll(losses: Any) -> float:
    return float((losses.cb0 + 7 * losses.depth + losses.activity).detach())


def _evaluate(
    model: DauysHybrid,
    windows: list[Window],
    silence: torch.Tensor,
    config: ModelConfig,
) -> dict[str, float]:
    model.eval()
    total_frames = 0
    total_nll = 0.0
    total_objective = 0.0
    with torch.inference_mode():
        for window in windows:
            user, target, activity = _on_device(window)
            prediction = model(user, target, silence)
            mask = torch.ones_like(activity, dtype=torch.bool)
            losses = training_loss(prediction, target, activity, mask, config)
            total_frames += window.frames
            total_nll += _nll(losses) * window.frames
            total_objective += float(losses.total) * window.frames
    return {
        "teacher_nll_nats_per_frame": total_nll / total_frames,
        "teacher_objective": total_objective / total_frames,
        "frames": float(total_frames),
    }


def _fixed_audio(
    label: str,
    window: Window,
    model: DauysHybrid,
    codec: Any,
    silence: torch.Tensor,
    output: Path,
) -> dict[str, float]:
    user, target, activity = _on_device(window)
    with torch.inference_mode():
        generated, predicted_activity = _rollout(model, user, silence)
        prediction = model(user, target, silence, history_codes=generated)
        losses = training_loss(
            prediction,
            target,
            activity,
            torch.ones_like(activity, dtype=torch.bool),
            model.config,
        )
        generated_pcm = decode(codec, generated)[0, 0].float().cpu().numpy()
    if window.user_pcm is not None and window.target_pcm is not None:
        sf.write(output / f"{label}_user.wav", window.user_pcm, 24_000)
        sf.write(output / f"{label}_target.wav", window.target_pcm, 24_000)
    sf.write(output / f"{label}_generated.wav", generated_pcm, 24_000)
    true_positive = int((predicted_activity.bool() & activity.bool()).sum())
    predicted = int(predicted_activity.sum())
    target_count = int(activity.sum())
    return {
        "free_running_history_nll_nats_per_frame": _nll(losses),
        "speech_activity_f1": 2 * true_positive / max(predicted + target_count, 1),
        "predicted_speech_frames": float(predicted),
        "target_speech_frames": float(target_count),
        "generated_rms": float(
            np.sqrt(np.mean(np.square(generated_pcm.astype(np.float64))))
        ),
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Multi-excerpt training requires CUDA")
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.steps < 1 or args.eval_every < 1 or not 8 <= args.frames <= 512:
        raise ValueError("Invalid training schedule or frame width")
    splits = json.loads(args.splits.read_text(encoding="utf-8"))
    if splits["dataset"] != DATASET or splits["dataset_revision"] != REVISION:
        raise ValueError("Data selection does not match the pinned dataset")
    experiment = splits[args.experiment]
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    args.output.mkdir(parents=True)
    started = time.monotonic()
    codec = load_mimi(args.mimi_checkpoint, "cuda")
    train_windows = _load_windows(experiment["train"], "train", codec, args.frames)
    validation_windows = _load_windows(
        experiment["validation"], "validation", codec, args.frames
    )
    with torch.inference_mode():
        silence = encode(codec, torch.zeros((1, 1, 1920), device="cuda"))[:, :, 0]
    config = ModelConfig.stage1()
    model = DauysHybrid(config).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    resume_sha = None
    resume_step = None
    if args.resume is not None:
        checkpoint = load_research_checkpoint(args.resume)
        if checkpoint["metadata"]["model_config"] != asdict(config):
            raise ValueError("Resume checkpoint has a different model configuration")
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        for group in optimizer.param_groups:
            group["lr"] = args.learning_rate
        resume_sha = _sha256(args.resume)
        resume_step = int(checkpoint["step"])
    metadata = {
        "experiment_id": args.output.name,
        "dataset": DATASET,
        "dataset_revision": REVISION,
        "manifest_sha256": splits["manifest_sha256"],
        "splits_sha256": _sha256(args.splits),
        "code_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "code_clean": not subprocess.check_output(
            ["git", "status", "--porcelain"], text=True
        ).strip(),
        "model_config": asdict(config),
        "model_parameters": model.trainable_parameters(),
        "seed": args.seed,
        "steps": args.steps,
        "frames_per_window": args.frames,
        "learning_rate": args.learning_rate,
        "train_conversations": [row["id"] for row in experiment["train"]],
        "validation_conversations": [row["id"] for row in experiment["validation"]],
        "train_frames": sum(row["frames"] for row in experiment["train"]),
        "validation_frames": sum(row["frames"] for row in experiment["validation"]),
        "train_windows": len(train_windows),
        "validation_windows": len(validation_windows),
        "resume_checkpoint_sha256": resume_sha,
        "resume_source_step": resume_step,
        "environment": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "cuda": torch.version.cuda,
        },
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    best_validation = float("inf")
    order = list(range(len(train_windows)))
    random.shuffle(order)
    with (args.output / "metrics.jsonl").open("w") as log:
        for step in range(1, args.steps + 1):
            if (step - 1) % len(order) == 0 and step > 1:
                random.shuffle(order)
            window = train_windows[order[(step - 1) % len(order)]]
            user, target, activity = _on_device(window)
            model.eval()
            with torch.inference_mode():
                history, _ = _rollout(model, user, silence)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            prediction = model(user, target, silence, history_codes=history.clone())
            loss = training_loss(
                prediction,
                target,
                activity,
                torch.ones_like(activity, dtype=torch.bool),
                config,
            )
            if not bool(torch.isfinite(loss.total)):
                raise RuntimeError(f"Non-finite loss at step {step}")
            loss.total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            record: dict[str, Any] = {
                "step": step,
                "train_conversation": window.conversation_id,
                "train_start_frame": window.start_frame,
                "train_nll_nats_per_frame": _nll(loss),
                "train_objective": float(loss.total.detach()),
                "elapsed_seconds": round(time.monotonic() - started, 3),
            }
            if step % args.eval_every == 0 or step == args.steps:
                validation = _evaluate(model, validation_windows, silence, config)
                record["validation"] = validation
                if validation["teacher_nll_nats_per_frame"] < best_validation:
                    best_validation = validation["teacher_nll_nats_per_frame"]
                    torch.save(
                        {
                            "model": model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "metadata": metadata,
                            "step": step,
                            "validation": validation,
                        },
                        args.output / "checkpoint_best.pt",
                    )
                    record["best_checkpoint"] = True
            log.write(json.dumps(record) + "\n")
            log.flush()
            if step == 1 or step % 10 == 0 or "validation" in record:
                print(json.dumps(record), flush=True)
    best = load_research_checkpoint(args.output / "checkpoint_best.pt")
    model.load_state_dict(best["model"], strict=True)
    model.eval()
    fixed_train = max(train_windows, key=lambda window: int(window.activity.sum()))
    fixed_validation = max(
        validation_windows, key=lambda window: int(window.activity.sum())
    )
    fixed = {
        "train": _fixed_audio("train", fixed_train, model, codec, silence, args.output),
        "validation": _fixed_audio(
            "validation", fixed_validation, model, codec, silence, args.output
        ),
    }
    summary = {
        "best_step": best["step"],
        "best_validation": best["validation"],
        "fixed_audio": fixed,
        "checkpoint_sha256": _sha256(args.output / "checkpoint_best.pt"),
        "duration_seconds": round(time.monotonic() - started, 3),
        "peak_vram_bytes": torch.cuda.max_memory_allocated(),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    report = f"""# {args.output.name}: Multi-excerpt concurrent-stream training

Train and validation conversations are disjoint. The selected source test split
was not read during this training run. Each optimizer step uses generated
response history while the input stream supplies one frame at a time.

- Code revision: `{metadata["code_revision"]}`
- Dataset revision: `{REVISION}`
- Selection SHA-256: `{metadata["splits_sha256"]}`
- Train: {metadata["train_frames"] / 12.5 / 60:.3f} minutes across {len(experiment["train"])} conversations
- Validation: {metadata["validation_frames"] / 12.5 / 60:.3f} minutes across {len(experiment["validation"])} conversations
- Best step: {best["step"]}
- Best validation teacher-forced NLL: {best_validation:.4f} nats/frame
- Fixed train activity F1: {fixed["train"]["speech_activity_f1"]:.4f}
- Fixed validation activity F1: {fixed["validation"]["speech_activity_f1"]:.4f}

The teacher-forced NLL is measured over all selected validation windows. The
fixed audio metrics are free-running on one train and one validation window.
Listen to the WAVs to assess intelligibility and response relevance. These
numbers alone do not establish full-duplex interaction or semantic relevance.
"""
    (args.output / "report.md").write_text(report)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
