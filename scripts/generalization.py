"""Train on disjoint full conversations and reserve the test split."""

import argparse
import hashlib
import json
import math
import platform
import shutil
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
from datasets import Audio, load_dataset
from overfit import DATASET, REVISION, _pcm
from overfit_ten import Conversation, on_device, rollout

from dauys.checkpoint import load_research_checkpoint
from dauys.codec import decode, encode, load_mimi
from dauys.config import ModelConfig
from dauys.loss import listening_loss, training_loss
from dauys.model import DauysHybrid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--mimi-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--persistent", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--history-refresh-every", type=int, default=1_000)
    parser.add_argument("--eval-every", type=int, default=5_000)
    parser.add_argument("--save-every", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_selection(splits: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    if splits["dataset"] != DATASET or splits["dataset_revision"] != REVISION:
        raise ValueError("Dataset revision mismatch")
    selection = splits["e005"]
    if set(selection) != {"train", "validation", "test"}:
        raise ValueError("Expected frozen train, validation, and test selections")
    groups: list[set[str]] = []
    for split in ("train", "validation", "test"):
        rows = selection[split]
        if not rows or any(row["source_split"] != split for row in rows):
            raise ValueError(f"Invalid {split} selection")
        ids = [row["id"] for row in rows]
        if len(ids) != len(set(ids)) or any(int(row["frames"]) < 8 for row in rows):
            raise ValueError(f"Duplicate or short {split} selection")
        groups.append({row["scenario"] for row in rows})
    if any(groups[i] & groups[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("Scenario leakage across splits")
    expected = {"train": 75_000, "validation": 7_500, "test": 7_500}
    for split, frames in expected.items():
        actual = sum(int(row["frames"]) for row in selection[split])
        if abs(actual - frames) > 5:
            raise ValueError(f"Expected {frames} frames for {split}, got {actual}")
    return selection


def selected_conversations(
    selected: list[dict[str, Any]], split: str, codec: Any
) -> list[Conversation]:
    requested = {row["id"]: row for row in selected}
    dataset = load_dataset(DATASET, split=split, revision=REVISION, streaming=True)
    for column in ("channel_a", "channel_b"):
        if isinstance(dataset.features[column], Audio):
            dataset = dataset.cast_column(column, Audio(decode=False))
    found: dict[str, Conversation] = {}
    for row in dataset:
        identifier = row["id"]
        if identifier not in requested:
            continue
        frames = int(requested[identifier]["frames"])
        samples = frames * 1920
        user_pcm = _pcm(row["channel_a"])
        target_pcm = _pcm(row["channel_b"])
        if min(len(user_pcm), len(target_pcm)) < samples:
            raise ValueError(f"Audio shorter than selection: {identifier}")
        user_pcm = user_pcm[:samples].copy()
        target_pcm = target_pcm[:samples].copy()
        with torch.inference_mode():
            user = encode(codec, torch.from_numpy(user_pcm)[None, None].cuda())
            target = encode(codec, torch.from_numpy(target_pcm)[None, None].cuda())
        if user.shape != (1, 8, frames) or target.shape != user.shape:
            raise ValueError(f"Codec alignment mismatch: {identifier}")
        blocks = target_pcm.reshape(frames, 1920)
        activity = torch.from_numpy(
            (np.sqrt(np.mean(np.square(blocks), axis=1)) > 1e-4).astype(np.int64)
        )[None]
        found[identifier] = Conversation(
            identifier,
            user.cpu().clone(),
            target.cpu().clone(),
            activity,
            samples,
            user_pcm,
            target_pcm,
        )
        if len(found) == len(requested):
            break
    if set(found) != set(requested):
        raise ValueError(
            f"Missing {split} conversations: {set(requested) - set(found)}"
        )
    return [found[row["id"]] for row in selected]


def evaluate_teacher(
    model: DauysHybrid,
    cases: list[Conversation],
    silence: torch.Tensor,
    config: ModelConfig,
) -> dict[str, float]:
    model.eval()
    weighted_nll = 0.0
    weighted_objective = 0.0
    frames = 0
    with torch.inference_mode():
        for case in cases:
            user, target, activity = on_device(case)
            output = model(user, target, silence)
            losses = training_loss(
                output,
                target,
                activity,
                torch.ones_like(activity, dtype=torch.bool),
                config,
            )
            nll = losses.cb0 + 7 * losses.depth + losses.activity
            weighted_nll += float(nll) * case.frames
            weighted_objective += float(losses.total) * case.frames
            frames += case.frames
    return {
        "teacher_nll_nats_per_frame": weighted_nll / frames,
        "teacher_objective": weighted_objective / frames,
        "frames": float(frames),
    }


def evaluate_free(
    model: DauysHybrid,
    case: Conversation,
    codec: Any,
    silence: torch.Tensor,
    output: Path,
    label: str,
) -> dict[str, float]:
    model.eval()
    user, target, activity = on_device(case)
    with torch.inference_mode():
        generated, predicted_activity = rollout(model, user, silence)
        wave = decode(codec, generated)[0, 0].float().cpu().numpy()
    sf.write(output / f"{label}_user.wav", case.user_pcm, 24_000)
    sf.write(output / f"{label}_target.wav", case.target_pcm, 24_000)
    sf.write(output / f"{label}_generated.wav", wave[: case.samples], 24_000)
    true_positive = int((predicted_activity.bool() & activity.bool()).sum())
    predicted = int(predicted_activity.sum())
    target_count = int(activity.sum())
    return {
        "exact_token_fraction": float((generated == target).float().mean()),
        "exact_frames": float((generated == target).all(dim=1).sum()),
        "speech_activity_f1": 2 * true_positive / max(predicted + target_count, 1),
        "predicted_speech_frames": float(predicted),
        "target_speech_frames": float(target_count),
        "generated_rms": float(np.sqrt(np.mean(np.square(wave.astype(np.float64))))),
    }


def save_checkpoint(
    path: Path,
    persistent: Path,
    model: DauysHybrid,
    optimizer: torch.optim.Optimizer,
    metadata: dict[str, Any],
    step: int,
    validation: dict[str, float] | None,
) -> str:
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "metadata": metadata,
        "step": step,
        "validation": validation,
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all(),
    }
    temp = path.with_suffix(".tmp")
    torch.save(checkpoint, temp)
    temp.replace(path)
    persistent.parent.mkdir(parents=True, exist_ok=True)
    copied = persistent.with_suffix(".tmp")
    shutil.copyfile(path, copied)
    copied.replace(persistent)
    digest = sha256(path)
    if sha256(persistent) != digest:
        raise RuntimeError("Persistent checkpoint verification failed")
    return digest


def main() -> None:
    args = parse_args()
    if (
        args.steps < 10
        or min(args.history_refresh_every, args.eval_every, args.save_every) < 1
    ):
        raise ValueError("Invalid training schedule")
    splits = json.loads(args.splits.read_text(encoding="utf-8"))
    selection = validate_selection(splits)
    listening_steps = round(args.steps * 0.4)
    if args.prepare_only:
        print(
            json.dumps(
                {
                    "splits_sha256": sha256(args.splits),
                    "counts": {name: len(rows) for name, rows in selection.items()},
                    "frames": {
                        name: sum(row["frames"] for row in rows)
                        for name, rows in selection.items()
                    },
                    "listening_steps": listening_steps,
                    "response_steps": args.steps - listening_steps,
                }
            )
        )
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.output.exists() and args.resume is None:
        raise FileExistsError(args.output)
    if args.persistent.exists() and args.resume is None:
        raise FileExistsError(args.persistent)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    started = time.monotonic()
    args.output.mkdir(parents=True, exist_ok=True)
    args.persistent.mkdir(parents=True, exist_ok=True)
    metadata_path = args.output / "metadata.json"
    codec = load_mimi(args.mimi_checkpoint, "cuda")
    train = selected_conversations(selection["train"], "train", codec)
    validation = selected_conversations(selection["validation"], "validation", codec)
    with torch.inference_mode():
        silence = encode(codec, torch.zeros((1, 1, 1920), device="cuda"))[:, :, 0]
    config = ModelConfig.stage1()
    model = DauysHybrid(config).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    metadata = {
        "experiment_id": args.output.name,
        "dataset": DATASET,
        "dataset_revision": REVISION,
        "manifest_sha256": splits["manifest_sha256"],
        "splits_sha256": sha256(args.splits),
        "code_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "code_clean": not subprocess.check_output(
            ["git", "status", "--porcelain"], text=True
        ).strip(),
        "model_config": asdict(config),
        "trainable_parameters": model.trainable_parameters(),
        "seed": args.seed,
        "steps": args.steps,
        "listening_steps": listening_steps,
        "learning_rate": args.learning_rate,
        "history_refresh_every": args.history_refresh_every,
        "eval_every": args.eval_every,
        "save_every": args.save_every,
        "train_ids": [x.identifier for x in train],
        "validation_ids": [x.identifier for x in validation],
        "train_frames": sum(x.frames for x in train),
        "validation_frames": sum(x.frames for x in validation),
        "test_frames_reserved": sum(row["frames"] for row in selection["test"]),
        "environment": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
        },
    }
    first_step = 1
    best_nll = math.inf
    last_validation: dict[str, float] | None = None
    if args.resume is not None:
        checkpoint = load_research_checkpoint(args.resume)
        old = checkpoint["metadata"]
        for key in (
            "dataset_revision",
            "splits_sha256",
            "model_config",
            "seed",
            "steps",
            "listening_steps",
            "history_refresh_every",
        ):
            if old[key] != metadata[key]:
                raise ValueError(f"Resume mismatch: {key}")
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        torch.set_rng_state(checkpoint["torch_rng"])
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
        first_step = int(checkpoint["step"]) + 1
        if first_step > args.steps:
            raise ValueError("Checkpoint is already at final step")
        best_path = args.persistent / "checkpoint_best.pt"
        if best_path.exists():
            best = load_research_checkpoint(best_path)
            best_nll = float(best["validation"]["teacher_nll_nats_per_frame"])
        last_validation = checkpoint["validation"]
        metadata = old
    if args.preflight_only:
        longest = max(train, key=lambda case: case.frames)
        user, target, activity = on_device(longest)
        mask = torch.ones_like(activity, dtype=torch.bool)
        model.train()
        prediction = model(user, target, silence, initial_user_codes=silence)
        listen_loss = training_loss(prediction, target, activity, mask, config).total
        listen_loss += listening_loss(prediction, user, mask, config)
        listen_loss.backward()
        model.zero_grad(set_to_none=True)
        model.eval()
        with torch.inference_mode():
            history, _ = rollout(model, user, silence)
        model.train()
        prediction = model(user, target, silence, history_codes=history.clone())
        response_loss = training_loss(prediction, target, activity, mask, config).total
        response_loss.backward()
        model.zero_grad(set_to_none=True)
        result = {
            "longest_train_id": longest.identifier,
            "longest_train_frames": longest.frames,
            "listen_loss": float(listen_loss.detach()),
            "response_loss": float(response_loss.detach()),
            "validation": evaluate_teacher(model, validation, silence, config),
            "duration_seconds": round(time.monotonic() - started, 3),
            "peak_vram_bytes": torch.cuda.max_memory_allocated(),
        }
        print(json.dumps(result), flush=True)
        return
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    shutil.copyfile(metadata_path, args.persistent / "metadata.json")
    metrics = args.output / "metrics.jsonl"
    if args.resume is not None and not metrics.exists():
        saved_metrics = args.persistent / "metrics.jsonl"
        if not saved_metrics.exists():
            raise FileNotFoundError(saved_metrics)
        shutil.copyfile(saved_metrics, metrics)
    if args.resume is not None:
        records = [json.loads(line) for line in metrics.read_text().splitlines()]
        records = [record for record in records if record["step"] < first_step]
        if not records or records[-1]["step"] != first_step - 1:
            raise ValueError("Metrics do not reach the resume checkpoint")
        metrics.write_text("".join(json.dumps(record) + "\n" for record in records))
    mode = "a" if args.resume is not None else "w"
    cache: dict[str, torch.Tensor] = {}
    with metrics.open(mode, encoding="utf-8") as log:
        for step in range(first_step, args.steps + 1):
            case = train[(step - 1) % len(train)]
            user, target, activity = on_device(case)
            listen = step <= listening_steps
            history = None
            refreshed = False
            if not listen:
                if (
                    not cache
                    or (step - listening_steps - 1) % args.history_refresh_every == 0
                ):
                    model.eval()
                    with torch.inference_mode():
                        for source in train:
                            source_user, _, _ = on_device(source)
                            generated, _ = rollout(model, source_user, silence)
                            cache[source.identifier] = generated.cpu().clone()
                    refreshed = True
                history = cache[case.identifier].cuda()
            model.train()
            optimizer.zero_grad(set_to_none=True)
            prediction = model(
                user,
                target,
                silence,
                initial_user_codes=silence if listen else None,
                history_codes=history,
            )
            mask = torch.ones_like(activity, dtype=torch.bool)
            response = training_loss(prediction, target, activity, mask, config)
            user_loss = (
                listening_loss(prediction, user, mask, config) if listen else None
            )
            loss = response.total + (user_loss if user_loss is not None else 0)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError(f"Non-finite loss at step {step}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            record: dict[str, Any] = {
                "step": step,
                "conversation_id": case.identifier,
                "phase": "listen" if listen else "respond",
                "loss": float(loss.detach()),
                "response_cb0": float(response.cb0.detach()),
                "response_depth": float(response.depth.detach()),
                "elapsed_seconds": round(time.monotonic() - started, 3),
            }
            if refreshed:
                record["history_refreshed"] = True
            if step % args.eval_every == 0 or step == args.steps:
                last_validation = evaluate_teacher(model, validation, silence, config)
                record["validation"] = last_validation
                nll = last_validation["teacher_nll_nats_per_frame"]
                if nll < best_nll:
                    best_nll = nll
                    record["best_checkpoint_sha256"] = save_checkpoint(
                        args.output / "checkpoint_best.pt",
                        args.persistent / "checkpoint_best.pt",
                        model,
                        optimizer,
                        metadata,
                        step,
                        last_validation,
                    )
            if step % args.save_every == 0 or step == args.steps:
                record["latest_checkpoint_sha256"] = save_checkpoint(
                    args.output / "checkpoint_latest.pt",
                    args.persistent / "checkpoint_latest.pt",
                    model,
                    optimizer,
                    metadata,
                    step,
                    last_validation,
                )
            log.write(json.dumps(record) + "\n")
            log.flush()
            if step % args.save_every == 0 or step == args.steps:
                shutil.copyfile(metrics, args.persistent / "metrics.jsonl")
            if step == first_step or step % 100 == 0 or "validation" in record:
                print(json.dumps(record), flush=True)
    if last_validation is None:
        raise RuntimeError("Validation was not evaluated")
    best = load_research_checkpoint(args.persistent / "checkpoint_best.pt")
    model.load_state_dict(best["model"], strict=True)
    train_case = max(train, key=lambda case: int(case.activity.sum()))
    val_case = max(validation, key=lambda case: int(case.activity.sum()))
    fixed = {
        "train": evaluate_free(model, train_case, codec, silence, args.output, "train"),
        "validation": evaluate_free(
            model, val_case, codec, silence, args.output, "validation"
        ),
    }
    for wav in args.output.glob("*.wav"):
        shutil.copyfile(wav, args.persistent / wav.name)
    summary = {
        "best_step": int(best["step"]),
        "best_validation": best["validation"],
        "fixed_audio": fixed,
        "best_checkpoint_sha256": sha256(args.persistent / "checkpoint_best.pt"),
        "latest_checkpoint_sha256": sha256(args.persistent / "checkpoint_latest.pt"),
        "duration_seconds": round(time.monotonic() - started, 3),
        "peak_vram_bytes": torch.cuda.max_memory_allocated(),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    shutil.copyfile(args.output / "summary.json", args.persistent / "summary.json")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
