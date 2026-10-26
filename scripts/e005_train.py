"""Train Hybrid v0.1 from scratch on the frozen E005 token cache."""

import argparse
import base64
import hashlib
import json
import math
import os
import random
import subprocess
import time
import urllib.request
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import wandb
from e005_data import DATASET, REVISION, digest

from dauys.checkpoint import load_research_checkpoint
from dauys.codec import decode, encode, load_mimi
from dauys.config import ModelConfig
from dauys.loss import training_loss
from dauys.model import DauysHybrid, FrameAttention
from dauys.schedule import e005_phase

RUN_NAME = "S1-E005-hohere-1m-hybrid-v01"
EPOCHS = 6
BATCH_FRAMES = 8 * 256


def read_manifest(path: Path, split: str) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if not rows or len({row["id"] for row in rows}) != len(rows):
        raise ValueError(f"Invalid {split} manifest")
    for row in rows:
        if (
            row["dataset"] != DATASET
            or row["dataset_revision"] != REVISION
            or row["split"] not in ("train", "validation", "test")
            or row.get("selected_split", row["split"]) != split
        ):
            raise ValueError(f"Dataset identity mismatch: {row['id']}")
    return rows


def make_batches(rows: list[dict[str, Any]], seed: int) -> list[list[dict[str, Any]]]:
    order = rows.copy()
    random.Random(seed).shuffle(order)
    batches: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    frames = 0
    for row in order:
        if (
            current
            and frames + row["frames"] > BATCH_FRAMES
            and abs(frames - BATCH_FRAMES) <= abs(frames + row["frames"] - BATCH_FRAMES)
        ):
            batches.append(current)
            current, frames = [], 0
        current.append(row)
        frames += row["frames"]
    if current:
        batches.append(current)
    return batches


def set_context(model: DauysHybrid, frames: int) -> None:
    for module in model.modules():
        if isinstance(module, FrameAttention):
            module.window = frames


def load_case(root: Path, row: dict[str, Any]) -> dict[str, torch.Tensor]:
    path = root / row["split"] / f"{row['id']}.pt"
    if not path.is_file():
        raise FileNotFoundError(path)
    cached = torch.load(path, map_location="cpu", weights_only=True)
    if set(cached) != {"user", "target", "activity", "user_activity"}:
        raise ValueError(f"Invalid token cache: {row['id']}")
    if cached["user"].shape != (8, row["frames"]) or cached["target"].shape != (
        8,
        row["frames"],
    ):
        raise ValueError(f"Token geometry mismatch: {row['id']}")
    return cached


def verify_cache(root: Path, rows: list[dict[str, Any]]) -> None:
    for row in rows:
        path = root / row["split"] / f"{row['id']}.pt"
        if not path.is_file() or digest(path) != row["sha256"]:
            raise ValueError(f"Token cache checksum mismatch: {row['id']}")


def on_gpu(case: dict[str, torch.Tensor]) -> tuple[torch.Tensor, ...]:
    return (
        case["user"][None].long().cuda(non_blocking=True),
        case["target"][None].long().cuda(non_blocking=True),
        case["activity"][None].long().cuda(non_blocking=True),
    )


def own_history(
    model: DauysHybrid,
    user: torch.Tensor,
    target: torch.Tensor,
    silence: torch.Tensor,
    length: int,
    seed: int,
) -> torch.Tensor:
    frames = user.shape[-1]
    start = random.Random(seed).randrange(max(1, frames - length + 1))
    end = min(frames, start + length)
    model.eval()
    with torch.inference_mode():
        state = model.prefill_teacher(user[:, :, :start], target[:, :, :start], silence)
        generated = []
        for index in range(start, end):
            codes, _ = model.generate_step(user[:, :, index], state)
            generated.append(codes)
        history = target.clone()
        history[:, :, start:end] = torch.stack(generated, dim=-1)
    return history


def private_wandb_entity() -> str:
    api = wandb.Api(timeout=30)
    entity = api.default_entity
    if not entity or not os.environ.get("WANDB_API_KEY"):
        raise RuntimeError("W&B authentication is unavailable")
    query = {
        "query": f'query {{ project(name: "dauys", entityName: "{entity}") {{ access }} }}'
    }
    key = base64.b64encode(("api:" + os.environ["WANDB_API_KEY"]).encode()).decode()
    request = urllib.request.Request(
        "https://api.wandb.ai/graphql",
        data=json.dumps(query).encode(),
        headers={"Authorization": "Basic " + key, "Content-Type": "application/json"},
    )
    result = json.load(urllib.request.urlopen(request, timeout=30))
    if result.get("data", {}).get("project", {}).get("access") != "PRIVATE":
        raise RuntimeError("W&B project must be private before uploading")
    return entity


def save_checkpoint(
    path: Path,
    model: DauysHybrid,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    metadata: dict[str, Any],
    epoch: int,
    batch_index: int,
    context_frames: int,
    step: int,
    seen_frames: int,
    best_nll: float,
    stale_checks: int,
) -> str:
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "metadata": metadata,
        "epoch": epoch,
        "batch_index": batch_index,
        "context_frames": context_frames,
        "step": step,
        "seen_frames": seen_frames,
        "best_nll": best_nll,
        "stale_checks": stale_checks,
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all(),
        "python_rng": random.getstate(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    return digest(path)


def write_completion(output: Path, step: int, reason: str) -> None:
    best = output / "best_nll.pt"
    latest = output / "latest.pt"
    if not best.is_file() or not latest.is_file():
        raise RuntimeError("Training ended without durable checkpoints")
    (output / "completed.json").write_text(
        json.dumps(
            {
                "step": step,
                "reason": reason,
                "best_nll_sha256": digest(best),
                "latest_sha256": digest(latest),
            },
            indent=2,
        )
        + "\n"
    )


def validation(
    model: DauysHybrid,
    rows: list[dict[str, Any]],
    root: Path,
    silence: torch.Tensor,
    config: ModelConfig,
) -> dict[str, Any]:
    model.eval()
    total_frames = 0
    sums = np.zeros(10, dtype=np.float64)
    true_positive = predicted_positive = target_positive = 0
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for row in rows:
            user, target, activity = on_gpu(load_case(root, row))
            output = model(user, target, silence)
            frames = row["frames"]
            cb0 = F.cross_entropy(
                output.cb0_logits.reshape(-1, config.vocab_size),
                target[:, 0].reshape(-1),
            )
            assert output.depth_logits is not None
            depth = F.cross_entropy(
                output.depth_logits.reshape(-1, config.vocab_size),
                target[:, 1:].transpose(1, 2).reshape(-1),
                reduction="none",
            ).reshape(1, frames, 7)
            activity_loss = F.cross_entropy(
                output.activity_logits.reshape(-1, 2), activity.reshape(-1)
            )
            sums[0] += float(cb0) * frames
            for codebook in range(7):
                sums[1 + codebook] += float(depth[:, :, codebook].mean()) * frames
            sums[8] += float(activity_loss) * frames
            sums[9] += (
                float((output.cb0_logits.argmax(-1) == target[:, 0]).float().mean())
                * frames
            )
            predicted = output.activity_logits.argmax(-1).bool()
            actual = activity.bool()
            true_positive += int((predicted & actual).sum())
            predicted_positive += int(predicted.sum())
            target_positive += int(actual.sum())
            total_frames += frames
    mean = sums / total_frames
    return {
        "nll": float(mean[:9].sum()),
        "cb0_nll": float(mean[0]),
        "depth_nll": float(mean[1:8].mean()),
        "per_codebook_nll": [float(value) for value in mean[:8]],
        "activity_nll": float(mean[8]),
        "cb0_accuracy": float(mean[9]),
        "activity_f1": 2 * true_positive / max(1, predicted_positive + target_positive),
        "frames": total_frames,
    }


def fixed_examples(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_skill: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_skill.setdefault(row["skill"], []).append(row)
    result: list[dict[str, Any]] = []
    for skill in sorted(by_skill):
        result.append(
            min(
                by_skill[skill],
                key=lambda row: hashlib.sha256(row["id"].encode()).hexdigest(),
            )
        )
    remaining = sorted(
        (row for row in rows if row not in result),
        key=lambda row: hashlib.sha256(("42:" + row["id"]).encode()).hexdigest(),
    )
    return (result + remaining)[:16]


def conditional_evaluation(
    model: DauysHybrid,
    selected: list[dict[str, Any]],
    all_validation: list[dict[str, Any]],
    root: Path,
    silence: torch.Tensor,
    config: ModelConfig,
) -> dict[str, Any]:
    model.eval()
    cases = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for row in selected:
            choices = [
                item
                for item in all_validation
                if item["scenario_family"] != row["scenario_family"]
            ]
            alternate = min(
                choices,
                key=lambda item: (
                    item["voice_a"] != row["voice_a"],
                    item["frames"] < row["frames"],
                    abs(item["frames"] - row["frames"]),
                    abs(
                        item["user_active_frames"] / item["frames"]
                        - row["user_active_frames"] / row["frames"]
                    ),
                    item["id"],
                ),
            )
            user, target, activity = on_gpu(load_case(root, row))
            other, _, _ = on_gpu(load_case(root, alternate))
            shuffled = silence[:, :, None].expand_as(user).clone()
            overlap = min(row["frames"], alternate["frames"])
            shuffled[:, :, :overlap] = other[:, :, :overlap]
            modes = {
                "original": user,
                "silent": silence[:, :, None].expand_as(user),
                "shuffled": shuffled,
            }
            losses = {}
            for name, incoming in modes.items():
                output = model(incoming, target, silence)
                loss = training_loss(
                    output,
                    target,
                    activity,
                    torch.ones_like(activity, dtype=torch.bool),
                    config,
                )
                losses[name] = float(loss.cb0 + 7 * loss.depth + loss.activity)
            cases.append(
                {
                    "id": row["id"],
                    "alternate_id": alternate["id"],
                    "same_voice": alternate["voice_a"] == row["voice_a"],
                    "alternate_frames": alternate["frames"],
                    "frames": row["frames"],
                    "nll": losses,
                }
            )
    total_frames = sum(item["frames"] for item in cases)
    aggregate = {
        mode: sum(item["nll"][mode] * item["frames"] for item in cases) / total_frames
        for mode in ("original", "silent", "shuffled")
    }
    return {
        "aggregate_nll": aggregate,
        "silent_delta_nll": aggregate["silent"] - aggregate["original"],
        "shuffled_delta_nll": aggregate["shuffled"] - aggregate["original"],
        "cases": cases,
    }


def free_evaluation(
    model: DauysHybrid,
    codec: Any,
    rows: list[dict[str, Any]],
    root: Path,
    silence: torch.Tensor,
    output_dir: Path,
    step: int,
) -> tuple[dict[str, float], wandb.Table]:
    model.eval()
    table = wandb.Table(
        columns=[
            "conversation_id",
            "checkpoint",
            "user_audio",
            "target_audio",
            "generated_audio",
            "activity_f1",
            "notes",
        ]
    )
    total_tp = total_pred = total_target = 0
    for row in rows:
        cached = load_case(root, row)
        user, target, activity = on_gpu(cached)
        start = 0
        end = min(row["frames"], 256)
        with torch.inference_mode():
            state = model.new_stream(silence)
            previous = silence
            for index in range(start):
                model.stream_step_teacher(user[:, :, index], previous, state)
                previous = target[:, :, index]
            state.previous_output = previous
            generated, predicted = [], []
            for index in range(start, end):
                codes, logits = model.generate_step(user[:, :, index], state)
                generated.append(codes)
                predicted.append(logits.argmax(-1))
            generated_codes = torch.stack(generated, -1)
            predicted_activity = torch.stack(predicted, -1).bool()
            actual = activity[:, start:end].bool()
            total_tp += int((predicted_activity & actual).sum())
            total_pred += int(predicted_activity.sum())
            total_target += int(actual.sum())
            audio = []
            for codes in (
                user[:, :, start:end],
                target[:, :, start:end],
                generated_codes,
            ):
                audio.append(decode(codec, codes)[0, 0].float().cpu().numpy())
        case_dir = output_dir / f"step_{step:06d}"
        case_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        for label, wave in zip(("user", "target", "generated"), audio):
            path = case_dir / f"{row['id']}_{label}.wav"
            sf.write(path, wave, 24_000)
            paths.append(path)
        f1 = (
            2
            * int((predicted_activity & actual).sum())
            / max(1, int(predicted_activity.sum()) + int(actual.sum()))
        )
        table.add_data(
            row["id"],
            step,
            wandb.Audio(str(paths[0]), sample_rate=24_000),
            wandb.Audio(str(paths[1]), sample_rate=24_000),
            wandb.Audio(str(paths[2]), sample_rate=24_000),
            f1,
            "Blind intelligibility and relevance review pending",
        )
    return {
        "free_running_activity_f1": 2 * total_tp / max(1, total_pred + total_target),
        "generated_frames": total_pred,
        "target_frames": total_target,
    }, table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=Path, required=True)
    parser.add_argument("--mimi-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--splits", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--smoke-output", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--resume-code-migration", type=str)
    args = parser.parse_args()
    if args.resume_code_migration is not None and args.resume is None:
        raise ValueError("Code migration requires a saved checkpoint")
    if args.smoke_output is not None and not args.preflight_only:
        raise ValueError("Smoke output is only valid for preflight")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA with BF16 is required")
    if args.preflight_only and args.splits is None:
        manifest_root = args.tokens
    elif args.splits is not None:
        manifest_root = args.splits
    else:
        raise ValueError("Frozen split selections are required before training")
    train_path = manifest_root / "train.jsonl"
    val_path = manifest_root / "validation.jsonl"
    train = read_manifest(train_path, "train")
    val = read_manifest(val_path, "validation")
    if not args.preflight_only:
        if not 2000 <= len(train) <= 3511 or not 200 <= len(val) <= 1000:
            raise ValueError("Frozen train or validation selection is incomplete")
        verify_cache(args.tokens, train + val)
        test = read_manifest(manifest_root / "test.jsonl", "test")
        if not 200 <= len(test) <= 1200 or any(
            {row["scenario_family"] for row in test}
            & {row["scenario_family"] for row in rows}
            for rows in (train, val)
        ):
            raise ValueError("Incomplete or leaking test metadata")
        if {r["scenario_family"] for r in train} & {r["scenario_family"] for r in val}:
            raise ValueError("Scenario family leakage")
    if args.output.exists() and not args.preflight_only and args.resume is None:
        raise FileExistsError(args.output)
    if (
        not args.preflight_only
        and subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
    ):
        raise RuntimeError("Training must start from a clean Git checkout")
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    random.seed(42)
    codec = load_mimi(args.mimi_checkpoint, "cuda")
    with torch.inference_mode():
        silence = encode(codec, torch.zeros((1, 1, 1920), device="cuda"))[:, :, 0]
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    random.seed(42)
    config = ModelConfig.stage1()
    model = DauysHybrid(config).cuda()
    if model.trainable_parameters() != 60_873_210:
        raise ValueError("Model parameter count changed")
    set_context(model, 256)
    counts = [
        sum(row["frames"] - row["active_frames"] for row in train),
        sum(row["active_frames"] for row in train),
    ]
    activity_weights = torch.tensor(
        [sum(counts) / (2 * count) for count in counts],
        device="cuda",
        dtype=torch.float32,
    )
    if args.preflight_only:
        longest = max(train, key=lambda row: row["frames"])
        user, target, activity = on_gpu(load_case(args.tokens, longest))
        model.train()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(user, target, silence)
            loss = training_loss(
                output,
                target,
                activity,
                torch.ones_like(activity, dtype=torch.bool),
                config,
                activity_weights,
            ).total
        loss.backward()
        result: dict[str, Any] = {
            "longest_id": longest["id"],
            "frames": longest["frames"],
            "loss": float(loss.detach()),
            "peak_gpu_bytes": torch.cuda.max_memory_allocated(),
            "activity_weights": activity_weights.tolist(),
        }
        if args.smoke_output is not None:
            if args.smoke_output.exists():
                raise FileExistsError(args.smoke_output)
            result["validation"] = validation(
                model, val[:1], args.tokens, silence, config
            )
            free, _ = free_evaluation(
                model,
                codec,
                val[:1],
                args.tokens,
                silence,
                args.smoke_output,
                0,
            )
            result["free_running"] = free
        print(json.dumps(result), flush=True)
        return
    entity = private_wandb_entity()
    batches = [make_batches(train, 42 + epoch) for epoch in range(EPOCHS)]
    total_steps = sum(len(group) for group in batches)
    warmup = max(1, math.ceil(total_steps * 0.03))
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.01)

    def lr_factor(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        return 0.5 * (
            1 + math.cos(math.pi * (step - warmup) / max(1, total_steps - warmup))
        )

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    if args.resume is None:
        args.output.mkdir(parents=True)
    metadata = {
        "experiment": RUN_NAME,
        "dataset": DATASET,
        "dataset_revision": REVISION,
        "train_manifest_sha256": digest(train_path),
        "validation_manifest_sha256": digest(val_path),
        "test_manifest_sha256": digest(manifest_root / "test.jsonl"),
        "split_audit_sha256": digest(manifest_root / "audit.json"),
        "mimi_sha256": digest(args.mimi_checkpoint),
        "model_config": asdict(config),
        "parameters": model.trainable_parameters(),
        "git_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "seed": 42,
        "optimizer": "AdamW",
        "peak_lr": 2e-4,
        "weight_decay": 0.01,
        "warmup_steps": warmup,
        "planned_steps": total_steps,
        "planned_epochs": EPOCHS,
        "early_stopping_metric": "validation_nll",
        "early_stopping_patience": 3,
        "effective_batch_target_frames": BATCH_FRAMES,
        "effective_batch_actual_frames": "variable by whole conversation",
        "activity_class_weights": activity_weights.tolist(),
        "fixed_validation_ids": [row["id"] for row in fixed_examples(val)],
    }
    start_epoch = start_index = step = seen_frames = stale_checks = 0
    best_nll = math.inf
    if args.resume is not None:
        saved = load_research_checkpoint(args.resume)
        old = saved["metadata"]
        previous_git_sha = old["git_sha"]
        for key, value in metadata.items():
            if (
                key == "git_sha"
                and old[key] != value
                and args.resume_code_migration == previous_git_sha
            ):
                continue
            if old[key] != value:
                raise ValueError(f"Resume mismatch: {key}")
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        torch.set_rng_state(saved["torch_rng"])
        torch.cuda.set_rng_state_all(saved["cuda_rng"])
        random.setstate(saved["python_rng"])
        start_epoch = int(saved["epoch"])
        start_index = int(saved["batch_index"]) + 1
        step = int(saved["step"])
        seen_frames = int(saved["seen_frames"])
        stale_checks = int(saved["stale_checks"])
        best_nll = float(saved["best_nll"])
        metadata = old.copy()
        if previous_git_sha != metadata["git_sha"]:
            raise RuntimeError("Unexpected checkpoint code identity")
        current_git_sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
        if previous_git_sha != current_git_sha:
            migration_path = args.output / f"code-migration-step-{step:06d}.json"
            if migration_path.exists():
                raise FileExistsError(migration_path)
            migration_path.write_text(
                json.dumps(
                    {
                        "checkpoint_step": step,
                        "from_git_sha": previous_git_sha,
                        "to_git_sha": current_git_sha,
                        "reason": "Equivalent parallel teacher-prefix prefill",
                    },
                    indent=2,
                )
                + "\n"
            )
            metadata["code_history"] = [
                *old.get("code_history", [previous_git_sha]),
                current_git_sha,
            ]
            metadata["git_sha"] = current_git_sha
    else:
        (args.output / "config.json").write_text(json.dumps(metadata, indent=2) + "\n")
    wandb_run = wandb.init(
        project="dauys",
        entity=entity,
        name=RUN_NAME,
        config=metadata,
        dir=str(args.output),
        id=metadata.get("wandb_run_id"),
        resume="must" if args.resume is not None else None,
    )
    if wandb_run is None:
        raise RuntimeError("W&B did not start")
    metadata["wandb_run_id"] = wandb_run.id
    if args.resume is None:
        artifact = wandb.Artifact("S1-E005-protocol", type="protocol")
        artifact.add_file(str(args.output / "config.json"))
        artifact.add_file(str(train_path))
        artifact.add_file(str(val_path))
        artifact.add_file(str(manifest_root / "test.jsonl"))
        artifact.add_file(str(manifest_root / "audit.json"))
        wandb_run.log_artifact(artifact)
    started = time.monotonic()
    metrics_path = args.output / "metrics.jsonl"
    with metrics_path.open("a" if args.resume else "w") as log:
        for epoch, epoch_batches in enumerate(batches):
            if epoch < start_epoch:
                continue
            for index, batch in enumerate(epoch_batches):
                if epoch == start_epoch and index < start_index:
                    continue
                context, rollout_length = e005_phase(epoch, index, len(epoch_batches))
                set_context(model, context)
                total = sum(row["frames"] for row in batch)
                optimizer.zero_grad(set_to_none=True)
                batch_losses = np.zeros(4, dtype=np.float64)
                accuracy_counts = np.zeros(8, dtype=np.float64)
                for row in batch:
                    user, target, activity = on_gpu(load_case(args.tokens, row))
                    history = None
                    if (
                        rollout_length
                        and (step + int(row["id"].split("_")[-1])) % 4 == 0
                    ):
                        history = own_history(
                            model,
                            user,
                            target,
                            silence,
                            rollout_length,
                            42 + epoch * 100000 + int(row["id"].split("_")[-1]),
                        )
                    model.train()
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        output = model(user, target, silence, history_codes=history)
                        losses = training_loss(
                            output,
                            target,
                            activity,
                            torch.ones_like(activity, dtype=torch.bool),
                            config,
                            activity_weights,
                        )
                        weighted = losses.total * (row["frames"] / total)
                    if (step + 1) % 100 == 0:
                        accuracy_counts[0] += int(
                            (
                                output.cb0_logits.detach().argmax(-1) == target[:, 0]
                            ).sum()
                        )
                        assert output.depth_logits is not None
                        correct = (
                            output.depth_logits.detach().argmax(-1)
                            == target[:, 1:].transpose(1, 2)
                        ).sum((0, 1))
                        accuracy_counts[1:] += correct.cpu().numpy()
                    if not bool(torch.isfinite(weighted)):
                        raise RuntimeError(f"Non-finite loss at step {step + 1}")
                    weighted.backward()
                    batch_losses += (
                        np.array(
                            [
                                float(losses.total.detach()),
                                float(losses.cb0.detach()),
                                float(losses.depth.detach()),
                                float(losses.activity.detach()),
                            ]
                        )
                        * row["frames"]
                        / total
                    )
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                step += 1
                seen_frames += total
                record = {
                    "step": step,
                    "epoch": epoch + (index + 1) / len(epoch_batches),
                    "train_loss": float(batch_losses[0]),
                    "cb0_loss": float(batch_losses[1]),
                    "depth_loss": float(batch_losses[2]),
                    "activity_loss": float(batch_losses[3]),
                    "lr": optimizer.param_groups[0]["lr"],
                    "grad_norm": float(norm),
                    "frames_seen": seen_frames,
                    "context_frames": context,
                    "rollout_frames": rollout_length,
                    "elapsed_seconds": time.monotonic() - started,
                }
                if step % 20 == 0 or step == 1:
                    wandb_run.log(
                        {
                            "train/loss": record["train_loss"],
                            "train/cb0_loss": record["cb0_loss"],
                            "train/depth_loss": record["depth_loss"],
                            "train/activity_loss": record["activity_loss"],
                            "train/lr": record["lr"],
                            "train/grad_norm": record["grad_norm"],
                            "train/epoch": record["epoch"],
                            "train/frames_seen": seen_frames,
                        },
                        step=step,
                    )
                if step % 100 == 0:
                    wandb_run.log(
                        {
                            "train/cb0_accuracy": float(accuracy_counts[0] / total),
                            **{
                                f"train/cb{codebook}_accuracy": float(
                                    accuracy_counts[codebook] / total
                                )
                                for codebook in range(1, 8)
                            },
                            "train/gpu_memory_bytes": torch.cuda.max_memory_allocated(),
                            "train/steps_per_sec": step
                            / max(1, time.monotonic() - started),
                        },
                        step=step,
                    )
                checkpoint_due = index + 1 == len(
                    epoch_batches
                ) // 2 or index + 1 == len(epoch_batches)
                if checkpoint_due:
                    set_context(model, context)
                    score = validation(model, val, args.tokens, silence, config)
                    record["validation"] = score
                    conditional = conditional_evaluation(
                        model, fixed_examples(val), val, args.tokens, silence, config
                    )
                    record["input_dependency"] = conditional
                    best = score["nll"] < best_nll
                    best_nll = min(best_nll, score["nll"])
                    stale_checks = 0 if best else stale_checks + 1
                    latest_hash = save_checkpoint(
                        args.output / "latest.pt",
                        model,
                        optimizer,
                        scheduler,
                        metadata,
                        epoch,
                        index,
                        context,
                        step,
                        seen_frames,
                        best_nll,
                        stale_checks,
                    )
                    record["latest_sha256"] = latest_hash
                    if best:
                        record["best_sha256"] = save_checkpoint(
                            args.output / "best_nll.pt",
                            model,
                            optimizer,
                            scheduler,
                            metadata,
                            epoch,
                            index,
                            context,
                            step,
                            seen_frames,
                            best_nll,
                            stale_checks,
                        )
                    free, table = free_evaluation(
                        model,
                        codec,
                        fixed_examples(val),
                        args.tokens,
                        silence,
                        args.output / "audio",
                        step,
                    )
                    record["free_evaluation"] = free
                    evaluation_path = args.output / f"evaluation_{step:06d}.json"
                    evaluation_path.write_text(
                        json.dumps(
                            {
                                "step": step,
                                "validation": score,
                                "input_dependency": conditional,
                                "free_running": free,
                                "checkpoint_path": str(args.output / "latest.pt"),
                                "checkpoint_sha256": latest_hash,
                            },
                            indent=2,
                        )
                        + "\n"
                    )
                    results_artifact = wandb.Artifact(
                        f"S1-E005-evaluation-{step:06d}", type="evaluation"
                    )
                    results_artifact.add_file(str(evaluation_path))
                    wandb_run.log_artifact(results_artifact)
                    wandb_run.log(
                        {
                            "val/nll": score["nll"],
                            "val/cb0_nll": score["cb0_nll"],
                            "val/depth_nll": score["depth_nll"],
                            "val/activity_f1": score["activity_f1"],
                            **{
                                f"val/cb{codebook}_nll": nll
                                for codebook, nll in enumerate(
                                    score["per_codebook_nll"]
                                )
                            },
                            "eval/original_nll": conditional["aggregate_nll"][
                                "original"
                            ],
                            "eval/silent_delta_nll": conditional["silent_delta_nll"],
                            "eval/shuffled_delta_nll": conditional[
                                "shuffled_delta_nll"
                            ],
                            "eval/free_running_activity_f1": free[
                                "free_running_activity_f1"
                            ],
                            "eval/audio_samples": table,
                        },
                        step=step,
                    )
                log.write(json.dumps(record) + "\n")
                log.flush()
                if step % 20 == 0 or checkpoint_due:
                    print(json.dumps(record), flush=True)
                if checkpoint_due and stale_checks >= 3:
                    print(json.dumps({"early_stopped": True, "step": step}), flush=True)
                    wandb_run.finish()
                    write_completion(args.output, step, "early_stopping")
                    return
    wandb_run.finish()
    write_completion(args.output, step, "six_epochs")


if __name__ == "__main__":
    main()
