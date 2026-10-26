"""Evaluate frozen E005 checkpoints on fixed train and validation conversations."""

import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import random
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import wandb
from e005_data import DATASET, REVISION, digest
from e005_train import (
    load_case,
    on_gpu,
    private_wandb_entity,
    read_manifest,
    set_context,
)

from dauys.checkpoint import load_research_checkpoint
from dauys.codec import decode, encode, load_mimi
from dauys.config import ModelConfig
from dauys.model import DauysHybrid, DialogueBlock

SAMPLE_SEED = 42
ROLLOUT_FRAMES = 160
WINDOW_FRAMES = 32
MIN_EARLY_ACTIVE_FRAMES = 16
MODES = ("original", "silent", "shuffled", "fusion_off", "acoustic")


def fixed_rows(
    rows: list[dict[str, Any]], count: int, token_root: Path
) -> list[dict[str, Any]]:
    ordered = sorted(
        rows,
        key=lambda row: hashlib.sha256(
            f"{SAMPLE_SEED}:{row['id']}".encode()
        ).hexdigest(),
    )
    selected = []
    for row in ordered:
        if row["frames"] < ROLLOUT_FRAMES:
            continue
        case = load_case(token_root, row)
        if (
            int(case["activity"][:ROLLOUT_FRAMES].sum()) >= MIN_EARLY_ACTIVE_FRAMES
            and int(case["user_activity"][:ROLLOUT_FRAMES].sum())
            >= MIN_EARLY_ACTIVE_FRAMES
        ):
            selected.append(row)
        if len(selected) == count:
            break
    if len(selected) < count:
        raise ValueError("Insufficient conversations with early speech on both streams")
    return selected


def matched_alternate(
    row: dict[str, Any], pool: list[dict[str, Any]]
) -> tuple[dict[str, Any], dict[str, Any]]:
    choices = [
        other
        for other in pool
        if other["id"] != row["id"]
        and other["scenario_family"] != row["scenario_family"]
        and other["frames"] >= row["frames"]
    ]
    if not choices:
        raise ValueError(f"No causal shuffled-input match for {row['id']}")
    alternate = min(
        choices,
        key=lambda other: (
            other["voice_a"] != row["voice_a"],
            abs(other["frames"] - row["frames"]) / row["frames"],
            abs(
                other["user_active_frames"] / other["frames"]
                - row["user_active_frames"] / row["frames"]
            ),
            other["id"],
        ),
    )
    duration_mismatch = (alternate["frames"] - row["frames"]) / row["frames"]
    activity_mismatch = abs(
        alternate["user_active_frames"] / alternate["frames"]
        - row["user_active_frames"] / row["frames"]
    )
    match = {
        "alternate_id": alternate["id"],
        "same_voice": alternate["voice_a"] == row["voice_a"],
        "duration_mismatch_fraction": duration_mismatch,
        "activity_mismatch_fraction": activity_mismatch,
        "controlled": (
            alternate["voice_a"] == row["voice_a"]
            and duration_mismatch <= 0.25
            and activity_mismatch <= 0.15
        ),
    }
    return alternate, match


@contextmanager
def without_fusion(model: DauysHybrid) -> Iterator[None]:
    handles = [
        block.cross_attention.register_forward_hook(
            lambda _module, _inputs, output: torch.zeros_like(output)
        )
        for block in model.dialogue
        if isinstance(block, DialogueBlock)
    ]
    if not handles:
        raise ValueError("The checkpoint has no fusion blocks to ablate")
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


def binary_metrics(predicted: torch.Tensor, target: torch.Tensor) -> dict[str, Any]:
    predicted = predicted.bool().flatten()
    target = target.bool().flatten()
    tp = int((predicted & target).sum())
    fp = int((predicted & ~target).sum())
    fn = int((~predicted & target).sum())
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": tp / max(1, tp + fp),
        "recall": tp / max(1, tp + fn),
        "f1": 2 * tp / max(1, 2 * tp + fp + fn),
        "predicted_active_frames": int(predicted.sum()),
        "target_active_frames": int(target.sum()),
    }


def onset_deviation(predicted: torch.Tensor, target: torch.Tensor) -> float | None:
    predicted = predicted.bool().flatten()
    target = target.bool().flatten()
    predicted_onsets = torch.where(predicted & ~F.pad(predicted[:-1], (1, 0)))[0]
    target_onsets = torch.where(target & ~F.pad(target[:-1], (1, 0)))[0]
    if not len(predicted_onsets) or not len(target_onsets):
        return None
    distances = torch.abs(target_onsets[:, None] - predicted_onsets[None, :])
    return float(distances.min(dim=1).values.float().mean())


def conditional(
    model: DauysHybrid,
    user: torch.Tensor,
    target: torch.Tensor,
    activity: torch.Tensor,
    silence: torch.Tensor,
    config: ModelConfig,
) -> tuple[dict[str, Any], torch.Tensor]:
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        output = model(user, target, silence)
        assert output.depth_logits is not None
        frames = target.shape[-1]
        cb0_loss = F.cross_entropy(
            output.cb0_logits.reshape(-1, config.vocab_size),
            target[:, 0].reshape(-1),
        )
        depth_loss = F.cross_entropy(
            output.depth_logits.reshape(-1, config.vocab_size),
            target[:, 1:].transpose(1, 2).reshape(-1),
            reduction="none",
        ).reshape(1, frames, 7)
        activity_loss = F.cross_entropy(
            output.activity_logits.reshape(-1, 2), activity.reshape(-1)
        )
        cb0_accuracy = (output.cb0_logits.argmax(-1) == target[:, 0]).float()
        depth_accuracy = (
            output.depth_logits.argmax(-1) == target[:, 1:].transpose(1, 2)
        ).float()
        predicted = output.activity_logits.argmax(-1).bool()
        per_codebook_nll = [float(cb0_loss)] + [
            float(depth_loss[:, :, index].mean()) for index in range(7)
        ]
        per_codebook_accuracy = [float(cb0_accuracy.mean())] + [
            float(depth_accuracy[:, :, index].mean()) for index in range(7)
        ]
        windows = []
        cb0_frames = F.cross_entropy(
            output.cb0_logits.reshape(-1, config.vocab_size),
            target[:, 0].reshape(-1),
            reduction="none",
        ).reshape(1, frames)
        for start in range(0, frames, WINDOW_FRAMES):
            end = min(frames, start + WINDOW_FRAMES)
            windows.append(
                {
                    "start_frame": start,
                    "end_frame": end,
                    "cb0_nll": float(cb0_frames[:, start:end].mean()),
                    "depth_nll": float(depth_loss[:, start:end].mean()),
                    "activity": binary_metrics(
                        predicted[:, start:end], activity[:, start:end]
                    ),
                }
            )
        result = {
            "nll": sum(per_codebook_nll) + float(activity_loss),
            "cb0_nll": per_codebook_nll[0],
            "depth_nll": float(depth_loss.mean()),
            "activity_nll": float(activity_loss),
            "per_codebook_nll": per_codebook_nll,
            "per_codebook_accuracy": per_codebook_accuracy,
            "activity": binary_metrics(predicted, activity),
            "windows": windows,
        }
        return result, predicted


def user_mode(
    mode: str,
    user: torch.Tensor,
    alternate: torch.Tensor,
    silence: torch.Tensor,
) -> torch.Tensor:
    if mode == "silent":
        return silence[:, :, None].expand_as(user)
    if mode == "shuffled":
        return alternate[:, :, : user.shape[-1]]
    return user


def generate(
    model: DauysHybrid,
    incoming: torch.Tensor,
    target: torch.Tensor,
    activity: torch.Tensor,
    user_activity: torch.Tensor,
    silence: torch.Tensor,
    codec: Any,
    cb0_reference: bool,
    limit_frames: int = ROLLOUT_FRAMES,
) -> tuple[dict[str, Any], np.ndarray, torch.Tensor]:
    frames = min(limit_frames, incoming.shape[-1])
    state = model.new_stream(silence)
    generated = []
    predicted_activity = []
    with torch.inference_mode():
        for index in range(frames):
            if cb0_reference:
                if state.previous_output is None:
                    raise ValueError("Missing causal output history")
                cb0, activity_logits, context = model.stream_step_teacher(
                    incoming[:, :, index], state.previous_output, state
                )
                del cb0
                depth_codes = [target[:, 0, index]]
                for depth_index in range(1, 8):
                    prefix = torch.stack(depth_codes, dim=-1)[:, None]
                    logits = model.depth(prefix, context)[:, 0, -1]
                    depth_codes.append(logits.argmax(dim=-1))
                frame = torch.stack(depth_codes, dim=1)
                state.previous_output = frame
            else:
                frame, activity_logits = model.generate_step(
                    incoming[:, :, index], state
                )
            generated.append(frame)
            predicted_activity.append(activity_logits.argmax(-1))
        if state.frame_index != frames or not state.listening_history:
            raise RuntimeError("Incoming stream did not advance during generation")
        codes = torch.stack(generated, dim=-1)
        activity_prediction = torch.stack(predicted_activity, dim=-1).bool()
        wave = decode(codec, codes)[0, 0].float().cpu().numpy()
    actual = activity[:, :frames].bool()
    user_active = user_activity[:, :frames].bool()
    overlap = binary_metrics(activity_prediction & user_active, actual & user_active)
    timing = {
        "activity": binary_metrics(activity_prediction, actual),
        "overlap": overlap,
        "overlap_annotation_frames": int((actual & user_active).sum()),
        "onset_deviation_frames": onset_deviation(activity_prediction, actual),
        "silence_ratio": 1 - float(activity_prediction.float().mean()),
        "rms": float(np.sqrt(np.mean(wave.astype(np.float64) ** 2))),
        "duration_seconds": len(wave) / 24_000,
        "distinct_cb0_fraction": float(
            torch.unique(codes[:, 0]).numel() / max(1, frames)
        ),
        "repeated_frame_fraction": float(
            (codes[:, :, 1:] == codes[:, :, :-1]).all(dim=1).float().mean()
        )
        if frames > 1
        else 0.0,
        "input_frames_processed": state.frame_index,
    }
    windows = []
    for start in range(0, frames, WINDOW_FRAMES):
        end = min(frames, start + WINDOW_FRAMES)
        window = binary_metrics(activity_prediction[:, start:end], actual[:, start:end])
        window["start_frame"] = start
        window["end_frame"] = end
        windows.append(window)
    timing["windows"] = windows
    return timing, wave, codes


def save_wave(path: Path, wave: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, wave, 24_000)


def mean_weighted(
    cases: list[dict[str, Any]], path: tuple[str | int, ...], controlled: bool = False
) -> float | None:
    selected = [row for row in cases if not controlled or row["match"]["controlled"]]
    if not selected:
        return None
    values = []
    for row in selected:
        value: Any = row
        for key in path:
            value = value[key]
        values.append(float(value) * row["frames"])
    return sum(values) / sum(row["frames"] for row in selected)


def aggregate(cases: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {"samples": len(cases)}
    for split in ("train", "validation"):
        selected = [row for row in cases if row["selected_split"] == split]
        controlled = [row for row in selected if row["match"]["controlled"]]
        overlap_cases = [
            row
            for row in selected
            if row["generated"]["original"]["overlap_annotation_frames"] > 0
        ]
        result[split] = {
            "samples": len(selected),
            "frames": sum(row["frames"] for row in selected),
            "controlled_matches": sum(row["match"]["controlled"] for row in selected),
            "valid_overlap_cases": len(overlap_cases),
            "original_nll": mean_weighted(selected, ("conditional", "original", "nll")),
            "controlled_original_nll": mean_weighted(
                controlled, ("conditional", "original", "nll")
            ),
            "silent_nll": mean_weighted(selected, ("conditional", "silent", "nll")),
            "shuffled_nll": mean_weighted(
                selected, ("conditional", "shuffled", "nll"), controlled=True
            ),
            "fusion_off_nll": mean_weighted(
                selected, ("conditional", "fusion_off", "nll")
            ),
            "cb0_nll": mean_weighted(selected, ("conditional", "original", "cb0_nll")),
            "depth_nll": mean_weighted(
                selected, ("conditional", "original", "depth_nll")
            ),
            "teacher_activity_f1": mean_weighted(
                selected, ("conditional", "original", "activity", "f1")
            ),
            "free_activity_f1": mean_weighted(
                selected, ("generated", "original", "activity", "f1")
            ),
            "free_overlap_f1": mean_weighted(
                overlap_cases, ("generated", "original", "overlap", "f1")
            ),
            "free_onset_deviation_frames": mean_weighted(
                [
                    row
                    for row in selected
                    if row["generated"]["original"]["onset_deviation_frames"]
                    is not None
                ],
                ("generated", "original", "onset_deviation_frames"),
            ),
            "free_silence_ratio": mean_weighted(
                selected, ("generated", "original", "silence_ratio")
            ),
            "free_repeated_frame_fraction": mean_weighted(
                selected, ("generated", "original", "repeated_frame_fraction")
            ),
            "fusion_output_token_difference_fraction": mean_weighted(
                selected,
                ("generated", "fusion_off", "token_difference_fraction"),
            ),
            "per_codebook_nll": [
                mean_weighted(
                    selected,
                    ("conditional", "original", "per_codebook_nll", index),
                )
                for index in range(8)
            ],
            "per_codebook_accuracy": [
                mean_weighted(
                    selected,
                    ("conditional", "original", "per_codebook_accuracy", index),
                )
                for index in range(8)
            ],
        }
        for mode in ("silent", "shuffled", "fusion_off"):
            baseline = (
                result[split]["controlled_original_nll"]
                if mode == "shuffled"
                else result[split]["original_nll"]
            )
            result[split][f"{mode}_delta_nll"] = (
                result[split][f"{mode}_nll"] - baseline
                if result[split][f"{mode}_nll"] is not None and baseline is not None
                else None
            )
    return result


def evaluate_checkpoint(
    label: str,
    checkpoint_path: Path,
    selected: list[dict[str, Any]],
    pools: dict[str, list[dict[str, Any]]],
    token_root: Path,
    codec: Any,
    silence: torch.Tensor,
    output: Path,
    entity: str | None,
) -> dict[str, Any]:
    checkpoint = load_research_checkpoint(checkpoint_path)
    config = ModelConfig.stage1()
    metadata = checkpoint["metadata"]
    if metadata["dataset_revision"] != REVISION or metadata["model_config"] != asdict(
        config
    ):
        raise ValueError(f"Checkpoint identity mismatch: {label}")
    model = DauysHybrid(config).cuda().eval()
    model.load_state_dict(checkpoint["model"], strict=True)
    model.requires_grad_(False)
    set_context(model, int(checkpoint["context_frames"]))
    cases: list[dict[str, Any]] = []
    table = wandb.Table(
        columns=[
            "sample_id",
            "split",
            "checkpoint",
            "mode",
            "user_audio",
            "reference_audio",
            "generated_audio",
            "nll",
            "activity_f1",
            "intelligibility",
            "relevance",
        ]
    )
    for index, row in enumerate(selected, 1):
        split = row["selected_split"]
        alternate_row, match = matched_alternate(row, pools[split])
        case = load_case(token_root, row)
        other = load_case(token_root, alternate_row)
        user, target, activity = on_gpu(case)
        alternate = other["user"][None].long().cuda()
        user_activity = case["user_activity"][None].long().cuda()
        frames = min(ROLLOUT_FRAMES, row["frames"])
        reference_paths = {
            mode: output / "audio" / "reference" / split / f"{row['id']}_{mode}.wav"
            for mode in ("user", "target")
        }
        for mode, codes in (
            ("user", user[:, :, :frames]),
            ("target", target[:, :, :frames]),
        ):
            if not reference_paths[mode].exists():
                save_wave(
                    reference_paths[mode],
                    decode(codec, codes)[0, 0].float().cpu().numpy(),
                )
        record: dict[str, Any] = {
            "id": row["id"],
            "selected_split": split,
            "frames": row["frames"],
            "scenario_family": row["scenario_family"],
            "match": match,
            "conditional": {},
            "generated": {},
            "codebook_injection": {
                "acoustic": "Current-frame ground-truth CB0 only; CB1-CB7 and all previous response frames are generated",
                "original": "No reference response tokens injected",
            },
        }
        original_codes = None
        for mode in MODES:
            incoming = user_mode(mode, user, alternate, silence)
            if mode == "acoustic":
                continue
            if mode == "fusion_off":
                with without_fusion(model):
                    scores, _ = conditional(
                        model, incoming, target, activity, silence, config
                    )
                    generated, wave, codes = generate(
                        model,
                        incoming,
                        target,
                        activity,
                        user_activity,
                        silence,
                        codec,
                        False,
                    )
            else:
                scores, _ = conditional(
                    model, incoming, target, activity, silence, config
                )
                generated, wave, codes = generate(
                    model,
                    incoming,
                    target,
                    activity,
                    user_activity,
                    silence,
                    codec,
                    False,
                )
            record["conditional"][mode] = scores
            record["generated"][mode] = generated
            audio_path = output / "audio" / mode / label / split / f"{row['id']}.wav"
            save_wave(audio_path, wave)
            if mode == "original":
                original_codes = codes
            elif original_codes is not None:
                generated["token_difference_fraction"] = float(
                    (codes != original_codes).float().mean()
                )
            table.add_data(
                row["id"],
                split,
                label,
                mode,
                wandb.Audio(str(reference_paths["user"]), sample_rate=24_000),
                wandb.Audio(str(reference_paths["target"]), sample_rate=24_000),
                wandb.Audio(str(audio_path), sample_rate=24_000),
                scores["nll"],
                generated["activity"]["f1"],
                None,
                None,
            )
        acoustic, wave, acoustic_codes = generate(
            model,
            user,
            target,
            activity,
            user_activity,
            silence,
            codec,
            True,
        )
        acoustic_path = (
            output
            / "audio"
            / "codebook_diagnostics"
            / label
            / split
            / f"{row['id']}_acoustic.wav"
        )
        save_wave(acoustic_path, wave)
        acoustic["cb0_reference_fraction"] = float(
            (acoustic_codes[:, 0] == target[:, 0, : acoustic_codes.shape[-1]])
            .float()
            .mean()
        )
        record["generated"]["acoustic"] = acoustic
        record["audio"] = {
            "user": str(reference_paths["user"].relative_to(output)),
            "reference": str(reference_paths["target"].relative_to(output)),
            **{
                mode: str(
                    (
                        output / "audio" / mode / label / split / f"{row['id']}.wav"
                    ).relative_to(output)
                )
                for mode in MODES
                if mode != "acoustic"
            },
            "acoustic": str(acoustic_path.relative_to(output)),
        }
        cases.append(record)
        table.add_data(
            row["id"],
            split,
            label,
            "acoustic",
            wandb.Audio(str(reference_paths["user"]), sample_rate=24_000),
            wandb.Audio(str(reference_paths["target"]), sample_rate=24_000),
            wandb.Audio(str(acoustic_path), sample_rate=24_000),
            record["conditional"]["original"]["nll"],
            acoustic["activity"]["f1"],
            None,
            None,
        )
        print(
            json.dumps(
                {
                    "checkpoint": label,
                    "sample": index,
                    "total": len(selected),
                    "id": row["id"],
                }
            ),
            flush=True,
        )
    summary = aggregate(cases)
    result = {
        "checkpoint_label": label,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": digest(checkpoint_path),
        "checkpoint_step": checkpoint["step"],
        "training_git_sha": metadata["git_sha"],
        "evaluation_git_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "summary": summary,
    }
    case_path = output / f"results_per_sample_{label}.jsonl"
    case_path.write_text(
        "".join(json.dumps(row) + "\n" for row in cases), encoding="utf-8"
    )
    (output / f"results_{label}.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    if entity is not None:
        run = wandb.init(
            project="dauys",
            entity=entity,
            group="S1-E005-DIAGNOSTICS",
            name=f"S1-E005-DIAGNOSTICS-{label}",
            dir=str(output / "wandb-local"),
            config={
                "checkpoint_sha256": result["checkpoint_sha256"],
                "checkpoint_step": result["checkpoint_step"],
                "git_sha": result["evaluation_git_sha"],
                "dataset_revision": REVISION,
                "seed": SAMPLE_SEED,
                "rollout_frames": ROLLOUT_FRAMES,
                "sample_manifest_sha256": digest(output / "sample_manifest.json"),
            },
        )
        if run is None:
            raise RuntimeError("W&B diagnostic run did not start")
        for split in ("train", "validation"):
            scores = summary[split]
            for name, namespace in (
                ("original_nll", "conditioning"),
                ("silent_nll", "conditioning"),
                ("shuffled_nll", "conditioning"),
                ("silent_delta_nll", "conditioning"),
                ("shuffled_delta_nll", "conditioning"),
                ("fusion_off_nll", "fusion"),
                ("fusion_off_delta_nll", "fusion"),
                ("fusion_output_token_difference_fraction", "fusion"),
                ("cb0_nll", "codebooks"),
                ("depth_nll", "codebooks"),
                ("teacher_activity_f1", "history"),
                ("free_activity_f1", "history"),
                ("free_overlap_f1", "activity"),
                ("free_silence_ratio", "activity"),
                ("free_onset_deviation_frames", "activity"),
            ):
                value = scores[name]
                if value is not None:
                    run.log({f"eval/{namespace}/{split}_{name}": value})
            for index in range(8):
                run.log(
                    {
                        f"eval/codebooks/{split}_cb{index}_nll": scores[
                            "per_codebook_nll"
                        ][index],
                        f"eval/codebooks/{split}_cb{index}_accuracy": scores[
                            "per_codebook_accuracy"
                        ][index],
                    }
                )
        for metric in ("original_nll", "cb0_nll", "depth_nll", "free_activity_f1"):
            train_value = summary["train"][metric]
            validation_value = summary["validation"][metric]
            if train_value is not None and validation_value is not None:
                run.log(
                    {
                        f"eval/generalization/validation_minus_train_{metric}": (
                            validation_value - train_value
                        )
                    }
                )
        run.log({"eval/audio": table})
        artifact = wandb.Artifact(
            f"S1-E005-DIAGNOSTICS-{label}-results", type="evaluation"
        )
        for path in (
            output / "metadata.json",
            output / "sample_manifest.json",
            output / f"results_{label}.json",
            case_path,
        ):
            artifact.add_file(str(path))
        run.log_artifact(artifact)
        result["wandb_url"] = run.url
        run.finish()
        (output / f"results_{label}.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8"
        )
    del model
    torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--tokens", type=Path, required=True)
    parser.add_argument("--best-checkpoint", type=Path, required=True)
    parser.add_argument("--final-checkpoint", type=Path, required=True)
    parser.add_argument("--mimi-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--train-count", type=int, default=10)
    parser.add_argument("--validation-count", type=int, default=5)
    args = parser.parse_args()
    if not (1 <= args.train_count <= 10 and 1 <= args.validation_count <= 20):
        raise ValueError("Diagnostic sample counts exceed the frozen protocol")
    if args.output.exists() and not args.resume:
        raise FileExistsError(args.output)
    completion = json.loads(
        (args.best_checkpoint.parent / "completed.json").read_text()
    )
    if (
        digest(args.best_checkpoint) != completion["best_nll_sha256"]
        or digest(args.final_checkpoint) != completion["latest_sha256"]
    ):
        raise ValueError("Selected checkpoint hashes differ from completion record")
    train = read_manifest(args.splits / "train.jsonl", "train")
    validation = read_manifest(args.splits / "validation.jsonl", "validation")
    if {row["scenario_family"] for row in train} & {
        row["scenario_family"] for row in validation
    }:
        raise ValueError("Train/validation family leakage")
    selected = fixed_rows(
        train, 1 if args.smoke else args.train_count, args.tokens
    ) + fixed_rows(validation, 1 if args.smoke else args.validation_count, args.tokens)
    pools = {"train": train, "validation": validation}
    selected_ids = {row["id"] for row in selected}
    match_rows = [
        matched_alternate(row, pools[row["selected_split"]])[0] for row in selected
    ]
    needed = selected + [row for row in match_rows if row["id"] not in selected_ids]
    for row in needed:
        load_case(args.tokens, row)
    args.output.mkdir(parents=True)
    sample_manifest = [
        {
            "id": row["id"],
            "selected_split": row["selected_split"],
            "source_split": row["split"],
            "frames": row["frames"],
            "scenario_family": row["scenario_family"],
            "token_sha256": row["sha256"],
            "match": matched_alternate(row, pools[row["selected_split"]])[1],
        }
        for row in selected
    ]
    (args.output / "sample_manifest.json").write_text(
        json.dumps(sample_manifest, indent=2) + "\n", encoding="utf-8"
    )
    torch.manual_seed(SAMPLE_SEED)
    torch.cuda.manual_seed_all(SAMPLE_SEED)
    random.seed(SAMPLE_SEED)
    codec = load_mimi(args.mimi_checkpoint, "cuda")
    with torch.inference_mode():
        silence = encode(codec, torch.zeros((1, 1, 1920), device="cuda"))[:, :, 0]
    metadata = {
        "experiment": "S1-E005-DIAGNOSTICS",
        "architecture": "Hybrid v0.1",
        "dataset": DATASET,
        "dataset_revision": REVISION,
        "git_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "sample_seed": SAMPLE_SEED,
        "rollout_frames": ROLLOUT_FRAMES,
        "window_frames": WINDOW_FRAMES,
        "minimum_active_frames_per_stream_in_rollout": MIN_EARLY_ACTIVE_FRAMES,
        "selected_train_count": 1 if args.smoke else args.train_count,
        "selected_validation_count": 1 if args.smoke else args.validation_count,
        "decoding": "deterministic greedy",
        "mimi_checkpoint_sha256": digest(args.mimi_checkpoint),
        "mimi_package_version": importlib.metadata.version("moshi"),
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
        "cuda_version": torch.version.cuda,
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_device_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
        "cuda_capability": torch.cuda.get_device_capability(0),
        "best_checkpoint_sha256": digest(args.best_checkpoint),
        "final_checkpoint_sha256": digest(args.final_checkpoint),
        "sample_manifest_sha256": digest(args.output / "sample_manifest.json"),
    }
    (args.output / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    entity = None if args.smoke else private_wandb_entity()
    results = []
    for label, path in (
        ("A", args.best_checkpoint),
        ("B", args.final_checkpoint),
    ):
        result_path = args.output / f"results_{label}.json"
        if args.resume and result_path.is_file():
            prior = json.loads(result_path.read_text())
            if prior["checkpoint_sha256"] == digest(path) and (
                args.smoke or prior.get("wandb_url")
            ):
                results.append(prior)
                continue
        results.append(
            evaluate_checkpoint(
                label,
                path,
                selected,
                pools,
                args.tokens,
                codec,
                silence,
                args.output,
                entity,
            )
        )
    (args.output / "results.json").write_text(
        json.dumps({"checkpoints": results}, indent=2) + "\n", encoding="utf-8"
    )
    with (args.output / "listening_review.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "sample_id",
                "split",
                "checkpoint_label",
                "user_audio",
                "reference_audio",
                "generated_audio",
                "reference_transcript",
                "intelligibility_0_to_3",
                "relevance_0_to_3",
                "notes",
            ]
        )
        for label in ("A", "B"):
            for row in selected:
                split = row["selected_split"]
                writer.writerow(
                    [
                        row["id"],
                        split,
                        label,
                        f"audio/reference/{split}/{row['id']}_user.wav",
                        f"audio/reference/{split}/{row['id']}_target.wav",
                        f"audio/original/{label}/{split}/{row['id']}.wav",
                        "",
                        "",
                        "",
                        "",
                    ]
                )
    (args.output / "report.md").write_text(
        "# S1-E005 diagnostics\n\n"
        "Automated metrics are in results.json. Human intelligibility and relevance "
        "scores are pending in listening_review.csv. Checkpoint labels are anonymized "
        "in that review file; metadata.json retains their mapping.\n",
        encoding="utf-8",
    )
    print(json.dumps({"completed": True, "results": str(args.output / "results.json")}))


if __name__ == "__main__":
    main()
