"""Run the frozen E005 test once after the final checkpoint is selected."""

import argparse
import json
import math
import random
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
import wandb
from datasets import Audio, load_dataset
from e005_data import (
    DATASET,
    FRAME_SAMPLES,
    REVISION,
    activity_from_events,
    digest,
)
from e005_train import (
    conditional_evaluation,
    fixed_examples,
    free_evaluation,
    load_case,
    on_gpu,
    private_wandb_entity,
    read_manifest,
    set_context,
    validation,
)
from overfit import _pcm

from dauys.checkpoint import load_research_checkpoint
from dauys.codec import decode, encode, load_mimi
from dauys.config import ModelConfig
from dauys.model import DauysHybrid


def cache_test(
    frozen: list[dict[str, Any]],
    root: Path,
    codec: Any,
) -> list[dict[str, Any]]:
    expected = {row["id"]: row for row in frozen}
    found: dict[str, dict[str, Any]] = {}
    dataset = load_dataset(DATASET, split="test", revision=REVISION, streaming=True)
    for column in ("channel_a", "channel_b"):
        if isinstance(dataset.features[column], Audio):
            dataset = dataset.cast_column(column, Audio(decode=False))
    for row in dataset:
        identifier = row["id"]
        if identifier not in expected or identifier in found:
            raise ValueError(f"Unexpected or duplicate test conversation: {identifier}")
        frozen_row = expected[identifier]
        samples = int(row["duration_samples"])
        frames = math.ceil(samples / FRAME_SAMPLES)
        if (
            samples != frozen_row["samples"]
            or frames != frozen_row["frames"]
            or row["scenario"] != frozen_row["scenario"]
            or row["voice_a"] != frozen_row["voice_a"]
        ):
            raise ValueError(f"Test metadata changed: {identifier}")
        user = _pcm(row["channel_a"])
        target = _pcm(row["channel_b"])
        if len(user) != samples or len(target) != samples:
            raise ValueError(f"Misaligned test audio: {identifier}")
        codes = []
        for pcm in (user, target):
            padded = np.pad(pcm, (0, frames * FRAME_SAMPLES - samples))
            with torch.inference_mode():
                codes.append(
                    encode(codec, torch.from_numpy(padded.copy())[None, None].cuda())
                )
        activity = activity_from_events(row["events"], frames)
        user_activity = activity_from_events(row["events"], frames, "A")
        cached = {
            "user": codes[0][0].short().cpu(),
            "target": codes[1][0].short().cpu(),
            "activity": torch.from_numpy(activity.astype(np.uint8)),
            "user_activity": torch.from_numpy(user_activity.astype(np.uint8)),
        }
        path = root / "test" / f"{identifier}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(cached, path)
        found[identifier] = {
            **frozen_row,
            "active_frames": int(activity.sum()),
            "user_active_frames": int(user_activity.sum()),
            "sha256": digest(path),
        }
        print(json.dumps({"cached_test_id": identifier}), flush=True)
    if set(found) != set(expected):
        raise ValueError("Test cache is incomplete")
    selected = [found[row["id"]] for row in frozen]
    (root / "test.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in selected), encoding="utf-8"
    )
    return selected


def rollout_control(
    model: DauysHybrid,
    incoming: torch.Tensor,
    target: torch.Tensor,
    silence: torch.Tensor,
    start: int,
    end: int,
) -> torch.Tensor:
    with torch.inference_mode():
        state = model.new_stream(silence)
        previous = silence
        for index in range(start):
            model.stream_step_teacher(incoming[:, :, index], previous, state)
            previous = target[:, :, index]
        state.previous_output = previous
        result = []
        for index in range(start, end):
            codes, _ = model.generate_step(incoming[:, :, index], state)
            result.append(codes)
    return torch.stack(result, -1)


def blind_controls(
    model: DauysHybrid,
    codec: Any,
    selected: list[dict[str, Any]],
    all_test: list[dict[str, Any]],
    root: Path,
    silence: torch.Tensor,
    output: Path,
    seed: int = 42,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    results = []
    output.mkdir(parents=True)
    for index, row in enumerate(selected[:4]):
        choices = [
            other
            for other in all_test
            if other["scenario_family"] != row["scenario_family"]
            and other["frames"] >= row["frames"]
        ]
        if not choices:
            raise ValueError("No long enough shuffled test input")
        alternate = min(
            choices,
            key=lambda other: (
                other["voice_a"] != row["voice_a"],
                abs(other["frames"] - row["frames"]),
                other["id"],
            ),
        )
        cached = load_case(root, row)
        user, target, _ = on_gpu(cached)
        other, _, _ = on_gpu(load_case(root, alternate))
        start = 0
        end = min(row["frames"], 256)
        modes = {
            "original": user,
            "silent": silence[:, :, None].expand_as(user),
            "shuffled": other[:, :, : row["frames"]],
        }
        labels = ["A", "B", "C"]
        rng.shuffle(labels)
        key = dict(zip(labels, modes))
        item: dict[str, Any] = {
            "conversation_id": row["id"],
            "alternate_id": alternate["id"],
            "start_frame": start,
            "end_frame": end,
            "hidden_mapping": {label: mode for label, mode in zip(labels, modes)},
        }
        for label, mode in key.items():
            codes = rollout_control(model, modes[mode], target, silence, start, end)
            with torch.inference_mode():
                wave = decode(codec, codes)[0, 0].float().cpu().numpy()
            sf.write(output / f"case_{index:02d}_{label}.wav", wave, 24_000)
        results.append(item)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--completed", type=Path, required=True)
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument("--mimi-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.checkpoint.name != "best_nll.pt":
        raise ValueError(
            "A fresh output and the selected best NLL checkpoint are required"
        )
    completion = json.loads(args.completed.read_text())
    if digest(args.checkpoint) != completion["best_nll_sha256"]:
        raise ValueError("Selected checkpoint differs from completed training")
    checkpoint = load_research_checkpoint(args.checkpoint)
    metadata = checkpoint["metadata"]
    frozen = read_manifest(args.test_manifest, "test")
    config = ModelConfig.stage1()
    if (
        len(frozen) != 397
        or metadata["dataset_revision"] != REVISION
        or metadata["test_manifest_sha256"] != digest(args.test_manifest)
        or metadata["model_config"] != asdict(config)
        or metadata["mimi_sha256"] != digest(args.mimi_checkpoint)
    ):
        raise ValueError("Final test identity mismatch")
    entity = private_wandb_entity()
    args.output.mkdir(parents=True)
    codec = load_mimi(args.mimi_checkpoint, "cuda")
    with torch.inference_mode():
        silence = encode(codec, torch.zeros((1, 1, 1920), device="cuda"))[:, :, 0]
    model = DauysHybrid(config).cuda().eval()
    model.load_state_dict(checkpoint["model"], strict=True)
    set_context(model, int(checkpoint["context_frames"]))
    token_root = args.output / "tokens"
    test = cache_test(frozen, token_root, codec)
    score = validation(model, test, token_root, silence, config)
    conditional = conditional_evaluation(model, test, test, token_root, silence, config)
    fixed = fixed_examples(test)
    free, table = free_evaluation(
        model,
        codec,
        fixed,
        token_root,
        silence,
        args.output / "audio",
        int(checkpoint["step"]),
    )
    blind = blind_controls(
        model, codec, fixed, test, token_root, silence, args.output / "blind"
    )
    (args.output / "unblind_key.json").write_text(json.dumps(blind, indent=2) + "\n")
    result = {
        "checkpoint_sha256": digest(args.checkpoint),
        "checkpoint_step": checkpoint["step"],
        "training_git_sha": metadata["git_sha"],
        "evaluation_git_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "dataset_revision": REVISION,
        "test_manifest_sha256": digest(args.test_manifest),
        "teacher_metrics": score,
        "input_dependency": conditional,
        "free_running": free,
        "fixed_ids": [row["id"] for row in fixed],
        "blind_control_cases": len(blind),
        "human_intelligibility_and_relevance": "pending",
    }
    (args.output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    run = wandb.init(
        project="dauys",
        entity=entity,
        name="S1-E005-final-test",
        config={
            key: value
            for key, value in result.items()
            if key not in ("input_dependency", "teacher_metrics")
        },
        dir=str(args.output),
    )
    if run is None:
        raise RuntimeError("W&B test run did not start")
    run.log(
        {
            "test/nll": score["nll"],
            "test/activity_f1": score["activity_f1"],
            "test/silent_delta_nll": conditional["silent_delta_nll"],
            "test/shuffled_delta_nll": conditional["shuffled_delta_nll"],
            "test/free_running_activity_f1": free["free_running_activity_f1"],
            "test/audio_samples": table,
        }
    )
    artifact = wandb.Artifact("S1-E005-final-test-results", type="evaluation")
    artifact.add_file(str(args.output / "results.json"))
    run.log_artifact(artifact)
    run.finish()
    print(
        json.dumps(
            {
                "checkpoint_step": checkpoint["step"],
                "test_nll": score["nll"],
                "silent_delta_nll": conditional["silent_delta_nll"],
                "shuffled_delta_nll": conditional["shuffled_delta_nll"],
                "free_running_activity_f1": free["free_running_activity_f1"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
