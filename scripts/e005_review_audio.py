"""Render complete validation conversations for E005 human listening review."""

import argparse
import csv
import json
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
import wandb
from e005_data import REVISION, digest
from e005_diagnostics import generate, save_wave
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
from dauys.model import DauysHybrid


def render_checkpoint(
    label: str,
    checkpoint_path: Path,
    selected: list[dict[str, Any]],
    tokens: Path,
    codec: Any,
    silence: torch.Tensor,
    output: Path,
    wandb_dir: Path,
    entity: str,
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
    table = wandb.Table(
        columns=[
            "sample_id",
            "split",
            "checkpoint",
            "mode",
            "user_audio",
            "reference_audio",
            "generated_audio",
            "combined_audio",
            "nll",
            "activity_f1",
            "intelligibility",
            "relevance",
        ]
    )
    cases = []
    for index, row in enumerate(selected, 1):
        cached = load_case(tokens, row)
        user, target, activity = on_gpu(cached)
        user_activity = cached["user_activity"][None].long().cuda()
        frames = row["frames"]
        case_dir = output / "audio" / label
        generated_path = case_dir / f"{row['id']}_generated.wav"
        combined_path = output / "audio" / "combined" / label / f"{row['id']}.wav"
        user_path = output / "audio" / "reference" / f"{row['id']}_user.wav"
        target_path = output / "audio" / "reference" / f"{row['id']}_target.wav"
        if not user_path.exists():
            with torch.inference_mode():
                save_wave(user_path, decode(codec, user)[0, 0].float().cpu().numpy())
                save_wave(
                    target_path, decode(codec, target)[0, 0].float().cpu().numpy()
                )
        metrics, wave, _ = generate(
            model,
            user,
            target,
            activity,
            user_activity,
            silence,
            codec,
            False,
            limit_frames=frames,
        )
        save_wave(generated_path, wave)
        user_wave, sample_rate = sf.read(user_path, dtype="float32")
        if sample_rate != 24_000 or len(user_wave) != len(wave):
            raise ValueError(f"Two streams are not aligned: {row['id']}")
        save_wave(combined_path, np.column_stack((user_wave, wave)))
        cases.append(
            {
                "id": row["id"],
                "frames": frames,
                "duration_seconds": len(wave) / 24_000,
                "activity": metrics["activity"],
                "rms": metrics["rms"],
                "silence_ratio": metrics["silence_ratio"],
                "input_frames_processed": metrics["input_frames_processed"],
                "user_audio": str(user_path.relative_to(output)),
                "reference_audio": str(target_path.relative_to(output)),
                "generated_audio": str(generated_path.relative_to(output)),
                "combined_audio": str(combined_path.relative_to(output)),
            }
        )
        table.add_data(
            row["id"],
            "validation",
            label,
            "full_conversation",
            wandb.Audio(str(user_path), sample_rate=24_000),
            wandb.Audio(str(target_path), sample_rate=24_000),
            wandb.Audio(str(generated_path), sample_rate=24_000),
            wandb.Audio(str(combined_path), sample_rate=24_000),
            None,
            metrics["activity"]["f1"],
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
                    "frames": frames,
                }
            ),
            flush=True,
        )
    result = {
        "checkpoint_label": label,
        "checkpoint_sha256": digest(checkpoint_path),
        "checkpoint_step": checkpoint["step"],
        "training_git_sha": metadata["git_sha"],
        "evaluation_git_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "cases": cases,
    }
    result_path = output / f"results_{label}.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    run_dir = wandb_dir / label
    run_dir.mkdir(parents=True, exist_ok=True)
    run = wandb.init(
        project="dauys",
        entity=entity,
        group="S1-E005-DIAGNOSTICS",
        name=f"S1-E005-DIAGNOSTICS-{label}-full-review",
        dir=str(run_dir),
        config={
            "checkpoint_sha256": result["checkpoint_sha256"],
            "dataset_revision": REVISION,
            "sample_manifest_sha256": digest(output / "sample_manifest.json"),
            "decoding": "deterministic greedy",
            "history": "fully model-generated",
        },
    )
    if run is None:
        raise RuntimeError("W&B listening-review run did not start")
    run.log(
        {
            "eval/human/full_conversation_audio": table,
            "eval/activity/full_conversation_f1": sum(
                case["activity"]["f1"] for case in cases
            )
            / len(cases),
        }
    )
    artifact = wandb.Artifact(
        f"S1-E005-DIAGNOSTICS-{label}-full-review", type="evaluation"
    )
    artifact.add_file(str(result_path))
    artifact.add_file(str(output / "sample_manifest.json"))
    run.log_artifact(artifact)
    result["wandb_url"] = run.url
    run.finish()
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    del model
    torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-manifest", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--tokens", type=Path, required=True)
    parser.add_argument("--best-checkpoint", type=Path, required=True)
    parser.add_argument("--final-checkpoint", type=Path, required=True)
    parser.add_argument("--mimi-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wandb-dir", type=Path, required=True)
    parser.add_argument("--sample-id")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    completion = json.loads(
        (args.best_checkpoint.parent / "completed.json").read_text()
    )
    if (
        digest(args.best_checkpoint) != completion["best_nll_sha256"]
        or digest(args.final_checkpoint) != completion["latest_sha256"]
    ):
        raise ValueError("Checkpoint differs from completion record")
    manifest = json.loads(args.sample_manifest.read_text())
    selected_ids = [
        row["id"] for row in manifest if row["selected_split"] == "validation"
    ]
    if len(selected_ids) != 5 or len(set(selected_ids)) != 5:
        raise ValueError("Five distinct validation samples are required")
    if args.sample_id is not None:
        if args.sample_id not in selected_ids:
            raise ValueError("Requested sample is outside the frozen review selection")
        selected_ids = [args.sample_id]
    validation = {
        row["id"]: row
        for row in read_manifest(args.splits / "validation.jsonl", "validation")
    }
    selected = [validation[identifier] for identifier in selected_ids]
    for manifest_row, row in zip(
        (
            item
            for item in manifest
            if item["selected_split"] == "validation" and item["id"] in selected_ids
        ),
        selected,
    ):
        if row["sha256"] != manifest_row["token_sha256"]:
            raise ValueError(f"Sample cache changed: {row['id']}")
        load_case(args.tokens, row)
    args.output.mkdir(parents=True)
    (args.output / "sample_manifest.json").write_text(
        json.dumps(
            [
                {
                    "id": row["id"],
                    "frames": row["frames"],
                    "token_sha256": row["sha256"],
                }
                for row in selected
            ],
            indent=2,
        )
        + "\n"
    )
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    codec = load_mimi(args.mimi_checkpoint, "cuda")
    with torch.inference_mode():
        silence = encode(codec, torch.zeros((1, 1, 1920), device="cuda"))[:, :, 0]
    entity = private_wandb_entity()
    results = [
        render_checkpoint(
            label,
            path,
            selected,
            args.tokens,
            codec,
            silence,
            args.output,
            args.wandb_dir,
            entity,
        )
        for label, path in (
            ("A", args.best_checkpoint),
            ("B", args.final_checkpoint),
        )
    ]
    (args.output / "results.json").write_text(
        json.dumps({"checkpoints": results}, indent=2) + "\n"
    )
    with (args.output / "listening_review.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "sample_id",
                "checkpoint_label",
                "user_audio",
                "reference_audio",
                "generated_audio",
                "combined_audio",
                "intelligibility_0_to_3",
                "relevance_0_to_3",
                "notes",
            ]
        )
        for result in results:
            for case in result["cases"]:
                writer.writerow(
                    [
                        case["id"],
                        result["checkpoint_label"],
                        case["user_audio"],
                        case["reference_audio"],
                        case["generated_audio"],
                        case["combined_audio"],
                        "",
                        "",
                        "",
                    ]
                )
    print(json.dumps({"completed": True, "output": str(args.output)}))


if __name__ == "__main__":
    main()
