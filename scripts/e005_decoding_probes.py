"""Compare frozen codec reconstruction and causal decoding on one E005 case."""

import argparse
import json
import math
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
import wandb
from datasets import Audio, load_dataset
from e005_data import DATASET, FRAME_SAMPLES, REVISION, digest
from e005_diagnostics import binary_metrics, save_wave
from e005_train import (
    load_case,
    on_gpu,
    private_wandb_entity,
    read_manifest,
    set_context,
)
from overfit import _pcm

from dauys.checkpoint import load_research_checkpoint
from dauys.codec import decode, encode, load_mimi
from dauys.config import ModelConfig
from dauys.model import DauysHybrid

TEMPERATURE = 0.8
TOP_K = 250
SEED = 42


def sample_topk(logits: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    values, indices = torch.topk(
        logits.float() / TEMPERATURE, min(TOP_K, logits.shape[-1]), dim=-1
    )
    probabilities = values.softmax(dim=-1)
    choice = torch.multinomial(probabilities, 1, generator=generator)
    return indices.gather(-1, choice).squeeze(-1)


def generate_sampled(
    model: DauysHybrid,
    user: torch.Tensor,
    silence: torch.Tensor,
    codec: Any,
    sample_cb0: bool,
) -> tuple[np.ndarray, torch.Tensor, torch.Tensor, dict[str, Any]]:
    label = "all_sampled" if sample_cb0 else "depth_sampled"
    cb0_rng = torch.Generator(device=user.device).manual_seed(SEED)
    depth_rng = torch.Generator(device=user.device).manual_seed(SEED + 1)
    state = model.new_stream(silence)
    codes = []
    activity = []
    began = time.monotonic()
    with torch.inference_mode():
        for index in range(user.shape[-1]):
            if state.previous_output is None:
                raise RuntimeError("Missing causal response history")
            cb0_logits, activity_logits, context = model.stream_step_teacher(
                user[:, :, index], state.previous_output, state
            )
            first = (
                sample_topk(cb0_logits, cb0_rng)
                if sample_cb0
                else cb0_logits.argmax(dim=-1)
            )
            frame_codes = [first]
            for codebook in range(1, 8):
                prefix = torch.stack(frame_codes, dim=-1)[:, None]
                logits = model.depth(prefix, context)[:, 0, -1]
                frame_codes.append(sample_topk(logits, depth_rng))
            frame = torch.stack(frame_codes, dim=1)
            state.previous_output = frame
            codes.append(frame)
            activity.append(activity_logits.argmax(dim=-1))
        if state.frame_index != user.shape[-1] or not state.listening_history:
            raise RuntimeError("The incoming stream did not advance")
        output_codes = torch.stack(codes, dim=-1)
        wave = decode(codec, output_codes)[0, 0].float().cpu().numpy()
    predicted_activity = torch.stack(activity, dim=-1)
    result = {
        "mode": label,
        "frames": state.frame_index,
        "elapsed_seconds": time.monotonic() - began,
        "rms": float(np.sqrt(np.mean(wave.astype(np.float64) ** 2))),
        "predicted_active_frames": int(predicted_activity.sum()),
    }
    return wave, output_codes, predicted_activity, result


def source_audio(row: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    dataset = load_dataset(
        DATASET, split=row["split"], revision=REVISION, streaming=True
    )
    for column in ("channel_a", "channel_b"):
        if isinstance(dataset.features[column], Audio):
            dataset = dataset.cast_column(column, Audio(decode=False))
    source = next((item for item in dataset if item["id"] == row["id"]), None)
    if source is None:
        raise ValueError("Selected conversation is absent from the source split")
    user = _pcm(source["channel_a"])
    target = _pcm(source["channel_b"])
    if (
        int(source["duration_samples"]) != row["samples"]
        or int(source["sample_rate"]) != 24_000
        or len(user) != len(target)
        or len(target) != row["samples"]
    ):
        raise ValueError("Source streams differ from frozen sample metadata")
    return user, target


def stereo(user: np.ndarray, output: np.ndarray) -> np.ndarray:
    if user.ndim != 1 or output.ndim != 1 or len(user) != len(output):
        raise ValueError("Two audio channels must be aligned mono waveforms")
    return np.column_stack((user, output))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--sample-manifest", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--tokens", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--greedy-generated", type=Path, required=True)
    parser.add_argument("--mimi-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wandb-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    completion = json.loads((args.checkpoint.parent / "completed.json").read_text())
    if digest(args.checkpoint) != completion["best_nll_sha256"]:
        raise ValueError("Selected checkpoint is not the best validation checkpoint")
    chosen = [
        item
        for item in json.loads(args.sample_manifest.read_text())
        if item["id"] == args.sample_id and item["selected_split"] == "validation"
    ]
    if len(chosen) != 1:
        raise ValueError("Sample is outside the frozen validation review selection")
    validation = {
        item["id"]: item
        for item in read_manifest(args.splits / "validation.jsonl", "validation")
    }
    row = validation[args.sample_id]
    if row["sha256"] != chosen[0]["token_sha256"]:
        raise ValueError("Token cache identity differs from the frozen selection")
    cached = load_case(args.tokens, row)
    user, target, target_activity = on_gpu(cached)
    if user.shape[-1] != row["frames"]:
        raise ValueError("Cached conversation length changed")
    args.output.mkdir(parents=True)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    codec = load_mimi(args.mimi_checkpoint, "cuda")
    with torch.inference_mode():
        silence = encode(codec, torch.zeros((1, 1, FRAME_SAMPLES), device="cuda"))[
            :, :, 0
        ]
        user_reconstruction = decode(codec, user)[0, 0].float().cpu().numpy()
    greedy_wave, sample_rate = sf.read(args.greedy_generated, dtype="float32")
    if sample_rate != 24_000 or len(greedy_wave) != len(user_reconstruction):
        raise ValueError("Greedy baseline is not aligned to the selected conversation")
    model_payload = load_research_checkpoint(args.checkpoint)
    config = ModelConfig.stage1()
    metadata = model_payload["metadata"]
    if metadata["dataset_revision"] != REVISION or metadata["model_config"] != asdict(
        config
    ):
        raise ValueError("Checkpoint model or dataset identity mismatch")
    model = DauysHybrid(config).cuda().eval()
    model.load_state_dict(model_payload["model"], strict=True)
    model.requires_grad_(False)
    set_context(model, int(model_payload["context_frames"]))
    for sample_cb0 in (False, True):
        wave, codes, predicted_activity, result = generate_sampled(
            model, user, silence, codec, sample_cb0
        )
        label = result["mode"]
        result["activity"] = binary_metrics(
            predicted_activity,
            target_activity[:, : codes.shape[-1]],
        )
        result["first_codebook_unique"] = int(torch.unique(codes[:, 0]).numel())
        path = args.output / "audio" / f"{label}_dialogue_stereo.wav"
        save_wave(path, stereo(user_reconstruction, wave))
        (args.output / f"result_{label}.json").write_text(
            json.dumps(result, indent=2) + "\n"
        )
        print(json.dumps({"generated": label, "path": str(path)}), flush=True)
    del model
    torch.cuda.empty_cache()
    source_user, source_target = source_audio(row)
    samples = row["samples"]
    frames = math.ceil(samples / FRAME_SAMPLES)
    padded_target = np.pad(source_target, (0, frames * FRAME_SAMPLES - samples))
    with torch.inference_mode():
        source_codes = encode(
            codec, torch.from_numpy(padded_target.copy())[None, None].cuda()
        )
        reconstruction = (
            decode(codec, source_codes)[0, 0].float().cpu().numpy()[:samples]
        )
    token_match_fraction = float((source_codes == target).float().mean())
    if token_match_fraction != 1.0:
        raise ValueError("Fresh codec tokens differ from the frozen token cache")
    source_user_padded = np.pad(source_user, (0, frames * FRAME_SAMPLES - samples))
    source_target_padded = np.pad(source_target, (0, frames * FRAME_SAMPLES - samples))
    reconstruction_padded = np.pad(
        reconstruction, (0, frames * FRAME_SAMPLES - samples)
    )
    save_wave(args.output / "audio" / "source_response.wav", source_target)
    save_wave(args.output / "audio" / "mimi_reconstruction.wav", reconstruction)
    save_wave(
        args.output / "audio" / "source_dialogue_stereo.wav",
        stereo(source_user_padded, source_target_padded),
    )
    save_wave(
        args.output / "audio" / "mimi_dialogue_stereo.wav",
        stereo(source_user_padded, reconstruction_padded),
    )
    save_wave(
        args.output / "audio" / "greedy_dialogue_stereo.wav",
        stereo(user_reconstruction, greedy_wave),
    )
    codec_result = {
        "source_samples": samples,
        "frames": frames,
        "token_match_fraction": token_match_fraction,
        "source_rms": float(np.sqrt(np.mean(source_target.astype(np.float64) ** 2))),
        "reconstruction_rms": float(
            np.sqrt(np.mean(reconstruction.astype(np.float64) ** 2))
        ),
        "waveform_correlation": float(np.corrcoef(source_target, reconstruction)[0, 1]),
    }
    (args.output / "codec_result.json").write_text(
        json.dumps(codec_result, indent=2) + "\n"
    )
    results = {
        "sample_id": row["id"],
        "selected_split": "validation",
        "source_split": row["split"],
        "dataset_revision": REVISION,
        "checkpoint_sha256": digest(args.checkpoint),
        "codec_sha256": digest(args.mimi_checkpoint),
        "evaluation_git_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "seed_cb0": SEED,
        "seed_depth": SEED + 1,
        "temperature": TEMPERATURE,
        "top_k": TOP_K,
        "codec": codec_result,
        "greedy_baseline_sha256": digest(args.greedy_generated),
        "modes": {
            label: json.loads((args.output / f"result_{label}.json").read_text())
            for label in ("depth_sampled", "all_sampled")
        },
    }
    (args.output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    entity = private_wandb_entity()
    args.wandb_dir.mkdir(parents=True, exist_ok=True)
    run = wandb.init(
        project="dauys",
        entity=entity,
        group="S1-E005-DIAGNOSTICS",
        name="S1-E005-DIAGNOSTICS-codec-and-sampling",
        dir=str(args.wandb_dir),
        config={
            "sample_id": row["id"],
            "checkpoint_sha256": results["checkpoint_sha256"],
            "dataset_revision": REVISION,
            "temperature": TEMPERATURE,
            "top_k": TOP_K,
            "seed_cb0": SEED,
            "seed_depth": SEED + 1,
        },
    )
    if run is None:
        raise RuntimeError("W&B probe run did not start")
    table = wandb.Table(
        columns=["mode", "conversation_audio"],
        data=[
            [
                label,
                wandb.Audio(
                    str(args.output / "audio" / f"{label}_dialogue_stereo.wav"),
                    sample_rate=24_000,
                ),
            ]
            for label in (
                "source",
                "mimi",
                "greedy",
                "depth_sampled",
                "all_sampled",
            )
        ],
    )
    run.log(
        {
            "eval/codebooks/codec_waveform_correlation": codec_result[
                "waveform_correlation"
            ],
            "eval/codebooks/codec_rms_ratio": (
                codec_result["reconstruction_rms"] / codec_result["source_rms"]
            ),
            "eval/audio/decoding_probes": table,
        }
    )
    artifact = wandb.Artifact("S1-E005-decoding-probe-results", type="evaluation")
    artifact.add_file(str(args.output / "results.json"))
    run.log_artifact(artifact)
    results["wandb_url"] = run.url
    run.finish()
    (args.output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps({"completed": True, "wandb_url": results["wandb_url"]}))


if __name__ == "__main__":
    main()
