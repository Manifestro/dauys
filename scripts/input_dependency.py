"""Evaluate response dependence on three synchronized user-audio inputs."""

import argparse
import hashlib
import json
import random
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
from overfit import DATASET, REVISION, _activity, _load_conversation

from dauys.checkpoint import load_research_checkpoint
from dauys.codec import decode, encode, load_mimi
from dauys.config import ModelConfig
from dauys.loss import training_loss
from dauys.model import DauysHybrid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--mimi-checkpoint", type=Path, required=True)
    parser.add_argument("--alternate-conversation-id", required=True)
    parser.add_argument("--output", type=Path, default=Path("runs/S1-E003"))
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rms(signal: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(signal.astype(np.float64)))))


def _correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    if len(left) != len(right):
        raise ValueError("Signals must have equal length")
    left_centered = left.astype(np.float64) - float(np.mean(left))
    right_centered = right.astype(np.float64) - float(np.mean(right))
    denominator = float(np.linalg.norm(left_centered) * np.linalg.norm(right_centered))
    return (
        float(np.dot(left_centered, right_centered) / denominator)
        if denominator
        else None
    )


def _conditional_nll(
    model: DauysHybrid,
    user: torch.Tensor,
    target: torch.Tensor,
    activity: torch.Tensor,
    silence: torch.Tensor,
    config: ModelConfig,
    history: torch.Tensor | None,
) -> dict[str, float]:
    prediction = model(user, target, silence, history_codes=history)
    masks = {
        "all": torch.ones_like(activity, dtype=torch.bool),
        "speech": activity.bool(),
    }
    result = {}
    for name, mask in masks.items():
        if not bool(mask.any()):
            raise ValueError("The target has no active speech frames")
        losses = training_loss(prediction, target, activity, mask, config)
        result[name] = float(losses.cb0 + 7 * losses.depth + losses.activity)
    return result


def _generate(
    model: DauysHybrid, user: torch.Tensor, silence: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    state = model.new_stream(silence)
    generated = []
    predicted_activity = []
    for frame in range(user.shape[-1]):
        codes, activity_logits = model.generate_step(user[:, :, frame], state)
        generated.append(codes)
        predicted_activity.append(activity_logits.argmax(dim=-1))
    return torch.stack(generated, dim=-1), torch.stack(predicted_activity, dim=-1)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Input dependency evaluation requires CUDA")
    if args.output.exists():
        raise FileExistsError(args.output)
    checkpoint = load_research_checkpoint(args.checkpoint)
    training_metadata = checkpoint["metadata"]
    if (
        training_metadata["dataset"] != DATASET
        or training_metadata["dataset_revision"] != REVISION
    ):
        raise ValueError("Checkpoint dataset version differs from evaluation data")
    frames = int(training_metadata["frames"])
    original_id = training_metadata["conversation_id"]
    if args.alternate_conversation_id == original_id:
        raise ValueError("Alternate input must come from a different conversation")
    config = ModelConfig.stage1()
    if training_metadata["model_config"] != asdict(config):
        raise ValueError(
            "Checkpoint model configuration differs from the installed code"
        )
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    original_pcm, target_pcm = _load_conversation(original_id)
    alternate_pcm, _ = _load_conversation(args.alternate_conversation_id)
    samples = frames * 1920
    if min(len(original_pcm), len(target_pcm), len(alternate_pcm)) < samples:
        raise ValueError("A conversation is shorter than the training frame window")
    original_pcm = original_pcm[:samples]
    target_pcm = target_pcm[:samples]
    alternate_pcm = alternate_pcm[:samples]
    inputs = {
        "A_original": original_pcm,
        "B_silence": np.zeros_like(original_pcm),
        "C_alternate": alternate_pcm,
    }
    codec = load_mimi(args.mimi_checkpoint, "cuda")
    model = DauysHybrid(config).cuda()
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    args.output.mkdir(parents=True)
    metadata = {
        "experiment_id": args.output.name,
        "hypothesis": "The trained response depends on the synchronized user stream.",
        "code_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "code_clean": not subprocess.check_output(
            ["git", "status", "--porcelain"], text=True
        ).strip(),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "training_metadata": training_metadata,
        "codec_checkpoint_sha256": _sha256(args.mimi_checkpoint),
        "dataset": DATASET,
        "dataset_revision": REVISION,
        "original_conversation_id": original_id,
        "alternate_conversation_id": args.alternate_conversation_id,
        "frames": frames,
        "seed": args.seed,
        "generation": "deterministic argmax",
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    sf.write(args.output / "target.wav", target_pcm, 24_000)
    activity = _activity(target_pcm, frames)
    with torch.inference_mode():
        target = encode(codec, torch.from_numpy(target_pcm.copy())[None, None].cuda())
        silence = encode(codec, torch.zeros((1, 1, 1920), device="cuda"))[:, :, 0]
        baseline_codes = None
        baseline_activity = None
        encoded_inputs = {}
        generations = {}
        for label, pcm in inputs.items():
            user = encode(codec, torch.from_numpy(pcm.copy())[None, None].cuda())
            if user.shape != target.shape:
                raise ValueError("Codec frame alignment differs across inputs")
            encoded_inputs[label] = user
            generated, predicted_activity = _generate(model, user, silence)
            generations[label] = (generated, predicted_activity)
            if label == "A_original":
                baseline_codes = generated
                baseline_activity = predicted_activity
        assert baseline_codes is not None and baseline_activity is not None
        results: dict[str, dict[str, Any]] = {}
        for label, pcm in inputs.items():
            generated, predicted_activity = generations[label]
            generated_pcm = decode(codec, generated)[0, 0].float().cpu().numpy()
            sf.write(args.output / f"{label}_input.wav", pcm, 24_000)
            sf.write(args.output / f"{label}_generated.wav", generated_pcm, 24_000)
            np.save(args.output / f"{label}_codes.npy", generated[0].cpu().numpy())
            shared = {
                "teacher_target_history": _conditional_nll(
                    model,
                    encoded_inputs[label],
                    target,
                    activity,
                    silence,
                    config,
                    None,
                ),
                "fixed_baseline_generated_history": _conditional_nll(
                    model,
                    encoded_inputs[label],
                    target,
                    activity,
                    silence,
                    config,
                    baseline_codes,
                ),
            }
            matches = (predicted_activity.bool() & activity.bool()).sum().item()
            speech_count = int(predicted_activity.sum())
            results[label] = {
                "conditional_nll_nats_per_frame": shared,
                "different_codebook_tokens_vs_A": int(
                    (generated != baseline_codes).sum()
                ),
                "different_frames_vs_A": int(
                    (generated != baseline_codes).any(dim=1).sum()
                ),
                "different_activity_frames_vs_A": int(
                    (predicted_activity != baseline_activity).sum()
                ),
                "generated_speech_frames": speech_count,
                "target_speech_frames": int(activity.sum()),
                "speech_activity_f1_vs_target": 2
                * matches
                / max(speech_count + int(activity.sum()), 1),
                "input_rms": _rms(pcm),
                "generated_rms": _rms(generated_pcm),
                "waveform_correlation_vs_target": _correlation(
                    generated_pcm, target_pcm
                ),
            }
    (args.output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    report = [
        f"# {args.output.name}: Input dependency test",
        "",
        "One checkpoint and fresh initial streaming state were used in all three modes.",
        "The seed and deterministic decoder were fixed. The original target is used",
        "for every conditional NLL comparison; the alternate input is not an answer target.",
        "",
        "| Mode | NLL, target history | NLL, fixed A history | Different frames | Speech frames | Activity F1 | RMS | Waveform correlation |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label, metrics in results.items():
        nll = metrics["conditional_nll_nats_per_frame"]
        report.append(
            f"| {label} | {nll['teacher_target_history']['all']:.4f} | "
            f"{nll['fixed_baseline_generated_history']['all']:.4f} | "
            f"{metrics['different_frames_vs_A']} | {metrics['generated_speech_frames']} | "
            f"{metrics['speech_activity_f1_vs_target']:.4f} | "
            f"{metrics['generated_rms']:.6f} | "
            f"{metrics['waveform_correlation_vs_target']} |"
        )
    report.extend(
        [
            "",
            "A changed output establishes sensitivity to this input intervention on one",
            "training excerpt. It does not establish semantic understanding, correct",
            "answers, transfer to unseen conversations, or live full-duplex behavior.",
            "Listen to all three generated WAVs before assigning qualitative quality.",
        ]
    )
    (args.output / "report.md").write_text("\n".join(report) + "\n")
    print(json.dumps(results), flush=True)


if __name__ == "__main__":
    main()
