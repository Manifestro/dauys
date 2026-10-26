"""Run the frozen E005 test selection once after checkpoint selection."""

import argparse
import json
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
from multi_excerpt import _evaluate, _load_windows, _nll, _rollout, _sha256
from overfit import DATASET, REVISION

from dauys.checkpoint import load_research_checkpoint
from dauys.codec import decode, encode, load_mimi
from dauys.config import ModelConfig
from dauys.loss import training_loss
from dauys.model import DauysHybrid


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--mimi-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=128)
    args = parser.parse_args()
    if not torch.cuda.is_available() or args.output.exists():
        raise RuntimeError("CUDA is required and output must not exist")
    splits = json.loads(args.splits.read_text())
    checkpoint = load_research_checkpoint(args.checkpoint)
    metadata = checkpoint["metadata"]
    config = ModelConfig.stage1()
    if (splits["dataset"], splits["dataset_revision"]) != (DATASET, REVISION):
        raise ValueError("Dataset revision mismatch")
    if metadata["model_config"] != asdict(config):
        raise ValueError("Model configuration mismatch")
    if metadata["splits_sha256"] != _sha256(args.splits):
        raise ValueError("Selection mismatch")
    if metadata["train_frames"] != sum(x["frames"] for x in splits["e005"]["train"]):
        raise ValueError("Checkpoint is not from the selected E005 training run")
    codec = load_mimi(args.mimi_checkpoint, "cuda")
    model = DauysHybrid(config).cuda().eval()
    model.load_state_dict(checkpoint["model"], strict=True)
    with torch.inference_mode():
        silence = encode(codec, torch.zeros((1, 1, 1920), device="cuda"))[:, :, 0]
    args.output.mkdir(parents=True)
    selected = splits["e005"]["test"]
    windows = _load_windows(selected, "test", codec, args.frames)
    with torch.inference_mode():
        aggregate = _evaluate(model, windows, silence, config)
        cases = [
            max(
                (w for w in windows if w.conversation_id == row["id"]),
                key=lambda w: int(w.activity.sum()),
            )
            for row in selected
        ]
        results: list[dict[str, Any]] = []
        for index, case in enumerate(cases):
            alternate = cases[(index + 1) % len(cases)]
            assert case.user_pcm is not None and case.target_pcm is not None
            assert alternate.user_pcm is not None
            length = min(args.frames, case.frames, alternate.frames)
            target = case.target[:, :length][None].cuda()
            activity = case.activity[:length][None].cuda()
            original = case.user[:, :length][None].cuda()
            shuffled = alternate.user[:, :length][None].cuda()
            baseline, baseline_activity = _rollout(model, original, silence)
            modes = {
                "original": (original, case.user_pcm[: length * 1920]),
                "silence": (
                    silence[:, :, None].expand_as(original),
                    np.zeros(length * 1920, dtype=np.float32),
                ),
                "shuffled": (shuffled, alternate.user_pcm[: length * 1920]),
            }
            item: dict[str, Any] = {
                "conversation_id": case.conversation_id,
                "scenario": selected[index]["scenario"],
                "alternate_id": alternate.conversation_id,
                "start_frame": case.start_frame,
                "frames": length,
                "target_speech_frames": int(activity.sum()),
                "modes": {},
            }
            prefix = f"test_{index}"
            sf.write(
                args.output / f"{prefix}_target.wav",
                case.target_pcm[: length * 1920],
                24_000,
            )
            for mode, (user, pcm) in modes.items():
                generated, predicted_activity = (
                    (baseline, baseline_activity)
                    if mode == "original"
                    else _rollout(model, user, silence)
                )
                prediction = model(user, target, silence, history_codes=baseline)
                loss = training_loss(
                    prediction,
                    target,
                    activity,
                    torch.ones_like(activity, dtype=torch.bool),
                    config,
                )
                wave = decode(codec, generated)[0, 0].float().cpu().numpy()
                sf.write(args.output / f"{prefix}_{mode}_input.wav", pcm, 24_000)
                sf.write(args.output / f"{prefix}_{mode}_generated.wav", wave, 24_000)
                predicted = int(predicted_activity.sum())
                target_count = int(activity.sum())
                true_positive = int((predicted_activity.bool() & activity.bool()).sum())
                input_activity = (
                    np.sqrt(
                        np.mean(
                            np.square(pcm.reshape(length, 1920).astype(np.float64)),
                            axis=1,
                        )
                    )
                    > 1e-4
                )
                output_activity = predicted_activity[0].bool().cpu().numpy()
                # Framewise offline proxy; this is not a measured live interruption latency.
                previous_input = np.r_[False, input_activity[:-1]]
                previous_output = np.r_[False, output_activity[:-1]]
                overlap_onsets = np.flatnonzero(
                    input_activity & ~previous_input & previous_output
                )
                cessation_latency_frames = []
                for onset in overlap_onsets:
                    stops = np.flatnonzero(~output_activity[onset:])
                    cessation_latency_frames.append(
                        int(stops[0]) if len(stops) else None
                    )
                item["modes"][mode] = {
                    "fixed_original_history_nll": _nll(loss),
                    "different_frames_vs_original": int(
                        (generated != baseline).any(dim=1).sum()
                    ),
                    "speech_frames": predicted,
                    "activity_f1": 2 * true_positive / max(predicted + target_count, 1),
                    "generated_rms": float(
                        np.sqrt(np.mean(np.square(wave.astype(np.float64))))
                    ),
                    "simultaneous_input_output_frames": int(
                        np.logical_and(input_activity, output_activity).sum()
                    ),
                    "overlap_onsets": len(overlap_onsets),
                    "simulated_cessation_latency_frames": cessation_latency_frames,
                }
            results.append(item)
            print(json.dumps(item), flush=True)
    document = {
        "checkpoint_sha256": _sha256(args.checkpoint),
        "splits_sha256": _sha256(args.splits),
        "code_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "test_teacher_metrics_all_windows": aggregate,
        "cases": results,
    }
    (args.output / "results.json").write_text(json.dumps(document, indent=2) + "\n")


if __name__ == "__main__":
    main()
