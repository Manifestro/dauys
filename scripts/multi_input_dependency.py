"""Measure input dependence on disjoint multi-excerpt train and validation cases."""

import argparse
import json
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
from multi_excerpt import _load_windows, _nll, _rollout, _sha256
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
    codec = load_mimi(args.mimi_checkpoint, "cuda")
    model = DauysHybrid(config).cuda().eval()
    model.load_state_dict(checkpoint["model"], strict=True)
    with torch.inference_mode():
        silence = encode(codec, torch.zeros((1, 1, 1920), device="cuda"))[:, :, 0]
    args.output.mkdir(parents=True)
    results = []
    with torch.inference_mode():
        for split in ("train", "validation"):
            # IDs and windows are fixed by the precommitted split manifest.
            selected = splits["e004"][split][:3]
            windows = _load_windows(selected, split, codec, args.frames)
            cases = [
                next(w for w in windows if w.conversation_id == row["id"])
                for row in selected
            ]
            for index, case in enumerate(cases):
                alternate = cases[(index + 1) % len(cases)]
                assert case.user_pcm is not None and case.target_pcm is not None
                assert alternate.user_pcm is not None
                length = min(case.frames, alternate.frames)
                target = case.target[:, :length][None].cuda()
                activity = case.activity[:length][None].cuda()
                original = case.user[:, :length][None].cuda()
                shuffled = alternate.user[:, :length][None].cuda()
                baseline_codes, baseline_activity = _rollout(model, original, silence)
                modes = {
                    "original": (original, case.user_pcm[: length * 1920]),
                    "silence": (
                        silence[:, :, None].expand_as(original),
                        np.zeros(length * 1920, dtype=np.float32),
                    ),
                    "shuffled": (shuffled, alternate.user_pcm[: length * 1920]),
                }
                case_result: dict[str, Any] = {
                    "split": split,
                    "conversation_id": case.conversation_id,
                    "alternate_id": alternate.conversation_id,
                    "frames": length,
                    "modes": {},
                }
                prefix = f"{split}_{index}"
                sf.write(
                    args.output / f"{prefix}_target.wav",
                    case.target_pcm[: length * 1920],
                    24_000,
                )
                for mode, (user, pcm) in modes.items():
                    generated, predicted_activity = (
                        (baseline_codes, baseline_activity)
                        if mode == "original"
                        else _rollout(model, user, silence)
                    )
                    mask = torch.ones_like(activity, dtype=torch.bool)
                    teacher = training_loss(
                        model(user, target, silence), target, activity, mask, config
                    )
                    fixed = training_loss(
                        model(user, target, silence, history_codes=baseline_codes),
                        target,
                        activity,
                        mask,
                        config,
                    )
                    wave = decode(codec, generated)[0, 0].float().cpu().numpy()
                    sf.write(args.output / f"{prefix}_{mode}_input.wav", pcm, 24_000)
                    sf.write(
                        args.output / f"{prefix}_{mode}_generated.wav", wave, 24_000
                    )
                    predicted = int(predicted_activity.sum())
                    target_count = int(activity.sum())
                    true_positive = int(
                        (predicted_activity.bool() & activity.bool()).sum()
                    )
                    case_result["modes"][mode] = {
                        "teacher_nll": _nll(teacher),
                        "fixed_original_history_nll": _nll(fixed),
                        "different_frames_vs_original": int(
                            (generated != baseline_codes).any(dim=1).sum()
                        ),
                        "different_activity_frames_vs_original": int(
                            (predicted_activity != baseline_activity).sum()
                        ),
                        "speech_frames": predicted,
                        "activity_f1": 2
                        * true_positive
                        / max(predicted + target_count, 1),
                        "generated_rms": float(
                            np.sqrt(np.mean(np.square(wave.astype(np.float64))))
                        ),
                    }
                results.append(case_result)
                print(json.dumps(case_result), flush=True)
    document = {
        "checkpoint_sha256": _sha256(args.checkpoint),
        "splits_sha256": _sha256(args.splits),
        "code_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "cases": results,
    }
    (args.output / "results.json").write_text(json.dumps(document, indent=2) + "\n")


if __name__ == "__main__":
    main()
