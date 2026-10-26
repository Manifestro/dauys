"""Evaluate the frozen generalization checkpoint on held-out dialogues."""

import argparse
import json
import random
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
from generalization import (
    evaluate_teacher,
    selected_conversations,
    sha256,
    validate_selection,
)
from overfit_ten import Conversation, on_device, rollout

from dauys.checkpoint import load_research_checkpoint
from dauys.codec import decode, encode, load_mimi
from dauys.config import ModelConfig
from dauys.loss import training_loss
from dauys.model import DauysHybrid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--mimi-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--blind-seed", type=int, default=19689)
    return parser.parse_args()


def nll(losses: Any) -> float:
    return float((losses.cb0 + 7 * losses.depth + losses.activity).detach())


def closest_other(
    cases: list[Conversation], case: Conversation, families: dict[str, str]
) -> Conversation:
    return min(
        (
            item
            for item in cases
            if item.identifier != case.identifier
            and families[item.identifier] != families[case.identifier]
        ),
        key=lambda item: (abs(item.samples - case.samples), item.identifier),
    )


def matched_input(
    alternate: Conversation, frames: int, samples: int, silence: torch.Tensor
) -> tuple[torch.Tensor, np.ndarray, int]:
    codes = alternate.user.cuda()
    pcm = alternate.user_pcm[: alternate.samples]
    if codes.shape[-1] >= frames:
        codes = codes[:, :, :frames]
        pcm = pcm[:samples]
    else:
        missing = frames - codes.shape[-1]
        codes = torch.cat((codes, silence[:, :, None].expand(-1, -1, missing)), dim=-1)
        pcm = np.pad(pcm, (0, max(samples - len(pcm), 0)))
    return codes, pcm, int(alternate.frames - frames)


def mode_result(
    model: DauysHybrid,
    codec: Any,
    user: torch.Tensor,
    target: torch.Tensor,
    activity: torch.Tensor,
    baseline: torch.Tensor,
    silence: torch.Tensor,
    samples: int,
    output: Path,
) -> dict[str, Any]:
    with torch.inference_mode():
        generated, predicted_activity = rollout(model, user, silence)
        target_prediction = model(user, target, silence)
        generated_prediction = model(user, target, silence, history_codes=baseline)
        mask = torch.ones_like(activity, dtype=torch.bool)
        target_loss = training_loss(
            target_prediction, target, activity, mask, model.config
        )
        generated_loss = training_loss(
            generated_prediction, target, activity, mask, model.config
        )
        wave = decode(codec, generated)[0, 0].float().cpu().numpy()[:samples]
    sf.write(output, wave, 24_000)
    true_positive = int((predicted_activity.bool() & activity.bool()).sum())
    predicted = int(predicted_activity.sum())
    target_count = int(activity.sum())
    return {
        "target_history_nll_nats_per_frame": nll(target_loss),
        "fixed_generated_history_nll_nats_per_frame": nll(generated_loss),
        "different_frames_vs_original": int((generated != baseline).any(dim=1).sum()),
        "speech_activity_f1": 2 * true_positive / max(predicted + target_count, 1),
        "predicted_speech_frames": predicted,
        "target_speech_frames": target_count,
        "generated_rms": float(np.sqrt(np.mean(np.square(wave.astype(np.float64))))),
        "output_sha256": sha256(output),
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available() or args.output.exists():
        raise RuntimeError("CUDA is required and output must not exist")
    splits = json.loads(args.splits.read_text(encoding="utf-8"))
    selection = validate_selection(splits)
    checkpoint = load_research_checkpoint(args.checkpoint)
    metadata = checkpoint["metadata"]
    config = ModelConfig.stage1()
    if metadata["model_config"] != asdict(config):
        raise ValueError("Model configuration mismatch")
    if metadata["splits_sha256"] != sha256(args.splits):
        raise ValueError("Selection mismatch")
    if metadata["training_mode"] != "joint_causal_mixed_history":
        raise ValueError("Checkpoint was not trained with the frozen method")
    if int(metadata["steps"]) != 100_000:
        raise ValueError("Unexpected training budget")
    if checkpoint["validation"] is None:
        raise ValueError("Checkpoint was not selected by validation")
    args.output.mkdir(parents=True)
    codec = load_mimi(args.mimi_checkpoint, "cuda")
    model = DauysHybrid(config).cuda().eval()
    model.load_state_dict(checkpoint["model"], strict=True)
    with torch.inference_mode():
        silence = encode(codec, torch.zeros((1, 1, 1920), device="cuda"))[:, :, 0]
    cases = selected_conversations(selection["test"], codec)
    families = {row["id"]: row["scenario_family"] for row in selection["test"]}
    aggregate = evaluate_teacher(model, cases, silence, config)
    aggregate_silent = evaluate_teacher(
        model, cases, silence, config, silent_input=True
    )
    rng = random.Random(args.blind_seed)
    report: list[dict[str, Any]] = []
    blind: list[dict[str, Any]] = []
    unblind: list[dict[str, Any]] = []
    for index, case in enumerate(cases):
        alternate = closest_other(cases, case, families)
        user, target, activity = on_device(case)
        shuffled, shuffled_pcm, length_delta = matched_input(
            alternate, case.frames, case.samples, silence
        )
        original_pcm = case.user_pcm[: case.samples]
        modes = {
            "original": (user, original_pcm),
            "shuffled": (shuffled, shuffled_pcm),
            "silence": (
                silence[:, :, None].expand_as(user),
                np.zeros(case.samples, dtype=np.float32),
            ),
        }
        with torch.inference_mode():
            baseline, _ = rollout(model, user, silence)
        label = f"case_{index:02d}"
        sf.write(args.output / f"{label}_prompt.wav", original_pcm, 24_000)
        sf.write(
            args.output / f"{label}_reference.wav",
            case.target_pcm[: case.samples],
            24_000,
        )
        shuffled_modes = list(modes)
        rng.shuffle(shuffled_modes)
        aliases = dict(zip(shuffled_modes, "ABC", strict=True))
        item: dict[str, Any] = {
            "case_id": label,
            "conversation_id": case.identifier,
            "scenario_family": selection["test"][index]["scenario_family"],
            "alternate_id": alternate.identifier,
            "alternate_length_delta_frames": length_delta,
            "frames": case.frames,
            "modes": {},
        }
        for mode, (input_codes, input_pcm) in modes.items():
            alias = aliases[mode]
            filename = f"{label}_{alias}.wav"
            metrics = mode_result(
                model,
                codec,
                input_codes,
                target,
                activity,
                baseline,
                silence,
                case.samples,
                args.output / filename,
            )
            metrics["input_rms"] = float(
                np.sqrt(np.mean(np.square(input_pcm.astype(np.float64))))
            )
            item["modes"][mode] = metrics
        original_nll = item["modes"]["original"]["target_history_nll_nats_per_frame"]
        item["shuffled_minus_original_nll"] = (
            item["modes"]["shuffled"]["target_history_nll_nats_per_frame"]
            - original_nll
        )
        item["silent_minus_original_nll"] = (
            item["modes"]["silence"]["target_history_nll_nats_per_frame"] - original_nll
        )
        report.append(item)
        blind.append(
            {
                "case_id": label,
                "prompt": f"{label}_prompt.wav",
                "answers": [f"{label}_{alias}.wav" for alias in "ABC"],
                "rating_fields": ["intelligibility_0_to_5", "relevance_0_to_5"],
            }
        )
        unblind.append({"case_id": label, "modes": aliases})
        print(json.dumps(item), flush=True)
    positive = sum(item["shuffled_minus_original_nll"] > 0 for item in report)
    result = {
        "checkpoint_sha256": sha256(args.checkpoint),
        "checkpoint_step": int(checkpoint["step"]),
        "splits_sha256": sha256(args.splits),
        "code_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "test_teacher_original": aggregate,
        "test_teacher_silent": aggregate_silent,
        "positive_shuffled_delta_cases": positive,
        "test_cases": len(cases),
        "cases": report,
        "blind_rating_status": "awaiting_human_ratings",
        "speaker_match_status": "unknown_no_speaker_metadata",
    }
    (args.output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    (args.output / "blind_manifest.json").write_text(json.dumps(blind, indent=2) + "\n")
    (args.output / "unblind_key.json").write_text(json.dumps(unblind, indent=2) + "\n")
    print(json.dumps({"positive_shuffled_delta_cases": positive}), flush=True)


if __name__ == "__main__":
    main()
