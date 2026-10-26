"""Audit codec reconstruction and generated speech tokens on active excerpts."""

import argparse
import json
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


def _rms(signal: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(signal.astype(np.float64)))))


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
    results: list[dict[str, Any]] = []
    with torch.inference_mode():
        for split in ("train", "validation"):
            selected = splits["e004"][split][:3]
            windows = _load_windows(selected, split, codec, args.frames)
            for row in selected:
                window = max(
                    (w for w in windows if w.conversation_id == row["id"]),
                    key=lambda w: int(w.activity.sum()),
                )
                assert window.target_pcm is not None and window.user_pcm is not None
                user = window.user[None].cuda()
                target = window.target[None].cuda()
                activity = window.activity[None].cuda()
                generated, predicted_activity = _rollout(model, user, silence)
                teacher = model(user, target, silence)
                self_history = model(user, target, silence, history_codes=generated)
                mask = activity.bool()
                silent_mask = ~mask
                teacher_speech = training_loss(teacher, target, activity, mask, config)
                self_speech = training_loss(
                    self_history, target, activity, mask, config
                )
                target_pcm = window.target_pcm
                roundtrip = decode(codec, target)[0, 0].float().cpu().numpy()
                wave = decode(codec, generated)[0, 0].float().cpu().numpy()
                cb0 = teacher.cb0_logits.argmax(dim=-1)
                depth = teacher.depth_logits.argmax(dim=-1)
                target_depth = target[:, 1:].transpose(1, 2)
                label = f"{split}_{row['id']}"
                sf.write(args.output / f"{label}_reference.wav", target_pcm, 24_000)
                sf.write(
                    args.output / f"{label}_codec_roundtrip.wav", roundtrip, 24_000
                )
                sf.write(args.output / f"{label}_generated.wav", wave, 24_000)
                result: dict[str, Any] = {
                    "split": split,
                    "conversation_id": row["id"],
                    "start_frame": window.start_frame,
                    "frames": window.frames,
                    "speech_frames": int(mask.sum()),
                    "teacher_speech_nll": _nll(teacher_speech),
                    "self_history_speech_nll": _nll(self_speech),
                    "teacher_cb0_top1_speech": float(
                        (cb0[mask] == target[:, 0][mask]).float().mean()
                    ),
                    "teacher_depth_top1_speech": float(
                        (depth[mask] == target_depth[mask]).float().mean()
                    ),
                    "generated_exact_speech_frames": int(
                        (generated == target).all(dim=1)[mask].sum()
                    ),
                    "generated_codebook_unique": [
                        int(torch.unique(generated[:, q]).numel()) for q in range(8)
                    ],
                    "target_codebook_unique": [
                        int(torch.unique(target[:, q]).numel()) for q in range(8)
                    ],
                    "predicted_activity_frames": int(predicted_activity.sum()),
                    "reference_rms": _rms(target_pcm),
                    "roundtrip_rms": _rms(roundtrip),
                    "generated_rms": _rms(wave),
                }
                if bool(silent_mask.any()):
                    silent = training_loss(
                        teacher, target, activity, silent_mask, config
                    )
                    result["teacher_silent_nll"] = _nll(silent)
                results.append(result)
                print(json.dumps(result), flush=True)
    (args.output / "results.json").write_text(
        json.dumps(
            {
                "checkpoint_sha256": _sha256(args.checkpoint),
                "splits_sha256": _sha256(args.splits),
                "cases": results,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
