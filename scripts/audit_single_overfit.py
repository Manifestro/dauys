"""Apply exact-token and listening gates to a single-excerpt checkpoint."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from multi_excerpt import _rollout, _sha256
from overfit import DATASET, REVISION, _activity, _load_conversation

from dauys.checkpoint import load_research_checkpoint
from dauys.codec import decode, encode, load_mimi
from dauys.config import ModelConfig
from dauys.model import DauysHybrid


def _rms(signal: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(signal.astype(np.float64)))))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--mimi-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available() or args.output.exists():
        raise RuntimeError("CUDA is required and output must not exist")
    checkpoint = load_research_checkpoint(args.checkpoint)
    metadata = checkpoint["metadata"]
    config = ModelConfig.stage1()
    if metadata["dataset"] != DATASET or metadata["dataset_revision"] != REVISION:
        raise ValueError("Dataset revision mismatch")
    if metadata["model_config"] != asdict(config):
        raise ValueError("Model configuration mismatch")
    frames = int(metadata["frames"])
    samples = frames * 1920
    user_pcm, target_pcm = _load_conversation(metadata["conversation_id"])
    user_pcm = user_pcm[:samples].copy()
    target_pcm = target_pcm[:samples].copy()
    codec = load_mimi(args.mimi_checkpoint, "cuda")
    model = DauysHybrid(config).cuda().eval()
    model.load_state_dict(checkpoint["model"], strict=True)
    args.output.mkdir(parents=True)
    with torch.inference_mode():
        user = encode(codec, torch.from_numpy(user_pcm)[None, None].cuda())
        target = encode(codec, torch.from_numpy(target_pcm)[None, None].cuda())
        silence = encode(codec, torch.zeros((1, 1, 1920), device="cuda"))[:, :, 0]
        generated, activity_logits = _rollout(model, user, silence)
        teacher = model(user, target, silence)
        activity = _activity(target_pcm, frames)
        roundtrip = decode(codec, target)[0, 0].float().cpu().numpy()
        wave = decode(codec, generated)[0, 0].float().cpu().numpy()
        speech = activity.bool().expand_as(target[:, :1]).expand_as(target)
        equality = generated == target
        teacher_cb0 = teacher.cb0_logits.argmax(dim=-1)
        assert teacher.depth_logits is not None
        teacher_depth = teacher.depth_logits.argmax(dim=-1)
        result = {
            "checkpoint_sha256": _sha256(args.checkpoint),
            "conversation_id": metadata["conversation_id"],
            "frames": frames,
            "speech_frames": int(activity.sum()),
            "free_running_exact_token_fraction": float(equality.float().mean()),
            "free_running_exact_speech_token_fraction": float(
                equality[speech].float().mean()
            ),
            "free_running_exact_frames": int(equality.all(dim=1).sum()),
            "free_running_exact_speech_frames": int(
                equality.all(dim=1)[activity.bool()].sum()
            ),
            "free_running_cb0_speech_top1": float(
                (generated[:, 0][activity.bool()] == target[:, 0][activity.bool()])
                .float()
                .mean()
            ),
            "teacher_cb0_speech_top1": float(
                (teacher_cb0[activity.bool()] == target[:, 0][activity.bool()])
                .float()
                .mean()
            ),
            "teacher_depth_speech_top1": float(
                (
                    teacher_depth[activity.bool()]
                    == target[:, 1:].transpose(1, 2)[activity.bool()]
                )
                .float()
                .mean()
            ),
            "generated_activity_frames": int(activity_logits.sum()),
            "reference_rms": _rms(target_pcm),
            "codec_roundtrip_rms": _rms(roundtrip),
            "generated_rms": _rms(wave),
        }
    sf.write(args.output / "reference.wav", target_pcm, 24_000)
    sf.write(args.output / "codec_roundtrip.wav", roundtrip, 24_000)
    sf.write(args.output / "generated.wav", wave, 24_000)
    (args.output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
