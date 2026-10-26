"""Generate a complete held-out conversation without updating model weights."""

import argparse
import hashlib
import json
import math
import subprocess
from dataclasses import asdict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from datasets import Audio, load_dataset
from overfit import DATASET, REVISION, _pcm
from overfit_ten import rollout

from dauys.checkpoint import load_research_checkpoint
from dauys.codec import decode, encode, load_mimi
from dauys.config import ModelConfig
from dauys.model import DauysHybrid


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--mimi-checkpoint", type=Path, required=True)
    parser.add_argument("--conversation-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available() or args.output.exists():
        raise RuntimeError("CUDA is required and output must not exist")
    selection = json.loads(args.selection.read_text())
    if args.conversation_id in {row["id"] for row in selection}:
        raise ValueError("Probe conversation is in the training selection")
    checkpoint = load_research_checkpoint(args.checkpoint)
    metadata = checkpoint["metadata"]
    config = ModelConfig.stage1()
    if metadata["dataset"] != DATASET or metadata["dataset_revision"] != REVISION:
        raise ValueError("Dataset revision mismatch")
    if metadata["model_config"] != asdict(config):
        raise ValueError("Model configuration mismatch")
    if metadata["selection_sha256"] != sha256(args.selection):
        raise ValueError("Training selection mismatch")
    if int(checkpoint["step"]) != 1000:
        raise ValueError("Probe requires the completed final checkpoint")
    dataset = load_dataset(
        DATASET, split="validation", revision=REVISION, streaming=True
    )
    for column in ("channel_a", "channel_b"):
        if isinstance(dataset.features[column], Audio):
            dataset = dataset.cast_column(column, Audio(decode=False))
    row = next((item for item in dataset if item["id"] == args.conversation_id), None)
    if row is None:
        raise ValueError("Conversation is absent from validation")
    user_pcm, target_pcm = _pcm(row["channel_a"]), _pcm(row["channel_b"])
    if len(user_pcm) != len(target_pcm):
        raise ValueError("Channel lengths differ")
    samples = len(user_pcm)
    frames = math.ceil(samples / 1920)
    user_pcm = np.pad(user_pcm, (0, frames * 1920 - samples))
    target_pcm = np.pad(target_pcm, (0, frames * 1920 - samples))
    codec = load_mimi(args.mimi_checkpoint, "cuda")
    model = DauysHybrid(config).cuda().eval()
    model.load_state_dict(checkpoint["model"], strict=True)
    with torch.inference_mode():
        user = encode(codec, torch.from_numpy(user_pcm.copy())[None, None].cuda())
        target = encode(codec, torch.from_numpy(target_pcm.copy())[None, None].cuda())
        silence = encode(codec, torch.zeros((1, 1, 1920), device="cuda"))[:, :, 0]
        generated, predicted_activity = rollout(model, user, silence)
        generated_pcm = decode(codec, generated)[0, 0].float().cpu().numpy()[:samples]
        codec_reference = decode(codec, target)[0, 0].float().cpu().numpy()[:samples]
    args.output.mkdir(parents=True)
    sf.write(args.output / "user.wav", user_pcm[:samples], 24_000)
    sf.write(args.output / "target.wav", target_pcm[:samples], 24_000)
    sf.write(args.output / "codec_reference.wav", codec_reference, 24_000)
    sf.write(args.output / "generated.wav", generated_pcm, 24_000)
    target_activity = (
        np.sqrt(np.mean(np.square(target_pcm.reshape(frames, 1920)), axis=1)) > 1e-4
    )
    generated_activity = predicted_activity[0].bool().cpu().numpy()
    tp = int(np.logical_and(target_activity, generated_activity).sum())
    exact = generated == target
    result = {
        "checkpoint_sha256": sha256(args.checkpoint),
        "selection_sha256": sha256(args.selection),
        "code_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "dataset_revision": REVISION,
        "split": "validation",
        "conversation_id": args.conversation_id,
        "scenario": row["scenario"],
        "samples": samples,
        "frames": frames,
        "exact_tokens": int(exact.sum()),
        "total_tokens": exact.numel(),
        "exact_frames": int(exact.all(dim=1).sum()),
        "target_speech_frames": int(target_activity.sum()),
        "predicted_speech_frames": int(generated_activity.sum()),
        "activity_f1": 2
        * tp
        / max(int(target_activity.sum() + generated_activity.sum()), 1),
        "generated_rms": float(
            np.sqrt(np.mean(np.square(generated_pcm.astype(np.float64))))
        ),
        "target_rms": float(
            np.sqrt(np.mean(np.square(target_pcm[:samples].astype(np.float64))))
        ),
    }
    (args.output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
