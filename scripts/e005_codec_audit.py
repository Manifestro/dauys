"""Check a seeded dataset audio round trip through the frozen codec."""

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from datasets import Audio, load_dataset
from e005_data import DATASET, FRAME_SAMPLES, REVISION
from e005_train import read_manifest
from overfit import _pcm

from dauys.codec import decode, encode, load_mimi


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=Path, required=True)
    parser.add_argument("--mimi-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    manifest = read_manifest(args.tokens / "train.jsonl", "train")
    if len(manifest) < 64:
        raise ValueError("At least 64 cached training conversations are required")
    chosen = random.Random(42).choice(manifest[:64])
    dataset = load_dataset(DATASET, split="train", revision=REVISION, streaming=True)
    for column in ("channel_a", "channel_b"):
        if isinstance(dataset.features[column], Audio):
            dataset = dataset.cast_column(column, Audio(decode=False))
    row = next(item for item in dataset if item["id"] == chosen["id"])
    user = _pcm(row["channel_a"])
    target = _pcm(row["channel_b"])
    samples = int(row["duration_samples"])
    frames = math.ceil(samples / FRAME_SAMPLES)
    if len(user) != len(target) or len(user) != samples or frames != chosen["frames"]:
        raise ValueError("Source channels are not aligned")
    codec = load_mimi(args.mimi_checkpoint, "cuda")
    audio = np.pad(target, (0, frames * FRAME_SAMPLES - samples))
    with torch.inference_mode():
        codes = encode(codec, torch.from_numpy(audio.copy())[None, None].cuda())
        reconstructed = decode(codec, codes)[0, 0].float().cpu().numpy()[:samples]
    if (
        codes.shape != (1, 8, frames)
        or reconstructed.shape != target.shape
        or not np.isfinite(reconstructed).all()
        or not np.any(reconstructed)
    ):
        raise RuntimeError("Dataset codec round trip failed")
    args.output.mkdir(parents=True)
    sf.write(args.output / "source.wav", target, 24_000)
    sf.write(args.output / "reconstructed.wav", reconstructed, 24_000)
    result = {
        "conversation_id": chosen["id"],
        "dataset_revision": REVISION,
        "samples": samples,
        "frames": frames,
        "source_rms": float(np.sqrt(np.mean(np.square(target.astype(np.float64))))),
        "reconstructed_rms": float(
            np.sqrt(np.mean(np.square(reconstructed.astype(np.float64))))
        ),
        "waveform_correlation": float(np.corrcoef(target, reconstructed)[0, 1]),
    }
    (args.output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
