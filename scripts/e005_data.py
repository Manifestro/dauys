"""Encode frozen, synchronized dialogue streams for the E005 run."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from datasets import Audio, load_dataset
from overfit import _pcm

from dauys.codec import decode, encode, load_mimi

DATASET = "manifestro/hohere-1m"
REVISION = "2e3880f30e8725ed242d9d2f71e0794d62ed1795"
FRAME_SAMPLES = 1920


def family(scenario: str) -> str:
    parts = scenario.split("/", 1)
    if len(parts) != 2 or not parts[1]:
        raise ValueError(f"Invalid scenario: {scenario}")
    return parts[1]


def activity_from_events(
    events: list[dict], frames: int, speaker: str = "B"
) -> np.ndarray:
    active = np.zeros(frames, dtype=np.int64)
    for event in events:
        if event["speaker"] != speaker:
            continue
        start = max(0, int(event["start_sample"]) // FRAME_SAMPLES)
        end = min(frames, math.ceil(int(event["end_sample"]) / FRAME_SAMPLES))
        if end <= start:
            raise ValueError("Invalid model-side speech event")
        active[start:end] = 1
    return active


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mimi-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation"), required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for codec preprocessing")
    if args.limit is not None and args.limit < 1:
        raise ValueError("Limit must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / f"{args.split}.jsonl"
    existing = {}
    if manifest_path.exists():
        for line in manifest_path.read_text().splitlines():
            row = json.loads(line)
            existing[row["id"]] = row
    codec = load_mimi(args.mimi_checkpoint, "cuda")
    dataset = load_dataset(DATASET, split=args.split, revision=REVISION, streaming=True)
    for column in ("channel_a", "channel_b"):
        if isinstance(dataset.features[column], Audio):
            dataset = dataset.cast_column(column, Audio(decode=False))
    seen = 0
    with manifest_path.open("a", encoding="utf-8") as manifest:
        for row in dataset:
            identifier = row["id"]
            if not identifier.startswith("conv_") or not identifier[5:].isdigit():
                raise ValueError(f"Unsafe conversation identifier: {identifier}")
            seen += 1
            if args.limit is not None and seen > args.limit:
                break
            path = args.output / args.split / f"{identifier}.pt"
            if identifier in existing:
                if not path.is_file() or digest(path) != existing[identifier]["sha256"]:
                    raise RuntimeError(f"Invalid cached conversation: {identifier}")
                continue
            user = _pcm(row["channel_a"])
            target = _pcm(row["channel_b"])
            samples = int(row["duration_samples"])
            if (
                len(user) != samples
                or len(target) != samples
                or int(row["sample_rate"]) != 24_000
                or samples < FRAME_SAMPLES
            ):
                raise ValueError(f"Invalid synchronized audio: {identifier}")
            frames = math.ceil(samples / FRAME_SAMPLES)
            padded = []
            for pcm in (user, target):
                padded.append(np.pad(pcm, (0, frames * FRAME_SAMPLES - samples)))
            with torch.inference_mode():
                encoded = [
                    encode(codec, torch.from_numpy(pcm.copy())[None, None].cuda())
                    for pcm in padded
                ]
            if any(codes.shape != (1, 8, frames) for codes in encoded):
                raise ValueError(f"Codec alignment mismatch: {identifier}")
            activity = activity_from_events(row["events"], frames)
            user_activity = activity_from_events(row["events"], frames, "A")
            rms = np.sqrt(
                np.mean(np.square(padded[1].reshape(frames, FRAME_SAMPLES)), axis=1)
            )
            discrepancy = int(np.count_nonzero(activity != (rms > 1e-4)))
            cache = {
                "user": encoded[0][0].short().cpu(),
                "target": encoded[1][0].short().cpu(),
                "activity": torch.from_numpy(activity.astype(np.uint8)),
                "user_activity": torch.from_numpy(user_activity.astype(np.uint8)),
            }
            path.parent.mkdir(parents=True, exist_ok=True)
            temp = path.with_suffix(".tmp")
            torch.save(cache, temp)
            temp.replace(path)
            item = {
                "id": identifier,
                "split": args.split,
                "dataset": DATASET,
                "dataset_revision": REVISION,
                "scenario": row["scenario"],
                "scenario_family": family(row["scenario"]),
                "skill": row["skill"],
                "voice_a": row["voice_a"],
                "voice_b": row["voice_b"],
                "samples": samples,
                "frames": frames,
                "active_frames": int(activity.sum()),
                "user_active_frames": int(user_activity.sum()),
                "activity_rms_disagreement": discrepancy,
                "sha256": digest(path),
            }
            manifest.write(json.dumps(item) + "\n")
            manifest.flush()
            print(json.dumps(item), flush=True)
            if seen == 1:
                with torch.inference_mode():
                    round_trip = decode(codec, encoded[1]).float()
                expected = torch.from_numpy(padded[1])[None, None]
                if round_trip.shape != expected.shape or not torch.isfinite(round_trip).all():
                    raise RuntimeError("Mimi round trip failed")
        if args.limit is None and len(existing) + seen < 1:
            raise RuntimeError("No conversations found")


if __name__ == "__main__":
    main()
