"""Freeze E005 split metadata and verify disjoint scenario families."""

import argparse
import json
import math
from pathlib import Path

from datasets import Audio, load_dataset
from e005_data import DATASET, FRAME_SAMPLES, REVISION, digest, family
from e005_train import read_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    train = read_manifest(args.tokens / "train.jsonl", "train")
    validation = read_manifest(args.tokens / "validation.jsonl", "validation")
    if (len(train), len(validation)) != (3139, 372):
        raise ValueError("Train and validation manifests are incomplete")
    dataset = load_dataset(DATASET, split="test", revision=REVISION, streaming=True)
    for column in ("channel_a", "channel_b"):
        if isinstance(dataset.features[column], Audio):
            dataset = dataset.cast_column(column, Audio(decode=False))
    test = []
    for row in dataset:
        test.append(
            {
                "id": row["id"],
                "split": "test",
                "dataset": DATASET,
                "dataset_revision": REVISION,
                "scenario": row["scenario"],
                "scenario_family": family(row["scenario"]),
                "skill": row["skill"],
                "voice_a": row["voice_a"],
                "voice_b": row["voice_b"],
                "samples": int(row["duration_samples"]),
                "frames": math.ceil(int(row["duration_samples"]) / FRAME_SAMPLES),
            }
        )
    if len(test) != 397:
        raise ValueError(f"Expected 397 frozen test conversations, got {len(test)}")
    splits = {"train": train, "validation": validation, "test": test}
    ids = [{row["id"] for row in rows} for rows in splits.values()]
    families = [{row["scenario_family"] for row in rows} for rows in splits.values()]
    for i in range(3):
        for j in range(i + 1, 3):
            if ids[i] & ids[j] or families[i] & families[j]:
                raise ValueError("Conversation or scenario leakage across splits")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row) + "\n" for row in test), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "dataset_revision": REVISION,
                "counts": {name: len(rows) for name, rows in splits.items()},
                "hours": {
                    name: sum(row["samples"] for row in rows) / 24_000 / 3600
                    for name, rows in splits.items()
                },
                "families": {name: len(group) for name, group in zip(splits, families)},
                "test_manifest_sha256": digest(args.output),
            }
        )
    )


if __name__ == "__main__":
    main()
