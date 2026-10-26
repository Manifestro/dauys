"""Record duration and scenario metadata without decoding audio."""

import argparse
import hashlib
import json
from pathlib import Path

from datasets import Audio, load_dataset
from huggingface_hub import hf_hub_download
from overfit import DATASET, REVISION


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("runs/dataset-inventory.json")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest = hf_hub_download(
        DATASET, "manifest.json", repo_type="dataset", revision=REVISION
    )
    manifest_sha256 = hashlib.sha256(Path(manifest).read_bytes()).hexdigest()
    rows = []
    for split in ("train", "validation", "test"):
        try:
            dataset = load_dataset(
                DATASET, split=split, revision=REVISION, streaming=True
            )
        except ValueError:
            continue
        for column in ("channel_a", "channel_b"):
            if isinstance(dataset.features[column], Audio):
                dataset = dataset.cast_column(column, Audio(decode=False))
        for row in dataset:
            rows.append(
                {
                    "id": row["id"],
                    "split": split,
                    "duration_samples": int(row["duration_samples"]),
                    "duration_seconds": float(row["duration_seconds"]),
                    "sample_rate": int(row["sample_rate"]),
                    "scenario": row["scenario"],
                    "skill": row["skill"],
                }
            )
    if not rows:
        raise ValueError("No conversations were found")
    result = {
        "dataset": DATASET,
        "revision": REVISION,
        "manifest_sha256": manifest_sha256,
        "conversation_count": len(rows),
        "duration_seconds": sum(row["duration_seconds"] for row in rows),
        "rows": rows,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}))


if __name__ == "__main__":
    main()
