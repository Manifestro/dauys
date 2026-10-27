"""Join Hohere training conversations with Stage 1 replay; hold out validation."""

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq
import torch
from huggingface_hub import HfApi, HfFileSystem


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _write(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _split_ids(dataset: str, revision: str, split: str) -> set[str]:
    files = [
        name
        for name in HfApi().list_repo_files(
            dataset, repo_type="dataset", revision=revision
        )
        if name.endswith(".parquet") and split in name
    ]
    if not files:
        raise ValueError(f"No {split} parquet files in {dataset}@{revision}")
    filesystem = HfFileSystem()
    ids: set[str] = set()
    for name in files:
        table = pq.read_table(
            f"datasets/{dataset}@{revision}/{name}",
            columns=["id"],
            filesystem=filesystem,
        )
        ids.update(str(value) for value in table.column("id").to_pylist())
    return ids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ls100-root", type=Path, required=True)
    parser.add_argument("--hohere-root", type=Path, required=True)
    parser.add_argument("--dataset", default="manifestro/hohere-1m")
    parser.add_argument(
        "--revision", default="2e3880f30e8725ed242d9d2f71e0794d62ed1795"
    )
    parser.add_argument("--heldout-items", type=int, default=200)
    parser.add_argument(
        "--cache-root", type=Path, required=True, help="Common root of both caches"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    silence = torch.tensor(
        json.loads((args.ls100_root / "silence_codes.json").read_text())
    )
    hohere_silence = torch.load(
        args.hohere_root / "silence_codes.pt", weights_only=True
    )
    if not torch.equal(silence.long(), hohere_silence["codes"].long()):
        raise ValueError("Stage 1 and Hohere caches use different silence codes")
    validation = _split_ids(args.dataset, args.revision, "validation")
    args.output.mkdir(parents=True, exist_ok=True)
    ls100_prefix = args.ls100_root.resolve().relative_to(args.cache_root.resolve())
    hohere_prefix = args.hohere_root.resolve().relative_to(args.cache_root.resolve())
    items = [
        json.loads(path.read_text())
        for path in sorted((args.hohere_root / "items").glob("*.json"))
    ]
    train_manifest, train_companion, held_manifest, held_companion = [], [], [], []
    for item in items:
        manifest = dict(
            item["manifest"],
            tokens_path=f"{hohere_prefix}/{item['manifest']['tokens_path']}",
        )
        if manifest["id"] in validation:
            held_manifest.append(manifest)
            held_companion.append(item["companion"])
        else:
            train_manifest.append(manifest)
            train_companion.append(item["companion"])
    held_manifest = held_manifest[: args.heldout_items]
    held_companion = held_companion[: args.heldout_items]
    if not held_manifest or not train_manifest:
        raise ValueError("Both Hohere training and held-out items are required")
    for row in _jsonl(args.ls100_root / "manifest.jsonl"):
        train_manifest.append(
            dict(row, tokens_path=f"{ls100_prefix}/{row['tokens_path']}")
        )
    train_companion.extend(_jsonl(args.ls100_root / "companion.jsonl"))
    _write(args.output / "train-manifest.jsonl", train_manifest)
    _write(args.output / "train-companion.jsonl", train_companion)
    _write(args.output / "heldout-manifest.jsonl", held_manifest)
    _write(args.output / "heldout-companion.jsonl", held_companion)
    (args.output / "silence_codes.json").write_text(json.dumps(silence.tolist()) + "\n")
    replay = sum(row["source"] != "hohere" for row in train_manifest)
    summary = {
        "hohere_train": len(train_manifest) - replay,
        "hohere_heldout": len(held_manifest),
        "validation_ids": len(validation),
        "replay": replay,
    }
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
