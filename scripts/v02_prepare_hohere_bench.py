"""Materialize the pinned Hohere benchmark as immutable Dauys suite manifests."""

import argparse
import hashlib
import json
import os
from pathlib import Path

from datasets import Audio, load_dataset
from huggingface_hub import HfApi

from dauys.v02_hohere import companion_from_label, materialize_bench_row

CONFIGS = {
    "E1": "e1_conversations",
    "E2": "e2_single_turn",
    "E3": "default",
    "E4": "e4_memory",
    "E5": "e5_interruptions",
    "E6": "e6_long_turns",
}


def _rows(repo: str, config: str, revision: str):
    dataset = load_dataset(repo, config, split="test", revision=revision, token=True)
    for name, feature in dataset.features.items():
        if isinstance(feature, Audio):
            dataset = dataset.cast_column(name, Audio(decode=False))
    return dataset


def _atomic_json(path: Path, value: dict) -> str:
    payload = (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    return hashlib.sha256(payload).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default="manifestro/hohere-bench")
    parser.add_argument("--revision", default="v0.2.0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.output
    root.mkdir(parents=True, exist_ok=True)
    resolved = (
        HfApi()
        .repo_info(
            repo_id=args.repo, repo_type="dataset", revision=args.revision, token=True
        )
        .sha
    )
    if not resolved:
        raise ValueError("Could not resolve the Hohere benchmark revision")
    labels = {str(row["id"]): row for row in _rows(args.repo, "e1_labels", resolved)}
    if not labels:
        raise ValueError("Hohere E1 frame labels are empty")
    e1_families: dict[str, str] = {}
    summaries = []
    for name, config in CONFIGS.items():
        items = []
        for row in _rows(args.repo, config, resolved):
            item = materialize_bench_row(row, name, root)
            if name == "E1":
                identity = item["id"]
                if identity not in labels:
                    raise ValueError(f"Missing E1 frame labels: {identity}")
                item["labels"] = companion_from_label(labels[identity], item["frames"])
                scenario = row.get("scenario")
                if scenario and "/" in scenario:
                    item["family"] = scenario.split("/", 1)[1]
                e1_families[identity] = item["family"]
            elif name in ("E2", "E5"):
                source_id = item.get("analysis", {}).get("source_id")
                if source_id not in e1_families:
                    raise ValueError(f"Unknown E1 source conversation: {source_id}")
                item["family"] = e1_families[source_id]
            items.append(item)
        if not items or len({item["id"] for item in items}) != len(items):
            raise ValueError(f"{name} has no items or duplicate IDs")
        if name == "E1" and set(labels) != {item["id"] for item in items}:
            raise ValueError("E1 labels and conversations differ")
        path = root / f"{name}.json"
        digest = _atomic_json(
            path,
            {
                "name": name,
                "version": args.revision,
                "source_repo": args.repo,
                "source_commit": resolved,
                "items": items,
            },
        )
        (root / f"{name}.sha256").write_text(digest + "\n")
        summaries.append({"suite": name, "items": len(items), "sha256": digest})
        print(json.dumps(summaries[-1]), flush=True)
    _atomic_json(
        root / "provenance.json",
        {
            "repo": args.repo,
            "revision": args.revision,
            "commit": resolved,
            "suites": summaries,
        },
    )


if __name__ == "__main__":
    main()
