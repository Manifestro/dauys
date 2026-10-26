"""Deterministic, nested dialogue selections for Stage 1 research."""

import hashlib
from typing import Any

FRAME_SAMPLES = 1920
FRAME_RATE = 12.5


def _rank(row: dict[str, Any], seed: int) -> str:
    value = f"{seed}:{row['id']}".encode()
    return hashlib.sha256(value).hexdigest()


def _select(
    rows: list[dict[str, Any]], split: str, minutes: int, seed: int
) -> list[dict[str, Any]]:
    remaining = minutes * 60 * 25 // 2
    selected = []
    for row in sorted(
        (item for item in rows if item["split"] == split),
        key=lambda item: _rank(item, seed),
    ):
        available = int(row["duration_samples"]) // FRAME_SAMPLES
        take = min(available, remaining)
        if take < 8:
            break
        selected.append(
            {
                "id": row["id"],
                "source_split": split,
                "scenario": row["scenario"],
                "frames": take,
            }
        )
        remaining -= take
        if not remaining:
            break
    if remaining >= 8:
        raise ValueError(f"Insufficient {split} audio for {minutes} minutes")
    return selected


def stage1_selections(inventory: dict[str, Any], seed: int = 42) -> dict[str, Any]:
    """Keep E004 inside E005 while reserving source test scenarios."""
    rows = inventory["rows"]
    e005 = {
        "train": _select(rows, "train", 100, seed),
        "validation": _select(rows, "validation", 10, seed),
        "test": _select(rows, "test", 10, seed),
    }
    e004 = {
        "train": _select(rows, "train", 25, seed),
        "validation": _select(rows, "validation", 5, seed),
    }
    for part in (e004, e005):
        groups = [{row["scenario"] for row in part[name]} for name in part]
        if any(
            groups[i] & groups[j]
            for i in range(len(groups))
            for j in range(i + 1, len(groups))
        ):
            raise ValueError("A scenario appears in multiple data splits")
    return {
        "seed": seed,
        "dataset": inventory["dataset"],
        "dataset_revision": inventory["revision"],
        "manifest_sha256": inventory["manifest_sha256"],
        "e004": e004,
        "e005": e005,
    }
