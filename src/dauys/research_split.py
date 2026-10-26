"""Deterministic, nested dialogue selections for Stage 1 research."""

import hashlib
from collections import defaultdict
from typing import Any

FRAME_SAMPLES = 1920
FRAME_RATE = 12.5


def leakage_free_source_splits(
    train: list[dict[str, Any]],
    validation: list[dict[str, Any]],
    test: list[dict[str, Any]],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[str]]]:
    """Keep test and validation families, excluding conflicting earlier rows."""
    test_families = {row["scenario_family"] for row in test}
    validation_families = {row["scenario_family"] for row in validation}
    selected = {
        "train": [
            row
            for row in train
            if row["scenario_family"] not in test_families | validation_families
        ],
        "validation": [
            row for row in validation if row["scenario_family"] not in test_families
        ],
        "test": test,
    }
    excluded = {
        "train": [
            row["id"]
            for row in train
            if row["scenario_family"] in test_families | validation_families
        ],
        "validation": [
            row["id"] for row in validation if row["scenario_family"] in test_families
        ],
    }
    families = [{row["scenario_family"] for row in rows} for rows in selected.values()]
    ids = [{row["id"] for row in rows} for rows in selected.values()]
    if any(
        families[i] & families[j] or ids[i] & ids[j]
        for i in range(3)
        for j in range(i + 1, 3)
    ):
        raise ValueError("Selected splits still overlap")
    return selected, excluded


def scenario_family(scenario: str) -> str:
    """Group the same dialogue topic across interaction styles."""
    parts = scenario.split("/", 1)
    if len(parts) != 2 or not parts[1]:
        raise ValueError(f"Invalid scenario: {scenario}")
    return parts[1]


def _whole_groups(
    groups: dict[str, list[dict[str, Any]]], target_seconds: int, seed: int
) -> set[str]:
    """Select complete scenario families nearest a duration target."""
    order = sorted(groups, key=lambda key: _rank({"id": key}, seed))
    durations = {
        key: round(sum(int(row["duration_samples"]) for row in groups[key]) / 24_000)
        for key in order
    }
    largest = max(durations.values())
    reachable: dict[int, tuple[str, ...]] = {0: ()}
    for key in order:
        for seconds, chosen in list(reachable.items()):
            total = seconds + durations[key]
            if total <= target_seconds + largest and total not in reachable:
                reachable[total] = (*chosen, key)
    candidates = [
        (seconds, chosen)
        for seconds, chosen in reachable.items()
        if chosen and sum(len(groups[key]) for key in chosen) >= 5
    ]
    if not candidates:
        raise ValueError("No whole-family selection has five conversations")
    _, selected = min(
        candidates,
        key=lambda item: (
            abs(
                sum(
                    int(row["duration_samples"])
                    for key in item[1]
                    for row in groups[key]
                )
                / 24_000
                - target_seconds
            ),
            abs(item[0] - target_seconds),
            -sum(len(groups[key]) for key in item[1]),
            item[1],
        ),
    )
    return set(selected)


def whole_dialogue_generalization_split(
    inventory: dict[str, Any],
    seed: int = 42,
    prior_exposed_families: set[str] | None = None,
) -> dict[str, Any]:
    """Choose 100/10/10 minutes without splitting conversations or topics."""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in inventory["rows"]:
        if int(row["sample_rate"]) != 24_000:
            raise ValueError("Expected 24 kHz source audio")
        groups[scenario_family(row["scenario"])].append(row)
    prior = prior_exposed_families or set()
    available = dict(groups)
    result: dict[str, list[dict[str, Any]]] = {}
    for split, seconds in (("test", 600), ("validation", 600), ("train", 6000)):
        candidates = {
            key: [row for row in value if row["split"] == split]
            for key, value in available.items()
            if split == "train" or key not in prior
        }
        candidates = {key: value for key, value in candidates.items() if value}
        selected = _whole_groups(candidates, seconds, seed + len(result))
        result[split] = sorted(
            (
                {
                    "id": row["id"],
                    "source_split": row["split"],
                    "scenario": row["scenario"],
                    "scenario_family": key,
                    "source_samples": int(row["duration_samples"]),
                    "codec_frames": (int(row["duration_samples"]) + FRAME_SAMPLES - 1)
                    // FRAME_SAMPLES,
                }
                for key in selected
                for row in candidates[key]
            ),
            key=lambda row: (row["scenario_family"], row["id"], row["source_split"]),
        )
        for key in selected:
            del available[key]
    return {
        "seed": seed,
        "dataset": inventory["dataset"],
        "dataset_revision": inventory["revision"],
        "manifest_sha256": inventory["manifest_sha256"],
        "grouping_rule": "scenario topic after the first slash",
        "prior_exposed_families": sorted(prior),
        "selection": result,
    }


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
