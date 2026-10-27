"""Frozen-suite metrics and checkpoint selection for Hybrid v0.2."""

import json
import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .v02_data import sha256


@dataclass(frozen=True)
class Suite:
    name: str
    version: str
    items: tuple[Mapping[str, Any], ...]
    manifest_sha256: str


def load_suite(path: str | Path, expected_sha256: str) -> Suite:
    source = Path(path)
    actual_sha256 = sha256(source)
    if actual_sha256 != expected_sha256:
        raise ValueError("Evaluation suite checksum mismatch")
    data = json.loads(source.read_text(encoding="utf-8"))
    if data.get("partial", False):
        raise ValueError(
            "Partial Hohere benchmark cannot be used for research evaluation"
        )
    if data["name"] not in {f"E{index}" for index in range(1, 7)}:
        raise ValueError("Unknown evaluation suite")
    items = tuple(data["items"])
    if not items or len({item["id"] for item in items}) != len(items):
        raise ValueError("Evaluation suite IDs must be unique")
    for item in items:
        for asset in item.get("audio", {}).values():
            candidate = (source.parent / asset["path"]).resolve()
            if not candidate.is_relative_to(source.parent.resolve()):
                raise ValueError("Evaluation audio escapes its suite root")
            if sha256(candidate) != asset["sha256"]:
                raise ValueError(f"Evaluation audio checksum mismatch: {item['id']}")
    return Suite(data["name"], data["version"], items, actual_sha256)


def crossed_minimal_pair_accuracy(
    nll: torch.Tensor, seed: int = 42, bootstrap_samples: int = 2000
) -> dict[str, float]:
    """Input rows and answer columns are paired in a [pairs, 2, 2] NLL matrix."""
    if nll.ndim != 3 or nll.shape[1:] != (2, 2) or nll.shape[0] < 1:
        raise ValueError("Minimal-pair NLL must have shape [pairs, 2, 2]")
    if bootstrap_samples < 1 or not torch.isfinite(nll).all():
        raise ValueError("Invalid minimal-pair scores or bootstrap count")
    first = nll[:, 0, 0] < nll[:, 0, 1]
    second = nll[:, 1, 1] < nll[:, 1, 0]
    correct = (nll[:, 0, 0] + nll[:, 1, 1] < nll[:, 0, 1] + nll[:, 1, 0]).float()
    rng = random.Random(seed)
    population = correct.tolist()
    bootstrap = sorted(
        sum(population[rng.randrange(len(population))] for _ in population)
        / len(population)
        for _ in range(bootstrap_samples)
    )
    low = bootstrap[int(0.025 * (bootstrap_samples - 1))]
    high = bootstrap[int(0.975 * (bootstrap_samples - 1))]
    return {
        "accuracy": float(correct.mean()),
        "both_directions_accuracy": float((first & second).float().mean()),
        "ci95_low": low,
        "ci95_high": high,
        "pairs": float(len(population)),
    }


def conditional_delta(
    original_nll: Sequence[float], alternate_nll: Sequence[float]
) -> float:
    if not original_nll or len(original_nll) != len(alternate_nll):
        raise ValueError("Conditioning scores must have matched nonempty samples")
    return sum(
        alternate - original for original, alternate in zip(original_nll, alternate_nll)
    ) / len(original_nll)


def bootstrap_mean_ci(
    values: Sequence[float], seed: int = 42, samples: int = 2000
) -> tuple[float, float]:
    if not values or samples < 1 or not all(math.isfinite(value) for value in values):
        raise ValueError("Bootstrap requires finite nonempty measurements")
    rng = random.Random(seed)
    size = len(values)
    estimates = sorted(
        sum(values[rng.randrange(size)] for _ in range(size)) / size
        for _ in range(samples)
    )
    return (
        estimates[int(0.025 * (samples - 1))],
        estimates[int(0.975 * (samples - 1))],
    )


def word_error_rate(reference: str, hypothesis: str) -> float:
    reference_words = reference.lower().split()
    hypothesis_words = hypothesis.lower().split()
    if not reference_words:
        raise ValueError("Reference transcript cannot be empty")
    previous = list(range(len(hypothesis_words) + 1))
    for row, word in enumerate(reference_words, 1):
        current = [row]
        for column, predicted in enumerate(hypothesis_words, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (word != predicted),
                )
            )
        previous = current
    return previous[-1] / len(reference_words)


def yield_within(
    activity: torch.Tensor, interrupt_frames: Sequence[int], maximum_frames: int = 6
) -> float:
    if activity.ndim != 1 or not interrupt_frames:
        raise ValueError("Expected one activity trace and interruption events")
    if maximum_frames < 1:
        raise ValueError("Yield window must be positive")
    successes = 0
    for frame in interrupt_frames:
        if not 0 <= frame < activity.numel():
            raise ValueError("Interruption frame is outside the trace")
        end = min(activity.numel(), frame + maximum_frames + 1)
        successes += int((~activity[frame:end].bool()).any())
    return successes / len(interrupt_frames)


@dataclass(frozen=True)
class CheckpointScores:
    checkpoint: str
    e1_cb0_nll: float
    e2_onset_latency_ms: float
    e5_yield_within_500ms: float


def select_checkpoint(
    candidates: Sequence[CheckpointScores], baseline: CheckpointScores
) -> CheckpointScores:
    """Choose CB0 NLL subject to at most 10% onset/yield regression."""
    eligible = [
        item
        for item in candidates
        if item.e2_onset_latency_ms <= baseline.e2_onset_latency_ms * 1.1
        and item.e5_yield_within_500ms >= baseline.e5_yield_within_500ms * 0.9
    ]
    if not eligible:
        raise ValueError("No checkpoint passes the timing guardrails")
    return min(eligible, key=lambda item: item.e1_cb0_nll)
