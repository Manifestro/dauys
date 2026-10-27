"""Frozen-suite statistics and checkpoint selection checks."""

import json

import pytest
import torch

from dauys.v02_data import sha256
from dauys.v02_evaluation import (
    CheckpointScores,
    conditional_delta,
    crossed_minimal_pair_accuracy,
    load_suite,
    select_checkpoint,
    word_error_rate,
    yield_within,
)


def test_crossed_pairs_require_both_directions():
    scores = torch.tensor(
        [
            [[0.1, 0.7], [0.7, 0.1]],
            [[0.1, 0.7], [0.1, 0.7]],
        ]
    )
    result = crossed_minimal_pair_accuracy(scores, bootstrap_samples=100)
    assert result["accuracy"] == 0.75
    assert result["both_directions_accuracy"] == 0.5
    assert result["ci95_low"] <= result["accuracy"] <= result["ci95_high"]


def test_minimal_pair_accuracy_has_fifty_percent_chance_reference():
    scores = torch.tensor([[[0.1, 0.2], [0.1, 0.2]]])
    result = crossed_minimal_pair_accuracy(scores, bootstrap_samples=10)
    assert result["accuracy"] == 0.5
    assert result["both_directions_accuracy"] == 0.0


def test_conditioning_wer_yield_and_checkpoint_guardrails():
    assert conditional_delta([1.0, 2.0], [1.5, 2.5]) == 0.5
    assert word_error_rate("red tomato", "red carrot") == 0.5
    activity = torch.tensor([1, 1, 1, 0, 0], dtype=torch.bool)
    assert yield_within(activity, [1], maximum_frames=2) == 1.0
    baseline = CheckpointScores("baseline", 2.0, 400, 0.8)
    candidates = [
        CheckpointScores("better_nll_but_slow", 1.0, 600, 0.9),
        CheckpointScores("eligible", 1.2, 420, 0.76),
    ]
    assert select_checkpoint(candidates, baseline).checkpoint == "eligible"


def test_suite_manifest_checksum_and_unique_ids(tmp_path):
    path = tmp_path / "E3.json"
    path.write_text(json.dumps({"name": "E3", "version": "1", "items": [{"id": "a"}]}))
    suite = load_suite(path, sha256(path))
    assert suite.name == "E3"
    with pytest.raises(ValueError, match="checksum"):
        load_suite(path, "0" * 64)
