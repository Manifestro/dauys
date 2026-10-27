"""Local checks for post-training ranking and audio-token preference losses."""

import pytest
import torch

from dauys.v02_feedback import (
    RewardWeights,
    audio_dpo_loss,
    best_of_n,
    clipped_policy_loss,
    group_relative_advantages,
    sequence_log_probability,
)
from dauys.v02_runner import TrainingPlan


def _report(relevance: float) -> dict[str, float]:
    return {
        "onset_human_range": 1.0,
        "yield_within_500ms": 1.0,
        "backchannel_appropriate": 1.0,
        "relevance_0_to_3": relevance,
        "task_success": 1.0,
        "false_starts": 0.0,
        "talk_over_frames": 0.0,
        "response_opportunities": 3.0,
        "duration_frames": 100.0,
    }


def test_best_of_n_selects_per_context_and_requires_complete_report():
    winners = best_of_n(
        [
            ("a", "first", _report(1.0)),
            ("b", "other", _report(2.0)),
            ("a", "second", _report(3.0)),
        ],
        RewardWeights(),
    )
    assert winners == {"a": "second", "b": "other"}
    with pytest.raises(ValueError, match="Missing reward"):
        best_of_n([("a", "broken", {})], RewardWeights())


def test_audio_dpo_rejects_different_contexts_and_prefers_chosen():
    chosen = torch.tensor([-3.0], requires_grad=True)
    rejected = torch.tensor([-5.0], requires_grad=True)
    reference = torch.tensor([-4.0])
    loss = audio_dpo_loss(
        chosen, rejected, reference, reference, ("same",), ("same",)
    )
    loss.backward()
    assert chosen.grad is not None and chosen.grad.item() < 0
    assert rejected.grad is not None and rejected.grad.item() > 0
    with pytest.raises(ValueError, match="identical contexts"):
        audio_dpo_loss(chosen, rejected, reference, reference, ("a",), ("b",))


def test_sequence_mask_and_clipped_frame_policy():
    logits = torch.zeros((1, 3, 2), requires_grad=True)
    score = sequence_log_probability(
        logits, torch.tensor([[0, 1, 0]]), torch.tensor([[True, True, False]])
    )
    assert torch.allclose(score, torch.tensor([-2 * torch.log(torch.tensor(2.0))]))
    loss = clipped_policy_loss(
        torch.tensor([[0.0, 10.0]], requires_grad=True),
        torch.zeros((1, 2)),
        torch.ones((1, 2)),
        torch.tensor([[True, False]]),
    )
    assert loss.item() == -1.0


def test_group_relative_advantages_are_context_local():
    result = group_relative_advantages(
        torch.tensor([1.0, 3.0, 10.0, 12.0]), ("a", "a", "b", "b")
    )
    assert result.tolist() == [-1.0, 1.0, -1.0, 1.0]


def test_targeted_sft_plan_caps_steps_and_learning_rate():
    values = {
        "track": "SFT",
        "stage": "posttrain",
        "steps": 2000,
        "batch_size": 2,
        "learning_rate": 2e-5,
        "checkpoint_interval": 100,
        "stage2_peak_learning_rate": 2e-4,
    }
    TrainingPlan(**values).validate()
    with pytest.raises(ValueError, match="3,000"):
        TrainingPlan(**{**values, "steps": 3001}).validate()
    with pytest.raises(ValueError, match="0.1x"):
        TrainingPlan(**{**values, "learning_rate": 3e-5}).validate()


def test_full_pretraining_track_is_available():
    TrainingPlan(
        track="PRE",
        stage="pretrain",
        steps=100,
        batch_size=2,
        learning_rate=2e-4,
        checkpoint_interval=10,
    ).validate()
