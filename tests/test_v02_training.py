"""Local checks for stage routing, transition windows and decoding decisions."""

import torch
from torch import nn

from dauys.decoding import (
    ActivityState,
    DecodingConfig,
    HysteresisScore,
    activity_decision,
    cb0_decision,
    choose_hysteresis,
    normalize_step_activity,
)
from dauys.v02_training import (
    Event,
    V02Batch,
    _contrast_negatives,
    history_span_dropout,
    masked_mean,
    optimizer_groups,
    response_onset_mask,
    schedule,
    source_flags,
    transition_mask,
)


def test_pretraining_masks_exclude_empty_channels():
    flags = source_flags(
        ("1a_on_A", "1a_on_B", "1b", "hohere"), "pretrain", torch.device("cpu")
    )
    assert flags["b"].tolist() == [False, True, True, True]
    assert flags["listen"].tolist() == [True, False, True, True]
    assert flags["contrast"].tolist() == [False, False, False, False]
    assert flags["detach_listening"].tolist() == [False, True, False, False]
    assert flags["acoustic"].tolist() == [True, False, True, True]
    flags = source_flags(("hohere",), "dialogue", torch.device("cpu"))
    assert flags["contrast"].item()


def test_masked_mean_ignores_disabled_items():
    values = torch.tensor([[2.0, 4.0], [1000.0, 1000.0]], requires_grad=True)
    mask = torch.tensor([[True, True], [False, False]])
    result = masked_mean(values, mask)
    assert result.item() == 3.0
    result.backward()
    assert values.grad is not None
    assert values.grad[1].abs().sum().item() == 0.0


def test_history_dropout_preserves_targets_and_disabled_items():
    target = torch.arange(2 * 8 * 40).reshape(2, 8, 40)
    original = target.clone()
    silence = torch.full((2, 8), 2047)
    valid = torch.ones((2, 40), dtype=torch.bool)
    enabled = torch.tensor([True, False])
    altered = history_span_dropout(
        target,
        silence,
        valid,
        enabled,
        0.25,
        training=True,
        generator=torch.Generator().manual_seed(7),
    )
    assert torch.equal(target, original)
    assert (altered[0] == 2047).any()
    assert torch.equal(altered[1], original[1])
    assert history_span_dropout(target, silence, valid, enabled, 0.25, False) is target


def test_transition_windows_and_explicit_response_onset():
    events = (
        (
            Event("a_turn_end", "A", 10, 11),
            Event("a_interrupts_b", "A", 30, 31),
            Event("backchannel", "B", 40, 43),
            Event("response_onset", "B", 12, 13),
        ),
        (),
    )
    mask = transition_mask(events, 50, torch.device("cpu"))
    assert mask[0, 4:17].all()
    assert mask[0, 24:37].all()
    assert mask[0, 37:43].all()
    assert not mask[1].any()
    activity = torch.zeros((2, 50), dtype=torch.long)
    activity[1, 20:22] = 1
    onset = response_onset_mask(events, activity)
    assert not onset.any()


def test_schedule_has_no_stage_one_contrast():
    assert schedule(0, "pretrain").contrast == 0
    assert schedule(0, "dialogue").contrast == 0.05
    assert schedule(0.4, "dialogue").transition == 3.0
    assert schedule(1.0, "dialogue").history_dropout == 0.25


def test_hysteresis_holds_state_between_thresholds_and_argmax_reference():
    config = DecodingConfig(activity_mode="hysteresis", theta_on=0.5, theta_off=0.2)
    state = ActivityState.silent(1, torch.device("cpu"))
    probabilities = (0.1, 0.6, 0.3, 0.4, 0.1)
    decisions = []
    for probability in probabilities:
        logits = torch.tensor([[1 - probability, probability]]).log()
        decisions.append(activity_decision(logits, config, state).item())
    assert decisions == [False, True, True, True, False]
    reference = DecodingConfig(activity_mode="hysteresis", theta_on=0.5, theta_off=0.5)
    state = ActivityState.silent(3, torch.device("cpu"))
    logits = torch.tensor([[0.0, 0.0], [0.1, 0.2], [0.3, 0.1]])
    assert torch.equal(
        activity_decision(logits, reference, state), logits.argmax(-1).bool()
    )


def test_step_activity_normalizes_legacy_and_configured_outputs():
    logits = torch.tensor([[0.1, 0.9], [0.9, 0.1]])
    decision = torch.tensor([True, False])
    assert torch.equal(normalize_step_activity(logits), decision)
    assert torch.equal(normalize_step_activity(decision), decision)


def test_activity_gated_cb0_does_not_sample_silent_rows():
    config = DecodingConfig(cb0_mode="activity_gated", cb0_top_k=2)
    logits = torch.tensor([[0.0, 100.0, 99.0], [100.0, 0.0, 0.0]])
    generator = torch.Generator().manual_seed(9)
    before = generator.get_state().clone()
    result = cb0_decision(logits, torch.zeros(2, dtype=torch.bool), config, generator)
    assert result.tolist() == [1, 0]
    assert torch.equal(before, generator.get_state())


def test_hysteresis_selection_respects_yield_and_false_start_gates():
    scores = [
        HysteresisScore(0.25, 0.1, 200, 0.7, 1.0),
        HysteresisScore(0.35, 0.2, 350, 0.85, 1.0),
        HysteresisScore(0.5, 0.3, 450, 0.9, 0.5),
    ]
    assert choose_hysteresis(scores, argmax_false_starts=1.0) == scores[1]


def test_weight_decay_excludes_embeddings_and_norms():
    model = nn.Sequential(nn.Embedding(12, 4), nn.Linear(4, 4), nn.RMSNorm(4))
    groups = optimizer_groups(model)
    decayed = {id(item) for item in groups[0]["params"]}
    assert id(model[1].weight) in decayed
    assert id(model[0].weight) not in decayed
    assert id(model[2].weight) not in decayed
    assert id(model[1].bias) not in decayed


def test_contrast_uses_another_conversation_when_matched_negative_missing():
    user = torch.stack((torch.ones((8, 3), dtype=torch.long), torch.full((8, 3), 2)))
    batch = V02Batch(
        user_codes=user,
        output_codes=user.clone(),
        user_activity=torch.ones((2, 3), dtype=torch.long),
        output_activity=torch.ones((2, 3), dtype=torch.long),
        valid_frames=torch.ones((2, 3), dtype=torch.bool),
        initial_output_codes=torch.zeros((2, 8), dtype=torch.long),
        silence_codes=torch.zeros((2, 8), dtype=torch.long),
        sources=("hohere", "hohere"),
        conversation_ids=("a", "b"),
        events=((), ()),
        negative_user_codes=user.clone(),
        negative_ids=("a", "b"),
    )
    negative, enabled = _contrast_negatives(batch, torch.ones(2, dtype=torch.bool))
    assert enabled.all()
    assert torch.equal(negative[0], user[1])
    assert torch.equal(negative[1], user[0])


def test_fallback_negatives_rotate_across_distinct_conversations():
    user = torch.arange(4).reshape(4, 1, 1).expand(4, 8, 3)
    batch = V02Batch(
        user_codes=user,
        output_codes=user.clone(),
        user_activity=torch.ones((4, 3), dtype=torch.long),
        output_activity=torch.ones((4, 3), dtype=torch.long),
        valid_frames=torch.ones((4, 3), dtype=torch.bool),
        initial_output_codes=torch.zeros((4, 8), dtype=torch.long),
        silence_codes=torch.zeros((4, 8), dtype=torch.long),
        sources=("hohere",) * 4,
        conversation_ids=("a", "b", "c", "d"),
        events=((),) * 4,
    )
    negatives, enabled = _contrast_negatives(batch, torch.ones(4, dtype=torch.bool))
    assert enabled.all()
    assert negatives[:, 0, 0].tolist() == [1, 2, 3, 0]


def test_overlap_response_onset_requires_an_explicit_response_event():
    output_activity = torch.tensor([[0, 0, 1, 1, 0, 0]])
    events = ((Event("response_onset", "B", 2, 3),),)
    mask = response_onset_mask(events, output_activity)
    assert mask[0].tolist() == [False, False, True, True, False, False]


def test_backchannel_and_b_interruption_cannot_create_c3_window():
    activity = torch.zeros((1, 50), dtype=torch.long)
    activity[0, 15:18] = 1
    backchannel = ((Event("backchannel", "B", 15, 18),),)
    assert not response_onset_mask(backchannel, activity).any()
    conflicting = ((
        Event("backchannel", "B", 15, 18),
        Event("response_onset", "B", 15, 16),
    ),)
    assert not response_onset_mask(conflicting, activity).any()
    interruption = ((
        Event("b_interrupts_a", "B", 15, 16),
        Event("response_onset", "B", 15, 16),
    ),)
    assert not response_onset_mask(interruption, activity).any()
