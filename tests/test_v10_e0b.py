"""CPU contract checks for the V10 E0b models, data, substitutions and rules."""

import importlib.util
import sys
import types
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from torch import nn

from dauys.v10_e0b import E0bConfig, E0bModel, flops_per_frame
from dauys.v10_e0b_data import (
    CONDITIONS,
    CONTEXT_FRAMES,
    PAUSE_FRAMES,
    SCORED_FRAMES,
    WINDOW_FRAMES,
    Clip,
    Place,
    build_condition,
    collate_windows,
    make_batch,
    reply_starts,
    silence_tokens,
    splice_timed,
    window_plan,
)
from dauys.v10_e0b_eval import (
    bootstrap_by_episode,
    causality_error,
    check_question_1,
    decide_question_2,
    echo_baseline,
    echo_counts,
    evaluate_split,
    place_nll,
    sample_replies,
)


class _CausalMamba(nn.Module):
    def __init__(self, d_model, **_kwargs):
        super().__init__()
        self.projection = nn.Linear(d_model, d_model)

    def forward(self, x, inference_params=None):
        return self.projection(x).cumsum(dim=1) / 1000


@pytest.fixture(scope="module", autouse=True)
def stand_in_mamba():
    module = types.ModuleType("mamba_ssm")
    module.Mamba2 = _CausalMamba
    old = sys.modules.get("mamba_ssm")
    sys.modules["mamba_ssm"] = module
    yield
    if old is None:
        del sys.modules["mamba_ssm"]
    else:
        sys.modules["mamba_ssm"] = old


def _tiny(scheme: str) -> E0bConfig:
    extra = {"ear_layers": 1, "cross_every": 2} if scheme == "rev1" else {}
    return E0bConfig(
        name=f"tiny-{scheme}",
        scheme=scheme,
        width=32,
        repeats=1,
        attention_heads=2,
        ffn_dim=32,
        attention_window=64,
        depth_dim=16,
        depth_heads=2,
        depth_layers=1,
        **extra,
    )


SILENCE = 5


def _clip(
    frames: int = 900,
    *,
    seed: int = 0,
    split: str = "train",
    episode: str = "e0",
    leak: float = 0.5,
) -> Clip:
    """Two channels taking turns every 40 frames, silent frames on one token."""
    generator = torch.Generator().manual_seed(seed)
    left = torch.randint(10, 2048, (8, frames), generator=generator)
    right = torch.randint(10, 2048, (8, frames), generator=generator)
    active_left = (torch.arange(frames) // 40) % 2 == 0
    active_right = ~active_left
    left[0, ~active_left] = SILENCE
    right[0, ~active_right] = SILENCE
    return Clip(
        file=f"{episode}_{seed}.npz",
        split=split,
        feed="feed",
        episode=episode,
        leak_pct=leak,
        codes=(left, right),
        active=(active_left, active_right),
    )


@pytest.mark.parametrize("scheme", ["rev2", "rev1"])
def test_configs_load_in_the_planned_shape(scheme):
    config = E0bConfig.for_scheme(scheme)
    assert config.scheme == scheme
    assert config.layers == 12
    assert config.attention_window == 1024


def _mamba_parameters(width: int, state: int = 64) -> int:
    """Mamba-2 2.3.1 with this repository's settings, plus the block's norm.

    The same formula gives 155,866,948 for the V10 step 1 model, the count measured
    with the real class.
    """
    inner = 2 * width
    heads = inner // 64
    conv = inner + 2 * state
    return (
        width * (2 * inner + 2 * state + heads)
        + 5 * conv
        + 3 * heads
        + inner
        + inner * width
        + width
    )


def _real_size(config: E0bConfig) -> int:
    """Swap the stand-in's parameters for the count of the real Mamba-2."""
    model = E0bModel(config)
    stand_ins = sum(
        parameter.numel()
        for module in model.modules()
        if isinstance(module, _CausalMamba)
        for parameter in module.parameters()
    )
    blocks = 3 * config.repeats + config.ear_layers
    block_norm = blocks * config.width
    # The formula counts each block's own norm, which the model already holds.
    return (
        model.trainable_parameters()
        - stand_ins
        - block_norm
        + blocks * _mamba_parameters(config.width, config.mamba_d_state)
    )


def test_schemes_match_in_size_and_rough_flops():
    rev2, rev1 = E0bConfig.for_scheme("rev2"), E0bConfig.for_scheme("rev1")
    rev2_size, rev1_size = _real_size(rev2), _real_size(rev1)
    assert 30_000_000 <= rev2_size <= 50_000_000
    assert 30_000_000 <= rev1_size <= 50_000_000
    assert abs(rev1_size / rev2_size - 1) <= 0.05
    assert abs(flops_per_frame(rev1) / flops_per_frame(rev2) - 1) <= 0.05


def test_rev2_has_no_ear_and_rev1_has_its_cross_blocks():
    assert E0bModel(_tiny("rev2")).ear is None
    model = E0bModel(_tiny("rev1"))
    assert model.ear is not None
    assert sorted(model.cross) == ["1", "3"]


@pytest.mark.parametrize("scheme", ["rev2", "rev1"])
def test_future_frames_do_not_reach_any_output(scheme):
    model = E0bModel(_tiny(scheme)).eval()
    clip = _clip()
    assert causality_error(model, [clip], SILENCE, frames=60) < 1e-5


@pytest.mark.parametrize("scheme", ["rev2", "rev1"])
def test_the_partner_reaches_the_next_frame_not_the_same_one(scheme):
    model = E0bModel(_tiny(scheme)).eval()
    own = torch.randint(0, 2048, (1, 20))
    partner = torch.randint(0, 2048, (1, 20))
    active = torch.ones(1, 20, dtype=torch.bool)
    changed = partner.clone()
    changed[0, 10] = (changed[0, 10] + 1) % 2048
    with torch.no_grad():
        base = model(own, partner, active, active).own_logits
        other = model(own, changed, active, active).own_logits
    assert torch.equal(base[0, :11], other[0, :11])
    assert not torch.equal(base[0, 11], other[0, 11])


@pytest.mark.parametrize("scheme", ["rev2", "rev1"])
def test_a_training_step_lowers_the_loss_and_depth_stops_at_the_tower(scheme):
    torch.manual_seed(0)
    model = E0bModel(_tiny(scheme))
    clips = [_clip()]
    plan = window_plan(clips, seed=0, step=1, batch_size=2)
    batch = make_batch(clips, plan, seed=0, step=1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)

    def run():
        output = model(
            batch["own"], batch["partner"], batch["own_active"], batch["partner_active"]
        )
        return output, model.loss(output, batch)

    output, losses = run()
    first = float(losses.total.detach())
    model.zero_grad()
    picks = batch["depth_frames"]
    depth = model.depth_logits(batch["depth_prefix"], output.hidden, picks)
    depth.sum().backward(retain_graph=True)
    assert model.blocks[0].mixer.projection.weight.grad is None
    assert model.depth_embeddings[0].weight.grad is not None
    for _ in range(5):
        optimizer.zero_grad()
        run()[1].total.backward()
        optimizer.step()
    assert float(run()[1].total.detach()) < first


def test_windows_depend_only_on_seed_and_step_and_keep_their_range():
    clips = [_clip(900, seed=index) for index in range(3)]
    first = window_plan(clips, seed=0, step=7, batch_size=8)
    assert first == window_plan(clips, seed=0, step=7, batch_size=8)
    assert first != window_plan(clips, seed=0, step=8, batch_size=8)
    roles = set()
    for step in range(30):
        for window in window_plan(clips, seed=0, step=step, batch_size=8):
            assert WINDOW_FRAMES[0] <= window.frames <= WINDOW_FRAMES[1]
            assert window.start + window.frames <= clips[window.clip].frames
            roles.add(window.role)
    assert roles == {0, 1}


def test_a_batch_swaps_the_channels_by_role():
    clip = _clip()
    plan = window_plan([clip], seed=0, step=1, batch_size=4)
    batch = make_batch([clip], plan, seed=0, step=1)
    for row, window in enumerate(plan):
        span = slice(window.start, window.start + window.frames)
        frames = window.frames
        assert torch.equal(batch["own"][row, :frames], clip.codes[window.role][0, span])
        assert torch.equal(
            batch["partner"][row, :frames], clip.codes[1 - window.role][0, span]
        )
        assert torch.equal(
            batch["depth_target"][row, :frames], clip.codes[window.role][1:, span].T
        )


def test_activity_targets_look_past_the_window_and_stop_at_the_clip_end():
    clip = _clip()
    plan = window_plan([clip], seed=3, step=1, batch_size=1)
    window = plan[0]
    batch = make_batch([clip], plan, seed=3, step=1)
    active = clip.active[window.role]
    frame = 5
    assert batch["activity_target"][0, frame, 0] == active[window.start + frame]
    assert batch["activity_target"][0, frame, 2] == active[window.start + frame + 11]
    edge = _clip(800)
    plan = [type(window)(0, 0, 0, 800)]
    batch = make_batch([edge], plan, seed=0, step=1)
    assert batch["activity_valid"][0, 799, 0]
    assert not batch["activity_valid"][0, 799, 2]
    assert batch["activity_valid"][0, 788, 2]


def test_reply_starts_follow_the_rule_of_the_brief():
    assert PAUSE_FRAMES == 7
    own = torch.zeros(100, dtype=torch.bool)
    partner = torch.zeros(100, dtype=torch.bool)
    own[20:30] = True
    own[40:50] = True  # a pause of ten frames, partner silent: not a start
    partner[52:56] = True
    own[60:70] = True  # the partner spoke in the pause: a start
    own[74:80] = True  # own silent for only four frames: not a start
    assert reply_starts(own, partner) == [60]
    late = torch.zeros(100, dtype=torch.bool)
    late[90:] = True
    assert reply_starts(late, torch.ones(100, dtype=torch.bool)) == []


def test_silence_comes_from_the_data_not_the_codec():
    clip = _clip()
    token, tokens = silence_tokens([clip])
    assert token == SILENCE
    assert SILENCE in tokens


def test_splicing_keeps_the_slots_and_rotates_the_speech():
    codes = torch.tensor([9, 10, 11, 0, 0, 12, 13, 0])
    active = codes > 0
    assert torch.equal(splice_timed(codes, active, active, 0, 0), codes)
    moved = splice_timed(codes, active, active, 0, 2)
    assert torch.equal(moved != 0, active)
    assert moved[active].tolist() == [11, 12, 13, 9, 10]


def test_each_condition_changes_only_its_channel():
    clip = _clip()
    place = Place(0, 0, 400)
    real = build_condition(clip, place, "real", SILENCE)
    t0 = int(real["t0"])
    silence = build_condition(clip, place, "silence", SILENCE)
    assert torch.equal(silence["own"], real["own"])
    assert (silence["partner"] == SILENCE).all() and not silence["partner_active"].any()
    spliced = build_condition(clip, place, "real_spliced", SILENCE)
    swapped = build_condition(clip, place, "same_voice", SILENCE)
    assert torch.equal(spliced["partner"], real["partner"])
    assert torch.equal(swapped["partner_active"], real["partner_active"])
    speech = real["partner_active"]
    assert torch.equal(swapped["own"], real["own"])
    assert (swapped["partner"][speech] != real["partner"][speech]).float().mean() > 0.9
    assert (swapped["partner"][~speech] == SILENCE).all()
    erased = build_condition(clip, place, "own_erased", SILENCE)
    assert (erased["own"][:t0] == SILENCE).all() and not erased["own_active"][:t0].any()
    assert torch.equal(erased["own"][t0:], real["own"][t0:])
    assert torch.equal(erased["partner"], real["partner"])
    shifted = build_condition(clip, place, "shifted", SILENCE)
    assert torch.equal(shifted["partner"][:-1], real["partner"][1:])


def test_windows_hold_the_context_and_scored_frames():
    clip = _clip(1200)
    late = build_condition(clip, Place(0, 0, 1120), "real", SILENCE)
    assert len(late["own"]) == CONTEXT_FRAMES + SCORED_FRAMES
    assert int(late["t0"]) == CONTEXT_FRAMES
    early = build_condition(clip, Place(0, 0, 80), "real", SILENCE)
    assert int(early["t0"]) == 80
    batch = collate_windows([late, early])
    assert batch["own"].shape == (2, CONTEXT_FRAMES + SCORED_FRAMES)


def test_padding_does_not_change_a_score():
    torch.manual_seed(0)
    model = E0bModel(_tiny("rev2")).eval()
    clip = _clip(1200)
    places = [Place(0, 0, 80), Place(0, 0, 1120)]
    together = place_nll(model, [clip], places, "real", SILENCE, batch_size=2)
    alone = place_nll(model, [clip], places, "real", SILENCE, batch_size=1)
    assert torch.allclose(together, alone, atol=1e-4)


def test_echo_counts_ignore_silence_and_find_copies():
    assert echo_counts([1, 2, 3, 4, 5], [9, 1, 2, 3, 4, 9], frozenset()) == (1, 2)
    assert echo_counts([7, 7, 7, 7], [7, 7, 7, 7], frozenset({7})) == (0, 0)


def test_sampling_changes_only_the_scored_frames():
    torch.manual_seed(0)
    model = E0bModel(_tiny("rev1")).eval()
    clip = _clip()
    replies = sample_replies(model, [clip], [Place(0, 0, 400)], SILENCE, seed=0)
    assert len(replies) == 1 and len(replies[0]) == SCORED_FRAMES


def test_samples_keep_the_lead_in_and_replace_the_model_frames():
    path = Path(__file__).parents[1] / "scripts" / "v10_e0b_samples.py"
    spec = importlib.util.spec_from_file_location("v10_e0b_samples", path)
    samples = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(samples)
    torch.manual_seed(0)
    model = E0bModel(_tiny("rev1")).eval()
    clip = _clip()
    place = Place(0, 0, 400)
    generator = torch.Generator().manual_seed(0)
    own, partner = samples.speak(model, clip, place, torch.device("cpu"), generator)
    frames = samples.LEAD_IN_FRAMES + samples.MODEL_FRAMES
    assert own.shape == (8, frames) and partner.shape == (8, frames)
    first = 400 - samples.LEAD_IN_FRAMES
    assert torch.equal(own[:, :38], clip.codes[0][:, first:400])
    assert not torch.equal(own[:, 38:], clip.codes[0][:, 400:550])
    assert torch.equal(partner, clip.codes[1][:, first:550])


def test_evaluation_reports_every_metric_for_both_schemes():
    for scheme in ("rev2", "rev1"):
        torch.manual_seed(0)
        model = E0bModel(_tiny(scheme))
        clip = _clip()
        out = evaluate_split(
            model,
            [clip],
            [Place(0, 0, 400), Place(0, 1, 440)],
            SILENCE,
            frozenset({SILENCE}),
            seed=0,
        )
        assert out["places"] == 2
        for name in ("g_silence", "g_voice", "own_history_gain", "shift_change"):
            assert out[name]["episodes"] == 1
        assert len(out["per_place"]["g_voice"]) == 2
        assert model.training


def test_all_conditions_are_known():
    clip = _clip()
    for condition in CONDITIONS:
        build_condition(clip, Place(0, 0, 400), condition, SILENCE)
    with pytest.raises(ValueError):
        build_condition(clip, Place(0, 0, 400), "other", SILENCE)


def test_echo_baseline_needs_ten_clean_clips():
    clean = [_clip(seed=index, leak=0.5) for index in range(10)]
    places = {"train": [Place(index, 0, 400) for index in range(10)]}
    assert echo_baseline(clean, places, SILENCE, frozenset())["percent"] is not None
    few = [replace(clip, leak_pct=3.0) for clip in clean[:9]] + clean[:1]
    assert echo_baseline(few, places, SILENCE, frozenset())["percent"] is None


def test_bootstrap_resamples_whole_episodes():
    out = bootstrap_by_episode([1.0, 1.0, 3.0], ["a", "a", "b"], reps=200)
    assert out["episodes"] == 2
    assert out["low"] == 1.0 and out["high"] == 3.0


FLAT = [0.0] * 10


@pytest.mark.parametrize(
    ("rev2", "rev1", "history", "outcome"),
    [
        ([0.6] * 10, [0.1] * 9 + [0.6], FLAT, "rev2_first_and_stronger"),
        (
            [0.0] * 3 + [0.6] * 7,
            [0.0, 0.6] + [1.0] * 8,
            FLAT,
            "rev1_earlier_and_stronger",
        ),
        (FLAT, [0.0] * 8 + [0.5, 0.5], FLAT, "rev1_earlier_and_stronger_guard"),
        (FLAT, FLAT, [float(i) for i in range(10)], "neither_detects_history_grows"),
        (FLAT, FLAT, FLAT, "other"),
        ([0.0, 0.6] + [0.7] * 8, [0.6] * 9 + [0.8], FLAT, "other"),
    ],
)
def test_question_2_table(rev2, rev1, history, outcome):
    result = decide_question_2(rev2, rev1, history, history)
    if outcome == "rev1_earlier_and_stronger_guard":
        # rev2 never detected, and 0.5 >= 1.5 * 0: the table sends this to rev1.
        assert result["outcome"] == "rev1_earlier_and_stronger"
        return
    assert result["outcome"] == outcome


def test_question_2_needs_matching_checkpoints():
    with pytest.raises(ValueError):
        decide_question_2([0.1], [0.1, 0.2], [0.1], [0.1, 0.2])


def _q1(**changes):
    values = {
        "train_loss_first": 7.0,
        "train_loss_last": 1.0,
        "echo_percent": 2.0,
        "echo_baseline_percent": 1.5,
        "shift_change": 0.2,
        "causality": 0.0,
        "kept_leak_max_pct": 5.0,
    }
    return check_question_1(**{**values, **changes})


def test_question_1_checks():
    assert _q1()["passed"]
    assert not _q1(train_loss_last=7.5)["passed"]
    assert not _q1(echo_percent=2.6)["passed"]
    assert not _q1(shift_change=-0.1)["passed"]
    assert not _q1(causality=0.1)["passed"]
    assert _q1(echo_baseline_percent=None)["echo_within_baseline"]
    assert not _q1(echo_baseline_percent=None, kept_leak_max_pct=6.0)["passed"]
