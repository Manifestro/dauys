"""Gate A bands, early abort, bigram reference and leakage checks."""

import math
import sys
import types
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from dauys.config import ModelConfig
from dauys.model import DauysHybrid
from dauys.v02_data import sha256
from dauys.v02_stage1 import Stage1Record, ab_layout
from dauys.v02_stage1_eval import (
    Cb0Bigram,
    assert_speakers_disjoint,
    chapter_control,
    early_abort,
    empty_rate,
    fusion_dependence,
    gate_a_bands,
    intent_selection,
    speaker_probe,
    speech_cb0_runs,
    speech_frame_scores,
)
from dauys.v03_intent import IntentConfig, IntentModule


def _record(identity: str, speaker: str) -> Stage1Record:
    return Stage1Record(
        id=identity,
        speaker=speaker,
        frames=100,
        cut_frame=50,
        token_path=Path(f"/nonexistent/{identity}.pt"),
        token_sha256="0" * 64,
    )


def _go_metrics() -> dict[str, float]:
    return {
        "m1_minus_m2": -0.6,
        "m3": 0.75,
        "m4": 0.4,
        "m5": 0.2,
        "m6": 5.0,
        "m8": 4.0,
    }


def test_gate_a_all_go_when_every_band_is_go():
    gate = gate_a_bands(_go_metrics())
    assert gate.outcome == "GO"
    assert set(gate.bands.values()) == {"GO"}


def test_gate_a_fix_band_when_one_metric_is_in_between():
    gate = gate_a_bands({**_go_metrics(), "m1_minus_m2": -0.3})
    assert gate.bands["m1_minus_m2"] == "FIX"
    assert gate.outcome == "FIX"


def test_gate_a_stop_on_any_fail():
    assert gate_a_bands({**_go_metrics(), "m3": 0.2}).outcome == "STOP"
    assert gate_a_bands({**_go_metrics(), "m5": 0.7}).outcome == "STOP"
    assert gate_a_bands({**_go_metrics(), "m8": 0.0}).outcome == "STOP"


def test_gate_a_boundaries_follow_the_document():
    assert (
        gate_a_bands({**_go_metrics(), "m1_minus_m2": -0.5}).bands["m1_minus_m2"]
        == "GO"
    )
    assert (
        gate_a_bands({**_go_metrics(), "m1_minus_m2": -0.2}).bands["m1_minus_m2"]
        == "FIX"
    )
    assert (
        gate_a_bands({**_go_metrics(), "m1_minus_m2": -0.19}).bands["m1_minus_m2"]
        == "FAIL"
    )


def test_gate_a_requires_every_metric():
    metrics = _go_metrics()
    del metrics["m6"]
    with pytest.raises(ValueError, match="missing"):
        gate_a_bands(metrics)


def test_early_abort_rules_fire_at_the_pre_registered_steps():
    assert early_abort(4000, {"m1_minus_m2": 0.1}) == "m1_not_below_bigram"
    assert early_abort(4000, {"m1_minus_m2": -0.1}) is None
    assert early_abort(8000, {"m1_minus_m2": -0.6, "m3": 0.2}) == "intent_near_chance"
    assert (
        early_abort(7000, {"m1_minus_m2": -0.6, "m3": 0.8, "m4": 0.005})
        == "fusion_ignored"
    )
    assert early_abort(6000, {"m1_minus_m2": -0.6, "m3": 0.8, "m4": 0.0}) is None
    assert early_abort(1, {"m1_minus_m2": float("nan")}) == "non_finite_metric"
    assert (
        early_abort(2000, {"m1_minus_m2": -0.6, "intent_spread": 0.01})
        == "intent_collapse"
    )
    assert early_abort(50, {"m1_minus_m2": -0.6, "intent_spread": 0.01}) is None


def test_bigram_prefers_a_seen_transition_over_an_unseen_one():
    bigram = Cb0Bigram.fit([[1, 2, 3, 2, 3]])
    seen = bigram.nll([[2, 3]])
    unseen = bigram.nll([[9, 9]])
    assert seen < unseen
    with pytest.raises(ValueError, match="transition"):
        bigram.nll([[]])


def test_leakage_check_rejects_shared_speakers():
    training = [_record("a", "s1"), _record("b", "s2")]
    heldout = [_record("c", "s3")]
    assert_speakers_disjoint(training, heldout)
    with pytest.raises(ValueError, match="leak"):
        assert_speakers_disjoint(training, [_record("d", "s2")])


def test_empty_rate_uses_the_energy_gate_and_optional_asr():
    silent = np.zeros(24_000, dtype=np.float32)
    loud = np.full(24_000, 0.1, dtype=np.float32)
    assert empty_rate([silent, loud], None) == pytest.approx(0.5)

    def texts(waves):
        return ["", "hello there"]

    assert empty_rate([loud, loud], texts) == pytest.approx(0.5)
    with pytest.raises(ValueError, match="waves"):
        empty_rate([], None)


def test_empty_rate_threshold_is_reachable():
    from dauys.v02_stage1_eval import EMPTY_RMS

    wave = np.full(24_000, EMPTY_RMS / 2, dtype=np.float32)
    assert empty_rate([wave], None) == 1.0


class _CausalMamba(nn.Module):
    def __init__(self, d_model, layer_idx=None, **_kwargs):
        super().__init__()
        self.layer_idx = layer_idx
        self.projection = nn.Linear(d_model, d_model)

    def forward(self, x, inference_params=None):
        return self.projection(x)


@pytest.fixture(autouse=True)
def stand_in_mamba():
    module = types.ModuleType("mamba_ssm")
    module.Mamba2 = _CausalMamba
    old = sys.modules.get("mamba_ssm")
    sys.modules["mamba_ssm"] = module
    try:
        yield
    finally:
        if old is None:
            del sys.modules["mamba_ssm"]
        else:
            sys.modules["mamba_ssm"] = old


def _tiny_config():
    return replace(
        ModelConfig.v02_proxy(),
        embedding_dim=64,
        listening_dim=64,
        listening_layers=1,
        dialogue_dim=64,
        dialogue_blocks=("mamba2", "transformer_cross"),
        attention_heads=4,
        ffn_dim=128,
        depth_dim=64,
        depth_layers=1,
        depth_heads=4,
        dropout=0.0,
        attention_dropout=0.0,
        activity_ahead_horizon=0,
        max_trainable_parameters=10_000_000,
    )


def _token_record(tmp_path, identity, speaker, frames=128, cut=64):
    codes = torch.randint(0, 2048, (8, frames))
    activity = torch.ones(frames, dtype=torch.long)
    activity[cut : cut + 3] = 0
    layout = ab_layout(codes, activity, cut, torch.zeros(8, dtype=torch.long))
    path = tmp_path / f"{identity}.pt"
    torch.save(
        {
            "user": layout.user.short(),
            "target": layout.target.short(),
            "user_activity": layout.user_activity.to(torch.uint8),
            "activity": layout.target_activity.to(torch.uint8),
        },
        path,
    )
    return Stage1Record(
        id=identity,
        speaker=speaker,
        frames=frames,
        cut_frame=cut,
        token_path=path,
        token_sha256=sha256(path),
    )


def test_gate_metrics_execute_on_a_tiny_model(tmp_path):
    torch.manual_seed(0)
    records = [
        _token_record(tmp_path, f"{speaker}-{index}", speaker)
        for speaker in ("s1", "s2", "s3")
        for index in range(3)
    ]
    model = DauysHybrid(_tiny_config()).eval()
    silence = torch.zeros(8, dtype=torch.long)
    scores = speech_frame_scores(model, records, silence, crop_frames=64)
    assert scores["frames"] > 0 and math.isfinite(scores["nll"])
    dependency = fusion_dependence(model, records, silence, crop_frames=64)
    assert math.isfinite(dependency)
    probe = speaker_probe(model, records, steps=5, crop_frames=16)
    assert 0.0 <= probe <= 1.0


def test_gate_a_empty_and_gap_boundaries_are_half_open():
    assert gate_a_bands({**_go_metrics(), "m5": 0.29}).bands["m5"] == "GO"
    assert gate_a_bands({**_go_metrics(), "m5": 0.30}).bands["m5"] == "FIX"
    assert gate_a_bands({**_go_metrics(), "m5": 0.50}).bands["m5"] == "FIX"
    assert gate_a_bands({**_go_metrics(), "m5": 0.51}).bands["m5"] == "FAIL"
    assert gate_a_bands({**_go_metrics(), "m6": 9.9}).bands["m6"] == "GO"
    assert gate_a_bands({**_go_metrics(), "m6": 10.0}).bands["m6"] == "FIX"
    assert gate_a_bands({**_go_metrics(), "m6": 15.0}).bands["m6"] == "FIX"
    assert gate_a_bands({**_go_metrics(), "m6": 15.1}).bands["m6"] == "FAIL"


def test_intent_selection_and_chapter_control_run(tmp_path):
    torch.manual_seed(0)
    records = [_token_record(tmp_path, f"s1-{index}", "s1") for index in range(8)]
    model = DauysHybrid(_tiny_config()).eval()
    intent = IntentModule(model, IntentConfig(horizon=2))
    silence = torch.zeros(8, dtype=torch.long)
    score = intent_selection(model, intent, records, silence, groups=1, candidates=8)
    control = chapter_control(model, intent, records, silence, groups=1, candidates=8)
    assert 0.0 <= score <= 1.0
    assert 0.0 <= control <= 1.0


def test_speech_cb0_runs_split_at_pauses(tmp_path):
    frames = 6
    codes = torch.zeros((8, frames), dtype=torch.short)
    codes[0] = torch.tensor([10, 11, 0, 12, 13, 14])
    path = tmp_path / "runs.pt"
    torch.save(
        {
            "user": codes.clone(),
            "target": codes,
            "user_activity": torch.zeros(frames, dtype=torch.uint8),
            "activity": torch.tensor([1, 1, 0, 1, 1, 1], dtype=torch.uint8),
        },
        path,
    )
    record = Stage1Record(
        id="runs",
        speaker="s1",
        frames=frames,
        cut_frame=3,
        token_path=path,
        token_sha256=sha256(path),
    )
    runs = speech_cb0_runs([record])
    assert [run.tolist() for run in runs] == [[10, 11], [12, 13, 14]]
