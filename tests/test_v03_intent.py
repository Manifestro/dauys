"""Intent objective: future targets, speech mask, EMA ear, gradient flow."""

import sys
import types
from dataclasses import replace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from dauys.config import ModelConfig
from dauys.model import DauysHybrid
from dauys.v03_intent import (
    IntentConfig,
    IntentModule,
    future_targets,
    intent_loss,
    speech_ahead,
)


class _CausalMamba(nn.Module):
    def __init__(self, d_model, layer_idx, **_kwargs):
        super().__init__()
        self.layer_idx = layer_idx
        self.projection = nn.Linear(d_model, d_model)

    def forward(self, x, inference_params=None):
        return self.projection(x).cumsum(dim=1)


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


def _model() -> DauysHybrid:
    torch.manual_seed(0)
    return DauysHybrid(
        replace(ModelConfig.v02_proxy(), dropout=0.0, attention_dropout=0.0)
    )


def test_future_targets_average_the_next_horizon_only():
    torch.manual_seed(1)
    states = torch.randn(2, 9, 4)
    valid = torch.ones(2, 9, dtype=torch.bool)
    valid[1, 6:] = False
    targets, has_future = future_targets(states, valid, horizon=3)
    for b in range(2):
        for t in range(9):
            frames = [s for s in range(t + 1, min(t + 4, 9)) if valid[b, s]]
            assert bool(has_future[b, t]) == bool(frames)
            if frames:
                expected = F.layer_norm(states[b, frames].mean(dim=0), (4,))
                torch.testing.assert_close(targets[b, t], expected)


def test_speech_ahead_looks_strictly_forward():
    activity = torch.tensor([[0, 0, 0, 1, 0, 0, 0, 0]])
    ahead = speech_ahead(activity, horizon=2)
    assert ahead.tolist() == [[False, True, True, False, False, False, False, False]]


def test_loss_ignores_frames_without_speech_ahead():
    prediction = torch.zeros(1, 6, 4)
    target_states = torch.randn(1, 6, 4)
    valid = torch.ones(1, 6, dtype=torch.bool)
    silent = intent_loss(
        prediction, target_states, valid, torch.zeros(1, 6), torch.tensor([True]), 2
    )
    assert int(silent.frames) == 0 and float(silent.loss) == 0
    no_b = intent_loss(
        prediction, target_states, valid, torch.ones(1, 6), torch.tensor([False]), 2
    )
    assert int(no_b.frames) == 0


def test_ema_update_follows_the_decay():
    model = _model()
    intent = IntentModule(model, IntentConfig())
    with torch.no_grad():
        model.listening_projection.weight.add_(1.0)
    before = intent.target.projection.weight.clone()
    intent.target.update(model, decay=1.0)
    assert torch.equal(intent.target.projection.weight, before)
    intent.target.update(model, decay=0.0)
    assert torch.equal(
        intent.target.projection.weight, model.listening_projection.weight
    )
    assert not any(p.requires_grad for p in intent.target.parameters())


def test_target_matches_the_online_ear_at_start():
    model = _model().eval()
    intent = IntentModule(model, IntentConfig())
    codes = torch.randint(0, 2048, (2, 8, 5))
    with torch.no_grad():
        online = model.listening_projection(model._embed(codes, 0))
        for block in model.listening:
            online = block(online)
    torch.testing.assert_close(intent.target(codes), online)


def test_intent_gradient_reaches_dialogue_tower_not_target():
    model = _model()
    intent = IntentModule(model, IntentConfig(horizon=2))
    user = torch.randint(0, 2048, (2, 8, 7))
    output = torch.randint(0, 2048, (2, 8, 7))
    initial = torch.randint(0, 2048, (2, 8))
    states = model(user, output, initial).dialogue_states
    scored = intent.loss(
        states,
        output,
        torch.ones(2, 7, dtype=torch.bool),
        torch.ones(2, 7),
        torch.tensor([True, True]),
    )
    scored.loss.backward()
    assert intent.head.net[0].weight.grad is not None
    assert model.dialogue_projection.weight.grad is not None
    assert all(p.grad is None for p in intent.target.parameters())
    assert float(scored.prediction_std.detach()) > 0


def test_ema_decay_ramps_from_start_to_end():
    config = IntentConfig(ema_start=0.99, ema_end=0.999)
    assert config.ema_decay(0.0) == pytest.approx(0.99)
    assert config.ema_decay(1.0) == pytest.approx(0.999)
    assert 0.99 < config.ema_decay(0.5) < 0.999
