"""CPU checks for the optional Stage 1 stateful stream side batch."""

import sys
import types
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from v02_stage1_train import optimize_step

from dauys.config import ModelConfig
from dauys.model import DauysHybrid, MambaBlock
from dauys.v02_stage1 import (
    INTENT_FRAMES,
    Stage1Batch,
    Stage1Weights,
    stage1_objective,
)
from dauys.v02_stateful import StreamCursor
from dauys.v03_intent import IntentConfig, IntentModule


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


def _tiny_config() -> ModelConfig:
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


def _batch(frames: int = 8) -> Stage1Batch:
    size = 2
    return Stage1Batch(
        user_codes=torch.randint(0, 2048, (size, 8, frames)),
        output_codes=torch.randint(1, 2048, (size, 8, frames)),
        user_activity=torch.ones((size, frames), dtype=torch.long),
        output_activity=torch.ones((size, frames), dtype=torch.long),
        valid_frames=torch.ones((size, frames), dtype=torch.bool),
        initial_output_codes=torch.zeros((size, 8), dtype=torch.long),
        silence_codes=torch.zeros((size, 8), dtype=torch.long),
        intent_frames=torch.tensor([1, 3]),
        candidate_codes=torch.randint(0, 2048, (size, 2, 8, INTENT_FRAMES)),
        candidate_activity=torch.ones((size, 2, INTENT_FRAMES), dtype=torch.bool),
        positive_index=torch.tensor([[0], [1]]),
        speaker_index=torch.arange(size),
        ids=("b0", "b1"),
        speakers=("s0", "s1"),
        candidate_ids=(("c0-0", "c0-1"), ("c1-0", "c1-1")),
    )


def _streams(frames: int = 16) -> list[dict]:
    return [
        {
            "user": torch.randint(0, 2048, (8, frames)),
            "target": torch.randint(0, 2048, (8, frames)),
            "user_activity": torch.ones(frames, dtype=torch.long),
            "activity": torch.ones(frames, dtype=torch.long),
        }
        for _ in range(3)
    ]


def _setup():
    model = DauysHybrid(_tiny_config()).train()
    intent = IntentModule(model, IntentConfig(horizon=2))
    params = list(model.parameters()) + list(intent.head.parameters())
    optimizer = torch.optim.AdamW(params, lr=1e-3)
    return model, intent, optimizer, params


def test_stream_step_runs_forward_backward_and_updates_parameters(monkeypatch):
    def fake_forward_with_state(self, x, state):
        return x, None

    monkeypatch.setattr(MambaBlock, "forward_with_state", fake_forward_with_state)
    torch.manual_seed(0)
    model, intent, optimizer, params = _setup()
    cursor = StreamCursor(
        _streams(), rows=2, chunk=4, seed=0, device=torch.device("cpu")
    )
    weights = Stage1Weights(acoustic_dropout=0.0, label_smoothing=0.0)
    before = model.dialogue_projection.weight.detach().clone()
    total, losses, stream_loss, stream_cb0, stream_chunk = optimize_step(
        model,
        intent,
        optimizer,
        _batch(),
        weights,
        trainable=params,
        stream_cursor=cursor,
        device=torch.device("cpu"),
    )
    assert torch.isfinite(total)
    assert torch.isfinite(losses.total)
    assert stream_loss is not None and torch.isfinite(stream_loss)
    assert stream_cb0 is not None and torch.isfinite(stream_cb0)
    assert stream_chunk == 0
    assert not torch.equal(model.dialogue_projection.weight, before)
    assert cursor.index == 1
    assert ("ear", 0) in cursor.states


def test_stream_rows_zero_keeps_the_stage1_loss_unchanged():
    torch.manual_seed(0)
    model, intent, optimizer, params = _setup()
    batch = _batch()
    weights = Stage1Weights(acoustic_dropout=0.0, label_smoothing=0.0)
    expected = stage1_objective(model, batch, weights, intent).total.detach()
    model.zero_grad(set_to_none=True)
    total, losses, stream_loss, stream_cb0, stream_chunk = optimize_step(
        model,
        intent,
        optimizer,
        batch,
        weights,
        trainable=params,
        stream_cursor=None,
        device=torch.device("cpu"),
    )
    assert torch.allclose(total.detach(), expected)
    assert torch.allclose(losses.total.detach(), expected)
    assert stream_loss is None
    assert stream_cb0 is None
    assert stream_chunk is None
