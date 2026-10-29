"""Separate frozen ear embeddings for the Stage 1 memory-ear run."""

import math
import sys
import types
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from torch import nn

from dauys.config import ModelConfig
from dauys.model import DauysHybrid, MambaBlock
from dauys.v02_runner import load_initial_weights
from dauys.v03_intent import IntentConfig, IntentModule

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from v02_stage1_train import filter_intent_state, map_ear_state


class _CausalMamba(nn.Module):
    def __init__(self, d_model, layer_idx, **_kwargs):
        super().__init__()
        self.layer_idx = layer_idx
        self.projection = nn.Linear(d_model, d_model)

    def forward(self, x, inference_params=None):
        if inference_params is not None:
            state = inference_params.key_value_memory_dict[self.layer_idx]
            out = self.projection(x).cumsum(dim=1)
            state[0].copy_(out[:, -1:])
            return out
        return self.projection(x).cumsum(dim=1)

    def allocate_inference_cache(self, batch_size, max_seqlen):
        width = self.projection.in_features
        return (torch.zeros(batch_size, 1, width), torch.zeros(1))

    def step(self, x, previous, unused):
        previous.add_(self.projection(x))
        return previous.clone(), previous, unused

    def prefill(self, x, cache):
        out = self(x)
        cache[0].copy_(out[:, -1:])
        return out


@pytest.fixture(autouse=True)
def stand_in_mamba():
    module = types.ModuleType("mamba_ssm")
    module.Mamba2 = _CausalMamba
    modules = types.ModuleType("mamba_ssm.modules")
    mamba2 = types.ModuleType("mamba_ssm.modules.mamba2")
    mamba2.causal_conv1d_fn = None
    modules.mamba2 = mamba2
    names = ("mamba_ssm", "mamba_ssm.modules", "mamba_ssm.modules.mamba2")
    old = {name: sys.modules.get(name) for name in names}
    sys.modules.update(zip(names, (module, modules, mamba2), strict=True))
    try:
        yield
    finally:
        for name, previous in old.items():
            if previous is None:
                del sys.modules[name]
            else:
                sys.modules[name] = previous


def _config(**changes) -> ModelConfig:
    return replace(
        ModelConfig.v02_proxy(),
        dropout=0.0,
        attention_dropout=0.0,
        max_trainable_parameters=200_000_000,
        **changes,
    )


def _codes(seed: int = 0, frames: int = 5):
    generator = torch.Generator().manual_seed(seed)
    user = torch.randint(0, 2048, (2, 8, frames), generator=generator)
    output = torch.randint(0, 2048, (2, 8, frames), generator=generator)
    initial = torch.randint(0, 2048, (2, 8), generator=generator)
    return user, output, initial


def test_shared_ear_embedding_uses_the_original_path():
    model = DauysHybrid(_config()).eval()
    assert model.ear_embeddings is None
    user, _, _ = _codes()
    with torch.no_grad():
        expected = model._embed(user, 0)
        actual = model._ear_embed(user)
    torch.testing.assert_close(actual, expected)


def test_separate_ear_uses_its_own_embeddings_without_role_or_dropout():
    model = DauysHybrid(_config(ear_embeddings="separate")).train()
    assert model.ear_embeddings is not None
    user, _, _ = _codes()
    with torch.no_grad():
        expected = sum(
            model.ear_embeddings[q](user[:, q]) for q in range(8)
        ) / math.sqrt(8)
        actual = model._ear_embed(user, acoustic_dropout=0.9)
    torch.testing.assert_close(actual, expected)
    with torch.no_grad():
        model.embeddings[0].weight.add_(1)
        model.roles.weight.add_(2)
    torch.testing.assert_close(model._ear_embed(user, acoustic_dropout=0.9), expected)
    with torch.no_grad():
        model.ear_embeddings[0].weight.add_(1)
    assert not torch.allclose(model._ear_embed(user), expected)


def test_separate_ear_rejects_self_hearing():
    with pytest.raises(ValueError, match="self-hearing"):
        DauysHybrid(_config(ear_embeddings="separate", self_hearing="mixed"))


def test_separate_ear_accepts_mamba_states(monkeypatch):
    def fake_forward_with_state(self, x, state):
        return x, None

    monkeypatch.setattr(MambaBlock, "forward_with_state", fake_forward_with_state)
    model = DauysHybrid(_config(ear_embeddings="separate")).eval()
    user, output, initial = _codes(frames=4)
    states = {}
    with torch.no_grad():
        result = model(user, output, initial, mamba_states=states)
    assert result.cb0_logits.shape == (2, 4, 2048)
    assert ("ear", 0) in states
    assert ("dialogue", 0) in states


def test_freeze_ear_drops_ear_gradients_and_keeps_blocks_in_eval():
    model = DauysHybrid(_config(ear_embeddings="separate", freeze_ear=True)).train()
    assert all(not p.requires_grad for p in model.ear_embeddings.parameters())
    assert all(not p.requires_grad for p in model.listening_projection.parameters())
    assert all(not p.requires_grad for p in model.listening.parameters())
    assert all(not block.training for block in model.listening)
    assert model.embeddings[0].weight.requires_grad
    assert model.dialogue_projection.weight.requires_grad

    user, output, initial = _codes()
    model(user, output, initial).cb0_logits.sum().backward()
    assert all(p.grad is None for p in model.listening.parameters())
    assert all(p.grad is None for p in model.ear_embeddings.parameters())
    assert model.dialogue_projection.weight.grad is not None


def test_intent_target_mirrors_the_separate_ear():
    model = DauysHybrid(_config(ear_embeddings="separate")).eval()
    target = IntentModule(model, IntentConfig(horizon=2)).target
    assert target.ear_embeddings is not None
    assert target.embeddings is None
    user, _, _ = _codes()
    with torch.no_grad():
        expected = sum(
            target.ear_embeddings[q](user[:, q]) for q in range(8)
        ) / math.sqrt(8)
        expected = target.projection(expected)
        for block in target.blocks:
            expected = block(expected)
        actual = target(user)
    torch.testing.assert_close(actual, expected)


def test_map_ear_state_maps_only_the_three_ear_prefixes():
    screen_state = {
        "embeddings.0.weight": torch.zeros(1),
        "embeddings.7.weight": torch.ones(1),
        "projection.weight": torch.zeros(2),
        "projection.bias": torch.ones(2),
        "blocks.0.projection.weight": torch.zeros(3),
        "blocks.5.projection.weight": torch.ones(3),
        "next_head.weight": torch.zeros(4),
        "memory_head.weight": torch.zeros(4),
        "norm.weight": torch.zeros(4),
        "histogram_head.weight": torch.zeros(4),
    }

    mapped = map_ear_state(screen_state)

    assert mapped == {
        "ear_embeddings.0.weight": screen_state["embeddings.0.weight"],
        "ear_embeddings.7.weight": screen_state["embeddings.7.weight"],
        "listening_projection.weight": screen_state["projection.weight"],
        "listening_projection.bias": screen_state["projection.bias"],
        "listening.0.projection.weight": screen_state["blocks.0.projection.weight"],
        "listening.5.projection.weight": screen_state["blocks.5.projection.weight"],
    }


def test_filter_intent_state_drops_only_ema_target_keys():
    state = {
        "head.net.0.weight": torch.zeros(1),
        "head.net.0.bias": torch.ones(1),
        "target.embeddings.0.weight": torch.zeros(2),
        "target.role": torch.zeros(2),
        "target.projection.weight": torch.ones(2),
        "target.blocks.0.projection.weight": torch.zeros(2),
    }
    filtered = filter_intent_state(state)
    assert filtered == {
        "head.net.0.weight": state["head.net.0.weight"],
        "head.net.0.bias": state["head.net.0.bias"],
    }


def test_initial_checkpoint_without_separate_ear_still_loads(tmp_path):
    old = DauysHybrid(_config(ear_embeddings="separate"))
    state = {
        key: value
        for key, value in old.state_dict().items()
        if not key.startswith("ear_embeddings.")
    }
    path = tmp_path / "old.pt"
    torch.save({"model": state}, path)
    new = DauysHybrid(_config(ear_embeddings="separate"))
    load_initial_weights(new, path, from_v01=False)
    assert new.ear_embeddings is not None


def test_memear_config_is_separate_and_frozen():
    path = (
        Path(__file__).parents[1]
        / "src/dauys/configs/dauys-hybrid-v0.2-stage1-memear.yaml"
    )
    config = ModelConfig.from_yaml(path)
    assert config.ear_embeddings == "separate"
    assert config.freeze_ear
