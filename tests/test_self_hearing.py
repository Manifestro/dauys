"""The ear hearing the model's own voice: no leakage, silent start, streaming parity."""

import sys
import types
from dataclasses import replace

import pytest
import torch
from torch import nn

from dauys.config import ModelConfig
from dauys.model import DauysHybrid
from dauys.v02_runner import load_initial_weights


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


def _model(mode: str, seed: int = 3) -> DauysHybrid:
    torch.manual_seed(seed)
    config = replace(
        ModelConfig.v02_proxy(), dropout=0.0, attention_dropout=0.0, self_hearing=mode
    )
    return DauysHybrid(config).eval()


def _wake(model: DauysHybrid) -> None:
    """Give the silent self-hearing input real weight so the tests see it."""
    with torch.no_grad():
        if model.self_hearing_gate is not None:
            model.self_hearing_gate.fill_(0.7)
        if model.self_hearing_projection is not None:
            nn.init.normal_(model.self_hearing_projection.weight, std=0.2)


def _codes(seed: int, frames: int = 7) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(seed)
    user = torch.randint(0, 2048, (2, 8, frames), generator=generator)
    output = torch.randint(0, 2048, (2, 8, frames), generator=generator)
    initial = torch.randint(0, 2048, (2, 8), generator=generator)
    return user, output, initial


@pytest.mark.parametrize("mode", ["labeled", "mixed"])
def test_ear_never_hears_the_frame_being_predicted(mode):
    model = _model(mode)
    _wake(model)
    user, output, initial = _codes(1)
    with torch.no_grad():
        base = model(user, output, initial).cb0_logits
        for frame in range(output.shape[-1]):
            changed = output.clone()
            changed[:, :, frame] = (changed[:, :, frame] + 17) % 2048
            logits = model(user, changed, initial).cb0_logits
            # B_t is the target at frame t; predictions up to t must not move.
            torch.testing.assert_close(
                logits[:, : frame + 1], base[:, : frame + 1], rtol=0, atol=0
            )
            if frame + 1 < output.shape[-1]:
                assert not torch.allclose(logits[:, frame + 1 :], base[:, frame + 1 :])


@pytest.mark.parametrize("mode", ["labeled", "mixed"])
def test_self_hearing_starts_silent(mode):
    plain = _model("none")
    hearing = _model(mode)
    hearing.load_state_dict(plain.state_dict(), strict=False)
    user, output, initial = _codes(2)
    with torch.no_grad():
        torch.testing.assert_close(
            hearing(user, output, initial).cb0_logits,
            plain(user, output, initial).cb0_logits,
        )


@pytest.mark.parametrize("mode", ["labeled", "mixed"])
def test_checkpoint_without_self_hearing_loads(mode, tmp_path):
    plain = _model("none")
    path = tmp_path / "plain.pt"
    torch.save({"model": plain.state_dict()}, path)
    hearing = _model(mode, seed=9)
    load_initial_weights(hearing, path, from_v01=False)
    for name, tensor in plain.state_dict().items():
        assert torch.equal(hearing.state_dict()[name], tensor)
    assert all(
        not tensor.any()
        for name, tensor in hearing.state_dict().items()
        if name.startswith("self_hearing_")
    )


@pytest.mark.parametrize("mode", ["none", "labeled", "mixed"])
def test_streaming_matches_teacher_forcing(mode):
    model = _model(mode)
    _wake(model)
    user, output, initial = _codes(4, frames=6)
    with torch.no_grad():
        expected = model(user, output, initial).cb0_logits
        state = model.prefill_teacher(user[:, :, :2], output[:, :, :2], initial)
        steps = []
        for frame in range(2, 6):
            cb0, _, _ = model.stream_step_teacher(
                user[:, :, frame], output[:, :, frame - 1], state
            )
            steps.append(cb0)
    torch.testing.assert_close(
        torch.stack(steps, dim=1), expected[:, 2:], rtol=1e-4, atol=1e-4
    )
