"""Batched probe scoring must reproduce the per-item backend."""

import sys
import types
from dataclasses import replace

import pytest
import torch
from torch import nn

from dauys.config import ModelConfig
from dauys.decoding import DecodingConfig
from dauys.model import DauysHybrid
from dauys.v02_batched_probes import BatchedProbeScorer, _groups
from dauys.v02_hohere_backend import HohereProbeBackend


class _CausalMamba(nn.Module):
    def __init__(self, d_model, layer_idx, **_kwargs):
        super().__init__()
        self.layer_idx = layer_idx
        self.projection = nn.Linear(d_model, d_model)

    def forward(self, x, inference_params=None):
        out = self.projection(x).cumsum(dim=1)
        if inference_params is not None:
            state = inference_params.key_value_memory_dict[self.layer_idx]
            state[0].copy_(out[:, -1:])
        return out

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


@pytest.fixture(params=["none", "mixed"])
def setup(request, tmp_path):
    torch.manual_seed(0)
    config = replace(
        ModelConfig.v02_proxy(),
        dropout=0.0,
        attention_dropout=0.0,
        self_hearing=request.param,
    )
    model = DauysHybrid(config).eval()
    if model.self_hearing_gate is not None:
        with torch.no_grad():
            model.self_hearing_gate.fill_(0.5)
    silence = torch.randint(0, 2048, (8,))
    backend = HohereProbeBackend(tmp_path, {}, None, silence, seeds=1)
    generator = torch.Generator().manual_seed(1)
    items = []
    for index, frames in enumerate((9, 14, 6, 11)):
        identity = f"item-{index}"
        backend._token_cache[identity] = (
            torch.randint(0, 2048, (8, frames), generator=generator),
            torch.randint(0, 2048, (8, frames), generator=generator),
        )
        items.append({"id": identity})
    return model, backend, BatchedProbeScorer(backend, None, None), items


def test_groups_cover_every_index_once():
    lengths = [5, 300, 40, 40, 1000, 7]
    groups = _groups(lengths, budget=1200)
    assert sorted(i for group in groups for i in group) == list(range(len(lengths)))
    assert all(max(lengths[i] for i in g) * len(g) <= 1200 for g in groups)


def test_teacher_forced_windows_match_per_item(setup):
    model, backend, scorer, items = setup
    cases = []
    for item in items:
        user, output = backend._channels(item)
        cases.append((user, output, 2, output.shape[1] - 1))
    batched = scorer.cb0_nll_many(model, cases)
    single = [backend._cb0_nll(model, *case) for case in cases]
    torch.testing.assert_close(torch.tensor(batched), torch.tensor(single))


def test_e1_matches_per_item(setup):
    model, backend, scorer, items = setup
    batched = scorer.e1_many(model, items)
    for item, result in zip(items, batched, strict=True):
        single = backend.e1(model, item)
        for key in ("cb0_nll", "cb0_accuracy", "depth_nll"):
            assert result[key] == pytest.approx(single[key], rel=1e-5, abs=1e-6)
        for block, value in single["fusion_off_cb0_nll_by_block"].items():
            assert result["fusion_off_cb0_nll_by_block"][block] == pytest.approx(
                value, rel=1e-5
            )


def test_greedy_generation_matches_per_item(setup):
    model, backend, scorer, items = setup
    decoding = replace(DecodingConfig(), activity_mode="hysteresis")
    requests = []
    for index, item in enumerate(items):
        user, output = backend._channels(item)
        requests.append((user, output, 1 + index))
    batched = scorer.generate_many(model, requests, decoding, seed=3)
    for (user, output, start), (codes, activity) in zip(
        requests, batched, strict=True
    ):
        single_codes, single_activity = backend._generate(
            model, user, output, start, decoding, seed=3
        )
        assert torch.equal(codes, single_codes.cpu())
        assert torch.equal(activity, single_activity)
