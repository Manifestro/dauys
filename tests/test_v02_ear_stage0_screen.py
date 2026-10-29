"""Checks for the Stage 0 ear-memory screen."""

import sys
import types
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

import v02_ear_stage0_screen as screen


def test_parse_cell_accepts_mimi_memory():
    cell = screen.parse_cell("all8:c1memmimi:12:64")
    assert cell == {
        "name": "ear_all8_c1memmimi_h12_w64",
        "codebooks": 8,
        "lags": screen.memory_lags(12),
        "histogram": False,
        "memory_target": "cb0",
        "width": 64,
    }
    assert screen.parse_cell("cb0:c1mem")["memory_target"] == "units"


def test_cb0_memory_uses_mimi_codes_as_targets(monkeypatch):
    class _Block(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

        def forward(self, x):
            return x

    monkeypatch.setattr(screen, "MambaBlock", _Block)
    config = types.SimpleNamespace(
        vocab_size=13,
        embedding_dim=4,
        dropout=0.0,
        listening_d_state=4,
        mamba_norm_eps=1e-5,
        listening_layers=1,
    )
    ear = screen.Ear(
        config,
        codebooks=1,
        clusters=3,
        lags=(0, 1),
        width=4,
        memory_target="cb0",
    )
    assert ear.memory_head is not None
    assert ear.memory_head.out_features == config.vocab_size * 2
    codes = torch.arange(2 * 8 * 5).reshape(2, 8, 5) % config.vocab_size
    units = torch.zeros((2, 5), dtype=torch.long)
    speech = torch.ones((2, 5), dtype=torch.bool)
    targets = []
    original = screen.F.cross_entropy

    def cross_entropy(inputs, target, *args, **kwargs):
        targets.append(target.detach().clone())
        return original(inputs, target, *args, **kwargs)

    monkeypatch.setattr(screen.F, "cross_entropy", cross_entropy)
    c1, memory = screen.losses(ear, codes, units, speech)

    assert torch.isfinite(c1)
    assert torch.isfinite(memory)
    assert torch.equal(targets[-2], codes[:, 0, :].reshape(-1))
    assert torch.equal(targets[-1], codes[:, 0, :4].reshape(-1))
