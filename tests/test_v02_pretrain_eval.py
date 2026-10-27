"""Pre-training continuation evaluator on a CPU stand-in model."""

import math
import sys
import types
from dataclasses import replace

import numpy as np
import pytest
import torch
from torch import nn

from dauys.config import ModelConfig
from dauys.model import DauysHybrid
from dauys.v02_data import CorpusRecord, sha256
from dauys.v02_pretrain_eval import (
    CONTINUATION_FRAMES,
    FRAME_SAMPLES,
    PretrainEvaluator,
    word_statistics,
)
from dauys.v02_proxy_dialogue_eval import HeldoutDialogueEvaluator
from dauys.v02_runner import _validate_evaluation


class _CausalMamba(nn.Module):
    def __init__(self, d_model, layer_idx, **_kwargs):
        super().__init__()
        self.layer_idx = layer_idx
        self.projection = nn.Linear(d_model, d_model)

    def forward(self, x, inference_params=None):
        mixed = self.projection(x).cumsum(dim=1)
        if inference_params is not None:
            state = inference_params.key_value_memory_dict[self.layer_idx][0]
            state.copy_(mixed[:, -1:])
        return mixed

    def allocate_inference_cache(self, batch_size, max_seqlen):
        width = self.projection.in_features
        return (torch.zeros(batch_size, 1, width), torch.zeros(1))

    def step(self, x, previous, unused):
        previous.add_(self.projection(x))
        return previous.clone(), previous, unused


class _Codec:
    def decode(self, codes):
        return torch.zeros(codes.shape[0], 1, codes.shape[-1] * FRAME_SAMPLES)


@pytest.fixture(scope="module")
def model():
    module = types.ModuleType("mamba_ssm")
    module.Mamba2 = _CausalMamba
    modules = types.ModuleType("mamba_ssm.modules")
    modules.mamba2 = types.SimpleNamespace(causal_conv1d_fn=None)
    module.modules = modules
    names = ("mamba_ssm", "mamba_ssm.modules")
    old = {name: sys.modules.get(name) for name in names}
    sys.modules["mamba_ssm"] = module
    sys.modules["mamba_ssm.modules"] = modules
    try:
        torch.manual_seed(5)
        config = replace(ModelConfig.v02_proxy(), dropout=0.0, attention_dropout=0.0)
        yield DauysHybrid(config).eval()
    finally:
        for name, previous in old.items():
            if previous is None:
                del sys.modules[name]
            else:
                sys.modules[name] = previous


def test_word_statistics_excludes_empty_continuations():
    scores = word_statistics(
        ["the cat sat", "ok", "the zzqx dog"], {"the", "cat", "sat", "dog"}
    )
    assert scores["empty_rate"] == pytest.approx(1 / 3)
    assert scores["word_rate"] == pytest.approx((1 + 2 / 3) / 2)
    assert word_statistics([""], set())["word_rate"] == 0.0


def test_evaluator_reports_required_pretraining_metrics(model, tmp_path):
    items = [
        {"id": f"item-{i}", "text": "x", "codes": torch.randint(0, 2048, (8, 120))}
        for i in range(3)
    ]
    heard = []

    def transcribe(waves):
        heard.extend(waves)
        return ["the cat sat"] * len(waves)

    evaluator = PretrainEvaluator(
        {"items": items, "ceiling": {"word_rate": 0.9, "empty_rate": 0.0}},
        _Codec(),
        torch.zeros(8, dtype=torch.long),
        transcribe,
        {"the", "cat"},
        [torch.randint(0, 2048, (8, 40))],
        audio_dir=tmp_path,
        listen_items=2,
        batch_size=2,
    )
    scores = evaluator(model, 5)
    assert all(wave.shape == (CONTINUATION_FRAMES * FRAME_SAMPLES,) for wave in heard)
    assert scores["eval/pretrain/cont_word_rate"] == pytest.approx(2 / 3)
    assert scores["eval/pretrain/cont_empty_rate"] == 0.0
    assert math.isfinite(scores["eval/codebook/cb0_accuracy_gap"])
    assert len(list((tmp_path / "step-000005").glob("*.wav"))) == 2
    assert np.isfinite(scores["eval/codebook/heldout_cb0_nll"])


def _hohere_record(tmp_path, identity, frames=7):
    path = tmp_path / f"{identity}.pt"
    activity = torch.tensor([0, 1, 1, 0, 1, 1, 0][:frames], dtype=torch.uint8)
    torch.save(
        {
            "user": torch.randint(0, 2048, (8, frames), dtype=torch.int16),
            "target": torch.randint(0, 2048, (8, frames), dtype=torch.int16),
            "user_activity": 1 - activity,
            "activity": activity,
        },
        path,
    )
    return CorpusRecord(
        id=identity,
        source="hohere",
        skill="simple_dialogue",
        family=f"family_{identity}",
        user_voice="voice",
        frames=frames,
        token_path=path,
        token_sha256=sha256(path),
        activity_a=tuple((1 - activity).tolist()),
        activity_b=tuple(activity.tolist()),
        events=(),
    )


def test_heldout_dialogue_evaluator_reports_proxy_dialogue_metrics(model, tmp_path):
    heldout = [_hohere_record(tmp_path, f"held_{i}", 5 + i) for i in range(3)]
    training = [_hohere_record(tmp_path, f"train_{i}") for i in range(2)]
    evaluator = HeldoutDialogueEvaluator(
        heldout, training, torch.zeros(8, dtype=torch.long), batch_size=2
    )
    scores = evaluator(model, 3)
    _validate_evaluation(scores, "dialogue", "P")
    assert all(math.isfinite(value) for value in scores.values())
    assert 0 <= scores["eval/timing/heldout_activity_f1"] <= 1
    with pytest.raises(ValueError, match="overlap"):
        HeldoutDialogueEvaluator(heldout, heldout, torch.zeros(8, dtype=torch.long))
