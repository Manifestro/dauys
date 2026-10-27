"""All six suites must contribute to every checkpoint decision."""

import pytest
import torch

from dauys.decoding import DecodingConfig
from dauys.v02_evaluation import Suite
from dauys.v02_harness import FrozenSuiteHarness


class _Backend:
    def e1(self, _model, _item):
        return {"cb0_nll": 1.0, "cb0_accuracy": 0.8, "depth_nll": 2.0}

    def e2(self, _model, _item, _decoding):
        return {
            "original_nll": 1.0,
            "matched_nll": 1.5,
            "wer": 0.25,
            "relevance": 2.0,
            "onset_latency_ms": 300.0,
            "false_starts": 0.0,
        }

    def e3(self, _model, _item):
        return torch.tensor([[0.1, 0.9], [0.9, 0.1]])

    def e4(self, _model, _item):
        return True

    def e5(self, _model, _item, _decoding):
        return {"yield_within_500ms": 1.0}

    def e6(self, _model, _item, _decoding):
        return {"backchannel_rate": 0.75}


def test_per_checkpoint_harness_reports_every_suite():
    suites = {
        f"E{index}": Suite(f"E{index}", "1", ({"id": f"item_{index}"},), "a" * 64)
        for index in range(1, 7)
    }
    harness = FrozenSuiteHarness(suites, _Backend(), DecodingConfig(), 0.9)
    model = torch.nn.Linear(1, 1).eval()
    result = harness(model, 10)
    assert result["eval/codebook/e1_cb0_nll"] == 1.0
    assert result["eval/codebook/cb0_accuracy_gap"] == pytest.approx(0.1)
    assert result["eval/conditioning/e2_matched_delta_cb0_onset"] == 0.5
    assert result["eval/conditioning/e2_matched_delta_ci95_low"] == 0.5
    assert result["eval/semantic/e3_accuracy"] == 1.0
    assert result["eval/semantic/e4_accuracy"] == 1.0
    assert result["eval/timing/e5_yield_within_500ms"] == 1.0
    assert result["eval/timing/e6_backchannel_rate"] == 0.75


def test_harness_breaks_out_unseen_voice_and_memory_distance():
    suites = {
        f"E{index}": Suite(
            f"E{index}",
            "1",
            (
                {
                    "id": f"item_{index}",
                    "unseen_voice": True,
                    "probe": {"distance_turns": 3, "updates": 1},
                    "condition": "overlap",
                },
            ),
            "a" * 64,
        )
        for index in range(1, 7)
    }
    result = FrozenSuiteHarness(suites, _Backend(), DecodingConfig(), 0.9)(
        torch.nn.Linear(1, 1).eval(), 10
    )
    assert result["eval/generalization/e3_unseen_accuracy"] == 1.0
    assert result["eval/semantic/e4_distance_turns_3_accuracy"] == 1.0
    assert result["eval/semantic/e4_updates_1_accuracy"] == 1.0
    assert result["eval/timing/e5_overlap_yield"] == 1.0


def test_harness_requires_and_reports_every_fusion_block():
    class Model(torch.nn.Linear):
        def fusion_block_indices(self):
            return (1, 3)

    class Backend(_Backend):
        def e1(self, _model, _item):
            return {
                "cb0_nll": 1.0,
                "cb0_accuracy": 0.8,
                "depth_nll": 2.0,
                "fusion_off_cb0_nll_by_block": {"1": 1.2, "3": 1.4},
            }

    suites = {
        f"E{index}": Suite(f"E{index}", "1", ({"id": f"item_{index}"},), "a" * 64)
        for index in range(1, 7)
    }
    harness = FrozenSuiteHarness(suites, Backend(), DecodingConfig(), 0.9)
    result = harness(Model(1, 1).eval(), 1)
    assert result["eval/training/cross_attention/block_1_delta_cb0_nll"] == pytest.approx(0.2)
    assert result["eval/training/cross_attention/block_3_delta_cb0_nll"] == pytest.approx(0.4)
    with pytest.raises(ValueError, match="dialogue block 1"):
        FrozenSuiteHarness(suites, _Backend(), DecodingConfig(), 0.9)(
            Model(1, 1).eval(), 1
        )
