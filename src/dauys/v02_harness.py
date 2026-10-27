"""Per-checkpoint E1–E6 harness over immutable evaluation suites."""

from collections.abc import Callable, Mapping
from statistics import median
from typing import Any, Protocol

import torch

from .decoding import DecodingConfig
from .model import DauysHybrid
from .v02_evaluation import Suite, bootstrap_mean_ci, crossed_minimal_pair_accuracy


class ProbeBackend(Protocol):
    """Hohere-owned suite adapter; text is used only for output evaluation."""

    def e1(self, model: DauysHybrid, item: Mapping[str, Any]) -> Mapping[str, Any]: ...

    def e2(
        self, model: DauysHybrid, item: Mapping[str, Any], decoding: DecodingConfig
    ) -> Mapping[str, float]: ...

    def e3(self, model: DauysHybrid, item: Mapping[str, Any]) -> torch.Tensor: ...

    def e4(self, model: DauysHybrid, item: Mapping[str, Any]) -> bool: ...

    def e5(
        self, model: DauysHybrid, item: Mapping[str, Any], decoding: DecodingConfig
    ) -> Mapping[str, float]: ...

    def e6(
        self, model: DauysHybrid, item: Mapping[str, Any], decoding: DecodingConfig
    ) -> Mapping[str, float]: ...


class FrozenSuiteHarness:
    def __init__(
        self,
        suites: Mapping[str, Suite],
        backend: ProbeBackend,
        decoding: DecodingConfig,
        train_cb0_accuracy: float | Callable[[DauysHybrid], float],
    ):
        if set(suites) != {f"E{index}" for index in range(1, 7)}:
            raise ValueError("All six frozen evaluation suites are required")
        ids = [item["id"] for suite in suites.values() for item in suite.items]
        if len(ids) != len(set(ids)):
            raise ValueError("Evaluation suite IDs overlap")
        if isinstance(train_cb0_accuracy, float) and not 0 <= train_cb0_accuracy <= 1:
            raise ValueError("Invalid training CB0 accuracy")
        decoding.validate()
        self.suites = dict(suites)
        self.backend = backend
        self.decoding = decoding
        self.train_cb0_accuracy = train_cb0_accuracy

    @staticmethod
    def _mean(values: list[float]) -> float:
        if not values:
            raise ValueError("Evaluation suite has no measurements")
        return sum(values) / len(values)

    def __call__(self, model: DauysHybrid, step: int) -> dict[str, float]:
        if step < 0 or model.training:
            raise ValueError("Evaluation requires a nonnegative step and eval mode")
        e1 = [self.backend.e1(model, item) for item in self.suites["E1"].items]
        e2 = [
            self.backend.e2(model, item, self.decoding)
            for item in self.suites["E2"].items
        ]
        e3_matrices = [self.backend.e3(model, item) for item in self.suites["E3"].items]
        matrices = torch.stack(e3_matrices)
        e3 = crossed_minimal_pair_accuracy(matrices)
        e4 = [float(self.backend.e4(model, item)) for item in self.suites["E4"].items]
        e5 = [
            self.backend.e5(model, item, self.decoding)
            for item in self.suites["E5"].items
        ]
        e6 = [
            self.backend.e6(model, item, self.decoding)
            for item in self.suites["E6"].items
        ]
        e1_accuracy = self._mean([item["cb0_accuracy"] for item in e1])
        train_accuracy = (
            self.train_cb0_accuracy(model)
            if callable(self.train_cb0_accuracy)
            else self.train_cb0_accuracy
        )
        if not 0 <= train_accuracy <= 1:
            raise ValueError("Invalid training CB0 accuracy")
        e2_deltas = [item["matched_nll"] - item["original_nll"] for item in e2]
        delta_low, delta_high = bootstrap_mean_ci(e2_deltas)
        metrics = {
            "eval/step": float(step),
            "eval/codebook/e1_cb0_nll": self._mean([item["cb0_nll"] for item in e1]),
            "eval/codebook/e1_cb0_accuracy": e1_accuracy,
            "eval/codebook/e1_depth_nll": self._mean(
                [item["depth_nll"] for item in e1]
            ),
            "eval/codebook/cb0_accuracy_gap": train_accuracy - e1_accuracy,
            "eval/conditioning/e2_matched_delta_cb0_onset": self._mean(e2_deltas),
            "eval/conditioning/e2_matched_delta_ci95_low": delta_low,
            "eval/conditioning/e2_matched_delta_ci95_high": delta_high,
            "eval/semantic/e2_wer": self._mean([item["wer"] for item in e2]),
            "eval/semantic/e2_relevance": self._mean(
                [item["relevance"] for item in e2]
            ),
            "eval/semantic/e3_accuracy": e3["accuracy"],
            "eval/semantic/e3_ci95_low": e3["ci95_low"],
            "eval/semantic/e3_ci95_high": e3["ci95_high"],
            "eval/semantic/e4_accuracy": self._mean(e4),
            "eval/timing/e2_onset_latency_ms": median(
                [item["onset_latency_ms"] for item in e2]
            ),
            "eval/timing/e2_false_starts": self._mean(
                [item["false_starts"] for item in e2]
            ),
            "eval/timing/e5_yield_within_500ms": self._mean(
                [item["yield_within_500ms"] for item in e5]
            ),
            "eval/timing/e6_backchannel_rate": self._mean(
                [item["backchannel_rate"] for item in e6]
            ),
        }
        if all("premature_takeover" in item for item in e6):
            metrics["eval/timing/e6_premature_takeover"] = self._mean(
                [item["premature_takeover"] for item in e6]
            )
        results: dict[str, list[Any]] = {
            "E1": e1,
            "E2": e2,
            "E3": e3_matrices,
            "E4": e4,
            "E5": e5,
            "E6": e6,
        }
        for name, suite in self.suites.items():
            indices = [
                index
                for index, item in enumerate(suite.items)
                if item.get("unseen_voice", False)
            ]
            if not indices:
                continue
            values = results[name]
            if name == "E1":
                metrics["eval/generalization/e1_unseen_cb0_nll"] = self._mean(
                    [values[index]["cb0_nll"] for index in indices]
                )
            elif name == "E2":
                metrics["eval/generalization/e2_unseen_relevance"] = self._mean(
                    [values[index]["relevance"] for index in indices]
                )
            elif name == "E3":
                metrics["eval/generalization/e3_unseen_accuracy"] = (
                    crossed_minimal_pair_accuracy(
                        torch.stack([values[index] for index in indices])
                    )["accuracy"]
                )
            elif name == "E4":
                metrics["eval/generalization/e4_unseen_accuracy"] = self._mean(
                    [float(values[index]) for index in indices]
                )
            elif name == "E5":
                metrics["eval/generalization/e5_unseen_yield"] = self._mean(
                    [values[index]["yield_within_500ms"] for index in indices]
                )
            else:
                metrics["eval/generalization/e6_unseen_backchannel_rate"] = self._mean(
                    [values[index]["backchannel_rate"] for index in indices]
                )
        for item in self.suites["E4"].items:
            probe = item.get("probe", item)
            for field in ("distance_turns", "updates"):
                if field in probe:
                    key = f"eval/semantic/e4_{field}_{probe[field]}_accuracy"
                    peers = [
                        float(e4[position])
                        for position, other in enumerate(self.suites["E4"].items)
                        if other.get("probe", other).get(field) == probe[field]
                    ]
                    metrics[key] = self._mean(peers)
        for item in self.suites["E5"].items:
            condition = item.get("condition")
            if condition is not None:
                metrics[f"eval/timing/e5_{condition}_yield"] = self._mean(
                    [
                        value["yield_within_500ms"]
                        for value, other in zip(e5, self.suites["E5"].items)
                        if other.get("condition") == condition
                    ]
                )
        placement = [
            item["backchannel_placement"]
            for item in e6
            if "backchannel_placement" in item
        ]
        if placement:
            metrics["eval/timing/e6_backchannel_placement"] = self._mean(placement)
        blocks = (
            model.fusion_block_indices()
            if hasattr(model, "fusion_block_indices")
            else ()
        )
        if blocks:
            for block in blocks:
                off_scores = []
                for item in e1:
                    ablations = item.get("fusion_off_cb0_nll_by_block")
                    if not isinstance(ablations, Mapping) or (
                        str(block) not in ablations and block not in ablations
                    ):
                        raise ValueError(
                            f"E1 fusion ablation is missing dialogue block {block}"
                        )
                    off_scores.append(
                        float(
                            ablations[str(block)]
                            if str(block) in ablations
                            else ablations[block]
                        )
                    )
                metrics[
                    f"eval/training/cross_attention/block_{block}_delta_cb0_nll"
                ] = self._mean(off_scores) - metrics["eval/codebook/e1_cb0_nll"]
        return metrics
