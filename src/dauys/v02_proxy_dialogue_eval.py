"""Teacher-forced held-out Hohere checks for the Track P dialogue stage."""

import json
import os
from collections.abc import Sequence
from pathlib import Path

import torch
from torch.nn import functional as F

from .model import DauysHybrid
from .v02_data import CorpusDataset, CorpusRecord, collate_cases, load_corpus


class HeldoutDialogueEvaluator:
    """CB0, depth and activity on held-out conversations, plus the CB0 gap."""

    def __init__(
        self,
        heldout: Sequence[CorpusRecord],
        training: Sequence[CorpusRecord],
        silence_codes: torch.Tensor,
        batch_size: int = 16,
    ) -> None:
        if not heldout or not training or silence_codes.shape != (8,):
            raise ValueError("Held-out and training records and silence are required")
        if {record.id for record in heldout} & {record.id for record in training}:
            raise ValueError("Held-out conversations overlap the training sample")
        self.heldout = CorpusDataset(list(heldout))
        self.training = CorpusDataset(list(training))
        self.silence = silence_codes.long()
        self.batch_size = batch_size

    @torch.inference_mode()
    def _scores(self, model: DauysHybrid, dataset: CorpusDataset) -> dict[str, float]:
        device = next(model.parameters()).device
        totals = dict.fromkeys(
            ("frames", "correct", "cb0", "depth", "tp", "fp", "fn"), 0.0
        )
        for start in range(0, len(dataset), self.batch_size):
            cases = [
                dataset[index]
                for index in range(start, min(start + self.batch_size, len(dataset)))
            ]
            batch = collate_cases(cases, self.silence).to(device)
            output = model(
                batch.user_codes, batch.output_codes, batch.initial_output_codes
            )
            valid = batch.valid_frames
            target = batch.output_codes
            cb0 = output.cb0_logits.float()
            totals["cb0"] += float(
                F.cross_entropy(cb0[valid], target[:, 0][valid], reduction="sum")
            )
            totals["correct"] += float(((cb0.argmax(-1) == target[:, 0]) & valid).sum())
            if output.depth_logits is not None:
                depth = output.depth_logits.float()[valid]
                totals["depth"] += (
                    float(
                        F.cross_entropy(
                            depth.reshape(-1, depth.shape[-1]),
                            target[:, 1:].transpose(1, 2)[valid].reshape(-1),
                            reduction="sum",
                        )
                    )
                    / 7
                )
            speaking = output.activity_logits.argmax(-1).bool() & valid
            truth = batch.output_activity.bool() & valid
            totals["tp"] += float((speaking & truth).sum())
            totals["fp"] += float((speaking & ~truth).sum())
            totals["fn"] += float((~speaking & truth).sum())
            totals["frames"] += float(valid.sum())
        precision = totals["tp"] / max(1.0, totals["tp"] + totals["fp"])
        recall = totals["tp"] / max(1.0, totals["tp"] + totals["fn"])
        return {
            "cb0_nll": totals["cb0"] / totals["frames"],
            "cb0_accuracy": totals["correct"] / totals["frames"],
            "depth_nll": totals["depth"] / totals["frames"],
            "activity_f1": 2 * precision * recall / max(1e-9, precision + recall),
        }

    def __call__(self, model: DauysHybrid, step: int) -> dict[str, float]:
        if step < 0 or model.training:
            raise ValueError("Evaluation requires a nonnegative step and eval mode")
        heldout = self._scores(model, self.heldout)
        training = self._scores(model, self.training)
        return {
            "eval/step": float(step),
            "eval/codebook/heldout_cb0_nll": heldout["cb0_nll"],
            "eval/codebook/heldout_cb0_accuracy": heldout["cb0_accuracy"],
            "eval/codebook/heldout_depth_nll": heldout["depth_nll"],
            "eval/timing/heldout_activity_f1": heldout["activity_f1"],
            "eval/codebook/train_cb0_nll": training["cb0_nll"],
            "eval/codebook/train_cb0_accuracy": training["cb0_accuracy"],
            "eval/codebook/cb0_accuracy_gap": training["cb0_accuracy"]
            - heldout["cb0_accuracy"],
        }


def create_heldout_dialogue_evaluator(
    *, suites: dict, records: list[CorpusRecord]
) -> HeldoutDialogueEvaluator:
    """Factory configured by DAUYS_HELDOUT_{ROOT,MANIFEST,COMPANION} and silence."""
    del suites
    required = (
        "DAUYS_HELDOUT_ROOT",
        "DAUYS_HELDOUT_MANIFEST",
        "DAUYS_HELDOUT_COMPANION",
        "DAUYS_SILENCE_CODES",
    )
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        raise ValueError(f"Missing evaluator configuration: {', '.join(missing)}")
    heldout = load_corpus(
        os.environ["DAUYS_HELDOUT_ROOT"],
        os.environ["DAUYS_HELDOUT_MANIFEST"],
        os.environ["DAUYS_HELDOUT_COMPANION"],
    )
    training = sorted(
        (record for record in records if record.source == "hohere"),
        key=lambda record: record.id,
    )[: len(heldout)]
    silence = torch.tensor(
        json.loads(Path(os.environ["DAUYS_SILENCE_CODES"]).read_text()),
        dtype=torch.long,
    )
    return HeldoutDialogueEvaluator(heldout, training, silence)
