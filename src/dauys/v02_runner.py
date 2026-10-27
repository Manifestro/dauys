"""Local, stage-aware training loop with durable checkpoint boundaries."""

import json
import math
import os
import random
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from functools import partial
from pathlib import Path
from typing import Literal

import torch
from torch.utils.data import DataLoader

from .model import DauysHybrid
from .v02_data import CorpusDataset, CorpusRecord, collate_cases, sample_records
from .v02_training import (
    Stage,
    Toggles,
    V02Batch,
    objective,
    optimizer_groups,
    schedule,
)

Track = Literal["FT", "P", "PRE", "S0", "S1", "S2", "S3", "S4", "S5", "SFT"]
EvaluationCallback = Callable[[DauysHybrid, int], dict[str, float]]
LogCallback = Callable[[dict[str, float]], None]


@dataclass(frozen=True)
class TrainingPlan:
    track: Track
    stage: Stage
    steps: int
    batch_size: int
    learning_rate: float
    checkpoint_interval: int
    num_workers: int = 4
    seed: int = 42
    context_frames: int = 256
    own_history_after_epochs: int = 2
    own_history_frames: int = 64
    stage2_peak_learning_rate: float | None = None
    early_stop_patience: int = 3

    def validate(self) -> None:
        if min(self.steps, self.batch_size, self.checkpoint_interval) < 1:
            raise ValueError(
                "Steps, batch size and checkpoint interval must be positive"
            )
        if self.learning_rate <= 0 or self.num_workers < 0:
            raise ValueError("Invalid learning rate or worker count")
        if self.context_frames < 1:
            raise ValueError("Context length must be positive")
        if self.stage == "pretrain" and self.track not in ("P", "PRE"):
            raise ValueError("Pre-training requires P or PRE track")
        if self.track == "PRE" and self.stage != "pretrain":
            raise ValueError("PRE track requires pretrain stage")
        if self.track == "FT" and self.stage != "dialogue":
            raise ValueError("Fine-tuning requires dialogue data")
        if self.track == "FT" and self.steps > 1000:
            raise ValueError("R0 fine-tuning is limited to 1,000 optimizer steps")
        if self.track == "FT" and self.learning_rate > 2e-5:
            raise ValueError("R0 fine-tuning learning rate cannot exceed 0.1x peak")
        if (self.stage == "posttrain") != (self.track == "SFT"):
            raise ValueError("Targeted SFT requires posttrain stage and SFT track")
        if self.track == "SFT":
            if self.steps > 3000:
                raise ValueError("Targeted SFT is limited to 3,000 steps")
            if self.stage2_peak_learning_rate is None or (
                self.learning_rate > 0.1 * self.stage2_peak_learning_rate
            ):
                raise ValueError("Targeted SFT LR must be at most 0.1x Stage 2 peak")
            if self.early_stop_patience < 1:
                raise ValueError("SFT early stop patience must be positive")


def toggles_for_track(track: Track, stage: Stage) -> Toggles:
    if stage == "pretrain":
        return Toggles(contrast=False)
    if track == "FT":
        return Toggles(
            regularization=False,
            listening=False,
            history_dropout=True,
            acoustic_dropout=False,
            transition=True,
            contrast=False,
        )
    if track == "SFT":
        return Toggles()
    levels = {"S0": 0, "S1": 1, "S2": 2, "S3": 3, "S4": 4, "S5": 5, "P": 5}
    level = levels[track]
    return Toggles(
        regularization=level >= 1,
        listening=level >= 2,
        history_dropout=level >= 3,
        acoustic_dropout=level >= 3,
        transition=level >= 4,
        contrast=level >= 5,
    )


def optimizer_groups_for_track(model: DauysHybrid, toggles: Toggles) -> list[dict]:
    if toggles.regularization:
        return optimizer_groups(model)
    return [{"params": list(model.parameters()), "weight_decay": 0.01}]


def load_initial_weights(
    model: DauysHybrid, checkpoint: str | Path, *, from_v01: bool
) -> None:
    from .checkpoint import load_research_checkpoint

    payload = load_research_checkpoint(checkpoint)
    result = model.load_state_dict(payload["model"], strict=not from_v01)
    if from_v01:
        allowed = (
            "listening_norm.",
            "listening_cb0_head.",
            "listening_activity_head.",
            "activity_ahead_head.",
        )
        if result.unexpected_keys or any(
            not key.startswith(allowed) for key in result.missing_keys
        ):
            raise ValueError("R0 checkpoint is incompatible with v0.2 geometry")


def _batch_indices(
    records: list[CorpusRecord], plan: TrainingPlan
) -> list[list[tuple[int, int]]]:
    lookup = {id(record): index for index, record in enumerate(records)}
    batches = []
    for step in range(plan.steps):
        selected = sample_records(
            records,
            plan.stage,
            step / max(1, plan.steps - 1),
            plan.batch_size,
            plan.seed + step,
            replay_fraction=0.0 if plan.track in ("FT", "SFT") else 0.15,
            replay_offset=step * plan.batch_size,
            single_stream_only=plan.stage == "pretrain" and plan.track == "P",
        )
        batches.append(
            [
                (lookup[id(record)], plan.seed + step * plan.batch_size + item)
                for item, record in enumerate(selected)
            ]
        )
    return batches


def _durable_save(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    with temporary.open("rb") as stream:
        os.fsync(stream.fileno())
    temporary.replace(path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _durable_link(source: Path, target: Path) -> None:
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    os.link(source, temporary)
    temporary.replace(target)
    descriptor = os.open(target.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _truncate_metrics_to_step(path: Path, step: int) -> None:
    if not path.is_file():
        raise ValueError("Resume requires the existing metrics log")
    retained = [
        line
        for line in path.read_text(encoding="utf-8").splitlines()
        if line and int(json.loads(line)["step"]) <= step
    ]
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write("\n".join(retained) + ("\n" if retained else ""))
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _validate_evaluation(scores: dict[str, float], stage: Stage, track: Track) -> None:
    if stage == "pretrain":
        required = {
            "eval/pretrain/cont_word_rate",
            "eval/pretrain/cont_empty_rate",
            "eval/codebook/cb0_accuracy_gap",
        }
        if track == "PRE":
            required.add("eval/timing/heldout_1b_activity_f1")
    elif track == "P":
        required = {
            "eval/codebook/heldout_cb0_nll",
            "eval/codebook/cb0_accuracy_gap",
        }
    else:
        required = {
            "eval/codebook/e1_cb0_nll",
            "eval/conditioning/e2_matched_delta_cb0_onset",
            "eval/semantic/e2_wer",
            "eval/semantic/e2_relevance",
            "eval/semantic/e3_accuracy",
            "eval/semantic/e4_accuracy",
            "eval/timing/e2_onset_latency_ms",
            "eval/timing/e5_yield_within_500ms",
            "eval/timing/e6_backchannel_rate",
        }
    if not required <= scores.keys():
        raise ValueError(
            f"Evaluator omitted required metrics: {sorted(required - scores.keys())}"
        )
    if any(not math.isfinite(value) for value in scores.values()):
        raise ValueError("Evaluation contains non-finite metrics")


def _module_gradient_norms(model: DauysHybrid) -> dict[str, float]:
    totals = {
        name: 0.0 for name in ("listening", "dialogue", "depth", "heads", "shared")
    }
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        if name.startswith(("listening", "user_prediction")):
            group = "listening"
        elif name.startswith(("dialogue", "final_norm")):
            group = "dialogue"
        elif name.startswith("depth"):
            group = "depth"
        elif name.startswith(("cb0_head", "activity_head", "activity_ahead_head")):
            group = "heads"
        else:
            group = "shared"
        totals[group] += float(parameter.grad.detach().float().square().sum())
    return {
        f"train/gradient_norm/{key}": math.sqrt(value) for key, value in totals.items()
    }


@torch.inference_mode()
def own_history_rollout(
    model: DauysHybrid,
    batch: V02Batch,
    frames: int,
    seed: int,
) -> torch.Tensor:
    """Replace one bounded B span with free model output while preserving targets."""
    minimum = int(batch.valid_frames.sum(dim=1).min())
    if frames < 1 or minimum <= frames:
        return batch.output_codes
    start = random.Random(seed).randrange(minimum - frames + 1)
    end = start + frames
    was_training = model.training
    model.eval()
    try:
        # Mamba's streaming conv state is float32; a bf16 autocast step input
        # makes the fused causal-conv update reject the mixed dtypes.
        with torch.autocast(batch.user_codes.device.type, enabled=False):
            state = model.prefill_teacher(
                batch.user_codes[:, :, :start],
                batch.output_codes[:, :, :start],
                batch.initial_output_codes,
            )
            generated = [
                model.generate_step(batch.user_codes[:, :, index], state)[0]
                for index in range(start, end)
            ]
        history = batch.output_codes.clone()
        history[:, :, start:end] = torch.stack(generated, dim=-1)
        return history
    finally:
        model.train(was_training)


def train(
    model: DauysHybrid,
    records: list[CorpusRecord],
    silence_codes: torch.Tensor,
    plan: TrainingPlan,
    output_dir: str | Path,
    *,
    evaluator: EvaluationCallback | None,
    allow_unscored_smoke: bool = False,
    provenance: dict[str, str] | None = None,
    selection_baseline: dict[str, float] | None = None,
    log_callback: LogCallback | None = None,
    resume: bool = False,
) -> Path:
    plan.validate()
    if evaluator is None and not allow_unscored_smoke:
        raise ValueError("A frozen-suite evaluator is required for a research run")
    if (
        plan.stage in ("dialogue", "posttrain")
        and plan.track != "P"
        and not allow_unscored_smoke
        and selection_baseline is None
    ):
        raise ValueError("Dialogue selection requires a frozen baseline evaluation")
    if model.config.name != "dauys-hybrid-v0.2":
        raise ValueError("Training requires a Hybrid v0.2 model")
    torch.manual_seed(plan.seed)
    random.seed(plan.seed)
    device = next(model.parameters()).device
    toggles = toggles_for_track(plan.track, plan.stage)
    model.set_regularization(toggles.regularization)
    model.train()
    dataset = CorpusDataset(records, plan.context_frames)
    collate = partial(collate_cases, silence_codes=silence_codes.cpu())
    optimizer = torch.optim.AdamW(
        optimizer_groups_for_track(model, toggles),
        lr=plan.learning_rate,
    )
    warmup = max(1, math.ceil(plan.steps * 0.03))

    def lr_factor(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        return 0.5 * (
            1 + math.cos(math.pi * (step - warmup) / max(1, plan.steps - warmup))
        )

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    metadata = {
        "plan": asdict(plan),
        "model": asdict(model.config),
        "provenance": provenance or {},
        "records": len(records),
        "scored": evaluator is not None,
        "selection_baseline": selection_baseline,
    }
    metadata_file = root / "metadata.json"
    if resume:
        if not metadata_file.is_file() or not (root / "latest.pt").is_file():
            raise ValueError("Resume requires metadata and latest checkpoint")
        if json.loads(metadata_file.read_text()) != json.loads(json.dumps(metadata)):
            raise ValueError("Resume metadata differs from the original run")
    else:
        if metadata_file.exists() or (root / "metrics.jsonl").exists():
            raise FileExistsError("Run directory already contains a training run")
        metadata_file.write_text(json.dumps(metadata, indent=2) + "\n")
    steps_per_epoch = max(1, (len(records) + plan.batch_size - 1) // plan.batch_size)
    metrics_file = root / "metrics.jsonl"
    best_cb0_nll = float("inf")
    checkpoints_without_improvement = 0
    start_step = 0
    if resume:
        saved = torch.load(root / "latest.pt", map_location=device, weights_only=False)
        if saved["metadata"] != metadata:
            raise ValueError("Resume checkpoint metadata differs from the run")
        start_step = int(saved["step"])
        if not 0 <= start_step <= plan.steps:
            raise ValueError("Resume step is outside the remaining plan")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        best_cb0_nll = float(saved.get("best_cb0_nll", float("inf")))
        checkpoints_without_improvement = int(
            saved.get("checkpoints_without_improvement", 0)
        )
        torch.set_rng_state(saved["torch_rng"].cpu())
        random.setstate(saved["python_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
        if saved.get("stopped_early", False) or start_step == plan.steps:
            return root / "latest.pt"
        _truncate_metrics_to_step(metrics_file, start_step)
    batches = _batch_indices(records, plan)[start_step:]
    if plan.num_workers:
        loader = DataLoader(
            dataset,
            batch_sampler=batches,
            collate_fn=collate,
            num_workers=plan.num_workers,
            pin_memory=device.type == "cuda",
            persistent_workers=True,
            prefetch_factor=2,
            generator=torch.Generator().manual_seed(plan.seed),
        )
    else:
        loader = DataLoader(
            dataset,
            batch_sampler=batches,
            collate_fn=collate,
            generator=torch.Generator().manual_seed(plan.seed),
        )
    with metrics_file.open("a" if resume else "w", encoding="utf-8") as metrics_stream:
        for step, cpu_batch in enumerate(loader, start_step + 1):
            batch = cpu_batch.to(device)
            checkpoint_due = step % plan.checkpoint_interval == 0 or step == plan.steps
            weights = schedule((step - 1) / max(1, plan.steps - 1), plan.stage)
            if plan.track == "FT":
                weights = replace(weights, transition=3.0, history_dropout=0.1)
            history = None
            if (
                plan.stage in ("dialogue", "posttrain")
                and plan.track != "FT"
                and step > plan.own_history_after_epochs * steps_per_epoch
                and step % 2 == 0
            ):
                history = own_history_rollout(
                    model, batch, plan.own_history_frames, plan.seed + step
                )
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
            ):
                losses = objective(
                    model,
                    batch,
                    plan.stage,
                    weights,
                    toggles,
                    history_codes_override=history,
                    measure_delta=checkpoint_due,
                )
            losses.total.backward()
            module_norms = _module_gradient_norms(model) if checkpoint_due else {}
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            log = {
                "step": step,
                "train/total": float(losses.total.detach()),
                "train/cb0": float(losses.cb0.detach()),
                "train/depth": float(losses.depth.detach()),
                "train/activity": float(losses.activity.detach()),
                "train/listening_cb0": float(losses.listening_cb0.detach()),
                "train/ahead": float(losses.ahead.detach()),
                "train/contrast": float(losses.contrast.detach()),
                "train/gradient_norm": float(gradient_norm),
                "train/lr": optimizer.param_groups[0]["lr"],
            }
            if losses.delta_cb0_onset is not None:
                log["train/diagnostic_delta_cb0_onset_single_batch"] = float(
                    losses.delta_cb0_onset
                )
            if checkpoint_due:
                log["train/diagnostic_delta_available"] = float(
                    losses.delta_cb0_onset is not None
                )
            log.update(module_norms)
            metrics_stream.write(json.dumps(log) + "\n")
            metrics_stream.flush()
            if log_callback is not None:
                log_callback(log)
            if checkpoint_due:
                evaluation = None
                if evaluator is not None:
                    model.eval()
                    try:
                        evaluation = evaluator(model, step)
                    finally:
                        model.train()
                    if not allow_unscored_smoke:
                        _validate_evaluation(evaluation, plan.stage, plan.track)
                    metrics_stream.write(
                        json.dumps({"step": step, **evaluation}) + "\n"
                    )
                    metrics_stream.flush()
                    if log_callback is not None:
                        log_callback({"step": float(step), **evaluation})
                improved = False
                if evaluation is not None and plan.stage in ("dialogue", "posttrain"):
                    cb0_nll = evaluation.get(
                        "eval/codebook/e1_cb0_nll",
                        evaluation.get("eval/codebook/heldout_cb0_nll"),
                    )
                    if cb0_nll is not None and cb0_nll < best_cb0_nll:
                        eligible = selection_baseline is None or (
                            evaluation["eval/timing/e2_onset_latency_ms"]
                            <= selection_baseline["eval/timing/e2_onset_latency_ms"]
                            * 1.1
                            and evaluation["eval/timing/e5_yield_within_500ms"]
                            >= selection_baseline["eval/timing/e5_yield_within_500ms"]
                            * 0.9
                        )
                        if eligible:
                            best_cb0_nll = cb0_nll
                            improved = True
                    if plan.track == "SFT":
                        checkpoints_without_improvement = (
                            0 if improved else checkpoints_without_improvement + 1
                        )
                payload = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "step": step,
                    "metadata": metadata,
                    "evaluation": evaluation,
                    "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all()
                    if device.type == "cuda"
                    else [],
                    "python_rng": random.getstate(),
                    "best_cb0_nll": best_cb0_nll,
                    "checkpoints_without_improvement": checkpoints_without_improvement,
                    "stopped_early": (
                        plan.track == "SFT"
                        and checkpoints_without_improvement >= plan.early_stop_patience
                    ),
                }
                _durable_save(root / "latest.pt", payload)
                if improved:
                    _durable_link(root / "latest.pt", root / "best.pt")
                if (
                    plan.track == "SFT"
                    and checkpoints_without_improvement >= plan.early_stop_patience
                ):
                    break
    return root / "latest.pt"
