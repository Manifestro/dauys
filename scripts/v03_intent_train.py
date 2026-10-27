"""Stage 2 with the intent objective (step 1 of reports/intent-spec.md).

A separate loop next to scripts/v02_train.py. It reuses the v0.2 data, batch
plan, objective, own-history rollouts and held-out evaluator unchanged and adds
L_intent on top. --intent-weight 0 runs the control through the same loop.
"""

import argparse
import json
import math
import random
import subprocess
from dataclasses import asdict
from functools import partial
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dauys.config import ModelConfig
from dauys.model import DauysHybrid
from dauys.v02_data import CorpusDataset, collate_cases, load_corpus, sha256
from dauys.v02_proxy_dialogue_eval import create_heldout_dialogue_evaluator
from dauys.v02_runner import (
    TrainingPlan,
    _batch_indices,
    _durable_save,
    load_initial_weights,
    optimizer_groups_for_track,
    own_history_rollout,
    toggles_for_track,
)
from dauys.v02_training import objective, schedule, source_flags
from dauys.v02_wandb import start_private_run
from dauys.v03_intent import IntentConfig, IntentModule, speech_frame_scores


def _silence(path: Path) -> torch.Tensor:
    codes = torch.tensor(json.loads(path.read_text()), dtype=torch.long)
    if codes.shape != (8,) or ((codes < 0) | (codes >= 2048)).any():
        raise ValueError("Silence codes must be eight Mimi indices")
    return codes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--companion", type=Path, required=True)
    parser.add_argument("--silence-codes", type=Path, required=True)
    parser.add_argument("--init-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--context-frames", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--checkpoint-interval", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--intent-weight", type=float, default=1.0)
    parser.add_argument("--intent-horizon", type=int, default=12)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument(
        "--smoke", action="store_true", help="Allow a dirty checkout; no W&B"
    )
    args = parser.parse_args()

    config = ModelConfig.from_yaml(args.config)
    records = load_corpus(args.cache_root, args.manifest, args.companion)
    silence = _silence(args.silence_codes)
    plan = TrainingPlan(
        track="P",
        stage="dialogue",
        steps=args.steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        checkpoint_interval=args.checkpoint_interval,
        num_workers=args.workers,
        seed=args.seed,
        context_frames=args.context_frames,
    )
    plan.validate()
    intent_config = IntentConfig(horizon=args.intent_horizon, weight=args.intent_weight)
    intent_config.validate()
    git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all"], text=True
    ).strip()
    if dirty and not args.smoke:
        raise ValueError("Research training requires a clean Git checkout")
    provenance = {
        "config_sha256": sha256(args.config),
        "manifest_sha256": sha256(args.manifest),
        "init_checkpoint": str(args.init_checkpoint),
        "git_sha": git_sha,
        "git_dirty": str(bool(dirty)).lower(),
        "intent": json.dumps(asdict(intent_config), sort_keys=True),
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    model = DauysHybrid(config)
    load_initial_weights(model, args.init_checkpoint, from_v01=False)
    model = model.to(device)
    intent = IntentModule(model, intent_config).to(device)
    evaluator = create_heldout_dialogue_evaluator(suites={}, records=records)
    heldout = evaluator.heldout.records

    # Reseed so data order, dropout and rollouts match the v0.2 runner.
    torch.manual_seed(plan.seed)
    random.seed(plan.seed)
    toggles = toggles_for_track(plan.track, plan.stage)
    model.set_regularization(toggles.regularization)
    model.train()
    dataset = CorpusDataset(records, plan.context_frames)
    loader = DataLoader(
        dataset,
        batch_sampler=_batch_indices(records, plan),
        collate_fn=partial(collate_cases, silence_codes=silence),
        num_workers=plan.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=plan.num_workers > 0,
        prefetch_factor=2 if plan.num_workers else None,
        generator=torch.Generator().manual_seed(plan.seed),
    )
    groups = optimizer_groups_for_track(model, toggles)
    groups.append({"params": list(intent.head.parameters()), "weight_decay": 0.05})
    optimizer = torch.optim.AdamW(groups, lr=plan.learning_rate)
    warmup = max(1, math.ceil(plan.steps * 0.03))

    def lr_factor(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        return 0.5 * (
            1 + math.cos(math.pi * (step - warmup) / max(1, plan.steps - warmup))
        )

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    trainable = [p for group in groups for p in group["params"]]

    captured: dict[str, object] = {}
    model.register_forward_hook(lambda _m, _i, output: captured.update(output=output))

    root = args.output
    root.mkdir(parents=True, exist_ok=True)
    if (root / "metrics.jsonl").exists():
        raise FileExistsError("Run directory already contains a training run")
    metadata = {
        "plan": asdict(plan),
        "model": asdict(model.config),
        "intent": asdict(intent_config),
        "provenance": provenance,
    }
    (root / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    steps_per_epoch = max(1, (len(records) + plan.batch_size - 1) // plan.batch_size)
    run = (
        start_private_run(
            "dauys", f"v03-intent-w{args.intent_weight:g}-s{args.seed}", provenance
        )
        if args.wandb and not args.smoke
        else None
    )
    try:
        with (root / "metrics.jsonl").open("w", encoding="utf-8") as stream:
            for step, cpu_batch in enumerate(loader, 1):
                batch = cpu_batch.to(device)
                progress = (step - 1) / max(1, plan.steps - 1)
                checkpoint_due = (
                    step % plan.checkpoint_interval == 0 or step == plan.steps
                )
                weights = schedule(progress, plan.stage)
                history = None
                if (
                    step > plan.own_history_after_epochs * steps_per_epoch
                    and step % 2 == 0
                ):
                    history = own_history_rollout(
                        model, batch, plan.own_history_frames, plan.seed + step
                    )
                optimizer.zero_grad(set_to_none=True)
                captured.clear()
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
                    main_output = captured["output"]
                    scored = intent.loss(
                        main_output.dialogue_states,  # type: ignore[attr-defined]
                        batch.output_codes,
                        batch.valid_frames,
                        batch.output_activity,
                        source_flags(batch.sources, plan.stage, device)["b"],
                    )
                    total = losses.total + intent_config.weight * scored.loss
                total.backward()
                gradient_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                scheduler.step()
                intent.target.update(model, intent_config.ema_decay(progress))
                log = {
                    "step": step,
                    "train/total": float(total.detach()),
                    "train/cb0": float(losses.cb0.detach()),
                    "train/depth": float(losses.depth.detach()),
                    "train/activity": float(losses.activity.detach()),
                    "train/contrast": float(losses.contrast.detach()),
                    "train/intent": float(scored.loss.detach()),
                    "train/intent_cosine": float(scored.cosine.detach()),
                    "train/intent_prediction_std": float(
                        scored.prediction_std.detach()
                    ),
                    "train/gradient_norm": float(gradient_norm),
                    "train/lr": optimizer.param_groups[0]["lr"],
                }
                if checkpoint_due:
                    model.eval()
                    try:
                        evaluation = evaluator(model, step)
                        evaluation.update(
                            speech_frame_scores(model, intent, heldout, silence)
                        )
                    finally:
                        model.train()
                    log.update(evaluation)
                    _durable_save(
                        root / "latest.pt",
                        {
                            "model": model.state_dict(),
                            "intent": intent.state_dict(),
                            "step": step,
                            "metadata": metadata,
                            "evaluation": evaluation,
                        },
                    )
                stream.write(json.dumps(log) + "\n")
                stream.flush()
                if run is not None:
                    run.log(log, step=step)
    finally:
        if run is not None:
            run.finish()
    print(json.dumps({"checkpoint": str(root / "latest.pt")}))


if __name__ == "__main__":
    main()
