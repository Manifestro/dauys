"""Stage 1 training loop from the pre-registered recipe.

The loop is intentionally narrow: A→B continuation on speech frames, a
same-chapter contrastive intent head, the pre-registered early-abort rule and
one best checkpoint by held-out speech-frame CB0 NLL. Bands and the GO/FIX/
STOP outcome live in ``dauys.v02_stage1_eval``.
"""

import argparse
import dataclasses
import json
import math
import random
import subprocess
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dauys.backbone_transfer import (
    load_language_model_state,
    transfer_mamba2_backbone,
)
from dauys.checkpoint import load_research_checkpoint
from dauys.config import ModelConfig
from dauys.model import DauysHybrid
from dauys.v02_data import sha256
from dauys.v02_runner import (
    _durable_save,
    _truncate_metrics_to_step,
    load_initial_weights,
    optimizer_groups,
)
from dauys.v02_stage1 import (
    CROP_FRAMES,
    Stage1BatchSource,
    assert_stage1_pools,
    intent_spread,
    load_batch,
    load_stage1_corpus,
    make_batch_spec,
    stage1_objective,
    stage1_weights,
)
from dauys.v02_stage1_eval import (
    Cb0Bigram,
    assert_speakers_disjoint,
    chapter_control,
    continuation_waves,
    early_abort,
    empty_rate,
    fusion_dependence,
    gate_a_bands,
    intent_selection,
    overfit_gap,
    save_wavs,
    speaker_probe,
    speech_cb0_runs,
    speech_frame_scores,
)
from dauys.v02_wandb import start_private_run
from dauys.v03_intent import IntentConfig, IntentModule


def _spread(records, limit: int):
    """Round-robin by speaker so the overfit sample stays speaker-diverse."""
    if limit < 1:
        raise ValueError("A spread sample needs a positive limit")
    by_speaker: dict[str, list] = {}
    for record in records:
        by_speaker.setdefault(record.speaker, []).append(record)
    keys = sorted(by_speaker)
    queues = {key: list(by_speaker[key]) for key in keys}
    chosen = []
    while len(chosen) < limit:
        progressed = False
        for key in keys:
            if queues[key]:
                chosen.append(queues[key].pop(0))
                progressed = True
                if len(chosen) >= limit:
                    break
        if not progressed:
            break
    if not chosen:
        raise ValueError("The spread sample is empty")
    return chosen


def _grouped(records, per_group: int, limit: int):
    """Take whole speaker-chapter groups so M3 always has eight candidates."""
    if per_group < 1 or limit < 1:
        raise ValueError("A grouped sample needs positive sizes")
    by_chapter: dict[tuple[str, str], list] = {}
    for record in records:
        key = (record.speaker, record.chapter or record.speaker)
        by_chapter.setdefault(key, []).append(record)
    groups = [items for items in by_chapter.values() if len(items) >= per_group]
    if not groups:
        raise ValueError("No held-out speaker-chapter group has enough phrases")
    chosen = []
    for items in groups:
        chosen.extend(items[:per_group])
        if len(chosen) >= limit:
            break
    return chosen


def _silence(path: Path) -> torch.Tensor:
    codes = torch.tensor(json.loads(path.read_text()), dtype=torch.long)
    if codes.shape != (8,):
        raise ValueError("Silence codes must be eight Mimi indices")
    return codes


def _git() -> tuple[str, bool]:
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    dirty = bool(
        subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=all"], text=True
        ).strip()
    )
    return sha, dirty


def _evaluate(
    model: DauysHybrid,
    intent: IntentModule,
    heldout,
    silence: torch.Tensor,
    bigram: Cb0Bigram,
    heldout_runs,
    *,
    batch_size: int,
    train_sample,
) -> dict[str, float]:
    on = speech_frame_scores(model, heldout, silence, batch_size=batch_size)
    reference = bigram.nll(heldout_runs)
    metrics = {
        "m1_cb0_nll": on["nll"],
        "m1_cb0_accuracy": on["accuracy"],
        "m2_bigram_nll": reference,
        "m1_minus_m2": on["nll"] - reference,
        "m3": intent_selection(model, intent, heldout, silence),
        "m3_chapter_control": chapter_control(model, intent, heldout, silence),
        "m4": fusion_dependence(model, heldout, silence, batch_size=batch_size),
        "m6": overfit_gap(model, train_sample, heldout, silence, batch_size=batch_size),
    }
    spec = make_batch_spec(
        heldout,
        0,
        seed=11,
        speakers=4,
        anchors=4,
        pool=8,
        chapter_only=True,
    )
    probe = load_batch(heldout, spec, silence).to(next(model.parameters()).device)
    metrics["intent_spread"] = intent_spread(model, probe, intent)
    return metrics


def _final_evaluation(
    model: DauysHybrid,
    heldout,
    silence: torch.Tensor,
    args,
    device: torch.device,
    best_metrics: dict[str, float] | None,
) -> dict:
    from dauys.codec import load_mimi

    codec = load_mimi(args.mimi_checkpoint, device)
    transcribe = None
    if args.asr_model:
        from dauys.v02_pretrain_eval import WhisperTranscriber

        transcribe = WhisperTranscriber(args.asr_model, str(device))
    with torch.inference_mode():
        waves = continuation_waves(
            model, heldout, silence, codec, items=args.eval_items
        )
        m5 = empty_rate(waves, transcribe)
    # The probe trains its own classifier, so it cannot run under inference_mode.
    try:
        m7 = speaker_probe(model, heldout)
    except ValueError:
        m7 = float("nan")
    result: dict = {
        "m5_empty_rate": m5,
        "m7_speaker_probe": m7,
        "listening_samples": len(waves[:8]),
    }
    if args.listen_dir is not None:
        save_wavs(waves[:8], args.listen_dir)
    if best_metrics is not None and args.m8_words is not None:
        gate = gate_a_bands(
            {
                "m1_minus_m2": best_metrics["m1_minus_m2"],
                "m3": best_metrics["m3"],
                "m4": best_metrics["m4"],
                "m5": result["m5_empty_rate"],
                "m6": best_metrics["m6"],
                "m8": float(args.m8_words),
            }
        )
        result["gate"] = {"bands": gate.bands, "outcome": gate.outcome}
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--heldout-root", type=Path, required=True)
    parser.add_argument("--heldout-manifest", type=Path, required=True)
    parser.add_argument("--silence-codes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--init-backbone",
        default="state-spaces/mamba2-130m",
        help="Dialogue Mamba-2 backbone; pass an empty string to disable",
    )
    parser.add_argument("--init-checkpoint", type=Path)
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--speakers", type=int, default=16)
    parser.add_argument("--anchors", type=int, default=4)
    parser.add_argument("--pool", type=int, default=8)
    parser.add_argument("--crop-frames", type=int, default=CROP_FRAMES)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--checkpoint-interval", type=int, default=2_000)
    parser.add_argument("--eval-records", type=int, default=256)
    parser.add_argument("--train-sample", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mimi-checkpoint", type=Path)
    parser.add_argument("--asr-model")
    parser.add_argument("--eval-items", type=int, default=100)
    parser.add_argument("--listen-dir", type=Path)
    parser.add_argument("--m8-words", type=int)
    parser.add_argument(
        "--history-dropout",
        type=float,
        default=0.0,
        help="Share of items whose B input after the cut is replaced by silence",
    )
    parser.add_argument("--intent-weight", type=float, default=0.5)
    parser.add_argument(
        "--listening-horizon",
        type=int,
        default=1,
        help="C1 predicts the A CB0 code this many frames ahead",
    )
    parser.add_argument(
        "--listening-scale",
        type=float,
        default=1.0,
        help="Multiplies the C1 weight schedule (spec M4 fix: 0.5)",
    )
    parser.add_argument(
        "--acoustic-dropout",
        type=float,
        default=0.5,
        help="Acoustic dropout on A (spec M4 fix: 0.3)",
    )
    parser.add_argument(
        "--ignore-abort",
        nargs="*",
        default=[],
        help="Early-abort reasons this run deliberately overrides; recorded in metadata",
    )
    parser.add_argument("--run-name", default="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--require-clean", action="store_true")
    args = parser.parse_args()

    if args.m8_words is not None and not 0 <= args.m8_words <= 8:
        raise ValueError("M8 is the count of recognisable items out of eight")
    config = ModelConfig.from_yaml(args.config)
    records = load_stage1_corpus(args.cache_root, args.manifest)
    heldout = _grouped(
        load_stage1_corpus(args.heldout_root, args.heldout_manifest),
        per_group=args.pool,
        limit=args.eval_records,
    )
    assert_speakers_disjoint(records, heldout)
    assert_stage1_pools(records, speakers=args.speakers, pool=args.pool)
    assert_stage1_pools(heldout, speakers=4, pool=args.pool, chapter_only=True)
    silence = _silence(args.silence_codes)
    git_sha, dirty = _git()
    if dirty and args.require_clean and not args.smoke:
        raise ValueError("This run requires a clean checkout")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    model = DauysHybrid(config)
    if args.init_backbone and not args.init_checkpoint and not args.resume:
        pairs = transfer_mamba2_backbone(
            model, load_language_model_state(args.init_backbone)
        )
        print(json.dumps({"init_backbone": args.init_backbone, "layers": pairs}))
    if args.init_checkpoint and not args.resume:
        load_initial_weights(model, args.init_checkpoint, from_v01=False)
    model = model.to(device)
    print(json.dumps({"trainable_parameters": model.trainable_parameters()}))
    intent_config = IntentConfig(horizon=12, weight=args.intent_weight)
    intent = IntentModule(model, intent_config).to(device)
    if args.init_checkpoint and not args.resume:
        initial = load_research_checkpoint(args.init_checkpoint)
        if "intent" in initial:
            intent.load_state_dict(initial["intent"])
    groups = optimizer_groups(model, args.weight_decay)
    groups.append(
        {"params": list(intent.head.parameters()), "weight_decay": args.weight_decay}
    )
    optimizer = torch.optim.AdamW(groups, lr=args.learning_rate)
    trainable = [parameter for group in groups for parameter in group["params"]]
    warmup = max(1, math.ceil(args.steps * 0.03))

    def lr_factor(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        return 0.5 * (
            1 + math.cos(math.pi * (step - warmup) / max(1, args.steps - warmup))
        )

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    root = args.output
    root.mkdir(parents=True, exist_ok=True)
    if not args.resume and (root / "metrics.jsonl").exists():
        raise FileExistsError("Run directory already contains a run")
    start_step = 0
    best = float("inf")
    best_step = 0
    if args.resume:
        latest = root / "latest.pt"
        if not latest.is_file():
            raise FileNotFoundError("Resume requires latest.pt")
        payload = load_research_checkpoint(latest)
        model.load_state_dict(payload["model"])
        intent.load_state_dict(payload["intent"])
        optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["scheduler"])
        start_step = int(payload["step"])
        best = float(payload.get("best", float("inf")))
        best_step = int(payload.get("best_step", start_step))
        torch.set_rng_state(payload["torch_rng"].cpu())
        _truncate_metrics_to_step(root / "metrics.jsonl", start_step)
    source = Stage1BatchSource(
        records,
        silence,
        steps=args.steps,
        seed=args.seed,
        speakers=args.speakers,
        anchors=args.anchors,
        pool=args.pool,
        crop_frames=args.crop_frames,
        start=start_step,
    )
    loader = DataLoader(
        source,
        batch_size=None,
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
        prefetch_factor=2 if args.workers else None,
    )
    train_sample = _spread(records, args.train_sample)
    bigram = Cb0Bigram.fit(speech_cb0_runs(records))
    heldout_runs = speech_cb0_runs(heldout)
    provenance = {
        "git_sha": git_sha,
        "git_dirty": str(dirty).lower(),
        "config_sha256": sha256(args.config),
        "manifest_sha256": sha256(args.manifest),
        "heldout_manifest_sha256": sha256(args.heldout_manifest),
        "init_backbone": args.init_backbone or "none",
        "init_checkpoint": str(args.init_checkpoint or "none"),
    }
    metadata = {
        "plan": {
            "steps": args.steps,
            "speakers": args.speakers,
            "anchors": args.anchors,
            "pool": args.pool,
            "crop_frames": args.crop_frames,
            "learning_rate": args.learning_rate,
            "seed": args.seed,
            "history_dropout": args.history_dropout,
            "intent_weight": args.intent_weight,
            "listening_horizon": args.listening_horizon,
            "listening_scale": args.listening_scale,
            "acoustic_dropout": args.acoustic_dropout,
            "ignore_abort": list(args.ignore_abort),
        },
        "model": asdict(config),
        "intent": asdict(intent_config),
        "provenance": provenance,
        "records": len(records),
        "heldout": len(heldout),
    }
    if not args.resume:
        (root / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    run = (
        start_private_run(
            "dauys", f"v02-stage1{args.run_name}-s{args.seed}", provenance
        )
        if args.wandb and not args.smoke
        else None
    )
    aborted: str | None = None
    best_metrics: dict[str, float] | None = None
    mode = "a" if args.resume else "w"
    with (root / "metrics.jsonl").open(mode, encoding="utf-8") as stream:
        try:
            for step, batch in enumerate(loader, start_step + 1):
                batch = batch.to(device)
                progress = (step - 1) / max(1, args.steps - 1)
                base = stage1_weights(progress)
                weights = dataclasses.replace(
                    base,
                    listening=base.listening * args.listening_scale,
                    acoustic_dropout=args.acoustic_dropout,
                    intent=args.intent_weight,
                    history_dropout=args.history_dropout,
                    listening_horizon=args.listening_horizon,
                )
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(
                    device.type,
                    dtype=torch.bfloat16,
                    enabled=device.type == "cuda",
                ):
                    losses = stage1_objective(model, batch, weights, intent)
                losses.total.backward()
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                scheduler.step()
                intent.target.update(model, intent_config.ema_decay(progress))
                log: dict[str, float] = {
                    "step": float(step),
                    "train/total": float(losses.total.detach()),
                    "train/cb0": float(losses.cb0.detach()),
                    "train/depth": float(losses.depth.detach()),
                    "train/activity": float(losses.activity.detach()),
                    "train/listening_cb0": float(losses.listening_cb0.detach()),
                    "train/intent": float(losses.intent.detach()),
                    "train/intent_accuracy": float(losses.intent_accuracy.detach()),
                    "train/lr": optimizer.param_groups[0]["lr"],
                }
                due = step % args.checkpoint_interval == 0 or step == args.steps
                if due:
                    model.eval()
                    with torch.inference_mode():
                        metrics = _evaluate(
                            model,
                            intent,
                            heldout,
                            silence,
                            bigram,
                            heldout_runs,
                            batch_size=args.speakers * args.anchors,
                            train_sample=train_sample,
                        )
                    model.train()
                    log.update(metrics)
                    reason = early_abort(step, metrics)
                    if reason in args.ignore_abort:
                        log["overridden_abort"] = 1.0
                        print(json.dumps({"step": step, "overridden_abort": reason}))
                        reason = None
                    aborted = aborted or reason
                    if metrics["m1_cb0_nll"] < best:
                        best = metrics["m1_cb0_nll"]
                        best_step = step
                        best_metrics = dict(metrics)
                        _durable_save(
                            root / "best.pt",
                            {
                                "model": model.state_dict(),
                                "intent": intent.state_dict(),
                                "step": step,
                                "metadata": metadata,
                                "metrics": metrics,
                            },
                        )
                    _durable_save(
                        root / "latest.pt",
                        {
                            "model": model.state_dict(),
                            "intent": intent.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "scheduler": scheduler.state_dict(),
                            "step": step,
                            "metadata": metadata,
                            "metrics": metrics,
                            "best": best,
                            "best_step": best_step,
                            "torch_rng": torch.get_rng_state(),
                        },
                    )
                stream.write(json.dumps(log) + "\n")
                stream.flush()
                if run is not None:
                    run.log(log, step=step)
                if aborted is not None:
                    (root / "abort.json").write_text(
                        json.dumps({"step": step, "reason": aborted}) + "\n"
                    )
                    break
        finally:
            if run is not None:
                run.finish()
    summary: dict = {
        "best_cb0_nll": best,
        "best_step": best_step,
        "aborted": aborted,
        "checkpoint": str(root / "best.pt"),
    }
    if args.mimi_checkpoint is not None:
        best_file = root / "best.pt"
        if best_file.is_file():
            payload = load_research_checkpoint(best_file)
            model.load_state_dict(payload["model"])
            intent.load_state_dict(payload["intent"])
            best_metrics = payload.get("metrics", best_metrics)
            best_step = int(payload.get("step", best_step))
        summary.update(
            _final_evaluation(model, heldout, silence, args, device, best_metrics)
        )
        summary["gate_step"] = best_step
    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
