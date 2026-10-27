"""Preflight or train Hybrid v0.2 from immutable token manifests."""

import argparse
import importlib
import json
import subprocess
from pathlib import Path
from typing import Any

import torch

from dauys.config import ModelConfig
from dauys.model import DauysHybrid
from dauys.v02_data import assert_suite_is_disjoint, load_corpus, sha256
from dauys.v02_evaluation import Suite, load_suite
from dauys.v02_runner import TrainingPlan, load_initial_weights, train
from dauys.v02_wandb import start_private_run


def _suites(directory: Path) -> dict[str, Suite]:
    suites = {}
    for index in range(1, 7):
        name = f"E{index}"
        path = directory / f"{name}.json"
        digest = (directory / f"{name}.sha256").read_text().strip()
        suites[name] = load_suite(path, digest)
    return suites


def _silence(path: Path) -> torch.Tensor:
    if path.suffix == ".json":
        codes = torch.tensor(json.loads(path.read_text()), dtype=torch.long)
    else:
        saved = torch.load(path, map_location="cpu", weights_only=True)
        codes = (saved["codes"] if isinstance(saved, dict) else saved).long()
    if codes.shape != (8,) or ((codes < 0) | (codes >= 2048)).any():
        raise ValueError("Silence codes must be eight Mimi indices")
    return codes


def _evaluation_factory(
    specification: str, suites: dict[str, Suite], records: list
) -> Any:
    module_name, separator, function_name = specification.partition(":")
    if not separator:
        raise ValueError("Evaluator must be module:function")
    factory = getattr(importlib.import_module(module_name), function_name)
    evaluator = factory(suites=suites, records=records)
    if not callable(evaluator):
        raise TypeError("Evaluator factory must return a callable")
    return evaluator


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--companion", type=Path, required=True)
    parser.add_argument("--silence-codes", type=Path, required=True)
    parser.add_argument("--suites-dir", type=Path)
    parser.add_argument(
        "--track",
        choices=("FT", "P", "PRE", "S0", "S1", "S2", "S3", "S4", "S5", "SFT"),
        required=True,
    )
    parser.add_argument(
        "--stage", choices=("pretrain", "dialogue", "posttrain"), required=True
    )
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--context-frames", type=int, default=256)
    parser.add_argument("--checkpoint-interval", type=int, default=100)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--stage2-peak-learning-rate", type=float)
    parser.add_argument("--early-stop-patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--init-checkpoint", type=Path)
    parser.add_argument(
        "--evaluator", help="Python module:function returning a suite evaluator"
    )
    parser.add_argument("--baseline-metrics", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--unscored-smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()

    config = ModelConfig.from_yaml(args.config)
    if config.name != "dauys-hybrid-v0.2":
        raise ValueError("A v0.2 model configuration is required")
    records = load_corpus(args.cache_root, args.manifest, args.companion)
    silence_codes = _silence(args.silence_codes)
    suites = _suites(args.suites_dir) if args.suites_dir else {}
    for suite in suites.values():
        assert_suite_is_disjoint(records, list(suite.items))
    plan = TrainingPlan(
        track=args.track,
        stage=args.stage,
        steps=args.steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        checkpoint_interval=args.checkpoint_interval,
        num_workers=args.workers,
        seed=args.seed,
        context_frames=args.context_frames,
        stage2_peak_learning_rate=args.stage2_peak_learning_rate,
        early_stop_patience=args.early_stop_patience,
    )
    plan.validate()
    if args.track == "P" and not config.proxy:
        raise ValueError("Track P requires the proxy configuration")
    if args.track == "PRE" and config.proxy:
        raise ValueError("Full pre-training requires the full configuration")
    git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    git_status = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all"], text=True
    ).strip()
    provenance = {
        "config_sha256": sha256(args.config),
        "manifest_sha256": sha256(args.manifest),
        "companion_sha256": sha256(args.companion),
        "git_sha": git_sha,
        "git_dirty": str(bool(git_status)).lower(),
        **{f"{name}_sha256": suite.manifest_sha256 for name, suite in suites.items()},
    }
    print(
        json.dumps(
            {
                "records": len(records),
                "suites": sorted(suites),
                "plan": vars(args),
                "provenance": provenance,
            },
            default=str,
            indent=2,
        )
    )
    if args.preflight:
        return
    if git_status and not args.unscored_smoke:
        raise ValueError("Research training requires a clean Git checkout")
    if args.output is None:
        raise ValueError("Training requires an output directory")
    if args.track == "FT" and args.init_checkpoint is None and not args.resume:
        raise ValueError("Track FT requires the R0 checkpoint")
    if args.track == "SFT" and args.init_checkpoint is None and not args.resume:
        raise ValueError("Targeted SFT requires a selected Stage 2 checkpoint")
    if args.unscored_smoke and args.steps > 2:
        raise ValueError("An unscored smoke run is limited to two optimizer steps")
    proxy_track = args.track == "P"
    if not args.unscored_smoke and not args.evaluator:
        raise ValueError("Research training requires an evaluator")
    if not args.unscored_smoke and not proxy_track and not suites:
        raise ValueError("Research training requires E1–E6")
    baseline = (
        json.loads(args.baseline_metrics.read_text())
        if args.baseline_metrics is not None
        else None
    )
    if (
        not args.unscored_smoke
        and not proxy_track
        and args.stage in ("dialogue", "posttrain")
        and baseline is None
    ):
        raise ValueError("Dialogue training requires frozen R0 baseline metrics")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DauysHybrid(config).to(device)
    if args.init_checkpoint is not None and not args.resume:
        load_initial_weights(model, args.init_checkpoint, from_v01=args.track == "FT")
    evaluator = (
        _evaluation_factory(args.evaluator, suites, records) if args.evaluator else None
    )
    run = (
        start_private_run("dauys", f"v02-{args.track}-{args.seed}", provenance)
        if args.wandb
        else None
    )
    try:
        checkpoint = train(
            model,
            records,
            silence_codes,
            plan,
            args.output,
            evaluator=evaluator,
            allow_unscored_smoke=args.unscored_smoke,
            provenance=provenance,
            selection_baseline=baseline,
            log_callback=(
                (lambda metrics: run.log(metrics, step=int(metrics["step"])))
                if run is not None
                else None
            ),
            resume=args.resume,
        )
    finally:
        if run is not None:
            run.finish()
    print(json.dumps({"checkpoint": str(checkpoint)}))


if __name__ == "__main__":
    main()
