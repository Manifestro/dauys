"""Train one scheme of V10 E0b and score it at every checkpoint.

    python scripts/v10_e0b_train.py --scheme rev2 --seed 0 --output RUN \
        --hf-repo manifestro/dauys-runs --hf-prefix e0b/rev2-s0

Both schemes run the same fixed number of steps on the same batches: nothing stops
early. The first step under the NLL level is recorded. Checkpoints resume from
``latest.pt``.
"""

import argparse
import dataclasses
import json
import math
import subprocess
import time
import uuid
from pathlib import Path

import airon
import torch

from dauys.v02_runner import _durable_save, _truncate_metrics_to_step
from dauys.v10_e0b import SCHEMES, E0bConfig, E0bModel, flops_per_frame
from dauys.v10_e0b_data import (
    REVISION,
    TEST_SPLITS,
    TRAIN_SPLIT,
    fetch_dataset,
    find_places,
    load_clips,
    make_batch,
    silence_tokens,
    window_plan,
)
from dauys.v10_e0b_eval import causality_error, echo_baseline, evaluate_split

STEPS = 12_000
BATCH_SIZE = 8
CHECKPOINTS = 10
NLL_LEVEL = 1.0
PEAK_LR = 3e-4
WARMUP_STEPS = 500
LR_FLOOR = 0.1
MAX_GRAD_NORM = 1.0


def _git() -> tuple[str, bool]:
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    dirty = bool(
        subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"], text=True
        ).strip()
    )
    return sha, dirty


def learning_rate(step: int, steps: int) -> float:
    """Linear warmup, then cosine decay to ``LR_FLOOR`` of the peak."""
    if step < WARMUP_STEPS:
        return PEAK_LR * (step + 1) / WARMUP_STEPS
    progress = (step - WARMUP_STEPS) / max(1, steps - WARMUP_STEPS)
    return PEAK_LR * (
        LR_FLOOR + (1 - LR_FLOOR) * 0.5 * (1 + math.cos(math.pi * progress))
    )


def _autocast(device: torch.device):
    return torch.autocast(
        device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scheme", choices=SCHEMES, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--hf-repo")
    parser.add_argument("--hf-prefix")
    args = parser.parse_args()
    if bool(args.hf_repo) != bool(args.hf_prefix):
        raise ValueError("Give both --hf-repo and --hf-prefix, or neither")
    if args.steps < CHECKPOINTS:
        raise ValueError("Every run needs its ten checkpoints")

    config = E0bConfig.for_scheme(args.scheme)
    root = args.output
    start_step = 0
    payload = None
    if args.resume:
        if not (root / "latest.pt").is_file():
            raise ValueError("Resume needs an existing latest.pt")
        payload = torch.load(root / "latest.pt", map_location="cpu", weights_only=True)
        if E0bConfig(**payload["config"]) != config or payload["seed"] != args.seed:
            raise ValueError("The checkpoint belongs to another scheme or seed")
        start_step = int(payload["step"])
        _truncate_metrics_to_step(root / "metrics.jsonl", start_step)
    elif (root / "metrics.jsonl").exists():
        raise FileExistsError(f"{root} already holds a run; use --resume")
    root.mkdir(parents=True, exist_ok=True)
    (root / "eval").mkdir(exist_ok=True)

    data_root = args.data_root or fetch_dataset()
    clips = load_clips(data_root)
    train_clips = [clip for clip in clips if clip.split == TRAIN_SPLIT]
    silence, silent = silence_tokens(train_clips)
    places = {split: find_places(clips, split) for split in (TRAIN_SPLIT, *TEST_SPLITS)}
    baseline = echo_baseline(clips, places, silence, silent)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    model = E0bModel(config).to(device)
    # No regularizer in this test, so no weight decay either.
    optimizer = torch.optim.AdamW(model.parameters(), lr=PEAK_LR, weight_decay=0.0)
    if payload is not None:
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])

    raw = root / "metadata.json"
    if args.resume:
        metadata = json.loads(raw.read_text(encoding="utf-8"))
    else:
        git_sha, dirty = _git()
        metadata = {
            "config": dataclasses.asdict(config),
            "seed": args.seed,
            "trainable_parameters": model.trainable_parameters(),
            "flops_per_frame_estimate": flops_per_frame(config),
            "dataset_revision": REVISION,
            "steps": args.steps,
            "batch_size": BATCH_SIZE,
            "learning_rate": PEAK_LR,
            "silence_token": silence,
            "silent_tokens": sorted(silent),
            "echo_baseline": baseline,
            "places": {split: len(group) for split, group in places.items()},
            "train_clips": [clip.file for clip in train_clips],
            "git": git_sha,
            "dirty": dirty,
            "airon_run_id": str(uuid.uuid4()),
            "first_step_under_nll_level": None,
        }
        raw.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "config": config.name,
                "trainable_parameters": model.trainable_parameters(),
                "steps": args.steps,
                "start_step": start_step,
                "device": str(device),
                "silence_token": silence,
                "places": metadata["places"],
                "echo_baseline": baseline,
            }
        ),
        flush=True,
    )
    tracked = airon.init(
        project="dauys",
        name=f"v10-e0b-{args.scheme}-s{args.seed}",
        config=metadata,
        experiment="v10-E0b",
        tags=["e0b", args.scheme, f"seed-{args.seed}"],
        total_steps=args.steps,
        run_id=metadata["airon_run_id"],
        resume="allow" if args.resume else "never",
        mode="disabled" if args.smoke else None,
    )

    every = args.steps // CHECKPOINTS
    frames_seen = 0
    started = time.time()
    model.train()
    with (root / "metrics.jsonl").open(
        "a" if args.resume else "w", encoding="utf-8"
    ) as stream:
        for step in range(start_step + 1, args.steps + 1):
            plan = window_plan(
                train_clips, seed=args.seed, step=step, batch_size=BATCH_SIZE
            )
            batch = make_batch(train_clips, plan, seed=args.seed, step=step)
            batch = {key: value.to(device) for key, value in batch.items()}
            rate = learning_rate(step - 1, args.steps)
            for group in optimizer.param_groups:
                group["lr"] = rate
            optimizer.zero_grad(set_to_none=True)
            with _autocast(device):
                output = model(
                    batch["own"],
                    batch["partner"],
                    batch["own_active"],
                    batch["partner_active"],
                )
                losses = model.loss(output, batch)
            losses.total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()
            frames_seen += int(batch["lengths"].sum())
            own_nll = float(losses.own.detach())
            if own_nll < NLL_LEVEL and metadata["first_step_under_nll_level"] is None:
                metadata["first_step_under_nll_level"] = step
                raw.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
            if step % args.log_interval == 0 or step == args.steps:
                elapsed = max(1e-9, time.time() - started)
                log = {
                    "step": float(step),
                    "train/loss": float(losses.total.detach()),
                    "train/nll_cb0_own": own_nll,
                    "train/nll_cb0_partner": float(losses.partner.detach()),
                    "train/activity": float(losses.activity.detach()),
                    "train/depth": float(losses.depth.detach()),
                    "train/lr": rate,
                    "train/frames_per_second": frames_seen / elapsed,
                }
                stream.write(json.dumps(log) + "\n")
                stream.flush()
                print(json.dumps(log), flush=True)
                tracked.log(
                    {key: value for key, value in log.items() if key != "step"},
                    step=step,
                    tokens=frames_seen,
                )
                frames_seen = 0
                started = time.time()
            if step % every == 0 or step == args.steps:
                state = {
                    "model": model.state_dict(),
                    "step": step,
                    "config": dataclasses.asdict(config),
                    "seed": args.seed,
                }
                _durable_save(root / f"checkpoint-{step:06d}.pt", state)
                _durable_save(
                    root / "latest.pt", {**state, "optimizer": optimizer.state_dict()}
                )
                scores: dict = {"step": step}
                model.eval()
                with _autocast(device):
                    scores["causality_error"] = causality_error(model, clips, silence)
                    for split, group in places.items():
                        scores[split] = evaluate_split(
                            model,
                            clips,
                            group,
                            silence,
                            silent,
                            seed=args.seed + step,
                            batch_size=args.eval_batch_size,
                        )
                model.train()
                (root / "eval" / f"step-{step:06d}.json").write_text(
                    json.dumps(scores) + "\n", encoding="utf-8"
                )
                summary = {
                    f"{split}/{name}": scores[split][name]["mean"]
                    for split in places
                    for name in ("g_silence", "g_voice", "own_history_gain")
                }
                summary |= {
                    f"{split}/nll": scores[split]["nll_real"] for split in places
                }
                summary |= {
                    f"{split}/echo": scores[split]["echo_percent"] for split in places
                }
                summary["causality_error"] = scores["causality_error"]
                print(json.dumps({"step": step, **summary}), flush=True)
                tracked.log(summary, step=step)
                frames_seen = 0
                started = time.time()
    tracked.finish()
    if args.hf_repo:
        from huggingface_hub import HfApi

        HfApi().upload_folder(
            repo_id=args.hf_repo,
            folder_path=str(root),
            path_in_repo=args.hf_prefix,
            ignore_patterns=["latest.pt"],
        )


if __name__ == "__main__":
    main()
