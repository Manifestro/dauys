"""Check E0b before any paid training: real sizes, causality, a few real steps.

    python scripts/v10_e0b_preflight.py --output CHECK

Run on the pod with the real Mamba-2. It fails if the two schemes are not within 5%
in parameters and rough FLOPs, or if future frames reach the model. It prints the
measured steps per second and scoring time, which give the time of the full run.
"""

import argparse
import json
import time
from pathlib import Path

import torch

from dauys.v10_e0b import SCHEMES, E0bConfig, E0bModel, flops_per_frame
from dauys.v10_e0b_data import (
    TEST_SPLITS,
    TRAIN_SPLIT,
    fetch_dataset,
    find_places,
    load_clips,
    make_batch,
    silence_tokens,
    window_plan,
)
from dauys.v10_e0b_eval import CAUSALITY_TOLERANCE, causality_error, evaluate_split

TOLERANCE = 0.05
STEPS = 20
BATCH_SIZE = 8


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.data_root or fetch_dataset()
    clips = load_clips(root)
    train = [clip for clip in clips if clip.split == TRAIN_SPLIT]
    silence, silent = silence_tokens(train)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    total_places = sum(
        len(find_places(clips, split)) for split in (TRAIN_SPLIT, *TEST_SPLITS)
    )
    report: dict = {
        "silence_token": silence,
        "silent_tokens": sorted(silent),
        "places": {
            split: len(find_places(clips, split))
            for split in (TRAIN_SPLIT, *TEST_SPLITS)
        },
    }
    for scheme in SCHEMES:
        torch.manual_seed(0)
        config = E0bConfig.for_scheme(scheme)
        model = E0bModel(config).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.0)
        losses = []
        started = time.time()
        for step in range(1, STEPS + 1):
            plan = window_plan(train, seed=0, step=step, batch_size=BATCH_SIZE)
            batch = make_batch(train, plan, seed=0, step=step)
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
            ):
                output = model(
                    batch["own"],
                    batch["partner"],
                    batch["own_active"],
                    batch["partner_active"],
                )
                loss = model.loss(output, batch)
            loss.total.backward()
            optimizer.step()
            losses.append(float(loss.own.detach()))
        seconds = time.time() - started
        model.eval()
        with torch.autocast(
            device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            error = causality_error(model, clips, silence)
            started = time.time()
            strangers = find_places(clips, "test_strangers")
            evaluate_split(model, clips, strangers, silence, silent, seed=0)
            eval_seconds = (time.time() - started) * total_places / len(strangers)
        report[scheme] = {
            "trainable_parameters": model.trainable_parameters(),
            "flops_per_frame_estimate": flops_per_frame(config),
            "steps_per_second": STEPS / seconds,
            "nll_first": losses[0],
            "nll_last": losses[-1],
            "causality_error": error,
            "seconds_per_scored_checkpoint": eval_seconds,
        }
        print(json.dumps({scheme: report[scheme]}), flush=True)
        if error > CAUSALITY_TOLERANCE:
            raise SystemExit(f"{scheme}: future frames reach the model: {error}")
    for name in ("trainable_parameters", "flops_per_frame_estimate"):
        ratio = report["rev1"][name] / report["rev2"][name]
        report[f"{name}_ratio"] = ratio
        if abs(ratio - 1) > TOLERANCE:
            raise SystemExit(f"{name}: rev1 / rev2 = {ratio:.3f}, over 5%")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "preflight.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
