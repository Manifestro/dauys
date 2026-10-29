"""Evaluate semantic memory: control-minus-same CB0 NLL after filler."""

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import torch
from torch.nn import functional as F

from dauys.checkpoint import load_research_checkpoint
from dauys.config import ModelConfig
from dauys.model import DauysHybrid

sys.path.insert(0, str(Path(__file__).parent))
from v02_stateful_train import chunk_slice, load_streams, stack_streams


def bootstrap_ci(
    values: list[float], samples: int = 1000, seed: int = 0
) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    rng = random.Random(seed)
    size = len(values)
    means = sorted(
        sum(values[rng.randrange(size)] for _ in range(size)) / size
        for _ in range(samples)
    )
    return means[int(0.025 * samples)], means[int(0.975 * samples)]


@torch.inference_mode()
def probe_nlls(
    model: DauysHybrid,
    streams: list[dict],
    *,
    chunk: int,
    batch_size: int = 16,
    stateful: bool,
) -> list[float]:
    device = next(model.parameters()).device
    vocab = model.config.vocab_size
    results = [float("nan")] * len(streams)
    by_length: dict[int, list[int]] = defaultdict(list)
    for index, stream in enumerate(streams):
        by_length[int(stream["user"].shape[-1])].append(index)
    for length, indices in by_length.items():
        if length % chunk:
            raise ValueError("Probe stream length must be divisible by chunk")
        chunks = length // chunk
        for start in range(0, len(indices), batch_size):
            selected = indices[start : start + batch_size]
            batch = stack_streams([streams[index] for index in selected])
            batch["score"] = torch.stack(
                [streams[index]["score"] for index in selected]
            )
            batch = {key: value.to(device) for key, value in batch.items()}
            states = {} if stateful else None
            sums = torch.zeros(len(selected), device=device)
            counts = torch.zeros(len(selected), device=device)
            for index in range(chunks):
                user, target, _, _activity, initial = chunk_slice(batch, chunk, index)
                output = model(
                    user,
                    target,
                    initial,
                    mamba_states=states if stateful else None,
                )
                values = F.cross_entropy(
                    output.cb0_logits.reshape(-1, vocab),
                    target[:, 0].reshape(-1),
                    reduction="none",
                ).reshape(target.shape[0], target.shape[-1])
                score = batch["score"][:, index * chunk : (index + 1) * chunk].bool()
                sums += (values * score).sum(dim=1)
                counts += score.sum(dim=1)
            for offset, stream_index in enumerate(selected):
                results[stream_index] = float(
                    sums[offset] / counts[offset].clamp_min(1)
                )
    return results


def deltas_by_gap(streams: list[dict], nlls: list[float]) -> dict[int, list[float]]:
    lookup = {
        (stream["item"], int(stream["gap"]), stream["kind"]): nlls[index]
        for index, stream in enumerate(streams)
    }
    deltas: dict[int, list[float]] = defaultdict(list)
    keys = sorted({(stream["item"], int(stream["gap"])) for stream in streams})
    for item, gap in keys:
        same = lookup.get((item, gap, "same"))
        control = lookup.get((item, gap, "control"))
        if same is None or control is None:
            continue
        deltas[gap].append(control - same)
    return deltas


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--chunk", type=int, default=256)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = ModelConfig.from_yaml(args.config)
    streams = load_streams(args.data)
    rows = []
    for spec in args.checkpoints:
        name, path = spec.split("=", 1)
        payload = load_research_checkpoint(Path(path))
        model = DauysHybrid(config)
        model.load_state_dict(payload["model"])
        model = model.cuda().eval()
        for mode in ("carry", "reset"):
            nlls = probe_nlls(
                model,
                streams,
                chunk=args.chunk,
                batch_size=args.batch,
                stateful=mode == "carry",
            )
            deltas = deltas_by_gap(streams, nlls)
            for gap in sorted(deltas):
                values = deltas[gap]
                low, high = bootstrap_ci(values, samples=1000, seed=0)
                row = {
                    "checkpoint": name,
                    "mode": mode,
                    "gap": gap,
                    "items": len(values),
                    "mean_delta": round(sum(values) / len(values), 4),
                    "ci95_low": round(low, 4) if low is not None else None,
                    "ci95_high": round(high, 4) if high is not None else None,
                    "positive_rate": round(
                        sum(value > 0 for value in values) / len(values), 4
                    ),
                }
                print(json.dumps(row), flush=True)
                rows.append(row)
    args.output.write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
