"""Evaluate Mamba carry, reset, shuffled and foreign state on chunks."""

import argparse
import json
import random
from pathlib import Path

import torch
from torch.nn import functional as F

from dauys.checkpoint import load_research_checkpoint
from dauys.config import ModelConfig
from dauys.model import DauysHybrid
from dauys.v02_stateful import chunk_nlls, load_streams

MODES = ("carry", "reset", "shuffled", "foreign")
GAINS = {
    "gain_carry": ("reset", "carry"),
    "gain_shuffled": ("reset", "shuffled"),
    "gain_foreign": ("reset", "foreign"),
    "specific": ("foreign", "carry"),
    "order": ("shuffled", "carry"),
}


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


def aggregate_gains(values: list[float]) -> dict:
    if not values:
        return {
            "items": 0,
            "mean": None,
            "ci95_low": None,
            "ci95_high": None,
            "positive_rate": None,
        }
    low, high = bootstrap_ci(values, samples=1000, seed=0)
    return {
        "items": len(values),
        "mean": round(sum(values) / len(values), 4),
        "ci95_low": round(low, 4),
        "ci95_high": round(high, 4),
        "positive_rate": round(sum(value > 0 for value in values) / len(values), 4),
    }


def foreign_stream_index(
    streams: list[dict],
    index: int,
    *,
    min_frames: int | None = None,
) -> int:
    """Return the next stream by round-robin whose speaker differs."""
    if len(streams) < 2:
        raise ValueError("A foreign stream needs at least two streams")
    fallback = None
    for step in range(1, len(streams) + 1):
        candidate = (index + step) % len(streams)
        if streams[candidate]["speaker"] == streams[index]["speaker"]:
            continue
        if min_frames is None or streams[candidate]["user"].shape[-1] >= min_frames:
            return candidate
        if fallback is None:
            fallback = candidate
    if fallback is not None:
        return fallback
    raise ValueError("No stream with a different speaker is available")


def shuffled_order(k: int, seed: int, stream_index: int) -> list[int]:
    """Permutation of chunks 0..k-1 for one target chunk."""
    rng = random.Random(seed + stream_index + k)
    return rng.sample(range(k), k)


def _initial(stream: dict, chunk: int, index: int) -> torch.Tensor:
    if index == 0:
        return stream["target"][:, 0]
    return stream["target"][:, index * chunk - 1]


def _chunk_tensors(
    stream: dict, chunk: int, index: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    start, end = index * chunk, (index + 1) * chunk
    return (
        stream["user"][:, start:end][None].to(device),
        stream["target"][:, start:end][None].to(device),
        stream["activity"][start:end][None].to(device),
    )


@torch.inference_mode()
def _forward_chunk(model, stream, chunk, index, initial, states, device):
    user, target, _ = _chunk_tensors(stream, chunk, index, device)
    return model(
        user,
        target,
        initial[None].to(device),
        mamba_states=states,
    )


def _nll(output, target: torch.Tensor, activity: torch.Tensor, vocab: int) -> float:
    values = F.cross_entropy(
        output.cb0_logits.reshape(-1, vocab),
        target[:, 0].reshape(-1),
        reduction="none",
    ).reshape(target.shape[0], target.shape[-1])
    speech = activity.bool()
    return float((values * speech).sum() / speech.sum().clamp_min(1))


@torch.inference_mode()
def carry_row(model, stream, chunk, device) -> list[float]:
    states = {}
    row = []
    for index in range(stream["user"].shape[-1] // chunk):
        initial = _initial(stream, chunk, index)
        output = _forward_chunk(model, stream, chunk, index, initial, states, device)
        _, target, activity = _chunk_tensors(stream, chunk, index, device)
        row.append(_nll(output, target, activity, model.config.vocab_size))
    return row


@torch.inference_mode()
def mode_nlls(
    model,
    streams: list[dict],
    mode: str,
    *,
    chunk: int,
    seed: int,
) -> list[list[float]]:
    device = next(model.parameters()).device
    rows = []
    for stream_index, stream in enumerate(streams):
        chunks = stream["user"].shape[-1] // chunk
        if mode == "carry":
            rows.append(carry_row(model, stream, chunk, device))
            continue
        if mode == "reset":
            row = []
            for index in range(chunks):
                initial = _initial(stream, chunk, index)
                output = _forward_chunk(
                    model, stream, chunk, index, initial, None, device
                )
                _, target, activity = _chunk_tensors(stream, chunk, index, device)
                row.append(_nll(output, target, activity, model.config.vocab_size))
            rows.append(row)
            continue
        if mode == "shuffled":
            row = carry_row(model, stream, chunk, device)
            for index in range(2, chunks):
                states = {}
                initial = stream["target"][:, 0]
                for part in shuffled_order(index, seed, stream_index):
                    output = _forward_chunk(
                        model, stream, chunk, part, initial, states, device
                    )
                    del output
                    initial = stream["target"][:, (part + 1) * chunk - 1]
                output = _forward_chunk(
                    model, stream, chunk, index, initial, states, device
                )
                _, target, activity = _chunk_tensors(stream, chunk, index, device)
                row[index] = _nll(output, target, activity, model.config.vocab_size)
            rows.append(row)
            continue
        if mode == "foreign":
            other_index = foreign_stream_index(streams, stream_index)
            other = streams[other_index]
            other_chunks = other["user"].shape[-1] // chunk
            row = []
            for index in range(chunks):
                if index == 0:
                    initial = _initial(stream, chunk, 0)
                    output = _forward_chunk(
                        model, stream, chunk, 0, initial, {}, device
                    )
                    _, target, activity = _chunk_tensors(stream, chunk, 0, device)
                    row.append(_nll(output, target, activity, model.config.vocab_size))
                    continue
                states = {}
                initial = other["target"][:, 0]
                for part in range(min(index, other_chunks)):
                    output = _forward_chunk(
                        model, other, chunk, part, initial, states, device
                    )
                    del output
                    initial = other["target"][:, (part + 1) * chunk - 1]
                output = _forward_chunk(
                    model, stream, chunk, index, initial, states, device
                )
                _, target, activity = _chunk_tensors(stream, chunk, index, device)
                row.append(_nll(output, target, activity, model.config.vocab_size))
            rows.append(row)
            continue
        raise ValueError(f"Unknown mode {mode}")
    return rows


def summarize_modes(nll: dict[str, list[list[float]]]) -> dict:
    streams = len(nll["carry"])
    max_chunks = max(len(row) for row in nll["carry"])
    summary = {}
    for index in range(1, max_chunks):
        values = {name: [] for name in GAINS}
        for stream in range(streams):
            for name, (left, right) in GAINS.items():
                if left not in nll or right not in nll:
                    continue
                if index < len(nll[left][stream]) and index < len(nll[right][stream]):
                    values[name].append(
                        nll[left][stream][index] - nll[right][stream][index]
                    )
        summary[str(index)] = {
            name: aggregate_gains(found) for name, found in values.items() if found
        }
    averaged = {name: [] for name in GAINS}
    for stream in range(streams):
        for name, (left, right) in GAINS.items():
            if left not in nll or right not in nll:
                continue
            length = min(len(nll[left][stream]), len(nll[right][stream]))
            found = [
                nll[left][stream][index] - nll[right][stream][index]
                for index in range(2, length)
            ]
            if found:
                averaged[name].append(sum(found) / len(found))
    return {
        "by_chunk": summary,
        "k2plus": {
            name: aggregate_gains(found) for name, found in averaged.items() if found
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--heldout", type=Path, required=True)
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--chunk", type=int, default=256)
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = ModelConfig.from_yaml(args.config)
    streams = load_streams(args.heldout)
    modes = list(dict.fromkeys(args.modes))
    rows = []
    for spec in args.checkpoints:
        name, path = spec.split("=", 1)
        payload = load_research_checkpoint(Path(path))
        model = DauysHybrid(config)
        model.load_state_dict(payload["model"])
        model = model.cuda().eval()
        old_carry = chunk_nlls(
            model,
            streams,
            chunk=args.chunk,
            batch_size=args.batch,
            stateful=True,
        )
        old_reset = chunk_nlls(
            model,
            streams,
            chunk=args.chunk,
            batch_size=args.batch,
            stateful=False,
        )
        nll = {
            "carry": mode_nlls(
                model, streams, "carry", chunk=args.chunk, seed=args.seed
            ),
            "reset": mode_nlls(
                model, streams, "reset", chunk=args.chunk, seed=args.seed
            ),
        }
        for mode in ("shuffled", "foreign"):
            if mode in modes:
                nll[mode] = mode_nlls(
                    model,
                    streams,
                    mode,
                    chunk=args.chunk,
                    seed=args.seed,
                )
        row = {
            "checkpoint": name,
            "step": int(payload.get("step", 0)),
            "chunks": {
                str(index): {
                    "carry_nll": round(old_carry[index], 4),
                    "reset_nll": round(old_reset[index], 4),
                    "gain": round(old_reset[index] - old_carry[index], 4),
                }
                for index in sorted(old_carry)
            },
            "nll": {
                mode: [[round(value, 4) for value in row] for row in matrix]
                for mode, matrix in nll.items()
            },
            "streams": [
                {
                    "index": index,
                    "chapter": stream.get("chapter"),
                    "speaker": stream.get("speaker"),
                }
                for index, stream in enumerate(streams)
            ],
            "summary": summarize_modes(nll),
        }
        print(json.dumps(row), flush=True)
        rows.append(row)
    if args.output is not None:
        args.output.write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
