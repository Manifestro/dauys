"""Stateful stream helpers shared by training, evaluation and Stage 1."""

import random
from pathlib import Path

import torch
from torch.nn import functional as F

from .model import DauysHybrid
from .v02_training import masked_mean


def load_streams(path: str | Path) -> list[dict]:
    streams = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(streams, list) or not streams:
        raise ValueError("A stateful split must be a non-empty list of streams")
    return streams


def stack_streams(streams: list[dict]) -> dict[str, torch.Tensor]:
    return {
        key: torch.stack([stream[key] for stream in streams])
        for key in ("user", "target", "user_activity", "activity")
    }


def validate_streams(streams: list[dict], chunk: int) -> int:
    """Check one shared, chunk-divisible stream length."""
    if chunk < 1:
        raise ValueError("The chunk size must be positive")
    lengths = set()
    for stream in streams:
        user = stream["user"]
        if user.ndim != 2 or user.shape[0] != 8:
            raise ValueError("Stateful stream tensors must share [8, T] geometry")
        frames = int(user.shape[-1])
        if (
            stream["target"].shape != user.shape
            or stream["user_activity"].shape != (frames,)
            or stream["activity"].shape != (frames,)
        ):
            raise ValueError("Stateful stream tensors must share [8, T] geometry")
        lengths.add(frames)
    if len(lengths) != 1:
        raise ValueError("All stateful streams must have the same length")
    length = lengths.pop()
    if length < 1 or length % chunk:
        raise ValueError("Stream length must be divisible by the chunk size")
    return length


def _batch_to(batch: dict[str, torch.Tensor], device: torch.device) -> dict:
    return {key: value.to(device) for key, value in batch.items()}


def chunk_slice(
    batch: dict[str, torch.Tensor], chunk: int, index: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    start = index * chunk
    end = start + chunk
    initial = (
        batch["target"][:, :, 0] if index == 0 else batch["target"][:, :, start - 1]
    )
    return (
        batch["user"][:, :, start:end],
        batch["target"][:, :, start:end],
        batch["user_activity"][:, start:end],
        batch["activity"][:, start:end],
        initial,
    )


def chunk_loss(
    model: DauysHybrid,
    output,
    target: torch.Tensor,
    activity: torch.Tensor,
    label_smoothing: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, frames = target.shape[0], target.shape[-1]
    vocab = model.config.vocab_size
    speech = activity.bool()
    cb0_values = F.cross_entropy(
        output.cb0_logits.reshape(-1, vocab),
        target[:, 0].reshape(-1),
        reduction="none",
        label_smoothing=label_smoothing,
    ).reshape(batch, frames)
    assert output.depth_logits is not None
    depth_values = F.cross_entropy(
        output.depth_logits.reshape(-1, vocab),
        target[:, 1:].transpose(1, 2).reshape(-1),
        reduction="none",
    ).reshape(batch, frames, 7)
    activity_values = F.cross_entropy(
        output.activity_logits.reshape(-1, 2),
        activity.reshape(-1).long(),
        reduction="none",
    ).reshape(batch, frames)
    cb0 = masked_mean(cb0_values, speech)
    depth = masked_mean(depth_values.mean(-1), speech)
    activity_loss = masked_mean(
        activity_values, torch.ones_like(activity, dtype=torch.bool)
    )
    loss = cb0 + depth + 0.2 * activity_loss
    return loss, cb0, depth, activity_loss


@torch.inference_mode()
def chunk_nlls(
    model: DauysHybrid,
    streams: list[dict],
    *,
    chunk: int,
    batch_size: int = 16,
    stateful: bool,
) -> dict[int, float]:
    device = next(model.parameters()).device
    chunks = streams[0]["user"].shape[-1] // chunk
    totals = torch.zeros(chunks, dtype=torch.float64)
    counts = torch.zeros(chunks, dtype=torch.float64)
    vocab = model.config.vocab_size
    for start in range(0, len(streams), batch_size):
        batch = _batch_to(stack_streams(streams[start : start + batch_size]), device)
        states = {} if stateful else None
        for index in range(chunks):
            user, target, _, activity, initial = chunk_slice(batch, chunk, index)
            output = model(
                user,
                target,
                initial,
                mamba_states=states if stateful else None,
            )
            speech = activity.bool()
            values = F.cross_entropy(
                output.cb0_logits.reshape(-1, vocab),
                target[:, 0].reshape(-1),
                reduction="none",
            ).reshape(target.shape[0], target.shape[-1])
            totals[index] += float((values * speech).sum())
            counts[index] += float(speech.sum())
    return {
        index: float(totals[index] / counts[index].clamp_min(1))
        for index in range(chunks)
    }


def stateful_validation(
    model: DauysHybrid,
    heldout: list[dict],
    *,
    chunk: int,
    batch_size: int,
    prefix: str = "eval/",
    gain: bool = False,
) -> dict[str, float]:
    """Carry/reset CB0 NLL by chunk, optionally with the k2+ gain."""
    carry = chunk_nlls(
        model, heldout, chunk=chunk, batch_size=batch_size, stateful=True
    )
    reset = chunk_nlls(
        model, heldout, chunk=chunk, batch_size=batch_size, stateful=False
    )
    metrics = {
        f"{prefix}carry_cb0_k{index}": value for index, value in carry.items()
    } | {f"{prefix}reset_cb0_k{index}": value for index, value in reset.items()}
    metrics[f"{prefix}carry_cb0_mean"] = sum(carry.values()) / len(carry)
    metrics[f"{prefix}reset_cb0_mean"] = sum(reset.values()) / len(reset)
    if gain:
        indices = [index for index in carry if index >= 2]
        metrics[f"{prefix}gain"] = (
            sum(reset[index] - carry[index] for index in indices) / len(indices)
            if indices
            else 0.0
        )
    return metrics


class StreamCursor:
    """Synchronous stream rows with a persistent Mamba state across chunks."""

    def __init__(
        self,
        streams: list[dict],
        *,
        rows: int,
        chunk: int,
        seed: int,
        device: torch.device | None = None,
    ) -> None:
        if rows < 1:
            raise ValueError("A stream cursor needs a positive row count")
        if rows > len(streams):
            raise ValueError("The stream row count exceeds the split size")
        self.streams = streams
        self.rows = rows
        self.chunk = chunk
        self.device = device
        self.length = validate_streams(streams, chunk)
        self.chunks = self.length // chunk
        self.rng = random.Random(seed + 1)
        self.index = 0
        self.last_index = 0
        self.states: dict = {}
        self.batch: dict[str, torch.Tensor] = {}
        self._sample()

    def _sample(self) -> None:
        chosen = self.rng.sample(self.streams, self.rows)
        self.batch = _batch_to(
            stack_streams(chosen),
            self.device if self.device is not None else torch.device("cpu"),
        )
        self.index = 0
        self.last_index = 0
        self.states = {}

    def next_chunk(
        self,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        if self.index >= self.chunks:
            self._sample()
        self.last_index = self.index
        user, target, user_activity, activity, initial = chunk_slice(
            self.batch, self.chunk, self.index
        )
        self.index += 1
        return user, target, user_activity, activity, initial
