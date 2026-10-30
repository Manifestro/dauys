"""Model-free arithmetic for the listening bench: pairing, n-gram echo, CIs."""

import math
import random
from collections.abc import Sequence

import torch


def bootstrap_ci(
    values: Sequence[float], *, reps: int = 2000, seed: int = 0
) -> dict[str, float | int]:
    """Mean with a 95% percentile bootstrap interval over the given units."""
    clean = [float(value) for value in values if not math.isnan(float(value))]
    if not clean:
        return {"mean": float("nan"), "low": float("nan"), "high": float("nan"), "n": 0}
    rng = random.Random(seed)
    count = len(clean)
    means = sorted(
        sum(clean[rng.randrange(count)] for _ in range(count)) / count
        for _ in range(reps)
    )
    return {
        "mean": sum(clean) / count,
        "low": means[int(0.025 * (reps - 1))],
        "high": means[int(0.975 * (reps - 1))],
        "n": count,
    }


def length_matched_partners(lengths: Sequence[int]) -> list[int]:
    """Give every item a different partner of similar length.

    Items are sorted by length and paired with the next one in that order
    (the last with the first), so no item is its own partner and the swapped
    partner channel needs little cropping or padding.
    """
    if len(lengths) < 2:
        raise ValueError("Swapping partners needs at least two conversations")
    order = sorted(range(len(lengths)), key=lambda index: (lengths[index], index))
    partners = [0] * len(lengths)
    for rank, index in enumerate(order):
        partners[index] = order[(rank + 1) % len(order)]
    return partners


def fit_length(codes: torch.Tensor, frames: int, silence: torch.Tensor) -> torch.Tensor:
    """Crop [K, T] codes to ``frames`` or pad them with the silence frame."""
    if codes.shape[-1] >= frames:
        return codes[:, :frames]
    pad = silence[:, None].expand(-1, frames - codes.shape[-1]).to(codes)
    return torch.cat((codes, pad), dim=-1)


def ngrams(tokens: Sequence[int], n: int) -> list[tuple[int, ...]]:
    return [tuple(tokens[index : index + n]) for index in range(len(tokens) - n + 1)]


def echo_rate(
    spoken: Sequence[int],
    heard: Sequence[int],
    *,
    n: int = 4,
    ignore: frozenset[tuple[int, ...]] = frozenset(),
) -> float:
    """Share of the model's spoken n-grams that also occur in what it heard.

    N-grams in ``ignore`` (for example runs of the codec's silence token) are
    left out on both sides; NaN when the model said nothing countable.
    """
    said = [gram for gram in ngrams(spoken, n) if gram not in ignore]
    if not said:
        return float("nan")
    pool = set(ngrams(heard, n)) - ignore
    return sum(gram in pool for gram in said) / len(said)


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0))


def pair_accuracy(good: Sequence[float], bad: Sequence[float]) -> float:
    """Share of pairs where the right item scores higher; ties count half."""
    if len(good) != len(bad) or not good:
        raise ValueError("Pair scores must be non-empty and aligned")
    wins = sum(1.0 if g > b else 0.5 if g == b else 0.0 for g, b in zip(good, bad))
    return wins / len(good)


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> float:
    """Mean of ``values`` over ``mask``; NaN when the mask is empty."""
    picked = values[mask & ~torch.isnan(values)]
    return float(picked.mean()) if picked.numel() else float("nan")
