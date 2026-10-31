"""Marginal ``p(s)`` scoring for the V10 step 1 arms.

All estimators are batched: many items times K particles in one tensor pass,
frame by frame. The monologue is never taken from the stimulus; each particle
samples its own ``m_t`` from ``p(m_t | history)`` and pays ``log p(s_t | ...)``.
The stimuli, the 0.5 s silence lead and the summed frames follow the listening
bench in branch ``bench/duplex-listening``.
"""

import math
from collections.abc import Sequence

import torch
from torch.nn import functional as F

from .v10_step1 import V10Step1Model

LEAD_FRAMES = 6  # the bench's 0.5 s of silence before every single-stream item


def log_marginal(log_weights: torch.Tensor) -> torch.Tensor:
    """``logsumexp(log w) - log K``, never the mean of the logs."""
    return torch.logsumexp(log_weights, dim=-1) - math.log(log_weights.shape[-1])


def _log_prob(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    target = target.long().clamp(min=0)
    picked = F.log_softmax(logits.float(), dim=-1).gather(-1, target[..., None])
    return picked[..., 0]


def _frame_mask(
    frames: int, lengths: torch.Tensor, score_from: int, device: torch.device
) -> torch.Tensor:
    positions = torch.arange(frames, device=device)
    return (positions[None] >= score_from) & (positions[None] < lengths[:, None])


def systematic_resample(
    weights: torch.Tensor, generator: torch.Generator
) -> torch.Tensor:
    """Systematic resampling of one row of weights; returns local indices."""
    rows, particles = weights.shape
    offsets = torch.arange(particles, device=weights.device)
    draws = torch.rand(rows, 1, generator=generator, device=weights.device)
    positions = (draws + offsets[None, :]) / particles
    cumulative = weights.cumsum(dim=-1)
    cumulative[..., -1] = 1.0
    indices = torch.searchsorted(cumulative, positions, right=False)
    return indices.clamp(max=particles - 1)


@torch.inference_mode()
def marginal_log_ps(
    model: V10Step1Model,
    s: torch.Tensor,
    lengths: torch.Tensor,
    k: int,
    generator: torch.Generator,
    resample: bool = False,
    score_from: int = 0,
) -> torch.Tensor:
    """Estimate ``log p(s)`` from K sampled monologue paths.

    The estimate of ``p(s)`` is unbiased; its log reads slightly low.
    ``s`` is ``[items, frames]`` CB0 tokens; ``lengths`` gives the valid frames.
    Frames before ``score_from`` (the bench's silence lead) weight the paths like
    any other frame, and their own estimate is taken out at the end, so the result
    is ``log p(item | lead)``. ``resample`` turns the pass into a particle filter
    that resamples when the effective sample size falls below K/2.
    """
    if s.ndim != 2 or lengths.shape != (s.shape[0],):
        raise ValueError("Scores need [items, frames] tokens and [items] lengths")
    if k < 1:
        raise ValueError("At least one partition is needed")
    items, frames = s.shape
    device = s.device
    lengths = lengths.to(device).long()
    item_of = torch.arange(items, device=device).repeat_interleave(k)
    tokens = s[item_of]
    length = lengths[item_of]
    state = model.init_state(items * k)
    log_w = torch.zeros(items * k, device=device, dtype=torch.float32)
    lead = torch.zeros(items, device=device, dtype=torch.float32)
    carried = torch.zeros(items, device=device, dtype=torch.float32)
    m_prev = s.new_full((items * k,), model.config.m_bos_id)
    s_prev = s.new_full((items * k,), model.config.s_bos_id)
    zero = torch.zeros((), device=device, dtype=torch.float32)
    for frame in range(frames):
        active = length > frame
        if not bool(active.any()):
            break
        h, state = model.step(state, m_prev, s_prev)
        probs = F.softmax(model.head_m(h).float(), dim=-1)
        m_t = torch.multinomial(probs, 1, generator=generator)[:, 0]
        s_logits = model.head_s(h + model.emb_m(m_t))
        target = tokens[:, frame]
        log_p_s = _log_prob(s_logits, target)
        log_w = log_w + torch.where(active, log_p_s, zero)
        if frame == score_from - 1:
            lead = carried + log_marginal(log_w.view(items, k))
        m_prev = m_t
        s_prev = torch.where(active, target, s_prev)
        if resample:
            weights = log_w.view(items, k)
            log_ess = 2 * torch.logsumexp(weights, -1) - torch.logsumexp(
                2 * weights, -1
            )
            go = (log_ess.exp() < k / 2) & (lengths > frame + 1)
            if bool(go.any()):
                picked = go.nonzero()[:, 0]
                carried = carried + torch.where(
                    go,
                    torch.logsumexp(weights, -1) - math.log(k),
                    torch.zeros_like(carried),
                )
                draws = systematic_resample(F.softmax(weights[picked], -1), generator)
                index = torch.arange(items * k, device=device).view(items, k)
                index[picked] = picked[:, None] * k + draws
                index = index.reshape(-1)
                state = model.reindex_state(state, index)
                log_w = torch.where(
                    go.repeat_interleave(k), torch.zeros_like(log_w), log_w
                )
                m_prev = m_prev[index]
                s_prev = s_prev[index]
    return carried + log_marginal(log_w.view(items, k)) - lead


@torch.inference_mode()
def exact_log_ps_pad(
    model: V10Step1Model,
    s: torch.Tensor,
    lengths: torch.Tensor,
    score_from: int = 0,
) -> torch.Tensor:
    """Arm A: ``m`` is always PAD, so ``p(s)`` is exact, with no sampling."""
    m = torch.full_like(s, model.config.pad_id)
    output = model(m, s)
    values = _log_prob(output.s_logits, s)
    valid = _frame_mask(s.shape[1], lengths.to(s.device), score_from, s.device)
    return (values * valid).sum(dim=-1)


@torch.inference_mode()
def joint_log_pms(
    model: V10Step1Model,
    m: torch.Tensor,
    s: torch.Tensor,
    lengths: torch.Tensor,
    score_from: int = 0,
) -> torch.Tensor:
    """``log p(m, s)`` with a given monologue (the bench's "hint at test")."""
    output = model(m, s)
    values = _log_prob(output.m_logits, m) + _log_prob(output.s_logits, s)
    valid = _frame_mask(s.shape[1], lengths.to(s.device), score_from, s.device)
    return (values * valid).sum(dim=-1)


@torch.inference_mode()
def nll_s_given_true_m(
    model: V10Step1Model,
    m: torch.Tensor,
    s: torch.Tensor,
    lengths: torch.Tensor,
    score_from: int = 0,
) -> torch.Tensor:
    """Per-item mean NLL of ``s`` when the true monologue is fed: the leak."""
    output = model(m, s)
    values = _log_prob(output.s_logits, s)
    valid = _frame_mask(s.shape[1], lengths.to(s.device), score_from, s.device)
    count = valid.sum(dim=-1).clamp_min(1)
    return -(values * valid).sum(dim=-1) / count


def _generator(model: V10Step1Model, seed: int) -> torch.Generator:
    device = next(model.parameters()).device
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return generator


def pair_marginal_log_ps(
    model: V10Step1Model,
    good: torch.Tensor,
    bad: torch.Tensor,
    good_lengths: torch.Tensor,
    bad_lengths: torch.Tensor,
    *,
    k: int,
    seed: int,
    resample: bool = False,
    score_from: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Score both items of every pair with common random numbers."""
    generator = _generator(model, seed)
    state = generator.get_state()
    first = marginal_log_ps(
        model, good, good_lengths, k, generator, resample, score_from
    )
    generator.set_state(state)
    second = marginal_log_ps(
        model, bad, bad_lengths, k, generator, resample, score_from
    )
    return first, second


def pair_outcomes(
    good_scores: Sequence[float], bad_scores: Sequence[float]
) -> tuple[float, ...]:
    """Per pair: 1 for the right item, 0 for the wrong one, 0.5 for a tie."""
    if len(good_scores) != len(bad_scores) or not good_scores:
        raise ValueError("Pair scores must be non-empty and aligned")
    return tuple(
        1.0 if good > bad else 0.5 if good == bad else 0.0
        for good, bad in zip(good_scores, bad_scores)
    )


def accuracy(outcomes: Sequence[float]) -> float:
    if not outcomes:
        raise ValueError("Accuracy needs at least one pair")
    return sum(outcomes) / len(outcomes)


def normalized_scores(scores: Sequence[float], lengths: Sequence[int]) -> list[float]:
    """Per-frame mean log probability, the bench's ``accuracy_mean``."""
    return [
        float(score) / max(1, int(length)) for score, length in zip(scores, lengths)
    ]


def _answer(outcome: float) -> int:
    if outcome > 0.5:
        return 1
    return 0 if outcome < 0.5 else -1


def stability(outcomes_a: Sequence[float], outcomes_b: Sequence[float]) -> float:
    """Share of pairs whose answer flips between two seeds."""
    if len(outcomes_a) != len(outcomes_b) or not outcomes_a:
        raise ValueError("Stability needs aligned, non-empty outcomes")
    flips = sum(
        _answer(first) != _answer(second)
        for first, second in zip(outcomes_a, outcomes_b)
    )
    return flips / len(outcomes_a)
