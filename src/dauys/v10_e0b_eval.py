"""V10 E0b metrics and the decision rules fixed in the brief.

All gaps are on the own channel's CB0, over the first 25 frames of replies that
start after a pause, with teacher forcing.
"""

import math
import random
from collections.abc import Iterable, Sequence

import torch
from torch.nn import functional as F

from .v10_e0b import E0bModel
from .v10_e0b_data import (
    CONDITIONS,
    SCORED_FRAMES,
    Clip,
    Place,
    build_condition,
    collate_windows,
)

DETECTION_GAP = 0.5
EARLIER_CHECKPOINTS = 2
STRONGER_RATIO = 1.5
ECHO_MARGIN_POINTS = 1.0
ECHO_NGRAM = 4
MIN_ECHO_BASELINE_CLIPS = 10
BASELINE_LEAK_PCT = 1.0
FLAGGED_LEAK_PCT = 2.0
CAUSALITY_TOLERANCE = 1e-3


def _batches(items: Sequence, size: int) -> Iterable[Sequence]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _to_device(batch: dict[str, torch.Tensor], device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


@torch.no_grad()
def place_nll(
    model: E0bModel,
    clips: list[Clip],
    places: list[Place],
    condition: str,
    silence: int,
    *,
    batch_size: int = 16,
) -> torch.Tensor:
    """Mean NLL of the own CB0 over the scored frames, one value per place."""
    device = next(model.parameters()).device
    values = []
    for group in _batches(places, batch_size):
        batch = collate_windows(
            [
                build_condition(clips[place.clip], place, condition, silence)
                for place in group
            ]
        )
        batch = _to_device(batch, device)
        output = model(
            batch["own"], batch["partner"], batch["own_active"], batch["partner_active"]
        )
        for row, t0 in enumerate(batch["t0"].tolist()):
            logits = output.own_logits[row, t0 : t0 + SCORED_FRAMES]
            target = batch["own"][row, t0 : t0 + SCORED_FRAMES]
            values.append(F.cross_entropy(logits.float(), target).cpu())
    return torch.stack(values) if values else torch.zeros(0)


def ngrams(tokens: Sequence[int], n: int = ECHO_NGRAM) -> list[tuple[int, ...]]:
    return [tuple(tokens[index : index + n]) for index in range(len(tokens) - n + 1)]


def echo_counts(
    spoken: Sequence[int], heard: Sequence[int], silent: frozenset[int]
) -> tuple[int, int]:
    """Spoken n-grams also heard, and all spoken n-grams, without those of silence."""
    said = [gram for gram in ngrams(spoken) if not set(gram) <= silent]
    pool = {gram for gram in ngrams(heard) if not set(gram) <= silent}
    return sum(gram in pool for gram in said), len(said)


def _echo_percent(counts: Iterable[tuple[int, int]]) -> float:
    hits = total = 0
    for hit, count in counts:
        hits += hit
        total += count
    return 100 * hits / total if total else float("nan")


def real_echo(
    clips: list[Clip], places: list[Place], silence: int, silent: frozenset[int]
) -> float:
    """The echo level of the data itself, in percent."""
    counts = []
    for place in places:
        window = build_condition(clips[place.clip], place, "real", silence)
        t0 = int(window["t0"])
        spoken = window["own"][t0 : t0 + SCORED_FRAMES].tolist()
        counts.append(echo_counts(spoken, window["partner"].tolist(), silent))
    return _echo_percent(counts)


@torch.no_grad()
def sample_replies(
    model: E0bModel,
    clips: list[Clip],
    places: list[Place],
    silence: int,
    *,
    seed: int,
    batch_size: int = 16,
    temperature: float = 1.0,
) -> list[list[int]]:
    """Own CB0 sampled over the scored frames, with the true partner and timing."""
    device = next(model.parameters()).device
    generator = torch.Generator(device=device).manual_seed(seed)
    replies: list[list[int]] = []
    for group in _batches(places, batch_size):
        batch = collate_windows(
            [
                build_condition(clips[place.clip], place, "real", silence)
                for place in group
            ]
        )
        batch = _to_device(batch, device)
        own = batch["own"].clone()
        rows = torch.arange(len(group), device=device)
        for offset in range(SCORED_FRAMES):
            frame = batch["t0"] + offset
            logits = model(
                own, batch["partner"], batch["own_active"], batch["partner_active"]
            ).own_logits[rows, frame]
            probabilities = (logits.float() / temperature).softmax(-1)
            own[rows, frame] = torch.multinomial(probabilities, 1, generator=generator)[
                :, 0
            ]
        for row, t0 in enumerate(batch["t0"].tolist()):
            replies.append(own[row, t0 : t0 + SCORED_FRAMES].tolist())
    return replies


def model_echo(
    clips: list[Clip],
    places: list[Place],
    replies: list[list[int]],
    silence: int,
    silent: frozenset[int],
) -> float:
    counts = []
    for place, reply in zip(places, replies):
        window = build_condition(clips[place.clip], place, "real", silence)
        counts.append(echo_counts(reply, window["partner"].tolist(), silent))
    return _echo_percent(counts)


@torch.no_grad()
def causality_error(
    model: E0bModel, clips: list[Clip], silence: int, *, frames: int = 256
) -> float:
    """How much the outputs up to frame ``k`` move when frames after ``k`` change.

    The direct test that the partner's future cannot reach the model. Frame ``k`` of
    the input is seen first at output ``k + 1``, so outputs up to ``k`` must not move.
    """
    device = next(model.parameters()).device
    clip = max(clips, key=lambda item: item.frames)
    frames = min(frames, clip.frames)
    own = clip.codes[0][0, :frames][None].to(device)
    partner = clip.codes[1][0, :frames][None].to(device)
    own_active = clip.active[0][:frames][None].to(device)
    partner_active = clip.active[1][:frames][None].to(device)
    cut = frames // 2
    base = model(own, partner, own_active, partner_active)
    changed = partner.clone()
    changed[:, cut + 1 :] = silence
    active = partner_active.clone()
    active[:, cut + 1 :] = ~active[:, cut + 1 :]
    other = model(own, changed, own_active, active)
    return float(
        (base.own_logits[:, : cut + 1] - other.own_logits[:, : cut + 1])
        .abs()
        .max()
        .float()
    )


def bootstrap_by_episode(
    values: Sequence[float], episodes: Sequence[str], *, reps: int = 2000, seed: int = 0
) -> dict[str, float]:
    """Mean with a 95% interval, resampling whole episodes."""
    grouped: dict[str, list[float]] = {}
    for value, episode in zip(values, episodes):
        grouped.setdefault(episode, []).append(float(value))
    if not grouped:
        return {"mean": math.nan, "low": math.nan, "high": math.nan, "episodes": 0}
    names = list(grouped)
    rng = random.Random(seed)
    means = []
    for _ in range(reps):
        picked = [
            value
            for _unused in names
            for value in grouped[names[rng.randrange(len(names))]]
        ]
        means.append(sum(picked) / len(picked))
    means.sort()
    flat = [value for group in grouped.values() for value in group]
    return {
        "mean": sum(flat) / len(flat),
        "low": means[int(0.025 * (reps - 1))],
        "high": means[int(0.975 * (reps - 1))],
        "episodes": len(names),
    }


def evaluate_split(
    model: E0bModel,
    clips: list[Clip],
    places: list[Place],
    silence: int,
    silent: frozenset[int],
    *,
    seed: int,
    batch_size: int = 16,
) -> dict:
    """Every metric of one split at one checkpoint, with the per-place values."""
    was_training = model.training
    model.eval()
    usable = [
        place
        for place in places
        if int(clips[place.clip].active[1 - place.role].sum()) >= 2
    ]
    nll = {
        condition: place_nll(
            model, clips, usable, condition, silence, batch_size=batch_size
        )
        for condition in CONDITIONS
    }
    gaps = {
        "g_silence": nll["silence"] - nll["real"],
        "g_voice": nll["same_voice"] - nll["real_spliced"],
        "own_history_gain": nll["own_erased"] - nll["real"],
        "shift_change": nll["shifted"] - nll["real"],
    }
    episodes = [clips[place.clip].episode for place in usable]
    feeds = [clips[place.clip].feed for place in usable]
    flagged = [clips[place.clip].leak_pct >= FLAGGED_LEAK_PCT for place in usable]
    replies = sample_replies(
        model, clips, usable, silence, seed=seed, batch_size=batch_size
    )
    result: dict = {
        "places": len(usable),
        "nll_real": float(nll["real"].mean()) if usable else math.nan,
        "echo_percent": model_echo(clips, usable, replies, silence, silent),
        "echo_percent_data": real_echo(clips, usable, silence, silent),
    }
    for name, values in gaps.items():
        result[name] = bootstrap_by_episode(values.tolist(), episodes)
    silence_gap = gaps["g_silence"].tolist()
    result["g_silence_by_feed"] = {
        feed: _mean(value for value, own in zip(silence_gap, feeds) if own == feed)
        for feed in sorted(set(feeds))
    }
    result["g_silence_leak_2_to_5"] = _mean(
        value for value, mark in zip(silence_gap, flagged) if mark
    )
    result["g_silence_leak_under_2"] = _mean(
        value for value, mark in zip(silence_gap, flagged) if not mark
    )
    result["per_place"] = {
        "clip": [clips[place.clip].file for place in usable],
        "role": [place.role for place in usable],
        "t0": [place.t0 for place in usable],
        **{name: values.tolist() for name, values in gaps.items()},
    }
    if was_training:
        model.train()
    return result


def _mean(values: Iterable[float]) -> float:
    items = list(values)
    return sum(items) / len(items) if items else math.nan


def echo_baseline(
    clips: list[Clip],
    places_by_split: dict[str, list[Place]],
    silence: int,
    silent: frozenset[int],
) -> dict:
    """The temporary echo baseline: clean clips only, or none if under ten of them."""
    clean = {
        index for index, clip in enumerate(clips) if clip.leak_pct < BASELINE_LEAK_PCT
    }
    places = [
        place
        for group in places_by_split.values()
        for place in group
        if place.clip in clean
    ]
    used = len({place.clip for place in places})
    return {
        "clips": len(clean),
        "clips_with_places": used,
        "places": len(places),
        "percent": (
            real_echo(clips, places, silence, silent)
            if len(clean) >= MIN_ECHO_BASELINE_CLIPS
            else None
        ),
    }


def detection_checkpoint(curve: Sequence[float]) -> int | None:
    """The index of the first checkpoint with G-silence at the detection level."""
    for index, value in enumerate(curve):
        if value >= DETECTION_GAP:
            return index
    return None


def decide_question_2(
    rev2_gap: Sequence[float],
    rev1_gap: Sequence[float],
    rev2_history: Sequence[float],
    rev1_history: Sequence[float],
) -> dict:
    """The table of question 2, read top to bottom on the training curves."""
    if len(rev2_gap) != len(rev1_gap) or not rev2_gap:
        raise ValueError("Both schemes need the same checkpoints")
    rev2_at = detection_checkpoint(rev2_gap)
    rev1_at = detection_checkpoint(rev1_gap)
    facts = {"rev2_detected_at": rev2_at, "rev1_detected_at": rev1_at}
    never = len(rev2_gap)
    rev2_when = never if rev2_at is None else rev2_at
    rev1_when = never if rev1_at is None else rev1_at
    if rev2_at is not None and rev2_when <= rev1_when and rev2_gap[-1] >= rev1_gap[-1]:
        return {
            **facts,
            "outcome": "rev2_first_and_stronger",
            "action": "rev2 is the base; E2 is cut, cross only as a guard if E7 fails",
        }
    if (
        rev1_at is not None
        and rev2_when - rev1_when >= EARLIER_CHECKPOINTS
        and rev1_gap[-1] >= STRONGER_RATIO * rev2_gap[-1]
    ):
        return {
            **facts,
            "outcome": "rev1_earlier_and_stronger",
            "action": "E2 runs in full; the burden of proof stays on rev1",
        }
    history_grows = (
        rev2_history[-1] > rev2_history[0] or rev1_history[-1] > rev1_history[0]
    )
    if rev2_at is None and rev1_at is None and history_grows:
        return {
            **facts,
            "outcome": "neither_detects_history_grows",
            "action": "E2 is lowered; E5 and E6 go first after step 1",
        }
    return {**facts, "outcome": "other", "action": "rev2; E2 stays as it is"}


def check_question_1(
    *,
    train_loss_first: float,
    train_loss_last: float,
    echo_percent: float,
    echo_baseline_percent: float | None,
    shift_change: float,
    causality: float,
    kept_leak_max_pct: float,
) -> dict:
    """The parts of question 1 a script can check; listening stays with a person.

    Decoded speech and the roles on five clips are checked by hand.
    """
    echo_ok = (
        echo_percent <= echo_baseline_percent + ECHO_MARGIN_POINTS
        if echo_baseline_percent is not None
        else kept_leak_max_pct <= 5.0
    )
    checks = {
        "loss_falls": train_loss_last < train_loss_first,
        "echo_within_baseline": echo_ok,
        "shift_makes_nll_worse": shift_change > 0,
        "future_frames_do_not_reach_the_model": causality <= CAUSALITY_TOLERANCE,
    }
    return {**checks, "passed": all(checks.values())}
