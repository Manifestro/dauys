"""V10 E0b data: the separated podcast clips, training windows and test substitutions.

The set is fixed by the ``split`` field at one Hugging Face revision. Windows depend
only on the seed and step, so both schemes see the same batches. Left and right are
random per clip and speakers are not linked across clips, so "the same speaker" can
only mean the same channel of the same clip.
"""

import json
import math
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .v10_e0b import ACTIVITY_HORIZONS, DEPTH_CODEBOOKS

DATASET = "manifestro/hohere-e0b-duplexchat"
REVISION = "e6b82f8dbc973276bd573705a4286221da6d37ac"
TRAIN_SPLIT = "train"
TEST_SPLITS = ("test_familiar", "test_strangers")
KEPT_SPLITS = (TRAIN_SPLIT, "train_reserve", *TEST_SPLITS)
FRAME_RATE = 12.5
WINDOW_FRAMES = (750, 1125)  # 60-90 s
PAUSE_FRAMES = math.ceil(0.5 * FRAME_RATE)
SCORED_FRAMES = 25
CONTEXT_FRAMES = 1000
DEPTH_FRAMES_PER_BATCH = 2048
SILENCE_COVERAGE = 0.9


@dataclass(frozen=True)
class Clip:
    file: str
    split: str
    feed: str
    episode: str
    leak_pct: float
    codes: tuple[torch.Tensor, torch.Tensor]  # left, right: int64 [8, T]
    active: tuple[torch.Tensor, torch.Tensor]  # left, right: bool [T]

    @property
    def frames(self) -> int:
        return int(self.codes[0].shape[-1])


def fetch_dataset() -> Path:
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            DATASET,
            repo_type="dataset",
            revision=REVISION,
            allow_patterns=["manifest.jsonl", "*/*.npz"],
        )
    )


def load_clips(root: Path, splits: tuple[str, ...] = KEPT_SPLITS) -> list[Clip]:
    clips = []
    with (root / "manifest.jsonl").open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle]
    for row in rows:
        if row["split"] not in splits:
            continue
        with np.load(root / row["file"]) as data:
            clips.append(
                Clip(
                    file=row["file"],
                    split=row["split"],
                    feed=row["feed"],
                    episode=row["episode"],
                    leak_pct=float(row["leak_pct"]),
                    codes=(
                        torch.from_numpy(data["codes_left"].astype(np.int64)),
                        torch.from_numpy(data["codes_right"].astype(np.int64)),
                    ),
                    active=(
                        torch.from_numpy(data["active_left"]),
                        torch.from_numpy(data["active_right"]),
                    ),
                )
            )
    return clips


def silence_tokens(clips: list[Clip]) -> tuple[int, frozenset[int]]:
    """The CB0 token of silence here, and the tokens of most silent frames.

    Taken from the channels' own silent frames: after separation they do not hold
    the token of digital silence (1049), which the model would never have heard.
    """
    counts: Counter[int] = Counter()
    for clip in clips:
        for codes, active in zip(clip.codes, clip.active):
            counts.update(codes[0, ~active].tolist())
    total = sum(counts.values())
    if not total:
        raise ValueError("No silent frames to take the silence token from")
    covered = 0
    tokens = []
    for token, count in counts.most_common():
        tokens.append(token)
        covered += count
        if covered >= SILENCE_COVERAGE * total:
            break
    return tokens[0], frozenset(tokens)


@dataclass(frozen=True)
class Window:
    clip: int
    role: int  # the index of the model's own channel: 0 left, 1 right
    start: int
    frames: int


def window_plan(
    clips: list[Clip], *, seed: int, step: int, batch_size: int
) -> list[Window]:
    """The windows of one step, a pure function of the seed and step."""
    rng = random.Random(f"e0b:{seed}:{step}")
    weights = [clip.frames for clip in clips]
    plan = []
    for _ in range(batch_size):
        index = rng.choices(range(len(clips)), weights)[0]
        frames = min(rng.randint(*WINDOW_FRAMES), clips[index].frames)
        start = rng.randint(0, clips[index].frames - frames)
        plan.append(Window(index, rng.randrange(2), start, frames))
    return plan


def _activity_targets(
    clip: Clip, window: Window, length: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Own and partner activity at each horizon; labels may come from past the window."""
    target = torch.zeros(length, 2 * len(ACTIVITY_HORIZONS), dtype=torch.bool)
    valid = torch.zeros_like(target)
    channels = (clip.active[window.role], clip.active[1 - window.role])
    for h_index, horizon in enumerate(ACTIVITY_HORIZONS):
        for c_index, active in enumerate(channels):
            column = 2 * h_index + c_index
            first = window.start + horizon - 1
            last = min(first + window.frames, clip.frames)
            count = max(0, last - first)
            target[:count, column] = active[first:last]
            valid[:count, column] = True
    return target, valid


def make_batch(
    clips: list[Clip], plan: list[Window], *, seed: int, step: int
) -> dict[str, torch.Tensor]:
    length = max(window.frames for window in plan)
    size = len(plan)
    own = torch.zeros(size, length, dtype=torch.long)
    partner = torch.zeros_like(own)
    own_active = torch.zeros(size, length, dtype=torch.bool)
    partner_active = torch.zeros_like(own_active)
    prefix = torch.zeros(size, length, DEPTH_CODEBOOKS, dtype=torch.long)
    depth_target = torch.zeros_like(prefix)
    activity_target = torch.zeros(
        size, length, 2 * len(ACTIVITY_HORIZONS), dtype=torch.bool
    )
    activity_valid = torch.zeros_like(activity_target)
    lengths = torch.zeros(size, dtype=torch.long)
    for row, window in enumerate(plan):
        clip = clips[window.clip]
        span = slice(window.start, window.start + window.frames)
        mine = clip.codes[window.role][:, span]
        own[row, : window.frames] = mine[0]
        partner[row, : window.frames] = clip.codes[1 - window.role][0, span]
        own_active[row, : window.frames] = clip.active[window.role][span]
        partner_active[row, : window.frames] = clip.active[1 - window.role][span]
        prefix[row, : window.frames] = mine[:DEPTH_CODEBOOKS].T
        depth_target[row, : window.frames] = mine[1:].T
        target, valid = _activity_targets(clip, window, length)
        activity_target[row] = target
        activity_valid[row] = valid
        lengths[row] = window.frames
    generator = torch.Generator().manual_seed(seed * 1_000_003 + step)
    frames = torch.cat(
        [
            torch.stack((torch.full((n,), row), torch.arange(n)), dim=1)
            for row, n in enumerate(lengths.tolist())
        ]
    )
    picks = torch.randperm(len(frames), generator=generator)[:DEPTH_FRAMES_PER_BATCH]
    return {
        "own": own,
        "partner": partner,
        "own_active": own_active,
        "partner_active": partner_active,
        "lengths": lengths,
        "activity_target": activity_target,
        "activity_valid": activity_valid,
        "depth_prefix": prefix,
        "depth_target": depth_target,
        "depth_frames": frames[picks],
    }


@dataclass(frozen=True)
class Place:
    """The start of a reply the metrics score, as ``t0`` in ``clip`` from ``role``."""

    clip: int
    role: int
    t0: int


def reply_starts(own: torch.Tensor, partner: torch.Tensor) -> list[int]:
    """Own silent for >= 0.5 s, the partner spoke in that pause, 25 frames ahead.

    The rule hohere used for the place counts in the brief (236, 84, 24).
    """
    return [
        t
        for t in range(PAUSE_FRAMES, own.numel() - SCORED_FRAMES)
        if own[t]
        and not own[t - PAUSE_FRAMES : t].any()
        and partner[t - PAUSE_FRAMES : t].any()
    ]


def find_places(clips: list[Clip], split: str) -> list[Place]:
    places = []
    for index, clip in enumerate(clips):
        if clip.split != split:
            continue
        for role in range(2):
            places += [
                Place(index, role, t0)
                for t0 in reply_starts(clip.active[role], clip.active[1 - role])
            ]
    return places


def splice_timed(
    speech_of: torch.Tensor,
    active_of: torch.Tensor,
    slots: torch.Tensor,
    silence: int,
    offset: int,
) -> torch.Tensor:
    """Speech frames, rotated by ``offset``, placed in ``slots``; silence elsewhere."""
    speech = speech_of[active_of]
    out = torch.full_like(speech_of, silence)
    where = slots.nonzero().flatten()
    if len(speech) and len(where):
        out[where] = speech[(torch.arange(len(where)) + offset) % len(speech)]
    return out


CONDITIONS = ("real", "silence", "real_spliced", "same_voice", "own_erased", "shifted")


def build_condition(
    clip: Clip, place: Place, condition: str, silence: int
) -> dict[str, torch.Tensor]:
    """One test window ending ``SCORED_FRAMES`` past ``t0``, with one channel changed.

    ``same_voice`` fills the partner's speech slots with other speech of the same
    channel and ``real_spliced`` puts the real speech back through the same cut, so
    both have the same pauses and the same seams at their borders. A fragment of
    ``same_voice`` can also cross a pause of its source; the real one cannot.
    """
    own_cb0 = clip.codes[place.role][0]
    partner_cb0 = clip.codes[1 - place.role][0]
    own_active = clip.active[place.role]
    partner_active = clip.active[1 - place.role]
    if condition == "silence":
        partner_cb0 = torch.full_like(partner_cb0, silence)
        partner_active = torch.zeros_like(partner_active)
    elif condition == "real_spliced":
        partner_cb0 = splice_timed(
            partner_cb0, partner_active, partner_active, silence, 0
        )
    elif condition == "same_voice":
        count = int(partner_active.sum())
        if count < 2:
            raise ValueError("A partner without speech cannot be replaced by itself")
        partner_cb0 = splice_timed(
            partner_cb0, partner_active, partner_active, silence, count // 2
        )
    elif condition == "own_erased":
        own_cb0 = own_cb0.clone()
        own_active = own_active.clone()
        own_cb0[: place.t0] = silence
        own_active[: place.t0] = False
    elif condition == "shifted":
        # With the frame shift of the model this feeds the partner's same frame.
        partner_cb0 = torch.cat((partner_cb0[1:], partner_cb0.new_tensor([silence])))
        partner_active = torch.cat((partner_active[1:], partner_active.new_zeros(1)))
    elif condition != "real":
        raise ValueError(f"Unknown test condition {condition}")
    start = max(0, place.t0 - CONTEXT_FRAMES)
    end = place.t0 + SCORED_FRAMES
    return {
        "own": own_cb0[start:end],
        "partner": partner_cb0[start:end],
        "own_active": own_active[start:end],
        "partner_active": partner_active[start:end],
        "t0": torch.tensor(place.t0 - start),
    }


def collate_windows(windows: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Pad on the right: the model is causal, so padding never reaches a scored frame."""
    length = max(len(window["own"]) for window in windows)
    out = {}
    for key in ("own", "partner", "own_active", "partner_active"):
        rows = [
            torch.cat((window[key], window[key].new_zeros(length - len(window[key]))))
            for window in windows
        ]
        out[key] = torch.stack(rows)
    out["t0"] = torch.stack([window["t0"] for window in windows])
    return out
