"""Stage 1: A→B continuation, speech-frame losses and contrastive intent.

Stage 1 trains the full model on unlabeled single-speaker speech. Channel A
carries the utterance context; channel B carries its continuation, which never
appears in A. The ear predicts the next A code on speech frames, the dialogue
tower predicts B on speech frames, and a contrastive intent head picks the true
continuation among same-speaker alternatives.

The recipe is fixed in ``docs/version2-stage1-spec.md``.
"""

import json
import math
import random
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.nn import functional as F

from .model import DauysHybrid, ModelOutput
from .v02_data import sha256
from .v02_training import masked_mean
from .v03_intent import IntentModule

FRAME_RATE = 12.5
SECONDS_BEFORE = 3.0
SECONDS_AFTER = 2.0
MIN_PAUSE_FRAMES = 3
CROP_FRAMES = 256
INTENT_FRAMES = 32
INTENT_TEMPERATURE = 0.1
CANDIDATES = 8
SPEAKERS_PER_BATCH = 16
ANCHORS_PER_SPEAKER = 4


@dataclass(frozen=True)
class Stage1Record:
    id: str
    speaker: str
    frames: int
    cut_frame: int
    token_path: Path
    token_sha256: str
    chapter: str = ""


def _chapter_of(identity: str) -> str:
    """LibriSpeech IDs are speaker-chapter-utterance; keep speaker-chapter."""
    return identity.rsplit("-", 1)[0] if "-" in identity else identity


def load_stage1_corpus(
    root: str | Path,
    manifest: str | Path,
    *,
    verify_hashes: bool = True,
) -> list[Stage1Record]:
    """Load an A→B manifest produced by ``scripts/v02_stage1_build.py``."""
    root = Path(root)
    rows = [
        json.loads(line)
        for line in Path(manifest).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError("The Stage 1 manifest is empty")
    records: list[Stage1Record] = []
    seen: set[str] = set()
    for row in rows:
        identity = str(row["id"])
        if identity in seen:
            raise ValueError(f"Duplicate Stage 1 ID: {identity}")
        seen.add(identity)
        frames = int(row["frames"])
        cut_frame = int(row["cut_frame"])
        if frames < 1 or not 0 < cut_frame < frames:
            raise ValueError(f"Invalid cut for {identity}")
        token_path = (root / str(row["tokens_path"])).resolve()
        if not token_path.is_relative_to(root.resolve()):
            raise ValueError("Token path escapes its corpus root")
        if verify_hashes and sha256(token_path) != str(row["sha256"]):
            raise ValueError(f"Token checksum mismatch: {identity}")
        speakers = str(row["speaker"])
        chapter = str(row.get("chapter") or _chapter_of(identity))
        records.append(
            Stage1Record(
                id=identity,
                speaker=speakers,
                frames=frames,
                cut_frame=cut_frame,
                token_path=token_path,
                token_sha256=str(row["sha256"]),
                chapter=chapter,
            )
        )
    return records


def _candidate_groups(
    records: Sequence[Stage1Record],
    pool: int,
    *,
    same_chapter: bool = True,
    chapter_only: bool = False,
) -> list[list[int]]:
    if chapter_only and not same_chapter:
        raise ValueError("Chapter-only pools require same_chapter")
    chapter_groups: dict[tuple[str, str], list[int]] = {}
    speaker_groups: dict[str, list[int]] = {}
    for index, record in enumerate(records):
        chapter_groups.setdefault(
            (record.speaker, record.chapter or record.speaker), []
        ).append(index)
        speaker_groups.setdefault(record.speaker, []).append(index)
    if not same_chapter:
        return [indices for indices in speaker_groups.values() if len(indices) >= pool]
    groups = [indices for indices in chapter_groups.values() if len(indices) >= pool]
    if chapter_only:
        return groups
    covered = {records[indices[0]].speaker for indices in groups}
    groups += [
        indices
        for speaker, indices in speaker_groups.items()
        if speaker not in covered and len(indices) >= pool
    ]
    return groups


def assert_stage1_pools(
    records: Sequence[Stage1Record],
    *,
    speakers: int = SPEAKERS_PER_BATCH,
    pool: int = CANDIDATES,
    chapter_only: bool = False,
) -> None:
    """Fail fast when too few groups can supply a same-speaker candidate pool.

    ``chapter_only`` counts speaker-chapter groups, the strict pool used by M3
    so candidates cannot be separated by room or microphone alone.
    """
    groups = _candidate_groups(records, pool, chapter_only=chapter_only)
    label = "speaker-chapter" if chapter_only else "speaker"
    if len(groups) < speakers:
        raise ValueError(
            f"Only {len(groups)} {label} groups have at least {pool} items; "
            f"need {speakers}"
        )


def find_cut(
    activity: Sequence[int],
    *,
    min_before: int,
    min_after: int,
    min_pause: int = MIN_PAUSE_FRAMES,
    max_pause: int | None = None,
    seed: int | None = None,
) -> int | None:
    """Return a pause start with enough speech before and after it.

    The pause belongs to B, so B opens with the natural silence and then the
    continuation. ``max_pause`` keeps the reply close to the cut, so a 256-frame
    crop always contains B speech. ``seed`` chooses among all valid pauses, so
    the model sees long and short contexts; ``None`` keeps the first valid one.
    ``None`` is also returned when no valid cut exists.
    """
    if min_before < 1 or min_after < 1 or min_pause < 1:
        raise ValueError("Pause and context requirements must be positive")
    if max_pause is not None and max_pause < min_pause:
        raise ValueError("The maximum pause cannot be shorter than the minimum")
    total = len(activity)
    if total < min_before + min_after + min_pause:
        return None
    prefix = [0]
    for value in activity:
        prefix.append(prefix[-1] + (1 if value else 0))
    candidates: list[int] = []
    index = 0
    while index < total:
        if activity[index]:
            index += 1
            continue
        pause_start = index
        while index < total and not activity[index]:
            index += 1
        pause_end = index
        if pause_end - pause_start < min_pause:
            continue
        if max_pause is not None and pause_end - pause_start > max_pause:
            continue
        if prefix[pause_start] < min_before:
            continue
        if prefix[total] - prefix[pause_end] < min_after:
            continue
        candidates.append(pause_start)
    if not candidates:
        return None
    if seed is None:
        return candidates[0]
    return random.Random(seed).choice(candidates)


@dataclass(frozen=True)
class AbLayout:
    user: torch.Tensor
    target: torch.Tensor
    user_activity: torch.Tensor
    target_activity: torch.Tensor


def ab_layout(
    codes: torch.Tensor,
    activity: torch.Tensor,
    cut: int,
    silence_codes: torch.Tensor,
) -> AbLayout:
    """Split one recording into A = speech before the cut, B = speech after it."""
    if codes.ndim != 2 or codes.shape[0] != 8 or codes.dtype != torch.long:
        raise ValueError("Codes must be long [8, frames]")
    frames = codes.shape[-1]
    if activity.shape != (frames,) or silence_codes.shape != (8,):
        raise ValueError("Activity and silence codes must match the frames")
    if not 0 < cut < frames:
        raise ValueError("The cut must fall inside the recording")
    silence = silence_codes[:, None]
    user = torch.cat((codes[:, :cut], silence.expand(8, frames - cut)), dim=-1)
    target = torch.cat((silence.expand(8, cut), codes[:, cut:]), dim=-1)
    zeros = torch.zeros(frames - cut, dtype=activity.dtype)
    user_activity = torch.cat((activity[:cut], zeros))
    target_activity = torch.cat(
        (torch.zeros(cut, dtype=activity.dtype), activity[cut:])
    )
    return AbLayout(
        user=user,
        target=target,
        user_activity=user_activity,
        target_activity=target_activity,
    )


def crop_around_cut(
    frames: int,
    cut: int,
    crop_frames: int,
    seed: int,
    *,
    min_before: int,
    min_after: int,
) -> tuple[int, int]:
    """Pick a crop that keeps at least ``min_before``/``min_after`` frames."""
    if crop_frames < min_before + min_after + 1:
        raise ValueError("The crop cannot hold both context windows")
    if frames <= crop_frames:
        return 0, frames
    low = max(0, cut + min_after - crop_frames)
    high = min(cut - min_before, frames - crop_frames)
    if low > high:
        start = min(max(0, cut - min_before), frames - crop_frames)
        return start, start + crop_frames
    start = random.Random(seed).randint(low, high)
    return start, start + crop_frames


def continuation_window(
    target: torch.Tensor,
    activity: torch.Tensor,
    cut: int,
    length: int = INTENT_FRAMES,
) -> tuple[torch.Tensor, torch.Tensor]:
    """First ``length`` speech frames of B after the cut, padded with silence."""
    if target.ndim != 2 or target.shape[0] != 8:
        raise ValueError("Target must have shape [8, frames]")
    frames = target.shape[-1]
    if not 0 <= cut < frames:
        raise ValueError("The cut must fall inside the continuation")
    active = activity.bool()
    speech = torch.nonzero(active[cut:], as_tuple=False)
    start = cut + speech[0].item() if speech.numel() else cut
    window = target[:, start : start + length]
    mask = active[start : start + length]
    pad = length - window.shape[-1]
    if pad > 0:
        window = torch.cat((window, torch.zeros((8, pad), dtype=window.dtype)), dim=-1)
        mask = torch.cat((mask, torch.zeros(pad, dtype=torch.bool)))
    return window.contiguous(), mask.contiguous()


def _load_tokens(path: Path) -> dict[str, torch.Tensor]:
    cached = torch.load(path, map_location="cpu", weights_only=True)
    user = cached["user"].long()
    target = cached["target"].long()
    if user.shape != target.shape or user.ndim != 2 or user.shape[0] != 8:
        raise ValueError(f"Invalid Stage 1 token geometry: {path}")
    frames = user.shape[-1]
    if "user_activity" not in cached or "activity" not in cached:
        raise ValueError(f"Stage 1 tokens lack speech activity: {path}")
    user_activity = cached["user_activity"].long()
    target_activity = cached["activity"].long()
    if user_activity.shape != (frames,) or target_activity.shape != (frames,):
        raise ValueError(f"Stage 1 activity does not match frames: {path}")
    return {
        "user": user,
        "target": target,
        "user_activity": user_activity,
        "target_activity": target_activity,
    }


@dataclass
class Stage1Batch:
    user_codes: torch.Tensor
    output_codes: torch.Tensor
    user_activity: torch.Tensor
    output_activity: torch.Tensor
    valid_frames: torch.Tensor
    initial_output_codes: torch.Tensor
    silence_codes: torch.Tensor
    intent_frames: torch.Tensor
    candidate_codes: torch.Tensor
    candidate_activity: torch.Tensor
    positive_index: torch.Tensor
    speaker_index: torch.Tensor
    ids: tuple[str, ...]
    speakers: tuple[str, ...]
    candidate_ids: tuple[tuple[str, ...], ...]

    def to(self, device: torch.device) -> "Stage1Batch":
        return Stage1Batch(
            user_codes=self.user_codes.to(device),
            output_codes=self.output_codes.to(device),
            user_activity=self.user_activity.to(device),
            output_activity=self.output_activity.to(device),
            valid_frames=self.valid_frames.to(device),
            initial_output_codes=self.initial_output_codes.to(device),
            silence_codes=self.silence_codes.to(device),
            intent_frames=self.intent_frames.to(device),
            candidate_codes=self.candidate_codes.to(device),
            candidate_activity=self.candidate_activity.to(device),
            positive_index=self.positive_index.to(device),
            speaker_index=self.speaker_index.to(device),
            ids=self.ids,
            speakers=self.speakers,
            candidate_ids=self.candidate_ids,
        )

    def validate(self) -> None:
        batch, _, frames = self.user_codes.shape
        groups = self.positive_index.shape[0]
        anchors = self.positive_index.shape[1]
        candidates = self.candidate_codes.shape[1]
        if self.output_codes.shape != (batch, 8, frames):
            raise ValueError("Both channels must share one geometry")
        if self.user_activity.shape != (batch, frames):
            raise ValueError("User activity must match the frames")
        if self.output_activity.shape != (batch, frames):
            raise ValueError("Output activity must match the frames")
        if self.valid_frames.shape != (batch, frames):
            raise ValueError("Valid frames must match the frames")
        if not bool(((self.intent_frames >= 0) & (self.intent_frames < frames)).all()):
            raise ValueError("Intent frames must fall inside the batch")
        if self.candidate_codes.shape[:3] != (groups, candidates, 8):
            raise ValueError("Candidate codes must have shape [groups, K, 8, T]")
        if self.candidate_activity.shape != (
            groups,
            candidates,
            self.candidate_codes.shape[-1],
        ):
            raise ValueError("Candidate activity must match candidate frames")
        if anchors * groups != batch:
            raise ValueError("Anchors must be grouped by speaker")


def make_batch_spec(
    records: Sequence[Stage1Record],
    step: int,
    *,
    seed: int,
    speakers: int = SPEAKERS_PER_BATCH,
    anchors: int = ANCHORS_PER_SPEAKER,
    pool: int = CANDIDATES,
    same_chapter: bool = True,
    chapter_only: bool = False,
) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
    """Choose ``speakers`` groups, a same-speaker pool and its anchors.

    Candidates come from one chapter when possible, so room and microphone
    identity cannot separate the true continuation from its alternatives.
    Speakers without a chapter pool fall back to a speaker pool unless
    ``chapter_only`` is set (M3 and its control).
    """
    if speakers < 1 or anchors < 1 or pool < anchors:
        raise ValueError("Invalid speaker, anchor or pool count")
    groups = _candidate_groups(
        records, pool, same_chapter=same_chapter, chapter_only=chapter_only
    )
    if chapter_only and not groups:
        raise ValueError("No speaker-chapter group has the required pool")
    if len(groups) < speakers:
        raise ValueError("Not enough speaker groups with the required candidate pool")
    rng = random.Random(seed * 1_000_003 + step)
    chosen = rng.sample(groups, speakers)
    spec: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    for indices in chosen:
        pool_indices = tuple(rng.sample(indices, pool))
        anchor_positions = tuple(sorted(rng.sample(range(pool), anchors)))
        spec.append((pool_indices, anchor_positions))
    return spec


def load_batch(
    records: Sequence[Stage1Record],
    spec: Sequence[tuple[Sequence[int], Sequence[int]]],
    silence_codes: torch.Tensor,
    *,
    crop_frames: int = CROP_FRAMES,
    seed: int = 0,
    min_before: int = math.ceil(SECONDS_BEFORE * FRAME_RATE),
    min_after: int = math.ceil(SECONDS_AFTER * FRAME_RATE),
    swap_user: bool = False,
) -> Stage1Batch:
    """Materialize one same-speaker batch with its candidate continuations.

    ``swap_user`` replaces A with another pool phrase (same chapter under M3)
    while keeping B, the M3 control that must fall back to chance.
    """
    silence = silence_codes.long()
    groups = len(spec)
    anchors = len(spec[0][1])
    candidates = len(spec[0][0])
    loaded: list[dict[str, torch.Tensor]] = []
    crops: list[tuple[int, int, int]] = []
    swap_loaded: list[dict[str, torch.Tensor] | None] = []
    swap_cuts: list[int] = []
    for group, (pool, anchor_positions) in enumerate(spec):
        for local, position in enumerate(anchor_positions):
            record = records[pool[position]]
            cached = _load_tokens(record.token_path)
            start, end = crop_around_cut(
                record.frames,
                record.cut_frame,
                crop_frames,
                seed + group * 10_000 + local,
                min_before=min_before,
                min_after=min_after,
            )
            loaded.append(cached)
            crops.append((start, end, record.cut_frame - start))
            if swap_user and len(pool) > 1:
                other = records[pool[(position + 1) % len(pool)]]
                swap_loaded.append(_load_tokens(other.token_path))
                swap_cuts.append(other.cut_frame)
            else:
                swap_loaded.append(None)
                swap_cuts.append(0)
    frames = max(end - start for start, end, _ in crops)
    batch = groups * anchors
    user = silence[None, :, None].expand(batch, 8, frames).clone()
    target = user.clone()
    user_activity = torch.zeros((batch, frames), dtype=torch.long)
    target_activity = torch.zeros((batch, frames), dtype=torch.long)
    valid = torch.zeros((batch, frames), dtype=torch.bool)
    intent_frames = torch.zeros(batch, dtype=torch.long)
    for index, (cached, (start, end, cut_local)) in enumerate(zip(loaded, crops)):
        length = end - start
        user[index, :, :length] = cached["user"][:, start:end]
        target[index, :, :length] = cached["target"][:, start:end]
        user_activity[index, :length] = cached["user_activity"][start:end]
        target_activity[index, :length] = cached["target_activity"][start:end]
        valid[index, :length] = True
        # The intent is read at the last frame before B starts.
        intent_frames[index] = max(0, cut_local - 1)
        swap = swap_loaded[index]
        if swap is not None:
            # Replace A with silence plus the other phrase's own context,
            # placed so it ends exactly at the anchor cut. No anchor A survives.
            user[index] = silence[:, None].expand(8, frames)
            user_activity[index].zero_()
            other_cut = swap_cuts[index]
            take = min(cut_local, other_cut)
            if take > 0:
                user[index, :, cut_local - take : cut_local] = swap["user"][
                    :, other_cut - take : other_cut
                ]
                user_activity[index, cut_local - take : cut_local] = swap[
                    "user_activity"
                ][other_cut - take : other_cut]
    candidate_codes = (
        silence[None, None, :, None]
        .expand(groups, candidates, 8, INTENT_FRAMES)
        .clone()
    )
    candidate_activity = torch.zeros(
        (groups, candidates, INTENT_FRAMES), dtype=torch.bool
    )
    positive_index = torch.zeros((groups, anchors), dtype=torch.long)
    candidate_ids: list[tuple[str, ...]] = []
    for group, (pool, anchor_positions) in enumerate(spec):
        identifiers = []
        for slot, record_index in enumerate(pool):
            record = records[record_index]
            cached = _load_tokens(record.token_path)
            window, mask = continuation_window(
                cached["target"], cached["target_activity"], record.cut_frame
            )
            candidate_codes[group, slot] = window
            candidate_activity[group, slot] = mask
            identifiers.append(record.id)
        candidate_ids.append(tuple(identifiers))
        for local, position in enumerate(anchor_positions):
            positive_index[group, local] = position
    speaker_index = torch.arange(groups).repeat_interleave(anchors)
    return Stage1Batch(
        user_codes=user,
        output_codes=target,
        user_activity=user_activity,
        output_activity=target_activity,
        valid_frames=valid,
        initial_output_codes=silence[None].expand(batch, 8).clone(),
        silence_codes=silence[None].expand(batch, 8).clone(),
        intent_frames=intent_frames,
        candidate_codes=candidate_codes,
        candidate_activity=candidate_activity,
        positive_index=positive_index,
        speaker_index=speaker_index,
        ids=tuple(
            records[spec[group][0][position]].id
            for group in range(groups)
            for position in spec[group][1]
        ),
        speakers=tuple(records[spec[group][0][0]].speaker for group in range(groups)),
        candidate_ids=tuple(candidate_ids),
    )


@dataclass(frozen=True)
class Stage1Weights:
    cb0: float = 1.0
    depth: float = 0.5
    activity: float = 0.2
    intent: float = 0.5
    listening: float = 1.0
    listening_activity: float = 0.2
    acoustic_dropout: float = 0.5
    label_smoothing: float = 0.05
    history_dropout: float = 0.0
    listening_horizon: int = 1


def _interpolate(progress: float, values: tuple[float, float, float, float]) -> float:
    points = (0.0, 0.1, 0.4, 1.0)
    for index in range(3):
        if progress <= points[index + 1]:
            ratio = (progress - points[index]) / (points[index + 1] - points[index])
            return values[index] + ratio * (values[index + 1] - values[index])
    return values[-1]


def stage1_weights(progress: float) -> Stage1Weights:
    if not 0 <= progress <= 1:
        raise ValueError("Training progress must lie in [0, 1]")
    return Stage1Weights(
        listening=_interpolate(progress, (1.0, 1.0, 0.5, 0.3)),
    )


@dataclass(frozen=True)
class Stage1Masks:
    cb0: torch.Tensor
    activity: torch.Tensor
    listening: torch.Tensor


def stage1_masks(
    valid: torch.Tensor, output_activity: torch.Tensor, user_activity: torch.Tensor
) -> Stage1Masks:
    """CB0/depth on B speech, activity on every valid frame, C1 on A speech."""
    speech = output_activity.bool()
    listen = valid[:, :-1] & valid[:, 1:] & user_activity[:, 1:].bool()
    return Stage1Masks(cb0=valid & speech, activity=valid, listening=listen)


def infonce(
    prediction: torch.Tensor,
    candidates: torch.Tensor,
    target: torch.Tensor,
    temperature: float = INTENT_TEMPERATURE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """InfoNCE over same-speaker candidates; target is the positive slot."""
    if temperature <= 0:
        raise ValueError("Temperature must be positive")
    if prediction.shape[:-1] != target.shape:
        raise ValueError("Prediction and target must share [groups, anchors]")
    normalized = F.normalize(prediction.float(), dim=-1)
    normalized_candidates = F.normalize(candidates.float(), dim=-1)
    logits = torch.einsum("sad,skd->sak", normalized, normalized_candidates)
    logits = logits / temperature
    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1))
    accuracy = (logits.argmax(-1) == target).float().mean()
    return loss, accuracy


@dataclass
class Stage1Loss:
    total: torch.Tensor
    cb0: torch.Tensor
    depth: torch.Tensor
    activity: torch.Tensor
    listening_cb0: torch.Tensor
    listening_activity: torch.Tensor
    intent: torch.Tensor
    intent_accuracy: torch.Tensor


def _hide_history(
    batch: Stage1Batch, probability: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Replace B after the cut with silence on the input side for some items.

    Labels stay the true B, so those frames can only be predicted from A.
    Teacher forcing otherwise lets B's own past explain almost every frame and
    the tower learns to ignore what it heard.
    """
    batch_size, _, frames = batch.output_codes.shape
    device = batch.output_codes.device
    if probability <= 0:
        return batch.output_codes, torch.zeros(
            batch_size, dtype=torch.bool, device=device
        )
    hidden = torch.rand(batch_size, device=device) < probability
    cut = batch.intent_frames + 1
    after = torch.arange(frames, device=device)[None] >= cut[:, None]
    blank = (hidden[:, None] & after)[:, None, :]
    silence = batch.initial_output_codes[:, :, None].to(batch.output_codes.dtype)
    return torch.where(blank, silence, batch.output_codes), hidden


def stage1_objective(
    model: DauysHybrid,
    batch: Stage1Batch,
    weights: Stage1Weights,
    intent: IntentModule,
) -> Stage1Loss:
    """Speech-frame continuation loss plus the same-speaker intent choice."""
    if model.config.name != "dauys-hybrid-v0.2":
        raise ValueError("Stage 1 requires a v0.2 model")
    batch.validate()
    inputs, hidden = _hide_history(
        batch, weights.history_dropout if model.training else 0.0
    )
    output = model(
        batch.user_codes,
        inputs,
        batch.initial_output_codes,
        acoustic_dropout=weights.acoustic_dropout if model.training else 0.0,
    )
    masks = stage1_masks(batch.valid_frames, batch.output_activity, batch.user_activity)
    batch_size, frames = batch.output_activity.shape
    vocab = model.config.vocab_size
    cb0_values = F.cross_entropy(
        output.cb0_logits.reshape(-1, vocab),
        batch.output_codes[:, 0].reshape(-1),
        reduction="none",
        label_smoothing=weights.label_smoothing if model.training else 0.0,
    ).reshape(batch_size, frames)
    cb0 = masked_mean(cb0_values, masks.cb0)
    assert output.depth_logits is not None
    depth_values = F.cross_entropy(
        output.depth_logits.reshape(-1, vocab),
        batch.output_codes[:, 1:].transpose(1, 2).reshape(-1),
        reduction="none",
    ).reshape(batch_size, frames, 7)
    # Depth reads the frame's own CB0 from the input, which is silence when hidden.
    depth = masked_mean(depth_values.mean(dim=-1), masks.cb0 & ~hidden[:, None])
    activity_values = F.cross_entropy(
        output.activity_logits.reshape(-1, 2),
        batch.output_activity.reshape(-1).long(),
        reduction="none",
    ).reshape(batch_size, frames)
    activity = masked_mean(activity_values, masks.activity)
    zero = output.cb0_logits.sum() * 0
    listening_cb0 = zero
    listening_activity = zero
    horizon = weights.listening_horizon
    if horizon < 1:
        raise ValueError("The listening horizon must be at least one frame")
    if output.listening_cb0_logits is not None and frames > horizon:
        # A horizon beyond the next frame cannot be met from local acoustics,
        # so the ear has to carry syllables and words forward.
        ahead = (
            batch.valid_frames[:, :-horizon]
            & batch.valid_frames[:, horizon:]
            & batch.user_activity[:, horizon:].bool()
        )
        listening_cb0 = masked_mean(
            F.cross_entropy(
                output.listening_cb0_logits[:, :-horizon].reshape(-1, vocab),
                batch.user_codes[:, 0, horizon:].reshape(-1),
                reduction="none",
            ).reshape(batch_size, frames - horizon),
            ahead,
        )
        if output.listening_activity_logits is not None:
            listening_activity = masked_mean(
                F.cross_entropy(
                    output.listening_activity_logits[:, :-1].reshape(-1, 2),
                    batch.user_activity[:, 1:].reshape(-1).long(),
                    reduction="none",
                ).reshape(batch_size, frames - 1),
                masks.listening,
            )
    intent_loss, intent_accuracy = _intent_choice(model, batch, output, intent)
    total = (
        weights.cb0 * cb0
        + weights.depth * depth
        + weights.activity * activity
        + weights.intent * intent_loss
        + weights.listening
        * (listening_cb0 + weights.listening_activity * listening_activity)
    )
    return Stage1Loss(
        total=total,
        cb0=cb0,
        depth=depth,
        activity=activity,
        listening_cb0=listening_cb0,
        listening_activity=listening_activity,
        intent=intent_loss,
        intent_accuracy=intent_accuracy,
    )


def _intent_choice(
    model: DauysHybrid,
    batch: Stage1Batch,
    output: ModelOutput,
    intent: IntentModule,
) -> tuple[torch.Tensor, torch.Tensor]:
    groups, candidates, _, _ = batch.candidate_codes.shape
    anchors = batch.positive_index.shape[1]
    frames = batch.candidate_codes.shape[-1]
    batch_size = batch.positive_index.numel()
    states = output.dialogue_states
    positions = batch.intent_frames
    rows = torch.arange(batch_size, device=states.device)
    with torch.autocast(states.device.type, enabled=False):
        prediction = intent.head(states[rows, positions].float())
        target_states = intent.target(
            batch.candidate_codes.reshape(groups * candidates, 8, frames).long()
        )
        mask = batch.candidate_activity.reshape(groups * candidates, frames).float()
        pooled = (target_states.float() * mask[..., None]).sum(dim=1)
        pooled = pooled / mask.sum(dim=1, keepdim=True).clamp_min(1)
    prediction = prediction.reshape(groups, anchors, -1)
    pooled = pooled.reshape(groups, candidates, -1)
    return infonce(prediction, pooled, batch.positive_index)


def intent_accuracy(
    model: DauysHybrid, batch: Stage1Batch, intent: IntentModule
) -> float:
    """Held-out future selection accuracy for metric M3."""
    batch.validate()
    output = model(batch.user_codes, batch.output_codes, batch.initial_output_codes)
    _, accuracy = _intent_choice(model, batch, output, intent)
    return float(accuracy)


@torch.inference_mode()
def intent_spread(
    model: DauysHybrid, batch: Stage1Batch, intent: IntentModule
) -> float:
    """Normalized spread of intent vectors; collapse is a red flag."""
    batch.validate()
    output = model(batch.user_codes, batch.output_codes, batch.initial_output_codes)
    states = output.dialogue_states
    rows = torch.arange(batch.positive_index.numel(), device=states.device)
    with torch.autocast(states.device.type, enabled=False):
        prediction = intent.head(states[rows, batch.intent_frames].float())
    return spread_score(prediction)


def spread_score(prediction: torch.Tensor) -> float:
    """Scale-normalized spread: ~1 for random unit vectors, ~0 on collapse.

    A unit vector in ``d`` dimensions has per-coordinate standard deviation
    about ``1/sqrt(d)``; multiplying by ``sqrt(d)`` makes the numbers comparable
    across widths, so the fixed 0.05 abort threshold means what it says.
    """
    normalized = F.normalize(prediction.float(), dim=-1)
    return float(normalized.std(dim=0).mean() * math.sqrt(normalized.shape[-1]))


class Stage1BatchSource(torch.utils.data.Dataset[Stage1Batch]):
    """Lazily builds one same-speaker batch per index for the training loop."""

    def __init__(
        self,
        records: Sequence[Stage1Record],
        silence_codes: torch.Tensor,
        *,
        steps: int,
        seed: int,
        speakers: int = SPEAKERS_PER_BATCH,
        anchors: int = ANCHORS_PER_SPEAKER,
        pool: int = CANDIDATES,
        crop_frames: int = CROP_FRAMES,
        start: int = 0,
        min_before: int = math.ceil(SECONDS_BEFORE * FRAME_RATE),
        min_after: int = math.ceil(SECONDS_AFTER * FRAME_RATE),
    ) -> None:
        if steps < 1:
            raise ValueError("A Stage 1 source needs at least one step")
        if not 0 <= start <= steps:
            raise ValueError("The Stage 1 start step must lie inside the plan")
        self.records = list(records)
        self.silence = silence_codes.long()
        self.steps = steps
        self.start = start
        self.seed = seed
        self.speakers = speakers
        self.anchors = anchors
        self.pool = pool
        self.crop_frames = crop_frames
        self.min_before = min_before
        self.min_after = min_after

    def __len__(self) -> int:
        return self.steps - self.start

    def __getitem__(self, index: int) -> Stage1Batch:
        step = self.start + index
        spec = make_batch_spec(
            self.records,
            step,
            seed=self.seed,
            speakers=self.speakers,
            anchors=self.anchors,
            pool=self.pool,
        )
        return load_batch(
            self.records,
            spec,
            self.silence,
            crop_frames=self.crop_frames,
            seed=self.seed + step,
            min_before=self.min_before,
            min_after=self.min_after,
        )
