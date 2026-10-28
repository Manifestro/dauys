"""Gate A measurements and decisions for Stage 1.

The bands and the GO/FIX/STOP outcome are fixed in
``docs/version2-stage1-spec.md`` and must not be tuned after a run.
"""

import math
import random
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .decoding import DecodingConfig
from .model import DauysHybrid
from .v02_stage1 import (
    CROP_FRAMES,
    Stage1Record,
    _load_tokens,
    crop_around_cut,
    intent_accuracy,
    load_batch,
)
from .v03_intent import IntentModule

MIN_BEFORE = 38
MIN_AFTER = 25
EMPTY_RMS = 0.003


@dataclass(frozen=True)
class RecordTensors:
    user: torch.Tensor
    target: torch.Tensor
    user_activity: torch.Tensor
    target_activity: torch.Tensor
    cut_local: int
    frames: int


def record_tensors(
    record: Stage1Record,
    silence_codes: torch.Tensor,
    *,
    crop_frames: int = CROP_FRAMES,
    seed: int = 0,
) -> RecordTensors:
    """Load one A→B record and crop it around the cut."""
    del silence_codes
    cached = _load_tokens(record.token_path)
    start, end = crop_around_cut(
        record.frames,
        record.cut_frame,
        crop_frames,
        seed,
        min_before=MIN_BEFORE,
        min_after=MIN_AFTER,
    )
    return RecordTensors(
        user=cached["user"][:, start:end].contiguous(),
        target=cached["target"][:, start:end].contiguous(),
        user_activity=cached["user_activity"][start:end].contiguous(),
        target_activity=cached["target_activity"][start:end].contiguous(),
        cut_local=record.cut_frame - start,
        frames=end - start,
    )


def _padded_records(
    records: Sequence[Stage1Record],
    silence_codes: torch.Tensor,
    crop_frames: int,
) -> tuple[torch.Tensor, ...]:
    items = [
        record_tensors(record, silence_codes, crop_frames=crop_frames)
        for record in records
    ]
    frames = max(item.frames for item in items)
    batch = len(items)
    silence = silence_codes.long()
    user = silence[None, :, None].expand(batch, 8, frames).clone()
    target = user.clone()
    user_activity = torch.zeros((batch, frames), dtype=torch.long)
    target_activity = torch.zeros((batch, frames), dtype=torch.long)
    valid = torch.zeros((batch, frames), dtype=torch.bool)
    for index, item in enumerate(items):
        length = item.frames
        user[index, :, :length] = item.user
        target[index, :, :length] = item.target
        user_activity[index, :length] = item.user_activity
        target_activity[index, :length] = item.target_activity
        valid[index, :length] = True
    initial = silence[None].expand(batch, 8).clone()
    return user, target, user_activity, target_activity, valid, initial


@torch.inference_mode()
def speech_frame_scores(
    model: DauysHybrid,
    records: Sequence[Stage1Record],
    silence_codes: torch.Tensor,
    *,
    batch_size: int = 16,
    crop_frames: int = CROP_FRAMES,
    fusion_off: bool = False,
) -> dict[str, float]:
    """M1: teacher-forced CB0 NLL and accuracy on B speech frames only."""
    if not records or batch_size < 1:
        raise ValueError("Speech-frame scoring needs records and a batch size")
    device = next(model.parameters()).device
    fused_off = frozenset(model.fusion_block_indices()) if fusion_off else frozenset()
    total = correct = 0
    nll = 0.0
    for start in range(0, len(records), batch_size):
        chunk = list(records[start : start + batch_size])
        user, target, _, ta, valid, initial = _padded_records(
            chunk, silence_codes, crop_frames
        )
        output = model(
            user.to(device),
            target.to(device),
            initial.to(device),
            fusion_off_blocks=fused_off,
        )
        speech = valid.to(device) & ta.to(device).bool()
        logits = output.cb0_logits.float()
        labels = target.to(device)[:, 0]
        nll += float(F.cross_entropy(logits[speech], labels[speech], reduction="sum"))
        correct += int(((logits.argmax(-1) == labels) & speech).sum())
        total += int(speech.sum())
    if total == 0:
        raise ValueError("No speech frames were scored")
    return {"nll": nll / total, "accuracy": correct / total, "frames": float(total)}


@torch.inference_mode()
def fusion_dependence(
    model: DauysHybrid,
    records: Sequence[Stage1Record],
    silence_codes: torch.Tensor,
    **kwargs,
) -> float:
    """M4: NLL with every cross-attention block off minus NLL with them on."""
    off = speech_frame_scores(model, records, silence_codes, fusion_off=True, **kwargs)[
        "nll"
    ]
    on = speech_frame_scores(model, records, silence_codes, **kwargs)["nll"]
    return off - on


@torch.inference_mode()
def overfit_gap(
    model: DauysHybrid,
    training: Sequence[Stage1Record],
    heldout: Sequence[Stage1Record],
    silence_codes: torch.Tensor,
    *,
    count: int = 64,
    **kwargs,
) -> float:
    """M6: training CB0 accuracy minus held-out CB0 accuracy, in points."""
    sample = list(training)[:count]
    train = speech_frame_scores(model, sample, silence_codes, **kwargs)["accuracy"]
    hold = speech_frame_scores(model, heldout, silence_codes, **kwargs)["accuracy"]
    return (train - hold) * 100


@dataclass
class Cb0Bigram:
    probabilities: torch.Tensor
    vocabulary: int = 2048
    smoothing: float = 1e-3

    @classmethod
    def fit(
        cls,
        runs: Iterable[Sequence[int] | torch.Tensor],
        *,
        vocabulary: int = 2048,
        smoothing: float = 1e-3,
        chunk_transitions: int = 5_000_000,
    ) -> "Cb0Bigram":
        """Vectorized add-k bigram, matching ``scripts/v02_cb0_baselines.py``."""
        if vocabulary < 1 or smoothing <= 0 or chunk_transitions < 1:
            raise ValueError("Invalid bigram vocabulary, smoothing or chunk")
        counts = torch.full((vocabulary, vocabulary), smoothing, dtype=torch.float64)
        buffer: list[torch.Tensor] = []
        buffered = 0
        for run in runs:
            codes = torch.as_tensor(run, dtype=torch.long)
            if codes.numel() < 2:
                continue
            pairs = codes[:-1] * vocabulary + codes[1:]
            buffer.append(pairs)
            buffered += int(pairs.numel())
            if buffered >= chunk_transitions:
                flat = torch.cat(buffer)
                counts += (
                    torch.bincount(flat, minlength=vocabulary * vocabulary)
                    .view(vocabulary, vocabulary)
                    .double()
                )
                buffer = []
                buffered = 0
        if buffer:
            flat = torch.cat(buffer)
            counts += (
                torch.bincount(flat, minlength=vocabulary * vocabulary)
                .view(vocabulary, vocabulary)
                .double()
            )
        return cls(
            probabilities=counts / counts.sum(dim=1, keepdim=True),
            vocabulary=vocabulary,
            smoothing=smoothing,
        )

    def nll(self, runs: Iterable[Sequence[int] | torch.Tensor]) -> float:
        total = 0.0
        frames = 0
        for run in runs:
            codes = torch.as_tensor(run, dtype=torch.long)
            if codes.numel() < 2:
                continue
            total -= float(self.probabilities[codes[:-1], codes[1:]].log().sum())
            frames += int(codes.numel() - 1)
        if frames == 0:
            raise ValueError("The bigram needs at least one transition")
        return total / frames


def speech_cb0_runs(records: Sequence[Stage1Record]) -> list[torch.Tensor]:
    """Contiguous B-speech CB0 runs, so bigrams never cross a pause."""
    runs: list[torch.Tensor] = []
    for record in records:
        cached = _load_tokens(record.token_path)
        active = cached["target_activity"].bool()[: record.frames]
        codes = cached["target"][0, : record.frames].long()
        indices = torch.nonzero(active, as_tuple=False).flatten()
        if indices.numel() == 0:
            continue
        breaks = torch.nonzero(indices[1:] != indices[:-1] + 1).flatten() + 1
        starts = torch.cat((torch.zeros(1, dtype=torch.long), breaks))
        ends = torch.cat((breaks, torch.tensor([indices.numel()])))
        for start, end in zip(starts.tolist(), ends.tolist()):
            runs.append(codes[indices[start:end]].to(torch.int16))
    return runs


@torch.inference_mode()
def _chapter_groups(
    records: Sequence[Stage1Record],
) -> dict[tuple[str, str], list[int]]:
    groups: dict[tuple[str, str], list[int]] = {}
    for index, record in enumerate(records):
        key = (record.speaker, record.chapter or record.speaker)
        groups.setdefault(key, []).append(index)
    return groups


@torch.inference_mode()
def intent_selection(
    model: DauysHybrid,
    intent: IntentModule,
    records: Sequence[Stage1Record],
    silence_codes: torch.Tensor,
    *,
    groups: int = 32,
    candidates: int = 8,
    seed: int = 7,
    crop_frames: int = CROP_FRAMES,
    swap_user: bool = False,
) -> float:
    """M3: pick the true continuation among same-chapter candidates.

    Candidates share the speaker and the chapter, so room, microphone and
    recording conditions match the positive; only the content differs.
    ``swap_user`` is the control: replacing A with another phrase of the same
    chapter must drop accuracy towards chance.
    """
    eligible = [
        indices
        for indices in _chapter_groups(records).values()
        if len(indices) >= candidates
    ]
    if not eligible:
        raise ValueError("No held-out speaker-chapter group has enough candidates")
    rng = random.Random(seed)
    chosen = rng.sample(eligible, min(groups, len(eligible)))
    device = next(model.parameters()).device
    scores = []
    for indices in chosen:
        pool = tuple(rng.sample(indices, candidates))
        spec = [(pool, tuple(range(candidates)))]
        batch = load_batch(
            records,
            spec,
            silence_codes,
            crop_frames=crop_frames,
            seed=seed,
            swap_user=swap_user,
        ).to(device)
        scores.append(intent_accuracy(model, batch, intent))
    return float(np.mean(scores))


@torch.inference_mode()
def chapter_control(
    model: DauysHybrid,
    intent: IntentModule,
    records: Sequence[Stage1Record],
    silence_codes: torch.Tensor,
    **kwargs,
) -> float:
    """M3 control: same chapter but the wrong user phrase must approach chance."""
    return intent_selection(
        model, intent, records, silence_codes, swap_user=True, **kwargs
    )


def speaker_probe(
    model: DauysHybrid,
    records: Sequence[Stage1Record],
    *,
    max_per_speaker: int = 8,
    crop_frames: int = CROP_FRAMES,
    steps: int = 200,
    learning_rate: float = 1e-2,
    seed: int = 0,
) -> float:
    """M7: linear speaker-ID probe on frozen ear states (report only)."""
    by_speaker: dict[str, list[Stage1Record]] = {}
    for record in records:
        by_speaker.setdefault(record.speaker, []).append(record)
    speakers = sorted(by_speaker)
    if len(speakers) < 2:
        raise ValueError("The speaker probe needs at least two speakers")
    device = next(model.parameters()).device
    torch.manual_seed(seed)
    features: list[torch.Tensor] = []
    labels: list[int] = []
    with torch.no_grad():
        for index, speaker in enumerate(speakers):
            for record in by_speaker[speaker][:max_per_speaker]:
                cached = _load_tokens(record.token_path)
                start = record.cut_frame
                end = min(record.frames, start + crop_frames)
                codes = cached["target"][None, :, start:end].long().to(device)
                states = model.encode_listening(codes)
                mask = cached["target_activity"].bool()[start:end].to(device)
                pooled = states[0][mask].mean(dim=0).float().cpu()
                features.append(pooled)
                labels.append(index)
    matrix = torch.stack(features)
    order = list(range(len(labels)))
    random.Random(seed).shuffle(order)
    test_mask = torch.zeros(len(labels), dtype=torch.bool)
    test_mask[order[: max(1, len(order) // 5)]] = True
    train_labels = sorted(
        {labels[index] for index in range(len(labels)) if not bool(test_mask[index])}
    )
    if len(train_labels) < 2:
        raise ValueError("The speaker probe needs two speakers in its train split")
    mapping = {label: index for index, label in enumerate(train_labels)}
    keep = torch.tensor([label in mapping for label in labels])
    train = ~test_mask & keep
    test = test_mask & keep
    if not test.any() or not train.any():
        raise ValueError("The speaker split needs nonempty train and test parts")
    targets = torch.tensor([mapping[label] for label in labels])
    classifier = nn.Linear(matrix.shape[1], len(train_labels))
    optimizer = torch.optim.Adam(classifier.parameters(), lr=learning_rate)
    for _ in range(steps):
        optimizer.zero_grad()
        loss = F.cross_entropy(classifier(matrix[train]), targets[train])
        loss.backward()
        optimizer.step()
    predictions = classifier(matrix[test]).argmax(-1)
    return float((predictions == targets[test]).float().mean())


@torch.inference_mode()
def continuation_waves(
    model: DauysHybrid,
    records: Sequence[Stage1Record],
    silence_codes: torch.Tensor,
    codec,
    *,
    items: int = 100,
    frames: int = 75,
    crop_frames: int = CROP_FRAMES,
    decoding: DecodingConfig | None = None,
) -> list[np.ndarray]:
    """M5/M8: generate B after the cut and return the model-channel waveform."""
    from .codec import decode

    config = decoding or DecodingConfig.v02_default()
    config.validate()
    device = next(model.parameters()).device
    silence = silence_codes.long().to(device)[None].expand(1, 8)
    generator = torch.Generator(device=device).manual_seed(0)
    waves: list[np.ndarray] = []
    for record in list(records)[:items]:
        item = record_tensors(record, silence_codes, crop_frames=crop_frames)
        cut = item.cut_local
        if cut < 1:
            continue
        user = item.user[:, :cut].to(device)[None]
        prompt = silence[:, :, None].expand(1, 8, cut)
        state = model.prefill_teacher(user, prompt, silence)
        generated: list[torch.Tensor] = []
        active: list[torch.Tensor] = []
        for _ in range(frames):
            codes, speaking = model.generate_step(silence, state, config, generator)
            generated.append(codes)
            active.append(speaking)
        codes = torch.stack(generated, dim=-1)
        speaking = torch.stack(active, dim=-1)[:, None]
        playable = torch.where(speaking, codes, silence[:, :, None])
        pcm = decode(codec, playable.long())
        waves.append(pcm[0, 0].float().cpu().numpy())
    return waves


def empty_rate(
    waves: Sequence[np.ndarray],
    transcribe: Callable[[list[np.ndarray]], list[str]] | None,
) -> float:
    """M5: empty = no ASR words or RMS below the energy gate."""
    if not waves:
        raise ValueError("Empty-rate scoring needs generated waves")
    texts = transcribe(list(waves)) if transcribe is not None else None
    empty = 0
    for index, wave in enumerate(waves):
        rms = float(np.sqrt(np.mean(np.square(wave))))
        silent = rms < EMPTY_RMS
        no_words = texts is not None and not texts[index].split()
        if silent or no_words:
            empty += 1
    return empty / len(waves)


@dataclass(frozen=True)
class GateA:
    bands: dict[str, str]
    outcome: str


BANDS = {
    "m1_minus_m2": ("GO", "FIX", "FAIL"),
    "m3": ("GO", "FIX", "FAIL"),
    "m4": ("GO", "FIX", "FAIL"),
    "m5": ("GO", "FIX", "FAIL"),
    "m6": ("GO", "FIX", "FAIL"),
    "m8": ("GO", "FIX", "FAIL"),
}


def _band(
    value: float,
    go: float,
    fix: float,
    higher_is_better: bool,
    *,
    strict_go: bool = False,
) -> str:
    if higher_is_better:
        if value >= go:
            return "GO"
        if value >= fix:
            return "FIX"
        return "FAIL"
    go_hit = value < go if strict_go else value <= go
    if go_hit:
        return "GO"
    if value <= fix:
        return "FIX"
    return "FAIL"


def gate_a_bands(metrics: dict[str, float]) -> GateA:
    """Apply the fixed Gate A bands; missing metrics raise."""
    required = set(BANDS)
    if not required <= metrics.keys():
        raise ValueError(
            f"Gate A is missing metrics: {sorted(required - metrics.keys())}"
        )
    bands = {
        "m1_minus_m2": _band(metrics["m1_minus_m2"], -0.5, -0.2, False),
        "m3": _band(metrics["m3"], 0.70, 0.40, True),
        "m4": _band(metrics["m4"], 0.30, 0.05, True),
        "m5": _band(metrics["m5"], 0.30, 0.50, False, strict_go=True),
        "m6": _band(metrics["m6"], 10.0, 15.0, False, strict_go=True),
        "m8": _band(metrics["m8"], 3.0, 1.0, True),
    }
    if any(band == "FAIL" for band in bands.values()):
        outcome = "STOP"
    elif all(band == "GO" for band in bands.values()):
        outcome = "GO"
    else:
        outcome = "FIX"
    return GateA(bands=bands, outcome=outcome)


def early_abort(step: int, metrics: dict[str, float]) -> str | None:
    """Return a reason when the pre-registered early-abort rule fires."""
    for value in metrics.values():
        if isinstance(value, float) and not math.isfinite(value):
            return "non_finite_metric"
    # Before the intent head has learned anything its outputs share one
    # direction, so a low spread early is the starting point, not a collapse.
    if step >= 2000 and metrics.get("intent_spread", 1.0) < 0.05:
        return "intent_collapse"
    if step >= 4000 and metrics.get("m1_minus_m2", 0.0) >= 0.0:
        return "m1_not_below_bigram"
    if step >= 8000 and metrics.get("m3", 1.0) < 0.25:
        return "intent_near_chance"
    if step > 6000 and metrics.get("m4", 1.0) < 0.01:
        return "fusion_ignored"
    return None


def assert_speakers_disjoint(
    training: Sequence[Stage1Record], heldout: Sequence[Stage1Record]
) -> None:
    """Gate A must never score a speaker that appeared in training."""
    overlap = {record.speaker for record in training} & {
        record.speaker for record in heldout
    }
    if overlap:
        sample = ", ".join(sorted(overlap)[:5])
        raise ValueError(f"Held-out speakers leak into training: {sample}")


def save_wavs(
    waves: Sequence[np.ndarray], directory: str | Path, sample_rate: int = 24_000
) -> list[Path]:
    """Write the M8 listening samples (model channel only)."""
    import soundfile as sf

    folder = Path(directory)
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for index, wave in enumerate(waves):
        path = folder / f"stage1-{index:03d}.wav"
        sf.write(path, wave, sample_rate)
        paths.append(path)
    return paths
