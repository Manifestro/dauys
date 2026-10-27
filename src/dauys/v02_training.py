"""Stage-aware Hybrid v0.2 objectives on synchronized Mimi frames."""

from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F

from .model import DauysHybrid, previous_frames

Source = Literal["1a_on_A", "1a_on_B", "1b", "hohere"]
Stage = Literal["pretrain", "dialogue", "posttrain"]


@dataclass(frozen=True)
class Event:
    type: str
    speaker: str
    start_frame: int
    end_frame: int
    related: str | None = None


@dataclass(frozen=True)
class V02Batch:
    user_codes: torch.Tensor
    output_codes: torch.Tensor
    user_activity: torch.Tensor
    output_activity: torch.Tensor
    valid_frames: torch.Tensor
    initial_output_codes: torch.Tensor
    silence_codes: torch.Tensor
    sources: tuple[Source, ...]
    conversation_ids: tuple[str, ...]
    events: tuple[tuple[Event, ...], ...]
    negative_user_codes: torch.Tensor | None = None
    negative_ids: tuple[str, ...] | None = None

    def to(self, device: torch.device) -> "V02Batch":
        return self._map_tensors(lambda value: value.to(device, non_blocking=True))

    def pin_memory(self) -> "V02Batch":
        return self._map_tensors(lambda value: value.pin_memory())

    def _map_tensors(
        self, transform: Callable[[torch.Tensor], torch.Tensor]
    ) -> "V02Batch":
        return V02Batch(
            user_codes=transform(self.user_codes),
            output_codes=transform(self.output_codes),
            user_activity=transform(self.user_activity),
            output_activity=transform(self.output_activity),
            valid_frames=transform(self.valid_frames),
            initial_output_codes=transform(self.initial_output_codes),
            silence_codes=transform(self.silence_codes),
            sources=self.sources,
            conversation_ids=self.conversation_ids,
            events=self.events,
            negative_user_codes=(
                transform(self.negative_user_codes)
                if self.negative_user_codes is not None
                else None
            ),
            negative_ids=self.negative_ids,
        )

    def validate(self) -> None:
        batch, codebooks, frames = self.user_codes.shape
        if codebooks != 8 or self.output_codes.shape != (batch, 8, frames):
            raise ValueError("Both token streams must have shape [batch, 8, frames]")
        if self.user_activity.shape != (
            batch,
            frames,
        ) or self.output_activity.shape != (batch, frames):
            raise ValueError("Activity must have shape [batch, frames]")
        if (
            self.valid_frames.shape != (batch, frames)
            or self.valid_frames.dtype != torch.bool
        ):
            raise ValueError("Valid frames must be a boolean [batch, frames] tensor")
        if self.initial_output_codes.shape != (
            batch,
            8,
        ) or self.silence_codes.shape != (batch, 8):
            raise ValueError("Initial and silence codes must have shape [batch, 8]")
        if not all(
            len(items) == batch
            for items in (self.sources, self.conversation_ids, self.events)
        ):
            raise ValueError("Source, ID and event lists must match batch size")
        if (
            self.negative_user_codes is not None
            and self.negative_user_codes.shape != self.user_codes.shape
        ):
            raise ValueError("Negative user codes must match the input shape")
        if self.negative_ids is not None and len(self.negative_ids) != batch:
            raise ValueError("Negative IDs must match batch size")
        if (self.negative_user_codes is None) != (self.negative_ids is None):
            raise ValueError("Negative codes and IDs must be supplied together")


@dataclass(frozen=True)
class Toggles:
    regularization: bool = True
    listening: bool = True
    history_dropout: bool = True
    acoustic_dropout: bool = True
    transition: bool = True
    contrast: bool = True


@dataclass(frozen=True)
class Weights:
    cb0: float
    depth: float
    activity: float
    listening: float
    listening_activity: float
    contrast: float
    ahead: float
    transition: float
    history_dropout: float
    acoustic_dropout: float


@dataclass
class V02Loss:
    total: torch.Tensor
    cb0: torch.Tensor
    depth: torch.Tensor
    activity: torch.Tensor
    listening_cb0: torch.Tensor
    listening_activity: torch.Tensor
    ahead: torch.Tensor
    contrast: torch.Tensor
    delta_cb0_onset: torch.Tensor | None
    raw_cb0_nll: torch.Tensor


def _interpolate(progress: float, values: tuple[float, float, float, float]) -> float:
    points = (0.0, 0.1, 0.4, 1.0)
    for index in range(3):
        if progress <= points[index + 1]:
            ratio = (progress - points[index]) / (points[index + 1] - points[index])
            return values[index] + ratio * (values[index + 1] - values[index])
    return values[-1]


def schedule(progress: float, stage: Stage) -> Weights:
    if not 0 <= progress <= 1 or stage not in ("pretrain", "dialogue", "posttrain"):
        raise ValueError("Invalid training stage or progress")
    if stage == "posttrain":
        progress = 1.0
    return Weights(
        cb0=1.0,
        depth=0.5,
        activity=0.2,
        listening=_interpolate(progress, (1.0, 1.0, 0.5, 0.3)),
        listening_activity=0.2,
        contrast=0.0
        if stage == "pretrain"
        else _interpolate(progress, (0.05, 0.05, 0.3, 0.3)),
        ahead=_interpolate(progress, (0.1, 0.2, 0.2, 0.2)),
        transition=_interpolate(progress, (1.0, 2.0, 3.0, 3.0)),
        history_dropout=_interpolate(progress, (0.1, 0.1, 0.25, 0.25)),
        acoustic_dropout=0.5,
    )


def source_flags(
    sources: tuple[Source, ...], stage: Stage, device: torch.device
) -> dict[str, torch.Tensor]:
    if stage not in ("pretrain", "dialogue", "posttrain"):
        raise ValueError("Invalid training stage")
    allowed = {"1a_on_A", "1a_on_B", "1b", "hohere"}
    if not set(sources) <= allowed:
        raise ValueError("Unknown training source")
    tensor = lambda matches: torch.tensor(matches, dtype=torch.bool, device=device)
    return {
        "b": tensor([source != "1a_on_A" for source in sources]),
        "listen": tensor([source != "1a_on_B" for source in sources]),
        "history": tensor([source != "1a_on_A" for source in sources]),
        "acoustic": tensor([source != "1a_on_B" for source in sources]),
        "transition": tensor([source in ("1b", "hohere") for source in sources]),
        "contrast": tensor(
            [stage != "pretrain" and source == "hohere" for source in sources]
        ),
        "detach_listening": tensor([source == "1a_on_B" for source in sources]),
    }


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if values.shape != mask.shape:
        raise ValueError("Values and mask must have equal shape")
    if values.ndim < 2:
        raise ValueError("Masked means require a batch and time dimension")
    flat_values = values.reshape(values.shape[0], -1)
    flat_mask = mask.reshape(mask.shape[0], -1).to(values.dtype)
    counts = flat_mask.sum(dim=1)
    means = (flat_values * flat_mask).sum(dim=1) / counts.clamp_min(1)
    selected = counts > 0
    return (means * selected).sum() / selected.sum().clamp_min(1)


def transition_mask(
    events: tuple[tuple[Event, ...], ...], frames: int, device: torch.device
) -> torch.Tensor:
    mask = torch.zeros((len(events), frames), dtype=torch.bool, device=device)
    for item, item_events in enumerate(events):
        for event in item_events:
            if event.start_frame < 0 or event.end_frame < event.start_frame:
                raise ValueError("Invalid event frame interval")
            if event.type in ("a_turn_end", "a_interrupts_b"):
                mask[
                    item,
                    max(0, event.start_frame - 6) : min(frames, event.start_frame + 7),
                ] = True
            elif event.type == "backchannel" and event.speaker == "B":
                mask[
                    item, max(0, event.start_frame - 3) : min(frames, event.end_frame)
                ] = True
    return mask


def response_onset_mask(
    events: tuple[tuple[Event, ...], ...],
    output_activity: torch.Tensor,
) -> torch.Tensor:
    """Select explicit response starts, excluding B backchannels and interruptions."""
    batch, frames = output_activity.shape
    mask = torch.zeros((batch, frames), dtype=torch.bool, device=output_activity.device)
    for item, item_events in enumerate(events):
        excluded = tuple(
            event
            for event in item_events
            if event.speaker == "B" and event.type in ("backchannel", "b_interrupts_a")
        )
        for event in item_events:
            if event.type != "response_onset" or event.speaker != "B":
                continue
            start = event.start_frame
            if (
                start < 0
                or start >= frames
                or any(
                    other.start_frame <= start < other.end_frame for other in excluded
                )
            ):
                continue
            stop = start
            while stop < min(frames, start + 25) and bool(output_activity[item, stop]):
                stop += 1
            mask[item, start:stop] = True
    return mask


def history_span_dropout(
    targets: torch.Tensor,
    silence_codes: torch.Tensor,
    valid_frames: torch.Tensor,
    enabled: torch.Tensor,
    rate: float,
    training: bool,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    if not training or rate == 0:
        return targets
    if not 0 <= rate < 1:
        raise ValueError("Invalid history dropout rate")
    result = targets.clone()
    for item in range(targets.shape[0]):
        if not bool(enabled[item]):
            continue
        length = int(valid_frames[item].sum())
        if length < 6:
            continue
        covered = torch.zeros(length, dtype=torch.bool, device=targets.device)
        budget = round(length * rate)
        attempts = 0
        while int(covered.sum()) < budget and attempts < length:
            span = min(length, int(torch.randint(6, 26, (), generator=generator)))
            start = int(torch.randint(0, length - span + 1, (), generator=generator))
            covered[start : start + span] = True
            attempts += 1
        result[item, :, :length] = torch.where(
            covered[None], silence_codes[item, :, None], result[item, :, :length]
        )
    return result


def _contrast_negatives(
    batch: V02Batch, enabled: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    alternatives: list[int] = []
    valid_items: list[bool] = []
    for position, current in enumerate(batch.conversation_ids):
        candidate = None
        for offset in range(1, len(batch.conversation_ids)):
            index = (position + offset) % len(batch.conversation_ids)
            if batch.conversation_ids[index] != current:
                candidate = index
                break
        alternatives.append(0 if candidate is None else candidate)
        valid_items.append(candidate is not None)
    indices = torch.tensor(alternatives, device=batch.user_codes.device)
    fallback = batch.user_codes[indices]
    fallback_valid = torch.tensor(valid_items, device=enabled.device)
    if batch.negative_user_codes is None:
        return fallback, enabled & fallback_valid
    assert batch.negative_ids is not None
    matched = torch.tensor(
        [
            negative != current
            for current, negative in zip(batch.conversation_ids, batch.negative_ids)
        ],
        dtype=torch.bool,
        device=enabled.device,
    )
    negative_codes = torch.where(
        matched[:, None, None], batch.negative_user_codes, fallback
    )
    return negative_codes, enabled & (matched | fallback_valid)


def objective(
    model: DauysHybrid,
    batch: V02Batch,
    stage: Stage,
    weights: Weights,
    toggles: Toggles | None = None,
    generator: torch.Generator | None = None,
    history_codes_override: torch.Tensor | None = None,
    measure_delta: bool = False,
) -> V02Loss:
    batch.validate()
    if toggles is None:
        toggles = Toggles()
    if model.config.name != "dauys-hybrid-v0.2":
        raise ValueError("v0.2 objective requires a v0.2 model")
    device = batch.user_codes.device
    flags = source_flags(batch.sources, stage, device)
    acoustic_keep_mask = None
    if toggles.acoustic_dropout and model.training and weights.acoustic_dropout > 0:
        acoustic_keep_mask = (
            torch.rand(
                batch.user_codes.shape[0],
                batch.user_codes.shape[-1],
                7,
                device=device,
            )
            >= weights.acoustic_dropout
        )
        acoustic_keep_mask |= ~flags["acoustic"][:, None, None]
    if (
        history_codes_override is not None
        and history_codes_override.shape != batch.output_codes.shape
    ):
        raise ValueError("Own-history codes must match target shape")
    history = history_span_dropout(
        batch.output_codes
        if history_codes_override is None
        else history_codes_override,
        batch.silence_codes,
        batch.valid_frames,
        flags["history"],
        weights.history_dropout if toggles.history_dropout else 0.0,
        model.training,
        generator,
    )
    output = model(
        batch.user_codes,
        batch.output_codes,
        batch.initial_output_codes,
        history_codes=history,
        detach_listening=flags["detach_listening"],
        acoustic_dropout=weights.acoustic_dropout if toggles.acoustic_dropout else 0.0,
        acoustic_dropout_items=flags["acoustic"],
        acoustic_keep_mask=acoustic_keep_mask,
    )
    batch_size, _, frames = batch.output_codes.shape
    valid = batch.valid_frames
    b_mask = valid & flags["b"][:, None]
    transition = transition_mask(batch.events, frames, device)
    transition &= flags["transition"][:, None] if toggles.transition else False
    multiplier = 1 + (weights.transition - 1) * transition.to(output.cb0_logits.dtype)
    cb0_logits = output.cb0_logits.reshape(-1, model.config.vocab_size)
    cb0_targets = batch.output_codes[:, 0].reshape(-1)
    raw_cb0 = F.cross_entropy(cb0_logits, cb0_targets, reduction="none").reshape(
        batch_size, frames
    )
    cb0_per_frame = F.cross_entropy(
        cb0_logits,
        cb0_targets,
        reduction="none",
        label_smoothing=0.05 if toggles.regularization and model.training else 0.0,
    ).reshape(batch_size, frames)
    cb0 = masked_mean(cb0_per_frame * multiplier, b_mask)
    if output.depth_logits is None:
        raise ValueError("Depth logits are required")
    depth_values = F.cross_entropy(
        output.depth_logits.reshape(-1, model.config.vocab_size),
        batch.output_codes[:, 1:].transpose(1, 2).reshape(-1),
        reduction="none",
    ).reshape(batch_size, frames, 7)
    depth = masked_mean(depth_values.mean(dim=-1), b_mask)
    activity_values = F.cross_entropy(
        output.activity_logits.reshape(-1, 2),
        batch.output_activity.reshape(-1).long(),
        reduction="none",
    ).reshape(batch_size, frames)
    activity = masked_mean(activity_values * multiplier, b_mask)
    zero = output.cb0_logits.sum() * 0
    listening_cb0 = zero
    listening_activity = zero
    if toggles.listening and output.listening_cb0_logits is not None and frames > 1:
        listen_mask = valid[:, :-1] & valid[:, 1:] & flags["listen"][:, None]
        logits = output.listening_cb0_logits[:, :-1]
        values = F.cross_entropy(
            logits.reshape(-1, model.config.vocab_size),
            batch.user_codes[:, 0, 1:].reshape(-1),
            reduction="none",
        ).reshape(batch_size, frames - 1)
        listening_cb0 = masked_mean(values, listen_mask)
        if output.listening_activity_logits is not None:
            values = F.cross_entropy(
                output.listening_activity_logits[:, :-1].reshape(-1, 2),
                batch.user_activity[:, 1:].reshape(-1).long(),
                reduction="none",
            ).reshape(batch_size, frames - 1)
            listening_activity = masked_mean(values, listen_mask)
    ahead = zero
    if toggles.transition and output.activity_ahead_logits is not None:
        horizons = output.activity_ahead_logits.shape[2]
        horizon_losses = []
        horizon_masks = []
        for horizon in range(1, horizons + 1):
            if horizon >= frames:
                break
            target = F.one_hot(batch.output_activity[:, horizon:].long(), 2).to(
                output.activity_ahead_logits.dtype
            )
            logits = output.activity_ahead_logits[:, :-horizon, horizon - 1]
            values = F.binary_cross_entropy_with_logits(
                logits, target, reduction="none"
            ).mean(dim=-1)
            mask = (
                valid[:, :-horizon] & valid[:, horizon:] & flags["transition"][:, None]
            )
            horizon_losses.append(masked_mean(values, mask))
            horizon_masks.append(mask.any())
        if any(bool(item) for item in horizon_masks):
            ahead = sum(horizon_losses) / sum(bool(item) for item in horizon_masks)
    contrast = zero
    delta = None
    if (toggles.contrast and weights.contrast > 0) or measure_delta:
        negative_codes, contrast_items = _contrast_negatives(batch, flags["contrast"])
        onset = response_onset_mask(batch.events, batch.output_activity) & valid
        contrast_mask = onset & contrast_items[:, None]
        if contrast_mask.any():
            context = (
                nullcontext()
                if toggles.contrast and weights.contrast > 0
                else torch.no_grad()
            )
            with context:
                negative_states = model.encode_listening(
                    negative_codes,
                    weights.acoustic_dropout if toggles.acoustic_dropout else 0.0,
                    flags["acoustic"],
                    acoustic_keep_mask,
                    previous_frames(history, batch.initial_output_codes),
                )
                negative_context = model.run_dialogue(
                    negative_states,
                    batch.output_codes,
                    batch.initial_output_codes,
                    history,
                )
                negative_cb0 = model.cb0_head(negative_context)
                negative_nll = F.cross_entropy(
                    negative_cb0.reshape(-1, model.config.vocab_size),
                    cb0_targets,
                    reduction="none",
                ).reshape(batch_size, frames)
            per_item_mask = contrast_mask.to(raw_cb0.dtype)
            counts = per_item_mask.sum(dim=1)
            positive = (raw_cb0 * per_item_mask).sum(dim=1) / counts.clamp_min(1)
            negative = (negative_nll * per_item_mask).sum(dim=1) / counts.clamp_min(1)
            selected = counts > 0
            delta_items = negative - positive
            if toggles.contrast and weights.contrast > 0:
                contrast = F.relu(0.3 - delta_items[selected]).mean()
            delta = delta_items[selected].detach().mean()
    total = (
        weights.cb0 * cb0
        + weights.depth * depth
        + weights.activity * activity
        + (
            weights.listening
            * (listening_cb0 + weights.listening_activity * listening_activity)
            if toggles.listening
            else zero
        )
        + (weights.ahead * ahead if toggles.transition else zero)
        + (weights.contrast * contrast if toggles.contrast else zero)
    )
    return V02Loss(
        total=total,
        cb0=cb0,
        depth=depth,
        activity=activity,
        listening_cb0=listening_cb0,
        listening_activity=listening_activity,
        ahead=ahead,
        contrast=contrast,
        delta_cb0_onset=delta,
        raw_cb0_nll=masked_mean(raw_cb0, b_mask),
    )


def optimizer_groups(model: nn.Module, weight_decay: float = 0.05) -> list[dict]:
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    excluded = {
        id(parameter)
        for module in model.modules()
        if isinstance(module, (nn.Embedding, nn.RMSNorm, nn.LayerNorm))
        for parameter in module.parameters(recurse=False)
    }
    for _name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim >= 2 and id(parameter) not in excluded:
            decay.append(parameter)
        else:
            no_decay.append(parameter)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
