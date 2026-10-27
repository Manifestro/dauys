"""Independent codebook and activity decisions for causal generation."""

from dataclasses import dataclass
from typing import Literal

import torch
from torch.nn import functional as F

TokenMode = Literal["greedy", "sample", "activity_gated"]
ActivityMode = Literal["argmax", "sample", "hysteresis"]


@dataclass(frozen=True)
class DecodingConfig:
    cb0_mode: TokenMode = "greedy"
    depth_mode: TokenMode = "greedy"
    activity_mode: ActivityMode = "argmax"
    cb0_temperature: float = 0.6
    cb0_top_k: int = 50
    depth_temperature: float = 0.8
    depth_top_k: int = 250
    activity_temperature: float = 1.0
    theta_on: float = 0.35
    theta_off: float = 0.2
    min_speaking_frames: int = 0
    min_silent_frames: int = 0

    @classmethod
    def v02_default(cls) -> "DecodingConfig":
        return cls(
            cb0_mode="activity_gated",
            depth_mode="sample",
            activity_mode="hysteresis",
        )

    def validate(self) -> None:
        if self.cb0_mode not in ("greedy", "sample", "activity_gated"):
            raise ValueError("Invalid CB0 decoding mode")
        if self.depth_mode not in ("greedy", "sample"):
            raise ValueError("Invalid depth decoding mode")
        if self.activity_mode not in ("argmax", "sample", "hysteresis"):
            raise ValueError("Invalid activity decoding mode")
        if (
            min(self.cb0_temperature, self.depth_temperature, self.activity_temperature)
            <= 0
        ):
            raise ValueError("Temperatures must be positive")
        if self.cb0_top_k < 1 or self.depth_top_k < 1:
            raise ValueError("Top-k must be positive")
        if not 0 <= self.theta_off <= self.theta_on <= 1:
            raise ValueError("Hysteresis requires theta_off <= theta_on")
        if self.min_speaking_frames < 0 or self.min_silent_frames < 0:
            raise ValueError("Minimum durations cannot be negative")


@dataclass
class ActivityState:
    speaking: torch.Tensor
    speaking_frames: torch.Tensor
    silent_frames: torch.Tensor

    @classmethod
    def silent(cls, batch: int, device: torch.device) -> "ActivityState":
        return cls(
            speaking=torch.zeros(batch, dtype=torch.bool, device=device),
            speaking_frames=torch.zeros(batch, dtype=torch.long, device=device),
            silent_frames=torch.full(
                (batch,), 1_000_000, dtype=torch.long, device=device
            ),
        )


@dataclass(frozen=True)
class HysteresisScore:
    theta_on: float
    theta_off: float
    median_onset_ms: float
    yield_within_500ms: float
    false_starts: float


def normalize_step_activity(value: torch.Tensor) -> torch.Tensor:
    """Normalize legacy activity logits or configured decisions to booleans."""
    if value.ndim == 2 and value.shape[1] == 2:
        return value.argmax(dim=-1).bool()
    if value.ndim == 1 and value.dtype == torch.bool:
        return value
    raise ValueError("Expected legacy [batch, 2] logits or [batch] decisions")


def choose_hysteresis(
    scores: list[HysteresisScore], argmax_false_starts: float
) -> HysteresisScore:
    """Freeze the fastest E2 onset satisfying the E5 yield and false-start gates."""
    grid = {
        (on, off) for on in (0.25, 0.35, 0.5) for off in (0.1, 0.2, 0.3) if off < on
    }
    eligible = [
        score
        for score in scores
        if (score.theta_on, score.theta_off) in grid
        and score.yield_within_500ms >= 0.8
        and score.false_starts <= argmax_false_starts
    ]
    if not eligible:
        raise ValueError("No hysteresis setting passes yield and false-start gates")
    return min(
        eligible,
        key=lambda item: (item.median_onset_ms, item.false_starts, item.theta_on),
    )


def sample_logits(
    logits: torch.Tensor,
    temperature: float,
    top_k: int,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    if temperature <= 0 or top_k < 1:
        raise ValueError("Temperature and top-k must be positive")
    width = min(top_k, logits.shape[-1])
    values, indices = torch.topk(logits.float() / temperature, width, dim=-1)
    probabilities = F.softmax(values, dim=-1)
    selected = torch.multinomial(probabilities, 1, generator=generator)
    return indices.gather(-1, selected).squeeze(-1)


def activity_decision(
    logits: torch.Tensor,
    config: DecodingConfig,
    state: ActivityState,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    if logits.ndim != 2 or logits.shape != (state.speaking.shape[0], 2):
        raise ValueError("Activity logits must have shape [batch, 2]")
    if config.activity_mode == "argmax":
        speaking = logits.argmax(dim=-1).bool()
    elif config.activity_mode == "sample":
        speaking = sample_logits(
            logits, config.activity_temperature, 2, generator
        ).bool()
    elif (
        config.theta_on == config.theta_off == 0.5
        and config.min_speaking_frames == config.min_silent_frames == 0
    ):
        speaking = logits.argmax(dim=-1).bool()
    else:
        probability = F.softmax(logits.float(), dim=-1)[:, 1]
        can_start = state.silent_frames >= config.min_silent_frames
        can_stop = state.speaking_frames >= config.min_speaking_frames
        speaking = torch.where(
            state.speaking,
            ~(can_stop & (probability <= config.theta_off)),
            can_start & (probability >= config.theta_on),
        )
    state.speaking_frames = torch.where(
        speaking, state.speaking_frames + 1, torch.zeros_like(state.speaking_frames)
    )
    state.silent_frames = torch.where(
        speaking, torch.zeros_like(state.silent_frames), state.silent_frames + 1
    )
    state.speaking = speaking
    return speaking


def cb0_decision(
    logits: torch.Tensor,
    speaking: torch.Tensor,
    config: DecodingConfig,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    greedy = logits.argmax(dim=-1)
    if config.cb0_mode == "greedy":
        return greedy
    if config.cb0_mode == "sample":
        return sample_logits(
            logits, config.cb0_temperature, config.cb0_top_k, generator
        )
    if not speaking.any():
        return greedy
    result = greedy.clone()
    result[speaking] = sample_logits(
        logits[speaking], config.cb0_temperature, config.cb0_top_k, generator
    )
    return result
