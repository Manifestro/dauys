"""Masked audio-only training objective."""

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from .config import ModelConfig
from .model import ModelOutput


@dataclass
class Losses:
    total: torch.Tensor
    cb0: torch.Tensor
    depth: torch.Tensor
    activity: torch.Tensor


def training_loss(
    output: ModelOutput,
    target_codes: torch.Tensor,
    target_activity: torch.Tensor,
    valid_frames: torch.Tensor,
    config: ModelConfig,
) -> Losses:
    batch, codebooks, frames = target_codes.shape
    if codebooks != 8 or target_activity.shape != (batch, frames):
        raise ValueError("Invalid target shapes")
    if valid_frames.shape != (batch, frames) or not valid_frames.any():
        raise ValueError("At least one valid frame is required")
    if output.depth_logits is None:
        raise ValueError("Teacher-forced depth logits are required")
    mask = valid_frames.to(output.cb0_logits.dtype)
    cb0_per_frame = F.cross_entropy(
        output.cb0_logits.reshape(-1, config.vocab_size),
        target_codes[:, 0].reshape(-1),
        reduction="none",
    ).reshape(batch, frames)
    cb0 = (cb0_per_frame * mask).sum() / mask.sum()
    depth_per_code = F.cross_entropy(
        output.depth_logits.reshape(-1, config.vocab_size),
        target_codes[:, 1:].transpose(1, 2).reshape(-1),
        reduction="none",
    ).reshape(batch, frames, 7)
    depth = (depth_per_code * mask[:, :, None]).sum() / (mask.sum() * 7)
    activity_per_frame = F.cross_entropy(
        output.activity_logits.reshape(-1, 2),
        target_activity.reshape(-1).long(),
        reduction="none",
    ).reshape(batch, frames)
    activity = (activity_per_frame * mask).sum() / mask.sum()
    total = (
        config.cb0_weight * cb0
        + config.depth_weight * depth
        + config.activity_weight * activity
    )
    return Losses(total, cb0, depth, activity)
