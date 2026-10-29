"""Intent: predict the meaning of what the model is about to say.

Step 1 of reports/intent-spec.md. A small head reads the dialogue state h_t and
predicts z_t, the representation an EMA copy of the ear assigns to the model's
own speech over the next `horizon` frames. The target is a slow, gradient-free
copy of the ear (as in data2vec / JEPA), so the model learns what its reply will
mean to a listener rather than which Mimi code comes next. Everything stays
audio-only.
"""

import copy
import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .model import DauysHybrid
from .v02_data import CorpusDataset, CorpusRecord, collate_cases
from .v02_training import masked_mean


@dataclass(frozen=True)
class IntentConfig:
    horizon: int = 12
    weight: float = 1.0
    hidden: int = 1024
    ema_start: float = 0.996
    ema_end: float = 0.9999

    def validate(self) -> None:
        if self.horizon < 1 or self.hidden < 1 or self.weight < 0:
            raise ValueError("Invalid intent horizon, width or weight")
        if not 0 <= self.ema_start <= self.ema_end < 1:
            raise ValueError("EMA decay must rise within [0, 1)")

    def ema_decay(self, progress: float) -> float:
        """Cosine ramp from ema_start to ema_end over training progress in [0, 1]."""
        ramp = 0.5 * (1 - math.cos(math.pi * min(max(progress, 0.0), 1.0)))
        return self.ema_start + (self.ema_end - self.ema_start) * ramp


class IntentHead(nn.Module):
    def __init__(self, dialogue_dim: int, target_dim: int, hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dialogue_dim, hidden), nn.GELU(), nn.Linear(hidden, target_dim)
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.net(states)


class EarTarget(nn.Module):
    """Gradient-free EMA copy of the ear: embeddings, projection, listening blocks."""

    def __init__(self, model: DauysHybrid) -> None:
        super().__init__()
        self.embeddings = (
            copy.deepcopy(model.embeddings)
            if model.config.ear_embeddings == "shared"
            else None
        )
        self.ear_embeddings = (
            copy.deepcopy(model.ear_embeddings)
            if model.config.ear_embeddings == "separate"
            else None
        )
        self.role = nn.Parameter(model.roles.weight[0].detach().clone())
        self.projection = copy.deepcopy(model.listening_projection)
        self.blocks = copy.deepcopy(model.listening)
        self.requires_grad_(False)

    def _pairs(self, model: DauysHybrid) -> list[tuple[torch.Tensor, torch.Tensor]]:
        pairs: list[tuple[torch.Tensor, torch.Tensor]] = [
            (self.role, model.roles.weight[0])
        ]
        if self.ear_embeddings is not None:
            assert model.ear_embeddings is not None
            pairs.extend(
                zip(
                    self.ear_embeddings.parameters(),
                    model.ear_embeddings.parameters(),
                    strict=True,
                )
            )
        else:
            assert self.embeddings is not None
            pairs.extend(
                zip(
                    self.embeddings.parameters(),
                    model.embeddings.parameters(),
                    strict=True,
                )
            )
        for mine, online in (
            (self.projection, model.listening_projection),
            (self.blocks, model.listening),
        ):
            pairs.extend(zip(mine.parameters(), online.parameters(), strict=True))
        return pairs

    @torch.no_grad()
    def update(self, model: DauysHybrid, decay: float) -> None:
        for mine, online in self._pairs(model):
            mine.mul_(decay).add_(online.detach().to(mine.dtype), alpha=1 - decay)

    @torch.no_grad()
    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        """Ear states for a [batch, 8, frames] stream, heard as speech (role 0)."""
        if self.ear_embeddings is not None:
            x = sum(self.ear_embeddings[q](codes[:, q]) for q in range(codes.shape[1]))
            x = self.projection(x / math.sqrt(codes.shape[1]))
        else:
            assert self.embeddings is not None
            x = sum(self.embeddings[q](codes[:, q]) for q in range(codes.shape[1]))
            x = self.projection(x / math.sqrt(codes.shape[1]) + self.role)
        for block in self.blocks:
            x = block(x)
        return x


def future_targets(
    states: torch.Tensor, valid: torch.Tensor, horizon: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Layer-normed mean of states over (t, t+horizon]; mask of frames with any."""
    _, frames, width = states.shape
    weights = valid.to(states.dtype)[..., None]
    padded = F.pad(states * weights, (0, 0, 1, horizon))
    counts = F.pad(weights, (0, 0, 1, horizon))
    total = padded.cumsum(dim=1)
    count = counts.cumsum(dim=1)
    # Window (t, t+h] in original frames is (t+1, t+1+h] after the leading pad.
    summed = total[:, 1 + horizon : 1 + horizon + frames] - total[:, 1 : 1 + frames]
    seen = count[:, 1 + horizon : 1 + horizon + frames] - count[:, 1 : 1 + frames]
    mean = summed / seen.clamp_min(1)
    return F.layer_norm(mean, (width,)), seen[..., 0] > 0


def speech_ahead(activity: torch.Tensor, horizon: int) -> torch.Tensor:
    """True at t when the speaker is active on at least one frame in (t, t+h]."""
    active = F.pad(activity.float(), (0, horizon))
    window = active.unfold(1, horizon, 1)[:, 1 : activity.shape[1] + 1]
    return window.amax(dim=-1) > 0


@dataclass
class IntentLoss:
    loss: torch.Tensor
    cosine: torch.Tensor
    prediction_std: torch.Tensor
    frames: torch.Tensor


def intent_loss(
    prediction: torch.Tensor,
    target_states: torch.Tensor,
    valid: torch.Tensor,
    output_activity: torch.Tensor,
    has_b: torch.Tensor,
    horizon: int,
) -> IntentLoss:
    targets, has_future = future_targets(target_states, valid, horizon)
    mask = valid & has_future & speech_ahead(output_activity, horizon) & has_b[:, None]
    prediction = prediction.float()
    targets = targets.float()
    values = (prediction - targets).pow(2).mean(dim=-1)
    selected = prediction[mask]
    return IntentLoss(
        loss=masked_mean(values, mask),
        cosine=F.cosine_similarity(selected, targets[mask], dim=-1).mean()
        if mask.any()
        else values.new_zeros(()),
        prediction_std=selected.std(dim=0).mean()
        if int(mask.sum()) > 1
        else values.new_zeros(()),
        frames=mask.sum(),
    )


class IntentModule(nn.Module):
    """Everything step 1 adds next to the base model; never used at inference."""

    def __init__(self, model: DauysHybrid, config: IntentConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.head = IntentHead(
            model.config.dialogue_dim, model.config.listening_dim, config.hidden
        )
        self.target = EarTarget(model)

    def loss(
        self,
        dialogue_states: torch.Tensor,
        output_codes: torch.Tensor,
        valid: torch.Tensor,
        output_activity: torch.Tensor,
        has_b: torch.Tensor,
    ) -> IntentLoss:
        with torch.autocast(output_codes.device.type, enabled=False):
            target_states = self.target(output_codes)
        return intent_loss(
            self.head(dialogue_states),
            target_states,
            valid,
            output_activity,
            has_b,
            self.config.horizon,
        )


@torch.inference_mode()
def speech_frame_scores(
    model: DauysHybrid,
    intent: IntentModule,
    records: Sequence[CorpusRecord],
    silence_codes: torch.Tensor,
    batch_size: int = 16,
) -> dict[str, float]:
    """CB0 only where B speaks, plus held-out intent loss, cosine and spread."""
    device = next(model.parameters()).device
    dataset = CorpusDataset(list(records))
    totals = dict.fromkeys(("frames", "correct", "nll"), 0.0)
    intent_sums = dict.fromkeys(("loss", "cosine", "std", "frames"), 0.0)
    for start in range(0, len(dataset), batch_size):
        cases = [
            dataset[i] for i in range(start, min(start + batch_size, len(dataset)))
        ]
        batch = collate_cases(cases, silence_codes.cpu()).to(device)
        output = model(batch.user_codes, batch.output_codes, batch.initial_output_codes)
        speech = batch.valid_frames & batch.output_activity.bool()
        cb0 = output.cb0_logits.float()
        target = batch.output_codes[:, 0]
        totals["nll"] += float(
            F.cross_entropy(cb0[speech], target[speech], reduction="sum")
        )
        totals["correct"] += float(((cb0.argmax(-1) == target) & speech).sum())
        totals["frames"] += float(speech.sum())
        scored = intent.loss(
            output.dialogue_states,
            batch.output_codes,
            batch.valid_frames,
            batch.output_activity,
            torch.ones(batch.valid_frames.shape[0], dtype=torch.bool, device=device),
        )
        weight = float(scored.frames)
        intent_sums["loss"] += float(scored.loss) * weight
        intent_sums["cosine"] += float(scored.cosine) * weight
        intent_sums["std"] += float(scored.prediction_std) * weight
        intent_sums["frames"] += weight
    frames = max(totals["frames"], 1.0)
    weight = max(intent_sums["frames"], 1.0)
    return {
        "eval/speech/heldout_cb0_accuracy": totals["correct"] / frames,
        "eval/speech/heldout_cb0_nll": totals["nll"] / frames,
        "eval/intent/heldout_loss": intent_sums["loss"] / weight,
        "eval/intent/heldout_cosine": intent_sums["cosine"] / weight,
        "eval/intent/heldout_prediction_std": intent_sums["std"] / weight,
    }
