"""V10 E0b: two ways for the partner to reach one tower, for an overfit test.

``rev2`` sums both channels into the input of every frame. ``rev1`` keeps its own
channel in the tower and hears the partner through a jointly trained ear and
cross blocks. In both, ``h[t]`` has seen frames up to ``t - 1`` of both channels,
so the same frame of the partner never reaches the prediction of ``own[t]``.
The recipe is the "v10-E0b" brief in Airon.
"""

from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml
from torch import nn
from torch.nn import functional as F

from .model import DepthDecoder, FrameAttention, MambaBlock
from .v10_step1 import SelfAttentionBlock, _masked_sum, _token_ce

SCHEMES = ("rev2", "rev1")
MAMBA_PER_ATTENTION = 3
VOCAB = 2048
BOS = VOCAB
DEPTH_CODEBOOKS = 7
PARTNER_WEIGHT = 0.4
ACTIVITY_WEIGHT = 0.1
ACTIVITY_BOS_REV2 = 4
ACTIVITY_BOS_REV1 = 2
ACTIVITY_HORIZONS = (1, 12)  # frames past the last one the tower has seen
ACTIVITY_OUTPUTS = 2 * len(ACTIVITY_HORIZONS)  # own and partner at each horizon


@dataclass(frozen=True)
class E0bConfig:
    name: str
    scheme: str
    width: int
    repeats: int
    attention_heads: int
    ffn_dim: int
    attention_window: int
    mamba_d_state: int = 64
    ear_layers: int = 0
    cross_every: int = 0
    depth_dim: int = 256
    depth_heads: int = 8
    depth_layers: int = 2
    max_trainable_parameters: int = 50_000_000

    def validate(self) -> None:
        if self.scheme not in SCHEMES:
            raise ValueError("The scheme must be rev2 or rev1")
        if self.width % self.attention_heads:
            raise ValueError("The width must be divisible by the attention heads")
        if self.width % 32:
            raise ValueError("Mamba-2 needs 2 * width divisible by its head size 64")
        if self.scheme == "rev1" and (self.ear_layers < 1 or self.cross_every < 1):
            raise ValueError("rev1 needs an ear and cross blocks")
        if self.scheme == "rev2" and (self.ear_layers or self.cross_every):
            raise ValueError("rev2 has no ear and no cross blocks")

    @property
    def layers(self) -> int:
        return self.repeats * (MAMBA_PER_ATTENTION + 1)

    @classmethod
    def for_scheme(cls, scheme: str) -> "E0bConfig":
        name = f"configs/dauys-v1.0-e0b-{scheme}.yaml"
        with files("dauys").joinpath(name).open("r", encoding="utf-8") as handle:
            return cls._from_mapping(yaml.safe_load(handle))

    @classmethod
    def from_yaml(cls, path: str | Path) -> "E0bConfig":
        with Path(path).open(encoding="utf-8") as handle:
            return cls._from_mapping(yaml.safe_load(handle))

    @classmethod
    def _from_mapping(cls, data: dict) -> "E0bConfig":
        model = data["model"]
        config = cls(
            name=str(model["name"]),
            scheme=str(model["scheme"]),
            width=int(model["width"]),
            repeats=int(model["repeats"]),
            attention_heads=int(model["attention_heads"]),
            ffn_dim=int(model["ffn_dimension"]),
            attention_window=int(data["streaming"]["attention_window_frames"]),
            mamba_d_state=int(model.get("mamba_d_state", 64)),
            ear_layers=int(model.get("ear_layers", 0)),
            cross_every=int(model.get("cross_every", 0)),
            depth_dim=int(model["depth_dim"]),
            depth_heads=int(model["depth_heads"]),
            depth_layers=int(model["depth_layers"]),
            max_trainable_parameters=int(model["max_trainable_parameters"]),
        )
        config.validate()
        return config


def flops_per_frame(config: E0bConfig, frames: int = 940) -> int:
    """Rough operations per frame of the tower, ear and heads, for the size check.

    The depth decoder is left out: it is the same in both schemes and sees a sample
    of the frames. ``frames`` is the mean training window, which sets the attention
    cost until it reaches the window.
    """
    width = config.width
    inner = 2 * width
    state = config.mamba_d_state
    conv = inner + 2 * state
    mamba = (
        width * (2 * inner + 2 * state + inner // 64)
        + 4 * conv
        + 2 * inner * state
        + inner * width
    )
    span = 2 * width * min(frames, config.attention_window)
    attention = 4 * width * width + span
    swa = attention + 3 * config.ffn_dim * width
    tower = (
        config.repeats * (MAMBA_PER_ATTENTION * mamba + swa)
        + config.ear_layers * mamba
        + _cross_count(config) * attention
    )
    return tower + 2 * width * VOCAB + width * ACTIVITY_OUTPUTS


def _cross_count(config: E0bConfig) -> int:
    return config.layers // config.cross_every if config.cross_every else 0


class CrossBlock(nn.Module):
    """Tower states attend to the ear's states up to the same frame."""

    def __init__(self, width: int, heads: int, window: int):
        super().__init__()
        self.norm = nn.RMSNorm(width)
        self.attention = FrameAttention(width, width, heads, window)

    def forward(
        self, x: torch.Tensor, ear: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor:
        return x + self.attention(self.norm(x), ear, positions, positions)


@dataclass
class E0bOutput:
    own_logits: torch.Tensor
    partner_logits: torch.Tensor
    activity_logits: torch.Tensor
    hidden: torch.Tensor


@dataclass
class E0bLosses:
    total: torch.Tensor
    own: torch.Tensor
    partner: torch.Tensor
    activity: torch.Tensor
    depth: torch.Tensor


def _shifted(tokens: torch.Tensor, bos: int) -> torch.Tensor:
    return torch.cat((tokens.new_full((tokens.shape[0], 1), bos), tokens[:, :-1]), 1)


class E0bModel(nn.Module):
    """One causal tower, Mamba-2 : frame attention = 3 : 1, with all output heads."""

    def __init__(self, config: E0bConfig):
        super().__init__()
        config.validate()
        self.config = config
        width = config.width
        self.emb_own = nn.Embedding(VOCAB + 1, width)
        self.emb_partner = nn.Embedding(VOCAB + 1, width)
        self.emb_activity = nn.Embedding(
            ACTIVITY_BOS_REV2 + 1 if config.scheme == "rev2" else ACTIVITY_BOS_REV1 + 1,
            width,
        )
        blocks: list[nn.Module] = []
        for _ in range(config.repeats):
            blocks += [
                MambaBlock(width, 0.0, config.mamba_d_state)
                for _ in range(MAMBA_PER_ATTENTION)
            ]
            blocks.append(
                SelfAttentionBlock(
                    width,
                    config.attention_heads,
                    config.ffn_dim,
                    config.attention_window,
                )
            )
        self.blocks = nn.ModuleList(blocks)
        self.ear: nn.ModuleList | None = None
        self.cross: nn.ModuleDict | None = None
        if config.scheme == "rev1":
            self.emb_ear_activity = nn.Embedding(ACTIVITY_BOS_REV1 + 1, width)
            self.ear = nn.ModuleList(
                MambaBlock(width, 0.0, config.mamba_d_state)
                for _ in range(config.ear_layers)
            )
            self.ear_norm = nn.RMSNorm(width)
            self.cross = nn.ModuleDict(
                {
                    str(index): CrossBlock(
                        width, config.attention_heads, config.attention_window
                    )
                    for index in range(
                        config.cross_every - 1, config.layers, config.cross_every
                    )
                }
            )
        self.final_norm = nn.RMSNorm(width)
        self.head_own = nn.Linear(width, VOCAB)
        self.head_partner = nn.Linear(width, VOCAB)
        self.head_activity = nn.Linear(width, ACTIVITY_OUTPUTS)
        self.depth_embeddings = nn.ModuleList(
            nn.Embedding(VOCAB, config.depth_dim) for _ in range(DEPTH_CODEBOOKS)
        )
        self.depth = DepthDecoder(
            SimpleNamespace(  # type: ignore[arg-type]
                embedding_dim=config.depth_dim,
                dialogue_dim=width,
                depth_dim=config.depth_dim,
                depth_heads=config.depth_heads,
                depth_layers=config.depth_layers,
                vocab_size=VOCAB,
            ),
            self.depth_embeddings,
        )
        if self.trainable_parameters() > config.max_trainable_parameters:
            raise ValueError("Trainable parameter budget exceeded")

    def trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _inputs(
        self,
        own: torch.Tensor,
        partner: torch.Tensor,
        own_active: torch.Tensor,
        partner_active: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        own_in = self.emb_own(_shifted(own, BOS))
        partner_in = self.emb_partner(_shifted(partner, BOS))
        if self.config.scheme == "rev2":
            activity = _shifted(
                2 * own_active.long() + partner_active.long(), ACTIVITY_BOS_REV2
            )
            return own_in + partner_in + self.emb_activity(activity), None
        own_activity = _shifted(own_active.long(), ACTIVITY_BOS_REV1)
        partner_activity = _shifted(partner_active.long(), ACTIVITY_BOS_REV1)
        return (
            own_in + self.emb_activity(own_activity),
            partner_in + self.emb_ear_activity(partner_activity),
        )

    def forward(
        self,
        own: torch.Tensor,
        partner: torch.Tensor,
        own_active: torch.Tensor,
        partner_active: torch.Tensor,
    ) -> E0bOutput:
        """Teacher forcing: every input holds the true frames ``[batch, frames]``."""
        if own.ndim != 2 or partner.shape != own.shape:
            raise ValueError("Both channels need one [batch, frames] token per frame")
        x, ear = self._inputs(own, partner, own_active, partner_active)
        positions = torch.arange(own.shape[1], device=own.device)
        if self.ear is not None and ear is not None:
            for block in self.ear:
                ear = block(ear)
            ear = self.ear_norm(ear)
        for index, block in enumerate(self.blocks):
            x = (
                block(x, positions)
                if isinstance(block, SelfAttentionBlock)
                else block(x)
            )
            if self.cross is not None and ear is not None and str(index) in self.cross:
                x = self.cross[str(index)](x, ear, positions)
        h = self.final_norm(x)
        return E0bOutput(
            own_logits=self.head_own(h),
            partner_logits=self.head_partner(h),
            activity_logits=self.head_activity(h),
            hidden=h,
        )

    def depth_logits(
        self, prefix: torch.Tensor, hidden: torch.Tensor, frames: torch.Tensor
    ) -> torch.Tensor:
        """CB1..7 at the picked frames, from true CB0..6, with no gradient to the tower.

        ``prefix`` is ``[batch, frames, 7]``; ``frames`` is ``[picks, 2]`` of
        batch and frame indices.
        """
        picked = prefix[frames[:, 0], frames[:, 1]][None]
        context = hidden.detach()[frames[:, 0], frames[:, 1]][None]
        return self.depth(picked, context)[0]

    def loss(self, output: E0bOutput, batch: dict[str, torch.Tensor]) -> E0bLosses:
        own, partner = batch["own"], batch["partner"]
        valid = (
            torch.arange(own.shape[1], device=own.device)[None]
            < batch["lengths"][:, None]
        )
        ce_own = _masked_sum(_token_ce(output.own_logits, own), valid)
        ce_partner = _masked_sum(_token_ce(output.partner_logits, partner), valid)
        activity = F.binary_cross_entropy_with_logits(
            output.activity_logits,
            batch["activity_target"].float(),
            reduction="none",
        )
        activity = _masked_sum(activity, batch["activity_valid"])
        picks = batch["depth_frames"]
        depth = self.depth_logits(batch["depth_prefix"], output.hidden, picks)
        targets = batch["depth_target"][picks[:, 0], picks[:, 1]]
        ce_depth = F.cross_entropy(depth.reshape(-1, VOCAB), targets.reshape(-1))
        total = (
            ce_own + PARTNER_WEIGHT * ce_partner + ACTIVITY_WEIGHT * activity + ce_depth
        )
        return E0bLosses(total, ce_own, ce_partner, activity, ce_depth)

    @torch.no_grad()
    def sample_frame_codes(
        self,
        cb0: torch.Tensor,
        hidden: torch.Tensor,
        *,
        temperature: float = 0.8,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Sample CB1..7 of one frame given its CB0; ``hidden`` is ``[batch, width]``."""
        batch = cb0.shape[0]
        context = hidden[:, None]
        codes = cb0[:, None, None]
        for book in range(DEPTH_CODEBOOKS):
            logits = self.depth(codes, context)[:, 0, book] / temperature
            token = torch.multinomial(logits.softmax(-1), 1, generator=generator)
            codes = torch.cat((codes, token[:, None]), dim=-1)
        return codes[:, 0].reshape(batch, DEPTH_CODEBOOKS + 1)
