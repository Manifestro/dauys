"""V10 step 1: one stream that also predicts an inferred monologue.

The recording gives two tokens per Mimi frame: ``s`` (the CB0 code) and ``m``
(the ZeroSyl monologue: a unit, SIL or PAD). One hybrid body sees only frames up
to ``t - 1``; ``h[t]`` predicts ``m[t]``, then ``s[t]`` with ``emb_m(m[t])``
added. Only the content of ``m`` changes between the arms (see ``v10_step1_data``).
The recipe is fixed in ``runs/V10-STEP1/plan.md``.
"""

from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import cast

import torch
import yaml
from torch import nn
from torch.nn import functional as F

from .model import AttentionCache, FrameAttention, MambaBlock, SwiGLU

ARMS = ("A", "B-sil", "B")  # scored on p(s), the arms of the decision rule
TRAIN_ARMS = (*ARMS, "m-only")  # m-only reproduces ZeroSyl, without s
MAMBA_PER_ATTENTION = 5


@dataclass(frozen=True)
class V10Step1Config:
    name: str
    width: int
    units: int
    repeats: int
    attention_window: int
    attention_heads: int
    ffn_dim: int
    s_vocab_size: int = 2048
    dropout: float = 0.0
    attention_dropout: float = 0.0
    mamba_d_state: int = 64
    max_trainable_parameters: int = 200_000_000

    @property
    def m_vocab_size(self) -> int:
        """Units ``0..N-1``, then SIL, PAD and BOS in every arm."""
        return self.units + 3

    @property
    def sil_id(self) -> int:
        return self.units

    @property
    def pad_id(self) -> int:
        return self.units + 1

    @property
    def m_bos_id(self) -> int:
        return self.units + 2

    @property
    def s_vocab_with_bos(self) -> int:
        return self.s_vocab_size + 1

    @property
    def s_bos_id(self) -> int:
        return self.s_vocab_size

    def validate(self) -> None:
        if self.width % self.attention_heads:
            raise ValueError("The width must be divisible by the attention heads")
        if min(self.units, self.repeats, self.attention_window, self.ffn_dim) < 1:
            raise ValueError("Units, repeats, window and FFN sizes must be positive")
        if self.s_vocab_size != 2048:
            raise ValueError("The CB0 stream is Mimi's 2048-token codebook")
        if not 0 <= self.dropout < 1 or not 0 <= self.attention_dropout < 1:
            raise ValueError("Invalid dropout probability")
        if self.max_trainable_parameters < 1:
            raise ValueError("Invalid parameter cap")

    @classmethod
    def v10_step1(cls) -> "V10Step1Config":
        return cls._from_resource("configs/dauys-v1.0-step1.yaml")

    @classmethod
    def v10_step1_tiny(cls) -> "V10Step1Config":
        return cls._from_resource("configs/dauys-v1.0-step1-tiny.yaml")

    @classmethod
    def _from_resource(cls, name: str) -> "V10Step1Config":
        with files("dauys").joinpath(name).open("r", encoding="utf-8") as handle:
            return cls._from_mapping(yaml.safe_load(handle))

    @classmethod
    def from_yaml(cls, path: str | Path) -> "V10Step1Config":
        with Path(path).open(encoding="utf-8") as handle:
            return cls._from_mapping(yaml.safe_load(handle))

    @classmethod
    def _from_mapping(cls, data: dict) -> "V10Step1Config":
        model = data["model"]
        config = cls(
            name=str(model["name"]),
            width=int(model["width"]),
            units=int(model["units"]),
            repeats=int(model["repeats"]),
            attention_window=int(data["streaming"]["attention_window_frames"]),
            attention_heads=int(model["attention_heads"]),
            ffn_dim=int(model["ffn_dimension"]),
            s_vocab_size=int(data["codec"]["vocab_size"]),
            dropout=float(data.get("regularization", {}).get("dropout", 0.0)),
            attention_dropout=float(
                data.get("regularization", {}).get("attention_dropout", 0.0)
            ),
            mamba_d_state=int(model.get("mamba_d_state", 64)),
            max_trainable_parameters=int(
                model.get("max_trainable_parameters", 200_000_000)
            ),
        )
        config.validate()
        return config


class SelfAttentionBlock(nn.Module):
    """Frame self-attention plus SwiGLU, no cross attention."""

    def __init__(
        self,
        width: int,
        heads: int,
        ffn_dim: int,
        window: int,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
    ):
        super().__init__()
        self.self_norm = nn.RMSNorm(width)
        self.self_attention = FrameAttention(
            width, width, heads, window, attention_dropout
        )
        self.ffn_norm = nn.RMSNorm(width)
        self.ffn = SwiGLU(width, ffn_dim)
        self.residual_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        normalized = self.self_norm(x)
        x = x + self.residual_dropout(
            self.self_attention(normalized, normalized, positions, positions)
        )
        return x + self.residual_dropout(self.ffn(self.ffn_norm(x)))

    def step(
        self, x: torch.Tensor, position: int, cache: AttentionCache
    ) -> tuple[torch.Tensor, AttentionCache]:
        update, cache = self.self_attention.step_self(
            self.self_norm(x), position, cache
        )
        x = x + self.residual_dropout(update)
        return x + self.residual_dropout(self.ffn(self.ffn_norm(x))), cache


@dataclass
class V10Output:
    m_logits: torch.Tensor
    s_logits: torch.Tensor
    hidden: torch.Tensor


@dataclass
class V10Losses:
    total: torch.Tensor
    m: torch.Tensor
    s: torch.Tensor
    m_unit: torch.Tensor
    m_sil: torch.Tensor
    m_pad: torch.Tensor


@dataclass
class V10StreamState:
    frame_index: int
    mamba: list[tuple[torch.Tensor, torch.Tensor] | None]
    attention: list[AttentionCache | None]


def _state_batch(state: V10StreamState) -> int:
    sizes = [
        int(value.shape[0])
        for entry in state.mamba
        if entry is not None
        for value in entry
        if value is not None and value.ndim
    ]
    sizes += [
        int(cache.keys.shape[0])
        for cache in state.attention
        if cache is not None and cache.keys is not None
    ]
    return max(sizes) if sizes else 0


def _select(value: torch.Tensor | None, indices: torch.Tensor, batch: int):
    if value is None or not value.ndim or value.shape[0] != batch:
        return value
    return value.index_select(0, indices.to(value.device))


def _masked_sum(values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    picked = values * valid
    return picked.sum() / valid.sum().clamp_min(1)


def _token_ce(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        target.reshape(-1),
        reduction="none",
    ).reshape(target.shape)


class V10Step1Model(nn.Module):
    """Mamba-2 body with one self-attention frame every five Mamba blocks."""

    def __init__(self, config: V10Step1Config):
        super().__init__()
        config.validate()
        self.config = config
        self.emb_m = nn.Embedding(config.m_vocab_size, config.width)
        self.emb_s = nn.Embedding(config.s_vocab_with_bos, config.width)
        blocks: list[nn.Module] = []
        for _ in range(config.repeats):
            blocks += [
                MambaBlock(config.width, config.dropout, config.mamba_d_state)
                for _ in range(MAMBA_PER_ATTENTION)
            ]
            blocks.append(
                SelfAttentionBlock(
                    config.width,
                    config.attention_heads,
                    config.ffn_dim,
                    config.attention_window,
                    config.dropout,
                    config.attention_dropout,
                )
            )
        self.blocks = nn.ModuleList(blocks)
        self.final_norm = nn.RMSNorm(config.width)
        self.head_m = nn.Linear(config.width, config.m_vocab_size)
        self.head_s = nn.Linear(config.width, config.s_vocab_with_bos)
        if self.trainable_parameters() > config.max_trainable_parameters:
            raise ValueError("Trainable parameter budget exceeded")

    def trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def block_kinds(self) -> tuple[str, ...]:
        return tuple(
            "attention" if isinstance(block, SelfAttentionBlock) else "mamba2"
            for block in self.blocks
        )

    def _check_tokens(self, m: torch.Tensor, s: torch.Tensor) -> None:
        if m.ndim != 2 or m.shape != s.shape:
            raise ValueError("Both streams need one [batch, frames] token per frame")
        if m.dtype not in (torch.int32, torch.int64):
            raise TypeError("Monologue tokens must be integer indices")
        if s.dtype not in (torch.int32, torch.int64):
            raise TypeError("CB0 tokens must be integer indices")
        if int(m.max()) >= self.config.m_vocab_size or int(m.min()) < 0:
            raise ValueError("Monologue token outside the vocabulary")
        if int(s.max()) >= self.config.s_vocab_with_bos or int(s.min()) < 0:
            raise ValueError("CB0 token outside the vocabulary")

    def forward(self, m: torch.Tensor, s: torch.Tensor) -> V10Output:
        """Teacher forcing: ``m`` and ``s`` are the true tokens of each frame."""
        self._check_tokens(m, s)
        batch, frames = s.shape
        m_input = torch.cat(
            (m.new_full((batch, 1), self.config.m_bos_id), m[:, :-1]), dim=1
        )
        s_input = torch.cat(
            (s.new_full((batch, 1), self.config.s_bos_id), s[:, :-1]), dim=1
        )
        x = self.emb_m(m_input) + self.emb_s(s_input)
        positions = torch.arange(frames, device=x.device)
        for block in self.blocks:
            if isinstance(block, SelfAttentionBlock):
                x = block(x, positions)
            else:
                x = block(x)
        h = self.final_norm(x)
        return V10Output(
            m_logits=self.head_m(h),
            s_logits=self.head_s(h + self.emb_m(m)),
            hidden=h,
        )

    def loss(
        self,
        output: V10Output,
        m: torch.Tensor,
        s: torch.Tensor,
        lengths: torch.Tensor,
        *,
        use_s: bool = True,
    ) -> V10Losses:
        """CE(m) + CE(s) over valid frames, each logged, CE(m) also by token kind.

        A kind absent from the batch reports zero. ``use_s=False`` leaves CE(s) out
        of the total, for the m-only arm.
        """
        frames = s.shape[1]
        valid = (
            torch.arange(frames, device=s.device)[None] < lengths.to(s.device)[:, None]
        )
        m = m.long()
        per_frame_m = _token_ce(output.m_logits, m)
        ce_m = _masked_sum(per_frame_m, valid)
        ce_s = _masked_sum(_token_ce(output.s_logits, s.long()), valid)
        by_kind = per_frame_m.detach()
        return V10Losses(
            total=ce_m + ce_s if use_s else ce_m,
            m=ce_m,
            s=ce_s,
            m_unit=_masked_sum(by_kind, valid & (m < self.config.units)),
            m_sil=_masked_sum(by_kind, valid & (m == self.config.sil_id)),
            m_pad=_masked_sum(by_kind, valid & (m == self.config.pad_id)),
        )

    def init_state(self, batch: int) -> V10StreamState:
        mamba: list[tuple[torch.Tensor, torch.Tensor] | None] = []
        attention: list[AttentionCache | None] = []
        for block in self.blocks:
            if isinstance(block, MambaBlock):
                mamba.append(block.allocate_state(batch))
                attention.append(None)
            else:
                mamba.append(None)
                attention.append(AttentionCache())
        return V10StreamState(0, mamba, attention)

    def step(
        self,
        state: V10StreamState,
        m_prev: torch.Tensor,
        s_prev: torch.Tensor,
    ) -> tuple[torch.Tensor, V10StreamState]:
        """One streaming frame from the tokens of frame ``t - 1`` (BOS at 0)."""
        x = self.emb_m(m_prev.long()) + self.emb_s(s_prev.long())
        x = x[:, None, :]
        for index, block in enumerate(self.blocks):
            if isinstance(block, MambaBlock):
                cache = state.mamba[index]
                if cache is None:
                    raise ValueError("Mamba block has no streaming state")
                x = block.step(x, cache)
            else:
                attention = state.attention[index]
                if attention is None:
                    raise ValueError("Attention block has no streaming state")
                x, attention = cast(SelfAttentionBlock, block).step(
                    x, state.frame_index, attention
                )
                state.attention[index] = attention
        state.frame_index += 1
        return self.final_norm(x)[:, 0], state

    def reindex_state(
        self, state: V10StreamState, indices: torch.Tensor
    ) -> V10StreamState:
        """Select particles, so a resampled filter continues their parents."""
        if indices.ndim != 1:
            raise ValueError("Particle indices must be one-dimensional")
        batch = _state_batch(state)
        mamba = [
            None
            if entry is None
            else tuple(_select(value, indices, batch) for value in entry)
            for entry in state.mamba
        ]
        attention = [
            None
            if cache is None
            else AttentionCache(
                _select(cache.keys, indices, batch),
                _select(cache.values, indices, batch),
                _select(cache.positions, indices, batch),
            )
            for cache in state.attention
        ]
        return V10StreamState(state.frame_index, mamba, attention)
