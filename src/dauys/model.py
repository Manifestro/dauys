"""Causal two-stream Hybrid v0.1 dialogue model."""

import math
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import cast

import torch
from torch import nn
from torch.nn import functional as F

from .config import ModelConfig
from .decoding import (
    ActivityState,
    DecodingConfig,
    activity_decision,
    cb0_decision,
    sample_logits,
)


def _rope(x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    frequency = torch.arange(half, device=x.device, dtype=torch.float32)
    frequency = 1.0 / (10_000.0 ** (frequency / half))
    phase = positions.to(torch.float32)[:, None] * frequency[None, :]
    cosine = phase.cos().to(x.dtype)[None, None]
    sine = phase.sin().to(x.dtype)[None, None]
    even, odd = x[..., 0::2], x[..., 1::2]
    return torch.stack(
        (even * cosine - odd * sine, even * sine + odd * cosine), dim=-1
    ).flatten(-2)


def _attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    allowed: torch.Tensor,
    dropout: float = 0.0,
    training: bool = False,
) -> torch.Tensor:
    scores = torch.matmul(q.float(), k.float().transpose(-2, -1))
    scores *= q.shape[-1] ** -0.5
    scores = scores.masked_fill(~allowed[None, None], torch.finfo(scores.dtype).min)
    weights = scores.softmax(dim=-1)
    weights = weights * allowed[None, None]
    weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-9)
    weights = F.dropout(weights, p=dropout, training=training)
    return torch.matmul(weights.to(v.dtype), v)


@dataclass
class AttentionCache:
    keys: torch.Tensor | None = None
    values: torch.Tensor | None = None
    positions: torch.Tensor | None = None


class FrameAttention(nn.Module):
    def __init__(
        self,
        query_dim: int,
        context_dim: int,
        heads: int,
        window: int,
        attention_dropout: float = 0.0,
    ):
        super().__init__()
        if query_dim % heads:
            raise ValueError("Attention width must be divisible by heads")
        self.heads = heads
        self.head_dim = query_dim // heads
        self.window = window
        self.attention_dropout = attention_dropout
        self.query = nn.Linear(query_dim, query_dim)
        self.key = nn.Linear(context_dim, query_dim)
        self.value = nn.Linear(context_dim, query_dim)
        self.output = nn.Linear(query_dim, query_dim)

    def _heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, width = x.shape
        return x.view(batch, length, self.heads, width // self.heads).transpose(1, 2)

    def forward(
        self,
        query: torch.Tensor,
        context: torch.Tensor,
        query_positions: torch.Tensor,
        context_positions: torch.Tensor,
        input_delay: int = 0,
    ) -> torch.Tensor:
        q = _rope(self._heads(self.query(query)), query_positions)
        k = _rope(self._heads(self.key(context)), context_positions)
        v = self._heads(self.value(context))
        distance = query_positions[:, None] - input_delay - context_positions[None, :]
        allowed = (distance >= 0) & (distance < self.window)
        result = _attention(q, k, v, allowed, self.attention_dropout, self.training)
        result = result.transpose(1, 2).contiguous().view(query.shape)
        output = self.output(result)
        return output * allowed.any(dim=-1)[None, :, None]

    def step_self(
        self, x: torch.Tensor, position: int, cache: AttentionCache
    ) -> tuple[torch.Tensor, AttentionCache]:
        index = torch.tensor([position], device=x.device)
        q = _rope(self._heads(self.query(x)), index)
        k = _rope(self._heads(self.key(x)), index)
        v = self._heads(self.value(x))
        if cache.keys is not None:
            k = torch.cat((cache.keys, k), dim=2)
            assert cache.values is not None
            v = torch.cat((cache.values, v), dim=2)
        k = k[:, :, -self.window :]
        v = v[:, :, -self.window :]
        allowed = torch.ones((1, k.shape[2]), dtype=torch.bool, device=x.device)
        result = _attention(q, k, v, allowed, self.attention_dropout, self.training)
        result = result.transpose(1, 2).contiguous().view(x.shape)
        return self.output(result), AttentionCache(k, v, None)


class SwiGLU(nn.Module):
    def __init__(self, width: int, hidden: int):
        super().__init__()
        self.up = nn.Linear(width, 2 * hidden)
        self.down = nn.Linear(hidden, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value, gate = self.up(x).chunk(2, dim=-1)
        return self.down(value * F.silu(gate))


class MambaBlock(nn.Module):
    def __init__(
        self,
        width: int,
        dropout: float = 0.0,
        d_state: int = 64,
        norm_eps: float | None = None,
    ):
        super().__init__()
        from mamba_ssm import Mamba2

        self.norm = nn.RMSNorm(width, eps=norm_eps)
        self.residual_dropout = nn.Dropout(dropout)
        self.mixer = Mamba2(
            d_model=width,
            d_state=d_state,
            d_conv=4,
            expand=2,
            headdim=64,
            ngroups=1,
            layer_idx=0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.residual_dropout(self.mixer(self.norm(x)))

    def forward_with_state(
        self,
        x: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Mamba2's continuous path with an explicit convolution/SSM carry."""
        from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined

        mixer = self.mixer
        residual = x
        normed = self.norm(x)
        batch, frames, _ = normed.shape
        zxbcdt = mixer.in_proj(normed)
        d_mlp = (
            zxbcdt.shape[-1]
            - 2 * mixer.d_ssm
            - 2 * mixer.ngroups * mixer.d_state
            - mixer.nheads
        ) // 2
        z0, x0, z, xBC, dt = torch.split(
            zxbcdt,
            [
                d_mlp,
                d_mlp,
                mixer.d_ssm,
                mixer.d_ssm + 2 * mixer.ngroups * mixer.d_state,
                mixer.nheads,
            ],
            dim=-1,
        )
        conv_dim = mixer.d_ssm + 2 * mixer.ngroups * mixer.d_state
        if state is None:
            conv_tail = normed.new_zeros(batch, conv_dim, mixer.d_conv - 1)
            ssm_state = normed.new_zeros(
                batch, mixer.nheads, mixer.headdim, mixer.d_state
            )
        else:
            conv_tail, ssm_state = state
            if conv_tail.shape != (batch, conv_dim, mixer.d_conv - 1):
                raise ValueError("Mamba convolution state has the wrong shape")
            if ssm_state.shape != (
                batch,
                mixer.nheads,
                mixer.headdim,
                mixer.d_state,
            ):
                raise ValueError("Mamba SSM state has the wrong shape")
            conv_tail = conv_tail.to(device=normed.device, dtype=normed.dtype)
            ssm_state = ssm_state.to(device=normed.device, dtype=normed.dtype)
        conv_input = torch.cat((conv_tail, xBC.transpose(1, 2)), dim=-1)
        convolved = F.conv1d(
            conv_input,
            mixer.conv1d.weight,
            mixer.conv1d.bias,
            padding=mixer.d_conv - 1,
            groups=conv_dim,
        )
        convolved = convolved[:, :, mixer.d_conv - 1 : mixer.d_conv - 1 + frames]
        xBC = mixer.act(convolved.transpose(1, 2))
        new_conv_tail = conv_input[:, :, -(mixer.d_conv - 1) :].detach()
        x_ssm, B, C = torch.split(
            xBC,
            [mixer.d_ssm, mixer.ngroups * mixer.d_state, mixer.ngroups * mixer.d_state],
            dim=-1,
        )
        dt_limit_kwargs = (
            {}
            if mixer.dt_limit == (0.0, float("inf"))
            else {"dt_limit": mixer.dt_limit}
        )
        y, final_state = mamba_chunk_scan_combined(
            x_ssm.reshape(batch, frames, mixer.nheads, mixer.headdim),
            dt,
            -torch.exp(mixer.A_log.float()),
            B.reshape(batch, frames, mixer.ngroups, mixer.d_state),
            C.reshape(batch, frames, mixer.ngroups, mixer.d_state),
            chunk_size=mixer.chunk_size,
            D=(
                mixer.D.reshape(mixer.nheads, mixer.headdim)
                if mixer.D_has_hdim
                else mixer.D
            ),
            z=(
                z.reshape(batch, frames, mixer.nheads, mixer.headdim)
                if not mixer.rmsnorm
                else None
            ),
            dt_bias=mixer.dt_bias,
            dt_softplus=True,
            seq_idx=None,
            initial_states=ssm_state,
            return_final_states=True,
            **dt_limit_kwargs,
        )
        y = y.reshape(batch, frames, mixer.d_ssm)
        if mixer.rmsnorm:
            y = mixer.norm(y, z)
        if d_mlp > 0:
            y = torch.cat([F.silu(z0) * x0, y], dim=-1)
        y = mixer.out_proj(y)
        return residual + self.residual_dropout(y), (
            new_conv_tail,
            final_state.detach(),
        )

    def step(
        self, x: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        mixed, _, _ = self.mixer.step(self.norm(x), *state)
        return x + self.residual_dropout(mixed)

    def prefill(
        self, x: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        """Run a teacher-forced prefix in parallel and populate inference state."""
        from mamba_ssm.modules import mamba2

        params = SimpleNamespace(
            seqlen_offset=0, key_value_memory_dict={self.mixer.layer_idx: state}
        )
        # The fused convolution requires a channel stride divisible by eight;
        # this model's projection has a different stride. Mamba's PyTorch
        # convolution path still uses the parallel SSM scan and fills the cache.
        fused_conv = mamba2.causal_conv1d_fn
        try:
            mamba2.causal_conv1d_fn = None
            mixed = self.mixer(self.norm(x), inference_params=params)
        finally:
            mamba2.causal_conv1d_fn = fused_conv
        return x + self.residual_dropout(mixed)

    def allocate_state(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen=1)


class DialogueBlock(nn.Module):
    def __init__(
        self,
        width: int,
        listening_dim: int,
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
        self.cross_norm = nn.RMSNorm(width)
        self.cross_attention = FrameAttention(
            width, listening_dim, heads, window, attention_dropout
        )
        self.ffn_norm = nn.RMSNorm(width)
        self.ffn = SwiGLU(width, ffn_dim)
        self.residual_dropout = nn.Dropout(dropout)

    def zero_init_outputs(self) -> None:
        """Make the block an identity map at initialization."""
        for layer in (
            self.self_attention.output,
            self.cross_attention.output,
            self.ffn.down,
        ):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(
        self,
        x: torch.Tensor,
        listening: torch.Tensor,
        positions: torch.Tensor,
        delay: int,
        cross_enabled: bool = True,
    ) -> torch.Tensor:
        normalized = self.self_norm(x)
        x = x + self.residual_dropout(
            self.self_attention(normalized, normalized, positions, positions)
        )
        if cross_enabled:
            x = x + self.residual_dropout(
                self.cross_attention(
                    self.cross_norm(x), listening, positions, positions, delay
                )
            )
        return x + self.residual_dropout(self.ffn(self.ffn_norm(x)))

    def step(
        self,
        x: torch.Tensor,
        listening: torch.Tensor,
        position: int,
        listening_positions: torch.Tensor,
        delay: int,
        cache: AttentionCache,
    ) -> tuple[torch.Tensor, AttentionCache]:
        update, cache = self.self_attention.step_self(
            self.self_norm(x), position, cache
        )
        x = x + self.residual_dropout(update)
        query_position = torch.tensor([position], device=x.device)
        x = x + self.residual_dropout(
            self.cross_attention(
                self.cross_norm(x),
                listening,
                query_position,
                listening_positions,
                delay,
            )
        )
        return x + self.residual_dropout(self.ffn(self.ffn_norm(x))), cache


class DepthBlock(nn.Module):
    def __init__(self, width: int, heads: int):
        super().__init__()
        self.attention_norm = nn.RMSNorm(width)
        self.attention = nn.MultiheadAttention(width, heads, batch_first=True)
        self.ffn_norm = nn.RMSNorm(width)
        self.ffn = SwiGLU(width, 4 * width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        length = x.shape[1]
        causal = torch.ones(length, length, device=x.device, dtype=torch.bool).triu(1)
        normed = self.attention_norm(x)
        update, _ = self.attention(
            normed, normed, normed, attn_mask=causal, need_weights=False
        )
        x = x + update
        return x + self.ffn(self.ffn_norm(x))


class DepthDecoder(nn.Module):
    def __init__(self, config: ModelConfig, embeddings: nn.ModuleList):
        super().__init__()
        self.embeddings = tuple(embeddings)
        self.input_projection = nn.Linear(config.embedding_dim, config.depth_dim)
        self.context_projection = nn.Linear(config.dialogue_dim, config.depth_dim)
        self.depth_positions = nn.Embedding(7, config.depth_dim)
        self.blocks = nn.ModuleList(
            DepthBlock(config.depth_dim, config.depth_heads)
            for _ in range(config.depth_layers)
        )
        self.norm = nn.RMSNorm(config.depth_dim)
        self.heads = nn.ModuleList(
            nn.Linear(config.depth_dim, config.vocab_size) for _ in range(7)
        )

    def forward(self, prefix: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """Prefix codebooks 0..Q-1 predict target codebooks 1..Q."""
        batch, length, depth = prefix.shape
        expected = (batch, length, self.context_projection.in_features)
        if not 1 <= depth <= 7 or context.shape != expected:
            raise ValueError("Invalid depth decoder inputs")
        embedded = torch.stack(
            [self.embeddings[q](prefix[:, :, q]) for q in range(depth)], dim=2
        )
        positions = torch.arange(depth, device=prefix.device)
        x = self.input_projection(embedded)
        x = x + self.depth_positions(positions)[None, None]
        x = x + self.context_projection(context)[:, :, None, :]
        x = x.reshape(batch * length, depth, -1)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x).reshape(batch, length, depth, -1)
        return torch.stack([self.heads[q](x[:, :, q]) for q in range(depth)], dim=2)


@dataclass
class ModelOutput:
    cb0_logits: torch.Tensor
    activity_logits: torch.Tensor
    depth_logits: torch.Tensor | None
    dialogue_states: torch.Tensor
    user_cb0_logits: torch.Tensor | None = None
    user_depth_logits: torch.Tensor | None = None
    listening_cb0_logits: torch.Tensor | None = None
    listening_activity_logits: torch.Tensor | None = None
    activity_ahead_logits: torch.Tensor | None = None


@dataclass
class StreamingState:
    frame_index: int
    listening_mamba: list[tuple[torch.Tensor, torch.Tensor]]
    dialogue_mamba: list[tuple[torch.Tensor, torch.Tensor] | None]
    dialogue_attention: list[AttentionCache | None]
    listening_history: list[torch.Tensor] = field(default_factory=list)
    previous_output: torch.Tensor | None = None
    activity_state: ActivityState | None = None


def previous_frames(codes: torch.Tensor, initial_codes: torch.Tensor) -> torch.Tensor:
    """Shift a [batch, 8, frames] stream one frame right behind its initial frame."""
    return torch.cat((initial_codes.unsqueeze(-1), codes[:, :, :-1]), dim=-1)


class DauysHybrid(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        if config.ear_embeddings == "separate" and config.self_hearing != "none":
            raise ValueError("Separate ear embeddings cannot use self-hearing")
        self.config = config
        self.embeddings = nn.ModuleList(
            nn.Embedding(config.vocab_size, config.embedding_dim)
            for _ in range(config.codebooks)
        )
        self.ear_embeddings = (
            nn.ModuleList(
                nn.Embedding(config.vocab_size, config.embedding_dim)
                for _ in range(config.codebooks)
            )
            if config.ear_embeddings == "separate"
            else None
        )
        self.roles = nn.Embedding(2, config.embedding_dim)
        self.listening_projection = nn.Linear(
            config.embedding_dim, config.listening_dim
        )
        # The ear can also hear the model's own previous frame. "labeled" gives
        # it a separate input path; "mixed" adds it to the user's sound through
        # the same path, as one microphone would. Both start silent.
        self.self_hearing_projection = (
            nn.Linear(config.embedding_dim, config.listening_dim)
            if config.self_hearing == "labeled"
            else None
        )
        if self.self_hearing_projection is not None:
            nn.init.zeros_(self.self_hearing_projection.weight)
            nn.init.zeros_(self.self_hearing_projection.bias)
        self.self_hearing_gate = (
            nn.Parameter(torch.zeros(())) if config.self_hearing == "mixed" else None
        )
        self.frame_dropout = nn.Dropout(config.dropout)
        self.dialogue_projection = nn.Linear(config.embedding_dim, config.dialogue_dim)
        self.listening = nn.ModuleList(
            MambaBlock(
                config.listening_dim,
                config.dropout,
                config.listening_d_state,
                config.mamba_norm_eps,
            )
            for _ in range(config.listening_layers)
        )
        self.dialogue = nn.ModuleList(
            MambaBlock(
                config.dialogue_dim,
                config.dropout,
                config.dialogue_d_state,
                config.mamba_norm_eps,
            )
            if kind == "mamba2"
            else DialogueBlock(
                config.dialogue_dim,
                config.listening_dim,
                config.attention_heads,
                config.ffn_dim,
                config.attention_window,
                config.dropout,
                config.attention_dropout,
            )
            for kind in config.dialogue_blocks
        )
        self.final_norm = nn.RMSNorm(config.dialogue_dim)
        self.user_prediction_projection = nn.Linear(
            config.listening_dim, config.dialogue_dim
        )
        self.cb0_head = nn.Linear(config.dialogue_dim, config.vocab_size)
        self.activity_head = nn.Linear(config.dialogue_dim, 2)
        self.listening_norm = (
            nn.RMSNorm(config.listening_dim) if config.listening_prediction else None
        )
        self.listening_cb0_head = (
            nn.Linear(config.listening_dim, config.vocab_size)
            if config.listening_prediction
            else None
        )
        self.listening_activity_head = (
            nn.Linear(config.listening_dim, 2) if config.listening_prediction else None
        )
        self.activity_ahead_head = (
            nn.Linear(config.dialogue_dim, 2 * config.activity_ahead_horizon)
            if config.activity_ahead_horizon
            else None
        )
        self.depth = DepthDecoder(config, self.embeddings)
        if config.zero_init_fusion_outputs:
            for block in self.dialogue:
                if isinstance(block, DialogueBlock):
                    block.zero_init_outputs()
        if config.freeze_ear:
            self.freeze_ear_parameters()
        if self.trainable_parameters() > config.max_trainable_parameters:
            raise ValueError("Trainable parameter budget exceeded")

    def trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def freeze_ear_parameters(self) -> None:
        if self.ear_embeddings is not None:
            self.ear_embeddings.requires_grad_(False)
        self.listening_projection.requires_grad_(False)
        self.listening.requires_grad_(False)
        for block in self.listening:
            block.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.config.freeze_ear:
            for block in self.listening:
                block.eval()
        return self

    def fusion_block_indices(self) -> tuple[int, ...]:
        return tuple(
            index
            for index, block in enumerate(self.dialogue)
            if isinstance(block, DialogueBlock)
        )

    def set_regularization(self, enabled: bool) -> None:
        """Apply the Stage 2 ablation toggle without changing model geometry."""
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.p = self.config.dropout if enabled else 0.0
            elif isinstance(module, FrameAttention):
                module.attention_dropout = (
                    self.config.attention_dropout if enabled else 0.0
                )

    def _embed(
        self,
        codes: torch.Tensor,
        role: int,
        acoustic_dropout: float = 0.0,
        acoustic_dropout_items: torch.Tensor | None = None,
        acoustic_keep_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if codes.ndim != 3 or codes.shape[1] != self.config.codebooks:
            raise ValueError("Codes must have shape [batch, 8, frames]")
        if codes.dtype not in (torch.int32, torch.int64):
            raise TypeError("Codes must be integer indices")
        if acoustic_dropout and self.training and role == 0:
            if not 0 <= acoustic_dropout < 1:
                raise ValueError("Invalid acoustic dropout probability")
            acoustic = torch.stack(
                [self.embeddings[q](codes[:, q]) for q in range(1, 8)], dim=2
            )
            if acoustic_keep_mask is not None:
                if acoustic_keep_mask.shape != acoustic.shape[:3]:
                    raise ValueError(
                        "Acoustic keep mask must have shape [batch, frames, 7]"
                    )
                keep = acoustic_keep_mask
            else:
                keep = (
                    torch.rand(acoustic.shape[:3], device=codes.device)
                    >= acoustic_dropout
                )
            if acoustic_dropout_items is not None:
                if acoustic_dropout_items.shape != (codes.shape[0],):
                    raise ValueError(
                        "Acoustic dropout item mask must have shape [batch]"
                    )
                keep = keep | ~acoustic_dropout_items[:, None, None]
            count = keep.sum(dim=2).clamp_min(1)
            scale = torch.sqrt(7 / count.to(acoustic.dtype))
            x = (
                self.embeddings[0](codes[:, 0])
                + (acoustic * keep[..., None]).sum(dim=2) * scale[..., None]
            )
        else:
            x = sum(self.embeddings[q](codes[:, q]) for q in range(8))
        return self.frame_dropout(x / math.sqrt(8) + self.roles.weight[role])

    def _ear_embed(
        self,
        codes: torch.Tensor,
        acoustic_dropout: float = 0.0,
        acoustic_dropout_items: torch.Tensor | None = None,
        acoustic_keep_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.config.ear_embeddings == "separate":
            if codes.ndim != 3 or codes.shape[1] != self.config.codebooks:
                raise ValueError("Codes must have shape [batch, 8, frames]")
            if codes.dtype not in (torch.int32, torch.int64):
                raise TypeError("Codes must be integer indices")
            assert self.ear_embeddings is not None
            x = sum(
                self.ear_embeddings[q](codes[:, q])
                for q in range(self.config.codebooks)
            )
            return x / math.sqrt(self.config.codebooks)
        return self._embed(
            codes,
            0,
            acoustic_dropout,
            acoustic_dropout_items,
            acoustic_keep_mask,
        )

    def ear_input(
        self,
        user_embedding: torch.Tensor,
        own_previous_codes: torch.Tensor | None,
    ) -> torch.Tensor:
        """Project what the ear hears at frame t: A_t, plus B_(t-1) if enabled.

        The model's own sound always arrives one frame late, so the ear never
        hears the frame the dialogue tower is predicting.
        """
        if self.config.self_hearing == "none":
            return self.listening_projection(user_embedding)
        if own_previous_codes is None:
            raise ValueError("Self-hearing requires the model's previous frames")
        if self.self_hearing_gate is not None:
            own = self._embed(own_previous_codes, 0)
            return self.listening_projection(
                user_embedding + self.self_hearing_gate * own
            )
        assert self.self_hearing_projection is not None
        return self.listening_projection(user_embedding) + self.self_hearing_projection(
            self._embed(own_previous_codes, 1)
        )

    def encode_listening(
        self,
        user_codes: torch.Tensor,
        acoustic_dropout: float = 0.0,
        acoustic_dropout_items: torch.Tensor | None = None,
        acoustic_keep_mask: torch.Tensor | None = None,
        own_previous_codes: torch.Tensor | None = None,
        mamba_states: dict | None = None,
    ) -> torch.Tensor:
        listening = self.ear_input(
            self._ear_embed(
                user_codes,
                acoustic_dropout,
                acoustic_dropout_items,
                acoustic_keep_mask,
            ),
            own_previous_codes,
        )
        for index, block in enumerate(self.listening):
            if mamba_states is None:
                listening = block(listening)
            else:
                listening, mamba_states[("ear", index)] = block.forward_with_state(
                    listening, mamba_states.get(("ear", index))
                )
        return listening

    def run_dialogue(
        self,
        listening: torch.Tensor,
        output_codes: torch.Tensor,
        initial_output_codes: torch.Tensor,
        history_codes: torch.Tensor | None = None,
        fusion_off_blocks: frozenset[int] = frozenset(),
        mamba_states: dict | None = None,
    ) -> torch.Tensor:
        if output_codes.ndim != 3 or output_codes.shape[1] != 8:
            raise ValueError("Output codes must have shape [batch, 8, frames]")
        batch, _, frames = output_codes.shape
        if listening.shape != (batch, frames, self.config.listening_dim):
            raise ValueError("Listening states must match output frames")
        if initial_output_codes.shape != (batch, 8):
            raise ValueError("Initial output codes must have shape [batch, 8]")
        if history_codes is not None and history_codes.shape != output_codes.shape:
            raise ValueError("History codes must match target code shape")
        if not fusion_off_blocks.issubset(self.fusion_block_indices()):
            raise ValueError("Only dialogue fusion blocks can be ablated")
        history = output_codes if history_codes is None else history_codes
        previous = torch.cat(
            (initial_output_codes.unsqueeze(-1), history[:, :, :-1]), dim=-1
        )
        x = self.dialogue_projection(self._embed(previous, 1))
        positions = torch.arange(frames, device=x.device)
        for index, block in enumerate(self.dialogue):
            if isinstance(block, MambaBlock):
                if mamba_states is None:
                    x = block(x)
                else:
                    x, mamba_states[("dialogue", index)] = block.forward_with_state(
                        x, mamba_states.get(("dialogue", index))
                    )
            else:
                x = block(
                    x,
                    listening,
                    positions,
                    self.config.input_delay_frames,
                    cross_enabled=index not in fusion_off_blocks,
                )
        return self.final_norm(x)

    def forward(
        self,
        user_codes: torch.Tensor,
        output_codes: torch.Tensor,
        initial_output_codes: torch.Tensor,
        initial_user_codes: torch.Tensor | None = None,
        history_codes: torch.Tensor | None = None,
        detach_listening: torch.Tensor | None = None,
        acoustic_dropout: float = 0.0,
        acoustic_dropout_items: torch.Tensor | None = None,
        acoustic_keep_mask: torch.Tensor | None = None,
        fusion_off_blocks: frozenset[int] = frozenset(),
        mamba_states: dict | None = None,
    ) -> ModelOutput:
        """Teacher-forced prediction of B_t using A_<=t and B_<t."""
        if user_codes.shape != output_codes.shape:
            raise ValueError("Two channels must have identical code shapes")
        batch, _, frames = user_codes.shape
        if frames < 1 or initial_output_codes.shape != (batch, 8):
            raise ValueError("An initial valid silence frame is required")
        if mamba_states is not None:
            if self.config.self_hearing != "none":
                raise ValueError("Stateful training does not support self-hearing")
            if self.config.ear_embeddings != "shared":
                raise ValueError("Stateful training does not support separate ears")
            if initial_user_codes is not None:
                raise ValueError("Stateful forward does not support initial user codes")
        own_previous = previous_frames(
            output_codes if history_codes is None else history_codes,
            initial_output_codes,
        )
        listening = self.encode_listening(
            user_codes,
            acoustic_dropout,
            acoustic_dropout_items,
            acoustic_keep_mask,
            own_previous,
            mamba_states,
        )
        dialogue_listening = listening
        if detach_listening is not None:
            if detach_listening.shape != (batch,):
                raise ValueError("Listening detach mask must have shape [batch]")
            dialogue_listening = torch.where(
                detach_listening[:, None, None],
                listening.detach(),
                listening,
            )
        x = self.run_dialogue(
            dialogue_listening,
            output_codes,
            initial_output_codes,
            history_codes,
            fusion_off_blocks,
            mamba_states,
        )
        user_cb0_logits = None
        user_depth_logits = None
        if initial_user_codes is not None:
            if initial_user_codes.shape != (batch, 8):
                raise ValueError("Initial user codes must have shape [batch, 8]")
            previous_user = torch.cat(
                (initial_user_codes.unsqueeze(-1), user_codes[:, :, :-1]), dim=-1
            )
            user_state = self.listening_projection(self._ear_embed(previous_user))
            for block in self.listening:
                user_state = block(user_state)
            user_context = self.final_norm(self.user_prediction_projection(user_state))
            user_cb0_logits = self.cb0_head(user_context)
            user_depth_logits = self.depth(
                user_codes[:, :7].transpose(1, 2), user_context
            )
        listening_cb0_logits = None
        listening_activity_logits = None
        if self.listening_norm is not None:
            assert self.listening_cb0_head is not None
            assert self.listening_activity_head is not None
            normalized_listening = self.listening_norm(listening)
            listening_cb0_logits = self.listening_cb0_head(normalized_listening)
            listening_activity_logits = self.listening_activity_head(
                normalized_listening
            )
        activity_ahead_logits = None
        if self.activity_ahead_head is not None:
            activity_ahead_logits = self.activity_ahead_head(x).reshape(
                batch, frames, self.config.activity_ahead_horizon, 2
            )
        return ModelOutput(
            cb0_logits=self.cb0_head(x),
            activity_logits=self.activity_head(x),
            depth_logits=self.depth(output_codes[:, :7].transpose(1, 2), x),
            dialogue_states=x,
            user_cb0_logits=user_cb0_logits,
            user_depth_logits=user_depth_logits,
            listening_cb0_logits=listening_cb0_logits,
            listening_activity_logits=listening_activity_logits,
            activity_ahead_logits=activity_ahead_logits,
        )

    def new_stream(self, initial_output_codes: torch.Tensor) -> StreamingState:
        if initial_output_codes.ndim != 2 or initial_output_codes.shape[1] != 8:
            raise ValueError("Initial output must contain eight valid silence codes")
        if initial_output_codes.dtype not in (torch.int32, torch.int64):
            raise TypeError("Initial silence codes must be integer indices")
        if torch.any(initial_output_codes < 0) or torch.any(
            initial_output_codes >= 2048
        ):
            raise ValueError("Initial silence code outside vocabulary")
        batch = initial_output_codes.shape[0]
        return StreamingState(
            frame_index=0,
            listening_mamba=[
                cast(MambaBlock, block).allocate_state(batch)
                for block in self.listening
            ],
            dialogue_mamba=[
                block.allocate_state(batch) if isinstance(block, MambaBlock) else None
                for block in self.dialogue
            ],
            dialogue_attention=[
                AttentionCache() if isinstance(block, DialogueBlock) else None
                for block in self.dialogue
            ],
            previous_output=initial_output_codes,
        )

    @torch.inference_mode()
    def prefill_teacher(
        self,
        user_codes: torch.Tensor,
        output_codes: torch.Tensor,
        initial_output_codes: torch.Tensor,
    ) -> StreamingState:
        """Build the same stream state as sequential teacher steps, in parallel."""
        if user_codes.shape != output_codes.shape or user_codes.ndim != 3:
            raise ValueError(
                "Synchronized prefix codes must have shape [batch, 8, frames]"
            )
        batch, codebooks, frames = user_codes.shape
        if codebooks != 8 or initial_output_codes.shape != (batch, 8):
            raise ValueError("Invalid prefix code geometry")
        state = self.new_stream(initial_output_codes)
        if frames == 0:
            return state
        previous = torch.cat(
            (initial_output_codes.unsqueeze(-1), output_codes[:, :, :-1]), dim=-1
        )
        listening = self.ear_input(self._ear_embed(user_codes), previous)
        for index, block in enumerate(self.listening):
            listening = cast(MambaBlock, block).prefill(
                listening, state.listening_mamba[index]
            )
        x = self.dialogue_projection(self._embed(previous, 1))
        positions = torch.arange(frames, device=x.device)
        for index, block in enumerate(self.dialogue):
            if isinstance(block, MambaBlock):
                cache = state.dialogue_mamba[index]
                assert cache is not None
                x = block.prefill(x, cache)
            else:
                dialogue_block = cast(DialogueBlock, block)
                normalized = dialogue_block.self_norm(x)
                attention = dialogue_block.self_attention
                keys = _rope(attention._heads(attention.key(normalized)), positions)
                values = attention._heads(attention.value(normalized))
                state.dialogue_attention[index] = AttentionCache(
                    keys[:, :, -attention.window :].contiguous(),
                    values[:, :, -attention.window :].contiguous(),
                    None,
                )
                x = x + attention(normalized, normalized, positions, positions)
                x = x + dialogue_block.cross_attention(
                    dialogue_block.cross_norm(x),
                    listening,
                    positions,
                    positions,
                    self.config.input_delay_frames,
                )
                x = x + dialogue_block.ffn(dialogue_block.ffn_norm(x))
        history_limit = self.config.attention_window + self.config.input_delay_frames
        state.listening_history = [
            listening[:, frame : frame + 1]
            for frame in range(max(0, frames - history_limit), frames)
        ]
        state.frame_index = frames
        state.previous_output = output_codes[:, :, -1]
        return state

    @torch.inference_mode()
    def stream_step_teacher(
        self,
        user_codes: torch.Tensor,
        previous_output: torch.Tensor,
        state: StreamingState,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Advance one synchronized frame while preserving both stream states."""
        batch = previous_output.shape[0]
        if user_codes.shape != (batch, 8) or previous_output.shape != (batch, 8):
            raise ValueError("Each stream step requires [batch, 8] codes")
        listening = self.ear_input(
            self._ear_embed(user_codes[:, :, None]), previous_output[:, :, None]
        )
        for block, cache in zip(self.listening, state.listening_mamba):
            listening = cast(MambaBlock, block).step(listening, cache)
        state.listening_history.append(listening)
        history_limit = self.config.attention_window + self.config.input_delay_frames
        if len(state.listening_history) > history_limit:
            state.listening_history.pop(0)
        history = torch.cat(state.listening_history, dim=1)
        start = state.frame_index - history.shape[1] + 1
        listening_positions = torch.arange(
            start, state.frame_index + 1, device=history.device
        )
        x = self.dialogue_projection(self._embed(previous_output[:, :, None], 1))
        for index, block in enumerate(self.dialogue):
            if isinstance(block, MambaBlock):
                mamba_cache = state.dialogue_mamba[index]
                assert mamba_cache is not None
                x = cast(MambaBlock, block).step(x, mamba_cache)
            else:
                attention_cache = state.dialogue_attention[index]
                assert attention_cache is not None
                x, updated = cast(DialogueBlock, block).step(
                    x,
                    history,
                    state.frame_index,
                    listening_positions,
                    self.config.input_delay_frames,
                    attention_cache,
                )
                state.dialogue_attention[index] = updated
        state.frame_index += 1
        x = self.final_norm(x)
        return self.cb0_head(x[:, 0]), self.activity_head(x[:, 0]), x

    @torch.inference_mode()
    def generate_step(
        self,
        user_codes: torch.Tensor,
        state: StreamingState,
        decoding: DecodingConfig | None = None,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate one frame; return logits for legacy calls, else a decision."""
        if state.previous_output is None:
            raise ValueError("Streaming state has no previous output frame")
        cb0, activity, context = self.stream_step_teacher(
            user_codes, state.previous_output, state
        )
        if decoding is None:
            generated = [cb0.argmax(dim=-1)]
        else:
            decoding.validate()
            if state.activity_state is None:
                state.activity_state = ActivityState.silent(cb0.shape[0], cb0.device)
            speaking = activity_decision(
                activity, decoding, state.activity_state, generator
            )
            generated = [cb0_decision(cb0, speaking, decoding, generator)]
        for target in range(1, 8):
            prefix = torch.stack(generated, dim=-1)[:, None]
            logits = self.depth(prefix, context)[:, 0, -1]
            if decoding is not None and decoding.depth_mode == "sample":
                generated.append(
                    sample_logits(
                        logits,
                        decoding.depth_temperature,
                        decoding.depth_top_k,
                        generator,
                    )
                )
            else:
                generated.append(logits.argmax(dim=-1))
        generated_codes = torch.stack(generated, dim=1)
        state.previous_output = generated_codes
        return generated_codes, speaking if decoding is not None else activity
