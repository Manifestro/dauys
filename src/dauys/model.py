"""Causal two-stream Hybrid v0.1 dialogue model."""

import math
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import cast

import torch
from torch import nn
from torch.nn import functional as F

from .config import ModelConfig


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
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, allowed: torch.Tensor
) -> torch.Tensor:
    scores = torch.matmul(q.float(), k.float().transpose(-2, -1))
    scores *= q.shape[-1] ** -0.5
    scores = scores.masked_fill(~allowed[None, None], torch.finfo(scores.dtype).min)
    weights = scores.softmax(dim=-1)
    weights = weights * allowed[None, None]
    weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-9)
    return torch.matmul(weights.to(v.dtype), v)


@dataclass
class AttentionCache:
    keys: torch.Tensor | None = None
    values: torch.Tensor | None = None
    positions: torch.Tensor | None = None


class FrameAttention(nn.Module):
    def __init__(self, query_dim: int, context_dim: int, heads: int, window: int):
        super().__init__()
        if query_dim % heads:
            raise ValueError("Attention width must be divisible by heads")
        self.heads = heads
        self.head_dim = query_dim // heads
        self.window = window
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
        result = _attention(q, k, v, allowed)
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
        result = _attention(q, k, v, allowed)
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
    def __init__(self, width: int):
        super().__init__()
        from mamba_ssm import Mamba2

        self.norm = nn.RMSNorm(width)
        self.mixer = Mamba2(
            d_model=width,
            d_state=64,
            d_conv=4,
            expand=2,
            headdim=64,
            ngroups=1,
            layer_idx=0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mixer(self.norm(x))

    def step(
        self, x: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        mixed, _, _ = self.mixer.step(self.norm(x), *state)
        return x + mixed

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
        return x + mixed

    def allocate_state(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen=1)


class DialogueBlock(nn.Module):
    def __init__(self, width: int, heads: int, ffn_dim: int, window: int):
        super().__init__()
        self.self_norm = nn.RMSNorm(width)
        self.self_attention = FrameAttention(width, width, heads, window)
        self.cross_norm = nn.RMSNorm(width)
        self.cross_attention = FrameAttention(width, 384, heads, window)
        self.ffn_norm = nn.RMSNorm(width)
        self.ffn = SwiGLU(width, ffn_dim)

    def forward(
        self,
        x: torch.Tensor,
        listening: torch.Tensor,
        positions: torch.Tensor,
        delay: int,
    ) -> torch.Tensor:
        normalized = self.self_norm(x)
        x = x + self.self_attention(normalized, normalized, positions, positions)
        x = x + self.cross_attention(
            self.cross_norm(x), listening, positions, positions, delay
        )
        return x + self.ffn(self.ffn_norm(x))

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
        x = x + update
        query_position = torch.tensor([position], device=x.device)
        x = x + self.cross_attention(
            self.cross_norm(x), listening, query_position, listening_positions, delay
        )
        return x + self.ffn(self.ffn_norm(x)), cache


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


@dataclass
class StreamingState:
    frame_index: int
    listening_mamba: list[tuple[torch.Tensor, torch.Tensor]]
    dialogue_mamba: list[tuple[torch.Tensor, torch.Tensor] | None]
    dialogue_attention: list[AttentionCache | None]
    listening_history: list[torch.Tensor] = field(default_factory=list)
    previous_output: torch.Tensor | None = None


class DauysHybrid(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.embeddings = nn.ModuleList(
            nn.Embedding(config.vocab_size, config.embedding_dim)
            for _ in range(config.codebooks)
        )
        self.roles = nn.Embedding(2, config.embedding_dim)
        self.listening_projection = nn.Linear(
            config.embedding_dim, config.listening_dim
        )
        self.dialogue_projection = nn.Linear(config.embedding_dim, config.dialogue_dim)
        self.listening = nn.ModuleList(
            MambaBlock(config.listening_dim) for _ in range(config.listening_layers)
        )
        self.dialogue = nn.ModuleList(
            MambaBlock(config.dialogue_dim)
            if kind == "mamba2"
            else DialogueBlock(
                config.dialogue_dim,
                config.attention_heads,
                config.ffn_dim,
                config.attention_window,
            )
            for kind in config.dialogue_blocks
        )
        self.final_norm = nn.RMSNorm(config.dialogue_dim)
        self.user_prediction_projection = nn.Linear(
            config.listening_dim, config.dialogue_dim
        )
        self.cb0_head = nn.Linear(config.dialogue_dim, config.vocab_size)
        self.activity_head = nn.Linear(config.dialogue_dim, 2)
        self.depth = DepthDecoder(config, self.embeddings)
        if self.trainable_parameters() > config.max_trainable_parameters:
            raise ValueError("Trainable parameter budget exceeded")

    def trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _embed(self, codes: torch.Tensor, role: int) -> torch.Tensor:
        if codes.ndim != 3 or codes.shape[1] != self.config.codebooks:
            raise ValueError("Codes must have shape [batch, 8, frames]")
        if codes.dtype not in (torch.int32, torch.int64):
            raise TypeError("Codes must be integer indices")
        x = sum(self.embeddings[q](codes[:, q]) for q in range(8))
        return x / math.sqrt(8) + self.roles.weight[role]

    def forward(
        self,
        user_codes: torch.Tensor,
        output_codes: torch.Tensor,
        initial_output_codes: torch.Tensor,
        initial_user_codes: torch.Tensor | None = None,
        history_codes: torch.Tensor | None = None,
    ) -> ModelOutput:
        """Teacher-forced prediction of B_t using A_<=t and B_<t."""
        if user_codes.shape != output_codes.shape:
            raise ValueError("Two channels must have identical code shapes")
        batch, _, frames = user_codes.shape
        if frames < 1 or initial_output_codes.shape != (batch, 8):
            raise ValueError("An initial valid silence frame is required")
        if history_codes is not None and history_codes.shape != output_codes.shape:
            raise ValueError("History codes must match target code shape")
        history = output_codes if history_codes is None else history_codes
        previous = torch.cat(
            (initial_output_codes.unsqueeze(-1), history[:, :, :-1]), dim=-1
        )
        listening = self.listening_projection(self._embed(user_codes, 0))
        for block in self.listening:
            listening = block(listening)
        x = self.dialogue_projection(self._embed(previous, 1))
        positions = torch.arange(frames, device=x.device)
        for block in self.dialogue:
            if isinstance(block, MambaBlock):
                x = block(x)
            else:
                x = block(x, listening, positions, self.config.input_delay_frames)
        x = self.final_norm(x)
        user_cb0_logits = None
        user_depth_logits = None
        if initial_user_codes is not None:
            if initial_user_codes.shape != (batch, 8):
                raise ValueError("Initial user codes must have shape [batch, 8]")
            previous_user = torch.cat(
                (initial_user_codes.unsqueeze(-1), user_codes[:, :, :-1]), dim=-1
            )
            user_state = self.listening_projection(self._embed(previous_user, 0))
            for block in self.listening:
                user_state = block(user_state)
            user_context = self.final_norm(self.user_prediction_projection(user_state))
            user_cb0_logits = self.cb0_head(user_context)
            user_depth_logits = self.depth(
                user_codes[:, :7].transpose(1, 2), user_context
            )
        return ModelOutput(
            cb0_logits=self.cb0_head(x),
            activity_logits=self.activity_head(x),
            depth_logits=self.depth(output_codes[:, :7].transpose(1, 2), x),
            dialogue_states=x,
            user_cb0_logits=user_cb0_logits,
            user_depth_logits=user_depth_logits,
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
        listening = self.listening_projection(self._embed(user_codes, 0))
        for index, block in enumerate(self.listening):
            listening = cast(MambaBlock, block).prefill(
                listening, state.listening_mamba[index]
            )
        previous = torch.cat(
            (initial_output_codes.unsqueeze(-1), output_codes[:, :, :-1]), dim=-1
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
        listening = self.listening_projection(self._embed(user_codes[:, :, None], 0))
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
        self, user_codes: torch.Tensor, state: StreamingState
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate one complete Mimi frame while continuing to listen."""
        if state.previous_output is None:
            raise ValueError("Streaming state has no previous output frame")
        cb0, activity, context = self.stream_step_teacher(
            user_codes, state.previous_output, state
        )
        generated = [cb0.argmax(dim=-1)]
        for target in range(1, 8):
            prefix = torch.stack(generated, dim=-1)[:, None]
            logits = self.depth(prefix, context)[:, 0, -1]
            generated.append(logits.argmax(dim=-1))
        generated_codes = torch.stack(generated, dim=1)
        state.previous_output = generated_codes
        return generated_codes, activity
