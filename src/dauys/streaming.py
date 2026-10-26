"""Frame-synchronous, full-duplex PCM interface."""

from contextlib import ExitStack
from typing import Self

import torch

from .codec import validate_codes, validate_pcm
from .model import DauysHybrid, StreamingState


class AudioSession:
    def __init__(self, model: DauysHybrid, codec, batch_size: int):
        if batch_size < 1:
            raise ValueError("Batch size must be positive")
        self.model = model
        self.codec = codec
        self.batch_size = batch_size
        self._stack = ExitStack()
        self.state: StreamingState | None = None

    def __enter__(self) -> Self:
        if self.state is not None:
            raise RuntimeError("Session is already active")
        device = next(self.model.parameters()).device
        silence = torch.zeros(self.batch_size, 1, 1920, device=device)
        with torch.no_grad():
            silence_codes = self.codec.encode(silence)
        validate_codes(silence_codes)
        if silence_codes.shape != (self.batch_size, 8, 1):
            raise ValueError("Mimi did not return exactly one silence frame")
        self._stack.enter_context(self.codec.streaming(batch_size=self.batch_size))
        self.state = self.model.new_stream(silence_codes[:, :, 0])
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self._stack.close()
        self.state = None

    @torch.inference_mode()
    def step(self, user_pcm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.state is None:
            raise RuntimeError("Start the audio session before stepping")
        validate_pcm(user_pcm)
        if user_pcm.shape != (self.batch_size, 1, 1920):
            raise ValueError("Exactly one 80 ms user frame is required")
        user_codes = self.codec.encode(user_pcm)
        validate_codes(user_codes)
        if user_codes.shape != (self.batch_size, 8, 1):
            raise ValueError("Mimi did not return exactly one user frame")
        output_codes, activity_logits = self.model.generate_step(
            user_codes[:, :, 0], self.state
        )
        output_pcm = self.codec.decode(output_codes[:, :, None])
        if output_pcm.shape != (self.batch_size, 1, 1920):
            raise ValueError("Mimi did not decode exactly one output frame")
        return output_pcm, activity_logits
