"""Frozen Mimi access through an explicitly supplied local checkpoint."""

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import torch


def load_mimi(checkpoint: str | Path, device: str | torch.device):
    """Load only the codec weights; never fetch a dialogue-model checkpoint."""
    from moshi.models import loaders

    path = Path(checkpoint)
    if not path.is_file():
        raise FileNotFoundError(path)
    codec = loaders.get_mimi(path, device=device, num_codebooks=8)
    codec.set_num_codebooks(8)
    codec.eval()
    codec.requires_grad_(False)
    if codec.sample_rate != 24_000 or codec.frame_rate != 12.5:
        raise ValueError("Unexpected Mimi frame geometry")
    return codec


def validate_pcm(pcm: torch.Tensor) -> None:
    if pcm.ndim != 3 or pcm.shape[1] != 1:
        raise ValueError("PCM must have shape [batch, 1, samples]")
    if pcm.shape[-1] % 1920:
        raise ValueError("PCM length must be a multiple of 1920 samples")


def validate_codes(codes: torch.Tensor) -> None:
    if codes.ndim != 3 or codes.shape[1] != 8:
        raise ValueError("Mimi codes must have shape [batch, 8, frames]")
    if codes.numel() and (torch.any(codes < 0) or torch.any(codes >= 2048)):
        raise ValueError("Mimi code index outside vocabulary")


@torch.no_grad()
def encode(codec, pcm: torch.Tensor) -> torch.Tensor:
    validate_pcm(pcm)
    codes = codec.encode(pcm)
    validate_codes(codes)
    return codes


@torch.no_grad()
def decode(codec, codes: torch.Tensor) -> torch.Tensor:
    validate_codes(codes)
    return codec.decode(codes)


@contextmanager
def streaming(codec, batch_size: int) -> Iterator:
    with codec.streaming(batch_size=batch_size):
        yield codec
