"""Opt-in integration test against a locally supplied Mimi checkpoint."""

import os

import pytest
import torch

from dauys.codec import load_mimi, validate_codes, validate_pcm


def test_audio_geometry_validation():
    validate_pcm(torch.zeros(2, 1, 1920))
    validate_codes(torch.zeros(2, 8, 1, dtype=torch.long))
    with pytest.raises(ValueError, match="multiple of 1920"):
        validate_pcm(torch.zeros(2, 1, 1919))
    with pytest.raises(ValueError, match="outside vocabulary"):
        validate_codes(torch.full((2, 8, 1), 2048, dtype=torch.long))


def test_mimi_offline_streaming_parity():
    checkpoint = os.environ.get("DAUYS_MIMI_CHECKPOINT")
    if not checkpoint:
        pytest.skip("Set DAUYS_MIMI_CHECKPOINT for the codec integration check")
    if not torch.cuda.is_available():
        pytest.skip("The codec integration check requires CUDA")
    pytest.importorskip("moshi")
    codec = load_mimi(checkpoint, "cuda")
    pcm = torch.randn(1, 1, 1920 * 5, device="cuda")
    with torch.inference_mode():
        offline_codes = codec.encode(pcm)
        offline_pcm = codec.decode(offline_codes)
        with codec.streaming(batch_size=1):
            pieces = [
                codec.encode(pcm[:, :, index : index + 1920])
                for index in range(0, pcm.shape[-1], 1920)
            ]
        streaming_codes = torch.cat(pieces, dim=-1)
        with codec.streaming(batch_size=1):
            decoded = [
                codec.decode(streaming_codes[:, :, index : index + 1])
                for index in range(streaming_codes.shape[-1])
            ]
        streaming_pcm = torch.cat(decoded, dim=-1)
    torch.testing.assert_close(streaming_codes, offline_codes)
    assert offline_pcm.shape == pcm.shape
    assert streaming_pcm.shape == pcm.shape
