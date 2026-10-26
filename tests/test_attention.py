"""CPU checks for the causal attention and depth paths."""

import torch
from torch import nn

from dauys.config import ModelConfig
from dauys.model import AttentionCache, DepthDecoder, FrameAttention


def test_self_attention_matches_incremental_window():
    torch.manual_seed(11)
    attention = FrameAttention(32, 32, heads=4, window=3).eval()
    inputs = torch.randn(2, 6, 32)
    positions = torch.arange(6)
    with torch.inference_mode():
        batched = attention(inputs, inputs, positions, positions)
        cache = AttentionCache()
        streamed = []
        for frame in range(6):
            output, cache = attention.step_self(inputs[:, frame : frame + 1], frame, cache)
            streamed.append(output)
    torch.testing.assert_close(torch.cat(streamed, dim=1), batched)


def test_cross_attention_hides_future_input():
    torch.manual_seed(12)
    attention = FrameAttention(32, 24, heads=4, window=3).eval()
    query = torch.randn(1, 5, 32)
    context = torch.randn(1, 5, 24)
    changed = context.clone()
    changed[:, 3:] += 100
    positions = torch.arange(5)
    with torch.inference_mode():
        baseline = attention(query, context, positions, positions)
        modified = attention(query, changed, positions, positions)
    torch.testing.assert_close(baseline[:, :3], modified[:, :3])


def test_depth_decoder_hides_later_codebooks():
    torch.manual_seed(13)
    base = ModelConfig.stage1()
    config = ModelConfig(
        **{
            **vars(base),
            "vocab_size": 32,
            "embedding_dim": 16,
            "dialogue_dim": 32,
            "depth_dim": 24,
            "depth_heads": 6,
        }
    )
    embeddings = nn.ModuleList(nn.Embedding(32, 16) for _ in range(8))
    decoder = DepthDecoder(config, embeddings).eval()
    prefix = torch.randint(0, 32, (2, 3, 7))
    changed = prefix.clone()
    changed[:, :, 4:] = torch.randint(0, 32, (2, 3, 3))
    context = torch.randn(2, 3, 32)
    with torch.inference_mode():
        baseline = decoder(prefix, context)
        modified = decoder(changed, context)
    assert baseline.shape == (2, 3, 7, 32)
    torch.testing.assert_close(baseline[:, :, :4], modified[:, :, :4])
