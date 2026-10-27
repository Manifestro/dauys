"""Batched audio tokenization and event alignment without a codec download."""

from contextlib import nullcontext
from dataclasses import replace

import numpy as np
import pytest
import torch

from dauys.v02_encode import (
    AudioConversation,
    companion_from_events,
    encode_conversations,
    encode_long_conversation,
)


class _Codec:
    def streaming(self, batch_size):
        assert batch_size == 2
        return nullcontext()

    def encode(self, pcm):
        batch, _, samples = pcm.shape
        frames = samples // 1920
        means = pcm.reshape(batch, frames, 1920).mean(dim=-1)
        coarse = (means * 100).round().long().clamp(0, 2040)
        return torch.stack([coarse + index for index in range(8)], dim=1)


class _BatchJitterCodec(_Codec):
    def encode(self, pcm):
        codes = super().encode(pcm)
        if pcm.shape[0] > 1:
            codes[0, 0, 0] += 1
        return codes


def _conversation(identity, frames):
    events = (
        {
            "event_type": "normal_turn",
            "speaker": "A",
            "start_sample": 0,
            "end_sample": 2 * 1920,
        },
        {
            "event_type": "normal_turn",
            "speaker": "B",
            "start_sample": 2 * 1920,
            "end_sample": 4 * 1920,
        },
    )
    return AudioConversation(
        id=identity,
        user=np.ones(frames * 1920, dtype=np.float32),
        target=np.full(frames * 1920, 2, dtype=np.float32),
        samples=frames * 1920,
        events=events,
        scenario="topic/family",
        skill="simple_dialogue",
        voice_a="voice-a",
    )


def test_batched_encoding_matches_single_item_and_keeps_frame_lengths():
    long = _conversation("long", 8)
    short = _conversation("short", 5)
    codec = _Codec()
    batched = encode_conversations(
        codec, [long, short], torch.device("cpu"), check_first_against_single=True
    )
    single = encode_conversations(codec, [short], torch.device("cpu"))
    assert batched[0]["user"].shape == (8, 8)
    assert batched[1]["target"].shape == (8, 5)
    torch.testing.assert_close(batched[1]["user"], single[0]["user"])
    torch.testing.assert_close(batched[1]["target"], single[0]["target"])


def test_batch_check_allows_small_quantization_differences_but_rejects_large_ones():
    with pytest.warns(RuntimeWarning, match="1/3200 tokens"):
        encode_conversations(
            _BatchJitterCodec(),
            [_conversation("long", 400)],
            torch.device("cpu"),
            check_first_against_single=True,
        )
    with pytest.raises(ValueError, match="1/800 tokens"):
        encode_conversations(
            _BatchJitterCodec(),
            [_conversation("short", 100)],
            torch.device("cpu"),
            check_first_against_single=True,
        )


def test_companion_activity_and_response_onset_align_to_audio():
    companion = companion_from_events(_conversation("example", 8))
    assert companion["activity_a"][:4] == [1, 1, 0, 0]
    assert companion["activity_b"][:4] == [0, 0, 1, 1]
    events = companion["event_frames"]
    assert any(
        item["type"] == "a_turn_end" and item["start_frame"] == 2 for item in events
    )
    assert any(
        item["type"] == "response_onset" and item["start_frame"] == 2 for item in events
    )


def test_long_conversation_encodes_in_bounded_streaming_chunks():
    item = _conversation("long", 12)
    result = encode_long_conversation(_Codec(), item, torch.device("cpu"), 4)
    reference = encode_conversations(_Codec(), [item], torch.device("cpu"))[0]
    torch.testing.assert_close(result["user"], reference["user"])
    torch.testing.assert_close(result["target"], reference["target"])


def test_event_companion_marks_only_real_overlapping_responses():
    item = _conversation("overlap", 50)

    def speech(kind, speaker, start, end):
        return {
            "event_type": kind,
            "speaker": speaker,
            "start_sample": start * 1920,
            "end_sample": end * 1920,
        }

    item = replace(
        item,
        events=(
            speech("normal_turn", "A", 0, 20),
            speech("backchannel", "B", 15, 18),
            speech("normal_turn", "B", 18, 25),
            speech("interruption", "B", 35, 38),
        ),
    )
    events = companion_from_events(item)["event_frames"]
    onsets = [
        event["start_frame"] for event in events if event["type"] == "response_onset"
    ]
    assert onsets == [18]
