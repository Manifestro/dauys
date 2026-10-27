"""Canonical Hohere labels and benchmark audio integration."""

import hashlib
import json

import pytest
import torch

from dauys.v02_data import sha256
from dauys.v02_evaluation import load_suite
from dauys.v02_hohere import (
    bench_item,
    cache_with_label,
    companion_from_label,
    materialize_bench_row,
)


def _label():
    return {
        "id": "conv_1",
        "sample_rate": 24_000,
        "frame_samples": 1920,
        "num_frames": 8,
        "activity_a": [0, 1, 1, 0, 0, 0, 0, 0],
        "activity_b": [0, 0, 0, 1, 1, 0, 0, 0],
        "events": [
            {
                "type": "turn_end",
                "speaker": "A",
                "start_frame": 1,
                "end_frame": 3,
                "next_speaker": "B",
                "event_id": "a",
            },
            {
                "type": "response_onset",
                "speaker": "B",
                "start_frame": 3,
                "end_frame": 5,
                "event_id": "b",
            },
            {
                "type": "backchannel",
                "speaker": "B",
                "start_frame": 6,
                "end_frame": 7,
                "event_id": "bc",
            },
        ],
    }


def test_canonical_labels_preserve_timing_and_override_approximate_activity():
    companion = companion_from_label(_label(), 8)
    assert [
        (event["type"], event["start_frame"]) for event in companion["event_frames"]
    ] == [("a_turn_end", 3), ("response_onset", 3), ("backchannel", 6)]
    codes = {
        "user": torch.ones(8, 8),
        "target": torch.ones(8, 8),
        "activity": torch.zeros(8),
    }
    updated = cache_with_label(codes, companion)
    assert updated["activity"].tolist() == _label()["activity_b"]
    assert updated["user_activity"].tolist() == _label()["activity_a"]
    assert torch.equal(updated["user"], codes["user"])


def test_canonical_labels_reject_frame_mismatch():
    with pytest.raises(ValueError, match="label count"):
        companion_from_label(_label(), 7)


def test_benchmark_rows_preserve_seen_voice_and_separate_analysis(tmp_path):
    wav = b"RIFFfake-pcm"
    row = {
        "id": "e2-conv_1",
        "voice_a": "af_sarah",
        "voice_a_seen_in_training": True,
        "sample_rate": 24_000,
        "total_samples": 3840,
        "num_frames": 2,
        "channel_a": {"bytes": wav, "path": "source.wav"},
        "channel_b": {"bytes": wav, "path": "source.wav"},
        "info_json": json.dumps({"source_id": "conv_1", "user_text": "Hello"}),
    }
    item = materialize_bench_row(row, "E2", tmp_path)
    assert not item["unseen_voice"]
    assert item["analysis"]["user_text"] == "Hello"
    assert item["audio"]["channel_a"]["sha256"] == hashlib.sha256(wav).hexdigest()
    assert (tmp_path / item["audio"]["channel_a"]["path"]).read_bytes() == wav
    assert bench_item({**row, "voice_a_seen_in_training": False}, "E2")["unseen_voice"]
    manifest = tmp_path / "E2.json"
    manifest.write_text(
        json.dumps({"name": "E2", "version": "v0.2.0", "items": [item]})
    )
    assert load_suite(manifest, sha256(manifest)).items[0]["id"] == row["id"]
    (tmp_path / item["audio"]["channel_a"]["path"]).write_bytes(b"tampered")
    with pytest.raises(ValueError, match="audio checksum"):
        load_suite(manifest, sha256(manifest))


def test_benchmark_rejects_inconsistent_frame_count():
    with pytest.raises(ValueError, match="frame geometry"):
        bench_item(
            {
                "id": "e1",
                "voice_a": "v",
                "voice_a_seen_in_training": False,
                "sample_rate": 24_000,
                "total_samples": 3841,
                "num_frames": 2,
            },
            "E1",
        )
