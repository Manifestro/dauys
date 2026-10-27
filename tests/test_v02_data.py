"""Immutable corpus and companion contract checks using local token fixtures."""

import json
import random
from pathlib import Path

import pytest
import torch

from dauys.v02_data import (
    CorpusDataset,
    assert_suite_is_disjoint,
    collate_cases,
    load_corpus,
    sample_records,
    sha256,
)


def _corpus(tmp_path: Path):
    manifest = tmp_path / "manifest.jsonl"
    companion = tmp_path / "companion.jsonl"
    rows = []
    details = []
    sources = ("1a_on_A", "1a_on_B", "1b", "hohere", "hohere")
    skills = ("speech", "speech", "two_party", "simple_dialogue", "interruption")
    for index, (source, skill) in enumerate(zip(sources, skills)):
        identity = f"conv_{index}"
        frames = 12 + index
        path = tmp_path / f"{identity}.pt"
        activity_a = torch.zeros(frames, dtype=torch.long)
        activity_b = torch.zeros(frames, dtype=torch.long)
        if source != "1a_on_B":
            activity_a[4:8] = 1
        if source != "1a_on_A":
            activity_b[4:8] = 1
        torch.save(
            {
                "user": torch.full(
                    (8, frames),
                    0 if source == "1a_on_B" else index + 1,
                    dtype=torch.long,
                ),
                "target": torch.full(
                    (8, frames),
                    0 if source == "1a_on_A" else index + 2,
                    dtype=torch.long,
                ),
                "activity": activity_b,
                "user_activity": activity_a,
            },
            path,
        )
        rows.append(
            {
                "id": identity,
                "source": source,
                "skill": skill,
                "family": f"family_{index}",
                "user_voice": f"voice_{index}",
                "frames": frames,
                "tokens_path": path.name,
                "sha256": sha256(path),
            }
        )
        details.append(
            {
                "id": identity,
                "activity_a": activity_a.tolist(),
                "activity_b": activity_b.tolist(),
                "event_frames": [
                    {
                        "type": "a_turn_end",
                        "speaker": "A",
                        "start_frame": 3,
                        "end_frame": 4,
                    }
                ],
            }
        )
    manifest.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    companion.write_text("\n".join(json.dumps(row) for row in details) + "\n")
    return manifest, companion


def test_corpus_load_collate_and_disjointness(tmp_path):
    manifest, companion = _corpus(tmp_path)
    records = load_corpus(tmp_path, manifest, companion)
    dataset = CorpusDataset(records)
    batch = collate_cases([dataset[0], dataset[4]], torch.zeros(8, dtype=torch.long))
    assert batch.user_codes.shape == (2, 8, 16)
    assert batch.valid_frames.sum(dim=1).tolist() == [12, 16]
    assert batch.sources == ("1a_on_A", "hohere")
    assert batch.events[0][0].type == "a_turn_end"
    assert_suite_is_disjoint(
        records,
        [
            {
                "id": "unseen",
                "family": "new",
                "user_voice": "new_voice",
                "unseen_voice": True,
            }
        ],
    )
    with pytest.raises(ValueError, match="overlaps"):
        assert_suite_is_disjoint(records, [{"id": "unseen", "family": "family_0"}])
    assert_suite_is_disjoint(
        records,
        [
            {
                "id": "seen",
                "family": "new",
                "user_voice": "voice_0",
                "unseen_voice": False,
            }
        ],
    )
    with pytest.raises(ValueError, match="marks a training voice as unseen"):
        assert_suite_is_disjoint(
            records,
            [
                {
                    "id": "unseen",
                    "family": "new",
                    "user_voice": "voice_0",
                    "unseen_voice": True,
                }
            ],
        )


def test_corpus_detects_checksum_change(tmp_path):
    manifest, companion = _corpus(tmp_path)
    (tmp_path / "conv_0.pt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        load_corpus(tmp_path, manifest, companion)


def test_cropping_preserves_previous_b_frame_and_event_alignment(tmp_path):
    manifest, companion = _corpus(tmp_path)
    records = load_corpus(tmp_path, manifest, companion)
    dataset = CorpusDataset(records, crop_frames=6)
    case = dataset[(4, 3)]
    assert case["user"].shape == (8, 6)
    assert case["record"].frames == 6
    # The initial B token is the frame immediately before the selected window.
    full = torch.load(records[4].token_path, weights_only=True)["target"]
    start = random.Random(3).randrange(full.shape[1] - 6 + 1)
    if start:
        torch.testing.assert_close(case["initial_output"], full[:, start - 1])


def test_stage_sampler_is_seeded_and_uses_both_sources(tmp_path):
    manifest, companion = _corpus(tmp_path)
    records = load_corpus(tmp_path, manifest, companion)
    a = sample_records(records, "pretrain", 0.0, 100, 42)
    b = sample_records(records, "pretrain", 0.0, 100, 42)
    assert [item.id for item in a] == [item.id for item in b]
    assert any(item.source == "1b" for item in a)
    dialogue = sample_records(records, "dialogue", 0.5, 100, 42)
    assert any(item.source == "hohere" for item in dialogue)
    assert sum(item.source != "hohere" for item in dialogue) == 15


def test_targeted_posttraining_samples_only_selected_skill(tmp_path):
    manifest, companion = _corpus(tmp_path)
    records = load_corpus(tmp_path, manifest, companion)
    selected = sample_records(records, "posttrain", 1.0, 8, 42)
    assert {item.id for item in selected} == {records[4].id}


def test_replay_fraction_is_exact_across_small_batches(tmp_path):
    manifest, companion = _corpus(tmp_path)
    records = load_corpus(tmp_path, manifest, companion)
    selected = [
        sample_records(records, "dialogue", 0.5, 2, 42 + step, replay_offset=step * 2)
        for step in range(20)
    ]
    assert sum(item.source != "hohere" for batch in selected for item in batch) == 6
