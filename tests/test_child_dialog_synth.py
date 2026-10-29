"""Reply-voice planning checks for child-dialogue synthesis."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from child_dialog_synth import (
    REPLY_VOICE,
    TEST_VOICES,
    TRAIN_VOICES,
    clip_name,
    make_big_plan,
    make_plan,
    shard_plan,
)


def _pair(identifier: str, split: str, replies: list[str]) -> dict:
    return {
        "id": identifier,
        "split": split,
        "concept": f"c-{identifier}",
        "family": "color",
        "group": "color",
        "key": "sky",
        "frame": 0,
        "question": "What color is the sky?",
        "replies": replies,
    }


def test_single_reply_voice_keeps_existing_reply_clip_names():
    pairs = [
        _pair("p0", "test", ["It is blue.", "Blue."]),
        _pair("p1", "train", ["It is green."]),
    ]
    plan = make_plan(pairs, 1, reply_voices=(REPLY_VOICE,))
    assert all(row["reply_voice"] == REPLY_VOICE for row in plan)
    assert {row["id"]: row["b_wav"] for row in plan} == {
        "p0-af_nova": "af058b54967cbb2f.wav",
        "p0-am_eric": "af058b54967cbb2f.wav",
        "p0-bf_lily": "f8aad788d517da3d.wav",
        "p0-bm_lewis": "f8aad788d517da3d.wav",
        "p1-af_sky": "eed2334690208c63.wav",
    }


def test_six_reply_voices_are_all_used_and_get_distinct_clips():
    reply_voices = (
        "af_heart",
        "am_adam",
        "bf_emma",
        "af_bella",
        "am_michael",
        "bf_isabella",
    )
    pairs = [
        _pair(identifier, "train", ["It is blue."])
        for identifier in ("p0", "p1", "p3", "p9", "p19", "p21")
    ]
    plan = make_plan(pairs, 1, reply_voices=reply_voices)
    assert {row["reply_voice"] for row in plan} == set(reply_voices)
    assert len({row["b_wav"] for row in plan}) == len(reply_voices)
    assert all(
        row["b_wav"] == clip_name(row["reply"], row["reply_voice"]) for row in plan
    )


def test_make_big_plan_is_deterministic_and_train_only():
    pairs = [
        _pair("p0", "train", ["It is blue.", "Blue."]),
        _pair("p1", "train", ["It is green."]),
        _pair("t0", "test", ["It is red."]),
    ]
    first = make_big_plan(pairs, 20, seed=7)
    second = make_big_plan(pairs, 20, seed=7)
    assert first == second
    assert len(first) == 40
    assert all(row["split"] == "train" for row in first)
    assert not any(row["id"].startswith("t0") for row in first)


def test_make_big_plan_voices_speeds_names_and_ranks():
    pairs = [
        _pair("p0", "train", ["It is blue.", "Blue."]),
        _pair("p1", "train", ["It is green."]),
    ]
    plan = make_big_plan(pairs, 50, seed=0)
    assert any(row["voice"].startswith("blend:") for row in plan)
    assert any(not row["voice"].startswith("blend:") for row in plan)
    for row in plan:
        spec = row["voice"]
        assert 0.85 <= row["speed"] <= 1.15
        assert row["speed"] == round(row["speed"], 2)
        assert 0 <= row["rank"] < 1
        assert row["reply_voice"] == REPLY_VOICE
        assert spec not in TEST_VOICES
        assert spec != REPLY_VOICE
        if spec.startswith("blend:"):
            _, first, second, weight = spec.split(":")
            assert first != second
            assert first[0] == second[0]
            assert first in TRAIN_VOICES
            assert second in TRAIN_VOICES
            assert 0.3 <= float(weight) <= 0.7
            assert weight == f"{float(weight):.2f}"
        else:
            assert spec in TRAIN_VOICES
        assert row["a_wav"] == clip_name(row["question"], f"{spec}@{row['speed']}")
        assert row["b_wav"] == clip_name(row["reply"], REPLY_VOICE)


def test_shard_plan_partitions_by_rank_without_overlap():
    plan = [{"id": f"r{index}", "rank": index / 10} for index in range(10)]
    shards = [shard_plan(plan, index, 3) for index in range(3)]
    assert sorted(item["id"] for shard in shards for item in shard) == sorted(
        item["id"] for item in plan
    )
    assert len({item["id"] for shard in shards for item in shard}) == len(plan)
    assert max(item["rank"] for item in shards[0]) < min(
        item["rank"] for item in shards[1]
    )
    assert max(item["rank"] for item in shards[1]) < min(
        item["rank"] for item in shards[2]
    )
