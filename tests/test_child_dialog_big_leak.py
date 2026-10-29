"""Leak checks for the big child-dialogue train set."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from child_dialog_big_leak import check_leaks


def test_check_leaks_finds_question_voice_piper_and_frame_overlap():
    train = [
        {
            "question": "leak",
            "voice": "blend:af_sky:af_nova:0.5",
            "group": "color",
            "frame": 2,
        },
        {"question": "ok", "voice": "af_heart", "group": "want", "frame": 3},
        {
            "question": "q",
            "voice": "blend:am_echo:am_adam:0.4",
            "group": "color",
            "frame": 4,
        },
        {
            "question": "p",
            "voice": "en_US-lessac-medium",
            "group": "want",
            "frame": 5,
        },
    ]
    test = [
        {
            "question": "leak",
            "voice": "af_nova",
            "group": "color",
            "frame": 2,
            "split": "test",
        },
        {
            "question": "other",
            "voice": "am_eric",
            "group": "want",
            "frame": 6,
            "split": "test",
        },
    ]
    result = check_leaks(train, test, [{"voice": "en_US-lessac-medium"}])
    assert result["train_rows"] == 4
    assert result["train_questions"] == 4
    assert result["question_overlap"] == ["leak"]
    assert result["test_voice_specs"] == ["blend:af_sky:af_nova:0.5"]
    assert result["af_heart_specs"] == ["af_heart"]
    assert result["piper_voice_specs"] == ["en_US-lessac-medium"]
    assert result["frame_overlap"] == {"color": [2]}
    assert result["ok"] is False


def test_check_leaks_accepts_clean_train_rows():
    train = [
        {"question": "q", "voice": "af_sky", "group": "color", "frame": 1},
        {
            "question": "r",
            "voice": "blend:bf_emma:bf_isabella:0.4",
            "group": "color",
            "frame": 2,
        },
    ]
    test = [
        {
            "question": "z",
            "voice": "af_nova",
            "group": "color",
            "frame": 0,
            "split": "test",
        }
    ]
    result = check_leaks(train, test, [{"voice": "en_US-lessac-medium"}])
    assert result["question_overlap"] == []
    assert result["test_voice_specs"] == []
    assert result["af_heart_specs"] == []
    assert result["piper_voice_specs"] == []
    assert result["frame_overlap"] == {}
    assert result["ok"] is True


def test_topic_grouping_does_not_mix_families_but_catches_real_overlap():
    train = [
        {"question": "q1", "voice": "af_sky", "topic": "color", "frame": 0},
        {"question": "q2", "voice": "af_sky", "topic": "want", "frame": 2},
    ]
    clean = [
        {
            "question": "t1",
            "voice": "af_nova",
            "topic": "color",
            "frame": 1,
            "split": "test",
        },
        {
            "question": "t2",
            "voice": "af_nova",
            "topic": "want",
            "frame": 0,
            "split": "test",
        },
    ]
    result = check_leaks(train, clean)
    assert result["frame_overlap"] == {}
    assert result["ok"] is True

    overlap = [
        {
            "question": "t1",
            "voice": "af_nova",
            "topic": "color",
            "frame": 0,
            "split": "test",
        },
        {
            "question": "t2",
            "voice": "af_nova",
            "topic": "want",
            "frame": 3,
            "split": "test",
        },
    ]
    result = check_leaks(train, overlap)
    assert result["frame_overlap"] == {"color": [0]}
    assert result["ok"] is False


def test_frame_rows_without_a_group_key_raise():
    with pytest.raises(ValueError, match="group, topic or chapter"):
        check_leaks(
            [{"question": "q", "voice": "af_sky", "frame": 0}],
            [],
        )
