"""CPU checks for re-encoded child-dialogue question clips."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from child_dialog_reencode import build_rows, trim_silence


def test_trim_silence_keeps_two_frame_margins():
    codes = torch.arange(8 * 10).reshape(8, 10)
    activity = torch.zeros(10, dtype=torch.long)
    activity[3:6] = 1
    trimmed, trimmed_activity = trim_silence(codes, activity)
    assert torch.equal(trimmed, codes[:, 1:8])
    assert trimmed_activity.tolist() == [0, 0, 1, 1, 1, 0, 0]
    silent, silent_activity = trim_silence(codes, torch.zeros(10, dtype=torch.long))
    assert silent.shape == (8, 0)
    assert silent_activity.shape == (0,)
    with pytest.raises(ValueError, match="Activity"):
        trim_silence(codes, torch.zeros(9, dtype=torch.long))


def test_build_rows_copies_metadata_and_trims_question_codes():
    original = [
        {
            "id": "r1",
            "question": "What?",
            "b_clip": "clips/b1.pt",
            "topic": "color",
        }
    ]
    plan = [
        {
            "row": "r1",
            "voice": "v1",
            "variant": "clean",
            "wav": "clean/v1/r1.wav",
        },
        {
            "row": "r1",
            "voice": "v1",
            "variant": "phone",
            "wav": "phone/v1/r1.wav",
        },
    ]
    codes = torch.arange(8 * 5).reshape(8, 5)
    activity = torch.tensor([0, 1, 1, 0, 0], dtype=torch.long)
    encoded = {
        "clean/v1/r1.wav": (codes, activity),
        "phone/v1/r1.wav": (codes, activity),
    }
    variants = build_rows(original, plan, encoded, margin=1)
    assert set(variants) == {"clean", "phone"}
    row, new_codes, new_activity = variants["clean"][0]
    assert row["id"] == "r1-v1"
    assert row["speaker"] == "v1"
    assert row["a_clip"] == "clips/q-r1-v1.pt"
    assert row["b_clip"] == "clips/b1.pt"
    assert row["question"] == "What?"
    assert torch.equal(new_codes, codes[:, :4])
    assert new_activity.tolist() == [0, 1, 1, 0]
