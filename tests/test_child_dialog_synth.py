"""Reply-voice planning checks for child-dialogue synthesis."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from child_dialog_synth import REPLY_VOICE, clip_name, make_plan


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
