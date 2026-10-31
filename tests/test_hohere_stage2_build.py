"""Assistant-turn records from hohere-1m event labels."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from hohere_stage2_build import FRAME_SAMPLES, activity_from_events, reply_cuts


def _event(speaker, start, end):
    return {
        "speaker": speaker,
        "start_sample": start * FRAME_SAMPLES,
        "end_sample": end * FRAME_SAMPLES,
    }


def test_activity_covers_only_the_speakers_turns():
    events = [_event("A", 2, 4), _event(None, 4, 5), _event("B", 5, 7)]
    assert activity_from_events(events, "A", 8).tolist() == [0, 0, 1, 1, 0, 0, 0, 0]
    assert activity_from_events(events, "B", 8).tolist() == [0, 0, 0, 0, 0, 1, 1, 0]


def test_reply_cuts_need_heard_speech_and_room_to_reply():
    events = [
        _event("B", 10, 20),  # too early
        _event("A", 20, 60),
        _event("B", 62, 80),
        _event("B", 180, 199),  # too close to the end
    ]
    user = activity_from_events(events, "A", 200)
    assert reply_cuts(events, 200, user) == [62]
