"""Selection checks for the child-dialogue Piper re-voicing plan."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from child_dialog_piper import pick_rows, select_rows


def test_pick_rows_keeps_first_id_per_question():
    rows = [
        {"id": "c", "question": "same?"},
        {"id": "a", "question": "other?"},
        {"id": "b", "question": "same?"},
        {"id": "d", "question": "third?"},
    ]
    picked = pick_rows(rows)
    assert {row["question"]: row["id"] for row in picked} == {
        "same?": "b",
        "other?": "a",
        "third?": "d",
    }


def test_select_rows_limits_each_topic_then_sorts_by_id():
    rows = [
        {"id": f"{topic}{index}", "topic": topic}
        for topic in ("a", "b")
        for index in range(5)
    ]
    assert [row["id"] for row in select_rows(rows, 2)] == [
        "a1",
        "a3",
        "b2",
        "b3",
    ]
    assert [row["id"] for row in select_rows(rows, 0)] == [
        "a0",
        "a1",
        "a2",
        "a3",
        "a4",
        "b0",
        "b1",
        "b2",
        "b3",
        "b4",
    ]
    with pytest.raises(ValueError, match="negative"):
        select_rows(rows, -1)
