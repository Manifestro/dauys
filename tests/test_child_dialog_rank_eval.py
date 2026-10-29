"""Ranking checks for the N-way child-dialogue answer eval."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from child_dialog_rank_eval import (
    collect_candidates,
    score_matrix,
    select_questions,
)


def test_select_questions_uses_seeded_shuffle_per_topic():
    rows = [
        {"id": f"{topic}{index}", "topic": topic}
        for topic in "ab"
        for index in range(5)
    ]
    chosen = select_questions(rows, 2)
    assert [row["id"] for row in chosen] == ["b2", "b3", "a1", "a3"]
    assert select_questions(rows, 2) == chosen
    assert {row["id"] for row in select_questions(rows, 0)} == {
        row["id"] for row in rows
    }


def test_collect_candidates_deduplicates_clips_and_normalizes_answers():
    concepts = {
        "c1": {"answer": "It is blue.", "group": "color"},
        "c2": {"answer": "Blue.", "group": "color"},
        "c3": {"answer": "It is green.", "group": "color"},
        "c4": {"answer": "A vegetable.", "group": "category"},
    }
    rows = [
        {"topic": "color", "concept": "c1", "b_clip": "clips/b1.pt"},
        {"topic": "color", "concept": "c2", "b_clip": "clips/b1.pt"},
        {"topic": "color", "concept": "c3", "b_clip": "clips/b2.pt"},
        {"topic": "category", "concept": "c4", "b_clip": "clips/b3.pt"},
    ]
    candidates = collect_candidates(rows, concepts)
    assert candidates["color"] == [
        ("clips/b1.pt", "blue"),
        ("clips/b2.pt", "green"),
    ]
    assert candidates["category"] == [("clips/b3.pt", "vegetable")]


def test_score_matrix_tracks_top1_top3_chance_rank_and_prior():
    answers = [["blue", "blue", "green", "red"]] * 3
    nll = [
        [0.4, 0.1, 0.2, 0.3],
        [0.3, 0.4, 0.2, 0.1],
        [0.1, 0.3, 0.2, 0.4],
    ]
    prior = [
        [0.3, 0.2, 0.1, 0.4],
        [0.3, 0.4, 0.1, 0.2],
        [0.1, 0.2, 0.3, 0.4],
    ]
    result = score_matrix(
        nll,
        answers,
        ["blue", "green", "red"],
        ["color", "color", "shape"],
        prior,
        ["i1", "i2", "i3"],
        ["q1", "q2", "q3"],
        ["c1", "c2", "c3"],
    )
    assert result == {
        "items": [
            {
                "id": "i1",
                "question": "q1",
                "concept": "c1",
                "family": "color",
                "expected": "blue",
                "predicted": "blue",
                "top1": True,
                "prior_predicted": "green",
            },
            {
                "id": "i2",
                "question": "q2",
                "concept": "c2",
                "family": "color",
                "expected": "green",
                "predicted": "red",
                "top1": False,
                "prior_predicted": "green",
            },
            {
                "id": "i3",
                "question": "q3",
                "concept": "c3",
                "family": "shape",
                "expected": "red",
                "predicted": "blue",
                "top1": False,
                "prior_predicted": "blue",
            },
        ],
        "top1": 0.333,
        "top3": 1.0,
        "chance": 0.333,
        "mean_rank": 1.0,
        "top1_prior": 0.333,
        "prior_agree": 0.333,
        "by_family": {
            "color": {
                "items": 2,
                "top1": 0.5,
                "top3": 1.0,
                "chance": 0.333,
                "mean_rank": 0.5,
                "top1_prior": 0.5,
                "prior_agree": 0.0,
            },
            "shape": {
                "items": 1,
                "top1": 0.0,
                "top3": 1.0,
                "chance": 0.333,
                "mean_rank": 2.0,
                "top1_prior": 0.0,
                "prior_agree": 1.0,
            },
        },
    }


def test_score_matrix_without_prior_leaves_prior_metrics_empty():
    result = score_matrix(
        [[0.1, 0.2]],
        [["blue", "green"]],
        ["blue"],
        ["color"],
    )
    assert result["top1"] == 1.0
    assert result["top1_prior"] is None
    assert result["prior_agree"] is None
    assert result["items"][0]["id"] is None
    assert result["items"][0]["prior_predicted"] is None
