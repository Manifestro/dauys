"""Scoring checks for the child-dialogue spoken-reply eval."""

import json
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

import v02_word_meaning_speak_eval as speak_eval
from child_dialog_speak_eval import (
    best_answer,
    build_parser,
    choose_candidate,
    pick_other_reply,
    rescore,
    words,
)


def test_combine_guidance_endpoints_match_the_chosen_distribution():
    cond = torch.tensor([[1.0, 2.0, 3.0]])
    uncond = torch.tensor([[3.0, 2.0, 1.0]])
    assert torch.equal(
        speak_eval.combine_guidance(cond, uncond, 1.0),
        torch.log_softmax(cond, dim=-1),
    )
    assert torch.equal(
        speak_eval.combine_guidance(cond, uncond, 0.0),
        torch.log_softmax(uncond, dim=-1),
    )


def test_words_spells_numbers_from_digits():
    assert words("before 9 comes 8") == words("before nine comes eight")
    assert words("after 14 comes 15") == words("after fourteen comes fifteen")
    assert words("2 +5 is 7") == {"two", "five", "seven"}
    assert words("21") == {"twenty", "one"}
    assert words("100") == {"one", "hundred"}


def test_digit_transcript_matches_number_answer():
    answers = {"Eight.": ["Eight."], "Fifteen.": ["Fifteen."]}
    assert best_answer("before 9 comes 8", answers) == "Eight."
    assert best_answer("after 14 comes 15", answers) == "Fifteen."


def test_shared_answer_is_one_winner():
    answers = {
        "It is blue.": ["It is blue.", "The sky is blue."],
        "It is green.": ["It is green."],
    }
    assert best_answer("It is blue.", answers) == "It is blue."
    assert best_answer("The sky is blue.", answers) == "It is blue."


def test_different_answers_are_distinguished():
    answers = {
        "A vegetable.": ["A vegetable.", "It is a vegetable."],
        "A fruit.": ["A fruit.", "It is a fruit."],
    }
    assert best_answer("A vegetable.", answers) == "A vegetable."
    assert best_answer("A fruit.", answers) == "A fruit."
    assert best_answer("A rock.", answers) is None


def test_parser_knows_rerank_scoring_arguments():
    parser = build_parser()
    defaults = parser.parse_args(
        ["--concepts", "concepts.jsonl", "--output", "out.json"]
    )
    assert defaults.batch == 16
    assert defaults.onset_frames == 0
    explicit = parser.parse_args(
        [
            "--concepts",
            "concepts.jsonl",
            "--output",
            "out.json",
            "--batch",
            "4",
            "--onset-frames",
            "7",
        ]
    )
    assert explicit.batch == 4
    assert explicit.onset_frames == 7


def test_choose_candidate_handles_infinities_and_ties():
    assert choose_candidate([float("inf"), 1.0, 1.0]) == 1
    assert choose_candidate([2.0, 1.0, 1.0]) == 1
    assert choose_candidate([float("inf"), float("inf")]) == 0


def test_pick_other_reply_uses_a_different_answer_deterministically():
    rows = [
        {"id": "a", "topic": "want", "answer": "Yes.", "b_clip": "a.pt"},
        {"id": "b", "topic": "want", "answer": "No.", "b_clip": "b.pt"},
        {"id": "c", "topic": "want", "answer": "Yes.", "b_clip": "c.pt"},
    ]
    first = pick_other_reply(rows, rows[0], random.Random(0))
    second = pick_other_reply(rows, rows[0], random.Random(0))
    assert first == second == rows[1]


def test_speak_can_include_teacher_forced_seed_frames(monkeypatch):
    class _Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.parameter = torch.nn.Parameter(torch.zeros(()))

        def prefill_teacher(self, user, output, quiet):
            return None

        def generate_step(self, user, state, decoding, rng):
            return torch.full((1, 8), 2), torch.ones(1, dtype=torch.bool)

    class _Codec:
        def decode(self, codes):
            return codes.float()

    monkeypatch.setattr(speak_eval, "decode", lambda codec, codes: codec.decode(codes))
    silence = torch.zeros(8, dtype=torch.long)
    a = torch.zeros(8, 3, dtype=torch.long)
    b = torch.full((8, 5), 9, dtype=torch.long)
    kwargs = {"seed": 2, "frames": 4, "decoding": None, "gated": False, "rng": None}

    generated = speak_eval.speak(_Model(), _Codec(), silence, a, b, **kwargs)
    seeded = speak_eval.speak(
        _Model(), _Codec(), silence, a, b, include_seed=True, **kwargs
    )

    assert generated.tolist() == [2.0, 2.0, 2.0, 2.0]
    assert seeded.tolist() == [9.0, 9.0, 2.0, 2.0, 2.0, 2.0]


def test_speak_codes_returns_unbatched_code_rows():
    class _Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.parameter = torch.nn.Parameter(torch.zeros(()))

        def prefill_teacher(self, user, output, quiet):
            return None

        def generate_step(self, user, state, decoding, rng):
            return torch.full((1, 8), 2), torch.ones(1, dtype=torch.bool)

    silence = torch.zeros(8, dtype=torch.long)
    a = torch.zeros(8, 3, dtype=torch.long)
    b = torch.full((8, 5), 9, dtype=torch.long)
    codes = speak_eval.speak_codes(
        _Model(),
        None,
        silence,
        a,
        b,
        seed=2,
        frames=4,
        decoding=None,
        gated=False,
        rng=None,
    )
    assert codes.shape == (8, 4)
    assert codes.dtype == torch.long


def test_rescore_reuses_saved_transcripts(tmp_path):
    concepts = {
        "c0000": {
            "id": "c0000",
            "group": "color",
            "answer": "It is blue.",
            "replies": ["It is blue.", "Blue."],
        },
        "c0001": {
            "id": "c0001",
            "group": "color",
            "answer": "It is blue.",
            "replies": ["The sky is blue."],
        },
        "c0002": {
            "id": "c0002",
            "group": "color",
            "answer": "It is green.",
            "replies": ["It is green."],
        },
    }
    pairs = [
        {
            "question": "What color is the sky?",
            "concept": "c0000",
            "group": "color",
            "split": "test",
        },
        {
            "question": "What color is the grass?",
            "concept": "c0002",
            "group": "color",
            "split": "test",
        },
    ]
    previous = [
        {
            "variant": "greedy",
            "examples": [
                {
                    "question": "What color is the sky?",
                    "expected": "It is blue.",
                    "heard": "It is blue.",
                    "stopped": True,
                },
                {
                    "question": "What color is the grass?",
                    "expected": "It is green.",
                    "heard": "It is blue.",
                    "stopped": False,
                },
            ],
        }
    ]
    concepts_path = tmp_path / "concepts.jsonl"
    concepts_path.write_text(
        "".join(json.dumps(concept) + "\n" for concept in concepts.values())
    )
    pairs_path = tmp_path / "pairs.jsonl"
    pairs_path.write_text("".join(json.dumps(pair) + "\n" for pair in pairs))
    previous_path = tmp_path / "previous.json"
    previous_path.write_text(json.dumps(previous))

    results = rescore(previous_path, pairs_path, concepts)

    assert results == [
        {
            "variant": "greedy",
            "source": "true",
            "seed": 0,
            "guidance": 1.0,
            "rerank": 0,
            "items": 2,
            "correct": 0.5,
            "correct_first": 0.5,
            "stopped": 0.5,
            "chance": 0.5,
            "follows_seed": 0.5,
            "follows_seed_first": 0.5,
            "examples": [
                {
                    "question": "What color is the sky?",
                    "expected": "It is blue.",
                    "heard": "It is blue.",
                    "ok": True,
                    "ok_first": True,
                    "stopped": True,
                    "seed_answer": "It is blue.",
                    "ok_seed_answer": True,
                    "ok_first_seed_answer": True,
                },
                {
                    "question": "What color is the grass?",
                    "expected": "It is green.",
                    "heard": "It is blue.",
                    "ok": False,
                    "ok_first": False,
                    "stopped": False,
                    "seed_answer": "It is green.",
                    "ok_seed_answer": False,
                    "ok_first_seed_answer": False,
                },
            ],
        }
    ]
