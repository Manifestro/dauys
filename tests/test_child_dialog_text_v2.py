"""Checks for the v2 child-dialogue text set and its text checker."""

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

import child_dialog_build as builder
import child_dialog_text_check as checker
import child_dialog_text_review as reviewer

SCRIPTS = Path(__file__).parents[1] / "scripts"
V1_HASHES = {
    "concepts.jsonl": "9bc0a867e835abf45175d8d7c996a59f5d896a718a98cbde0860ef5562b77cbc",
    "frames.jsonl": "9c194ee72adee1bf763703706d9a9dfc1fe3cc85229adadd0fe5a2020b4d79ee",
    "pairs.jsonl": "6046f8c450213cd9ef3040fa8bd225f11c525193fdcba881e67291e63ee05e87",
}


def _concepts(name: str) -> list[dict]:
    path = SCRIPTS / name
    return [json.loads(line) for line in path.read_text().splitlines()]


def _write_build(directory: Path, concepts: list[dict], version: int) -> None:
    kept, frames, pairs = builder.build(concepts, 0, version)
    directory.mkdir(parents=True, exist_ok=True)
    for name, rows in (("concepts", kept), ("frames", frames), ("pairs", pairs)):
        (directory / f"{name}.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows)
        )


def test_v1_default_output_is_byte_identical(tmp_path):
    concepts = _concepts("child_dialog_concepts_v1.jsonl")
    _write_build(tmp_path, concepts, 1)
    for name, expected in V1_HASHES.items():
        digest = hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()
        assert digest == expected
    kept, frames, pairs = builder.build(concepts, 0)
    assert (len(kept), len(frames), len(pairs)) == (377, 285, 5655)


def test_v2_preserves_v1_test_frames_and_has_no_v1_test_leak(tmp_path):
    v1_concepts = _concepts("child_dialog_concepts_v1.jsonl")
    v2_concepts = _concepts("child_dialog_concepts_v2.jsonl")
    assert (
        (SCRIPTS / "child_dialog_concepts_v2.jsonl")
        .read_text()
        .startswith((SCRIPTS / "child_dialog_concepts_v1.jsonl").read_text())
    )
    v1_concepts_kept, _, v1_pairs = builder.build(v1_concepts, 0, 1)
    v2_concepts_kept, _, v2_pairs = builder.build(v2_concepts, 0, 2)
    assert 900 <= len(v2_concepts_kept) <= 920
    v1_ids = {concept["id"] for concept in v1_concepts_kept}
    v1_test = {
        (pair["concept"], pair["frame"]): pair["question"]
        for pair in v1_pairs
        if pair["split"] == "test"
    }
    v2_test = {
        (pair["concept"], pair["frame"]): pair["question"]
        for pair in v2_pairs
        if pair["split"] == "test" and pair["concept"] in v1_ids
    }
    assert v2_test == v1_test
    v1_test_text = {
        pair["question"].lower() for pair in v1_pairs if pair["split"] == "test"
    }
    assert not [
        pair
        for pair in v2_pairs
        if pair["split"] == "train" and pair["question"].lower() in v1_test_text
    ]
    _, frames, _ = builder.build(v2_concepts, 0, 2)
    split = {(frame["group"], frame["index"]): frame["split"] for frame in frames}
    for group, indexes in builder.HELD_V1.items():
        held = {
            index
            for index in range(len(builder.FRAMES_V2[group]))
            if split[(group, index)] == "test"
        }
        assert held == set(indexes)


def test_check_passes_clean_build_and_fails_on_bad_text(tmp_path):
    v1_dir = tmp_path / "v1"
    v2_dir = tmp_path / "v2"
    _write_build(v1_dir, _concepts("child_dialog_concepts_v1.jsonl"), 1)
    _write_build(v2_dir, _concepts("child_dialog_concepts_v2.jsonl"), 2)
    assert checker.check_dirs(v1_dir, v2_dir) == []

    pairs_path = v2_dir / "pairs.jsonl"
    pairs = [json.loads(line) for line in pairs_path.read_text().splitlines()]
    pairs[0]["question"] = pairs[0]["question"] + "  {oops}"
    pairs_path.write_text("".join(json.dumps(pair) + "\n" for pair in pairs))
    failures = checker.check_dirs(v1_dir, v2_dir)
    assert any("double space" in failure for failure in failures)
    assert any("braces" in failure for failure in failures)


def test_review_lists_every_new_concept_and_totals(tmp_path):
    v1_dir = tmp_path / "v1"
    v2_dir = tmp_path / "v2"
    _write_build(v1_dir, _concepts("child_dialog_concepts_v1.jsonl"), 1)
    _write_build(v2_dir, _concepts("child_dialog_concepts_v2.jsonl"), 2)
    output = tmp_path / "review.md"
    totals = reviewer.render_review(v1_dir, v2_dir, output)
    text = output.read_text()
    assert totals["concepts"] == 906
    assert totals["frames"] == 971
    assert totals["test_v1"] == 1131
    assert totals["test_new"] == 1587
    assert totals["new_concepts"] == 529
    v1_ids = {concept["id"] for concept in _concepts("child_dialog_concepts_v1.jsonl")}
    for concept in _concepts("child_dialog_concepts_v2.jsonl"):
        if concept["id"] not in v1_ids:
            assert f"- {concept['key']} -> {concept['answer']}" in text


def test_computed_concepts_and_category_options_are_consistent():
    number_words = [
        "zero",
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "nine",
        "ten",
        "eleven",
        "twelve",
        "thirteen",
        "fourteen",
        "fifteen",
        "sixteen",
        "seventeen",
        "eighteen",
        "nineteen",
        "twenty",
        "twenty-one",
    ]
    concepts = _concepts("child_dialog_concepts_v2.jsonl")
    new = [concept for concept in concepts if int(concept["id"][1:]) >= 1000]
    assert [int(concept["id"][1:]) for concept in new] == list(
        range(1000, 1000 + len(new))
    )
    for concept in new:
        if concept["family"] == "add":
            x, y = (number_words.index(word) for word in concept["key"].split())
            assert concept["answer"] == f"{number_words[x + y].capitalize()}."
        elif concept["family"] == "number_next":
            index = number_words.index(concept["key"])
            assert concept["answer"] == f"{number_words[index + 1].capitalize()}."
        elif concept["family"] == "number_before":
            index = number_words.index(concept["key"])
            assert concept["answer"] == f"{number_words[index - 1].capitalize()}."
        elif concept["family"] == "category":
            answer_group = concept["answer"].split()[1].rstrip(".").lower()
            assert answer_group in concept["options"]


def test_checker_catches_rendering_mass_and_reply_defects():
    cases = [
        ("The bear lives in the forest..", "repeated sentence punctuation"),
        ("You put socks. on your ankles", "period before lowercase continuation"),
        ("The shape is triangle", "shape word without an article"),
        ("What taste does a maple syrup have?", "article before a mass noun"),
        ("It is a fruit., of course.", "punctuation collision"),
        ("It is a fruit,.", "punctuation collision"),
        ("What is this ?", "space before punctuation"),
    ]
    for text, expected in cases:
        problems = checker.text_problems(text, "v2 test")
        assert any(expected in problem for problem in problems)

    concept = {
        "id": "c1",
        "family": "color",
        "key": "sky",
        "answer": "It is blue.",
        "replies": ["Yes."],
        "group": "color",
    }
    pair = {
        "id": "c1-f00",
        "concept": "c1",
        "family": "color",
        "group": "color",
        "key": "sky",
        "frame": 0,
        "question": "What color is the sky?",
        "answer": "It is blue.",
        "replies": ["Yes."],
        "split": "train",
    }
    failures = checker.check_general("v2", [concept], [pair])
    assert any("reply missing answer word" in failure for failure in failures)


def test_render_collapses_punctuation_and_shape_keeps_article():
    assert builder.render("It is {ans}", {"ans": "a toy."}) == "It is a toy."
    assert builder.render("It is {ans}", {"ans": "a toy.."}) == "It is a toy."
    assert (
        builder.render("It is {ans}", {"ans": "so much fun!"}) == "It is so much fun!"
    )
    assert (
        builder.render("It is {ans}, of course.", {"ans": "a fruit."})
        == "It is a fruit, of course."
    )
    assert (
        builder.render(
            "You put {ans_l} on your {k}.",
            {"ans_l": "socks.", "k": "ankles"},
        )
        == "You put socks on your ankles."
    )

    concepts = _concepts("child_dialog_concepts_v2.jsonl")
    kept, _, _ = builder.build(concepts, 0, 2)
    shape = {
        concept["key"]: concept for concept in kept if concept["family"] == "shape"
    }
    triangle = shape["slice of pizza"]
    assert all("is triangle" not in reply for reply in triangle["replies"])
    assert "The shape of the slice of pizza is a triangle." in triangle["replies"]
    assert (
        builder.fields(
            {"family": "taste", "key": "maple syrup", "answer": "It is sweet."}
        )["a"]
        == ""
    )
    category = {
        concept["key"]: concept for concept in kept if concept["family"] == "category"
    }
    assert "It is a fruit, of course." in category["apple"]["replies"]
    assert not [
        reply
        for concept in category.values()
        for reply in concept["replies"]
        if ".," in reply or ",." in reply
    ]


def test_body_extras_use_our_and_my_forms():
    for template in builder.EXTRA_FRAMES["body"]:
        assert "{k_a}" not in template
        assert "a {k}" not in template
    concepts = _concepts("child_dialog_concepts_v2.jsonl")
    kept, _, pairs = builder.build(concepts, 0, 2)
    new_body_ids = {
        concept["id"]
        for concept in kept
        if concept["family"] == "body" and int(concept["id"][1:]) >= 1000
    }
    questions = {
        pair["question"]
        for pair in pairs
        if pair["concept"] in new_body_ids and pair["split"] == "train"
    }
    assert not [
        question
        for question in questions
        if "What is a" in question and "used for" in question
    ]
    assert not [question for question in questions if "What does a" in question]
    assert any("our neck" in question for question in questions)
    assert any("our fingers" in question for question in questions)


def test_number_families_cover_eleven_through_twenty():
    concepts = _concepts("child_dialog_concepts_v2.jsonl")
    words = [
        "eleven",
        "twelve",
        "thirteen",
        "fourteen",
        "fifteen",
        "sixteen",
        "seventeen",
        "eighteen",
        "nineteen",
        "twenty",
    ]
    for family in ("number_next", "number_before"):
        keys = {concept["key"] for concept in concepts if concept["family"] == family}
        assert set(words) <= keys
