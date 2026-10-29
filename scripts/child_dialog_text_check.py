"""Check a v2 child-dialogue build against its v1 build.

The builder guarantees the v1 test frames and ids, but the surrounding text
still needs mechanical checks: key words, answer words in replies, rendering
cleanliness, duplicates, split coverage and v1-test leakage.

Usage:
    python scripts/child_dialog_text_check.py --v1 DIR --v2 DIR
"""

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from child_dialog_build import MASS, PLURAL, answer_word, article, plural

BAD_PUNCTUATION = (" ,", " .", " ?", " !", " ;", " :")
REPEATED_PUNCTUATION = re.compile(r"[.!?]{2,}")
PERIOD_BEFORE_LOWERCASE = re.compile(r"\.(?=\s*[a-z])")
PUNCTUATION_COLLISION = re.compile(r"[.!?][,;:]|[,;:][.!?]")
SPACE_BEFORE_PUNCTUATION = re.compile(r"\s[,.;:!?]")
SHAPE_WITHOUT_ARTICLE = re.compile(
    r"\bis\s+(?:triangle|square|rectangle|circle|oval|star|heart|cone|diamond)\b",
    re.IGNORECASE,
)
MASS_WITH_ARTICLE = re.compile(
    r"\b(?:a|an)\s+(?:"
    + "|".join(re.escape(word) for word in sorted(MASS, key=len, reverse=True))
    + r")\b(?!\s+cone\b)",
    re.IGNORECASE,
)


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def load_build(directory: Path) -> tuple[list[dict], list[dict], list[dict]]:
    return (
        load_jsonl(directory / "concepts.jsonl"),
        load_jsonl(directory / "frames.jsonl"),
        load_jsonl(directory / "pairs.jsonl"),
    )


def key_forms(concept: dict) -> set[str]:
    key = concept["key"]
    forms = {key, plural(key)}
    if key not in PLURAL:
        forms.add(f"{article(key)} {key}")
    return {form.lower() for form in forms if form}


def text_problems(text: str, where: str) -> list[str]:
    problems = []
    if not text:
        return [f"{where}: empty text"]
    if text != text.strip():
        problems.append(f"{where}: leading or trailing whitespace: {text!r}")
    if "{" in text or "}" in text:
        problems.append(f"{where}: unrendered braces: {text!r}")
    if "  " in text:
        problems.append(f"{where}: double space: {text!r}")
    for punctuation in BAD_PUNCTUATION:
        if punctuation in text:
            problems.append(f"{where}: space before punctuation: {text!r}")
    if SPACE_BEFORE_PUNCTUATION.search(text):
        problems.append(f"{where}: space before punctuation: {text!r}")
    if REPEATED_PUNCTUATION.search(text):
        problems.append(f"{where}: repeated sentence punctuation: {text!r}")
    if PERIOD_BEFORE_LOWERCASE.search(text):
        problems.append(f"{where}: period before lowercase continuation: {text!r}")
    if PUNCTUATION_COLLISION.search(text):
        problems.append(f"{where}: punctuation collision: {text!r}")
    if SHAPE_WITHOUT_ARTICLE.search(text):
        problems.append(f"{where}: shape word without an article: {text!r}")
    if MASS_WITH_ARTICLE.search(text):
        problems.append(f"{where}: article before a mass noun: {text!r}")
    return problems


def check_general(name: str, concepts: list[dict], pairs: list[dict]) -> list[str]:
    failures = []
    by_concept = {concept["id"]: concept for concept in concepts}
    for concept in concepts:
        word = answer_word(concept["answer"]).lower()
        for reply in concept["replies"]:
            failures.extend(
                text_problems(reply, f"{name} reply {concept['id']} {concept['key']}")
            )
            if word not in reply.lower():
                failures.append(
                    f"{name}: concept reply missing answer word {word!r} "
                    f"for {concept['id']}: {reply!r}"
                )
        failures.extend(
            text_problems(
                concept["answer"], f"{name} answer {concept['id']} {concept['key']}"
            )
        )
    for pair in pairs:
        failures.extend(
            text_problems(
                pair["question"],
                f"{name} question {pair['concept']} frame {pair['frame']}",
            )
        )

    questions: dict[str, set[str]] = defaultdict(set)
    for pair in pairs:
        questions[pair["question"].lower()].add(pair["concept"])
    for question, concept_ids in questions.items():
        if len(concept_ids) > 1:
            failures.append(
                f"{name}: duplicate question across concepts "
                f"{sorted(concept_ids)}: {question!r}"
            )

    for pair in pairs:
        concept = by_concept[pair["concept"]]
        question = pair["question"].lower()
        if concept["family"] == "add":
            missing = [
                token
                for token in concept["key"].split()
                if not re.search(rf"\b{re.escape(token)}\b", question)
            ]
            has_key = not missing
        else:
            has_key = any(form in question for form in key_forms(concept))
        if not has_key:
            failures.append(
                f"{name}: question missing key {concept['key']!r}: "
                f"{pair['question']!r} ({pair['concept']})"
            )
        word = answer_word(concept["answer"]).lower()
        if word not in pair["answer"].lower():
            failures.append(
                f"{name}: answer lost answer word {word!r}: {pair['answer']!r}"
            )
        for reply in pair["replies"]:
            if word not in reply.lower():
                failures.append(
                    f"{name}: reply missing answer word {word!r} "
                    f"for {concept['id']}: {reply!r}"
                )
    return failures


def check_v1_preserved(
    v1_concepts: list[dict],
    v1_pairs: list[dict],
    v2_concepts: list[dict],
    v2_pairs: list[dict],
) -> list[str]:
    failures = []
    v2_ids = {concept["id"] for concept in v2_concepts}
    v1_by_id = {concept["id"]: concept for concept in v1_concepts}
    for concept_id, concept in v1_by_id.items():
        if concept_id not in v2_ids:
            failures.append(f"v2 is missing v1 concept {concept_id} ({concept['key']})")

    v1_test = {
        (pair["concept"], pair["frame"]): pair["question"]
        for pair in v1_pairs
        if pair["split"] == "test"
    }
    v2_test = {
        (pair["concept"], pair["frame"]): pair["question"]
        for pair in v2_pairs
        if pair["split"] == "test"
    }
    for key, question in sorted(v1_test.items()):
        if key not in v2_test:
            failures.append(f"v2 is missing v1 test pair {key}")
        elif v2_test[key] != question:
            failures.append(
                f"v2 changed v1 test question {key}: {question!r} -> {v2_test[key]!r}"
            )
    for key in v2_test:
        if key in v1_test:
            continue
        concept_id, _ = key
        if concept_id in v1_by_id:
            failures.append(f"v2 added a test frame for v1 concept {concept_id}: {key}")

    v1_test_text = {
        pair["question"].lower() for pair in v1_pairs if pair["split"] == "test"
    }
    leaks = [
        pair
        for pair in v2_pairs
        if pair["split"] == "train" and pair["question"].lower() in v1_test_text
    ]
    for pair in leaks[:20]:
        failures.append(
            f"v2 train question equals a v1 test question: "
            f"{pair['question']!r} ({pair['concept']})"
        )
    if len(leaks) > 20:
        failures.append(f"... and {len(leaks) - 20} more v1 test leaks")
    return failures


def check_split_coverage(v2_concepts: list[dict], v2_pairs: list[dict]) -> list[str]:
    failures = []
    train: Counter[str] = Counter()
    test: Counter[str] = Counter()
    for pair in v2_pairs:
        (test if pair["split"] == "test" else train)[pair["concept"]] += 1
    for concept in v2_concepts:
        concept_id = concept["id"]
        if train[concept_id] == 0:
            failures.append(f"v2 concept has no train pair: {concept_id}")
        if test[concept_id] == 0:
            failures.append(f"v2 concept has no test pair: {concept_id}")
    return failures


def print_family_counts(
    title: str, concepts: list[dict], pairs: list[dict], failures: list[str]
) -> None:
    print(f"\n{title} per-family counts")
    print(f"{'family':16} {'concepts':>9} {'train':>8} {'test':>7}")
    train = Counter(pair["family"] for pair in pairs if pair["split"] == "train")
    test = Counter(pair["family"] for pair in pairs if pair["split"] == "test")
    concept_counts = Counter(concept["family"] for concept in concepts)
    for family in sorted(concept_counts):
        print(
            f"{family:16} {concept_counts[family]:9d} "
            f"{train[family]:8d} {test[family]:7d}"
        )
    print(
        f"{'TOTAL':16} {len(concepts):9d} "
        f"{sum(train.values()):8d} {sum(test.values()):7d}"
    )
    if failures:
        print(f"\n{len(failures)} failure(s):")
        for failure in failures:
            print(f"  - {failure}")
    else:
        print("\nOK: all text checks passed.")


def check_dirs(v1_dir: Path, v2_dir: Path) -> list[str]:
    v1_concepts, _, v1_pairs = load_build(v1_dir)
    v2_concepts, _, v2_pairs = load_build(v2_dir)
    failures = []
    failures.extend(check_general("v1", v1_concepts, v1_pairs))
    failures.extend(check_general("v2", v2_concepts, v2_pairs))
    failures.extend(check_v1_preserved(v1_concepts, v1_pairs, v2_concepts, v2_pairs))
    failures.extend(check_split_coverage(v2_concepts, v2_pairs))
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v1", type=Path, required=True, help="v1 build directory")
    parser.add_argument("--v2", type=Path, required=True, help="v2 build directory")
    args = parser.parse_args()
    try:
        v1_concepts, _, v1_pairs = load_build(args.v1)
        v2_concepts, _, v2_pairs = load_build(args.v2)
    except OSError as error:
        print(f"check failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error
    failures = check_dirs(args.v1, args.v2)
    print_family_counts("v1", v1_concepts, v1_pairs, [])
    print_family_counts("v2", v2_concepts, v2_pairs, failures)
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
