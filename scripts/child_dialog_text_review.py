"""Write a human review file for the v2 child-dialogue text set.

Usage:
    python scripts/child_dialog_text_review.py --v1 DIR --v2 DIR \
        --output runs/text-v2-build/review.md
"""

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def pair_line(pair: dict) -> str:
    replies = " / ".join(pair["replies"])
    return (
        f'- **Q:** "{pair["question"]}" **A:** "{pair["answer"]}" '
        f'**replies:** "{replies}"'
    )


def render_review(
    v1_dir: Path,
    v2_dir: Path,
    output: Path,
    seed: int = 0,
    train_examples: int = 12,
    test_examples: int = 3,
) -> dict:
    v1_concepts = load_jsonl(v1_dir / "concepts.jsonl")
    v2_concepts = load_jsonl(v2_dir / "concepts.jsonl")
    v2_frames = load_jsonl(v2_dir / "frames.jsonl")
    v2_pairs = load_jsonl(v2_dir / "pairs.jsonl")
    v1_ids = {concept["id"] for concept in v1_concepts}
    new_concepts = [concept for concept in v2_concepts if concept["id"] not in v1_ids]

    by_family: dict[str, list[dict]] = defaultdict(list)
    for concept in new_concepts:
        by_family[concept["family"]].append(concept)

    train_by_family: dict[str, list[dict]] = defaultdict(list)
    test_by_family: dict[str, list[dict]] = defaultdict(list)
    for pair in v2_pairs:
        (test_by_family if pair["split"] == "test" else train_by_family)[
            pair["family"]
        ].append(pair)

    lines = ["# Child-dialogue text v2 review", ""]
    train_questions = {
        pair["question"] for pair in v2_pairs if pair["split"] == "train"
    }
    unique_replies = {reply for concept in v2_concepts for reply in concept["replies"]}
    v1_test = sum(
        pair["split"] == "test" and pair["concept"] in v1_ids for pair in v2_pairs
    )
    new_test = sum(
        pair["split"] == "test" and pair["concept"] not in v1_ids for pair in v2_pairs
    )
    lines.extend(
        [
            "## Totals",
            "",
            f"- Concepts: {len(v2_concepts)}",
            f"- Frames: {len(v2_frames)}",
            f"- Pairs: {len(v2_pairs)}",
            f"- Unique train questions: {len(train_questions)}",
            f"- Unique replies: {len(unique_replies)}",
            f"- Test pairs, v1 part / new part: {v1_test} / {new_test}",
            f"- New concepts: {len(new_concepts)}",
            "",
        ]
    )

    for family in sorted(by_family):
        family_new = sorted(by_family[family], key=lambda row: row["id"])
        rng = random.Random(seed)
        train_pool = train_by_family[family]
        test_pool = test_by_family[family]
        train = (
            rng.sample(train_pool, min(train_examples, len(train_pool)))
            if train_pool
            else []
        )
        test = (
            rng.sample(test_pool, min(test_examples, len(test_pool)))
            if test_pool
            else []
        )
        lines.extend(
            [
                f"## {family} ({len(family_new)} new concepts)",
                "",
                "### New concepts (key -> answer)",
                "",
            ]
        )
        lines.extend(
            f"- {concept['key']} -> {concept['answer']}" for concept in family_new
        )
        lines.extend(["", f"### Random train pairs ({len(train)})", ""])
        lines.extend(pair_line(pair) for pair in train)
        lines.extend(["", f"### Random test pairs ({len(test)})", ""])
        lines.extend(pair_line(pair) for pair in test)
        lines.append("")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines).rstrip() + "\n")
    return {
        "concepts": len(v2_concepts),
        "frames": len(v2_frames),
        "pairs": len(v2_pairs),
        "train_questions": len(train_questions),
        "unique_replies": len(unique_replies),
        "test_v1": v1_test,
        "test_new": new_test,
        "new_concepts": len(new_concepts),
        "by_family": Counter(concept["family"] for concept in new_concepts),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v1", type=Path, required=True, help="v1 build directory")
    parser.add_argument("--v2", type=Path, required=True, help="v2 build directory")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/text-v2-build/review.md"),
        help="markdown output path",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    totals = render_review(args.v1, args.v2, args.output, args.seed)
    print(json.dumps({**totals, "by_family": dict(totals["by_family"])}))


if __name__ == "__main__":
    main()
