"""Build the deterministic child-dialogue W2 set: order and negation.

Every sentence is assembled from the hand-written lists below; no text is
generated. ``order`` compares animals whose sizes differ by at least two tiers
and swaps X/Y for the minimal pair. ``want`` pairs positive and negative
requests for the same item.

Splits: three frame indices per group are held out with a seeded sample, and a
pair takes its frame's split. Outputs ``concepts.jsonl``, ``frames.jsonl``,
``pairs.jsonl`` and ``summary.json`` in the same layout as
``scripts/child_dialog_build.py``.
"""

import argparse
import json
import random
import re
from pathlib import Path

TIERS = {
    1: ("ant", "bee", "ladybug"),
    2: ("mouse", "frog", "bird"),
    3: ("cat", "rabbit", "duck", "dog"),
    4: ("horse", "cow", "bear", "lion"),
    5: ("elephant", "whale", "giraffe"),
}
ANIMALS = tuple((animal, tier) for tier, animals in TIERS.items() for animal in animals)
ORDER_FRAMES = (
    "Is {x} {adj} than {y}?",
    "Mommy, is {x} {adj} than {y}?",
    "Daddy, is {x} {adj} than {y}?",
    "Tell me, is {x} {adj} than {y}?",
    "Is {x} really {adj} than {y}?",
    "Do you think {x} is {adj} than {y}?",
    "Mom, is {x} {adj} than {y}?",
    "Hey, is {x} {adj} than {y}?",
    "I wonder, is {x} {adj} than {y}?",
    "Is {x} {adj} than {y}, mommy?",
)
WANT_ITEMS = (
    "apple",
    "banana",
    "cookie",
    "orange",
    "sandwich",
    "ball",
    "book",
    "doll",
    "blanket",
    "teddy bear",
    "toy car",
    "juice",
    "milk",
    "water",
    "crayons",
    "grapes",
)
PLURAL_ITEMS = {"crayons", "grapes"}
WANT_FRAMES = (
    ("I want the {item}.", "I don't want the {item}."),
    ("Mommy, I want the {item}.", "Mommy, I don't want the {item}."),
    ("Daddy, I want the {item}.", "Daddy, I don't want the {item}."),
    ("I want the {item} now.", "I don't want the {item} now."),
    ("I really want the {item}.", "I really don't want the {item}."),
    ("I need the {item}.", "I don't need the {item}."),
    ("Mom, I want the {item}.", "Mom, I don't want the {item}."),
    ("I want to have the {item}.", "I don't want to have the {item}."),
    ("I would like the {item}.", "I would not like the {item}."),
    ("Please, I want the {item}.", "Please, I don't want the {item}."),
)
HELD_OUT = 3


def article(word: str) -> str:
    return "an" if word[0].lower() in "aeiou" else "a"


def render(template: str, **values) -> str:
    text = template.format(**values)
    text = re.sub(r"\s+", " ", text).strip().replace(" ,", ",")
    return text[:1].upper() + text[1:]


def _want_id(item: str, polarity: str) -> str:
    return f"want-{item.replace(' ', '-')}-{polarity}"


def _want_replies(item: str, polarity: str) -> list[str]:
    if polarity == "neg":
        return [f"Okay, no {item}.", f"Okay, no {item} for now."]
    be = "are" if item in PLURAL_ITEMS else "is"
    return [
        f"Here {be} the {item}.",
        f"Okay, here {be} the {item}.",
    ]


def _order_concepts() -> list[dict]:
    concepts = []
    for x, tier_x in ANIMALS:
        for y, tier_y in ANIMALS:
            if abs(tier_x - tier_y) < 2:
                continue
            for adj in ("bigger", "smaller"):
                yes = (adj == "bigger" and tier_x > tier_y) or (
                    adj == "smaller" and tier_x < tier_y
                )
                answer = "Yes." if yes else "No."
                replies = ["Yes.", "Yes, it is."] if yes else ["No.", "No, it is not."]
                concepts.append(
                    {
                        "id": f"order-{x}-{y}-{adj}",
                        "family": "order",
                        "group": "order",
                        "key": f"{x} {y} {adj}",
                        "answer": answer,
                        "replies": replies,
                    }
                )
    return concepts


def _want_concepts() -> list[dict]:
    concepts = []
    for item in WANT_ITEMS:
        for polarity in ("pos", "neg"):
            replies = _want_replies(item, polarity)
            concepts.append(
                {
                    "id": _want_id(item, polarity),
                    "family": "want",
                    "group": "want",
                    "key": f"{item} {polarity}",
                    "answer": replies[0],
                    "replies": replies,
                }
            )
    return concepts


def frame_splits(seed: int = 0) -> dict[str, set[int]]:
    rng = random.Random(seed)
    return {
        group: set(rng.sample(range(length), HELD_OUT))
        for group, length in (("order", len(ORDER_FRAMES)), ("want", len(WANT_FRAMES)))
    }


def _frames(seed: int) -> list[dict]:
    held = frame_splits(seed)
    frames = [
        {
            "group": "order",
            "index": index,
            "template": template,
            "split": "test" if index in held["order"] else "train",
        }
        for index, template in enumerate(ORDER_FRAMES)
    ]
    frames.extend(
        {
            "group": "want",
            "index": index,
            "template": pos,
            "pos": pos,
            "neg": neg,
            "split": "test" if index in held["want"] else "train",
        }
        for index, (pos, neg) in enumerate(WANT_FRAMES)
    )
    return frames


def _pairs(concepts: list[dict], frames: list[dict]) -> list[dict]:
    frame_split = {(frame["group"], frame["index"]): frame["split"] for frame in frames}
    pairs = []
    for concept in concepts:
        for index in range(10):
            if concept["family"] == "order":
                x, y, adj = concept["key"].split()
                question = render(
                    ORDER_FRAMES[index],
                    x=f"{article(x)} {x}",
                    y=f"{article(y)} {y}",
                    adj=adj,
                )
            else:
                item, polarity = concept["key"].rsplit(" ", 1)
                pos, neg = WANT_FRAMES[index]
                question = render(pos if polarity == "pos" else neg, item=item)
            pairs.append(
                {
                    "id": f"{concept['id']}-f{index:02d}",
                    "question": question,
                    "replies": concept["replies"],
                    "concept": concept["id"],
                    "family": concept["family"],
                    "group": concept["group"],
                    "key": concept["key"],
                    "frame": index,
                    "split": frame_split[(concept["group"], index)],
                }
            )
    return pairs


def _minimal_pair_id(concept: dict, frame: int) -> str:
    if concept["family"] == "order":
        x, y, adj = concept["key"].split()
        other = f"order-{y}-{x}-{adj}"
    else:
        item, polarity = concept["key"].rsplit(" ", 1)
        other = _want_id(item, "neg" if polarity == "pos" else "pos")
    return f"{other}-f{frame:02d}"


def check(concepts: list[dict], pairs: list[dict]) -> None:
    questions = [pair["question"] for pair in pairs]
    if len(questions) != len(set(questions)):
        raise ValueError("Two pairs share a question")
    for question in questions:
        if "  " in question or " ," in question:
            raise ValueError(f"Malformed question: {question!r}")
    by_id = {concept["id"]: concept for concept in concepts}
    pairs_by_id = {pair["id"]: pair for pair in pairs}
    for concept in concepts:
        if concept["family"] != "order":
            continue
        x, y, _ = concept["key"].split()
        wanted = (f"{article(x)} {x}", f"{article(y)} {y}")
        for pair in pairs:
            if pair["concept"] == concept["id"] and not all(
                phrase in pair["question"] for phrase in wanted
            ):
                raise ValueError(f"Wrong article in {pair['question']!r}")
    for pair in pairs:
        if pair["split"] != "test":
            continue
        other_id = _minimal_pair_id(by_id[pair["concept"]], pair["frame"])
        other = pairs_by_id.get(other_id)
        if other is None or other["split"] != "test":
            raise ValueError(f"Test pair {pair['id']} lacks a test minimal pair")
        if by_id[other["concept"]]["answer"] == by_id[pair["concept"]]["answer"]:
            raise ValueError(f"Minimal pair {pair['id']} does not flip the answer")


def build(seed: int = 0) -> tuple[list[dict], list[dict], list[dict]]:
    concepts = _order_concepts() + _want_concepts()
    concepts.sort(key=lambda concept: concept["id"])
    frames = _frames(seed)
    pairs = _pairs(concepts, frames)
    check(concepts, pairs)
    return concepts, frames, pairs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    concepts, frames, pairs = build(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    for name, rows in (("concepts", concepts), ("frames", frames), ("pairs", pairs)):
        (args.output / f"{name}.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows)
        )
    summary = {
        "concepts": len(concepts),
        "frames": len(frames),
        "pairs": len(pairs),
        "test_pairs": sum(pair["split"] == "test" for pair in pairs),
        "by_family": {
            family: sum(concept["family"] == family for concept in concepts)
            for family in sorted({concept["family"] for concept in concepts})
        },
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
