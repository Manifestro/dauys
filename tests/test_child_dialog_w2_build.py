"""Checks for the deterministic W2 order/negation text builder."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from child_dialog_w2_build import TIERS, build


def _tier(name: str) -> int:
    return next(tier for tier, names in TIERS.items() if name in names)


def test_elephant_mouse_bigger_smaller():
    concepts, _, _ = build()
    by_id = {concept["id"]: concept for concept in concepts}
    bigger = by_id["order-elephant-mouse-bigger"]
    smaller = by_id["order-elephant-mouse-smaller"]
    assert bigger["answer"] == "Yes."
    assert bigger["replies"] == ["Yes.", "Yes, it is."]
    assert smaller["answer"] == "No."
    assert smaller["replies"] == ["No.", "No, it is not."]


def test_swapping_x_and_y_flips_the_answer():
    concepts, _, _ = build()
    by_id = {concept["id"]: concept for concept in concepts}
    for concept in concepts:
        if concept["family"] != "order":
            continue
        x, y, adj = concept["key"].split()
        other = by_id[f"order-{y}-{x}-{adj}"]
        assert other["answer"] != concept["answer"]


def test_pairs_with_less_than_two_tiers_apart_are_excluded():
    concepts, _, _ = build()
    identifiers = {concept["id"] for concept in concepts}
    assert "order-cat-dog-bigger" not in identifiers
    assert "order-mouse-frog-smaller" not in identifiers
    assert "order-cat-horse-bigger" not in identifiers
    for concept in concepts:
        if concept["family"] != "order":
            continue
        x, y, _ = concept["key"].split()
        assert abs(_tier(x) - _tier(y)) >= 2


def test_want_replies_keep_number_and_plural_forms():
    concepts, _, _ = build()
    by_id = {concept["id"]: concept for concept in concepts}
    apple = by_id["want-apple-neg"]
    crayons = by_id["want-crayons-pos"]
    assert apple["answer"] == "Okay, no apple."
    assert apple["replies"] == ["Okay, no apple.", "Okay, no apple for now."]
    assert crayons["answer"] == "Here are the crayons."
    assert crayons["replies"] == [
        "Here are the crayons.",
        "Okay, here are the crayons.",
    ]


def test_split_is_frame_based_unique_and_minimal_pairs_agree():
    concepts, frames, pairs = build()
    split = {(frame["group"], frame["index"]): frame["split"] for frame in frames}
    test_frames = {
        group: {
            index
            for (name, index), value in split.items()
            if name == group and value == "test"
        }
        for group in ("order", "want")
    }
    assert all(len(indexes) == 3 for indexes in test_frames.values())
    by_id = {concept["id"]: concept for concept in concepts}
    by_pair_id = {pair["id"]: pair for pair in pairs}
    assert len(by_pair_id) == len(pairs)
    for pair in pairs:
        assert pair["split"] == split[(pair["group"], pair["frame"])]
    for pair in pairs:
        if pair["split"] != "test":
            continue
        concept = by_id[pair["concept"]]
        if concept["family"] == "order":
            x, y, adj = concept["key"].split()
            other_id = f"order-{y}-{x}-{adj}-f{pair['frame']:02d}"
        else:
            item, polarity = concept["key"].rsplit(" ", 1)
            other = "neg" if polarity == "pos" else "pos"
            other_id = f"want-{item.replace(' ', '-')}-{other}-f{pair['frame']:02d}"
        other = by_pair_id[other_id]
        assert other["split"] == "test"
        assert by_id[other["concept"]]["answer"] != concept["answer"]
