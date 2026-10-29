"""Nested rank-order cuts for child-dialogue training manifests."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from child_dialog_big_cut import select_cuts, select_rows

HOUR = 45_000


def test_cut_selection_is_nested_and_rank_ordered():
    rows = [
        {"id": "r3", "rank": 0.3, "frames": HOUR},
        {"id": "r0", "rank": 0.0, "frames": HOUR},
        {"id": "r4", "rank": 0.4, "frames": HOUR},
        {"id": "r1", "rank": 0.1, "frames": HOUR},
        {"id": "r2", "rank": 0.2, "frames": HOUR},
    ]
    small, medium, large = [select_rows(rows, hours) for hours in (1, 3, 5)]
    assert [[row["id"] for row in cut] for cut in (small, medium, large)] == [
        ["r0"],
        ["r0", "r1", "r2"],
        ["r0", "r1", "r2", "r3", "r4"],
    ]
    assert small == medium[: len(small)]
    assert medium == large[: len(medium)]


def test_select_cuts_rewrites_tokens_next_to_each_manifest(tmp_path):
    manifest = tmp_path / "shard-0" / "enc" / "train" / "manifest.jsonl"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        json.dumps(
            {
                "id": "r0",
                "rank": 0.0,
                "frames": HOUR,
                "tokens_path": "tokens/r0.pt",
            }
        )
        + "\n"
    )
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    selected = select_cuts([(manifest, rows)], 1, tmp_path)
    assert selected[0]["tokens_path"] == "shard-0/enc/train/tokens/r0.pt"


def test_select_cuts_defaults_to_base_speaker_and_keeps_the_spec(tmp_path):
    manifest = tmp_path / "train" / "manifest.jsonl"
    manifest.parent.mkdir(parents=True)
    rows = [
        {
            "id": "b",
            "rank": 0.1,
            "frames": HOUR,
            "voice": "blend:am_echo:am_adam:0.47",
            "speaker": "blend:am_echo:am_adam:0.47",
            "tokens_path": "tokens/b.pt",
        },
        {
            "id": "a",
            "rank": 0.0,
            "frames": HOUR,
            "voice": "af_sky",
            "speaker": "af_sky",
            "tokens_path": "tokens/a.pt",
        },
    ]
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
    base = select_cuts([(manifest, rows)], 2, None)
    assert [(row["id"], row["speaker"]) for row in base] == [
        ("a", "af_sky"),
        ("b", "am_echo"),
    ]
    assert base[1]["voice"] == "blend:am_echo:am_adam:0.47"
    spec = select_cuts([(manifest, rows)], 2, None, speaker_key="spec")
    assert [row["speaker"] for row in spec] == [
        "af_sky",
        "blend:am_echo:am_adam:0.47",
    ]


def test_select_cuts_accepts_two_id_tagged_pools_together(tmp_path):
    definitions = [
        ("pool1", "p-r000", 0.1, "tokens/a.pt"),
        ("pool2", "p-pool2-r000", 0.2, "tokens/b.pt"),
    ]
    manifests = []
    for pool, identifier, rank, tokens in definitions:
        manifest = tmp_path / pool / "enc" / "train" / "manifest.jsonl"
        manifest.parent.mkdir(parents=True)
        row = {
            "id": identifier,
            "rank": rank,
            "frames": HOUR,
            "tokens_path": tokens,
        }
        manifest.write_text(json.dumps(row) + "\n")
        manifests.append((manifest, [row]))
    selected = select_cuts(manifests, 2, tmp_path)
    assert [row["id"] for row in selected] == ["p-r000", "p-pool2-r000"]
    assert len({row["id"] for row in selected}) == len(selected)
    assert [row["tokens_path"] for row in selected] == [
        "pool1/enc/train/tokens/a.pt",
        "pool2/enc/train/tokens/b.pt",
    ]
