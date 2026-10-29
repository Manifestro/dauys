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
