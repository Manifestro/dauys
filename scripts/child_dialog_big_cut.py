"""Cut nested child-dialogue training manifests by cumulative audio hours."""

import argparse
import json
import os
from pathlib import Path

FRAME_RATE = 12.5


def select_rows(rows: list[dict], hours: float) -> list[dict]:
    """Take rows in rank order until their audio reaches ``hours``."""
    if hours < 0:
        raise ValueError("The hour target cannot be negative")
    if hours == 0:
        return []
    target = hours * FRAME_RATE * 3600
    ordered = sorted(
        rows,
        key=lambda row: (
            float(row["rank"]),
            str(row.get("id", "")),
            str(row.get("_source", "")),
        ),
    )
    selected = []
    frames = 0
    for row in ordered:
        selected.append(dict(row))
        frames += int(row["frames"])
        if frames >= target:
            break
    return selected


def select_cuts(
    manifests: list[tuple[Path, list[dict]]],
    hours: float,
    root: Path | None,
) -> list[dict]:
    """Select a cut and rewrite tokens paths next to their source manifests."""
    rows = []
    for manifest, items in manifests:
        for item in items:
            row = dict(item)
            row["_source"] = str(manifest.parent)
            row["_tokens"] = str((manifest.parent / str(item["tokens_path"])).resolve())
            rows.append(row)
    selected = select_rows(rows, hours)
    root_path = None if root is None else Path(root).resolve()
    for row in selected:
        row.pop("_source")
        tokens = row.pop("_tokens")
        row["tokens_path"] = (
            tokens if root_path is None else os.path.relpath(tokens, root_path)
        )
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifests", type=Path, nargs="+", required=True)
    parser.add_argument("--hours", type=float, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--root", type=Path)
    args = parser.parse_args()
    manifests = []
    for path in args.manifests:
        rows = [
            json.loads(line) for line in path.read_text().splitlines() if line.strip()
        ]
        manifests.append((path, rows))
    args.output.mkdir(parents=True, exist_ok=True)
    for hours in args.hours:
        rows = select_cuts(manifests, hours, args.root)
        path = args.output / f"train-{hours:g}h.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        print(
            json.dumps(
                {
                    "hours": hours,
                    "rows": len(rows),
                    "frames": sum(int(row["frames"]) for row in rows),
                    "manifest": str(path),
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
