"""Freeze nested dialogue selections before multi-excerpt training."""

import argparse
import json
from pathlib import Path

from dauys.research_split import stage1_selections


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("runs/stage1-splits.json"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    inventory = json.loads(args.inventory.read_text(encoding="utf-8"))
    result = stage1_selections(inventory, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for experiment in ("e004", "e005"):
        print(experiment)
        for split, rows in result[experiment].items():
            frames = sum(row["frames"] for row in rows)
            print(
                f"  {split}: {len(rows)} conversations, {frames / 12.5 / 60:.3f} minutes"
            )


if __name__ == "__main__":
    main()
