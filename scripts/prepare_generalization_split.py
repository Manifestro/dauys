"""Freeze whole-dialogue, topic-disjoint generalization selections."""

import argparse
import json
from pathlib import Path

from dauys.research_split import (
    scenario_family,
    whole_dialogue_generalization_split,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prior-train-selection", type=Path, required=True)
    parser.add_argument(
        "--prior-probe-scenario", default="simple_dialogue/a_weekend_farmers_market"
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    inventory = json.loads(args.inventory.read_text(encoding="utf-8"))
    prior_rows = json.loads(args.prior_train_selection.read_text(encoding="utf-8"))
    prior = {scenario_family(row["scenario"]) for row in prior_rows}
    prior.add(scenario_family(args.prior_probe_scenario))
    result = whole_dialogue_generalization_split(inventory, args.seed, prior)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for name, rows in result["selection"].items():
        seconds = sum(row["source_samples"] for row in rows) / 24_000
        print(f"{name}: {len(rows)} complete conversations, {seconds / 60:.3f} minutes")


if __name__ == "__main__":
    main()
