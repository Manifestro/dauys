"""Apply the pre-registered Piper scaling decision to child-dialogue results."""

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np

NAME_RE = re.compile(r"^H(\d+(?:\.\d+)?)-s(\d+)$")


def parse_name(name: str) -> tuple[float, int]:
    """Parse a result name such as H200-s1 into hours and seed."""
    match = NAME_RE.fullmatch(name)
    if match is None:
        raise ValueError(f"Invalid result name: {name}")
    return float(match.group(1)), int(match.group(2))


def parse_named(items: list[str]) -> dict[str, Path]:
    """Parse NAME=PATH arguments."""
    result = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Expected NAME=PATH, got {item}")
        name, path = item.split("=", 1)
        if not name or not path:
            raise ValueError(f"Expected NAME=PATH, got {item}")
        if name in result:
            raise ValueError(f"Duplicate result name: {name}")
        result[name] = Path(path)
    return result


def index_items(items: list[dict]) -> tuple[dict[str, bool], float]:
    """Return top1 by item id and the overall fraction."""
    indexed = {}
    for item in items:
        identity = str(item["id"])
        if identity in indexed:
            raise ValueError(f"Duplicate item id: {identity}")
        indexed[identity] = bool(item["top1"])
    if not indexed:
        raise ValueError("A ranking result needs at least one item")
    return indexed, sum(indexed.values()) / len(indexed)


def _mean(values: list[float]) -> float | None:
    return float(sum(values) / len(values)) if values else None


def paired_bootstrap(
    differences: list[float], samples: int = 10_000, seed: int = 0
) -> dict:
    """Bootstrap the mean paired difference over items."""
    if not differences:
        return {"mean": None, "ci_low": None, "ci_high": None, "samples": samples}
    values = np.asarray(differences, dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(samples, len(values)), replace=True).mean(axis=1)
    return {
        "mean": float(draws.mean()),
        "ci_low": float(np.percentile(draws, 2.5)),
        "ci_high": float(np.percentile(draws, 97.5)),
        "samples": samples,
    }


def decide(
    *,
    predicted_2000: float | None,
    ci_low: float | None,
    G: float | None,
    seed_spread_h200: float | None,
    seed_spread_h500: float | None,
) -> tuple[str, list[str]]:
    """Apply the fixed YES/NO rule and list failed conditions."""
    failed = []
    if predicted_2000 is None or predicted_2000 < 0.03:
        failed.append("predicted_2000_below_0.03")
    if ci_low is None or ci_low <= 0:
        failed.append("bootstrap_ci_low_not_positive")
    spread = max(seed_spread_h200 or 0.0, seed_spread_h500 or 0.0)
    if G is None or G <= spread:
        failed.append("gain_not_above_seed_spread")
    return ("YES" if not failed else "NO"), failed


def _shared_ids(
    indexed: dict[str, tuple[dict[str, bool], float]], label: str
) -> set[str] | None:
    all_ids = None
    for by_id, _ in indexed.values():
        ids = set(by_id)
        if all_ids is None:
            all_ids = ids
        elif ids != all_ids:
            raise ValueError(f"All {label} results must use the same item ids")
    return all_ids


def compute_metrics(
    piper: dict[str, list[dict]],
    kokoro: dict[str, list[dict]],
    *,
    samples: int = 10_000,
    seed: int = 0,
) -> dict:
    """Summarize runs and apply the decision rule."""
    if not piper and not kokoro:
        raise ValueError("At least one result is required")
    piper_index = {name: index_items(items) for name, items in piper.items()}
    kokoro_index = {name: index_items(items) for name, items in kokoro.items()}
    piper_ids = _shared_ids(piper_index, "Piper")
    _shared_ids(kokoro_index, "Kokoro")
    names = sorted(
        set(piper) | set(kokoro),
        key=lambda name: parse_name(name),
    )
    runs = {}
    by_hours: dict[float, dict] = {}
    for name in names:
        hours, run_seed = parse_name(name)
        piper_stats = piper_index.get(name)
        kokoro_stats = kokoro_index.get(name)
        runs[name] = {
            "hours": hours,
            "seed": run_seed,
            "piper": (
                None
                if piper_stats is None
                else {
                    "top1": round(piper_stats[1], 4),
                    "items": len(piper_stats[0]),
                }
            ),
            "kokoro": (
                None
                if kokoro_stats is None
                else {
                    "top1": round(kokoro_stats[1], 4),
                    "items": len(kokoro_stats[0]),
                }
            ),
        }
        group = by_hours.setdefault(
            hours, {"piper": [], "kokoro": [], "piper_seeds": [], "kokoro_seeds": []}
        )
        if piper_stats is not None:
            group["piper"].append(piper_stats[1])
            group["piper_seeds"].append(run_seed)
        if kokoro_stats is not None:
            group["kokoro"].append(kokoro_stats[1])
            group["kokoro_seeds"].append(run_seed)
    by_hours_out = {}
    for hours, group in sorted(by_hours.items()):
        by_hours_out[f"{hours:g}"] = {
            "piper": _mean(group["piper"]),
            "kokoro": _mean(group["kokoro"]),
            "piper_seeds": sorted(group["piper_seeds"]),
            "kokoro_seeds": sorted(group["kokoro_seeds"]),
        }

    def spread(hours: float) -> float | None:
        values = {}
        for name, (_, top1) in piper_index.items():
            run_hours, run_seed = parse_name(name)
            if run_hours == hours:
                values[run_seed] = top1
        if 0 not in values or 1 not in values:
            return None
        return abs(values[0] - values[1])

    def item_means(hours: float) -> dict[str, float] | None:
        selected = [
            piper_index[name][0] for name in piper if parse_name(name)[0] == hours
        ]
        if not selected:
            return None
        assert piper_ids is not None
        return {
            identity: sum(index[identity] for index in selected) / len(selected)
            for identity in sorted(piper_ids)
        }

    h200_items = item_means(200)
    h500_items = item_means(500)
    differences = (
        []
        if h200_items is None or h500_items is None
        else [
            h500_items[identity] - h200_items[identity]
            for identity in sorted(piper_ids)
        ]
    )
    bootstrap = paired_bootstrap(differences, samples=samples, seed=seed)
    piper_200 = by_hours_out.get("200", {}).get("piper")
    piper_500 = by_hours_out.get("500", {}).get("piper")
    kokoro_200 = by_hours_out.get("200", {}).get("kokoro")
    kokoro_500 = by_hours_out.get("500", {}).get("kokoro")
    G = None if piper_200 is None or piper_500 is None else piper_500 - piper_200
    g = None if G is None else G / math.log2(500 / 200)
    predicted = None if g is None else g * math.log2(2000 / 500)
    spread_200 = spread(200)
    spread_500 = spread(500)
    decision, failed = decide(
        predicted_2000=predicted,
        ci_low=bootstrap["ci_low"],
        G=G,
        seed_spread_h200=spread_200,
        seed_spread_h500=spread_500,
    )
    kokoro_delta = (
        None if kokoro_200 is None or kokoro_500 is None else kokoro_500 - kokoro_200
    )
    memorization = bool(
        kokoro_delta is not None and kokoro_delta > 0.03 and G is not None and G <= 0.01
    )
    return {
        "runs": runs,
        "by_hours": by_hours_out,
        "seed_spread": {"H200": spread_200, "H500": spread_500},
        "gain": {
            "G": G,
            "per_log2_hour": g,
            "predicted_2000": predicted,
        },
        "bootstrap": bootstrap,
        "decision": decision,
        "conditions_failed": failed,
        "memorization_flag": memorization,
        "kokoro_delta_H500_H200": kokoro_delta,
    }


def _load(path: Path) -> list[dict]:
    data = json.loads(path.read_text())
    items = data["items"]
    if not isinstance(items, list):
        raise TypeError(f"Ranking result has no item list: {path}")
    return items


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--piper", nargs="+", required=True, metavar="NAME=PATH")
    parser.add_argument("--kokoro", nargs="+", required=True, metavar="NAME=PATH")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    piper = {name: _load(path) for name, path in parse_named(args.piper).items()}
    kokoro = {name: _load(path) for name, path in parse_named(args.kokoro).items()}
    result = compute_metrics(piper, kokoro)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    gain = result["gain"]
    bootstrap = result["bootstrap"]
    print(f"Decision: {result['decision']}")
    print(
        "Piper H200={:.4f} H500={:.4f} G={} predicted_2000={}".format(
            result["by_hours"].get("200", {}).get("piper") or 0.0,
            result["by_hours"].get("500", {}).get("piper") or 0.0,
            "null" if gain["G"] is None else f"{gain['G']:.4f}",
            (
                "null"
                if gain["predicted_2000"] is None
                else f"{gain['predicted_2000']:.4f}"
            ),
        )
    )
    print(
        "Bootstrap CI=[{}, {}] seed spread H200={} H500={}".format(
            ("null" if bootstrap["ci_low"] is None else f"{bootstrap['ci_low']:.4f}"),
            ("null" if bootstrap["ci_high"] is None else f"{bootstrap['ci_high']:.4f}"),
            (
                "null"
                if result["seed_spread"]["H200"] is None
                else result["seed_spread"]["H200"]
            ),
            (
                "null"
                if result["seed_spread"]["H500"] is None
                else result["seed_spread"]["H500"]
            ),
        )
    )
    if result["conditions_failed"]:
        print("Failed: " + ", ".join(result["conditions_failed"]))
    if result["memorization_flag"]:
        print("Memorization flag: true")


if __name__ == "__main__":
    main()
