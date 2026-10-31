"""Score one V10 step 1 arm on the listening bench cache.

    python scripts/v10_step1_eval.py score --config C.yaml --checkpoint X.pt \
        --arm B --seed 0 --cache CACHE --out RESULTS
    python scripts/v10_step1_eval.py combine --results RESULTS

Stimuli, pairing and the summed frames are the listening bench's
(``bench/duplexbench/run.py: single_stream``): the cache holds one entry per
pair with ``id``, ``kind``, ``good`` and ``bad`` [8, T] Mimi codes; the model's
own channel is CB0 (row 0), behind 0.5 s (6 frames) of the cache's silence
frame, and only the item's own frames are summed. The primary score is mean
pairwise accuracy in points; the mean-normalized score divides each log
likelihood by the item's frame count.
"""

import argparse
import json
from pathlib import Path

import torch

from dauys.v10_step1 import ARMS, V10Step1Config, V10Step1Model
from dauys.v10_step1_eval import (
    LEAD_FRAMES,
    accuracy,
    exact_log_ps_pad,
    joint_log_pms,
    nll_s_given_true_m,
    normalized_scores,
    pair_marginal_log_ps,
    pair_outcomes,
    stability,
)

TESTS = ("swuggy", "sblimp", "tstorycloze")
DEFAULT_PARTICLES = {"swuggy": 16, "sblimp": 16, "tstorycloze": 64}
RESAMPLED = frozenset({"tstorycloze"})
FLIP_LIMIT = 0.03
SEEDS = 3


def _load(cache: Path, name: str):
    path = cache / f"{name}.pt"
    if not path.is_file():
        return None
    return torch.load(path, map_location="cpu", weights_only=True)


def _pad(tokens: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    lengths = torch.tensor([value.shape[-1] for value in tokens], dtype=torch.long)
    padded = torch.zeros((len(tokens), int(lengths.max())), dtype=torch.long)
    for row, value in enumerate(tokens):
        padded[row, : value.shape[-1]] = value
    return padded, lengths


def _tokens(cb0: torch.Tensor, lead: int) -> torch.Tensor:
    prefix = torch.full((LEAD_FRAMES,), int(lead), dtype=torch.long)
    return torch.cat((prefix, cb0.long().flatten()))


def _scores(
    model: V10Step1Model,
    pairs: list[dict],
    lead: int,
    *,
    k: int,
    seed: int,
    resample: bool,
    exact: bool,
    batch: int,
) -> tuple[list[float], list[float]]:
    """Good and bad log p(s) per pair, the good side first with the same draws."""
    order = sorted(
        range(len(pairs)),
        key=lambda index: max(
            pairs[index]["good"].shape[-1], pairs[index]["bad"].shape[-1]
        ),
    )
    good_scores = [0.0] * len(pairs)
    bad_scores = [0.0] * len(pairs)
    device = next(model.parameters()).device
    for begin in range(0, len(order), batch):
        block = [pairs[index] for index in order[begin : begin + batch]]
        good, good_lengths = _pad([_tokens(pair["good"][0], lead) for pair in block])
        bad, bad_lengths = _pad([_tokens(pair["bad"][0], lead) for pair in block])
        good, bad = good.to(device), bad.to(device)
        if exact:
            scored_good = exact_log_ps_pad(model, good, good_lengths, LEAD_FRAMES)
            scored_bad = exact_log_ps_pad(model, bad, bad_lengths, LEAD_FRAMES)
        else:
            scored_good, scored_bad = pair_marginal_log_ps(
                model,
                good,
                bad,
                good_lengths,
                bad_lengths,
                k=k,
                seed=seed + begin,
                resample=resample,
                score_from=LEAD_FRAMES,
            )
        for row, index in enumerate(order[begin : begin + batch]):
            good_scores[index] = float(scored_good[row])
            bad_scores[index] = float(scored_bad[row])
    return good_scores, bad_scores


def _accuracies(
    good: list[float],
    bad: list[float],
    pairs: list[dict],
) -> tuple[float, float, tuple[float, ...]]:
    outcomes = pair_outcomes(good, bad)
    good_lengths = [int(pair["good"].shape[-1]) for pair in pairs]
    bad_lengths = [int(pair["bad"].shape[-1]) for pair in pairs]
    normalized = pair_outcomes(
        normalized_scores(good, good_lengths), normalized_scores(bad, bad_lengths)
    )
    return accuracy(outcomes), accuracy(normalized), outcomes


def _score_test(
    model: V10Step1Model,
    pairs: list[dict],
    lead: int,
    *,
    arm: str,
    k: int,
    seed: int,
    resample: bool,
    batch: int,
    max_doublings: int,
) -> dict:
    exact = arm == "A"

    def run(run_seed: int, particles: int):
        good, bad = _scores(
            model,
            pairs,
            lead,
            k=particles,
            seed=run_seed,
            resample=resample,
            exact=exact,
            batch=batch,
        )
        return _accuracies(good, bad, pairs)

    first, normalized, outcomes = run(seed, k)
    if exact:
        second, flips = first, 0.0
    else:
        second, _, outcomes_two = run(seed + 1, k)
        flips = stability(outcomes, outcomes_two)
    doublings = 0
    while flips > FLIP_LIMIT and doublings < max_doublings:
        k *= 2
        doublings += 1
        first, normalized, outcomes = run(seed, k)
        second, _, outcomes_two = run(seed + 1, k)
        flips = stability(outcomes, outcomes_two)
    return {
        "pairs": len(pairs),
        "k": k,
        "resample": resample,
        "accuracy_sum": first,
        "accuracy_mean": normalized,
        "accuracy_sum_seed_two": second,
        "stability_flips": flips,
        "doublings": doublings,
    }


def _sidecar(cache: Path, name: str, arm: str, limit: int):
    """The arm's ZeroSyl labels for the stimuli, if a sidecar was built for the cache."""
    if arm == "A":  # m is always PAD: nothing to hint, and p(s) is already exact
        return None
    payload = _load(cache, f"{name}_m_{arm}")
    if payload is None:
        return None
    return payload[:limit] if limit else payload


def _monologue_tokens(
    tokens: torch.Tensor, label: torch.Tensor, config: V10Step1Config
) -> torch.Tensor:
    frames = int(tokens.shape[-1])
    if label.numel() != frames:
        raise ValueError("The monologue sidecar does not match the item length")
    prefix = torch.full((LEAD_FRAMES,), config.pad_id, dtype=torch.long)
    return torch.cat((prefix, label.long()))


def _hint_metrics(
    model: V10Step1Model,
    pairs: list[dict],
    labels: list[dict],
    config: V10Step1Config,
    *,
    lead: int,
    batch: int,
) -> dict:
    """Joint p(m, s) with the stimulus monologue, beside the marginal p(s)."""
    if len(labels) != len(pairs):
        raise ValueError("The monologue sidecar does not match the pair list")
    outcomes: list[float] = []
    nll: list[float] = []
    device = next(model.parameters()).device
    for begin in range(0, len(pairs), batch):
        block = pairs[begin : begin + batch]
        label_block = labels[begin : begin + batch]
        good_m, good_lengths = _pad(
            [
                _monologue_tokens(pair["good"][0], label["good"], config)
                for pair, label in zip(block, label_block)
            ]
        )
        bad_m, bad_lengths = _pad(
            [
                _monologue_tokens(pair["bad"][0], label["bad"], config)
                for pair, label in zip(block, label_block)
            ]
        )
        good_s, good_s_lengths = _pad(
            [_tokens(pair["good"][0], lead) for pair in block]
        )
        bad_s, bad_s_lengths = _pad([_tokens(pair["bad"][0], lead) for pair in block])
        if not (
            torch.equal(good_lengths, good_s_lengths)
            and torch.equal(bad_lengths, bad_s_lengths)
        ):
            raise ValueError("The monologue sidecar does not match the CB0 frames")
        good_m, good_s = good_m.to(device), good_s.to(device)
        bad_m, bad_s = bad_m.to(device), bad_s.to(device)
        good_scores = joint_log_pms(model, good_m, good_s, good_lengths, LEAD_FRAMES)
        bad_scores = joint_log_pms(model, bad_m, bad_s, bad_lengths, LEAD_FRAMES)
        outcomes.extend(pair_outcomes(good_scores.tolist(), bad_scores.tolist()))
        nll.extend(
            nll_s_given_true_m(
                model, good_m, good_s, good_lengths, LEAD_FRAMES
            ).tolist()
        )
        nll.extend(
            nll_s_given_true_m(model, bad_m, bad_s, bad_lengths, LEAD_FRAMES).tolist()
        )
    return {
        "available": True,
        "pairs": len(outcomes),
        "accuracy_sum": accuracy(tuple(outcomes)),
        "nll_s_given_true_m": sum(nll) / len(nll),
    }


def score(args) -> None:
    config = V10Step1Config.from_yaml(args.config)
    model = V10Step1Model(config)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if V10Step1Config(**payload["config"]) != config:
        raise ValueError("The checkpoint config does not match --config")
    if (payload["arm"], payload["seed"]) != (args.arm, args.seed):
        raise ValueError("The checkpoint belongs to another arm or seed")
    model.load_state_dict(payload["model"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    silence = _load(args.cache, "silence")
    if silence is None:
        raise ValueError("The cache has no silence.pt")
    lead = int(silence[0])
    result: dict = {
        "config": config.name,
        "arm": args.arm,
        "seed": args.seed,
        "checkpoint": str(args.checkpoint),
        "trainable_parameters": model.trainable_parameters(),
        "cache": str(args.cache),
        "estimator": "exact (m = PAD)" if args.arm == "A" else "marginal p(s)",
        "tests": {},
    }
    for name in TESTS:
        pairs = _load(args.cache, name)
        if pairs is None:
            continue
        if args.limit:
            pairs = pairs[: args.limit]
        if not pairs:
            continue
        k = args.particles or DEFAULT_PARTICLES[name]
        metrics = _score_test(
            model,
            pairs,
            lead,
            arm=args.arm,
            k=k,
            seed=args.seed * 1_000_003 + TESTS.index(name),
            resample=name in RESAMPLED and args.arm != "A",
            batch=args.batch,
            max_doublings=args.max_doublings,
        )
        labels = _sidecar(args.cache, name, args.arm, args.limit)
        if labels is not None:
            hint = _hint_metrics(
                model, pairs, labels, config, lead=lead, batch=args.batch
            )
            hint["marginal_accuracy_sum"] = metrics["accuracy_sum"]
            hint["difference"] = hint["accuracy_sum"] - metrics["accuracy_sum"]
            metrics["hint_at_test"] = hint
        else:
            metrics["hint_at_test"] = {
                "available": False,
                "note": "arm A has no labels"
                if args.arm == "A"
                else f"needs cache/{name}_m_{args.arm}.pt (v10_step1_labels.py)",
            }
        result["tests"][name] = metrics
    result["primary_points"] = (
        100.0
        * sum(test["accuracy_sum"] for test in result["tests"].values())
        / len(result["tests"])
        if result["tests"]
        else None
    )
    result["mean_normalized_points"] = (
        100.0
        * sum(test["accuracy_mean"] for test in result["tests"].values())
        / len(result["tests"])
        if result["tests"]
        else None
    )
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"{args.arm}-s{args.seed}.json"
    path.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "written": str(path),
                "primary_points": result["primary_points"],
                "mean_normalized_points": result["mean_normalized_points"],
            }
        ),
        flush=True,
    )


def _spread(values: list[float]) -> float:
    return max(values) - min(values) if values else float("nan")


def decision_rule(points: dict[str, list[float]]) -> dict:
    """Apply the pre-registered arithmetic on three seeds per arm.

    B against B-sil decides on units; B-sil against A asks whether pauses help.
    """
    result: dict = {}
    for front, back in (("B", "B-sil"), ("B-sil", "A")):
        if len(points.get(front, [])) != SEEDS or len(points.get(back, [])) != SEEDS:
            continue
        delta = sum(points[front]) / len(points[front]) - sum(points[back]) / len(
            points[back]
        )
        spread = max(_spread(points[front]), _spread(points[back]))
        result[f"{front}_vs_{back}"] = {
            "delta": delta,
            "spread": spread,
            "yes": delta >= 1.0 and delta > spread,
        }
    return result


def combine(args) -> None:
    per_arm: dict[str, list[tuple[int, float]]] = {}
    for path in sorted(args.results.glob("*.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        if row.get("primary_points") is None:
            continue
        per_arm.setdefault(row["arm"], []).append(
            (int(row["seed"]), float(row["primary_points"]))
        )
    points = {
        arm: [value for _, value in sorted(rows)] for arm, rows in per_arm.items()
    }
    summary = {
        "primary_points": points,
        "seeds": {
            arm: [seed for seed, _ in sorted(rows)] for arm, rows in per_arm.items()
        },
        "gate_medium": None,
        "decision": decision_rule(points),
    }
    if len(points.get("B", [])) == 1 and len(points.get("B-sil", [])) == 1:
        delta = points["B"][0] - points["B-sil"][0]
        summary["gate_medium"] = {"B_minus_B_sil": delta, "large_starts": delta >= 0}
    text = json.dumps(summary, indent=1) + "\n"
    if args.out is not None:
        args.out.write_text(text, encoding="utf-8")
    print(text, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("score")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--checkpoint", type=Path, required=True)
    run.add_argument("--arm", choices=ARMS, required=True)
    run.add_argument("--seed", type=int, required=True)
    run.add_argument("--cache", type=Path, required=True)
    run.add_argument("--out", type=Path, required=True)
    run.add_argument("--particles", type=int, default=0)
    run.add_argument("--batch", type=int, default=16)
    run.add_argument("--limit", type=int, default=0)
    run.add_argument("--max-doublings", type=int, default=3)
    combine_parser = sub.add_parser("combine")
    combine_parser.add_argument("--results", type=Path, required=True)
    combine_parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.command == "score":
        if args.batch < 1 or args.max_doublings < 0 or args.particles < 0:
            raise ValueError("Batch must be positive and counts non-negative")
        score(args)
    else:
        combine(args)


if __name__ == "__main__":
    main()
