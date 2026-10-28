"""Where does the ear's training signal come from in Stage 1?

For each checkpoint, several same-speaker batches go through the Stage 1
objective with the end-of-training weights (``stage1_weights(1.0)`` with the
run's intent weight). The gradient of every weighted loss term is taken with
respect to the ear alone (input projection and Mamba blocks), and reported as
its norm and its share of the summed squared norms. B's terms (CB0, depth,
activity, intent) reach the ear only through the fusion blocks; the listening
terms (C1 next code, A activity) train it directly. The cosine between the
B-term gradient and the C1 gradient says whether the two pull the same way.
"""

import argparse
import dataclasses
import json
from pathlib import Path

import torch

from dauys.checkpoint import load_research_checkpoint
from dauys.config import ModelConfig
from dauys.model import DauysHybrid
from dauys.v02_stage1 import (
    load_batch,
    load_stage1_corpus,
    make_batch_spec,
    stage1_objective,
    stage1_weights,
)
from dauys.v03_intent import IntentConfig, IntentModule

TERMS = ("cb0", "depth", "activity", "intent", "listening_cb0", "listening_activity")
B_TERMS = ("cb0", "depth", "activity", "intent")


def _flat(grads) -> torch.Tensor:
    return torch.cat(
        [(g if g is not None else torch.zeros_like(p)).reshape(-1) for g, p in grads]
    )


def gradient_share(
    model, intent, records, silence, *, batches, intent_weight, history, seed
):
    ear = list(model.listening_projection.parameters()) + [
        p for block in model.listening for p in block.parameters()
    ]
    weights = dataclasses.replace(
        stage1_weights(1.0), intent=intent_weight, history_dropout=history
    )
    scale = {
        "cb0": weights.cb0,
        "depth": weights.depth,
        "activity": weights.activity,
        "intent": weights.intent,
        "listening_cb0": weights.listening,
        "listening_activity": weights.listening * weights.listening_activity,
    }
    sums = {term: None for term in TERMS}
    for step in range(batches):
        spec = make_batch_spec(records, step, seed=seed, speakers=4, chapter_only=True)
        batch = load_batch(records, spec, silence, seed=seed + step).to("cuda")
        loss = stage1_objective(model, batch, weights, intent)
        for term in TERMS:
            value = scale[term] * getattr(loss, term)
            grads = torch.autograd.grad(
                value, ear, retain_graph=True, allow_unused=True
            )
            flat = _flat(zip(grads, ear)).float()
            sums[term] = flat if sums[term] is None else sums[term] + flat
    means = {term: total / batches for term, total in sums.items()}
    squared = {term: float(g.pow(2).sum()) for term, g in means.items()}
    total = sum(squared.values())
    b_grad = sum(means[term] for term in B_TERMS)
    c1 = means["listening_cb0"]
    cosine = float(
        torch.nn.functional.cosine_similarity(b_grad, c1, dim=0)
        if c1.norm() > 0 and b_grad.norm() > 0
        else torch.tensor(0.0)
    )
    return {
        "norm": {term: round(value**0.5, 6) for term, value in squared.items()},
        "share": {term: round(value / total, 4) for term, value in squared.items()},
        "b_terms_share": round(sum(squared[t] for t in B_TERMS) / total, 4),
        "b_vs_c1_cosine": round(cosine, 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--silence-codes", type=Path, required=True)
    parser.add_argument("--checkpoints", nargs="+", required=True, help="name=path")
    parser.add_argument("--intent-weight", type=float, default=1.0)
    parser.add_argument("--history-dropout", type=float, nargs="+", default=[0.0, 0.5])
    parser.add_argument("--batches", type=int, default=16)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    silence = torch.tensor(json.loads(args.silence_codes.read_text()), dtype=torch.long)
    records = load_stage1_corpus(args.root, args.manifest)
    config = ModelConfig.from_yaml(args.config)
    rows = []
    for spec in args.checkpoints:
        name, path = spec.split("=", 1)
        payload = load_research_checkpoint(Path(path))
        model = DauysHybrid(config)
        model.load_state_dict(payload["model"])
        model = model.cuda().train()
        intent = IntentModule(
            model, IntentConfig(horizon=12, weight=args.intent_weight)
        ).cuda()
        intent.load_state_dict(payload["intent"])
        for history in args.history_dropout:
            row = {
                "checkpoint": name,
                "step": int(payload["step"]),
                "history_dropout": history,
            }
            row.update(
                gradient_share(
                    model,
                    intent,
                    records,
                    silence,
                    batches=args.batches,
                    intent_weight=args.intent_weight,
                    history=history,
                    seed=args.seed,
                )
            )
            print(json.dumps(row), flush=True)
            rows.append(row)
    args.output.write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
