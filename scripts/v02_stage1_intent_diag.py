"""Can the intent targets tell same-chapter continuations apart at all?

For held-out same-chapter pools of eight, the EMA-ear targets are pooled the
way training pools them. Reported per checkpoint:

- ``within_cos``: mean cosine between different candidates of one pool.
- ``between_cos``: mean cosine between candidates of different pools.
- ``content_share``: variance left after removing each pool's mean, over the
  total variance; near 0 means the targets encode only voice and recording.
- ``tail_retrieval``: pick the true continuation by the cosine between the
  pooled EMA ear of A's last 32 frames and each candidate, after removing the
  pool mean; a head-free check of whether the target space carries continuity.
"""

import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F

from dauys.checkpoint import load_research_checkpoint
from dauys.config import ModelConfig
from dauys.model import DauysHybrid
from dauys.v02_stage1 import load_batch, load_stage1_corpus
from dauys.v02_stage1_eval import _chapter_groups
from dauys.v03_intent import IntentConfig, IntentModule

WINDOW = 32


def _pool(intent, codes, activity):
    states = intent.target(codes.long()).float()
    mask = activity.float()
    return (states * mask[..., None]).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)


@torch.inference_mode()
def diagnose(model, intent, records, silence, *, groups=32, seed=7) -> dict:
    eligible = [i for i in _chapter_groups(records).values() if len(i) >= 8]
    rng = random.Random(seed)
    chosen = rng.sample(eligible, min(groups, len(eligible)))
    device = next(model.parameters()).device
    pools, tails = [], []
    for indices in chosen:
        pool = tuple(rng.sample(indices, 8))
        batch = load_batch(records, [(pool, tuple(range(8)))], silence, seed=seed)
        batch = batch.to(device)
        targets = _pool(intent, batch.candidate_codes[0], batch.candidate_activity[0])
        rows = []
        for row in range(batch.user_codes.shape[0]):
            end = int(batch.intent_frames[row]) + 1
            start = max(0, end - WINDOW)
            rows.append(
                _pool(
                    intent,
                    batch.user_codes[row : row + 1, :, start:end],
                    batch.user_activity[row : row + 1, start:end].clamp_min(0),
                )
            )
        tail = torch.cat(rows)
        order = batch.positive_index[0]
        pools.append(targets)
        tails.append((tail, order))
    stacked = torch.stack(pools)  # [groups, 8, d]
    unit = F.normalize(stacked, dim=-1)
    sims = unit @ unit.transpose(1, 2)
    off = ~torch.eye(8, dtype=torch.bool, device=sims.device)
    within = float(sims[:, off].mean())
    flat = F.normalize(stacked.mean(1), dim=-1)
    between_matrix = flat @ flat.T
    between = float(
        between_matrix[
            ~torch.eye(len(pools), dtype=torch.bool, device=sims.device)
        ].mean()
    )
    centered = stacked - stacked.mean(1, keepdim=True)
    total = stacked - stacked.reshape(-1, stacked.shape[-1]).mean(0)
    content_share = float(centered.pow(2).sum() / total.pow(2).sum())
    correct = count = 0
    for targets, (tail, order) in zip(pools, tails):
        mean = targets.mean(0, keepdim=True)
        scores = (
            F.normalize(tail - mean, dim=-1) @ F.normalize(targets - mean, dim=-1).T
        )
        correct += int((scores.argmax(-1) == order).sum())
        count += order.numel()
    return {
        "within_cos": round(within, 4),
        "between_cos": round(between, 4),
        "content_share": round(content_share, 4),
        "tail_retrieval": round(correct / count, 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--heldout-root", type=Path, required=True)
    parser.add_argument("--heldout-manifest", type=Path, required=True)
    parser.add_argument("--silence-codes", type=Path, required=True)
    parser.add_argument("checkpoints", type=Path, nargs="+")
    args = parser.parse_args()
    silence = torch.tensor(json.loads(args.silence_codes.read_text()), dtype=torch.long)
    records = load_stage1_corpus(args.heldout_root, args.heldout_manifest)
    config = ModelConfig.from_yaml(args.config)
    for checkpoint in args.checkpoints:
        payload = load_research_checkpoint(checkpoint)
        model = DauysHybrid(config)
        model.load_state_dict(payload["model"])
        model = model.cuda().eval()
        intent = IntentModule(model, IntentConfig(horizon=12, weight=0.5)).cuda()
        intent.load_state_dict(payload["intent"])
        row = {"file": checkpoint.name, "step": int(payload["step"])}
        row.update(diagnose(model, intent, records, silence))
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
