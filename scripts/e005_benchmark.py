"""Measure one representative E005 optimizer microstep before training."""

import argparse
import json
import time
from pathlib import Path

import torch
from e005_train import load_case, on_gpu, read_manifest, set_context

from dauys.codec import encode, load_mimi
from dauys.config import ModelConfig
from dauys.loss import training_loss
from dauys.model import DauysHybrid


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=Path, required=True)
    parser.add_argument("--mimi-checkpoint", type=Path, required=True)
    parser.add_argument("--conversation-id", required=True)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    rows = read_manifest(args.tokens / "train.jsonl", "train")
    row = next((item for item in rows if item["id"] == args.conversation_id), None)
    if row is None:
        raise ValueError("Conversation is not in the training cache")
    torch.manual_seed(42)
    codec = load_mimi(args.mimi_checkpoint, "cuda")
    with torch.inference_mode():
        silence = encode(codec, torch.zeros((1, 1, 1920), device="cuda"))[:, :, 0]
    config = ModelConfig.stage1()
    model = DauysHybrid(config).cuda().train()
    set_context(model, 256)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.01)
    user, target, activity = on_gpu(load_case(args.tokens, row))
    durations = []
    for _ in range(args.repeats):
        start = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(user, target, silence)
            loss = training_loss(
                output,
                target,
                activity,
                torch.ones_like(activity, dtype=torch.bool),
                config,
            ).total
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        torch.cuda.synchronize()
        durations.append(time.monotonic() - start)
    print(
        json.dumps(
            {
                "conversation_id": row["id"],
                "frames": row["frames"],
                "durations_seconds": durations,
                "steady_state_seconds": sum(durations[1:]) / max(1, len(durations) - 1),
                "peak_gpu_bytes": torch.cuda.max_memory_allocated(),
            }
        )
    )


if __name__ == "__main__":
    main()
