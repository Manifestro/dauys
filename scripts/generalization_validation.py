"""Compare original and silent inputs on frozen validation conversations."""

import argparse
import json
import subprocess
from dataclasses import asdict
from pathlib import Path

import torch
from generalization import (
    evaluate_teacher,
    selected_conversations,
    sha256,
    validate_selection,
)

from dauys.checkpoint import load_research_checkpoint
from dauys.codec import encode, load_mimi
from dauys.config import ModelConfig
from dauys.model import DauysHybrid


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--mimi-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available() or args.output.exists():
        raise RuntimeError("CUDA is required and output must not exist")
    splits = json.loads(args.splits.read_text(encoding="utf-8"))
    selection = validate_selection(splits)
    checkpoint = load_research_checkpoint(args.checkpoint)
    metadata = checkpoint["metadata"]
    config = ModelConfig.stage1()
    if metadata["splits_sha256"] != sha256(args.splits):
        raise ValueError("Selection mismatch")
    if metadata["model_config"] != asdict(config):
        raise ValueError("Model configuration mismatch")
    codec = load_mimi(args.mimi_checkpoint, "cuda")
    cases = selected_conversations(selection["validation"], codec)
    model = DauysHybrid(config).cuda().eval()
    model.load_state_dict(checkpoint["model"], strict=True)
    with torch.inference_mode():
        silence = encode(codec, torch.zeros((1, 1, 1920), device="cuda"))[:, :, 0]
    original = evaluate_teacher(model, cases, silence, config)
    silent = evaluate_teacher(model, cases, silence, config, silent_input=True)
    result = {
        "checkpoint_sha256": sha256(args.checkpoint),
        "checkpoint_step": int(checkpoint["step"]),
        "splits_sha256": sha256(args.splits),
        "code_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "original": original,
        "silence": silent,
        "silent_minus_original_nll": (
            silent["teacher_nll_nats_per_frame"]
            - original["teacher_nll_nats_per_frame"]
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
