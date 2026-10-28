"""One-time Stage 1 scoring on the test split (test-clean + test-other).

The test speakers are disjoint from training and from validation, and the
split is touched once, at the end, on the checkpoints already chosen on
validation. Reports the same M1–M4/M6 numbers the training loop logs.
"""

import argparse
import importlib.util
import json
from pathlib import Path

import torch

from dauys.checkpoint import load_research_checkpoint
from dauys.config import ModelConfig
from dauys.model import DauysHybrid
from dauys.v02_stage1 import load_stage1_corpus
from dauys.v02_stage1_eval import (
    Cb0Bigram,
    assert_speakers_disjoint,
    speech_cb0_runs,
)
from dauys.v03_intent import IntentConfig, IntentModule


def _train_module():
    path = Path(__file__).with_name("v02_stage1_train.py")
    spec = importlib.util.spec_from_file_location("v02_stage1_train", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--test-root", type=Path, required=True)
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument("--silence-codes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("checkpoints", type=Path, nargs="+")
    args = parser.parse_args()

    train = _train_module()
    records = load_stage1_corpus(args.cache_root, args.manifest)
    test = train._grouped(
        load_stage1_corpus(args.test_root, args.test_manifest), per_group=8, limit=256
    )
    assert_speakers_disjoint(records, test)
    silence = train._silence(args.silence_codes)
    bigram = Cb0Bigram.fit(speech_cb0_runs(records))
    test_runs = speech_cb0_runs(test)
    train_sample = train._spread(records, 64)
    config = ModelConfig.from_yaml(args.config)
    rows = []
    for checkpoint in args.checkpoints:
        payload = load_research_checkpoint(checkpoint)
        model = DauysHybrid(config)
        model.load_state_dict(payload["model"])
        model = model.cuda().eval()
        intent = IntentModule(model, IntentConfig(horizon=12, weight=0.5)).cuda()
        intent.load_state_dict(payload["intent"])
        intent.eval()
        with torch.inference_mode():
            metrics = train._evaluate(
                model,
                intent,
                test,
                silence,
                bigram,
                test_runs,
                batch_size=32,
                train_sample=train_sample,
            )
        row = {"checkpoint": str(checkpoint), "step": int(payload["step"])}
        row.update({key: round(value, 4) for key, value in metrics.items()})
        print(json.dumps(row), flush=True)
        rows.append(row)
    args.output.write_text("\n".join(json.dumps(row) for row in rows) + "\n")


if __name__ == "__main__":
    main()
