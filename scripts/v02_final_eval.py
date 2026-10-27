"""Score one checkpoint on the frozen Hohere E1–E6 suites."""

import argparse
import json
import time
from pathlib import Path

from dauys.checkpoint import load_research_checkpoint
from dauys.config import ModelConfig
from dauys.model import DauysHybrid
from dauys.v02_data import load_corpus
from dauys.v02_evaluation import load_suite
from dauys.v02_hohere_backend import create_evaluator


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--suites-dir", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--companion", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    suites = {
        f"E{index}": load_suite(
            args.suites_dir / f"E{index}.json",
            (args.suites_dir / f"E{index}.sha256").read_text().strip(),
        )
        for index in range(1, 7)
    }
    records = [
        record
        for record in load_corpus(args.cache_root, args.manifest, args.companion)
        if record.source == "hohere"
    ]
    model = DauysHybrid(ModelConfig.from_yaml(args.config)).cuda()
    payload = load_research_checkpoint(args.checkpoint)
    model.load_state_dict(payload["model"])
    model.eval()
    harness = create_evaluator(
        suites=suites, records=sorted(records, key=lambda r: r.id)
    )
    started = time.time()
    metrics = harness(model, int(payload.get("step", 0)))
    metrics["eval/seconds"] = time.time() - started
    harness.backend.save_token_cache()  # type: ignore[attr-defined]
    args.output.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metrics, sort_keys=True))


if __name__ == "__main__":
    main()
