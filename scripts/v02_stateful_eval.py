"""Evaluate Mamba carry versus reset CB0 NLL on stateful chunks."""

import argparse
import json
import sys
from pathlib import Path

from dauys.checkpoint import load_research_checkpoint
from dauys.config import ModelConfig
from dauys.model import DauysHybrid

sys.path.insert(0, str(Path(__file__).parent))
from v02_stateful_train import chunk_nlls, load_streams


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--heldout", type=Path, required=True)
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--chunk", type=int, default=256)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = ModelConfig.from_yaml(args.config)
    streams = load_streams(args.heldout)
    rows = []
    for spec in args.checkpoints:
        name, path = spec.split("=", 1)
        payload = load_research_checkpoint(Path(path))
        model = DauysHybrid(config)
        model.load_state_dict(payload["model"])
        model = model.cuda().eval()
        carry = chunk_nlls(
            model,
            streams,
            chunk=args.chunk,
            batch_size=args.batch,
            stateful=True,
        )
        reset = chunk_nlls(
            model,
            streams,
            chunk=args.chunk,
            batch_size=args.batch,
            stateful=False,
        )
        row = {
            "checkpoint": name,
            "step": int(payload.get("step", 0)),
            "chunks": {
                str(index): {
                    "carry_nll": round(carry[index], 4),
                    "reset_nll": round(reset[index], 4),
                    "gain": round(reset[index] - carry[index], 4),
                }
                for index in sorted(carry)
            },
        }
        print(json.dumps(row), flush=True)
        rows.append(row)
    if args.output is not None:
        args.output.write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
