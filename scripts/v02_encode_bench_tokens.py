"""Encode every frozen Hohere probe through Mimi once and save the codes.

Probe audio and its Mimi codes never depend on the model being scored, so
evaluations load this file (DAUYS_TOKEN_CACHE) instead of re-encoding about
ten thousand clips per run.
"""

import argparse
import json
import time
from pathlib import Path

import torch

from dauys.codec import load_mimi
from dauys.v02_evaluation import load_suite
from dauys.v02_hohere_backend import HohereProbeBackend


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suites-dir", type=Path, required=True)
    parser.add_argument("--mimi-checkpoint", type=Path, required=True)
    parser.add_argument("--silence-codes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    suites = {
        f"E{index}": load_suite(
            args.suites_dir / f"E{index}.json",
            (args.suites_dir / f"E{index}.sha256").read_text().strip(),
        )
        for index in range(1, 7)
    }
    silence = torch.tensor(json.loads(args.silence_codes.read_text()), dtype=torch.long)
    codec = load_mimi(args.mimi_checkpoint, "cuda")
    backend = HohereProbeBackend(
        args.suites_dir, suites, codec, silence, token_cache=args.output
    )
    started = time.time()
    for name in ("E1", "E2", "E5", "E6"):
        for item in suites[name].items:
            backend._channels(item)
    for item in suites["E3"].items:
        for question in ("x", "y"):
            for answer in ("x", "y"):
                backend._compose(item, question, answer)
    for item in suites["E4"].items:
        for answer in range(4):
            backend._compose(item, "base", str(answer))
    backend.save_token_cache()
    print(
        json.dumps(
            {
                "entries": len(backend._token_cache),
                "seconds": round(time.time() - started, 1),
                "bytes": args.output.stat().st_size,
            }
        )
    )


if __name__ == "__main__":
    main()
