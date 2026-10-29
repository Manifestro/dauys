"""GPU check: continuous Mamba forward equals chunked carry."""

import argparse
import json
import sys
from pathlib import Path

import torch

from dauys.model import MambaBlock


def _max_diff(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left - right).abs().max())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--frames", type=int, default=256)
    parser.add_argument("--chunk", type=int, default=64)
    parser.add_argument("--threshold", type=float, default=1e-3)
    parser.add_argument("--hybrid", action="store_true")
    parser.add_argument("--config", type=Path, help="model config for --hybrid")
    args = parser.parse_args()
    if args.frames % args.chunk:
        raise ValueError("Frames must be divisible by the chunk size")
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    block = MambaBlock(args.width, dropout=0.0, d_state=64).to(device).eval()
    x = torch.randn(args.batch, args.frames, args.width, device=device)
    results = {}
    with torch.no_grad():
        direct = block(x)
        stateless, _ = block.forward_with_state(x, None)
        results["stateless"] = _max_diff(direct, stateless)
        pieces = []
        state = None
        for start in range(0, args.frames, args.chunk):
            piece, state = block.forward_with_state(
                x[:, start : start + args.chunk], state
            )
            pieces.append(piece)
        carried = torch.cat(pieces, dim=1)
        results["chunked"] = _max_diff(direct, carried)
        if args.hybrid:
            from dauys.config import ModelConfig
            from dauys.model import DauysHybrid

            config = (
                ModelConfig.v02_stage1()
                if args.config is None
                else ModelConfig.from_yaml(args.config)
            )
            model = DauysHybrid(config).to(device).eval()
            codes = torch.randint(0, 2048, (args.batch, 8, args.frames), device=device)
            full = model.encode_listening(codes)
            states = {}
            chunks = []
            for start in range(0, args.frames, args.chunk):
                chunk = codes[:, :, start : start + args.chunk]
                chunks.append(model.encode_listening(chunk, mamba_states=states))
            results["hybrid"] = _max_diff(full, torch.cat(chunks, dim=1))
    ok = all(value < args.threshold for value in results.values())
    print(json.dumps({**results, "ok": ok}))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
