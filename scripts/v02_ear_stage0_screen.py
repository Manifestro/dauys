"""Stage 0 screen: what makes a causal ear keep what was said?

Two hypotheses for why the Stage 1 ear forgets content faster than a plain
four-frame CB0 buffer:

- H1, input: the ear hears the sum of eight code embeddings, and CB1–7
  (sound, voice, intonation) drown CB0, which carries the content.
- H2, objective: next-code prediction (C1) needs only the last frame or two,
  so nothing asks the ear to remember.

A 2×2 screen trains the ear alone (same Mamba2 512×6 as Stage 1, from
scratch) on single utterances: input CB0 only or all eight codebooks, and
objective C1 alone or C1 plus memory (the HuBERT units of frames t−lag for
several lags, HuBERT base being self-supervised on audio only). Every cell is
then scored with the ear probe of ``v02_ear_hubert_probe.py``: the same probe
utterances, units and linear probes, so its numbers compare directly with the
earlier rows (CB0 buffer, ear fix1).
"""

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from dauys.codec import load_mimi
from dauys.config import ModelConfig
from dauys.model import MambaBlock

sys.path.insert(0, str(Path(__file__).parent))
from v02_ear_hubert_probe import (
    _pairs,
    assign_units,
    hubert_frames,
    kmeans,
    load_utterances,
    unit_probe,
)

MEMORY_LAGS = (0, 1, 2, 3, 4, 6, 8)


class Ear(nn.Module):
    def __init__(self, config: ModelConfig, codebooks: int, clusters: int, memory):
        super().__init__()
        self.codebooks = codebooks
        self.embeddings = nn.ModuleList(
            nn.Embedding(config.vocab_size, config.embedding_dim)
            for _ in range(codebooks)
        )
        self.projection = nn.Linear(config.embedding_dim, config.listening_dim)
        self.blocks = nn.ModuleList(
            MambaBlock(
                config.listening_dim,
                config.dropout,
                config.listening_d_state,
                config.mamba_norm_eps,
            )
            for _ in range(config.listening_layers)
        )
        self.norm = nn.RMSNorm(config.listening_dim)
        self.next_head = nn.Linear(config.listening_dim, config.vocab_size)
        self.memory_head = (
            nn.Linear(config.listening_dim, clusters * len(MEMORY_LAGS))
            if memory
            else None
        )
        self.clusters = clusters

    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        x = sum(self.embeddings[q](codes[:, q]) for q in range(self.codebooks))
        x = self.projection(x / math.sqrt(self.codebooks))
        for block in self.blocks:
            x = block(x)
        return x


def _batch(items, silence):
    frames = max(len(item["units"]) for item in items)
    codes = silence[None, :, None].expand(len(items), 8, frames).clone()
    units = torch.zeros(len(items), frames, dtype=torch.long)
    speech = torch.zeros(len(items), frames, dtype=torch.bool)
    for row, item in enumerate(items):
        length = len(item["units"])
        codes[row, :, :length] = item["codes"]
        units[row, :length] = item["units"]
        speech[row, :length] = item["activity"]
    return codes.cuda(), units.cuda(), speech.cuda()


def _masked(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (values * mask).sum() / mask.sum().clamp_min(1)


def losses(ear: Ear, codes, units, speech):
    states = ear(codes)
    normed = ear.norm(states)
    batch, frames, _ = states.shape
    next_logits = ear.next_head(normed[:, :-1])
    c1 = _masked(
        F.cross_entropy(
            next_logits.reshape(-1, next_logits.shape[-1]),
            codes[:, 0, 1:].reshape(-1),
            reduction="none",
        ).reshape(batch, frames - 1),
        speech[:, 1:],
    )
    memory = torch.zeros((), device=codes.device)
    if ear.memory_head is not None:
        logits = ear.memory_head(normed).reshape(
            batch, frames, len(MEMORY_LAGS), ear.clusters
        )
        terms = []
        for index, lag in enumerate(MEMORY_LAGS):
            view = logits[:, lag:, index]
            terms.append(
                _masked(
                    F.cross_entropy(
                        view.reshape(-1, ear.clusters),
                        units[:, : frames - lag].reshape(-1),
                        reduction="none",
                    ).reshape(batch, frames - lag),
                    speech[:, : frames - lag],
                )
            )
        memory = torch.stack(terms).mean()
    return c1, memory


def train_cell(config, codebooks, memory, train, silence, args, log):
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    ear = Ear(config, codebooks, args.clusters, memory).cuda().train()
    optimiser = torch.optim.AdamW(ear.parameters(), lr=args.lr, weight_decay=0.01)
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimiser,
        lambda step: (
            min(1.0, (step + 1) / 200)
            * 0.5
            * (1 + math.cos(math.pi * min(step, args.steps) / args.steps))
        ),
    )
    name = f"ear_{'cb0' if codebooks == 1 else 'all8'}_{'c1mem' if memory else 'c1'}"
    for step in range(1, args.steps + 1):
        codes, units, speech = _batch(rng.sample(train, args.batch), silence)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            c1, mem = losses(ear, codes, units, speech)
        loss = c1 + mem
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(ear.parameters(), 1.0)
        optimiser.step()
        schedule.step()
        if step % 500 == 0 or step == args.steps:
            log(
                {
                    "cell": name,
                    "step": step,
                    "c1": round(c1.item(), 4),
                    "memory": round(mem.item(), 4),
                }
            )
    return name, ear.eval()


@torch.no_grad()
def heldout_c1(ear: Ear, items, silence) -> float:
    total = count = 0.0
    for start in range(0, len(items), 32):
        codes, units, speech = _batch(items[start : start + 32], silence)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            c1, _ = losses(ear, codes, units, speech)
        weight = float(speech[:, 1:].sum())
        total += c1.item() * weight
        count += weight
    return total / count


@torch.no_grad()
def attach_features(name: str, ear: Ear, items, buffer: int = 0) -> None:
    """The ear's states, and with ``buffer`` the cell's own input embeddings of
    the last ``buffer`` frames: a delay line the ear has to beat."""
    for item in items:
        codes = item["codes"][None].cuda()
        features = item.setdefault("features", {})
        features[name] = ear(codes)[0].float().half().cpu()
        if buffer:
            x = sum(ear.embeddings[q](codes[:, q]) for q in range(ear.codebooks))[0]
            padded = torch.cat((x[:1].expand(buffer - 1, -1), x))
            window = torch.cat([padded[k : k + len(x)] for k in range(buffer)], -1)
            features[f"{name}_buffer{buffer}"] = window.float().half().cpu()


def _units(args, log):
    """Probe, eval and train utterances with HuBERT units attached."""
    # The probe split is drawn exactly as v02_ear_hubert_probe.py draws it.
    probe = load_utterances(args.train_roots, args.probe_items, 240, args.seed)
    test = load_utterances(args.eval_roots, args.eval_items, 240, args.seed)
    held = {item["id"] for item in probe}
    train = [
        item
        for item in load_utterances(
            args.train_roots, args.train_items + args.probe_items, 256, args.seed + 1
        )
        if item["id"] not in held
    ][: args.train_items]
    overlap = {u["speaker"] for u in probe + train} & {u["speaker"] for u in test}
    if overlap:
        raise ValueError(f"Train and eval speakers overlap: {sorted(overlap)[:5]}")
    log({"probe": len(probe), "train": len(train), "eval": len(test)})

    from transformers import HubertModel

    codec = load_mimi(args.mimi_checkpoint, "cuda")
    hubert = HubertModel.from_pretrained(args.hubert).cuda().eval()
    hubert_frames(probe + test, codec, hubert, args.layer)
    speech = torch.cat([u["hubert"][u["activity"]] for u in probe]).cuda()
    centres = kmeans(speech, args.clusters, seed=args.seed)
    assign_units(probe + test, centres)
    for item in probe + test:
        del item["hubert"]
    for start in range(0, len(train), 256):
        chunk = train[start : start + 256]
        hubert_frames(chunk, codec, hubert, args.layer)
        assign_units(chunk, centres)
        for item in chunk:
            del item["hubert"]
    del codec, hubert
    torch.cuda.empty_cache()
    log({"units": "done"})
    return probe, test, train


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mimi-checkpoint", type=Path, required=True)
    parser.add_argument("--train-roots", type=Path, nargs="+", required=True)
    parser.add_argument("--eval-roots", type=Path, nargs="+", required=True)
    parser.add_argument("--silence-codes", type=Path, required=True)
    parser.add_argument("--train-items", type=int, default=20000)
    parser.add_argument("--probe-items", type=int, default=1500)
    parser.add_argument("--eval-items", type=int, default=500)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--clusters", type=int, default=200)
    parser.add_argument("--hubert", default="facebook/hubert-base-ls960")
    parser.add_argument("--layer", type=int, default=9)
    parser.add_argument("--lags", type=int, nargs="+", default=[0, 1, 2, 4, 8])
    parser.add_argument(
        "--buffer",
        type=int,
        default=0,
        help="Also probe a delay line of this many input frames per cell",
    )
    parser.add_argument("--save-dir", type=Path, help="Save each trained ear here")
    parser.add_argument(
        "--cells", nargs="*", default=["cb0:c1", "cb0:c1mem", "all8:c1", "all8:c1mem"]
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--units-cache",
        type=Path,
        help="Save the utterances with units here, or load them if present, so "
        "cells can run as parallel processes",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if "ft" in args.hubert.split("/")[-1]:
        raise ValueError("Use a self-supervised HuBERT, not a text-finetuned one")
    started = time.time()

    def log(row):
        print(json.dumps({**row, "seconds": round(time.time() - started)}), flush=True)

    silence = torch.tensor(json.loads(args.silence_codes.read_text()), dtype=torch.long)
    if args.units_cache and args.units_cache.exists():
        probe, test, train = torch.load(args.units_cache, weights_only=False)
        log({"cache": "loaded", "train": len(train)})
    else:
        probe, test, train = _units(args, log)
        if args.units_cache:
            torch.save((probe, test, train), args.units_cache)
    if not args.cells:
        return

    config = ModelConfig.from_yaml(args.config)
    rows = []
    for cell in args.cells:
        inputs, objective = cell.split(":")
        name, ear = train_cell(
            config,
            1 if inputs == "cb0" else 8,
            objective == "c1mem",
            train,
            silence,
            args,
            log,
        )
        if args.save_dir:
            args.save_dir.mkdir(parents=True, exist_ok=True)
            torch.save(ear.state_dict(), args.save_dir / f"{name}.pt")
        attach_features(name, ear, probe + test, args.buffer)
        names = [name] + ([f"{name}_buffer{args.buffer}"] if args.buffer else [])
        for feature in names:
            row = {"features": feature}
            if feature == name:
                row["heldout_c1"] = round(heldout_c1(ear, test, silence), 4)
            for lag in args.lags:
                row[f"unit@{lag}"] = round(
                    unit_probe(
                        *_pairs(probe, feature, lag),
                        *_pairs(test, feature, lag),
                        args.clusters,
                    ),
                    4,
                )
            log(row)
            rows.append(row)
        del ear
        torch.cuda.empty_cache()
    summary = {
        "hubert": args.hubert,
        "layer": args.layer,
        "clusters": args.clusters,
        "steps": args.steps,
        "train_utterances": len(train),
        "memory_lags": list(MEMORY_LAGS),
        "seconds": round(time.time() - started),
        "cells": rows,
    }
    args.output.write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
