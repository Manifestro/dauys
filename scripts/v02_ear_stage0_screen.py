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

A cell is ``inputs:objective[:horizon[:width]]``: ``cb0`` or ``all8``;
``c1``, ``c1mem``, ``c1memhist`` (memory plus the unit histogram of the last
two seconds) or ``c1memmimi`` (the same lags but Mimi CB0 codes as the memory
target, so training needs no HuBERT); the memory horizon in frames (8 by
default); the ear width (512 by default). HuBERT units are still attached for
the probes. Two probes test what no cell is trained on: the units of frames
t+12 and t+25 (``unit@-12``, ``unit@-25``, the next 1–2 s), and the unit
histogram of the last two seconds (``hist2s_r2``, ridge R²).
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
    assign_units,
    hubert_frames,
    kmeans,
    load_utterances,
    ridge_r2,
    unit_probe,
)

LAG_GRID = (0, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32)
HISTOGRAM_FRAMES = 25


def memory_lags(horizon: int) -> tuple[int, ...]:
    return tuple(lag for lag in LAG_GRID if lag <= horizon)


class Ear(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        codebooks: int,
        clusters: int,
        lags: tuple[int, ...],
        width: int,
        histogram: bool = False,
        memory_target: str = "units",
    ):
        super().__init__()
        if memory_target not in ("units", "cb0"):
            raise ValueError(f"Unknown memory target {memory_target}")
        self.codebooks = codebooks
        self.embeddings = nn.ModuleList(
            nn.Embedding(config.vocab_size, config.embedding_dim)
            for _ in range(codebooks)
        )
        self.projection = nn.Linear(config.embedding_dim, width)
        self.blocks = nn.ModuleList(
            MambaBlock(
                width,
                config.dropout,
                config.listening_d_state,
                config.mamba_norm_eps,
            )
            for _ in range(config.listening_layers)
        )
        self.norm = nn.RMSNorm(width)
        self.next_head = nn.Linear(width, config.vocab_size)
        self.lags = lags
        self.memory_target = memory_target
        self.memory_classes = config.vocab_size if memory_target == "cb0" else clusters
        self.memory_head = (
            nn.Linear(width, self.memory_classes * len(lags)) if lags else None
        )
        self.histogram_head = nn.Linear(width, clusters) if histogram else None
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


def unit_histograms(units, speech, clusters: int):
    """[batch, frames, clusters] share of each unit among the speech frames of
    the last HISTOGRAM_FRAMES frames, and where it is defined."""
    onehot = F.one_hot(units, clusters).float() * speech[..., None]
    total = F.pad(onehot.cumsum(1), (0, 0, 1, 0))
    end = torch.arange(1, units.shape[1] + 1, device=units.device)
    start = (end - HISTOGRAM_FRAMES).clamp_min(0)
    counts = total[:, end] - total[:, start]
    mass = counts.sum(-1, keepdim=True)
    valid = (mass[..., 0] > 0) & (end >= HISTOGRAM_FRAMES)[None]
    return counts / mass.clamp_min(1), valid


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
            batch, frames, len(ear.lags), ear.memory_classes
        )
        terms = []
        for index, lag in enumerate(ear.lags):
            view = logits[:, lag:, index]
            target = (
                codes[:, 0, : frames - lag]
                if ear.memory_target == "cb0"
                else units[:, : frames - lag]
            )
            terms.append(
                _masked(
                    F.cross_entropy(
                        view.reshape(-1, ear.memory_classes),
                        target.reshape(-1),
                        reduction="none",
                    ).reshape(batch, frames - lag),
                    speech[:, : frames - lag],
                )
            )
        memory = torch.stack(terms).mean()
    if ear.histogram_head is not None:
        target, valid = unit_histograms(units, speech, ear.clusters)
        predicted = F.log_softmax(ear.histogram_head(normed).float(), -1)
        memory = memory + _masked(-(target * predicted).sum(-1), valid)
    return c1, memory


def parse_cell(cell: str) -> dict:
    parts = cell.split(":")
    inputs, objective = parts[0], parts[1]
    horizon = int(parts[2]) if len(parts) > 2 else 8
    width = int(parts[3]) if len(parts) > 3 else 512
    if inputs not in ("cb0", "all8") or objective not in (
        "c1",
        "c1mem",
        "c1memhist",
        "c1memmimi",
    ):
        raise ValueError(f"Unknown cell {cell}")
    return {
        "name": f"ear_{inputs}_{objective}_h{horizon}_w{width}",
        "codebooks": 1 if inputs == "cb0" else 8,
        "lags": memory_lags(horizon) if objective != "c1" else (),
        "histogram": objective == "c1memhist",
        "memory_target": "cb0" if objective == "c1memmimi" else "units",
        "width": width,
    }


def train_cell(config, cell, train, silence, args, log):
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    ear = (
        Ear(
            config,
            cell["codebooks"],
            args.clusters,
            cell["lags"],
            cell["width"],
            cell["histogram"],
            memory_target=cell.get("memory_target", "units"),
        )
        .cuda()
        .train()
    )
    log({"cell": cell["name"], "parameters": sum(p.numel() for p in ear.parameters())})
    optimiser = torch.optim.AdamW(ear.parameters(), lr=args.lr, weight_decay=0.01)
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimiser,
        lambda step: (
            min(1.0, (step + 1) / 200)
            * 0.5
            * (1 + math.cos(math.pi * min(step, args.steps) / args.steps))
        ),
    )
    name = cell["name"]
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


def lag_pairs(utterances, name: str, lag: int):
    """Feature at t and the unit of frame t − lag; a negative lag is the future."""
    xs, ys = [], []
    for item in utterances:
        x, units, speech = item["features"][name], item["units"], item["activity"]
        frames, shift = len(units), abs(lag)
        if shift >= frames:
            continue
        if lag >= 0:
            feature, target, keep = (
                x[lag:],
                units[: frames - lag],
                speech[: frames - lag],
            )
        else:
            feature, target, keep = x[: frames - shift], units[shift:], speech[shift:]
        xs.append(feature[keep])
        ys.append(target[keep])
    return torch.cat(xs).float().cuda(), torch.cat(ys).cuda()


def histogram_pairs(utterances, name: str, clusters: int):
    xs, ys = [], []
    for item in utterances:
        target, valid = unit_histograms(
            item["units"][None], item["activity"][None].bool(), clusters
        )
        xs.append(item["features"][name][valid[0]])
        ys.append(target[0][valid[0]])
    return torch.cat(xs).float().cuda(), torch.cat(ys).float().cuda()


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
    parser.add_argument(
        "--lags", type=int, nargs="+", default=[-25, -12, 0, 2, 4, 8, 12, 16, 24, 32]
    )
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
        name, ear = train_cell(config, parse_cell(cell), train, silence, args, log)
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
                        *lag_pairs(probe, feature, lag),
                        *lag_pairs(test, feature, lag),
                        args.clusters,
                    ),
                    4,
                )
            row["hist2s_r2"] = round(
                ridge_r2(
                    *histogram_pairs(probe, feature, args.clusters),
                    *histogram_pairs(test, feature, args.clusters),
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
        "lag_grid": list(LAG_GRID),
        "seconds": round(time.time() - started),
        "cells": rows,
    }
    args.output.write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
