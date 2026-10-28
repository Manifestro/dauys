"""Audit what a Stage 1 checkpoint's dependence on A is made of.

For every held-out A→B item, B's speech frames are scored under four inputs:
the true A, two different same-chapter phrases in A's place (``swap``,
``swap2``: same speaker, room and microphone, wrong content, ending at the same
cut), and silence. Per item the script keeps CB0 and depth (CB1–7) NLL sums,
over all B speech frames and over the first 12 (the onset). The model is also
run with all fusion blocks off, and with each fusion block off alone.

Reported per checkpoint, each with a 95% bootstrap interval over items:

- ``content``: swap − true. What the wrong same-voice phrase loses; only the
  content of A can explain it.
- ``voice_timing``: silent − swap. What any same-voice speech ending at the cut
  gives: voice, room and timing.
- ``swap_noise``: swap2 − swap. Two equally wrong phrases; the content gain
  must stand clear of it.
- ``m4``: fusion off − true, and per fusion block.
- ``meaning``: shuffle − true. A keeps its last second untouched; everything
  before it is cut into 1 s chunks whose order is shuffled. Voice, room,
  sounds, words and the ending stay; only the order, the phrase, is gone.
  ``shuffle_noise`` (shuffle2 − shuffle, another order) bounds it. Only items
  with at least two chunks to shuffle count.
"""

import argparse
import importlib.util
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F

from dauys.checkpoint import load_research_checkpoint
from dauys.config import ModelConfig
from dauys.model import DauysHybrid
from dauys.v02_stage1 import load_stage1_corpus
from dauys.v02_stage1_eval import record_tensors

ONSET_FRAMES = 12
CHUNK_FRAMES = 12


def shuffled_user(item, seed: int) -> torch.Tensor | None:
    """A with its last second kept and the 1 s chunks before it reordered;
    None when fewer than two whole chunks precede the last second."""
    tail = item.cut_local - CHUNK_FRAMES
    chunks = tail // CHUNK_FRAMES
    if chunks < 2:
        return None
    head = tail - chunks * CHUNK_FRAMES
    order = list(range(chunks))
    rng = random.Random(seed)
    while order == sorted(order):
        rng.shuffle(order)
    user = item.user.clone()
    pieces = [
        item.user[:, head + k * CHUNK_FRAMES : head + (k + 1) * CHUNK_FRAMES]
        for k in order
    ]
    user[:, head:tail] = torch.cat(pieces, dim=1)
    return user


def _loopholes():
    path = Path(__file__).with_name("v02_stage1_loopholes.py")
    spec = importlib.util.spec_from_file_location("v02_stage1_loopholes", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@torch.inference_mode()
def item_scores(model, items, partner, silence, mode, *, off=frozenset(), batch=16):
    """Per item: [cb0_sum, depth_sum, frames, onset_cb0, onset_depth, onset_frames]."""
    loopholes = _loopholes()
    device = next(model.parameters()).device
    rows = []
    for start in range(0, len(items), batch):
        chunk = range(start, min(start + batch, len(items)))
        frames = max(items[i].frames for i in chunk)
        size = len(chunk)
        user = silence[None, :, None].expand(size, 8, frames).clone()
        target = user.clone()
        speech = torch.zeros((size, frames), dtype=torch.bool)
        onset = torch.zeros((size, frames), dtype=torch.bool)
        for row, index in enumerate(chunk):
            item = items[index]
            length = item.frames
            if mode in ("shuffle", "shuffle2"):
                shuffled = shuffled_user(item, 2 * index + (mode == "shuffle2"))
                user[row, :, :length] = item.user if shuffled is None else shuffled
            else:
                variant = "swap" if mode == "swap2" else mode
                user[row, :, :length] = loopholes._user_variant(
                    item, items[partner[index]], variant, silence
                )
            target[row, :, :length] = item.target
            active = item.target_activity.bool()
            speech[row, :length] = active
            positions = active.nonzero().flatten()
            positions = positions[positions >= item.cut_local]
            onset[row, positions[:ONSET_FRAMES]] = True
        initial = silence[None].expand(size, 8).clone()
        output = model(
            user.to(device),
            target.to(device),
            initial.to(device),
            fusion_off_blocks=off,
        )
        target = target.to(device)
        cb0 = F.cross_entropy(
            output.cb0_logits.float().transpose(1, 2), target[:, 0], reduction="none"
        )
        depth = F.cross_entropy(
            output.depth_logits.float().permute(0, 3, 1, 2),
            target[:, 1:].transpose(1, 2),
            reduction="none",
        ).mean(-1)
        speech, onset = speech.to(device), onset.to(device)
        for row in range(size):
            s, o = speech[row], onset[row]
            rows.append(
                [
                    float(cb0[row][s].sum()),
                    float(depth[row][s].sum()),
                    float(s.sum()),
                    float(cb0[row][o].sum()),
                    float(depth[row][o].sum()),
                    float(o.sum()),
                ]
            )
    return torch.tensor(rows, dtype=torch.float64)


def _rate(scores, column, frames_column, index=None):
    s = scores if index is None else scores[index]
    return s[:, column].sum() / s[:, frames_column].sum().clamp_min(1)


def contrast(high, low, *, resamples=1000, seed=0):
    """Per-frame NLL difference high − low with a bootstrap 95% interval."""
    generator = torch.Generator().manual_seed(seed)
    out = {}
    for name, column, frames in (
        ("cb0", 0, 2),
        ("depth", 1, 2),
        ("onset_cb0", 3, 5),
        ("onset_depth", 4, 5),
    ):
        point = float(_rate(high, column, frames) - _rate(low, column, frames))
        draws = []
        for _ in range(resamples):
            index = torch.randint(len(high), (len(high),), generator=generator)
            draws.append(
                float(
                    _rate(high, column, frames, index)
                    - _rate(low, column, frames, index)
                )
            )
        draws.sort()
        out[name] = [
            round(point, 4),
            round(draws[int(0.025 * resamples)], 4),
            round(draws[int(0.975 * resamples)], 4),
        ]
    return out


def _partners(records, seed):
    loopholes = _loopholes()
    return loopholes._partners(records, seed=seed)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--heldout-root", type=Path, required=True)
    parser.add_argument("--heldout-manifest", type=Path, required=True)
    parser.add_argument("--silence-codes", type=Path, required=True)
    parser.add_argument("--checkpoints", nargs="+", required=True, help="name=path")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--meaning-only",
        action="store_true",
        help="Score only true and shuffled A (the meaning contrast)",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    silence = torch.tensor(json.loads(args.silence_codes.read_text()), dtype=torch.long)
    records = load_stage1_corpus(args.heldout_root, args.heldout_manifest)
    items = [record_tensors(record, silence) for record in records]
    partner = _partners(records, seed=5)
    # A second wrong phrase; in two-phrase chapters it is the same one.
    partner2 = _partners(records, seed=11)
    config = ModelConfig.from_yaml(args.config)
    rows = []
    for spec in args.checkpoints:
        name, path = spec.split("=", 1)
        payload = load_research_checkpoint(Path(path))
        model = DauysHybrid(config)
        model.load_state_dict(payload["model"])
        model = model.cuda().eval()
        shuffled = torch.tensor(
            [shuffled_user(item, 0) is not None for item in items], dtype=torch.bool
        )
        meaning = {
            mode: item_scores(
                model, items, partner, silence, mode, batch=args.batch_size
            )[shuffled]
            for mode in ("true", "shuffle", "shuffle2")
        }
        meaning_row = {
            "checkpoint": name,
            "step": int(payload["step"]),
            "shuffled_items": int(shuffled.sum()),
            "meaning": contrast(meaning["shuffle"], meaning["true"]),
            "shuffle_noise": contrast(meaning["shuffle2"], meaning["shuffle"]),
        }
        if args.meaning_only:
            print(json.dumps(meaning_row), flush=True)
            rows.append(meaning_row)
            continue
        run = {
            mode: item_scores(
                model,
                items,
                partner2 if mode == "swap2" else partner,
                silence,
                mode,
                batch=args.batch_size,
            )
            for mode in ("true", "swap", "swap2", "silent")
        }
        blocks = model.fusion_block_indices()
        off_all = item_scores(
            model,
            items,
            partner,
            silence,
            "true",
            off=frozenset(blocks),
            batch=args.batch_size,
        )
        row = {
            "checkpoint": name,
            "step": int(payload["step"]),
            "items": len(items),
            "true_nll": {
                key: round(float(_rate(run["true"], c, f)), 4)
                for key, c, f in (("cb0", 0, 2), ("depth", 1, 2))
            },
            "content": contrast(run["swap"], run["true"]),
            "voice_timing": contrast(run["silent"], run["swap"]),
            "swap_noise": contrast(run["swap2"], run["swap"]),
            "m4": contrast(off_all, run["true"]),
            "m4_per_block": {},
            **{k: v for k, v in meaning_row.items() if k not in ("checkpoint", "step")},
        }
        for block in blocks:
            off_one = item_scores(
                model,
                items,
                partner,
                silence,
                "true",
                off=frozenset({block}),
                batch=args.batch_size,
            )
            row["m4_per_block"][str(block)] = contrast(off_one, run["true"])["cb0"]
        print(json.dumps(row), flush=True)
        rows.append(row)
    args.output.write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
