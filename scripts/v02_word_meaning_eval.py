"""Minimal-pair test: does the model's reply depend on which word it heard?

For two words w, v of one topic spoken by one voice, the model scores each
reply B after each phrase A (teacher-forced NLL of B's speech frames). The
pair counts as correct when the matched pairing is cheaper than the crossed
one: nll(Aw→Bw) + nll(Av→Bv) < nll(Aw→Bv) + nll(Av→Bw). How easy a reply is
on its own cancels out, so chance is exactly 50%.

Reported per split, with a 95% bootstrap interval over pairs:

- ``test_combo``: training voices, word–frame combinations never trained.
- ``test_voice``: voices never trained.
- ``control``: nonsense words in place of the real ones; the "matched" side
  is arbitrary, so it must stay at chance.

``by_length_gap`` splits each real split's accuracy by how many Mimi frames the
two phrases differ in. If the model read the word's length rather than the
word, pairs of equal length would fall towards chance. Any other directory in
the same format (``--splits``, e.g. live recordings) is scored the same way.
"""

import argparse
import itertools
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F

from dauys.checkpoint import load_research_checkpoint
from dauys.config import ModelConfig
from dauys.model import DauysHybrid

PAUSE = 6
LENGTH_BINS = ((0, 0), (1, 1), (2, 3), (4, 999))


def _clip(root: Path, name: str) -> tuple[torch.Tensor, torch.Tensor]:
    data = torch.load(root / name, weights_only=True)
    return data["codes"].long(), data["activity"].bool()


@torch.inference_mode()
def reply_nll(model, silence, pairs, *, batch: int = 32) -> torch.Tensor:
    """Mean CB0 + depth NLL over B's speech frames for each (A, B) clip pair."""
    out = []
    for start in range(0, len(pairs), batch):
        chunk = pairs[start : start + batch]
        frames = max(a.shape[-1] + PAUSE + b.shape[-1] for (a, _), (b, _) in chunk)
        user = silence[None, :, None].expand(len(chunk), 8, frames).clone()
        target = user.clone()
        speech = torch.zeros(len(chunk), frames, dtype=torch.bool)
        for row, ((a, _), (b, b_active)) in enumerate(chunk):
            user[row, :, : a.shape[-1]] = a
            begin = a.shape[-1] + PAUSE
            target[row, :, begin : begin + b.shape[-1]] = b
            speech[row, begin : begin + b.shape[-1]] = b_active
        initial = silence[None].expand(len(chunk), 8).clone()
        output = model(user.cuda(), target.cuda(), initial.cuda())
        target, speech = target.cuda(), speech.cuda()
        cb0 = F.cross_entropy(
            output.cb0_logits.float().transpose(1, 2), target[:, 0], reduction="none"
        )
        depth = F.cross_entropy(
            output.depth_logits.float().permute(0, 3, 1, 2),
            target[:, 1:].transpose(1, 2),
            reduction="none",
        ).mean(-1)
        total = ((cb0 + depth) * speech).sum(1) / speech.sum(1).clamp_min(1)
        out.append(total.cpu())
    return torch.cat(out)


def pair_accuracy(model, silence, quads) -> list[float]:
    """quads: (Aw, Av, Bw, Bv) clips; returns 1.0/0.0 per quad."""
    flat = []
    for a_w, a_v, b_w, b_v in quads:
        flat += [(a_w, b_w), (a_v, b_v), (a_w, b_v), (a_v, b_w)]
    nll = reply_nll(model, silence, flat).reshape(-1, 4)
    return ((nll[:, 0] + nll[:, 1]) < (nll[:, 2] + nll[:, 3])).float().tolist()


def by_length_gap(scores, quads) -> dict:
    gaps = [abs(q[0][0].shape[-1] - q[1][0].shape[-1]) for q in quads]
    out = {}
    for low, high in LENGTH_BINS:
        chosen = [s for s, g in zip(scores, gaps) if low <= g <= high]
        if chosen:
            label = f"{low}" if low == high else f"{low}-{high}"
            out[label] = {"pairs": len(chosen), "accuracy": summarize(chosen)}
    return out


def summarize(scores: list[float], resamples: int = 2000) -> list[float]:
    values = torch.tensor(scores, dtype=torch.float64)
    generator = torch.Generator().manual_seed(0)
    draws = sorted(
        float(
            values[
                torch.randint(len(values), (len(values),), generator=generator)
            ].mean()
        )
        for _ in range(resamples)
    )
    return [
        round(float(values.mean()), 4),
        round(draws[int(0.025 * resamples)], 4),
        round(draws[int(0.975 * resamples)], 4),
    ]


def real_quads(root: Path):
    rows = [
        json.loads(line) for line in (root / "manifest.jsonl").read_text().splitlines()
    ]
    groups: dict[tuple[str, str], dict[str, dict]] = {}
    for row in rows:
        groups.setdefault((row["topic"], row["speaker"]), {}).setdefault(
            row["word"], row
        )
    quads = []
    for words in groups.values():
        for w, v in itertools.combinations(sorted(words), 2):
            a_w, a_v = words[w], words[v]
            quads.append(
                tuple(
                    _clip(root, name)
                    for name in (
                        a_w["a_clip"],
                        a_v["a_clip"],
                        a_w["b_clip"],
                        a_v["b_clip"],
                    )
                )
            )
    return quads, groups


def control_quads(control_root: Path, reply_root: Path, groups, limit: int, seed: int):
    rows = [
        json.loads(line)
        for line in (control_root / "manifest.jsonl").read_text().splitlines()
    ]
    nonsense: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        nonsense.setdefault((row["topic"], row["speaker"]), []).append(row)
    replies: dict[str, list[str]] = {}
    for (topic, _), words in groups.items():
        replies.setdefault(topic, [])
        for row in words.values():
            if row["b_clip"] not in replies[topic]:
                replies[topic].append(row["b_clip"])
    rng = random.Random(seed)
    quads = []
    for (topic, _), options in nonsense.items():
        for _ in range(limit):
            n1, n2 = rng.sample(options, 2)
            b1, b2 = rng.sample(replies[topic], 2)
            quads.append(
                (
                    _clip(control_root, n1["a_clip"]),
                    _clip(control_root, n2["a_clip"]),
                    _clip(reply_root, b1),
                    _clip(reply_root, b2),
                )
            )
    return quads


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True, help="encode output dir")
    parser.add_argument("--silence-codes", type=Path, required=True)
    parser.add_argument("--checkpoints", nargs="+", required=True, help="name=path")
    parser.add_argument("--control-per-group", type=int, default=60)
    parser.add_argument("--splits", nargs="+", default=["test_combo", "test_voice"])
    parser.add_argument("--no-control", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    silence = torch.tensor(json.loads(args.silence_codes.read_text()), dtype=torch.long)
    splits = {}
    for split in args.splits:
        splits[split] = real_quads(args.data / split)
    if not args.no_control:
        splits["control"] = (
            control_quads(
                args.data / "control",
                args.data / "test_voice",
                real_quads(args.data / "test_voice")[1],
                args.control_per_group,
                seed=0,
            ),
            None,
        )
    config = ModelConfig.from_yaml(args.config)
    rows = []
    for spec in args.checkpoints:
        name, path = spec.split("=", 1)
        payload = load_research_checkpoint(Path(path))
        model = DauysHybrid(config)
        model.load_state_dict(payload["model"])
        model = model.cuda().eval()
        row = {"checkpoint": name, "step": int(payload.get("step", 0))}
        for split, (quads, groups) in splits.items():
            scores = pair_accuracy(model, silence, quads)
            row[split] = {"pairs": len(scores), "accuracy": summarize(scores)}
            if groups is not None:
                row[split]["by_length_gap"] = by_length_gap(scores, quads)
        print(json.dumps(row), flush=True)
        rows.append(row)
    args.output.write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
