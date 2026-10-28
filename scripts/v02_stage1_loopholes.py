"""Shortcut probes for Stage 1 checkpoints, run beside training.

Each probe asks whether a metric can be met without understanding A:

- L1 tail: intent selection with A cut to its last second before the pause.
- L2 voice: continuation NLL with the true A, a same-chapter A and a silent A;
  ``voice_share`` is the part of the A gain a same-voice wrong phrase keeps.
- L3 timing: activity accuracy from the cut to the B onset with true vs silent A.
- L4 empty: share of generated continuations below the energy gate.
- L5 voice quality: a few model-channel wavs per checkpoint for listening.

With ``--watch`` it scores every new ``step-*.pt`` in a directory.
"""

import argparse
import dataclasses
import importlib.util
import json
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from dauys.checkpoint import load_research_checkpoint
from dauys.config import ModelConfig
from dauys.model import DauysHybrid
from dauys.v02_stage1 import intent_accuracy, load_batch, load_stage1_corpus
from dauys.v02_stage1_eval import (
    _chapter_groups,
    continuation_waves,
    empty_rate,
    intent_selection,
    record_tensors,
    save_wavs,
)
from dauys.v03_intent import IntentConfig, IntentModule

TAIL_FRAMES = 12
ONSET_FRAMES = 12


def _train_module():
    path = Path(__file__).with_name("v02_stage1_train.py")
    spec = importlib.util.spec_from_file_location("v02_stage1_train", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _partners(records, seed: int = 5) -> list[int]:
    """A same-chapter other phrase for every record."""
    rng = random.Random(seed)
    partner = list(range(len(records)))
    for indices in _chapter_groups(records).values():
        if len(indices) < 2:
            continue
        for index in indices:
            partner[index] = rng.choice([i for i in indices if i != index])
    return partner


def _user_variant(item, other, mode: str, silence: torch.Tensor) -> torch.Tensor:
    user = silence[:, None].expand(8, item.frames).clone()
    cut = item.cut_local
    if mode == "true":
        return item.user.clone()
    if mode == "tail":
        start = max(0, cut - TAIL_FRAMES)
        user[:, start:cut] = item.user[:, start:cut]
        return user
    if mode == "swap":
        length = min(cut, other.cut_local)
        user[:, cut - length : cut] = other.user[
            :, other.cut_local - length : other.cut_local
        ]
        return user
    if mode == "silent":
        return user
    raise ValueError(mode)


@torch.inference_mode()
def continuation_scores(model, records, silence, mode: str, batch_size: int = 32):
    device = next(model.parameters()).device
    partner = _partners(records)
    items = [record_tensors(record, silence) for record in records]
    nll = onset_nll = 0.0
    speech_total = onset_total = act_correct = act_total = 0
    for start in range(0, len(items), batch_size):
        chunk = range(start, min(start + batch_size, len(items)))
        frames = max(items[i].frames for i in chunk)
        size = len(chunk)
        user = silence[None, :, None].expand(size, 8, frames).clone()
        target = user.clone()
        activity = torch.zeros((size, frames), dtype=torch.long)
        onset = torch.zeros((size, frames), dtype=torch.bool)
        window = torch.zeros((size, frames), dtype=torch.bool)
        for row, index in enumerate(chunk):
            item = items[index]
            length = item.frames
            user[row, :, :length] = _user_variant(
                item, items[partner[index]], mode, silence
            )
            target[row, :, :length] = item.target
            activity[row, :length] = item.target_activity
            speech = item.target_activity.nonzero().flatten()
            speech = speech[speech >= item.cut_local]
            if speech.numel():
                onset[row, speech[:ONSET_FRAMES]] = True
                end = int(speech[0]) + ONSET_FRAMES
                window[row, item.cut_local : min(end, length)] = True
        initial = silence[None].expand(size, 8).clone()
        output = model(user.to(device), target.to(device), initial.to(device))
        labels = target[:, 0].to(device)
        logits = output.cb0_logits.float()
        speech_mask = activity.bool().to(device)
        onset = onset.to(device)
        losses = F.cross_entropy(logits.transpose(1, 2), labels, reduction="none")
        nll += float(losses[speech_mask].sum())
        speech_total += int(speech_mask.sum())
        onset_nll += float(losses[onset].sum())
        onset_total += int(onset.sum())
        predicted = output.activity_logits.float().argmax(-1)
        window = window.to(device)
        act_correct += int(((predicted == activity.to(device)) & window).sum())
        act_total += int(window.sum())
    return {
        "nll": nll / max(1, speech_total),
        "onset_nll": onset_nll / max(1, onset_total),
        "activity_acc": act_correct / max(1, act_total),
    }


@torch.inference_mode()
def tail_intent(model, intent, records, silence, *, groups=32, candidates=8, seed=7):
    """M3 with A reduced to its last second; mirrors ``intent_selection``."""
    eligible = [
        indices
        for indices in _chapter_groups(records).values()
        if len(indices) >= candidates
    ]
    rng = random.Random(seed)
    chosen = rng.sample(eligible, min(groups, len(eligible)))
    device = next(model.parameters()).device
    scores = []
    for indices in chosen:
        pool = tuple(rng.sample(indices, candidates))
        batch = load_batch(
            records, [(pool, tuple(range(candidates)))], silence, seed=seed
        )
        frames = batch.user_codes.shape[-1]
        positions = torch.arange(frames)[None]
        keep_from = (batch.intent_frames + 1 - TAIL_FRAMES)[:, None]
        blank = (positions < keep_from)[:, None, :]
        user = torch.where(blank, silence[None, :, None], batch.user_codes)
        batch = dataclasses.replace(batch, user_codes=user)
        scores.append(intent_accuracy(model, batch.to(device), intent))
    return sum(scores) / len(scores)


def probe(checkpoint: Path, args, heldout, silence, codec) -> dict:
    payload = load_research_checkpoint(checkpoint)
    config = ModelConfig.from_yaml(args.config)
    model = DauysHybrid(config)
    model.load_state_dict(payload["model"])
    device = torch.device("cuda")
    model = model.to(device).eval()
    intent = IntentModule(model, IntentConfig(horizon=12, weight=0.5)).to(device)
    intent.load_state_dict(payload["intent"])
    intent.eval()
    step = int(payload["step"])
    out: dict = {"step": step}
    variants = {
        m: continuation_scores(model, heldout, silence, m)
        for m in ("true", "swap", "silent", "tail")
    }
    for mode, scores in variants.items():
        for key, value in scores.items():
            out[f"{mode}/{key}"] = round(value, 4)
    gain = variants["silent"]["nll"] - variants["true"]["nll"]
    kept = variants["silent"]["nll"] - variants["swap"]["nll"]
    out["L2_a_gain_nll"] = round(gain, 4)
    out["L2_voice_share"] = round(kept / gain, 3) if gain > 1e-3 else None
    onset_gain = variants["silent"]["onset_nll"] - variants["true"]["onset_nll"]
    onset_kept = variants["silent"]["onset_nll"] - variants["swap"]["onset_nll"]
    out["L2_onset_voice_share"] = (
        round(onset_kept / onset_gain, 3) if onset_gain > 1e-3 else None
    )
    out["L3_activity_gain"] = round(
        variants["true"]["activity_acc"] - variants["silent"]["activity_acc"], 4
    )
    out["L1_m3_full"] = round(intent_selection(model, intent, heldout, silence), 4)
    out["L1_m3_tail"] = round(tail_intent(model, intent, heldout, silence), 4)
    with torch.inference_mode():
        waves = continuation_waves(model, heldout, silence, codec, items=args.items)
    out["L4_empty_rate"] = round(empty_rate(waves, None), 3)
    save_wavs(waves[:3], args.output / "wavs" / f"step-{step:05d}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--heldout-root", type=Path, required=True)
    parser.add_argument("--heldout-manifest", type=Path, required=True)
    parser.add_argument("--silence-codes", type=Path, required=True)
    parser.add_argument("--mimi-checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--items", type=int, default=24)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--stop-file", type=Path)
    args = parser.parse_args()

    from dauys.codec import load_mimi

    train = _train_module()
    heldout = train._grouped(
        load_stage1_corpus(args.heldout_root, args.heldout_manifest),
        per_group=8,
        limit=256,
    )
    silence = train._silence(args.silence_codes)
    codec = load_mimi(args.mimi_checkpoint, "cuda")
    args.output.mkdir(parents=True, exist_ok=True)
    results = args.output / "loopholes.jsonl"
    done = set()
    if results.exists():
        done = {json.loads(line)["file"] for line in results.read_text().splitlines()}
    while True:
        for path in sorted(args.checkpoints.glob("step-*.pt")):
            if path.name in done:
                continue
            row = probe(path, args, heldout, silence, codec)
            row["file"] = path.name
            with results.open("a") as stream:
                stream.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)
            done.add(path.name)
            torch.cuda.empty_cache()
        if not args.watch or (args.stop_file and args.stop_file.exists()):
            break
        time.sleep(30)


if __name__ == "__main__":
    main()
