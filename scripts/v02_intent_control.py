"""Can a dedicated classifier pick B's continuation from A's audio at all?

The control for M3. The task is M3's: given the speech before the cut (A),
pick the true first ~2.5 s of B (``continuation_window``) among eight
continuations from the same speaker and chapter, so voice, room and microphone
cannot separate them. A dual encoder is trained on train phrases and scored on
held-out speakers; its accuracy bounds what an ear could reach on this data.

Variants (same phrases, same A frames):

- ``hubert``: HuBERT base features (self-supervised, audio only) of A and of
  each candidate, each decoded from Mimi on its own so B never leaks into A;
  a bidirectional encoder. The bound with strong hearing.
- ``codes``: Mimi code embeddings learned from scratch, bidirectional over A.
  How much content the codes carry.
- ``causal``: as ``codes``, but A is read causally at its last frame, as the
  ear and intent head do. What reading without the future costs.

Each variant reports accuracy on all held-out chapter pools, on the exact M3
pools (seed 7, 32 groups), and with A cut to its last 12 frames (the L1 tail).
"""

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from dauys.v02_stage1 import (
    INTENT_FRAMES,
    _load_tokens,
    continuation_window,
    load_stage1_corpus,
)
from dauys.v02_stage1_eval import _chapter_groups

POOL = 8
TAIL_FRAMES = 12
M3_GROUPS = 32
M3_SEED = 7


def pick_grouped(records, limit: int, seed: int):
    """Whole speaker-chapter groups of at least POOL phrases, up to ``limit``."""
    groups = [g for g in _chapter_groups(records).values() if len(g) >= POOL]
    random.Random(seed).shuffle(groups)
    chosen = []
    for indices in groups:
        if len(chosen) >= limit:
            break
        chosen.extend(indices)
    return [records[i] for i in sorted(chosen)]


def load_codes(records, context: int, silence: torch.Tensor):
    """A codes right-aligned at the cut and B's continuation window."""
    count = len(records)
    a_codes = silence[None, :, None].expand(count, 8, context).clone()
    a_mask = torch.zeros(count, context, dtype=torch.bool)
    b_codes = torch.zeros(count, 8, INTENT_FRAMES, dtype=torch.long)
    b_mask = torch.zeros(count, INTENT_FRAMES, dtype=torch.bool)
    for index, record in enumerate(records):
        cached = _load_tokens(record.token_path)
        cut = record.cut_frame
        a = cached["user"][:, max(0, cut - context) : cut]
        a_codes[index, :, context - a.shape[-1] :] = a
        a_mask[index, context - a.shape[-1] :] = True
        window, mask = continuation_window(
            cached["target"], cached["target_activity"], cut
        )
        b_codes[index], b_mask[index] = window, mask
    return {
        "a_codes": a_codes,
        "a_mask": a_mask,
        "b_codes": b_codes,
        "b_mask": b_mask,
        "silence": silence,
    }


@torch.inference_mode()
def _hubert_batch(codes, codec, hubert, layer: int) -> torch.Tensor:
    """[batch, 8, frames] codes → [batch, frames, 768] pooled HuBERT features."""
    from scipy.signal import resample_poly

    from dauys.codec import decode

    pcm = decode(codec, codes.cuda()).float()[:, 0].cpu().numpy()
    wav = torch.from_numpy(resample_poly(pcm, 2, 3, axis=-1)).float().cuda()
    wav = (wav - wav.mean(-1, keepdim=True)) / wav.std(-1, keepdim=True).clamp_min(1e-5)
    hidden = hubert(wav, output_hidden_states=True).hidden_states[layer]
    frames = codes.shape[-1]
    need = frames * 4
    if hidden.shape[1] < need:
        pad = hidden[:, -1:].expand(-1, need - hidden.shape[1], -1)
        hidden = torch.cat((hidden, pad), 1)
    return hidden[:, :need].reshape(len(codes), frames, 4, -1).mean(2).half()


def add_hubert(data, codec, hubert, layer: int, batch: int = 32) -> None:
    """HuBERT features for A (each length bucket decoded alone) and B windows."""
    a_codes, a_mask = data["a_codes"], data["a_mask"]
    count, _, context = a_codes.shape
    lengths = a_mask.sum(1)
    a_feats = torch.zeros(count, context, 768, dtype=torch.half)
    order = lengths.argsort()
    for chunk in order.split(batch):
        # Crop the chunk to its shortest A so no silence padding is decoded.
        take = int(lengths[chunk].min())
        feats = _hubert_batch(a_codes[chunk, :, context - take :], codec, hubert, layer)
        a_feats[chunk, context - take :] = feats.cpu()
        a_mask[chunk] = False
        a_mask[chunk, context - take :] = True
    b_feats = torch.cat(
        [
            _hubert_batch(chunk, codec, hubert, layer).cpu()
            for chunk in data["b_codes"].split(batch)
        ]
    )
    data["a_feats"], data["b_feats"] = a_feats, b_feats


class Encoder(nn.Module):
    def __init__(self, features: str, width: int, layers: int, causal: bool):
        super().__init__()
        self.features, self.causal = features, causal
        if features == "hubert":
            self.input = nn.Linear(768, width)
        else:
            self.input = nn.ModuleList(nn.Embedding(2048, width) for _ in range(8))
        self.position = nn.Embedding(512, width)
        layer = nn.TransformerEncoderLayer(
            width, 4, 4 * width, dropout=0.1, batch_first=True, norm_first=True
        )
        self.blocks = nn.TransformerEncoder(layer, layers)
        self.norm = nn.LayerNorm(width)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if self.features == "hubert":
            h = self.input(x.float())
        else:
            h = sum(self.input[q](x[:, q]) for q in range(8)) / math.sqrt(8)
        frames = h.shape[1]
        h = h + self.position(torch.arange(frames, device=h.device))
        if self.causal:
            # Like the ear: no padding mask, the frames before A are silence
            # codes, and A is read at its last frame.
            attention = torch.triu(
                torch.ones(frames, frames, dtype=torch.bool, device=h.device), 1
            )
            return self.norm(self.blocks(h, mask=attention))[:, -1]
        # A fully masked row would give NaN; keep at least the last frame.
        padding = ~mask
        padding[:, -1] = False
        h = self.norm(self.blocks(h, src_key_padding_mask=padding))
        weights = mask.float()
        return (h * weights[..., None]).sum(1) / weights.sum(1, keepdim=True).clamp_min(
            1
        )


class DualEncoder(nn.Module):
    def __init__(self, variant: str, width: int = 256, layers: int = 4):
        super().__init__()
        features = "hubert" if variant == "hubert" else "codes"
        self.features = features
        self.a = Encoder(features, width, layers, causal=variant == "causal")
        self.b = Encoder(features, width, layers, causal=False)
        self.scale = nn.Parameter(torch.tensor(math.log(10.0)))

    def inputs(self, data, rows, side: str):
        key = f"{side}_feats" if self.features == "hubert" else f"{side}_codes"
        return data[key][rows], data[f"{side}_mask"][rows]

    def scores(self, data, pools: torch.Tensor, tail: int | None = None):
        """pools: [groups, POOL] item rows → [groups, POOL, POOL] logits."""
        device = self.scale.device
        rows = pools.reshape(-1)
        a, a_mask = (t.to(device) for t in self.inputs(data, rows, "a"))
        if tail is not None:
            a_mask = a_mask.clone()
            a_mask[:, : a_mask.shape[1] - tail] = False
            if self.a.causal:
                silence = data["silence"].to(device)[None, :, None]
                a = torch.where(a_mask[:, None], a, silence)
        b, b_mask = (t.to(device) for t in self.inputs(data, rows, "b"))
        za = F.normalize(self.a(a, a_mask), dim=-1).reshape(*pools.shape, -1)
        zb = F.normalize(self.b(b, b_mask), dim=-1).reshape(*pools.shape, -1)
        return self.scale.exp() * za @ zb.transpose(1, 2)


def sample_pools(groups, count: int, rng: random.Random) -> torch.Tensor:
    chosen = [rng.sample(rng.choice(groups), POOL) for _ in range(count)]
    return torch.tensor(chosen)


def _accuracy(model, data, pools: torch.Tensor, tail=None, batch: int = 32) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for chunk in pools.split(batch):
            logits = model.scores(data, chunk, tail)
            target = torch.arange(POOL, device=logits.device).expand(len(chunk), -1)
            correct += int((logits.argmax(-1) == target).sum())
            total += target.numel()
    model.train()
    return correct / total


def train_variant(
    variant, train, train_groups, heldout, heldout_pools, *, steps, batch, seed, log
):
    torch.manual_seed(seed)
    rng = random.Random(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DualEncoder(variant).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimiser,
        lambda step: (
            min(1.0, (step + 1) / 200)
            * 0.5
            * (1 + math.cos(math.pi * min(step, steps) / steps))
        ),
    )
    history = []
    for step in range(1, steps + 1):
        pools = sample_pools(train_groups, batch, rng)
        logits = model.scores(train, pools)
        target = torch.arange(POOL, device=logits.device).expand(batch, -1)
        loss = F.cross_entropy(logits.reshape(-1, POOL), target.reshape(-1))
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimiser.step()
        schedule.step()
        if step % max(1, steps // 8) == 0 or step == steps:
            row = {
                "variant": variant,
                "step": step,
                "loss": round(loss.item(), 4),
                "heldout": round(_accuracy(model, heldout, heldout_pools["all"]), 4),
            }
            log(row)
            history.append(row)
    train_pools = sample_pools(train_groups, 512, random.Random(seed + 1))
    result = {
        "variant": variant,
        "train_acc": round(_accuracy(model, train, train_pools), 4),
        "heldout_all": round(_accuracy(model, heldout, heldout_pools["all"]), 4),
        "heldout_m3_pools": round(_accuracy(model, heldout, heldout_pools["m3"]), 4),
        "heldout_tail12": round(
            _accuracy(model, heldout, heldout_pools["all"], TAIL_FRAMES), 4
        ),
        "history": history,
    }
    return result


def heldout_pools(records, draws: int, seed: int, m3_records: int = 256):
    """All eligible chapter groups (``draws`` pools each) and the exact M3 pools."""
    eligible = [g for g in _chapter_groups(records).values() if len(g) >= POOL]
    rng = random.Random(seed)
    every = [rng.sample(g, POOL) for g in eligible for _ in range(draws)]
    # M3 runs on the training script's ``_grouped`` subset (the first POOL
    # phrases of each chapter, up to 256) with intent_selection's draw order.
    subset = []
    for indices in _chapter_groups(records).values():
        if len(indices) >= POOL and len(subset) < m3_records:
            subset.extend(indices[:POOL])
    sub_eligible = [
        [subset[i] for i in g]
        for g in _chapter_groups([records[i] for i in subset]).values()
        if len(g) >= POOL
    ]
    rng = random.Random(M3_SEED)
    chosen = rng.sample(sub_eligible, min(M3_GROUPS, len(sub_eligible)))
    m3 = [rng.sample(indices, POOL) for indices in chosen]
    return {"all": torch.tensor(every), "m3": torch.tensor(m3)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-root", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--heldout-root", type=Path, required=True)
    parser.add_argument("--heldout-manifest", type=Path, required=True)
    parser.add_argument("--silence-codes", type=Path, required=True)
    parser.add_argument("--mimi-checkpoint", type=Path, required=True)
    parser.add_argument("--variants", nargs="+", default=["hubert", "codes", "causal"])
    parser.add_argument("--train-items", type=int, default=30000)
    parser.add_argument("--context", type=int, default=192)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--batch", type=int, default=32, help="pools per step")
    parser.add_argument("--draws", type=int, default=4)
    parser.add_argument("--hubert", default="facebook/hubert-base-ls960")
    parser.add_argument("--layer", type=int, default=9)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if "ft" in args.hubert.split("/")[-1]:
        raise ValueError("Use a self-supervised HuBERT, not a text-finetuned one")
    started = time.time()

    def log(row):
        row = {**row, "seconds": round(time.time() - started)}
        print(json.dumps(row), flush=True)

    silence = torch.tensor(json.loads(args.silence_codes.read_text()), dtype=torch.long)
    train_records = pick_grouped(
        load_stage1_corpus(args.train_root, args.train_manifest, verify_hashes=False),
        args.train_items,
        args.seed,
    )
    heldout_records = load_stage1_corpus(args.heldout_root, args.heldout_manifest)
    overlap = {r.speaker for r in train_records} & {r.speaker for r in heldout_records}
    if overlap:
        raise ValueError(f"Train and held-out speakers overlap: {sorted(overlap)[:5]}")
    train = load_codes(train_records, args.context, silence)
    heldout = load_codes(heldout_records, args.context, silence)
    train_groups = [
        g for g in _chapter_groups(train_records).values() if len(g) >= POOL
    ]
    pools = heldout_pools(heldout_records, args.draws, args.seed)
    log(
        {
            "train_items": len(train_records),
            "train_groups": len(train_groups),
            "heldout_items": len(heldout_records),
            "heldout_pools": len(pools["all"]),
            "m3_pools": len(pools["m3"]),
        }
    )

    if "hubert" in args.variants:
        from transformers import HubertModel

        from dauys.codec import load_mimi

        codec = load_mimi(args.mimi_checkpoint, "cuda")
        hubert = HubertModel.from_pretrained(args.hubert).cuda().eval()
        for data in (train, heldout):
            add_hubert(data, codec, hubert, args.layer)
        del codec, hubert
        torch.cuda.empty_cache()
        log({"hubert": "done"})

    results = []
    for variant in args.variants:
        result = train_variant(
            variant,
            train,
            train_groups,
            heldout,
            pools,
            steps=args.steps,
            batch=args.batch,
            seed=args.seed,
            log=log,
        )
        log({key: value for key, value in result.items() if key != "history"})
        results.append(result)
    summary = {
        "task": "pick B's continuation among 8 same-chapter candidates from A",
        "chance": 1 / POOL,
        "train_items": len(train_records),
        "heldout_items": len(heldout_records),
        "context_frames": args.context,
        "steps": args.steps,
        "seconds": round(time.time() - started),
        "results": results,
    }
    args.output.write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
