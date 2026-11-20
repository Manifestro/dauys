"""DY-40: a causal LM over Mimi codes (8 codebooks x 2048, 12.5 Hz) with a depth transformer (Moshi-style RQ layout).

Layout (Moshi's acoustic delay): the acoustic codes 1..7 are shifted one frame later than the semantic code 0. A run of
n frames becomes n + 1 positions: position p holds code 0 of frame p and codes 1..7 of frame p - 1 (PAD where a frame
does not exist). So when the model draws the acoustic codes of frame f it already holds code 0 of frame f + 1.
Code ids: K (= 2048) = EOS (every level of the frame between runs, its own position, not delayed), K + 1 = PAD.

Temporal transformer (the DY-21/DY-32 b backbone, dy21_lm.Block): the input at position p is the sum of the 8 code
embeddings at p; its state h_p predicts position p + 1. Depth transformer: for position p + 1 it runs over the 8 levels;
level k gets proj(h_p) + (k > 0: embedding of level k - 1's code at p + 1) + a level embedding; its head predicts
level k at p + 1. Loss: weighted mean of the 8 CEs (PAD targets ignored) with weight 10 on code 0 and 1 on each of
codes 1..7, i.e. (10 CE_0 + sum CE_1..7) / 17.
"""

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from dauys.dy21_lm import Block

W0 = 10.0  # weight of code 0; each of codes 1..7 has weight 1 (owner, 20.11)


@dataclass
class DLConfig:
    levels: int = 8
    k: int = 2048
    d: int = 1024
    layers: int = 24
    heads: int = 16
    ffn: int = 4096
    ctx: int = 2048
    dd: int = 768
    dlayers: int = 4
    dheads: int = 12
    dffn: int = 3072

    @property
    def vocab(self) -> int:  # K codes + EOS + PAD
        return self.k + 2


def delay(frames: np.ndarray, pad: int) -> np.ndarray:
    """[n, 8] frames of one run -> [n + 1, 8]: code 0 of frame p at p, codes 1..7 of frame p at p + 1."""
    n, L = frames.shape
    out = np.full((n + 1, L), pad, frames.dtype)
    out[:n, 0] = frames[:, 0]
    out[1:, 1:] = frames[:, 1:]
    return out


def undelay(d: np.ndarray, n: int, start: int = 0) -> np.ndarray:
    """Positions of a delayed run -> frames start .. start + n - 1."""
    return np.concatenate([d[start : start + n, :1], d[start + 1 : start + n + 1, 1:]], 1)


class DepthLM(nn.Module):
    def __init__(self, cfg: DLConfig) -> None:
        super().__init__()
        self.cfg = cfg
        L, V = cfg.levels, cfg.vocab
        self.emb = nn.Embedding(L * V, cfg.d)
        self.blocks = nn.ModuleList(Block(cfg.d, cfg.heads, cfg.ffn) for _ in range(cfg.layers))
        self.norm = nn.RMSNorm(cfg.d)
        self.proj = nn.Linear(cfg.d, cfg.dd, bias=False)
        self.demb = nn.Embedding(L * V, cfg.dd)  # level k-1's code at the predicted position, per level table
        self.lemb = nn.Embedding(L, cfg.dd)
        self.dblocks = nn.ModuleList(Block(cfg.dd, cfg.dheads, cfg.dffn) for _ in range(cfg.dlayers))
        self.dnorm = nn.RMSNorm(cfg.dd)
        self.head = nn.Parameter(torch.empty(L, cfg.dd, V))
        self.register_buffer("offs", torch.arange(L) * V, persistent=False)
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, std=0.02)
        nn.init.normal_(self.head, std=0.02)
        for blk in self.blocks:
            nn.init.normal_(blk.out.weight, std=0.02 / math.sqrt(2 * cfg.layers))
            nn.init.normal_(blk.down.weight, std=0.02 / math.sqrt(2 * cfg.layers))
        for blk in self.dblocks:
            nn.init.normal_(blk.out.weight, std=0.02 / math.sqrt(2 * cfg.dlayers))
            nn.init.normal_(blk.down.weight, std=0.02 / math.sqrt(2 * cfg.dlayers))

    def states(self, ids: torch.Tensor) -> torch.Tensor:
        """ids [B, T, L] delayed positions -> [B, T, d]; state p sees positions <= p."""
        x = self.emb(ids + self.offs).sum(2)
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)

    def depth_logits(self, h: torch.Tensor, nxt: torch.Tensor) -> torch.Tensor:
        """h [N, d], nxt [N, L] codes at the predicted position (teacher forcing; levels 0..L-2 are read) ->
        logits [N, L, V] float32, level k conditioned on levels < k of that position."""
        N, L = nxt.shape
        prev = self.demb(nxt[:, :-1] + self.offs[:-1])
        x = self.proj(h)[:, None].expand(N, L, -1) + self.lemb.weight[None]
        x = torch.cat([x[:, :1], x[:, 1:] + prev], 1)
        for blk in self.dblocks:
            x = blk(x)
        x = self.dnorm(x)
        return torch.einsum("nld,ldv->nlv", x, self.head.to(x.dtype)).float()

    def parameters_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


def level_ce(model: DepthLM, ids: torch.Tensor) -> torch.Tensor:
    """ids [B, T + 1, L] delayed -> CE [B, T, L] of position p + 1 given positions <= p (PAD targets give 0 here;
    callers mask them)."""
    B, T1, L = ids.shape
    h = model.states(ids[:, :-1])
    y = ids[:, 1:]
    lg = model.depth_logits(h.reshape(-1, h.shape[-1]), y.reshape(-1, L))
    pad = model.cfg.k + 1
    return F.cross_entropy(lg.flatten(0, 1), y.reshape(-1), ignore_index=pad, reduction="none").view(B, T1 - 1, L)


def loss(model: DepthLM, ids: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
    ce = level_ce(model, ids).float()
    valid = (ids[:, 1:] != model.cfg.k + 1).float()
    per = (ce * valid).sum((0, 1)) / valid.sum((0, 1)).clamp(min=1)
    tot = (W0 * per[0] + per[1:].sum()) / (W0 + len(per) - 1)
    return tot, {"ce": tot.item(), "ce_mean": per.mean().item(), **{f"ce_l{k}": v for k, v in enumerate(per.tolist())}}


def _draw(lg: torch.Tensor, prev: torch.Tensor, mode: str, g: torch.Generator) -> torch.Tensor:
    """lg [B, V] float32 (banned ids -inf), prev [B] this level's code one position earlier. t10: temperature 1.0;
    ra08: p(prev) kept, the rest temperature 0.8 renormalised to the remaining mass (DY-33 S2)."""
    p = torch.softmax(lg, -1)
    if mode == "ra08":
        keep = p.gather(1, prev[:, None])
        q = torch.softmax(lg / 0.8, -1).scatter(1, prev[:, None], 0.0)
        q = q / q.sum(-1, keepdim=True).clamp(min=1e-12) * (1 - keep)
        p = q.scatter(1, prev[:, None], keep)
    return torch.multinomial(p, 1, generator=g)[:, 0]


@torch.no_grad()
def generate(model: DepthLM, prompts: np.ndarray, n_new: int, mode: str, seed: int, dev, trace: list | None = None) -> np.ndarray:
    """prompts [B, N, L] FRAMES (an EOS frame first, then the prompt's frames) -> [B, n_new, L] new frames.
    Works in the delayed layout: the EOS frame is one position, the prompt's n frames n + 1 positions whose last one
    holds only the acoustic codes of the last prompt frame (its code 0 slot is drawn: the first new frame's code 0).
    One extra position at the end draws the acoustic codes of the last new frame. EOS and PAD are never drawn."""
    cfg = model.cfg
    B, N, L = prompts.shape
    K, PAD = cfg.k, cfg.k + 1
    eos = prompts[:, :1]
    run = np.stack([delay(p[1:], PAD) for p in prompts])  # [B, N, L] (n = N - 1 frames -> N positions)
    seq = np.concatenate([eos, run], 1)  # [B, N + 1, L]
    P0 = seq.shape[1] - 1  # last prompt position: code 0 unknown, acoustic known
    total = P0 + n_new + 1  # positions 0 .. total - 1
    ids = torch.full((B, total, L), PAD, dtype=torch.long, device=dev)
    ids[:, : P0 + 1] = torch.from_numpy(seq.astype(np.int64)).to(dev)
    g = torch.Generator(device=dev).manual_seed(seed)
    model.eval()
    for p in range(P0, total):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h = model.states(ids[:, max(0, p - cfg.ctx) : p])[:, -1]
        cur = ids[:, p].clone()
        prev = ids[:, p - 1].clamp(max=K - 1)
        for k in range(L):
            known = (p == P0 and k > 0) or (p == total - 1 and k == 0)
            if known:
                continue  # acoustic codes of the last prompt frame / no frame after the last new one (PAD)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lg = model.depth_logits(h, cur)[:, k]
            lg[:, K:] = -float("inf")
            cur[:, k] = _draw(lg, prev[:, k], mode, g)
            if trace is not None:  # checklist: (position, level, drawn ids, their log-prob under the raw logits)
                trace.append((p, k, cur[:, k].clone(), torch.log_softmax(lg, -1).gather(1, cur[:, k : k + 1])[:, 0]))
        ids[:, p] = cur
    if trace is not None:
        trace.append(("ids", ids.clone()))
    d = ids[:, 1:].cpu().numpy()  # drop the EOS position: delayed run of n + n_new frames
    n = N - 1
    return np.stack([undelay(x, n_new, start=n) for x in d])
