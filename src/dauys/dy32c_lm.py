"""DY-32 track C: a causal LM over several parallel streams of codes per 20 ms frame (delay pattern).

Each frame has S codes (one per stream, K values each). Stream k of the model is delayed by k positions
(MusicGen-style delay pattern): at position t the model holds code [t - k] of stream k, so when it predicts stream k
of a frame it has already seen streams 0..k-1 of that frame. The input at a position is the mean of the S stream
embeddings (GSLM2-style); one next-code head per stream; the backbone is the DY-21 422M transformer (dy21_lm.Block).
Per stream, ids K = EOS (between pieces) and K + 1 = PAD (the delay gap before the first code).
"""

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from dauys.dy21_lm import Block


@dataclass
class MSConfig:
    streams: int = 6
    k: int = 1000
    d: int = 1024
    layers: int = 24
    heads: int = 16
    ffn: int = 4096
    ctx: int = 2048

    @property
    def vocab(self) -> int:  # per stream: K codes + EOS + PAD
        return self.k + 2


class MultiStreamLM(nn.Module):
    def __init__(self, cfg: MSConfig) -> None:
        super().__init__()
        self.cfg = cfg
        S, V = cfg.streams, cfg.vocab
        self.emb = nn.Embedding(S * V, cfg.d)
        self.blocks = nn.ModuleList(Block(cfg.d, cfg.heads, cfg.ffn) for _ in range(cfg.layers))
        self.norm = nn.RMSNorm(cfg.d)
        self.head = nn.Linear(cfg.d, S * V, bias=False)
        self.register_buffer("offs", torch.arange(S) * V, persistent=False)
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, std=0.02)
        for blk in self.blocks:
            nn.init.normal_(blk.out.weight, std=0.02 / math.sqrt(2 * cfg.layers))
            nn.init.normal_(blk.down.weight, std=0.02 / math.sqrt(2 * cfg.layers))

    def states(self, ids: torch.Tensor) -> torch.Tensor:
        """ids [B, T, S] (delayed) -> [B, T, d]."""
        x = self.emb(ids + self.offs).mean(2)
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)

    def logits(self, h: torch.Tensor) -> torch.Tensor:
        """[..., d] -> [..., S, V] float32."""
        return self.head(h).float().unflatten(-1, (self.cfg.streams, self.cfg.vocab))

    def parameters_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


def delay(x: np.ndarray, pad: int) -> np.ndarray:
    """[N, S] frames -> [N + S - 1, S] with stream k shifted down by k (PAD before and after)."""
    n, S = x.shape
    out = np.full((n + S - 1, S), pad, x.dtype)
    for k in range(S):
        out[k : k + n, k] = x[:, k]
    return out


def undelay(d: np.ndarray, n: int) -> np.ndarray:
    """Inverse of delay for the first n frames."""
    S = d.shape[1]
    return np.stack([d[k : k + n, k] for k in range(S)], 1)


def loss(model: MultiStreamLM, ids: torch.Tensor, pad: int) -> tuple[torch.Tensor, dict[str, float]]:
    """ids [B, T + 1, S]: next-position CE for every stream (PAD targets ignored), mean over streams."""
    x, y = ids[:, :-1], ids[:, 1:]
    lg = model.logits(model.states(x))  # [B, T, S, V]
    ce = F.cross_entropy(lg.flatten(0, 2), y.flatten(), ignore_index=pad, reduction="none").view_as(y).float()
    valid = (y != pad).float()
    per = (ce * valid).sum((0, 1)) / valid.sum((0, 1)).clamp(min=1)
    tot = per.mean()
    return tot, {"ce": tot.item(), **{f"ce_s{k}": v for k, v in enumerate(per.tolist())}}


def sample_next(lg: torch.Tensor, temp: float, top_p: float, g: torch.Generator) -> torch.Tensor:
    """lg [N, V] -> [N] nucleus sample."""
    p = torch.softmax(lg / temp, -1)
    sp, si = p.sort(-1, descending=True)
    keep = sp.cumsum(-1) - sp < top_p
    sp = sp * keep
    sp = sp / sp.sum(-1, keepdim=True)
    j = torch.multinomial(sp, 1, generator=g)
    return si.gather(-1, j)[:, 0]


@torch.no_grad()
def generate(model: MultiStreamLM, prompts: np.ndarray, n_new: int, temp: float, top_p: float, seed: int,
             device) -> np.ndarray:
    """prompts [B, N, S] frames (same N; the caller puts an EOS frame first) -> [B, n_new, S] new frames.
    Positions are generated in the delayed layout; a stream's code of a frame still inside the prompt is forced."""
    cfg = model.cfg
    B, N, S = prompts.shape
    K, PAD = cfg.k, cfg.k + 1
    total = N + n_new
    full = np.full((B, total, S), PAD, np.int64)
    full[:, :N] = prompts
    d = np.stack([delay(f, PAD) for f in full])  # [B, total + S - 1, S]
    known = np.zeros_like(d, bool)
    for k in range(S):
        known[:, k : k + N, k] = True
    ids = torch.from_numpy(d).to(device)
    kn = torch.from_numpy(known).to(device)
    g = torch.Generator(device=device).manual_seed(seed)
    model.eval()
    for t in range(N, total + S - 1):  # position t is predicted from positions < t
        lo = max(0, t - cfg.ctx)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h = model.states(ids[:, lo:t])
        lg = model.logits(h[:, -1])  # [B, S, V]
        lg[..., K:] = -float("inf")  # never EOS or PAD inside the continuation
        nxt = sample_next(lg.flatten(0, 1), temp, top_p, g).view(B, S)
        for k in range(S):
            frame = t - k
            if frame < 0 or frame >= total:
                nxt[:, k] = PAD
        ids[:, t] = torch.where(kn[:, t], ids[:, t], nxt)
    out = ids.cpu().numpy()
    return np.stack([undelay(o, total)[N:] for o in out])
