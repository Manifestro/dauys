"""DY-21: a causal language model over DY-14 syllable units of one speech stream, with a JEPA objective.

Tokens are the units of the 10 Hz grid with "continue" and "silence" removed (GSLM-style deduplicated
units), ``EOS = K`` between pieces. The model predicts the next unit (cross-entropy) and, as JEPA,
the mean of an EMA copy's normalised states over the next ``JEPA_AHEAD`` positions (cosine loss through
a small predictor). The EMA copy is never trained by gradients.
"""

import copy
import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

JEPA_AHEAD = 8
JEPA_WEIGHT = 0.5


@dataclass
class LMConfig:
    vocab: int = 8193  # K units + EOS
    d: int = 1024
    layers: int = 24
    heads: int = 16
    ffn: int = 4096
    ctx: int = 1024


def rope(x: torch.Tensor, base: float = 10000.0) -> torch.Tensor:
    b, h, t, d = x.shape
    half = d // 2
    freq = base ** (-torch.arange(half, device=x.device, dtype=torch.float32) / half)
    ang = torch.arange(t, device=x.device, dtype=torch.float32)[:, None] * freq[None]
    cos, sin = ang.cos().to(x.dtype), ang.sin().to(x.dtype)
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1)


class Block(nn.Module):
    def __init__(self, d: int, heads: int, ffn: int) -> None:
        super().__init__()
        self.heads = heads
        self.n1, self.n2 = nn.RMSNorm(d), nn.RMSNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.out = nn.Linear(d, d, bias=False)
        self.up = nn.Linear(d, 2 * ffn, bias=False)
        self.down = nn.Linear(ffn, d, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        q, k, v = self.qkv(self.n1(x)).view(b, t, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(rope(q), rope(k), v, is_causal=True)
        x = x + self.out(a.transpose(1, 2).reshape(b, t, d))
        g, u = self.up(self.n2(x)).chunk(2, -1)
        return x + self.down(F.silu(g) * u)


class UnitLM(nn.Module):
    def __init__(self, cfg: LMConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.emb = nn.Embedding(cfg.vocab, cfg.d)
        self.blocks = nn.ModuleList(Block(cfg.d, cfg.heads, cfg.ffn) for _ in range(cfg.layers))
        self.norm = nn.RMSNorm(cfg.d)
        self.head = nn.Linear(cfg.d, cfg.vocab, bias=False)
        self.pred = nn.Sequential(nn.Linear(cfg.d, cfg.d), nn.GELU(), nn.Linear(cfg.d, cfg.d))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)
        for blk in self.blocks:
            nn.init.normal_(blk.out.weight, std=0.02 / math.sqrt(2 * cfg.layers))
            nn.init.normal_(blk.down.weight, std=0.02 / math.sqrt(2 * cfg.layers))

    def states(self, ids: torch.Tensor) -> torch.Tensor:
        x = self.emb(ids)
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)

    def logits(self, h: torch.Tensor) -> torch.Tensor:
        return self.head(h).float()

    def parameters_count(self) -> int:
        return sum(p.numel() for n, p in self.named_parameters())


def make_target(model: UnitLM) -> UnitLM:
    tgt = copy.deepcopy(model)
    for p in tgt.parameters():
        p.requires_grad_(False)
    return tgt


@torch.no_grad()
def ema_update(tgt: UnitLM, model: UnitLM, decay: float) -> None:
    for pt, p in zip(tgt.parameters(), model.parameters()):
        pt.lerp_(p.detach(), 1 - decay)


def jepa_target(h_tgt: torch.Tensor) -> torch.Tensor:
    """Mean of the normalised target states over the next JEPA_AHEAD positions: [B, T - JEPA_AHEAD, d]."""
    z = F.normalize(h_tgt.float(), dim=-1)
    c = torch.cat([torch.zeros_like(z[:, :1]), z.cumsum(1)], 1)
    t = z.shape[1] - JEPA_AHEAD
    return (c[:, 1 + JEPA_AHEAD : 1 + JEPA_AHEAD + t] - c[:, 1 : 1 + t]) / JEPA_AHEAD


def loss(model: UnitLM, tgt: UnitLM, ids: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
    """ids [B, T + 1]: next-unit CE on ids[:, 1:] and JEPA on the first T - JEPA_AHEAD positions."""
    x, y = ids[:, :-1], ids[:, 1:]
    h = model.states(x)
    ce = F.cross_entropy(model.logits(h).flatten(0, 1), y.flatten())
    with torch.no_grad():
        target = jepa_target(tgt.states(x))
    t = target.shape[1]
    pred = F.normalize(model.pred(h[:, :t]).float(), dim=-1)
    jepa = (1 - (pred * F.normalize(target, dim=-1)).sum(-1)).mean()
    total = ce + JEPA_WEIGHT * jepa
    return total, {"ce": ce.item(), "jepa": jepa.item(), "loss": total.item()}
