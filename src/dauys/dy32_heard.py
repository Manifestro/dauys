"""DY-32 track D: "do not code what the model hears".

One causal transformer (the DY-21 UnitLM size, 422M) over 50 Hz WavLM-Large layer-6 frames. A window is split at a
point: before it is the HEARD part, after it the OWN part. The own part is always discrete units (nearest centroid of a
codebook), fed back as unit embeddings and predicted with cross-entropy. The heard part has no loss and comes in one of
two modes, marked by a mode embedding:
  mode 0 (continuous): the frozen WavLM features themselves, LayerNorm + linear projection, no quantisation;
  mode 1 (discrete):   the same frames as unit embeddings.
A role embedding marks heard vs own positions. Both modes are trained in every batch (half of the examples each), so
one checkpoint can be evaluated both ways.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

FEAT = 1024  # WavLM-Large width
HOP = 320  # 16 kHz samples per 50 Hz frame
SR = 16000


@dataclass
class HeardConfig:
    units: int = 2000
    d: int = 1024
    layers: int = 24
    heads: int = 16
    ffn: int = 4096
    ctx: int = 1280


def rope(x: torch.Tensor, offset: int = 0, base: float = 10000.0) -> torch.Tensor:
    b, h, t, d = x.shape
    half = d // 2
    freq = base ** (-torch.arange(half, device=x.device, dtype=torch.float32) / half)
    ang = torch.arange(offset, offset + t, device=x.device, dtype=torch.float32)[:, None] * freq[None]
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

    def forward(self, x: torch.Tensor, cache: dict | None = None) -> torch.Tensor:
        b, t, d = x.shape
        q, k, v = self.qkv(self.n1(x)).view(b, t, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        off = 0 if cache is None or "k" not in cache else cache["k"].shape[2]
        q, k = rope(q, off), rope(k, off)
        if cache is not None:
            if "k" in cache:
                k, v = torch.cat([cache["k"], k], 2), torch.cat([cache["v"], v], 2)
            cache["k"], cache["v"] = k, v
        a = F.scaled_dot_product_attention(q, k, v, is_causal=(off == 0 and t > 1))
        x = x + self.out(a.transpose(1, 2).reshape(b, t, d))
        g, u = self.up(self.n2(x)).chunk(2, -1)
        return x + self.down(F.silu(g) * u)


class HeardLM(nn.Module):
    def __init__(self, cfg: HeardConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.emb = nn.Embedding(cfg.units, cfg.d)
        self.feat_norm = nn.LayerNorm(FEAT)
        self.feat_proj = nn.Linear(FEAT, cfg.d)
        self.mode_emb = nn.Embedding(2, cfg.d)
        self.role_emb = nn.Embedding(2, cfg.d)  # 0 heard, 1 own
        self.blocks = nn.ModuleList(Block(cfg.d, cfg.heads, cfg.ffn) for _ in range(cfg.layers))
        self.norm = nn.RMSNorm(cfg.d)
        self.head = nn.Linear(cfg.d, cfg.units, bias=False)
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, std=0.02)
                if getattr(m, "bias", None) is not None:
                    nn.init.zeros_(m.bias)
        for blk in self.blocks:
            nn.init.normal_(blk.out.weight, std=0.02 / math.sqrt(2 * cfg.layers))
            nn.init.normal_(blk.down.weight, std=0.02 / math.sqrt(2 * cfg.layers))

    def embed(self, ids: torch.Tensor, feats: torch.Tensor | None, heard: torch.Tensor, mode: torch.Tensor) -> torch.Tensor:
        """ids [B, T] unit ids; feats [B, T, FEAT] or None; heard [B, T] bool; mode [B] (0 continuous, 1 discrete)."""
        x = self.emb(ids)
        if feats is not None:
            f = self.feat_proj(self.feat_norm(feats.to(self.feat_proj.weight.dtype)))
            use = (heard & (mode[:, None] == 0))[..., None]
            x = torch.where(use, f.to(x.dtype), x)
        return x + self.mode_emb(mode)[:, None] + self.role_emb((~heard).long())

    def states(self, x: torch.Tensor, caches: list | None = None) -> torch.Tensor:
        for i, blk in enumerate(self.blocks):
            x = blk(x, None if caches is None else caches[i])
        return self.norm(x)

    def logits(self, h: torch.Tensor) -> torch.Tensor:
        return self.head(h).float()

    def parameters_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


def window_loss(model: HeardLM, ids: torch.Tensor, feats: torch.Tensor, split: torch.Tensor, mode: torch.Tensor,
                reduction: str = "mean") -> tuple[torch.Tensor, torch.Tensor]:
    """ids [B, T+1], feats [B, T+1, FEAT], split [B] = index of the first own frame (1 <= split <= T).
    Inputs are positions 0..T-1; CE only where the target ids[t+1] is an own frame (t + 1 >= split).
    Returns (loss sum or mean, number of target positions)."""
    B, T1 = ids.shape
    T = T1 - 1
    pos = torch.arange(T, device=ids.device)[None]
    heard = pos < split[:, None]
    x = model.embed(ids[:, :T], feats[:, :T], heard, mode)
    h = model.states(x)
    mask = (pos + 1) >= split[:, None]
    lg = model.logits(h[mask])
    ce = F.cross_entropy(lg, ids[:, 1:][mask], reduction="sum")
    n = mask.sum()
    return (ce / n if reduction == "mean" else ce), n


@torch.no_grad()
def generate(model: HeardLM, ids: torch.Tensor, feats: torch.Tensor | None, mode: int, n_new: int, temp: float = 0.8,
             top_p: float = 0.95, seed: int = 1234) -> torch.Tensor:
    """Prompt = all P frames heard (ids [B, P], feats [B, P, FEAT]); continue with n_new own units. KV cache."""
    B, P = ids.shape
    dev = ids.device
    g = torch.Generator(device=dev).manual_seed(seed)
    m = torch.full((B,), mode, device=dev, dtype=torch.long)
    caches = [{} for _ in model.blocks]
    x = model.embed(ids, feats, torch.ones(B, P, dtype=torch.bool, device=dev), m)
    h = model.states(x, caches)[:, -1]
    out = []
    own = torch.zeros(B, 1, dtype=torch.bool, device=dev)
    for _ in range(n_new):
        p = torch.softmax(model.logits(h) / temp, -1)
        sp, si = p.sort(-1, descending=True)
        cut = sp.cumsum(-1) - sp > top_p
        sp = sp.masked_fill(cut, 0)
        nxt = si.gather(1, torch.multinomial(sp, 1, generator=g))
        out.append(nxt)
        x = model.embed(nxt, None, own, m)
        h = model.states(x, caches)[:, -1]
    return torch.cat(out, 1)
