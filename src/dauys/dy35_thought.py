"""DY-35: a thought level on top of the DY-32 b UnitLM (single stream, K2000, 50 Hz).

* Gist encoder: bidirectional transformer 4 x 512 over the codes of ONE utterance (own embedding table; the codes were
  made by WavLM on that utterance alone) -> mean over time -> linear -> z (64-d, L2-normalised).
* Thought before words: last-layer state at the last frame before an utterance -> MLP -> z_hat (64-d, L2). InfoNCE
  (tau 0.1) of z_hat against the true z, 15 same-speaker same-recording z and 16 z of other recordings.
* Speech from the thought: at every input frame of an utterance a linear projection of that utterance's thought is
  added to the token embedding; outside utterances a learned null vector.

``ThoughtLM`` wraps ``dy21_lm.UnitLM`` (unchanged blocks, the same JEPA objective of DY-32 b) and adds ``proj``,
``null`` and ``zhead``; the EMA target copy (JEPA) is a deep copy of the whole ThoughtLM.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn

from dauys.dy21_lm import JEPA_WEIGHT, LMConfig, UnitLM, jepa_target, rope

ZDIM = 64
TAU = 0.1
NCE_WEIGHT = 1.0


class EncBlock(nn.Module):
    """dy21_lm.Block without the causal mask, with a key padding mask."""

    def __init__(self, d: int, heads: int, ffn: int) -> None:
        super().__init__()
        self.heads = heads
        self.n1, self.n2 = nn.RMSNorm(d), nn.RMSNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.out = nn.Linear(d, d, bias=False)
        self.up = nn.Linear(d, 2 * ffn, bias=False)
        self.down = nn.Linear(ffn, d, bias=False)

    def forward(self, x: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        q, k, v = self.qkv(self.n1(x)).view(b, t, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(rope(q), rope(k), v, attn_mask=keep[:, None, None, :])
        x = x + self.out(a.transpose(1, 2).reshape(b, t, d))
        g, u = self.up(self.n2(x)).chunk(2, -1)
        return x + self.down(F.silu(g) * u)


class GistEncoder(nn.Module):
    def __init__(self, vocab: int = 2000, d: int = 512, layers: int = 4, heads: int = 8, ffn: int = 1344) -> None:
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.blocks = nn.ModuleList(EncBlock(d, heads, ffn) for _ in range(layers))
        self.norm = nn.RMSNorm(d)
        self.out = nn.Linear(d, ZDIM)
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, std=0.02)
        nn.init.zeros_(self.out.bias)

    def forward(self, codes: torch.Tensor, lens: torch.Tensor) -> torch.Tensor:
        """codes [B, L] (padding anywhere past lens), lens [B] -> z [B, 64] L2-normalised."""
        keep = torch.arange(codes.shape[1], device=codes.device)[None] < lens[:, None]
        x = self.emb(codes.clamp(0, self.emb.num_embeddings - 1))
        for blk in self.blocks:
            x = blk(x, keep)
        x = self.norm(x) * keep[..., None]
        m = x.sum(1) / lens[:, None].to(x.dtype)
        return F.normalize(self.out(m).float(), dim=-1)

    def encode_many(self, utts: list[torch.Tensor], budget: int = 65536) -> torch.Tensor:
        """Variable-length code tensors -> z [n, 64]; batched by sorted length under a token budget."""
        dev = self.emb.weight.device
        order = sorted(range(len(utts)), key=lambda i: -len(utts[i]))
        out = [None] * len(utts)
        i = 0
        while i < len(order):
            L = len(utts[order[i]])
            n = max(1, budget // max(L, 1))
            idx = order[i : i + n]
            c = torch.zeros(len(idx), L, dtype=torch.long, device=dev)
            lens = torch.tensor([len(utts[j]) for j in idx], device=dev)
            for r, j in enumerate(idx):
                c[r, : len(utts[j])] = utts[j]
            z = self.forward(c, lens)
            for r, j in enumerate(idx):
                out[j] = z[r]
            i += n
        return torch.stack(out) if out else torch.zeros(0, ZDIM, device=dev)


class ThoughtLM(nn.Module):
    def __init__(self, cfg: LMConfig, z_layers: tuple = ()) -> None:
        super().__init__()
        self.lm = UnitLM(cfg)
        # optional: the thought (projection per layer, zero-initialised) also added before these blocks
        self.z_layers = tuple(z_layers)
        self.zproj = nn.ModuleDict({str(i): nn.Linear(ZDIM, cfg.d, bias=False) for i in self.z_layers})
        for m in self.zproj.values():
            nn.init.zeros_(m.weight)
        self.cfg = cfg
        d = cfg.d
        self.proj = nn.Linear(ZDIM, d, bias=False)
        nn.init.normal_(self.proj.weight, std=0.02)
        self.null = nn.Parameter(torch.randn(d) * 0.02)
        self.zhead = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, ZDIM))
        for m in self.zhead:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                nn.init.zeros_(m.bias)

    def thought_input(self, zin: torch.Tensor | None, has: torch.Tensor | None, b: int, t: int) -> torch.Tensor:
        """zin [B, T, 64] thoughts per input frame, has [B, T] bool (False -> null)."""
        if zin is None:
            return self.null.expand(b, t, -1)
        return torch.where(has[..., None], self.proj(zin.to(self.proj.weight.dtype)), self.null.to(self.proj.weight.dtype))

    def states(self, ids: torch.Tensor, zin: torch.Tensor | None = None, has: torch.Tensor | None = None) -> torch.Tensor:
        b, t = ids.shape
        x = self.lm.emb(ids) + self.thought_input(zin, has, b, t)
        for i, blk in enumerate(self.lm.blocks):
            if i in self.z_layers and i > 0 and zin is not None:
                x = x + self.zproj[str(i)](zin.to(self.proj.weight.dtype)) * has[..., None]
            x = blk(x)
        return self.lm.norm(x)

    def zhat(self, h: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.zhead(h).float(), dim=-1)

    def logits(self, h: torch.Tensor) -> torch.Tensor:
        return self.lm.logits(h)


def nce(zh: torch.Tensor, cand: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """zh [A, 64], cand [A, C, 64] with the true one at index 0 -> InfoNCE loss, accuracy among the first 16."""
    lg = torch.einsum("ad,acd->ac", zh.float(), cand.float()) / TAU
    loss = F.cross_entropy(lg, torch.zeros(len(lg), dtype=torch.long, device=lg.device))
    acc = (lg[:, 0] > lg[:, 1:16].max(1).values).float().mean()  # strict: a tie is a miss
    return loss, acc


def lm_jepa(model: ThoughtLM, h: torch.Tensor, h_tgt: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """dy21_lm.loss on given online and target states (CE on all frames, JEPA as DY-32 b)."""
    ce = F.cross_entropy(model.logits(h).flatten(0, 1), y.flatten())
    target = jepa_target(h_tgt)
    t = target.shape[1]
    pred = F.normalize(model.lm.pred(h[:, :t]).float(), dim=-1)
    jepa = (1 - (pred * F.normalize(target, dim=-1)).sum(-1)).mean()
    return ce, jepa


def param_counts(model: ThoughtLM, enc: GistEncoder) -> dict:
    n = lambda m: sum(p.numel() for p in m.parameters())
    return {"unit_lm": n(model.lm), "gist_encoder": n(enc), "thought_proj_null": model.proj.weight.numel() + model.null.numel(),
            "zhead": n(model.zhead), "total": n(model) + n(enc)}


__all__ = ["ThoughtLM", "GistEncoder", "nce", "lm_jepa", "param_counts", "ZDIM", "TAU", "NCE_WEIGHT", "JEPA_WEIGHT",
           "LMConfig", "math"]
