"""DY-14 tower: a causal transformer over both channels' syllable units, with three heads.

Each 10 Hz frame enters as the sum of the self unit embedding, the partner unit
embedding and a projection of both channels' prosody. No audio token enters, and no
head predicts acoustics. Heads:

* partner latent (main): a predictor maps the state at ``t`` to 256 numbers that must
  match a separate target encoder's summary of the partner's next units (InfoNCE
  against the same partner elsewhere in the conversation and the batch);
* own next unit: cross-entropy on the next self unit within 2 s;
* who and when: both channels' activity over the next second.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .dy14_data import MAX_UNITS, WHO_AHEAD


@dataclass
class TowerConfig:
    k: int = 8192
    d: int = 768
    layers: int = 12
    heads: int = 12
    ffn: int = 3072
    latent: int = 256
    target_d: int = 256
    target_layers: int = 2
    ctx: int = 600


def rope(x: torch.Tensor, base: float = 10000.0) -> torch.Tensor:
    _, _, t, dh = x.shape
    half = dh // 2
    freq = base ** (-torch.arange(half, device=x.device, dtype=torch.float32) / half)
    ang = torch.arange(t, device=x.device, dtype=torch.float32)[:, None] * freq[None]
    cos, sin = ang.cos().to(x.dtype), ang.sin().to(x.dtype)
    a, b = x[..., :half], x[..., half:]
    return torch.cat([a * cos - b * sin, a * sin + b * cos], -1)


def top1(scores: torch.Tensor) -> torch.Tensor:
    """True window strictly above every other; ties and NaN count as wrong."""
    return scores[:, 0] > scores[:, 1:].max(1).values


class Block(nn.Module):
    def __init__(self, d: int, heads: int, ffn: int) -> None:
        super().__init__()
        self.heads = heads
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.out = nn.Linear(d, d, bias=False)
        self.ff = nn.Sequential(nn.Linear(d, ffn), nn.GELU(), nn.Linear(ffn, d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        q, k, v = (
            self.qkv(self.n1(x))
            .view(b, t, 3, self.heads, d // self.heads)
            .permute(2, 0, 3, 1, 4)
        )
        y = F.scaled_dot_product_attention(rope(q), rope(k), v, is_causal=True)
        x = x + self.out(y.transpose(1, 2).reshape(b, t, d))
        return x + self.ff(self.n2(x))


class TargetEncoder(nn.Module):
    """Summary of a list of partner units (no timing): 2 layers, masked mean, linear."""

    def __init__(self, k: int, d: int, layers: int, out: int) -> None:
        super().__init__()
        self.emb = nn.Embedding(k + 1, d, padding_idx=k)
        self.pos = nn.Parameter(torch.zeros(MAX_UNITS, d))
        layer = nn.TransformerEncoderLayer(
            d, 4, 4 * d, dropout=0.0, batch_first=True, norm_first=True
        )
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.proj = nn.Linear(d, out)

    def forward(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = self.enc(
            self.emb(ids) + self.pos[: ids.shape[1]], src_key_padding_mask=~mask
        )
        m = mask.unsqueeze(-1).to(h.dtype)
        return self.proj((h * m).sum(1) / m.sum(1).clamp(min=1))


class Tower(nn.Module):
    def __init__(self, cfg: TowerConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.emb_self = nn.Embedding(cfg.k + 2, cfg.d)
        self.emb_partner = nn.Embedding(cfg.k + 2, cfg.d)
        self.pros = nn.Linear(6, cfg.d)
        self.blocks = nn.ModuleList(
            Block(cfg.d, cfg.heads, cfg.ffn) for _ in range(cfg.layers)
        )
        self.norm = nn.LayerNorm(cfg.d)
        self.predictor = nn.Sequential(
            nn.Linear(cfg.d, 1024), nn.GELU(), nn.Linear(1024, cfg.latent)
        )
        self.own_next = nn.Linear(cfg.d, cfg.k)
        self.who = nn.Linear(cfg.d, 2 * WHO_AHEAD)
        self.target = TargetEncoder(cfg.k, cfg.target_d, cfg.target_layers, cfg.latent)
        self.log_scale = nn.Parameter(torch.tensor(2.659))  # 1 / 0.07

    def states(
        self, tok_self: torch.Tensor, tok_partner: torch.Tensor, pros: torch.Tensor
    ) -> torch.Tensor:
        x = self.emb_self(tok_self) + self.emb_partner(tok_partner) + self.pros(pros)
        for block in self.blocks:
            x = block(x)
        return self.norm(x)

    def scores(
        self,
        h: torch.Tensor,
        anchor_row: torch.Tensor,
        anchor_pos: torch.Tensor,
        cand_ids: torch.Tensor,
        cand_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Cosine scores ``[N, 16]`` (column 0 is the true window) and in-batch ``[N, N]``."""
        q = F.normalize(self.predictor(h[anchor_row, anchor_pos]).float(), dim=-1)
        n, c, u = cand_ids.shape
        z = F.normalize(
            self.target(cand_ids.view(n * c, u), cand_mask.view(n * c, u)).float(),
            dim=-1,
        ).view(n, c, -1)
        return torch.einsum("nd,ncd->nc", q, z), q @ z[:, 0].T

    def loss(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, float]]:
        h = self.states(batch["tok_self"], batch["tok_partner"], batch["pros"])
        hard, inbatch = self.scores(
            h,
            batch["anchor_row"],
            batch["anchor_pos"],
            batch["cand_ids"],
            batch["cand_mask"],
        )
        scale = self.log_scale.clamp(max=4.6052).exp()
        eye = torch.eye(len(hard), device=hard.device, dtype=torch.bool)
        # mask after scaling: -inf times the learned scale gives a NaN gradient
        logits = torch.cat(
            [hard * scale, (inbatch * scale).masked_fill(eye, float("-inf"))], 1
        )
        nce = F.cross_entropy(
            logits, torch.zeros(len(hard), dtype=torch.long, device=hard.device)
        )
        own = F.cross_entropy(
            self.own_next(h).flatten(0, 1).float(),
            batch["own_next"].flatten(),
            ignore_index=-100,
        )
        who = F.binary_cross_entropy_with_logits(self.who(h).float(), batch["who"])
        total = nce + 0.5 * own + 0.2 * who
        acc = top1(hard).float().mean()
        return total, {
            "loss": total.item(),
            "nce": nce.item(),
            "own": own.item(),
            "who": who.item(),
            "acc16_train": acc.item(),
            "scale": scale.item(),
            "anchors": len(hard),
        }

    def parameters_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
