"""DY-37: one participant of the conversation speaks from a chosen thought.

The DY-32 b UnitLM (single stream, K2000 codes, 50 Hz, CE + JEPA) with three additions on the input of every frame:
* a role embedding: 0 no speech, 1 "me" (the participant the model is in this sequence), 2 "other";
* on frames of "me" utterances: a linear projection of the utterance's thought (a FROZEN 64-d unit vector read from a
  dictionary; no gradient reaches it); elsewhere a learned null vector;
* a mask code (vocab K + 1) that replaces the input codes inside some "me" utterances (targets unchanged).
The thought head (last-layer state at the last frame before a "me" utterance -> MLP -> 64-d unit vector) is trained
by InfoNCE against the dictionary thoughts.
"""

import torch
import torch.nn.functional as F
from torch import nn

from dauys.dy21_lm import LMConfig, UnitLM

ZDIM = 64
TAU = 0.1
NCE_WEIGHT = 1.0
ROLE_NONE, ROLE_ME, ROLE_OTHER = 0, 1, 2


class RoleThoughtLM(nn.Module):
    def __init__(self, cfg: LMConfig) -> None:
        super().__init__()
        self.lm = UnitLM(cfg)
        self.cfg = cfg
        d = cfg.d
        self.role = nn.Embedding(3, d)
        nn.init.normal_(self.role.weight, std=0.02)
        self.proj = nn.Linear(ZDIM, d, bias=False)
        nn.init.normal_(self.proj.weight, std=0.02)
        self.null = nn.Parameter(torch.randn(d) * 0.02)
        self.zhead = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, ZDIM))
        for m in self.zhead:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                nn.init.zeros_(m.bias)

    def inputs(self, ids: torch.Tensor, roles: torch.Tensor, zin: torch.Tensor | None, has: torch.Tensor | None) -> torch.Tensor:
        """ids, roles [B, T]; zin [B, T, 64] thought per input frame, has [B, T] bool (False -> null vector)."""
        x = self.lm.emb(ids) + self.role(roles)
        if zin is None:
            return x + self.null
        th = self.proj(zin.to(self.proj.weight.dtype))
        return x + torch.where(has[..., None], th, self.null.to(th.dtype))

    def states(self, ids, roles, zin=None, has=None) -> torch.Tensor:
        x = self.inputs(ids, roles, zin, has)
        for blk in self.lm.blocks:
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


def param_counts(model: RoleThoughtLM) -> dict:
    n = lambda m: sum(p.numel() for p in m.parameters())
    return {"unit_lm": n(model.lm), "role_emb": model.role.weight.numel(),
            "thought_proj_null": model.proj.weight.numel() + model.null.numel(), "zhead": n(model.zhead), "total": n(model)}


__all__ = ["RoleThoughtLM", "nce", "param_counts", "ZDIM", "TAU", "NCE_WEIGHT", "ROLE_NONE", "ROLE_ME", "ROLE_OTHER", "LMConfig"]
