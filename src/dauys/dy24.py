"""DY-24: three variants on the DY-23 data, looking for the lever on meaning (same data, size and steps).

B: one clean content token per syllable — the voice-invariant content code z rebuilt from the 8x256 codes and
   snapped to the nearest of 2048 content centroids (units_vinv.py books) — on the DY-22 grid (timing, pauses,
   prosody), DY-22 GridLM and loss; unit rows of the embedding and head are new.
C: the same tokens as one deduplicated syllable stream (no timing), DY-21 UnitLM and loss: about two passes over
   the data in the same steps and ~200 s of speech in the context.
D: DY-23 CodeLM, but the main target is the content of the next 2 s: the mean z of the syllables in the next
   20 frames (cosine), code heads weighted 0.1.
"""

import torch
import torch.nn.functional as F
from torch import nn

from dauys.dy21_lm import JEPA_WEIGHT, LMConfig, jepa_target
from dauys.dy22_lm import PROS_WEIGHT
from dauys.dy23_lm import G, UNIT, CodeLM

N_CONTENT = 2048
FUT = 20  # frames (2 s)


class ContentBooks:
    """z from 8x256 codes, and the nearest of the 2048 content centroids."""

    def __init__(self, coder_ckpt: dict, device) -> None:
        self.pq = torch.stack([b for b in coder_ckpt["pq"]["8x256"]]).to(device)  # [8, 256, 32]
        self.c2048 = F.normalize(coder_ckpt["books"][2048].to(device), dim=1)  # [2048, 256]

    def z(self, codes: torch.Tensor) -> torch.Tensor:
        """codes [..., 8] long -> z [..., 256] (L2-normalised)."""
        parts = [self.pq[g][codes[..., g]] for g in range(G)]
        return F.normalize(torch.cat(parts, -1), dim=-1)

    @torch.no_grad()
    def content_ids(self, codes: torch.Tensor, chunk: int = 1 << 20) -> torch.Tensor:
        out = torch.empty(len(codes), dtype=torch.long, device=codes.device)
        for i in range(0, len(codes), chunk):
            out[i : i + chunk] = (self.z(codes[i : i + chunk].long()) @ self.c2048.T).argmax(1)
        return out


def reinit_unit_rows(model: nn.Module, k: int) -> None:
    """Fresh rows for the unit ids of an embedding/head copied from a model with other units."""
    with torch.no_grad():
        model.emb.weight[:k].normal_(0, 0.02)
        model.head.weight[:k].normal_(0, 0.02)


class FutureLM(CodeLM):
    def __init__(self, cfg: LMConfig) -> None:
        super().__init__(cfg)
        self.fut = nn.Sequential(nn.Linear(cfg.d, cfg.d), nn.GELU(), nn.Linear(cfg.d, 256))


def future_target(ev: torch.Tensor, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean z of the syllables in the FUT frames after each position (y indices t..t+FUT-1): [B, T, 256] and a mask where at least one syllable falls in."""
    m = (ev == UNIT).float().unsqueeze(-1)
    zc = torch.cat([torch.zeros_like(z[:, :1]), (z * m).cumsum(1)], 1)
    mc = torch.cat([torch.zeros_like(m[:, :1]), m.cumsum(1)], 1)
    t = z.shape[1]
    lo = torch.arange(t, device=z.device)  # y index t is the frame right after position t
    hi = torch.clamp(lo + FUT, max=t)
    s = zc[:, hi] - zc[:, lo]
    n = mc[:, hi] - mc[:, lo]
    return F.normalize(s, dim=-1), (n[..., 0] > 0)


def loss_d(model: FutureLM, tgt: FutureLM, books: ContentBooks, ev, codes, pros, code_w: float = 0.1):
    """DY-23 loss with code heads weighted ``code_w`` plus the 2-s future-content cosine (weight 1)."""
    x_ev, x_c, y_ev, y_c = ev[:, :-1], codes[:, :-1], ev[:, 1:], codes[:, 1:]
    h = model.states(x_ev, x_c)
    ce_ev = F.cross_entropy(model.ev_head(h).float().flatten(0, 1), y_ev.flatten())
    nxt = y_ev == UNIT
    yc = y_c[nxt].long()
    if len(yc):
        per_g = F.cross_entropy(model.code_logits(h[nxt], yc).transpose(1, 2), yc, reduction="none").mean(0)
        ce_codes = per_g.mean()
    else:
        per_g = ce_codes = ce_ev * 0
    is_unit = x_ev == UNIT
    pf, pe, pv = model.prosody(h[is_unit])
    p = pros[:, :-1][is_unit].long()
    v = p[:, 2]
    l_v = F.cross_entropy(pv, v)
    l_e = F.cross_entropy(pe, p[:, 1])
    l_f = F.cross_entropy(pf[v == 1], p[v == 1, 0]) if (v == 1).any() else l_v * 0
    with torch.no_grad():
        target_j = jepa_target(tgt.states(x_ev, x_c))
        z = books.z(y_c.long())
        target_f, mask = future_target(y_ev, z)
    t = target_j.shape[1]
    pj = F.normalize(model.pred(h[:, :t]).float(), dim=-1)
    jepa = (1 - (pj * F.normalize(target_j, dim=-1)).sum(-1)).mean()
    pf_ = F.normalize(model.fut(h).float(), dim=-1)
    fut = (1 - (pf_ * target_f).sum(-1))[mask].mean() if mask.any() else ce_ev * 0
    ce = ce_ev + code_w * per_g.sum() * nxt.float().mean()
    total = ce + JEPA_WEIGHT * jepa + PROS_WEIGHT * (l_f + l_e + l_v) + fut
    return total, {"ce_ev": ce_ev.item(), "ce_code": ce_codes.item(), "fut": fut.item(), "jepa": jepa.item(),
                   "f0": l_f.item(), "energy": l_e.item(), "voiced": l_v.item(), "loss": total.item()}
