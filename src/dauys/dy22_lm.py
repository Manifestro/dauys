"""DY-22: the DY-21 unit LM on the 10 Hz grid, so it also says when and how a syllable is spoken.

Tokens are the DY-14 grid of one speech stream: a unit where a syllable starts, ``CONT`` while it lasts,
``SIL`` in pauses (runs capped at ``MAX_SIL`` frames), ``EOS`` between pieces. Timing and pauses are thus
predicted with the units. Prosody of every unit (F0 in semitones from the piece median, energy in dB from
the piece median, voiced) is predicted as classes from the state right after the unit is read, so it
follows the chosen unit and is sampled at generation. JEPA as in DY-21.
"""

import torch
import torch.nn.functional as F
from torch import nn

from dauys.dy21_lm import JEPA_WEIGHT, LMConfig, UnitLM, jepa_target

K = 8192
CONT, SIL, EOS = K, K + 1, K + 2
VOCAB = K + 3
MAX_SIL = 20  # pauses longer than 2 s are cut to 2 s
F0_BINS, F0_LO, F0_HI = 25, -12.0, 12.0  # 1 semitone per bin
E_BINS, E_LO, E_HI = 20, -30.0, 10.0  # 2 dB per bin
PROS_WEIGHT = 0.2


def f0_bin(st: torch.Tensor) -> torch.Tensor:
    return ((st.clamp(F0_LO, F0_HI) - F0_LO) / (F0_HI - F0_LO) * (F0_BINS - 1)).round().long()


def e_bin(db: torch.Tensor) -> torch.Tensor:
    return ((db.clamp(E_LO, E_HI) - E_LO) / (E_HI - E_LO) * (E_BINS - 1)).round().long()


def f0_of_bin(b: torch.Tensor) -> torch.Tensor:
    return F0_LO + b.float() * (F0_HI - F0_LO) / (F0_BINS - 1)


def e_of_bin(b: torch.Tensor) -> torch.Tensor:
    return E_LO + b.float() * (E_HI - E_LO) / (E_BINS - 1)


class GridLM(UnitLM):
    def __init__(self, cfg: LMConfig) -> None:
        super().__init__(cfg)
        self.pros = nn.Linear(cfg.d, F0_BINS + E_BINS + 2)
        nn.init.normal_(self.pros.weight, std=0.02)
        nn.init.zeros_(self.pros.bias)

    def prosody(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.pros(h).float()
        return z[..., :F0_BINS], z[..., F0_BINS : F0_BINS + E_BINS], z[..., F0_BINS + E_BINS :]


def from_dy21(state: dict, cfg: LMConfig) -> GridLM:
    """GridLM with DY-21 weights; DY-21 EOS (row K) becomes EOS here, CONT and SIL rows are new."""
    model = GridLM(cfg)
    own = model.state_dict()
    for k, v in state.items():
        if k in ("emb.weight", "head.weight"):
            own[k][:K] = v[:K]
            own[k][EOS] = v[K]
        else:
            own[k] = v
    model.load_state_dict(own)
    return model


def loss(model: GridLM, tgt: GridLM, ids: torch.Tensor, pros: torch.Tensor) -> tuple[torch.Tensor, dict]:
    """ids [B, T + 1] grid tokens; pros [B, T + 1, 3] (f0 bin, energy bin, voiced) valid at unit tokens."""
    x, y = ids[:, :-1], ids[:, 1:]
    h = model.states(x)
    logits = model.logits(h)
    ce = F.cross_entropy(logits.flatten(0, 1), y.flatten())
    unit_y = y < K
    ce_unit = F.cross_entropy(logits[unit_y], y[unit_y]) if unit_y.any() else ce * 0
    is_unit = x < K
    pf, pe, pv = model.prosody(h[is_unit])
    p = pros[:, :-1][is_unit]
    v = p[:, 2]
    l_v = F.cross_entropy(pv, v)
    l_e = F.cross_entropy(pe, p[:, 1])
    l_f = F.cross_entropy(pf[v == 1], p[v == 1, 0]) if (v == 1).any() else l_v * 0
    with torch.no_grad():
        target = jepa_target(tgt.states(x))
    t = target.shape[1]
    pred = F.normalize(model.pred(h[:, :t]).float(), dim=-1)
    jepa = (1 - (pred * F.normalize(target, dim=-1)).sum(-1)).mean()
    pros_l = l_f + l_e + l_v
    total = ce + JEPA_WEIGHT * jepa + PROS_WEIGHT * pros_l
    return total, {"ce": ce.item(), "ce_unit": ce_unit.item(), "jepa": jepa.item(), "f0": l_f.item(),
                   "energy": l_e.item(), "voiced": l_v.item(), "loss": total.item()}
