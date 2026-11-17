"""DY-23: the DY-22 grid LM over voice-invariant product codes instead of one DY-14 unit per syllable.

Every grid frame is an event: a new syllable (UNIT), ``CONT``, ``SIL`` or ``EOS``. A syllable is described by
8 codes of 256 (groups of the voice-invariant content code z, units_vinv.py). Input at a frame: the event's
embedding plus, at a syllable, the sum of its 8 code embeddings. Output from the state at frame t: the next
event (4 classes) and, when it is a syllable, its 8 codes through a chain (code g sees the state and the
codes before it, teacher-forced). Prosody of a syllable is predicted from the state after it is read, as in
DY-22; JEPA as in DY-21/22.
"""

import torch
import torch.nn.functional as F
from torch import nn

from dauys.dy21_lm import JEPA_WEIGHT, Block, LMConfig, jepa_target
from dauys.dy22_lm import E_BINS, F0_BINS, PROS_WEIGHT

UNIT, CONT, SIL, EOS = 0, 1, 2, 3
G, C = 8, 256


class CodeLM(nn.Module):
    def __init__(self, cfg: LMConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d = cfg.d
        self.ev_emb = nn.Embedding(4, d)
        self.code_emb = nn.Embedding(G * C, d)
        self.blocks = nn.ModuleList(Block(d, cfg.heads, cfg.ffn) for _ in range(cfg.layers))
        self.norm = nn.RMSNorm(d)
        self.ev_head = nn.Linear(d, 4, bias=False)
        self.code_in = nn.Embedding(G * C, d)  # chain: code g is added before predicting code g + 1
        self.code_norm = nn.ModuleList(nn.RMSNorm(d) for _ in range(G))
        self.code_head = nn.ModuleList(nn.Linear(d, C, bias=False) for _ in range(G))
        self.pred = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))
        self.pros = nn.Linear(d, F0_BINS + E_BINS + 2)
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, std=0.02)
                if getattr(m, "bias", None) is not None:
                    nn.init.zeros_(m.bias)
        self.register_buffer("offs", torch.arange(G) * C, persistent=False)

    def embed(self, ev: torch.Tensor, codes: torch.Tensor) -> torch.Tensor:
        """ev [B, T] events, codes [B, T, G] (ignored where ev != UNIT)."""
        x = self.ev_emb(ev)
        cs = self.code_emb(codes.long() + self.offs).sum(-2)
        return x + cs * (ev == UNIT).unsqueeze(-1)

    def states(self, ev: torch.Tensor, codes: torch.Tensor) -> torch.Tensor:
        x = self.embed(ev, codes)
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)

    def code_logits(self, h: torch.Tensor, codes: torch.Tensor) -> torch.Tensor:
        """Teacher-forced chain: h [N, d], codes [N, G] -> logits [N, G, C]."""
        q, out = h, []
        for g in range(G):
            out.append(self.code_head[g](self.code_norm[g](q)).float())
            q = q + self.code_in(codes[:, g].long() + g * C)
        return torch.stack(out, 1)

    @torch.no_grad()
    def sample_codes(self, h: torch.Tensor, temp: float, gen: torch.Generator) -> torch.Tensor:
        q, out = h, []
        for g in range(G):
            p = torch.softmax(self.code_head[g](self.code_norm[g](q)).float() / temp, -1)
            c = torch.multinomial(p, 1, generator=gen)[:, 0]
            out.append(c)
            q = q + self.code_in(c + g * C)
        return torch.stack(out, 1)

    def prosody(self, h: torch.Tensor):
        z = self.pros(h).float()
        return z[..., :F0_BINS], z[..., F0_BINS : F0_BINS + E_BINS], z[..., F0_BINS + E_BINS :]


def from_dy22(state: dict, cfg: LMConfig) -> CodeLM:
    """Transformer blocks, final norm, JEPA predictor and prosody head from DY-22; event embeddings from its
    CONT / SIL / EOS rows (UNIT from the mean of its unit rows); code embeddings and heads are new."""
    from dauys.dy22_lm import CONT as C22
    from dauys.dy22_lm import EOS as E22
    from dauys.dy22_lm import K
    from dauys.dy22_lm import SIL as S22

    m = CodeLM(cfg)
    own = m.state_dict()
    for k, v in state.items():
        if k.startswith(("blocks.", "norm.", "pred.", "pros.")) and k in own and own[k].shape == v.shape:
            own[k] = v
    e = state["emb.weight"]
    own["ev_emb.weight"] = torch.stack([e[:K].mean(0), e[C22], e[S22], e[E22]])
    m.load_state_dict(own)
    return m


def loss(model: CodeLM, tgt: CodeLM, ev: torch.Tensor, codes: torch.Tensor, pros: torch.Tensor):
    """ev [B, T+1], codes [B, T+1, G], pros [B, T+1, 3] (bins, valid at syllables)."""
    x_ev, x_c, y_ev, y_c = ev[:, :-1], codes[:, :-1], ev[:, 1:], codes[:, 1:]
    h = model.states(x_ev, x_c)
    ce_ev = F.cross_entropy(model.ev_head(h).float().flatten(0, 1), y_ev.flatten())
    nxt = y_ev == UNIT
    yc = y_c[nxt].long()
    lc = model.code_logits(h[nxt], yc)
    ce_codes = F.cross_entropy(lc.flatten(0, 1), yc.flatten()) if len(yc) else ce_ev * 0
    is_unit = x_ev == UNIT
    pf, pe, pv = model.prosody(h[is_unit])
    p = pros[:, :-1][is_unit].long()
    v = p[:, 2]
    l_v = F.cross_entropy(pv, v)
    l_e = F.cross_entropy(pe, p[:, 1])
    l_f = F.cross_entropy(pf[v == 1], p[v == 1, 0]) if (v == 1).any() else l_v * 0
    with torch.no_grad():
        target = jepa_target(tgt.states(x_ev, x_c))
    t = target.shape[1]
    pred = F.normalize(model.pred(h[:, :t]).float(), dim=-1)
    jepa = (1 - (pred * F.normalize(target, dim=-1)).sum(-1)).mean()
    # per syllable the 8 codes carry the content: their summed CE is the content term
    ce = ce_ev + ce_codes * G * nxt.float().mean()
    total = ce + JEPA_WEIGHT * jepa + PROS_WEIGHT * (l_f + l_e + l_v)
    return total, {"ce_ev": ce_ev.item(), "ce_code": ce_codes.item(), "jepa": jepa.item(), "f0": l_f.item(),
                   "energy": l_e.item(), "voiced": l_v.item(), "loss": total.item()}
