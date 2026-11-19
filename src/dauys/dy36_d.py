"""DY-36 candidate D: cross-voice reconstruction with a weak decoder (no text anywhere).

* Encoder: bidirectional transformer (6 x 512, RoPE, own embedding table) over the K2000 codes of an utterance in voice A
  (a share of input codes replaced by MASK) -> mean over time -> linear -> z (64-d, L2-normalised).
* Weak decoder: 2 blocks over the codes of the SAME utterance in voice B (kNN-converted in WavLM space, tempo-changed),
  most of them replaced by MASK (visibility in blocks of BLOCK frames, each visible with probability VIS), plus z
  (projected) and a voice-B vector (projected pool mean WavLM feature) added at every position -> logits over K.
  Loss: cross-entropy on the masked positions.
* Invariance: InfoNCE (tau 0.1) of z(A) against z(B) of the same utterance, other utterances of the batch as negatives.
"""

import torch
import torch.nn.functional as F
from torch import nn

from dauys.dy35_thought import EncBlock

K = 2000
MASK = K
ZDIM = 64
VDIM = 1024


def _init(m: nn.Module) -> None:
    for x in m.modules():
        if isinstance(x, (nn.Linear, nn.Embedding)):
            nn.init.normal_(x.weight, std=0.02)
            if isinstance(x, nn.Linear) and x.bias is not None:
                nn.init.zeros_(x.bias)


class Encoder(nn.Module):
    def __init__(self, d: int = 512, layers: int = 6, heads: int = 8, ffn: int = 1344) -> None:
        super().__init__()
        self.emb = nn.Embedding(K + 1, d)
        self.blocks = nn.ModuleList(EncBlock(d, heads, ffn) for _ in range(layers))
        self.norm = nn.RMSNorm(d)
        self.out = nn.Linear(d, ZDIM)
        _init(self)

    def forward(self, codes: torch.Tensor, lens: torch.Tensor) -> torch.Tensor:
        keep = torch.arange(codes.shape[1], device=codes.device)[None] < lens[:, None]
        x = self.emb(codes)
        for blk in self.blocks:
            x = blk(x, keep)
        x = self.norm(x) * keep[..., None]
        m = x.sum(1) / lens[:, None].to(x.dtype)
        return F.normalize(self.out(m).float(), dim=-1)

    @torch.no_grad()
    def encode_many(self, utts: list, budget: int = 16384) -> torch.Tensor:
        """Variable-length code arrays -> z [n, 64]; sorted by length, padded on the host, one copy per batch."""
        import numpy as np

        dev = self.emb.weight.device
        lens_all = np.array([len(u) for u in utts])
        order = np.argsort(-lens_all, kind="stable")
        out = torch.empty(len(utts), ZDIM, device=dev)
        i = 0
        while i < len(order):
            L = int(lens_all[order[i]])
            idx = order[i : i + max(1, budget // max(L, 1))]
            c = np.zeros((len(idx), L), np.int64)
            for r, j in enumerate(idx):
                c[r, : lens_all[j]] = np.asarray(utts[j])
            cc = torch.from_numpy(c).pin_memory().to(dev, non_blocking=True)
            lens = torch.from_numpy(lens_all[idx]).to(dev)
            out[torch.from_numpy(idx).to(dev)] = self.forward(cc, lens)
            i += len(idx)
        return out


class LocalEncoder(Encoder):
    """Variant E: no global mixing. Embedding -> `layers` residual conv blocks (kernel `kern`, GLU, RMSNorm), receptive
    field 1 + layers * (kern - 1) frames (6 x 7 -> 37 frames = +-360 ms at 50 Hz) -> per-frame MLP -> mean over the
    valid frames -> linear -> z (64-d, L2). Padding frames are zeroed before every convolution."""

    def __init__(self, d: int = 512, layers: int = 6, kern: int = 7) -> None:
        nn.Module.__init__(self)
        self.emb = nn.Embedding(K + 1, d)
        self.norms = nn.ModuleList(nn.RMSNorm(d) for _ in range(layers))
        self.convs = nn.ModuleList(nn.Conv1d(d, 2 * d, kern, padding=kern // 2) for _ in range(layers))
        self.mlp = nn.Sequential(nn.RMSNorm(d), nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
        self.norm = nn.RMSNorm(d)
        self.out = nn.Linear(d, ZDIM)
        _init(self)

    def forward(self, codes: torch.Tensor, lens: torch.Tensor) -> torch.Tensor:
        keep = (torch.arange(codes.shape[1], device=codes.device)[None] < lens[:, None])[..., None]
        x = self.emb(codes) * keep
        for n, c in zip(self.norms, self.convs):
            h = (n(x) * keep).transpose(1, 2)
            x = x + F.glu(c(h), dim=1).transpose(1, 2)
        x = x + self.mlp(x)
        x = self.norm(x) * keep
        m = x.sum(1) / lens[:, None].to(x.dtype)
        return F.normalize(self.out(m).float(), dim=-1)


class Decoder(nn.Module):
    def __init__(self, d: int = 512, layers: int = 2, heads: int = 8, ffn: int = 1344) -> None:
        super().__init__()
        self.emb = nn.Embedding(K + 1, d)
        self.zp = nn.Linear(ZDIM, d)
        self.vn = nn.LayerNorm(VDIM)
        self.vp = nn.Linear(VDIM, d)
        self.blocks = nn.ModuleList(EncBlock(d, heads, ffn) for _ in range(layers))
        self.norm = nn.RMSNorm(d)
        self.head = nn.Linear(d, K)
        _init(self)

    def forward(self, codes: torch.Tensor, lens: torch.Tensor, z: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        keep = torch.arange(codes.shape[1], device=codes.device)[None] < lens[:, None]
        x = self.emb(codes) + (self.zp(z) + self.vp(self.vn(v)))[:, None].to(self.emb.weight.dtype)
        for blk in self.blocks:
            x = blk(x, keep)
        return self.head(self.norm(x))


class ModelD(nn.Module):
    def __init__(self, enc_layers: int = 6, dec_layers: int = 2, enc_type: str = "global", bag: bool = False) -> None:
        super().__init__()
        self.enc = LocalEncoder(layers=enc_layers) if enc_type == "local" else Encoder(layers=enc_layers)
        self.dec = Decoder(layers=dec_layers)
        self.bag = nn.Linear(ZDIM, K) if bag else None


def mask_input(codes: torch.Tensor, lens: torch.Tensor, rate: float, gen=None) -> torch.Tensor:
    keep = torch.arange(codes.shape[1], device=codes.device)[None] < lens[:, None]
    m = (torch.rand(codes.shape, device=codes.device, generator=gen) < rate) & keep
    return torch.where(m, torch.full_like(codes, MASK), codes)


def block_visible(codes: torch.Tensor, lens: torch.Tensor, vis: float, block: int, gen=None) -> torch.Tensor:
    """Visible positions: blocks of `block` frames, each visible with probability vis (random block phase per row)."""
    b, t = codes.shape
    nb = t // block + 2
    vb = torch.rand(b, nb, device=codes.device, generator=gen) < vis
    ph = torch.randint(0, block, (b, 1), device=codes.device, generator=gen)
    pos = (torch.arange(t, device=codes.device)[None] + ph) // block
    keep = torch.arange(t, device=codes.device)[None] < lens[:, None]
    return torch.gather(vb, 1, pos) & keep


def info_nce(a: torch.Tensor, b: torch.Tensor, tau: float = 0.1) -> torch.Tensor:
    lg = a @ b.T / tau
    y = torch.arange(len(a), device=a.device)
    return (F.cross_entropy(lg, y) + F.cross_entropy(lg.T, y)) / 2
