"""DY-33 track S: slow codec for WavLM-Large layer-6 features (kNN-VC, 50 Hz, 1024-d).

Encoder: per-frame linear -> ConvNeXt-style 1-D blocks at 50 Hz -> strided conv x4 down -> blocks at 12.5 Hz ->
linear to the code space. Residual VQ: 4 levels x 1024 codes, EMA codebooks with dead-code restarts (codes unused for
a while are replaced by random residuals of the current batch), straight-through, commitment loss. Level dropout in
training (a random number 1..4 of levels used for half the batches) so fewer levels still decode. Decoder mirrors the
encoder (transposed conv x4 up) and predicts the 50 Hz features; loss = MSE on normalised features + (1 - cosine).
Audio only; the kNN-VC vocoder is never trained, only used to listen/check.
"""

from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class CodecConfig:
    feat: int = 1024
    d: int = 512
    n50: int = 2
    n12: int = 4
    dq: int = 256
    levels: int = 4
    k: int = 1024
    kernel: int = 7
    down: int = 4


class Block(nn.Module):
    def __init__(self, d: int, kernel: int) -> None:
        super().__init__()
        self.dw = nn.Conv1d(d, d, kernel, padding=kernel // 2, groups=d)
        self.norm = nn.LayerNorm(d)
        self.up = nn.Linear(d, 4 * d)
        self.down = nn.Linear(4 * d, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [B, T, d]
        h = self.dw(x.transpose(1, 2)).transpose(1, 2)
        return x + self.down(F.gelu(self.up(self.norm(h))))


class RVQ(nn.Module):
    def __init__(self, levels: int, k: int, dim: int, decay: float = 0.99, dead_after: int = 200) -> None:
        super().__init__()
        self.levels, self.k, self.decay, self.dead_after = levels, k, decay, dead_after
        self.register_buffer("books", torch.randn(levels, k, dim) * 0.1)
        self.register_buffer("ema_n", torch.ones(levels, k))
        self.register_buffer("ema_sum", self.books.clone())
        self.register_buffer("idle", torch.zeros(levels, k))
        self.register_buffer("inited", torch.zeros(()))

    def nearest(self, r: torch.Tensor, j: int) -> torch.Tensor:
        c = self.books[j]
        return ((c * c).sum(-1)[None] - 2 * r @ c.T).argmin(-1)

    @torch.no_grad()
    def _init(self, z: torch.Tensor) -> None:
        r = z
        for j in range(self.levels):
            pick = r[torch.randint(len(r), (self.k,), device=r.device)]
            self.books[j] = pick
            self.ema_sum[j] = pick
            self.ema_n[j] = 1.0
            r = r - self.books[j][self.nearest(r, j)]
        self.inited.fill_(1)

    def forward(self, z: torch.Tensor, n_levels: int | None = None):
        """z [B, T, dim] -> (quantised, codes [B, T, L], commitment loss, usage stats)."""
        B, T, D = z.shape
        flat = z.reshape(-1, D).float()
        if self.training and not bool(self.inited):
            self._init(flat.detach())
        n_levels = n_levels or self.levels
        r = flat
        q = torch.zeros_like(flat)
        codes = []
        for j in range(n_levels):
            a = self.nearest(r.detach(), j)
            qj = self.books[j][a]
            if self.training:
                with torch.no_grad():
                    oh = torch.zeros(len(a), self.k, device=a.device)
                    oh.scatter_(1, a[:, None], 1.0)
                    n = oh.sum(0)
                    self.ema_n[j].mul_(self.decay).add_(n, alpha=1 - self.decay)
                    self.ema_sum[j].mul_(self.decay).add_(oh.T @ r.detach(), alpha=1 - self.decay)
                    tot = self.ema_n[j].sum()
                    nn_ = (self.ema_n[j] + 1e-5) / (tot + self.k * 1e-5) * tot
                    self.books[j] = self.ema_sum[j] / nn_[:, None]
                    self.idle[j] = torch.where(n > 0, torch.zeros_like(n), self.idle[j] + 1)
                    dead = self.idle[j] >= self.dead_after
                    nd = int(dead.sum())
                    if nd:
                        new = r.detach()[torch.randint(len(r), (nd,), device=r.device)]
                        self.books[j][dead] = new
                        self.ema_sum[j][dead] = new
                        self.ema_n[j][dead] = 1.0
                        self.idle[j][dead] = 0
            q = q + qj
            r = r - qj
            codes.append(a)
        commit = F.mse_loss(flat, q.detach())
        qst = flat + (q - flat).detach()
        codes = torch.stack(codes, -1).view(B, T, n_levels)
        return qst.view(B, T, D).to(z.dtype), codes, commit

    def lookup(self, codes: torch.Tensor) -> torch.Tensor:
        """codes [B, T, L'] (first L' levels) -> summed vectors [B, T, dim]."""
        return sum(self.books[j][codes[..., j]] for j in range(codes.shape[-1]))


class Codec(nn.Module):
    def __init__(self, cfg: CodecConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d = cfg.d
        self.register_buffer("mu", torch.zeros(cfg.feat))
        self.register_buffer("sd", torch.ones(cfg.feat))
        self.inp = nn.Linear(cfg.feat, d)
        self.enc50 = nn.ModuleList(Block(d, cfg.kernel) for _ in range(cfg.n50))
        self.dn = nn.Conv1d(d, d, cfg.down * 2, stride=cfg.down, padding=cfg.down // 2)
        self.enc12 = nn.ModuleList(Block(d, cfg.kernel) for _ in range(cfg.n12))
        self.to_q = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, cfg.dq))
        self.rvq = RVQ(cfg.levels, cfg.k, cfg.dq)
        self.from_q = nn.Linear(cfg.dq, d)
        self.dec12 = nn.ModuleList(Block(d, cfg.kernel) for _ in range(cfg.n12))
        self.up = nn.ConvTranspose1d(d, d, cfg.down, stride=cfg.down)
        self.dec50 = nn.ModuleList(Block(d, cfg.kernel) for _ in range(cfg.n50))
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, cfg.feat))

    def encode_latent(self, x: torch.Tensor) -> torch.Tensor:
        """x [B, T, 1024] raw features, T multiple of 4 -> [B, T/4, dq]."""
        h = self.inp((x - self.mu) / self.sd)
        for b in self.enc50:
            h = b(h)
        h = self.dn(h.transpose(1, 2)).transpose(1, 2)
        for b in self.enc12:
            h = b(h)
        return self.to_q(h)

    def decode_latent(self, q: torch.Tensor) -> torch.Tensor:
        h = self.from_q(q)
        for b in self.dec12:
            h = b(h)
        h = self.up(h.transpose(1, 2)).transpose(1, 2)
        for b in self.dec50:
            h = b(h)
        return self.out(h) * self.sd + self.mu

    def forward(self, x: torch.Tensor, n_levels: int | None = None):
        z = self.encode_latent(x)
        q, codes, commit = self.rvq(z, n_levels)
        y = self.decode_latent(q)
        return y, codes, commit

    def loss(self, x: torch.Tensor, n_levels: int | None = None):
        y, codes, commit = self(x, n_levels)
        xn, yn = (x - self.mu) / self.sd, (y.float() - self.mu) / self.sd
        mse = F.mse_loss(yn, xn)
        cos = 1 - F.cosine_similarity(y.float(), x, dim=-1).mean()
        return mse + cos + 0.25 * commit, {"mse": mse.item(), "cos": cos.item(), "commit": commit.item()}

    @torch.no_grad()
    def codes(self, x: torch.Tensor) -> torch.Tensor:
        """[B, T, 1024] (T multiple of 4) -> [B, T/4, levels] int."""
        return self.rvq(self.encode_latent(x))[1]

    @torch.no_grad()
    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """[B, T/4, L'] -> [B, T, 1024] features."""
        return self.decode_latent(self.rvq.lookup(codes))

    def config(self) -> dict:
        return asdict(self.cfg)
