"""DY-36 thought model: a participant of a conversation that composes the thought (64-d unit vector) of its next
utterance. No sound in, no sound out, no text anywhere.

Input = the sequence of utterances of one recording in time order (BOS + up to the last 256 utterances). Per utterance:
  - its dictionary vector without recording statistics ("raw": the 64-d projection, L2-normalised), and the causal
    speaker-centred vector (raw minus the mean raw of that speaker's utterances up to and including this one inside the
    given history, L2-normalised; 0 for the first one). The dictionary thought itself is centred with the speaker's
    mean over the WHOLE recording, which contains the future (and the predicted utterance); it is therefore never used
    as an input, only as the target / candidates / bank.
  - role embedding (me / other), bucketed duration, bucketed pause before it (time since the previous utterance ended).
Causal transformer (pre-norm, learned positions), output at position p = thought of utterance p of the window
(position 0 = BOS); the loss is taken only where that utterance is by "me".
"""

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CTX = 256
DUR_EDGES = np.array([150, 200, 300, 400, 500, 650, 800])  # frames
PAUSE_EDGES = np.array([0.2, 0.5, 1.0, 2.0, 5.0, 15.0, 60.0])  # seconds
N_DUR, N_PAUSE = len(DUR_EDGES) + 1, len(PAUSE_EDGES) + 2  # last pause bucket = first in the history
GAP_S, N_SAME, TAU = 60.0, 15, 0.1


@dataclass
class Cfg:
    d: int = 512
    layers: int = 12
    heads: int = 8
    ffn: int = 2048
    dropout: float = 0.1
    zdim: int = 64


class ThoughtModel(nn.Module):
    def __init__(self, c: Cfg) -> None:
        super().__init__()
        self.c = c
        self.inp = nn.Linear(2 * c.zdim, c.d)
        self.role = nn.Embedding(2, c.d)
        self.dur = nn.Embedding(N_DUR, c.d)
        self.pause = nn.Embedding(N_PAUSE, c.d)
        self.bos = nn.Parameter(torch.zeros(c.d))
        self.pos = nn.Embedding(CTX + 1, c.d)
        layer = nn.TransformerEncoderLayer(c.d, c.heads, c.ffn, c.dropout, activation="gelu", batch_first=True,
                                           norm_first=True)
        self.tf = nn.TransformerEncoder(layer, c.layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(c.d)
        self.head = nn.Linear(c.d, c.zdim)
        nn.init.normal_(self.bos, std=0.02)

    def forward(self, x, role, dur, pause, pad):
        """x [B, L, 128], role/dur/pause [B, L] (L utterances), pad [B, L+1] True = padding -> out [B, L+1, zdim]."""
        B, L, _ = x.shape
        h = self.inp(x) + self.role(role) + self.dur(dur) + self.pause(pause)
        h = torch.cat([self.bos.expand(B, 1, -1), h], 1)
        h = h + self.pos(torch.arange(L + 1, device=x.device))[None]
        mask = torch.triu(torch.full((L + 1, L + 1), float("-inf"), device=x.device), 1)
        h = self.tf(h, mask=mask, src_key_padding_mask=pad, is_causal=True)
        return F.normalize(self.head(self.norm(h)).float(), dim=-1)


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def bucket_dur(lens: np.ndarray) -> np.ndarray:
    return np.searchsorted(DUR_EDGES, lens, side="right")


def bucket_pause(t0: np.ndarray, t1: np.ndarray) -> np.ndarray:
    gap = np.r_[np.inf, t0[1:] - t1[:-1]]
    b = np.searchsorted(PAUSE_EDGES, np.maximum(gap, 0), side="right")
    b[0] = N_PAUSE - 1
    return b


def unit(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return np.where(n > 1e-6, x / np.maximum(n, 1e-12), 0.0)


def seq_features(raw: np.ndarray, spk: np.ndarray) -> np.ndarray:
    """[L, 128]: unit(raw) and unit(raw - causal mean of the same speaker inside the sequence)."""
    r = unit(raw.astype(np.float64))
    cen = np.zeros_like(r)
    for s in np.unique(spk):
        i = np.flatnonzero(spk == s)
        cm = np.cumsum(r[i], 0) / np.arange(1, len(i) + 1)[:, None]
        cen[i] = unit(r[i] - cm)
    return np.concatenate([r, cen], 1).astype(np.float32)


class Data:
    """Utterances in (episode, time) order with their dictionary vectors.

    ep, spk, t0, t1, len, raw [n, 64], thought [n, 64], valid; episodes contiguous. ok_target: valid thought and
    >= 15 same-speaker same-recording utterances >= 60 s away with length within x2."""

    def __init__(self, ep, spk, t0, t1, ln, raw, thought, valid) -> None:
        self.ep, self.spk = np.asarray(ep, np.int64), np.asarray(spk, np.int64)
        self.t0, self.t1, self.len = np.asarray(t0, np.float64), np.asarray(t1, np.float64), np.asarray(ln, np.int64)
        self.raw, self.thought, self.valid = raw.astype(np.float32), thought.astype(np.float32), valid.astype(bool)
        self.n = len(self.ep)
        assert np.all(np.diff(self.ep) >= 0)
        b = np.flatnonzero(np.diff(np.r_[-1, self.ep, -2]))
        self.eb, self.ee = b[:-1], b[1:]
        self.ep_of = np.repeat(np.arange(len(self.eb)), self.ee - self.eb)
        _, self.grp = np.unique(self.spk, return_inverse=True)
        order = np.argsort(self.grp, kind="stable")
        gb = np.flatnonzero(np.diff(np.r_[-1, self.grp[order], 10**12]))
        self.groups = [order[x:y] for x, y in zip(gb[:-1], gb[1:])]
        self.n_elig = np.zeros(self.n, np.int64)
        for g in self.groups:
            self.n_elig[g] = self.eligible(g, g).sum(1)
        self.ok_target = self.valid & (self.n_elig >= N_SAME)
        self.durb = bucket_dur(self.len)
        self.pauseb = np.zeros(self.n, np.int64)
        for a, e in zip(self.eb, self.ee):
            self.pauseb[a:e] = bucket_pause(self.t0[a:e], self.t1[a:e])

    def eligible(self, us: np.ndarray, ds: np.ndarray) -> np.ndarray:
        far = (self.t1[ds][None] <= self.t0[us][:, None] - GAP_S) | (self.t0[ds][None] >= self.t1[us][:, None] + GAP_S)
        r = self.len[ds][None] / self.len[us][:, None]
        return far & (r >= 0.5) & (r <= 2.0) & (ds[None] != us[:, None]) & self.valid[ds][None]

    def decoys(self, u: int, rng, k: int = N_SAME) -> np.ndarray:
        g = self.groups[self.grp[u]]
        el = g[self.eligible(np.array([u]), g)[0]]
        assert len(el) >= k
        d = rng.choice(el, k, replace=False)
        assert np.all(np.abs(self.t0[d] - self.t0[u]) >= GAP_S) and np.all(self.ep[d] == self.ep[u]) and u not in d
        return d


def collate(seqs: list[dict], dev) -> dict:
    """seqs: dicts with x [L,128], role, dur, pause [L] -> padded tensors (pad at the end)."""
    L = max(len(s["role"]) for s in seqs)
    B = len(seqs)
    x = np.zeros((B, L, 128), np.float32)
    role = np.zeros((B, L), np.int64)
    dur = np.zeros((B, L), np.int64)
    pause = np.zeros((B, L), np.int64)
    pad = np.ones((B, L + 1), bool)
    for i, s in enumerate(seqs):
        n = len(s["role"])
        x[i, :n], role[i, :n], dur[i, :n], pause[i, :n] = s["x"], s["role"], s["dur"], s["pause"]
        pad[i, : n + 1] = False
    t = lambda a: torch.from_numpy(a).to(dev, non_blocking=True)
    return {"x": t(x), "role": t(role), "dur": t(dur), "pause": t(pause), "pad": t(pad)}


def make_seq(D: Data, idx: np.ndarray, me_spk: int, raw_override: dict | None = None) -> dict:
    """Input sequence of utterances idx (time order) with "me" = me_spk; raw_override {position: raw vector,
    "spk": {position: speaker key}} replaces the content of some positions (C3 swaps)."""
    raw = D.raw[idx].copy()
    spk = D.spk[idx].copy()
    if raw_override:
        for p, v in raw_override.get("raw", {}).items():
            raw[p] = v
        for p, v in raw_override.get("spk", {}).items():
            spk[p] = v
    return {"x": seq_features(raw, spk), "role": (D.spk[idx] == me_spk).astype(np.int64), "dur": D.durb[idx],
            "pause": D.pauseb[idx]}


def train_batch(D: Data, rng, B: int, ctx: int = CTX) -> tuple[list[dict], list[dict]]:
    """B windows from B distinct episodes; one speaker of each is "me". Returns sequences and per-sequence targets
    (positions p, target utterance ids u, decoys [k, 15]); the utterance at position p (output p) is never in inputs
    0..p-1 (inputs = utterances before it)."""
    w = D.ee - D.eb
    eps = rng.choice(len(w), B, replace=False, p=w / w.sum())
    seqs, tg = [], []
    for e in eps:
        a, b = int(D.eb[e]), int(D.ee[e])
        spks = np.unique(D.spk[a:b][D.ok_target[a:b]])
        if len(spks) == 0:
            spks = np.unique(D.spk[a:b])
        me = int(rng.choice(spks))
        s = int(rng.integers(a, max(a, b - ctx) + 1))
        idx = np.arange(s, min(b, s + ctx))
        tpos = np.flatnonzero((D.spk[idx] == me) & D.ok_target[idx])
        us = idx[tpos]
        dec = np.stack([D.decoys(int(u), rng) for u in us]) if len(us) else np.zeros((0, N_SAME), np.int64)
        seqs.append(make_seq(D, idx, me))
        tg.append({"pos": tpos, "u": us, "dec": dec, "ep": e, "idx": idx})
    return seqs, tg


def leak_assert(tg: list[dict]) -> None:
    """Output p sees inputs of positions < p only (BOS = output 0); the target utterance sits at input index p (seen
    by outputs > p only)."""
    for t in tg:
        for p, u in zip(t["pos"], t["u"]):
            assert t["idx"][p] == u and u not in t["idx"][:p]


def nce_loss(model, seqs, tg, D: Data, dev):
    bt = collate(seqs, dev)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(bt["x"], bt["role"], bt["dur"], bt["pause"], bt["pad"])
    bi = np.concatenate([np.full(len(t["pos"]), i) for i, t in enumerate(tg)])
    pos = np.concatenate([t["pos"] for t in tg])
    us = np.concatenate([t["u"] for t in tg])
    dec = np.concatenate([t["dec"] for t in tg])
    if len(us) == 0:
        return None, {}
    o = out[torch.from_numpy(bi).to(dev), torch.from_numpy(pos).to(dev)]  # output at p predicts utterance at p
    th = torch.from_numpy(D.thought).to(dev) if not hasattr(D, "_th") else D._th
    tt = th[torch.from_numpy(us).to(dev)]
    dd = th[torch.from_numpy(dec).to(dev)]
    own = torch.cat([(o * tt).sum(-1, keepdim=True), torch.einsum("nd,nkd->nk", o, dd)], 1) / TAU
    other = (o @ tt.T) / TAU
    bt_ = torch.from_numpy(bi).to(dev)
    other = other.masked_fill(bt_[:, None] == bt_[None], float("-inf"))  # in-batch = other recordings only
    logits = torch.cat([own, other], 1)
    loss = F.cross_entropy(logits, torch.zeros(len(us), dtype=torch.long, device=dev))
    acc16 = float((own[:, 0] > own[:, 1:].max(1).values).float().mean())
    return loss, {"loss": float(loss), "acc16": acc16, "targets": len(us)}


@torch.no_grad()
def last_outputs(model, seqs: list[dict], dev, batch: int = 64) -> torch.Tensor:
    """Output after the full given history (position len) for each sequence -> [n, 64]."""
    res = []
    for b in range(0, len(seqs), batch):
        S = seqs[b : b + batch]
        bt = collate(S, dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(bt["x"], bt["role"], bt["dur"], bt["pause"], bt["pad"])
        res.append(out[torch.arange(len(S), device=dev), torch.tensor([len(s["role"]) for s in S], device=dev)])
    return torch.cat(res)


def history(D: Data, u: int, n: int = CTX) -> np.ndarray:
    """Up to n utterances right before u in its recording (time order)."""
    a = int(D.eb[D.ep_of[u]])
    return np.arange(max(a, u - n), u)
