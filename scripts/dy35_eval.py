"""DY-35 evaluation after training (held-out shards only).

Thoughts at evaluation come from the model's own prediction by the pause rule (generation mode): a pause = >= 15
consecutive silence codes (silence.json of the pack: median frame energy <= -45 dBFS); when the first non-silence code
after a pause arrives, z_hat = head(state at the last pause frame) becomes the thought and is held until the next pause
(from the 15th silence frame on: null). Before the first pause: null. Everything runs token by token with a key/value
cache, so every thought is computed from what was heard before it.

  p1   recognition of the true next utterance among 16 same-speaker same-recording utterances (true + 15 decoys >= 60 s
       away, length within x2) on held-out anchors: (head) cosine(z_hat at the last heard frame, z); (b) model b's
       length-normalised log-likelihood of each candidate's codes after the heard codes; (acoustic) cosine of the mean
       WavLM-L6 feature of the candidate (alone) to the mean over the last 2 s heard (alone). Heard 15 s; head also by
       heard 5 / 15 / 30 s on anchors with >= 30 s heard.
  p2a  CE of the utterance frames (predicted from inputs inside the utterance) with the true z, z of another
       same-speaker utterance (>= 60 s, x2), null, predicted z_hat (all after the same heard 15 s).
  gen  continuations (10 s, temperature 1.0, top-p 1.0, one generator seed 1234, EOS banned).
  g1   DY-32 likelihood test of heard content (dy32b_heard_use windows: heard 15 s, own 10 s, same-episode swap
       >= 60 s, separate encoding), thoughts by the pause rule; (b - a) / a in % for seconds 3-10.

    python scripts/dy35_eval.py p1 --ckpt run/best.pt --bckpt b/best.pt --utt utt --held held --out p1.json
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from dy21_train import held_out
from dy32b_channel import SR, Channel, Codebook
from dy32b_gen import rope_at
from dy35_train import EOS, HEARD, N_SAME, load_corpus

from dauys.dy16_model import sample_next
from dauys.dy21_lm import LMConfig, UnitLM
from dauys.dy35_thought import GistEncoder, ThoughtLM

PAUSE = 15
CB = "/data/cb/km-corpus-K2000.npy"


def load_t(path: str, dev):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = LMConfig(**ck["cfg"])
    m = ThoughtLM(cfg, tuple(ck.get("z_layers", ()))).to(dev).eval()
    m.load_state_dict(ck["model"])
    e = GistEncoder(vocab=cfg.vocab - 1).to(dev).eval()
    e.load_state_dict(ck["enc"])
    return m, e, ck


def load_b(path: str, dev):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = UnitLM(LMConfig(**ck["cfg"])).to(dev).eval()
    m.load_state_dict(ck["model"])
    return m, ck


def lm_of(m):
    return m.lm if isinstance(m, ThoughtLM) else m


def make_cache(m, B: int, cap: int, dev) -> list:
    lm = lm_of(m)
    blk = lm.blocks[0]
    d = lm.emb.weight.shape[1]
    shape = (B, blk.heads, cap, d // blk.heads)
    return [(torch.zeros(shape, dtype=torch.bfloat16, device=dev), torch.zeros(shape, dtype=torch.bfloat16, device=dev))
            for _ in lm.blocks]


def blocks(m, x: torch.Tensor, cache: list, pos0: int) -> torch.Tensor:
    """Run the blocks on input embeddings x [B, t, d] at positions pos0..pos0+t-1; keys/values are written into the
    preallocated cache at those positions and attention covers positions < pos0 + t."""
    lm = lm_of(m)
    b, t, d = x.shape
    mask = None
    if t > 1:
        i = torch.arange(t, device=x.device)[:, None] + pos0
        mask = torch.arange(pos0 + t, device=x.device)[None] <= i
    for j, blk in enumerate(lm.blocks):
        q, k, v = blk.qkv(blk.n1(x)).view(b, t, 3, blk.heads, d // blk.heads).permute(2, 0, 3, 1, 4)
        q, k = rope_at(q, pos0), rope_at(k, pos0)
        K, V = cache[j]
        K[:, :, pos0 : pos0 + t] = k.to(K.dtype)
        V[:, :, pos0 : pos0 + t] = v.to(V.dtype)
        a = F.scaled_dot_product_attention(q.to(K.dtype), K[:, :, : pos0 + t], V[:, :, : pos0 + t], attn_mask=mask)
        x = x + blk.out(a.transpose(1, 2).reshape(b, t, d))
        g, u = blk.up(blk.n2(x)).chunk(2, -1)
        x = x + blk.down(F.silu(g) * u)
    return lm.norm(x)


def embed(m, ids: torch.Tensor, z: torch.Tensor | None, has: torch.Tensor | None) -> torch.Tensor:
    if isinstance(m, ThoughtLM):
        b, t = ids.shape
        return m.lm.emb(ids) + m.thought_input(z, has, b, t)
    return m.emb(ids)


@torch.no_grad()
def roll(m, prefix: torch.Tensor, sil: torch.Tensor, n_gen: int = 0, force=None,
         gen: torch.Generator | None = None, temp: float = 1.0, top_p: float = 1.0, near_bank: torch.Tensor | None = None,
         cap: int = 0):
    """Token-by-token run over prefix [B, P] (teacher forced) and n_gen sampled tokens, thoughts by the pause rule.
    force: [B, 64] thought set at the first generated token (or 'pred' handled by the caller via force=None and
    near_bank). Returns (tokens [B, P + n_gen], last states [B, d], per-position logits of the prefix targets [B, P-1]
    as log-probs of prefix[:, 1:] , cache, info)."""
    dev = prefix.device
    B, P = prefix.shape
    th = isinstance(m, ThoughtLM)
    cur = torch.zeros(B, 64, device=dev)
    has = torch.zeros(B, dtype=torch.bool, device=dev)
    run = torch.zeros(B, dtype=torch.long, device=dev)
    cache = make_cache(m, B, max(cap, P + n_gen), dev)
    toks = [prefix[:, i] for i in range(P)]
    lp = torch.zeros(B, P - 1, device=dev)
    h_prev = None
    switches = torch.zeros(B, dtype=torch.long, device=dev)
    eos = lm_of(m).head.out_features - 1
    for t in range(P + n_gen):
        x = toks[t]
        if th:
            is_sil = sil[x]
            ended = (~is_sil) & (run >= PAUSE) & (h_prev is not None)
            run = torch.where(is_sil, run + 1, torch.zeros_like(run))
            has = has & (run < PAUSE)
            if h_prev is not None and bool(ended.any()):
                zh = m.zhat(h_prev)
                cur = torch.where(ended[:, None], zh, cur)
                has = has | ended
                switches += ended.long()
            if t == P and n_gen and force is not None:
                if isinstance(force, str):  # "pred" or "near": z_hat at the last prompt frame (or its nearest neighbour)
                    f = m.zhat(h_prev)
                    if force == "near":
                        f = near_bank[(f @ near_bank.T).argmax(1)]
                else:
                    f = force
                cur, has, run = f.float(), torch.ones_like(has), torch.zeros_like(run)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            e = embed(m, x[:, None], cur[:, None] if th else None, has[:, None] if th else None)
            h = blocks(m, e, cache, t)[:, -1].float()
        h_prev = h
        lg = lm_of(m).logits(h)
        if t < P - 1:
            lp[:, t] = torch.log_softmax(lg, -1).gather(1, prefix[:, t + 1 : t + 2])[:, 0]
        elif t >= P - 1 and t < P + n_gen - 1:
            lg[:, eos] = -float("inf")
            toks.append(sample_next(lg, temp, top_p, gen))
    return torch.stack(toks, 1), h_prev, lp, cache, {"switches": switches.cpu().tolist(), "cur": cur, "has": has}


def sil_mask(path: str, dev) -> torch.Tensor:
    s = json.loads(Path(path).read_text())
    m = torch.zeros(2001, dtype=torch.bool, device=dev)
    m[torch.tensor(s["codes"])] = True
    return m


def chunk_codes(ch: Channel, cb: Codebook, wavs: list[np.ndarray]) -> list[torch.Tensor]:
    """Each waveform encoded alone (+80 zero samples, so 320 m samples -> m frames, as dy35_utt.py)."""
    x = [np.concatenate([w.astype(np.float32), np.zeros(80, np.float32)]) for w in wavs]
    return [cb.encode(f) for f in ch.feats(x, batch=16)]


# ---------------------------------------------------------------- anchors on held-out
def anchors(va, n: int, seed: int):
    rng = np.random.default_rng(seed)
    ids = np.flatnonzero(va.ok_anchor)
    ids = np.sort(rng.choice(ids, min(n, len(ids)), replace=False))
    out = []
    for u in ids:
        g = va.groups[va.grp[u]]
        pool = rng.permutation(g)
        m = va.eligible_mask(np.array([u]), pool)[0]
        out.append((int(u), [int(d) for d in pool[m][:N_SAME]]))
    return out


def zs(enc, va, us: list[int], dev) -> torch.Tensor:
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        return enc.encode_many([torch.from_numpy(va.codes(u).astype(np.int64)).to(dev) for u in us]).float()


def cmd_p1(a) -> None:
    dev = torch.device("cuda")
    va = load_corpus(sorted(f for f in Path(a.utt).glob("*.npz") if held_out(f.stem)))
    sil = sil_mask(a.silence, dev)
    A = anchors(va, a.n, 2026)
    res = {"anchors": len(A), "anchor_ids": [x[0] for x in A]}
    parts = a.parts.split(",")
    if "head" in parts:
        m, enc, ck = load_t(a.ckpt, dev)
        res.update({"step": ck["step"], "val": ck["val"]})
        allu = sorted({u for x in A for u in [x[0]] + x[1]})
        zz = zs(enc, va, allu, dev)
        Z = {u: zz[i] for i, u in enumerate(allu)}

        def head_acc(sub, H: int, null: bool = False) -> tuple[float, int, list]:
            hit, ranks = 0, []
            for b0 in range(0, len(sub), a.batch):
                S = sub[b0 : b0 + a.batch]
                pre = torch.stack([torch.from_numpy(va.stream[va.st[u] - H : va.st[u]].astype(np.int64)) for u, _ in S]).to(dev)
                if null:
                    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                        h = m.states(pre)[:, -1]
                else:
                    _, h, _, _, _ = roll(m, pre, sil)
                zh = m.zhat(h)
                C = torch.stack([torch.stack([Z[u]] + [Z[d] for d in ds]) for u, ds in S])
                s = torch.einsum("bd,bcd->bc", zh, C)
                hit += int((s[:, 0] > s[:, 1:].max(1).values).sum())  # strict: a tie is a miss
                ranks += (s[:, 1:] >= s[:, :1]).sum(1).cpu().tolist()
            return hit / len(sub), len(sub), ranks

        acc, n, ranks = head_acc(A, HEARD)
        res["head_15s"] = {"acc16": acc, "n": n, "ci95": wilson(round(acc * n), n), "mean_rank": float(np.mean(ranks))}
        acc, n, _ = head_acc(A, HEARD, null=True)
        res["head_15s_heard_without_thoughts"] = {"acc16": acc, "n": n, "ci95": wilson(round(acc * n), n)}
        sub = [x for x in A if va.sin[x[0]] >= 1500]
        res["head_by_heard"] = {}
        for H in (250, 750, 1500):
            acc, n, _ = head_acc(sub, H)
            res["head_by_heard"][f"{H // 50}s"] = {"acc16": acc, "n": n, "ci95": wilson(round(acc * n), n)}
        print(json.dumps({k: v for k, v in res.items() if k != "anchor_ids"}), flush=True)
        del m, enc
        torch.cuda.empty_cache()
    # model b: length-normalised log-likelihood of each candidate after the heard 15 s
    if "b" in parts:
        p1_b(a, va, A, res, dev)
    if "acoustic" in parts:
        p1_acoustic(a, va, A, res)
    res["chance"] = 1 / 16
    Path(a.out).write_text(json.dumps(res, indent=1))
    print("P1 DONE", json.dumps({k: v for k, v in res.items() if k != "anchor_ids"}), flush=True)


def p1_b(a, va, A, res, dev) -> None:
    bm, bck = load_b(a.bckpt, dev)
    hit = 0
    for b0 in range(0, len(A), 2):
        S = A[b0 : b0 + 2]
        seqs, lens = [], []
        for u, ds in S:
            pre = va.stream[va.st[u] - HEARD : va.st[u]].astype(np.int64)
            for c in [u] + ds:
                seqs.append(np.concatenate([pre, va.codes(c).astype(np.int64)]))
                lens.append(len(va.codes(c)))
        L = max(map(len, seqs))
        x = torch.zeros(len(seqs), L, dtype=torch.long)
        for i, s_ in enumerate(seqs):
            x[i, : len(s_)] = torch.from_numpy(s_)
        x = x.to(dev)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            h = bm.states(x[:, :-1])
        lp = torch.log_softmax(bm.logits(h), -1).gather(2, x[:, 1:, None])[..., 0]
        sc = []
        for i, l in enumerate(lens):
            sc.append(float(lp[i, HEARD - 1 : HEARD - 1 + l].mean()))
        sc = np.array(sc).reshape(len(S), 16)
        hit += int((sc[:, 0] > sc[:, 1:].max(1)).sum())
    res["model_b_15s"] = {"acc16": hit / len(A), "n": len(A), "ci95": wilson(hit, len(A)), "b_step": bck.get("step")}
    print(json.dumps(res["model_b_15s"]), flush=True)
    del bm
    torch.cuda.empty_cache()


def p1_acoustic(a, va, A, res) -> None:
    ch = Channel()
    piece_audio = PieceAudio(a.held, a.utt)
    hit = 0
    for b0 in range(0, len(A), 8):
        S = A[b0 : b0 + 8]
        wavs = []
        for u, ds in S:
            wavs.append(piece_audio.utt_wav(va, u, -100, 0))  # last 2 s before the utterance
            for c in [u] + ds:
                wavs.append(piece_audio.utt_wav(va, c, 0, None))
        fs = ch.feats([w.astype(np.float32) / 32768.0 for w in wavs], batch=32)
        mu = torch.stack([F.normalize(f.mean(0), dim=0) for f in fs]).view(len(S), 17, -1)
        s = torch.einsum("bd,bcd->bc", mu[:, 0], mu[:, 1:])
        hit += int((s[:, 0] > s[:, 1:].max(1).values).sum())
    res["acoustic_15s"] = {"acc16": hit / len(A), "n": len(A), "ci95": wilson(hit, len(A)),
                           "rule": "cosine of mean WavLM-L6 feature (candidate alone) to mean over the last 2 s heard (alone)"}


class PieceAudio:
    """Raw audio of utterances of the held-out corpus (va.file_idx -> shard, stream offset -> piece)."""

    def __init__(self, held: str, utt: str) -> None:
        self.held = Path(held)
        self.files = sorted(f for f in Path(utt).glob("*.npz") if held_out(f.stem))
        self.meta = []
        pos = 0
        for f in self.files:
            z = np.load(f)
            idx = np.load(self.held / f.stem / "index.npy")
            segs = np.memmap(self.held / f.stem / "segs.bin", dtype=np.int16, mode="r")
            for k, (st, n) in enumerate(z["index"]):
                if n < 16:
                    continue
                self.meta.append((pos, n, segs, int(idx[int(z["piece_orig"][k])][0])))
                pos += n + 1
        self.starts = np.array([x[0] for x in self.meta])

    def utt_wav(self, va, u: int, ds: int, de) -> np.ndarray:
        """Audio of frames [st + ds, (en if de is None else st + de)) of utterance u."""
        s = va.st[u] + ds
        e = va.en[u] if de is None else va.st[u] + de
        k = int(np.searchsorted(self.starts, va.st[u], side="right") - 1)
        p0, n, segs, a0 = self.meta[k]
        assert p0 <= s and e <= p0 + n
        return np.asarray(segs[a0 + (s - p0) * 320 : a0 + (e - p0) * 320], np.int16)


def wilson(k: int, n: int) -> list[float]:
    if n == 0:
        return [0.0, 0.0]
    p, z = k / n, 1.96
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 4), round(c + h, 4)]


def cmd_p2a(a) -> None:
    dev = torch.device("cuda")
    va = load_corpus(sorted(f for f in Path(a.utt).glob("*.npz") if held_out(f.stem)))
    sil = sil_mask(a.silence, dev)
    A = anchors(va, a.n, 2027)
    m, enc, ck = load_t(a.ckpt, dev)
    out = {k: [] for k in ("true", "swap", "null", "pred")}
    for b0 in range(0, len(A), a.batch):
        S = A[b0 : b0 + a.batch]
        pre = torch.stack([torch.from_numpy(va.stream[va.st[u] - HEARD : va.st[u]].astype(np.int64)) for u, _ in S]).to(dev)
        L = max(len(va.codes(u)) for u, _ in S)
        _, h, _, cache, _ = roll(m, pre, sil, cap=HEARD + L)
        zt = zs(enc, va, [u for u, _ in S], dev)
        zw = zs(enc, va, [ds[0] for _, ds in S], dev)
        zp = m.zhat(h)
        x = torch.full((len(S), L), 0, dtype=torch.long)
        mask = torch.zeros(len(S), L - 1, dtype=torch.bool)
        for i, (u, _) in enumerate(S):
            c = va.codes(u).astype(np.int64)
            x[i, : len(c)] = torch.from_numpy(c)
            mask[i, : len(c) - 1] = True
        x, mask = x.to(dev), mask.to(dev)
        for k, z in (("true", zt), ("swap", zw), ("null", None), ("pred", zp)):
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                hasv = torch.full((len(S), L), z is not None, dtype=torch.bool, device=dev)
                zin = (z if z is not None else torch.zeros(len(S), 64, device=dev))[:, None].expand(-1, L, -1)
                e = embed(m, x, zin, hasv)
                hh = blocks(m, e, cache, HEARD)
            l = F.cross_entropy(m.logits(hh[:, :-1]).flatten(0, 1), x[:, 1:].flatten(), reduction="none").view(len(S), L - 1)
            out[k] += ((l * mask).sum(1) / mask.sum(1)).cpu().tolist()
        print(json.dumps({"done": b0 + len(S)}), flush=True)
    r = {k: float(np.mean(v)) for k, v in out.items()}
    t, w = np.array(out["true"]), np.array(out["swap"])
    rel = (w - t) / t
    r.update({"n": len(t), "rel_swap_minus_true_pct": float(100 * (r["swap"] - r["true"]) / r["true"]),
              "rel_per_utt_pct_mean_ci95": [float(100 * rel.mean()), float(100 * (rel.mean() - 1.96 * rel.std(ddof=1) / math.sqrt(len(rel)))),
                                            float(100 * (rel.mean() + 1.96 * rel.std(ddof=1) / math.sqrt(len(rel))))],
              "share_true_lt_swap": float((t < w).mean()), "step": ck["step"],
              "notes": "heard 15 s before the utterance with thoughts by the pause rule; the utterance's frames all carry "
                       "the given thought; CE over utterance frames predicted from inputs inside the utterance"})
    Path(a.out).write_text(json.dumps(r, indent=1))
    print("P2A DONE", json.dumps(r), flush=True)


def write_gen(ch, cb, outdir: Path, names, toks, n, info):
    outdir.mkdir(parents=True, exist_ok=True)
    gen = toks[:, -n:]
    for pid, c in zip(names, gen):
        sf.write(outdir / f"{pid}.wav", ch.vocode(cb.decode(c)), SR)
    info["codes"] = {pid: r.tolist() for pid, r in zip(names, gen.cpu().numpy())}
    (outdir / "info.json").write_text(json.dumps(info, indent=1))


def cmd_gen(a) -> None:
    dev = torch.device("cuda")
    sil = sil_mask(a.silence, dev)
    ch, cb = Channel(), Codebook(CB)
    n = 500
    samp = {"temperature": 1.0, "top_p": 1.0, "seed": 1234, "eos_banned": True, "batch": a.batch,
            "generator": "one per folder (seed 1234), batches in prompt order"}
    base = {"sampling": samp, "generated_s": 10.0, "units": "WavLM-Large layer 6, 50 Hz, K2000 corpus codebook, prompt wav encoded alone",
            "decoder": "centroid lookup -> kNN-VC prematched HiFi-GAN, 16 kHz", "pause_rule": json.loads(Path(a.silence).read_text())["rule"] + "; pause = 15 frames"}

    def prompts(folder: str, pat: str):
        fs = sorted(Path(folder).glob(pat))
        names = [f.stem.split("_", 1)[1] for f in fs]
        wavs = [sf.read(f, dtype="float32")[0] for f in fs]
        codes = chunk_codes(ch, cb, [w * 1.0 for w in wavs])
        L = min(len(c) for c in codes)
        x = torch.stack([torch.cat([torch.tensor([EOS], device=dev), c[-L:]]) for c in codes]).long()
        return names, x, L

    def batched(m, x, **kw):
        g = torch.Generator(device=dev).manual_seed(1234)
        outs, sw = [], []
        for b0 in range(0, len(x), a.batch):
            kk = {k: (v[b0 : b0 + a.batch] if k == "force" and torch.is_tensor(v) else v) for k, v in kw.items()}
            t, _, _, _, info = roll(m, x[b0 : b0 + a.batch], sil, n_gen=n, gen=g, **kk)
            outs.append(t)
            sw += info["switches"]
        return torch.cat(outs), sw

    modes = a.modes.split(",")
    if any(md.startswith("t-") for md in modes):
        m, enc, ck = load_t(a.ckpt, dev)
        tinfo = {"checkpoint": a.ckpt, "step": ck["step"], "val": ck["val"], "config": ck["cfg"]}
    if any(md in ("t-pred", "t-oracle", "t-swap", "t-near", "b-turn") for md in modes):
        names, x, L = prompts(a.pack, "prompt_*.wav")
        cz = np.load(Path(a.pack) / "codes.npz")
    for md in modes:
        t0 = time.time()
        if md == "b-turn":
            bm, bck = load_b(a.bckpt, dev)
            toks, sw = batched(bm, x)
            write_gen(ch, cb, Path(a.out) / md, names, toks, n, {**base, "model": "DY-32 b", "checkpoint": a.bckpt,
                      "step": bck.get("step"), "val": bck.get("val"), "prompt_seen_s": L / 50, "thought": "none"})
            del bm
            torch.cuda.empty_cache()
        elif md in ("t-pred", "t-oracle", "t-swap", "t-near"):
            force, bank = "pred", None
            if md in ("t-oracle", "t-swap"):
                key = "real" if md == "t-oracle" else "swap"
                force = zs_codes(enc, [cz[f"{key}_{i}"] for i in names], dev)
            if md == "t-near":
                bank = torch.load(a.bank, map_location=dev)["z"].float()
                force = "near"
            toks, sw = batched(m, x, force=force, near_bank=bank)
            desc = {"t-pred": "z_hat at the last prompt frame, then the pause rule",
                    "t-oracle": "first utterance: z of real_NNN (its alone-encoded codes); then the pause rule",
                    "t-swap": "first utterance: z of swap_NNN; then the pause rule",
                    "t-near": "first utterance: nearest neighbour (cosine) of z_hat among z of 100,000 training utterances; then the pause rule"}[md]
            write_gen(ch, cb, Path(a.out) / md, names, toks, n, {**base, **tinfo, "prompt_seen_s": L / 50, "thought": desc,
                      "thought_switches_after_start": sw})
        elif md in ("t-t10", "t-t10-x"):
            nm, xx, LL = prompts(a.t10 if md == "t-t10" else a.t10x, "prompt_*.wav")
            toks, sw = batched(m, xx)
            write_gen(ch, cb, Path(a.out) / md, nm, toks, n, {**base, **tinfo, "prompt_seen_s": LL / 50,
                      "thought": "pause rule over prompt and continuation (null before the first pause)", "thought_switches": sw})
        print(json.dumps({"mode": md, "s": round(time.time() - t0, 1)}), flush=True)
    print("GEN DONE", flush=True)


def zs_codes(enc, codes: list[np.ndarray], dev) -> torch.Tensor:
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        return enc.encode_many([torch.from_numpy(c.astype(np.int64)).to(dev) for c in codes]).float()


def cmd_bank(a) -> None:
    dev = torch.device("cuda")
    files = sorted(f for f in Path(a.utt).glob("*.npz") if not held_out(f.stem))
    tr = load_corpus(files)
    rng = np.random.default_rng(7)
    us = np.sort(rng.choice(tr.n, min(100000, tr.n), replace=False))
    m, enc, ck = load_t(a.ckpt, dev)
    z = []
    for b0 in range(0, len(us), 4096):
        z.append(zs_codes(enc, [tr.codes(int(u)) for u in us[b0 : b0 + 4096]], dev))
    torch.save({"z": torch.cat(z).half().cpu(), "utt_ids": us, "step": ck["step"]}, a.bank)
    print("BANK DONE", len(us), flush=True)


def cmd_g1(a) -> None:
    from dy32b_heard_use import BUCKETS, OWN_S, mean_ci, windows

    dev = torch.device("cuda")
    sil = sil_mask(a.silence, dev)
    H = 15
    pieces, wins, winfo = windows(a.held, a.n, 2026, H)
    ch, cb = Channel(), Codebook(CB)
    m, enc, ck = load_t(a.ckpt, dev)

    def chunk(i, off, n):
        segs, st, _, _, _ = pieces[i]
        return np.asarray(segs[st + off : st + off + n], np.float32) / 32768.0

    hs, os_ = H * SR, OWN_S * SR
    res = {"a": [], "b": [], "a_null": [], "b_null": []}
    for b0 in range(0, len(wins), a.batch):
        W = wins[b0 : b0 + a.batch]
        own = chunk_codes(ch, cb, [chunk(i, off + hs, os_) for i, off, *_ in W])
        for c, aud in (("a", [chunk(i, off, hs) for i, off, *_ in W]), ("b", [chunk(j, offb, hs) for _, _, j, offb, _, _ in W])):
            hid = chunk_codes(ch, cb, aud)
            x = torch.stack([torch.cat([torch.tensor([EOS], device=dev), h_, o]) for h_, o in zip(hid, own)]).long()
            P = 1 + len(hid[0])
            _, _, lp, _, _ = roll(m, x, sil)
            res[c].append((-lp[:, P - 1 :]).cpu().numpy())
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                hh = m.states(x[:, :-1])
            l = F.cross_entropy(m.logits(hh[:, P - 1 :]).flatten(0, 1), x[:, P:].flatten(), reduction="none").view(len(W), -1)
            res[c + "_null"].append(l.cpu().numpy())
        print(json.dumps({"done": b0 + len(W)}), flush=True)
    res = {k: np.concatenate(v) for k, v in res.items()}
    out = {**winfo, "step": ck["step"], "heard_s": H, "own_frames": res["a"].shape[1], "results": {}}
    for mode, (ka, kb) in (("predicted_thought", ("a", "b")), ("no_thought_input", ("a_null", "b_null"))):
        r = {}
        for bk, (s, e) in BUCKETS.items():
            xa, xb = res[ka][:, s:e].mean(1), res[kb][:, s:e].mean(1)
            r[bk] = {"ce_a": float(xa.mean()), "ce_b": float(xb.mean()), "rel_b_minus_a_pct": float(100 * (xb.mean() - xa.mean()) / xa.mean()),
                     "share_a_lt_b": float((xa < xb).mean()), "share_a_lt_b_ci95": wilson(int((xa < xb).sum()), len(xa)),
                     "gap_b_minus_a": mean_ci(xb - xa)}
        out["results"][mode] = r
    Path(a.out).write_text(json.dumps(out, indent=1, default=float))
    print("G1 DONE", json.dumps(out["results"], default=float), flush=True)


def cmd_bce(a) -> None:
    """Model b's held-out CE (256 random 2048-frame windows, seed 1234, as dy21_train validation) on the DY-32 b codes
    (20 s windows) and on the DY-35 chunked codes (WavLM alone per utterance / gap chunk) of the same held-out shards."""
    from dy21_train import load_stream

    dev = torch.device("cuda")
    bm, bck = load_b(a.bckpt, dev)
    out = {"b_step": bck.get("step"), "b_val_reported": bck.get("val")}
    streams = {"dy32b_codes": load_stream(sorted(Path(a.b_units).glob("*.npz")), "tok"),
               "dy35_chunked_codes": load_corpus(sorted(f for f in Path(a.utt).glob("*.npz") if held_out(f.stem))).stream}
    for k, st in streams.items():
        T = 2048
        starts = np.random.default_rng(1234).integers(0, len(st) - T - 1, size=256)
        tot = n = 0
        for i in range(0, 256, 32):
            x = torch.from_numpy(np.stack([st[s_ : s_ + T + 1] for s_ in starts[i : i + 32]]).astype(np.int64)).to(dev)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                h = bm.states(x[:, :-1])
            tot += float(F.cross_entropy(bm.logits(h).flatten(0, 1), x[:, 1:].flatten(), reduction="sum"))
            n += x[:, 1:].numel()
        out[k] = {"ce": tot / n, "tokens": int(len(st))}
        print(k, out[k], flush=True)
    Path(a.out).write_text(json.dumps(out, indent=1))
    print("BCE DONE", json.dumps(out), flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    s = p.add_subparsers(dest="cmd", required=True)
    for name in ("p1", "p2a", "gen", "bank", "g1", "bce"):
        x = s.add_parser(name)
        x.add_argument("--ckpt", default="/data/t/run/best.pt")
        x.add_argument("--utt", default="/opt/dlami/nvme/utt")
        x.add_argument("--held", default="/opt/dlami/nvme/held")
        x.add_argument("--silence", default="/data/t/prompts_turn/silence.json")
        x.add_argument("--bckpt", default="/data/b/best.pt")
        x.add_argument("--out", default="")
        x.add_argument("--n", type=int, default=1200)
        x.add_argument("--batch", type=int, default=100)
    s.choices["p1"].add_argument("--parts", default="head,b,acoustic")
    g = s.choices["gen"]
    g.add_argument("--pack", default="/data/t/prompts_turn")
    g.add_argument("--t10", default="/data/t/pack32")
    g.add_argument("--t10x", default="/data/t/pack32x")
    g.add_argument("--bank", default="/data/t/bank.pt")
    g.add_argument("--modes", default="t-pred,t-oracle,t-swap,t-near,b-turn,t-t10,t-t10-x")
    s.choices["bank"].add_argument("--bank", default="/data/t/bank.pt")
    s.choices["bce"].add_argument("--b-units", default="/opt/dlami/nvme/b_held")
    a = p.parse_args()
    {"p1": cmd_p1, "p2a": cmd_p2a, "gen": cmd_gen, "bank": cmd_bank, "g1": cmd_g1, "bce": cmd_bce}[a.cmd](a)


if __name__ == "__main__":
    main()
