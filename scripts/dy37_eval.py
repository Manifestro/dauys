"""DY-37 evaluation after training (held-out shards only).

  pack   reply moments by rule: the next utterance is by "me" (its speaker), the segment before it is by another
         speaker; "me" utterance 4-15 s with a valid thought; >= 10 utterances before it in its piece; >= 40 s of its
         piece heard; a swap utterance (same speaker, >= 60 s away, length within x2, valid thought). 200 moments,
         seed 1234. Writes prompt_NNN.wav (the last 40 s before the reply), real_NNN.wav, swap_NNN.wav, codes.npz
         (training-identical stream codes and speakers of the prompt frames, codes of real / swap), manifest.json.
  p2a    CE of held-out "me" utterance frames (predicted from inputs inside the utterance) after the heard 15 s, with
         the true thought, a swapped one (same speaker, >= 60 s, x2), null and the decided one; unmasked and masked
         input; (swap - true) / true in %.
  p1     recognition of the next "me" utterance among 16 (true + 15 same-speaker, same-recording, >= 60 s, x2) by
         cosine(head output at the last heard frame, thought); 1,200 anchors, Wilson 95 % CI, split by previous speaker.
  gen    10 s per reply moment, temperature 1.0, top-p 1.0, EOS / mask code banned. The model sees the last 1,547
         frames of the 40 s prompt (context 2,048 = prompt + 500 generated + 1). Prompt frames: roles from the
         diarization (me = the reply's speaker), past "me" utterances carry their dictionary thoughts. At the first
         generated frame the thought is chosen by mode: decided (one of the 50 bank thoughts nearest to the head
         output, probability ~ exp(cos / 0.1)), nearest, true (thought of the real reply), swapped (thought of the swap
         utterance); generated frames have role "me" and keep the thought until a pause (>= 15 silence codes: role
         none, null thought); at the next non-silence frame role "me" and a new decided thought. Model b (DY-32 b) on
         the same prompt frames, plain sampling.

    python scripts/dy37_eval.py pack --out /data/t/reply_pack
    python scripts/dy37_eval.py gen --ckpt /data/t/run37/best.pt --pack /data/t/reply_pack --out /data/t/gen37
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
from dy32b_gen import rope_at
from dy35_eval import CB, PieceAudio, load_b, wilson
from dy35_train import EOS, GAP_S, HEARD, N_SAME, load_corpus
from dy37_train import MASK, Bank, attach_dict, prepare, segment_speakers, utterance_keys

from dauys.dy16_model import sample_next
from dauys.dy21_lm import LMConfig
from dauys.dy37_model import ROLE_ME, ROLE_NONE, ROLE_OTHER, RoleThoughtLM

PAUSE = 15
PROMPT_S = 40
SEEN = 1547
N_GEN = 500


def load_val(a, files_all=None):
    files = sorted(Path(a.utt).glob("*.npz"))
    va_files = [f for f in files if held_out(f.stem)]
    va = load_corpus(va_files)
    attach_dict(va, utterance_keys(va_files, files), a.dict, a.table, a.dict_key)
    prepare(va, segment_speakers(va_files, len(va.stream)), 1)
    return va


def load_train(a):
    files = sorted(Path(a.utt).glob("*.npz"))
    tr_files = [f for f in files if not held_out(f.stem)]
    tr = load_corpus(tr_files)
    attach_dict(tr, utterance_keys(tr_files, files), a.dict, a.table, a.dict_key, check_n=0)
    return tr


def load_m(path: str, dev):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = RoleThoughtLM(LMConfig(**ck["cfg"])).to(dev).eval()
    m.load_state_dict(ck["model"])
    return m, ck


def roles_for(va, w0: int, w1: int, me: int) -> np.ndarray:
    fs = va.fspk[w0:w1]
    return np.where(fs < 0, ROLE_NONE, np.where(fs == me, ROLE_ME, ROLE_OTHER)).astype(np.int64)


def history_thoughts(va, w0: int, w1: int, me: int) -> tuple[np.ndarray, np.ndarray]:
    """Dictionary thoughts on frames of past "me" utterances inside [w0, w1)."""
    z = np.zeros((w1 - w0, 64), np.float32)
    has = np.zeros(w1 - w0, bool)
    uo = va.utt_of[w0:w1]
    for u in np.unique(uo[uo >= 0]):
        if va.spk[u] == me and va.zvalid[u]:
            q0, q1 = max(va.st[u] - w0, 0), min(va.en[u] - w0, w1 - w0)
            z[q0:q1] = va.Z[u]
            has[q0:q1] = True
    return z, has


def partner(va, u: int, rng, heard0: int | None = None) -> int:
    g = va.groups[va.grp[u]]
    m = va.eligible_mask(np.array([u]), g)[0] & va.zvalid[g]
    if heard0 is not None:
        m &= (va.en[g] <= heard0) | (va.st[g] >= va.st[u])
    el = g[m]
    return int(rng.choice(el)) if len(el) else -1


# ---------------------------------------------------------------- pack
def cmd_pack(a) -> None:
    va = load_val(a)
    eos = np.flatnonzero(va.stream == EOS)
    piece0 = lambda s: int(eos[np.searchsorted(eos, s) - 1] + 1) if np.searchsorted(eos, s) > 0 else 0
    rng = np.random.default_rng(1234)
    cand = []
    for u in range(va.n):
        L = va.len[u]
        if not (200 <= L <= 750 and va.zvalid[u] and va.sin[u] >= PROMPT_S * 50):
            continue
        if not (va.prev_spk[u] >= 0 and va.prev_spk[u] != va.spk[u] % 64):
            continue
        p0 = piece0(va.st[u])
        if np.searchsorted(va.st, va.st[u]) - np.searchsorted(va.st, p0) < 10:
            continue
        cand.append(u)
    sel = np.sort(rng.choice(cand, min(a.n, len(cand)), replace=False))
    pa = PieceAudio(a.held, a.utt)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    codes, items = {}, []
    for i, u in enumerate(sel):
        u = int(u)
        sw = partner(va, u, rng, heard0=va.st[u] - PROMPT_S * 50)
        if sw < 0:
            continue
        k = f"{len(items):03d}"
        w0 = va.st[u] - PROMPT_S * 50
        sf.write(out / f"prompt_{k}.wav", pa.utt_wav(va, u, -PROMPT_S * 50, 0), 16000, subtype="PCM_16")
        sf.write(out / f"real_{k}.wav", pa.utt_wav(va, u, 0, None), 16000, subtype="PCM_16")
        sf.write(out / f"swap_{k}.wav", pa.utt_wav(va, sw, 0, None), 16000, subtype="PCM_16")
        codes[f"prompt_stream_{k}"] = va.stream[w0 : va.st[u]]
        codes[f"prompt_spk_{k}"] = va.fspk[w0 : va.st[u]]
        codes[f"real_{k}"] = va.codes(u)
        codes[f"swap_{k}"] = va.codes(sw)
        items.append({"id": k, "utt": u, "swap_utt": sw, "speaker": int(va.spk[u]), "prev_speaker_local": int(va.prev_spk[u]),
                      "reply_s": round(va.len[u] / 50, 2), "swap_s": round(va.len[sw] / 50, 2),
                      "reply_t0": float(va.t0[u]), "swap_t0": float(va.t0[sw]), "shard": va.names[int(va.file_idx[u])]})
        if len(items) == a.n:
            break
    np.savez(out / "codes.npz", **codes)
    (out / "manifest.json").write_text(json.dumps({"rule": __doc__.split("pack")[1].split("p2a")[0].strip(), "seed": 1234,
                                                   "n": len(items), "candidates": len(cand), "items": items}, indent=1))
    print("PACK DONE", len(items), "of", len(cand), flush=True)


# ---------------------------------------------------------------- p1 / p2a
def anchors(va, n: int, seed: int):
    rng = np.random.default_rng(seed)
    ids = np.flatnonzero(va.ok_anchor & va.zvalid)
    out = []
    for u in rng.permutation(ids):
        g = va.groups[va.grp[u]]
        pool = rng.permutation(g)
        m = va.eligible_mask(np.array([u]), pool)[0] & va.zvalid[pool]
        m &= (va.en[pool] <= va.st[u] - HEARD) | (va.st[pool] >= va.st[u])
        ds = pool[m][:N_SAME]
        if len(ds) == N_SAME:
            out.append((int(u), [int(d) for d in ds]))
        if len(out) == n:
            break
    return sorted(out)


def batch_inputs(va, S, dev, heard: int = HEARD, extra=None, masked=False):
    """[B, heard + L] inputs (heard part + the utterance), roles, history thoughts; utterance frames get the thought
    given per item (extra[i] or None = null) and are optionally masked."""
    L = max(va.len[u] for u in S) if extra is not None else 0
    B = len(S)
    T = heard + L
    x = np.zeros((B, T), np.int64)
    r = np.zeros((B, T), np.int64)
    z = np.zeros((B, T, 64), np.float32)
    has = np.zeros((B, T), bool)
    tmask = np.zeros((B, T), bool)
    for i, u in enumerate(S):
        w0 = va.st[u] - heard
        me = int(va.spk[u])
        n = heard + (va.len[u] if extra is not None else 0)
        x[i, :n] = va.stream[w0 : w0 + n]
        r[i, :n] = roles_for(va, w0, w0 + n, me)
        z[i, :heard], has[i, :heard] = history_thoughts(va, w0, w0 + heard, me)
        if extra is not None:
            if extra[i] is not None:
                z[i, heard:n] = extra[i]
                has[i, heard:n] = True
            if masked:
                x[i, heard:n] = MASK
            tmask[i, heard : n - 1] = True  # input and target both inside the utterance
    t = lambda v: torch.from_numpy(v).to(dev)
    return t(x), t(r), t(z), t(has), t(tmask)


def cmd_p1(a) -> None:
    dev = torch.device("cuda")
    va = load_val(a)
    m, ck = load_m(a.ckpt, dev)
    A = anchors(va, a.n, 2026)
    Z = torch.from_numpy(va.Z).to(dev)
    hits = []
    for b0 in range(0, len(A), a.batch):
        S = A[b0 : b0 + a.batch]
        x, r, z, has, _ = batch_inputs(va, [u for u, _ in S], dev)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            h = m.states(x, r, z, has)[:, -1]
        zh = m.zhat(h)
        C = torch.stack([Z[[u] + ds] for u, ds in S])
        s = torch.einsum("bd,bcd->bc", zh, C)
        hits += (s[:, 0] > s[:, 1:].max(1).values).cpu().tolist()
    hits = np.array(hits)
    prev = np.array([va.prev_spk[u] for u, _ in A])
    own = np.array([va.spk[u] % 64 for u, _ in A])
    res = {"anchors": len(A), "step": ck["step"], "acc16": float(hits.mean()), "ci95": wilson(int(hits.sum()), len(hits)),
           "chance": 1 / 16, "by_prev_speaker": {}}
    for k, sel in (("same", prev == own), ("different", (prev >= 0) & (prev != own)), ("none_in_piece", prev < 0)):
        res["by_prev_speaker"][k] = {"acc16": float(hits[sel].mean()) if sel.any() else 0.0, "n": int(sel.sum()),
                                     "ci95": wilson(int(hits[sel].sum()), int(sel.sum()))}
    Path(a.out).write_text(json.dumps(res, indent=1))
    print("P1 DONE", json.dumps(res), flush=True)


def cmd_p2a(a) -> None:
    dev = torch.device("cuda")
    va = load_val(a)
    tr = load_train(a)
    m, ck = load_m(a.ckpt, dev)
    bank = Bank(tr, len(ck["bank_ids"]), dev)
    bank.ids = ck["bank_ids"]
    bank.z = torch.from_numpy(tr.Z[bank.ids]).to(dev)
    bank.ep = torch.full((len(bank.ids),), -1, dtype=torch.long, device=dev)  # held-out recordings are never in the bank
    rng = np.random.default_rng(2027)
    ids = [u for u, _ in anchors(va, a.n * 2, 2027)]
    items = []
    for u in ids:
        p = partner(va, u, rng)
        if p >= 0:
            items.append((u, p))
        if len(items) == a.n:
            break
    out = {f"{k}{s}": [] for k in ("true", "swap", "null", "decided") for s in ("", "_masked")}
    for b0 in range(0, len(items), a.batch):
        S = items[b0 : b0 + a.batch]
        us = [u for u, _ in S]
        x0, r0, z0, h0, _ = batch_inputs(va, us, dev)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            zh = m.zhat(m.states(x0, r0, z0, h0)[:, -1])
        zd, _ = bank.nearest(zh, torch.zeros(len(S), dtype=torch.long, device=dev))
        variants = {"true": [va.Z[u] for u in us], "swap": [va.Z[p] for _, p in S], "null": [None] * len(S),
                    "decided": list(zd.cpu().numpy())}
        for masked in (False, True):
            for k, ex in variants.items():
                x, r, z, has, tm = batch_inputs(va, us, dev, extra=ex, masked=masked)
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    h = m.states(x[:, :-1], r[:, :-1], z[:, :-1], has[:, :-1])
                tgt = torch.from_numpy(np.stack([np.pad(va.stream[va.st[u] - HEARD : va.en[u]], (0, x.shape[1] - HEARD - va.len[u]))
                                                 for u in us]).astype(np.int64)).to(dev)
                l = F.cross_entropy(m.logits(h.float()).flatten(0, 1), tgt[:, 1:].flatten(), reduction="none").view(len(S), -1)
                mk = tm[:, :-1].float()
                out[k + ("_masked" if masked else "")] += ((l * mk).sum(1) / mk.sum(1)).cpu().tolist()
        print(json.dumps({"done": b0 + len(S)}), flush=True)
    r = {k: float(np.mean(v)) for k, v in out.items()}
    for s in ("", "_masked"):
        t, w = np.array(out["true" + s]), np.array(out["swap" + s])
        rel = (w - t) / t
        r[f"rel_swap_minus_true_pct{s}"] = float(100 * (w.mean() - t.mean()) / t.mean())
        r[f"rel_per_utt_pct_mean_ci95{s}"] = [float(100 * rel.mean()), float(100 * (rel.mean() - 1.96 * rel.std(ddof=1) / math.sqrt(len(rel)))),
                                               float(100 * (rel.mean() + 1.96 * rel.std(ddof=1) / math.sqrt(len(rel))))]
        r[f"share_true_lt_swap{s}"] = float((t < w).mean())
    r.update({"n": len(items), "step": ck["step"], "threshold": "rel_swap_minus_true_pct (unmasked) >= 2 %",
              "notes": "heard 15 s (roles from diarization, me = the utterance's speaker, past me utterances with their "
                       "dictionary thoughts); the utterance's frames carry the given thought; CE over frames predicted "
                       "from inputs inside the utterance; decided = nearest bank thought to the head output"})
    Path(a.out).write_text(json.dumps(r, indent=1))
    print("P2A DONE", json.dumps(r), flush=True)


# ---------------------------------------------------------------- generation
def make_cache(lm, B: int, cap: int, dev) -> list:
    blk = lm.blocks[0]
    d = lm.emb.weight.shape[1]
    shape = (B, blk.heads, cap, d // blk.heads)
    return [(torch.zeros(shape, dtype=torch.bfloat16, device=dev), torch.zeros(shape, dtype=torch.bfloat16, device=dev)) for _ in lm.blocks]


def blocks(lm, x, cache, pos0: int):
    b, t, d = x.shape
    for j, blk in enumerate(lm.blocks):
        q, k, v = blk.qkv(blk.n1(x)).view(b, t, 3, blk.heads, d // blk.heads).permute(2, 0, 3, 1, 4)
        q, k = rope_at(q, pos0), rope_at(k, pos0)
        K, V = cache[j]
        K[:, :, pos0 : pos0 + t] = k.to(K.dtype)
        V[:, :, pos0 : pos0 + t] = v.to(V.dtype)
        a = F.scaled_dot_product_attention(q.to(K.dtype), K[:, :, : pos0 + t], V[:, :, : pos0 + t])
        x = x + blk.out(a.transpose(1, 2).reshape(b, t, d))
        g, u = blk.up(blk.n2(x)).chunk(2, -1)
        x = x + blk.down(F.silu(g) * u)
    return lm.norm(x)


@torch.no_grad()
def roll(m, x, r, z, has, sil, mode: str, first, bank: Bank, gen, n_gen: int = N_GEN, b_model: bool = False):
    """Prompt [B, P] token by token (teacher forced), then n_gen sampled frames. Thoughts for generated frames as in
    the module docstring. Returns tokens [B, P + n_gen] and per item the bank rows chosen."""
    dev = x.device
    B, P = x.shape
    lm = m if b_model else m.lm
    cache = make_cache(lm, B, P + n_gen, dev)
    toks = [x[:, i] for i in range(P)]
    cur = torch.zeros(B, 64, device=dev)
    hold = torch.zeros(B, dtype=torch.bool, device=dev)
    role = torch.full((B,), ROLE_ME, dtype=torch.long, device=dev)
    run = torch.zeros(B, dtype=torch.long, device=dev)
    zero_ep = torch.full((B,), -1, dtype=torch.long, device=dev)
    picks = [[] for _ in range(B)]
    h = None
    for t in range(P + n_gen):
        xt = toks[t]
        if b_model:
            rt = zt = ht = None
        elif t < P:
            rt, zt, ht = r[:, t], z[:, t], has[:, t]
        else:
            s = sil[xt]
            if t == P:
                if mode in ("decided", "nearest"):
                    zz, pk = bank.sample(m.zhat(h), zero_ep, gen) if mode == "decided" else bank.nearest(m.zhat(h), zero_ep)
                    for i in range(B):
                        picks[i].append(int(bank.ids[int(pk[i])]))
                else:
                    zz = first
                cur, hold, run = zz.float(), torch.ones_like(hold), torch.zeros_like(run)
            else:
                start = (~s) & (run >= PAUSE)
                if bool(start.any()):  # speech after a pause: a new decided thought
                    zz, pk = bank.sample(m.zhat(h), zero_ep, gen)
                    cur = torch.where(start[:, None], zz.float(), cur)
                    hold = hold | start
                    for i in torch.flatnonzero(start).tolist():
                        picks[i].append(int(bank.ids[int(pk[i])]))
            run = torch.where(s, run + 1, torch.zeros_like(run))
            hold = hold & (run < PAUSE)
            role = torch.where(run >= PAUSE, torch.full_like(role, ROLE_NONE), torch.full_like(role, ROLE_ME))
            rt, zt, ht = role, cur, hold
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if b_model:
                e = lm.emb(xt[:, None])
            else:
                e = m.inputs(xt[:, None], rt[:, None], zt[:, None], ht[:, None])
            h = blocks(lm, e, cache, t)[:, -1].float()
        if t >= P - 1 and t < P + n_gen - 1:
            lg = lm.logits(h)
            lg[:, EOS:] = -float("inf")  # EOS and the mask code banned
            toks.append(sample_next(lg, 1.0, 1.0, gen))
    return torch.stack(toks, 1), picks


def cmd_gen(a) -> None:
    from dy32b_channel import SR, Channel, Codebook

    dev = torch.device("cuda")
    sil = torch.zeros(2002, dtype=torch.bool, device=dev)
    sil[torch.tensor(json.loads(Path(a.silence).read_text())["codes"])] = True
    pack = Path(a.pack)
    man = json.loads((pack / "manifest.json").read_text())
    cz = np.load(pack / "codes.npz")
    items = man["items"][: a.limit or None]
    B = len(items)
    x = np.stack([np.concatenate([[EOS], cz[f"prompt_stream_{it['id']}"][-(SEEN - 1):]]) for it in items]).astype(np.int64)
    fs = np.stack([np.concatenate([[-1], cz[f"prompt_spk_{it['id']}"][-(SEEN - 1):]]) for it in items])
    me = np.array([it["speaker"] for it in items])[:, None]
    r = np.where(fs < 0, ROLE_NONE, np.where(fs == me, ROLE_ME, ROLE_OTHER)).astype(np.int64)
    m, ck = load_m(a.ckpt, dev)
    tr = load_train(a)
    va = load_val(a)
    zh = np.zeros((B, SEEN, 64), np.float32)
    hh = np.zeros((B, SEEN), bool)
    for i, it in enumerate(items):
        u = it["utt"]
        w0 = va.st[u] - (SEEN - 1)
        zh[i, 1:], hh[i, 1:] = history_thoughts(va, w0, va.st[u], int(va.spk[u]))
        assert np.array_equal(va.stream[w0 : va.st[u]], x[i, 1:])
    bank = Bank(tr, len(ck["bank_ids"]), dev)
    bank.ids = ck["bank_ids"]
    bank.z = torch.from_numpy(tr.Z[bank.ids]).to(dev)
    bank.ep = torch.full((len(bank.ids),), -2, dtype=torch.long, device=dev)
    t_ = lambda v: torch.from_numpy(v).to(dev)
    X, R, Zh, Hh = t_(x), t_(r), t_(zh), t_(hh)
    ch, cb = Channel(), Codebook(CB)
    base = {"step": ck["step"], "val": ck["val"], "checkpoint": a.ckpt, "dict": ck.get("dict"), "sampling": {"temperature": 1.0,
            "top_p": 1.0, "seed": 1234, "eos_and_mask_banned": True}, "generated_s": N_GEN / 50, "prompt_seen_s": (SEEN - 1) / 50,
            "prompt_wav_s": PROMPT_S, "units": "WavLM-Large layer 6, 50 Hz, K2000; prompt = training-identical stream codes",
            "decoder": "centroid lookup -> kNN-VC prematched HiFi-GAN, 16 kHz", "pause_rule": "15 silence codes"}
    for md in a.modes.split(","):
        t0 = time.time()
        g = torch.Generator(device=dev).manual_seed(1234)
        outs, picks = [], []
        for b0 in range(0, B, a.batch):
            sl = slice(b0, b0 + a.batch)
            first = None
            if md in ("true", "swapped"):
                key = "real" if md == "true" else "swap"
                first = torch.from_numpy(np.stack([va.Z[it["utt" if md == "true" else "swap_utt"]] for it in items[sl]])).to(dev)
            if md == "b":
                bm, bck = load_b(a.bckpt, dev)
                tk, pk = roll(bm, X[sl], None, None, None, sil, md, None, bank, g, b_model=True)
                del bm
            else:
                tk, pk = roll(m, X[sl], R[sl], Zh[sl], Hh[sl], sil, md, first, bank, g)
            outs.append(tk[:, -N_GEN:])
            picks += pk
        gen = torch.cat(outs)
        od = Path(a.out) / md
        od.mkdir(parents=True, exist_ok=True)
        for it, c in zip(items, gen):
            sf.write(od / f"{it['id']}.wav", ch.vocode(cb.decode(c)), SR)
        desc = {"decided": "one of the 50 bank thoughts nearest to the head output (probability ~ exp(cos / 0.1))",
                "nearest": "the bank thought nearest to the head output", "true": "the dictionary thought of the real reply",
                "swapped": "the dictionary thought of the swap utterance", "b": "model b (DY-32 b), no thought, plain sampling"}[md]
        info = {**base, "mode": md, "first_thought": desc, "bank_picks_train_utt_ids": {it["id"]: p for it, p in zip(items, picks)},
                "codes": {it["id"]: c.tolist() for it, c in zip(items, gen.cpu().numpy())}}
        if md == "b":
            info.update({"model": "DY-32 b", "checkpoint": a.bckpt})
        (od / "info.json").write_text(json.dumps(info, indent=1, default=str))
        print(json.dumps({"mode": md, "s": round(time.time() - t0, 1)}), flush=True)
    print("GEN DONE", flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    s = p.add_subparsers(dest="cmd", required=True)
    for name in ("pack", "p1", "p2a", "gen"):
        x = s.add_parser(name)
        x.add_argument("--ckpt", default="/data/t/run37/best.pt")
        x.add_argument("--utt", default="/opt/dlami/nvme/utt")
        x.add_argument("--held", default="/opt/dlami/nvme/held")
        x.add_argument("--dict", default="")
        x.add_argument("--table", default="/opt/dlami/nvme/dict37/table.npz")
        x.add_argument("--dict-key", default="thought")
        x.add_argument("--silence", default="/data/t/prompts_turn/silence.json")
        x.add_argument("--bckpt", default="/data/b/best.pt")
        x.add_argument("--out", default="")
        x.add_argument("--n", type=int, default=200)
        x.add_argument("--batch", type=int, default=50)
    s.choices["gen"].add_argument("--pack", default="/data/t/reply_pack")
    s.choices["gen"].add_argument("--modes", default="decided,nearest,true,swapped,b")
    s.choices["gen"].add_argument("--limit", type=int, default=0)
    a = p.parse_args()
    if not a.dict and a.cmd != "pack":
        a.dict = torch.load(a.ckpt, map_location="cpu", weights_only=False)["dict"]
    {"pack": cmd_pack, "p1": cmd_p1, "p2a": cmd_p2a, "gen": cmd_gen}[a.cmd](a)


if __name__ == "__main__":
    main()
