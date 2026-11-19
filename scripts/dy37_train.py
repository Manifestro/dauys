"""DY-37 training: one participant speaks from a chosen thought (src/dauys/dy37_model.py).

Model b's recipe (24 x 1024, K2000 codes at 50 Hz, context 2048, random windows over the joined pieces, CE + JEPA,
AdamW, cosine) on the DY-35 chunked codes and utterance tables (scripts/dy35_utt.py), with:
* THOUGHTS ARE GIVEN: a frozen 64-d unit vector per utterance from a dictionary (``--dict``, DY-36 format: rows of
  ``--table`` = every utterance of the tables; matched by shard / piece / start / end and checked on the codes).
* ROLES: in every window each recording present has one "me" speaker, a fixed function of (recording, sequence
  number): the recording's eligible speakers (>= 16 utterances with a valid thought) in a seed-fixed order, cycled by
  the sequence number. Every input frame gets a role (no speech / me / other) from the diarization of all segments.
* THOUGHT INPUT only on frames of "me" utterances: true z (0.6), "decided" (0.3, from step --pred-from; before:
  true) or null (0.1). Decided = the nearest thought (cosine) in a fixed bank of 100,000 training utterances, other
  recordings only, to the head output of the EMA copy at the last frame before the utterance (detached).
* THOUGHT HEAD before "me" utterances only: InfoNCE tau 0.1, weight 1.0, against 15 "me" utterances of the same
  recording (>= 60 s from the target, length within x2, never inside the heard part) + 16 of other recordings.
* MASK: in 30 % of "me" utterances (independent draw) the INPUT codes inside the utterance are the mask code K + 1
  (targets unchanged).
* FRAME LOSS weight 1.0 on target frames of "me" segments, 0.1 on all other frames.

    python scripts/dy37_train.py --utt /opt/dlami/nvme/utt --dict DICT.npz --table table.npz --out /data/t/run37 \
        --cfg 1024,24,16,4096,2048 --batch 32 --micro 16 --lr 3e-4 --warmup 1000 --val-every 500 --steps 12257
"""

import argparse
import json
import math
import os
import queue
import sys
import threading
import time
import uuid
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from dy21_train import held_out, lr_at
from dy35_train import EOS, GAP_S, HEARD, K, N_OTHER, N_SAME, UMAX, UMIN, Corpus, load_corpus, rss_gb, wilson_lo

from dauys.dy21_lm import JEPA_WEIGHT, LMConfig, jepa_target
from dauys.dy37_model import NCE_WEIGHT, ROLE_ME, ROLE_NONE, ROLE_OTHER, RoleThoughtLM, nce, param_counts

MASK = K + 1
VOCAB = K + 2
MAX_ANCHORS = 8
P_TRUE, P_DEC = 0.6, 0.3
P_MASK = 0.3
W_OTHER = 0.1
BANK_N = 100000
MIN_ME_UTTS = N_SAME + 1
EXPERIMENT = "DY-37: замысел и речь, перезапуск — участник разговора говорит по выбранному замыслу; замыслы из словаря, который ловит содержание; без текста"


# ---------------------------------------------------------------- data
def segment_speakers(files: list[Path], n_stream: int) -> np.ndarray:
    """Global speaker (episode * 64 + speaker) of every stream frame from ALL diarized segments (any length), -1 for
    frames outside segments and EOS. Offsets as dy35_train.load_corpus."""
    fs = np.full(n_stream, -1, np.int32)
    pos, ep_base = 0, 0
    for f in files:
        z = np.load(f)
        index = z["index"]
        off = np.full(len(index), -1, np.int64)
        for k, (a, n) in enumerate(index):
            if n < 16:
                continue
            off[k] = pos
            pos += n + 1
        g = (ep_base + z["piece_ep"][z["seg_piece"]].astype(np.int64)) * 64 + z["seg_spk"].astype(np.int64)
        o = off[z["seg_piece"]]
        for p0, s, e, sp in zip(o.tolist(), z["seg_s"].tolist(), z["seg_e"].tolist(), g.tolist()):
            if p0 >= 0:
                fs[p0 + s : p0 + e] = sp
        ep_base += len(json.loads(str(z["eps"])))
    assert pos == n_stream, (pos, n_stream)
    return fs


def utterance_keys(files: list[Path], all_files: list[Path]) -> np.ndarray:
    """[n, 4] (file index among all shards, piece, start, end) of every utterance, in load_corpus order."""
    fidx = {f.stem: i for i, f in enumerate(all_files)}
    out = []
    for f in files:
        z = np.load(f)
        index = z["index"]
        ok = np.array([n >= 16 for _, n in index])
        L = z["seg_e"] - z["seg_s"]
        m = (L >= UMIN) & (L <= UMAX) & ok[z["seg_piece"]]
        k = np.stack([np.full(int(m.sum()), fidx[f.stem]), z["seg_piece"][m], z["seg_s"][m], z["seg_e"][m]], 1)
        out.append(k.astype(np.int64))
    return np.concatenate(out)


def attach_dict(c: Corpus, keys: np.ndarray, dict_path: str, table_path: str, key: str, check_n: int = 3000) -> dict:
    """c.Z [n, 64] float32 unit vectors, c.zvalid [n] from the dictionary rows matched to the utterances."""
    t = np.load(table_path)
    tk = np.stack([t["fi"], t["piece"], t["s"], t["e"]], 1).astype(np.int64)
    enc = lambda a: ((a[:, 0] * (1 << 20) + a[:, 1]) * (1 << 21) + a[:, 2]) * (1 << 11) + (a[:, 3] - a[:, 2])
    tkey, ckey = enc(tk), enc(keys)
    order = np.argsort(tkey)
    pos = np.searchsorted(tkey[order], ckey)
    pos = np.clip(pos, 0, len(order) - 1)
    rows = order[pos]
    found = tkey[rows] == ckey
    assert found.all(), f"{int((~found).sum())} utterances not in the table"
    rng = np.random.default_rng(0)
    co = t["code_off"]
    tc = t["codes"]
    for u in rng.choice(c.n, min(check_n, c.n), replace=False):
        r = rows[u]
        assert np.array_equal(tc[co[r] : co[r + 1]], c.codes(int(u))), ("codes differ", u, r)
    d = np.load(dict_path)
    Z = d[key][rows].astype(np.float32)
    nrm = np.linalg.norm(Z, axis=1)
    valid = d["valid"][rows] if "valid" in d.files else np.ones(c.n, bool)
    valid = valid & np.isfinite(nrm) & (nrm > 1e-6)
    c.Z = Z / np.maximum(nrm, 1e-6)[:, None]
    c.zvalid = valid
    return {"dict": dict_path, "key": key, "rows_matched": int(c.n), "codes_checked": int(min(check_n, c.n)),
            "valid_share": round(float(valid.mean()), 4), "norm_before_pct1_50_99": np.percentile(nrm, [1, 50, 99]).round(4).tolist(),
            "dict_info": str(d["info"]) if "info" in d.files else ""}


def prepare(c: Corpus, fspk: np.ndarray, seed: int) -> None:
    """Me-speaker lists per recording (seed-fixed order) and anchor eligibility for the role setting."""
    c.fspk = fspk
    nep = int(c.ep.max()) + 1 if c.n else 0
    rng = np.random.default_rng(seed)
    lists: list[np.ndarray] = [np.zeros(0, np.int64)] * nep
    by_ep: dict[int, list] = {}
    for g in c.groups:
        s = int(c.spk[g[0]])
        by_ep.setdefault(s // 64, []).append((s, int(c.zvalid[g].sum())))
    for e, sp in by_ep.items():
        el = sorted(s for s, n in sp if n >= MIN_ME_UTTS) or sorted(s for s, n in sp)
        lists[e] = rng.permutation(np.array(el, np.int64))
    c.me_lists = lists


def me_of(c: Corpus, ep: int, seq: int) -> int:
    """The "me" speaker of recording ep in sequence number seq: a fixed function, independent of the window."""
    lst = c.me_lists[ep]
    return int(lst[seq % len(lst)]) if len(lst) else -1


def frame_roles(c: Corpus, fs: np.ndarray, seq: int) -> tuple[np.ndarray, set]:
    eps = np.unique(fs[fs >= 0] // 64)
    me = {me_of(c, int(e), seq) for e in eps}
    r = np.where(fs < 0, ROLE_NONE, np.where(np.isin(fs, list(me)), ROLE_ME, ROLE_OTHER)).astype(np.int64)
    return r, me


class Builder:
    """Micro-batches on the CPU (numpy). Window starts uniform over the stream (as dy21_train)."""

    def __init__(self, c: Corpus, T: int, rng: np.random.Generator) -> None:
        self.c, self.T, self.rng = c, T, rng
        self.valid_ids = np.flatnonzero(c.zvalid)

    def build(self, starts: np.ndarray, seqs: np.ndarray, pred_on: bool, val: bool = False, p_mask: float = P_MASK,
              max_anchors: int = MAX_ANCHORS) -> dict:
        c, T, rng = self.c, self.T, self.rng
        B = len(starts)
        ids = np.stack([c.stream[s : s + T + 1] for s in starts]).astype(np.int64)
        x, y = ids[:, :-1].copy(), ids[:, 1:].copy()
        roles = np.zeros((B, T), np.int64)
        wts = np.full((B, T), W_OTHER, np.float32)
        upos = np.full((B, T), -1, np.int64)
        slots = []  # (b, u, mode 0 true 1 decided 2 null, anchor position or -1, masked)
        swap_u = []  # per slot (validation): a same-speaker utterance >= 60 s away, length within x2, valid (-1: none)
        anchors = []  # (b, position, [u] + 15 same + 16 other)
        mes = []
        for b, (w, sq) in enumerate(zip(starts, seqs)):
            fs = c.fspk[w : w + T + 1]
            r, me = frame_roles(c, fs, int(sq))
            roles[b] = r[:T]
            wts[b] = np.where(r[1:] == ROLE_ME, 1.0, W_OTHER)
            mes.append(sorted(me))
            uo = c.utt_of[w : w + T]
            for u in np.unique(uo[uo >= 0]):
                u = int(u)
                if c.spk[u] not in me or not c.zvalid[u]:
                    continue
                q0, q1 = max(c.st[u] - w, 0), min(c.en[u] - w, T)
                ap = int(c.st[u] - 1 - w)
                mode = 0
                if not val:
                    rr = rng.random()
                    mode = 0 if rr < P_TRUE else (1 if rr < P_TRUE + P_DEC else 2)
                    if mode == 1 and (not pred_on or ap < 0):
                        mode = 0
                masked = bool(rng.random() < p_mask)
                if masked:
                    x[b, q0:q1] = MASK
                upos[b, q0:q1] = len(slots)
                slots.append((b, u, mode, ap if ap >= 0 else -1, masked))
                if val:
                    g = c.groups[c.grp[u]]
                    el = g[c.eligible_mask(np.array([u]), g)[0] & c.zvalid[g]]
                    swap_u.append(int(rng.choice(el)) if len(el) else -1)
            lo, hi = np.searchsorted(c.st, w + HEARD), np.searchsorted(c.st, w + T, side="right")
            cand = np.arange(lo, hi)
            cand = cand[c.ok_anchor[cand] & c.zvalid[cand] & np.isin(c.spk[cand], list(me))]
            if len(cand) > max_anchors:
                cand = np.sort(rng.choice(cand, max_anchors, replace=False))
            pools: dict[int, np.ndarray] = {}
            for u in cand:
                u = int(u)
                g = c.grp[u]
                if g not in pools:
                    pools[g] = rng.permutation(c.groups[g])
                pool = pools[g]
                m = c.eligible_mask(np.array([u]), pool)[0] & c.zvalid[pool]
                m &= (c.en[pool] <= w) | (c.st[pool] >= c.st[u])  # never a candidate inside the heard part
                ds = pool[m][:N_SAME]
                if len(ds) < N_SAME:
                    continue
                oth = []
                while len(oth) < N_OTHER:
                    o = rng.choice(self.valid_ids, 64)
                    o = o[(c.ep[o] != c.ep[u]) & ((c.en[o] <= w) | (c.st[o] >= c.st[u]))]
                    oth += o.tolist()
                anchors.append((b, int(c.st[u] - 1 - w), [u] + [int(d) for d in ds] + oth[:N_OTHER]))
        return {"x": x, "y": y, "roles": roles, "wts": wts, "upos": upos, "slots": slots, "swap_u": swap_u,
                "anchors": anchors, "starts": np.asarray(starts), "seqs": np.asarray(seqs), "me": mes}


# ---------------------------------------------------------------- losses
class Bank:
    """Fixed bank of thoughts of training utterances (decided thoughts are real thoughts of other recordings)."""

    def __init__(self, c: Corpus, n: int, dev, seed: int = 7) -> None:
        ids = np.flatnonzero(c.zvalid)
        self.ids = np.sort(np.random.default_rng(seed).choice(ids, min(n, len(ids)), replace=False))
        self.z = torch.from_numpy(c.Z[self.ids]).to(dev)
        self.ep = torch.from_numpy(c.ep[self.ids]).to(dev)

    def nearest(self, zh: torch.Tensor, ep: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        s = zh.float() @ self.z.T
        s = s.masked_fill(self.ep[None] == ep[:, None], -float("inf"))
        pick = s.argmax(1)
        return self.z[pick], pick

    def sample(self, zh: torch.Tensor, ep: torch.Tensor, gen=None, k: int = 50, tau: float = 0.1):
        s = zh.float() @ self.z.T
        s = s.masked_fill(self.ep[None] == ep[:, None], -float("inf"))
        v, ix = s.topk(k, dim=1)
        j = torch.multinomial(torch.softmax(v / tau, -1), 1, generator=gen)
        pick = ix.gather(1, j)[:, 0]
        return self.z[pick], pick


def thought_table(mb: dict, Zg: torch.Tensor, dev, mode_over: list | None = None, z_over: dict | None = None):
    """Per-frame thoughts [B, T, 64] and has [B, T] (False -> null). mode_over: per-slot modes (0 true / 2 null);
    z_over: slot index -> 64-d vector (decided / swapped)."""
    B, T = mb["upos"].shape
    if not mb["slots"]:
        return torch.zeros(B, T, Zg.shape[1], device=dev), torch.zeros(B, T, dtype=torch.bool, device=dev)
    us = torch.tensor([s[1] for s in mb["slots"]], device=dev)
    modes = mode_over if mode_over is not None else [s[2] for s in mb["slots"]]
    mode = torch.tensor(modes, device=dev)
    tab = Zg[us].clone()
    if z_over:
        idx = torch.tensor(list(z_over.keys()), device=dev)
        tab[idx] = torch.stack([z_over[i] for i in z_over]).to(tab.dtype)
    upos = torch.from_numpy(mb["upos"]).to(dev)
    has = (upos >= 0) & (mode[upos.clamp(min=0)] != 2)
    return tab[upos.clamp(min=0)], has


def tensors(mb: dict, dev):
    t = lambda a: torch.from_numpy(a).to(dev, non_blocking=True)
    return t(mb["x"]), t(mb["y"]), t(mb["roles"]), t(mb["wts"])


def decided_for(mb: dict, slot_idx: list[int], h: torch.Tensor, head, bank: Bank, c_ep: torch.Tensor, dev, sample=False, gen=None):
    """Decided thought of the given slots: the bank thought nearest to the head output at the slot's anchor position."""
    b = torch.tensor([mb["slots"][i][0] for i in slot_idx], device=dev)
    p = torch.tensor([mb["slots"][i][3] for i in slot_idx], device=dev)
    u = torch.tensor([mb["slots"][i][1] for i in slot_idx], device=dev)
    zh = head(h[b, p])
    return (bank.sample(zh, c_ep[u], gen) if sample else bank.nearest(zh, c_ep[u]))


def step_loss(model, tgt, mb, Zg, c_ep, bank: Bank, dev, pred_on: bool, debug: dict | None = None):
    x, y, roles, wts = tensors(mb, dev)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        with torch.no_grad():
            zt, ht = thought_table(mb, Zg, dev, mode_over=[2 if s[2] == 2 else 0 for s in mb["slots"]])
            h_tgt = tgt.states(x, roles, zt, ht)
            over = {}
            dec = [i for i, s in enumerate(mb["slots"]) if s[2] == 1 and s[3] >= 0]
            if pred_on and dec:
                zd, pick = decided_for(mb, dec, h_tgt, tgt.zhat, bank, c_ep, dev)
                over = {i: zd[j] for j, i in enumerate(dec)}
                if debug is not None:
                    debug["decided"] = [(mb["slots"][i][1], int(bank.ids[int(pick[j])])) for j, i in enumerate(dec)]
        zin, has = thought_table(mb, Zg, dev, mode_over=[2 if s[2] == 2 else 0 for s in mb["slots"]], z_over=over)
        h = model.states(x, roles, zin, has)
        l = F.cross_entropy(model.logits(h).flatten(0, 1), y.flatten(), reduction="none").view_as(y)
        ce = (l * wts).sum() / wts.sum()
        me = wts == 1.0
        target = jepa_target(h_tgt)
        tt = target.shape[1]
        pred = F.normalize(model.lm.pred(h[:, :tt]).float(), dim=-1)
        jepa = (1 - (pred * F.normalize(target, dim=-1)).sum(-1)).mean()
        if mb["anchors"]:
            A = mb["anchors"]
            bb = torch.tensor([a[0] for a in A], device=dev)
            pp = torch.tensor([a[1] for a in A], device=dev)
            cand = torch.tensor([a[2] for a in A], device=dev)
            l_nce, acc = nce(model.zhat(h[bb, pp]), Zg[cand])
        else:
            l_nce, acc = torch.zeros((), device=dev), torch.zeros((), device=dev)
    total = ce + JEPA_WEIGHT * jepa + NCE_WEIGHT * l_nce
    return total, {"ce_w": ce.item(), "ce_me": float(l[me].mean()) if me.any() else 0.0, "ce_other": float(l[~me].mean()),
                   "jepa": jepa.item(), "nce": l_nce.item(), "acc16": acc.item(), "anchors": len(mb["anchors"]),
                   "me_utts": len(mb["slots"]), "decided": len(over), "loss": total.item()}


def inside_masks(mb: dict, dev):
    """Target positions predicted from inputs inside a "me" utterance (input frame t and target frame t + 1 both in
    the same utterance), split by masked / unmasked input; and the same restricted to slots with a swap partner."""
    up = torch.from_numpy(mb["upos"]).to(dev)
    nxt = torch.cat([up[:, 1:], torch.full_like(up[:, :1], -1)], 1)
    ins = (up >= 0) & (nxt == up)
    sl = mb["slots"]
    msk = torch.tensor([s[4] for s in sl] or [False], device=dev)
    sw = torch.tensor([u2 >= 0 for u2 in mb["swap_u"]] or [False], device=dev)
    m = msk[up.clamp(min=0)]
    s = sw[up.clamp(min=0)]
    return ins & ~m, ins & m, ins & ~m & s, ins & m & s


@torch.no_grad()
def validate(model, vbatches: list[dict], Zg, c_ep, bank: Bank, dev) -> dict:
    """CE on "me" utterance frames (unmasked: ceu_*, masked input: cem_*) with true / swapped / null / decided thoughts;
    weighted frame CE; recognition among true + 15 same-speaker "me" utterances (acc16) and + 16 others (acc32)."""
    model.eval()
    keys = ("true", "swap", "null", "decided")
    tot = {f"{p}_{k}": 0.0 for p in ("ceu", "cem") for k in keys}
    tot.update({"ceu_true_on_swap": 0.0, "cem_true_on_swap": 0.0, "ce_w_true": 0.0, "ce_w_decided": 0.0})
    n = {"u": 0, "m": 0, "us": 0, "ms": 0, "w": 0.0}
    hit = hit32 = n_a = 0
    for vb in vbatches:
        x, y, roles, wts = tensors(vb, dev)
        un, ma, uns, mas = inside_masks(vb, dev)
        res = {}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            zt, ht = thought_table(vb, Zg, dev)
            h = model.states(x, roles, zt, ht)
            res["true"] = h
            if vb["anchors"]:
                A = vb["anchors"]
                bb = torch.tensor([a[0] for a in A], device=dev)
                pp = torch.tensor([a[1] for a in A], device=dev)
                cand = torch.tensor([a[2] for a in A], device=dev)
                s = torch.einsum("ad,acd->ac", model.zhat(h[bb, pp]), Zg[cand].float())
                hit += int((s[:, 0] > s[:, 1:16].max(1).values).sum())  # strict: a tie is a miss
                hit32 += int((s[:, 0] > s[:, 1:].max(1).values).sum())
                n_a += len(A)
            res["null"] = model.states(x, roles, None, None)
            sw = {i: Zg[u2] for i, u2 in enumerate(vb["swap_u"]) if u2 >= 0}
            zs, hs = thought_table(vb, Zg, dev, mode_over=[0 if vb["swap_u"][i] >= 0 else 2 for i in range(len(vb["slots"]))], z_over=sw)
            res["swap"] = model.states(x, roles, zs, hs)
            dec = [i for i, s_ in enumerate(vb["slots"]) if s_[3] >= 0]
            over = {}
            if dec:
                zd, _ = decided_for(vb, dec, h, model.zhat, bank, c_ep, dev)
                over = {i: zd[j] for j, i in enumerate(dec)}
            zd_, hd_ = thought_table(vb, Zg, dev, mode_over=[0 if i in over else 2 for i in range(len(vb["slots"]))], z_over=over)
            res["decided"] = model.states(x, roles, zd_, hd_)
        for k, hh in res.items():
            l = F.cross_entropy(model.logits(hh).flatten(0, 1), y.flatten(), reduction="none").view_as(y)
            if k == "swap":
                tot["ceu_swap"] += float((l * uns).sum())
                tot["cem_swap"] += float((l * mas).sum())
            else:
                tot[f"ceu_{k}"] += float((l * un).sum())
                tot[f"cem_{k}"] += float((l * ma).sum())
            if k == "true":
                tot["ceu_true_on_swap"] += float((l * uns).sum())
                tot["cem_true_on_swap"] += float((l * mas).sum())
                tot["ce_w_true"] += float((l * wts).sum())
            if k == "decided":
                tot["ce_w_decided"] += float((l * wts).sum())
        n["u"] += int(un.sum()); n["m"] += int(ma.sum()); n["us"] += int(uns.sum()); n["ms"] += int(mas.sum())
        n["w"] += float(wts.sum())
    model.train()
    out = {}
    for k, v in tot.items():
        if k.startswith("ce_w"):
            d = n["w"]
        elif "swap" in k:
            d = n["us"] if k.startswith("ceu") else n["ms"]
        else:
            d = n["u"] if k.startswith("ceu") else n["m"]
        out[k] = v / max(d, 1)
    out["ceu_swap_minus_true_rel"] = (out["ceu_swap"] - out["ceu_true_on_swap"]) / max(out["ceu_true_on_swap"], 1e-9)
    out["cem_swap_minus_true_rel"] = (out["cem_swap"] - out["cem_true_on_swap"]) / max(out["cem_true_on_swap"], 1e-9)
    out.update({"acc16": hit / max(n_a, 1), "acc32": hit32 / max(n_a, 1), "anchors": n_a,
                "frames_unmasked": n["u"], "frames_masked": n["m"]})
    out["acc16_lo95"] = wilson_lo(hit, n_a)
    return out


# ---------------------------------------------------------------- training
def make_models(cfg: LMConfig, dev):
    import copy

    model = RoleThoughtLM(cfg).to(dev)
    tgt = copy.deepcopy(model)
    for p in tgt.parameters():
        p.requires_grad_(False)
    return model, tgt


def val_batches(va: Corpus, T: int, n_windows: int, micro: int = 16) -> list[dict]:
    b = Builder(va, T, np.random.default_rng(1234))
    starts = np.random.default_rng(1234).integers(0, len(va.stream) - T - 1, size=n_windows)
    return [b.build(starts[i : i + micro], np.arange(i, i + len(starts[i : i + micro])), pred_on=False, val=True)
            for i in range(0, len(starts), micro)]


def train(args, tr: Corpus, va: Corpus, cfg: LMConfig, out: Path, log=print, meta_extra: dict | None = None) -> dict:
    import airon

    dev = torch.device("cuda")
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    model, tgt = make_models(cfg, dev)
    named = list(model.named_parameters())
    decay = [q for n_, q in named if q.dim() >= 2 and "emb" not in n_ and "role" not in n_]
    other = [q for n_, q in named if not (q.dim() >= 2 and "emb" not in n_ and "role" not in n_)]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1}, {"params": other, "weight_decay": 0.0}],
                            lr=args.lr, betas=(0.9, 0.95), fused=True)
    T = cfg.ctx
    Zg = torch.from_numpy(np.concatenate([tr.Z, va.Z])).to(dev)  # tr ids first, va ids offset
    va_off = tr.n
    Ztr, Zva = Zg[:va_off], Zg[va_off:]
    ep_tr = torch.from_numpy(tr.ep).to(dev)
    ep_va = torch.from_numpy(va.ep + 10**6).to(dev)  # held-out recordings never equal a training recording
    bank = Bank(tr, args.bank_n, dev)
    vbatches = val_batches(va, T, args.val_windows)
    out.mkdir(parents=True, exist_ok=True)
    pc = param_counts(model)
    meta = {"run_id": str(uuid.uuid4()), "params": pc, "config": vars(cfg), "args": vars(args),
            "train_files": len(tr.names), "val_files": va.names, "train_tokens": int(len(tr.stream)),
            "val_tokens": int(len(va.stream)), "train_utterances": int(tr.n), "train_valid_thoughts": int(tr.zvalid.sum()),
            "val_utterances": int(va.n), "val_anchors_in_windows": int(sum(len(v["anchors"]) for v in vbatches)),
            "val_me_utterances": int(sum(len(v["slots"]) for v in vbatches)), "git": os.environ.get("DY21_COMMIT", ""),
            "bank": f"fixed: {len(bank.ids)} training utterances (seed 7), other recordings only at use",
            "decided_input_source": "nearest bank thought to the head output of the EMA copy (decay %.3f) at the last frame before the utterance, detached" % args.ema,
            **(meta_extra or {})}
    (out / "metadata.json").write_text(json.dumps(meta, indent=2))
    log(json.dumps(meta))
    run = airon.init(project="dauys", name=args.name, config=meta, experiment=args.experiment,
                     tags=[args.name.split("-")[0], f"seed-{args.seed}"], total_steps=args.steps, run_id=meta["run_id"],
                     mode="disabled" if args.tracking == "off" else None)
    builder = Builder(tr, T, rng)
    q: queue.Queue = queue.Queue(maxsize=6)
    stop = threading.Event()

    def producer() -> None:
        s = 0
        while not stop.is_set():
            starts = rng.integers(0, len(tr.stream) - T - 1, size=args.batch)
            seqs = s * args.batch + np.arange(args.batch)
            on = s >= args.pred_from
            mbs = [builder.build(starts[i : i + args.micro], seqs[i : i + args.micro], pred_on=on)
                   for i in range(0, args.batch, args.micro)]
            q.put((s, mbs))
            s += 1

    th = threading.Thread(target=producer, daemon=True)
    th.start()
    best, t0 = math.inf, time.time()
    chance_step = args.chance_step or int(math.ceil(0.2 * args.steps / args.val_every) * args.val_every)
    history, result = [], {}
    for step in range(args.steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, args.steps, args.lr, args.warmup)
        s, mbs = q.get()
        assert s == step
        opt.zero_grad(set_to_none=True)
        logs: dict[str, float] = {}
        for mb in mbs:
            total, part = step_loss(model, tgt, mb, Ztr, ep_tr, bank, dev, step >= args.pred_from)
            (total / len(mbs)).backward()
            for k, v in part.items():
                logs[k] = logs.get(k, 0.0) + v / len(mbs)
        if (step + 1) % args.log_every == 0:
            gn = lambda ps: float(torch.sqrt(sum((p_.grad.float() ** 2).sum() for p_ in ps if p_.grad is not None)))
            logs["gnorm_proj"] = gn([model.proj.weight, model.null])
            logs["gnorm_role"] = gn([model.role.weight])
            logs["gnorm_zhead"] = gn(list(model.zhead.parameters()))
        logs["grad_norm"] = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        opt.step()
        with torch.no_grad():
            for pt, p_ in zip(tgt.parameters(), model.parameters()):
                pt.lerp_(p_.detach(), 1 - args.ema)
        if (step + 1) % args.log_every == 0:
            logs["lr"] = opt.param_groups[0]["lr"]
            logs["steps_per_s"] = args.log_every / (time.time() - t0)
            logs["rss_gb"] = rss_gb()
            t0 = time.time()
            run.log({f"train/{k}": v for k, v in logs.items()}, step=step + 1)
            log(json.dumps({"step": step + 1, **{k: round(v, 4) for k, v in logs.items()}}))
        late = bool(args.stop_at and time.time() > args.stop_at)
        if (step + 1) % args.val_every == 0 or step + 1 == args.steps or late:
            res = validate(model, vbatches, Zva, ep_va, bank, dev)
            run.log({f"val/{k}": v for k, v in res.items()}, step=step + 1)
            log(json.dumps({"step": step + 1, "val": res}))
            history.append({"step": step + 1, **res})
            ck = {"model": model.state_dict(), "cfg": vars(cfg), "step": step + 1, "val": res, "dict": args.dict,
                  "dict_key": args.dict_key, "bank_ids": bank.ids, "mask_code": MASK}
            if res["ce_w_decided"] < best:
                best = res["ce_w_decided"]
                torch.save(ck, out / "best.pt")
            if not args.no_last:
                torch.save(ck, out / "last.pt")
            (out / "val_history.json").write_text(json.dumps(history, indent=1))
            if step + 1 >= chance_step and not result.get("chance_checked"):
                result["chance_checked"] = True
                if res["acc16_lo95"] <= 1 / 16:
                    log(json.dumps({"early_stop_rule": step + 1, "acc16": res["acc16"], "lo95": res["acc16_lo95"]}))
                    result["early_stop"] = step + 1
                    break
            if late:
                log(json.dumps({"stop_at_reached": step + 1}))
                break
    stop.set()
    try:
        q.get_nowait()
    except queue.Empty:
        pass
    (out / "DONE").write_text(f"{best}\n")
    run.finish()
    result.update({"best_ce_w_decided": best, "history": history, "model": model, "bank": bank})
    log(f"TRAIN DONE {best}")
    return result


def load_all(utt: str, dict_path: str, table: str, key: str, limit: int = 0, seed: int = 1):
    files = sorted(Path(utt).glob("*.npz"))
    tr_files = [f for f in files if not held_out(f.stem)]
    va_files = [f for f in files if held_out(f.stem)]
    assert va_files and not set(f.stem for f in tr_files) & set(f.stem for f in va_files)
    if limit:
        tr_files = tr_files[:limit]
    info = {}
    out = []
    for name, fl in (("train", tr_files), ("val", va_files)):
        c = load_corpus(fl)
        info[name] = attach_dict(c, utterance_keys(fl, files), dict_path, table, key)
        prepare(c, segment_speakers(fl, len(c.stream)), seed)
        out.append(c)
    return out[0], out[1], info


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--utt", required=True)
    p.add_argument("--dict", required=True)
    p.add_argument("--table", required=True)
    p.add_argument("--dict-key", default="thought")
    p.add_argument("--out", required=True)
    p.add_argument("--steps", type=int, default=12257)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--micro", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup", type=int, default=1000)
    p.add_argument("--val-every", type=int, default=500)
    p.add_argument("--val-windows", type=int, default=256)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--ema", type=float, default=0.999)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--pred-from", type=int, default=2000)
    p.add_argument("--bank-n", type=int, default=BANK_N)
    p.add_argument("--chance-step", type=int, default=0)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--cfg", default="1024,24,16,4096,2048")
    p.add_argument("--tracking", default="on", choices=["on", "off"])
    p.add_argument("--experiment", default=EXPERIMENT)
    p.add_argument("--name", default="dy37-t")
    p.add_argument("--stop-at", type=float, default=0)
    p.add_argument("--no-last", action="store_true")
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t = time.time()
    tr, va, info = load_all(args.utt, args.dict, args.table, args.dict_key, args.limit, args.seed)
    print(json.dumps({"loaded_s": round(time.time() - t, 1), "dict": info, "train_tokens": len(tr.stream), "train_utts": tr.n,
                      "val_utts": va.n, "rss_gb": round(rss_gb(), 2)}), flush=True)
    d, layers, heads, ffn, ctx = map(int, args.cfg.split(","))
    cfg = LMConfig(vocab=VOCAB, d=d, layers=layers, heads=heads, ffn=ffn, ctx=ctx)
    train(args, tr, va, cfg, out, log=lambda s: print(s, flush=True), meta_extra={"dict_info": info})


if __name__ == "__main__":
    main()
