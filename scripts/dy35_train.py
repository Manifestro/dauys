"""DY-35 training: the DY-32 b UnitLM recipe (24 x 1024, K2000, 50 Hz, context 2048, random windows over the joined
pieces, CE + JEPA, AdamW, cosine) plus a thought level (src/dauys/dy35_thought.py).

Data: ``--utt DIR`` of ``<shard>.npz`` from scripts/dy35_utt.py. Held-out shards (dy21_train.held_out) are only used
for validation. Stream = codes of every piece + EOS, as dy21_train.load_stream. Utterance = speaker segment of
100-1000 frames; its codes in the stream were encoded alone (dy35_utt.py), the gist encoder reads exactly those.

Per window (random start, as b): utterances whose frames lie in the window carry a thought: true z (p 0.6), predicted
z_hat (p 0.3; from step --pred-from, before that true z), null (p 0.1). z_hat for the input share is the prediction of
the EMA copy of the model + head (the JEPA target pass, in which those utterances carry true z); it is detached.
Anchors: utterances starting >= 750 frames after the window start and >= 750 frames into their piece, with >= 15
same-speaker same-recording utterances >= 60 s away from the anchor and from the utterance and of length within x2;
up to 8 per window. Candidates: true + 15 such decoys (a shuffled per-window, per-speaker pool shared between anchors)
+ 16 utterances of other recordings from the same micro-batch. InfoNCE tau 0.1, weight 1.0.

    python scripts/dy35_train.py --utt /opt/dlami/nvme/utt --out /data/t/run --cfg 1024,24,16,4096,2048 --batch 32 \
        --micro 16 --lr 3e-4 --warmup 1000 --val-every 500 --steps 12257 --seed 1 --name dy35-t --experiment "..."
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

from dauys.dy21_lm import JEPA_WEIGHT, LMConfig
from dauys.dy35_thought import NCE_WEIGHT, GistEncoder, ThoughtLM, lm_jepa, nce, param_counts

K = 2000
EOS = K
UMIN, UMAX = 100, 1000
HEARD = 750
GAP_S = 60.0
N_SAME, N_OTHER = 15, 16
MAX_ANCHORS = 8
P_TRUE, P_PRED = 0.6, 0.3
EXPERIMENT = "DY-35: мысль о всей следующей реплике — перед её началом модель узнаёт суть реплики среди других реплик того же человека, без текста"


def rss_gb() -> float:
    try:
        return int(open(f"/proc/{os.getpid()}/statm").read().split()[1]) * 4096 / 1e9
    except Exception:
        return -1.0


class Corpus:
    """Joined code stream + utterance table.

    stream int16 [N]; utt_of int32 [N] (utterance id or -1); per utterance: st, en (stream), ep, spk (global ids),
    t0, t1 (seconds in the episode), sin (start frame inside its piece), grp (index of its (ep, spk) group);
    groups: list of utterance id arrays; ok_anchor bool (heard >= 750 in piece and >= 15 eligible decoys)."""

    def __init__(self, stream, st, en, ep, spk, t0, t1, sin, names=None, emb=None, prev_spk=None, prev_gap=None,
                 file_idx=None) -> None:
        self.stream = stream
        self.emb, self.prev_spk, self.prev_gap, self.file_idx = emb, prev_spk, prev_gap, file_idx
        self.st, self.en = st.astype(np.int64), en.astype(np.int64)
        self.ep, self.spk = ep.astype(np.int64), spk.astype(np.int64)
        self.t0, self.t1, self.sin = t0.astype(np.float64), t1.astype(np.float64), sin.astype(np.int64)
        self.n = len(self.st)
        self.names = names or []
        self.utt_of = np.full(len(stream), -1, np.int32)
        for i in range(self.n):
            self.utt_of[self.st[i] : self.en[i]] = i
        self.len = self.en - self.st
        _, self.grp = np.unique(self.spk, return_inverse=True)
        order = np.argsort(self.grp, kind="stable")
        bounds = np.flatnonzero(np.diff(np.concatenate([[-1], self.grp[order], [10**12]])))
        self.groups = [order[a:b] for a, b in zip(bounds[:-1], bounds[1:])]
        self.n_elig = np.zeros(self.n, np.int64)
        for g in self.groups:
            m = self.eligible_mask(g, g)
            self.n_elig[g] = m.sum(1)
        self.ok_anchor = (self.sin >= HEARD) & (self.n_elig >= N_SAME)

    def eligible_mask(self, us: np.ndarray, ds: np.ndarray) -> np.ndarray:
        """[len(us), len(ds)]: d is a valid same-speaker decoy for true utterance u (caller passes one group)."""
        far = (self.t1[ds][None] <= self.t0[us][:, None] - GAP_S) | (self.t0[ds][None] >= self.t1[us][:, None] + GAP_S)
        r = self.len[ds][None] / self.len[us][:, None]
        return far & (r >= 0.5) & (r <= 2.0) & (ds[None] != us[:, None])

    def codes(self, u: int) -> np.ndarray:
        return self.stream[self.st[u] : self.en[u]]


def load_corpus(files: list[Path]) -> Corpus:
    parts, st, en, ep, spk, t0, t1, sin = [], [], [], [], [], [], [], []
    emb, prev_spk, prev_gap, fidx = [], [], [], []
    pos, ep_base = 0, 0
    names = []
    for f in files:
        z = np.load(f)
        tok, index = z["tok"], z["index"]
        neps = len(json.loads(str(z["eps"])))
        off = np.full(len(index), -1, np.int64)
        for k, (a, n) in enumerate(index):
            if n < 16:
                continue
            off[k] = pos
            parts.append(tok[a : a + n].astype(np.int16))
            parts.append(np.array([EOS], np.int16))
            pos += n + 1
        L = z["seg_e"] - z["seg_s"]
        m = (L >= UMIN) & (L <= UMAX) & (off[z["seg_piece"]] >= 0)
        # previous segment (any length) in the same piece: its speaker and the gap to it (frames)
        sp_, ss_, se_ = z["seg_piece"], z["seg_s"].astype(np.int64), z["seg_e"].astype(np.int64)
        o = np.lexsort((ss_, sp_))
        assert np.all(o == np.arange(len(o))), "segments are not in (piece, start) order"
        first = np.concatenate([[True], sp_[1:] != sp_[:-1]])
        pspk = np.where(first, -1, np.concatenate([[-1], z["seg_spk"][:-1].astype(np.int64)]))
        pgap = np.where(first, -1, np.concatenate([[0], ss_[1:] - se_[:-1]]))
        prev_spk.append(pspk[m])
        prev_gap.append(pgap[m])
        emb.append(z["seg_emb"][m])
        fidx.append(np.full(int(m.sum()), len(names)))
        p, s, e = z["seg_piece"][m], z["seg_s"][m].astype(np.int64), z["seg_e"][m].astype(np.int64)
        st.append(off[p] + s)
        en.append(off[p] + e)
        g_ep = ep_base + z["piece_ep"][p].astype(np.int64)
        ep.append(g_ep)
        spk.append(g_ep * 64 + z["seg_spk"][m].astype(np.int64))
        t0.append(z["piece_t0"][p] + s / 50)
        t1.append(z["piece_t0"][p] + e / 50)
        sin.append(s)
        ep_base += neps
        names.append(f.stem)
    cat = lambda xs: np.concatenate(xs) if xs else np.zeros(0)
    return Corpus(np.concatenate(parts), cat(st), cat(en), cat(ep), cat(spk), cat(t0), cat(t1), cat(sin), names,
                  emb=np.concatenate(emb) if emb else None, prev_spk=cat(prev_spk), prev_gap=cat(prev_gap), file_idx=cat(fidx))


class Builder:
    """Micro-batches on the CPU (numpy). Window starts follow dy21_train (uniform over the stream)."""

    def __init__(self, c: Corpus, T: int, rng: np.random.Generator) -> None:
        self.c, self.T, self.rng = c, T, rng
        self.anchor_ids = np.flatnonzero(c.ok_anchor)

    def build(self, starts: np.ndarray, pred_on: bool, modes: bool = True, max_anchors: int = MAX_ANCHORS,
              n_other: int = N_OTHER) -> dict:
        c, T, rng = self.c, self.T, self.rng
        B = len(starts)
        ids = np.stack([c.stream[s : s + T + 1] for s in starts]).astype(np.int64)
        enc: dict[int, int] = {}  # utterance id -> row in the encode list

        def row(u: int) -> int:
            if u not in enc:
                enc[u] = len(enc)
            return enc[u]

        upos = np.full((B, T), -1, np.int64)  # in-window utterance slot per input frame
        slots = []  # (b, u, enc row, mode 0 true 1 pred 2 null, anchor position or -1)
        anchors = []  # (b, position, true row, [15 same rows], u)
        for b, w in enumerate(starts):
            uo = c.utt_of[w : w + T]
            us = np.unique(uo[uo >= 0])
            for u in us:
                q0, q1 = max(c.st[u] - w, 0), min(c.en[u] - w, T)
                ap = int(c.st[u] - 1 - w)
                r = rng.random()
                mode = 0 if r < P_TRUE else (1 if r < P_TRUE + P_PRED else 2)
                if not modes:
                    mode = 0
                if mode == 1 and (not pred_on or ap < 0):
                    mode = 0
                upos[b, q0:q1] = len(slots)
                slots.append((b, int(u), row(int(u)), mode, ap if ap >= 0 else -1))
            # anchors: utterance start in [w + HEARD, w + T]
            lo, hi = np.searchsorted(c.st, w + HEARD), np.searchsorted(c.st, w + T, side="right")
            cand = np.arange(lo, hi)
            cand = cand[c.ok_anchor[cand]]
            if len(cand) > max_anchors:
                cand = np.sort(rng.choice(cand, max_anchors, replace=False))
            pools: dict[int, np.ndarray] = {}
            for u in cand:
                g = c.groups[c.grp[u]]
                if c.grp[u] not in pools:
                    pools[c.grp[u]] = rng.permutation(g)
                pool = pools[c.grp[u]]
                m = c.eligible_mask(np.array([u]), pool)[0]
                ds = pool[m][:N_SAME]
                assert len(ds) == N_SAME
                anchors.append((b, int(c.st[u] - 1 - w), row(int(u)), [row(int(d)) for d in ds], int(u)))
        # other recordings: from everything encoded in this micro-batch (+ random extra utterances)
        extra = rng.choice(c.n, 64, replace=False)
        for u in extra:
            row(int(u))
        rows_u = np.empty(len(enc), np.int64)
        for u, r in enc.items():
            rows_u[r] = u
        rows_ep = c.ep[rows_u]
        A = []
        for b, p, tr, same, u in anchors:
            oth = np.flatnonzero(rows_ep != c.ep[u])
            oth = rng.choice(oth, n_other, replace=len(oth) < n_other)
            A.append((b, p, [tr] + same + oth.tolist()))
        lens = c.len[rows_u]
        codes = np.concatenate([c.codes(int(u)) for u in rows_u]).astype(np.int64) if len(rows_u) else np.zeros(0, np.int64)
        return {"ids": ids, "upos": upos, "slots": slots, "anchors": A, "codes": codes, "lens": lens,
                "rows_u": rows_u, "starts": starts}


def encode_rows(enc: GistEncoder, batch: dict, dev) -> torch.Tensor:
    codes = torch.from_numpy(batch["codes"]).to(dev, non_blocking=True)
    utts = list(codes.split(batch["lens"].tolist()))
    return enc.encode_many(utts)


def thought_tables(batch: dict, Z: torch.Tensor, zh_slot: torch.Tensor | None, dev, override: dict | None = None):
    """Per-frame thoughts: table [n_slots, 64] (true z, or z_hat for mode 1), has [B, T] (False = null)."""
    slots = batch["slots"]
    if not slots:
        B, T = batch["upos"].shape
        return torch.zeros(B, T, Z.shape[1], device=dev), torch.zeros(B, T, dtype=torch.bool, device=dev)
    rows = torch.tensor([s[2] for s in slots], device=dev)
    mode = torch.tensor([s[3] for s in slots], device=dev)
    tab = Z[rows]
    if zh_slot is not None:
        tab = torch.where((mode == 1)[:, None], zh_slot.to(tab.dtype), tab)
    if override is not None:
        for i, v in override.items():
            tab = tab.index_copy(0, torch.tensor([i], device=dev), v[None].to(tab.dtype))
    upos = torch.from_numpy(batch["upos"]).to(dev)
    has = (upos >= 0) & (mode[upos.clamp(min=0)] != 2)
    return tab[upos.clamp(min=0)], has


def slot_anchor_states(batch: dict, h: torch.Tensor) -> tuple[list[int], torch.Tensor]:
    """States at the anchor position of every slot that has one (position >= 0)."""
    idx = [i for i, s in enumerate(batch["slots"]) if s[4] >= 0]
    if not idx:
        return idx, h.new_zeros(0, h.shape[-1])
    b = torch.tensor([batch["slots"][i][0] for i in idx], device=h.device)
    p = torch.tensor([batch["slots"][i][4] for i in idx], device=h.device)
    return idx, h[b, p]


def anchor_states(batch: dict, h: torch.Tensor):
    A = batch["anchors"]
    b = torch.tensor([a[0] for a in A], device=h.device)
    p = torch.tensor([a[1] for a in A], device=h.device)
    cand = torch.tensor([a[2] for a in A], device=h.device)
    return h[b, p], cand


def step_loss(model, tgt, enc, batch, dev, pred_on: bool):
    ids = torch.from_numpy(batch["ids"]).to(dev, non_blocking=True)
    x, y = ids[:, :-1], ids[:, 1:]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        Z = encode_rows(enc, batch, dev)
        with torch.no_grad():
            zin_t, has_t = thought_tables(batch, Z.detach(), None, dev)
            h_tgt = tgt.states(x, zin_t, has_t)
            zh_slot = None
            if pred_on and any(s[3] == 1 for s in batch["slots"]):
                idx, hs = slot_anchor_states(batch, h_tgt)
                zh_slot = torch.zeros(len(batch["slots"]), Z.shape[1], device=dev)
                if idx:
                    zh_slot[torch.tensor(idx, device=dev)] = tgt.zhat(hs)
        zin, has = thought_tables(batch, Z, zh_slot, dev)
        h = model.states(x, zin, has)
        ce, jepa = lm_jepa(model, h, h_tgt, y)
        if batch["anchors"]:
            ha, cand = anchor_states(batch, h)
            l_nce, acc = nce(model.zhat(ha), Z[cand])
        else:
            l_nce, acc = torch.zeros((), device=dev), torch.zeros((), device=dev)
    total = ce + JEPA_WEIGHT * jepa + NCE_WEIGHT * l_nce
    return total, {"ce": ce.item(), "jepa": jepa.item(), "nce": l_nce.item(), "acc16": acc.item(),
                   "anchors": len(batch["anchors"]), "enc_utts": len(batch["rows_u"]), "loss": total.item()}


@torch.no_grad()
def validate(model, enc, vbatches: list[dict], dev) -> dict:
    """CE (all frames, and utterance frames only) with true z / null / predicted z_hat (from the true-z pass), and
    recognition accuracy among true + 15 same-speaker utterances."""
    model.eval()
    enc.eval()
    tot = {k: 0.0 for k in ("ce_true", "ce_null", "ce_pred", "ceu_true", "ceu_null", "ceu_pred")}
    n_all = n_u = 0
    hit = n_a = 0
    hit32 = 0
    for vb in vbatches:
        ids = torch.from_numpy(vb["ids"]).to(dev)
        x, y = ids[:, :-1], ids[:, 1:]
        inu = torch.from_numpy(vb["upos"]).to(dev)
        inu = torch.cat([inu[:, 1:] >= 0, torch.zeros_like(inu[:, :1], dtype=torch.bool)], 1)  # target frame inside an utterance
        with torch.autocast("cuda", dtype=torch.bfloat16):
            Z = encode_rows(enc, vb, dev)
            zt, ht = thought_tables(vb, Z, None, dev)
            h = model.states(x, zt, ht)
            res = {"true": h}
            idx, hs = slot_anchor_states(vb, h)
            zh_slot = torch.zeros(len(vb["slots"]), Z.shape[1], device=dev)
            if idx:
                zh_slot[torch.tensor(idx, device=dev)] = model.zhat(hs)
            if vb["anchors"]:
                ha, cand = anchor_states(vb, h)
                zh = model.zhat(ha)
                s = torch.einsum("ad,acd->ac", zh.float(), Z[cand].float())
                hit += int((s[:, :16].argmax(1) == 0).sum())
                hit32 += int((s.argmax(1) == 0).sum())
                n_a += len(zh)
            nb = {**vb, "slots": [(b, u, r, 2 if a < 0 else 1, a) for b, u, r, m, a in vb["slots"]]}
            zp, hp = thought_tables(nb, Z, zh_slot, dev)
            res["pred"] = model.states(x, zp, hp)
            res["null"] = model.states(x, None, None)
        for k, hh in res.items():
            l = F.cross_entropy(model.logits(hh).flatten(0, 1), y.flatten(), reduction="none").view_as(y)
            tot[f"ce_{k}"] += float(l.sum())
            tot[f"ceu_{k}"] += float((l * inu).sum())
        n_all += y.numel()
        n_u += int(inu.sum())
    model.train()
    enc.train()
    out = {k: v / (n_all if k.startswith("ce_") else max(n_u, 1)) for k, v in tot.items()}
    out.update({"acc16": hit / max(n_a, 1), "acc32": hit32 / max(n_a, 1), "anchors": n_a})
    return out


def wilson_lo(k: int, n: int) -> float:
    if n == 0:
        return 0.0
    p, z = k / n, 1.96
    d = 1 + z * z / n
    return (p + z * z / (2 * n) - z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / d


def make_models(cfg: LMConfig, dev):
    import copy

    model = ThoughtLM(cfg).to(dev)
    enc = GistEncoder(vocab=K).to(dev)
    tgt = copy.deepcopy(model)
    for p in tgt.parameters():
        p.requires_grad_(False)
    return model, enc, tgt


def train(args, tr: Corpus, va: Corpus, cfg: LMConfig, out: Path, log=print, meta_extra: dict | None = None) -> dict:
    import airon

    dev = torch.device("cuda")
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    model, enc, tgt = make_models(cfg, dev)
    named = list(model.named_parameters()) + [("enc." + n, p) for n, p in enc.named_parameters()]
    decay = [q for n, q in named if q.dim() >= 2 and "emb" not in n]
    other = [q for n, q in named if not (q.dim() >= 2 and "emb" not in n)]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1}, {"params": other, "weight_decay": 0.0}],
                            lr=args.lr, betas=(0.9, 0.95), fused=True)
    T = cfg.ctx
    vb_builder = Builder(va, T, np.random.default_rng(1234))
    vstarts = np.random.default_rng(1234).integers(0, len(va.stream) - T - 1, size=args.val_windows)
    vbatches = [vb_builder.build(vstarts[i : i + 16], pred_on=False, modes=False) for i in range(0, len(vstarts), 16)]
    pc = param_counts(model, enc)
    meta = {"run_id": str(uuid.uuid4()), "params": pc, "config": vars(cfg), "args": vars(args),
            "train_files": len(tr.names), "val_files": va.names, "train_tokens": int(len(tr.stream)),
            "val_tokens": int(len(va.stream)), "train_utterances": int(tr.n), "train_anchor_utterances": int(tr.ok_anchor.sum()),
            "val_utterances": int(va.n), "val_anchor_utterances": int(va.ok_anchor.sum()),
            "val_anchors_in_windows": int(sum(len(v["anchors"]) for v in vbatches)), "git": os.environ.get("DY21_COMMIT", ""),
            "pred_input_source": "z_hat of the EMA copy (decay %.3f) of model + head from the JEPA target pass, detached" % args.ema,
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
            on = s >= args.pred_from
            mbs = [builder.build(starts[i : i + args.micro], pred_on=on) for i in range(0, args.batch, args.micro)]
            q.put((s, mbs))
            s += 1

    th = threading.Thread(target=producer, daemon=True)
    th.start()
    best, t0 = math.inf, time.time()
    chance_step = args.chance_step or int(math.ceil(0.2 * args.steps / args.val_every) * args.val_every)
    history = []
    result = {}
    for step in range(args.steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, args.steps, args.lr, args.warmup)
        s, mbs = q.get()
        assert s == step
        opt.zero_grad(set_to_none=True)
        logs: dict[str, float] = {}
        for mb in mbs:
            total, part = step_loss(model, tgt, enc, mb, dev, step >= args.pred_from)
            (total / len(mbs)).backward()
            for k, v in part.items():
                logs[k] = logs.get(k, 0.0) + v / len(mbs)
        logs["grad_norm"] = float(torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(enc.parameters()), 1.0))
        opt.step()
        with torch.no_grad():
            for pt, p in zip(tgt.parameters(), model.parameters()):
                pt.lerp_(p.detach(), 1 - args.ema)
        if (step + 1) % args.log_every == 0:
            logs["lr"] = opt.param_groups[0]["lr"]
            logs["steps_per_s"] = args.log_every / (time.time() - t0)
            logs["rss_gb"] = rss_gb()
            t0 = time.time()
            run.log({f"train/{k}": v for k, v in logs.items()}, step=step + 1)
            log(json.dumps({"step": step + 1, **{k: round(v, 4) for k, v in logs.items()}}))
        late = bool(args.stop_at and time.time() > args.stop_at)
        if (step + 1) % args.val_every == 0 or step + 1 == args.steps or late:
            res = validate(model, enc, vbatches, dev)
            res["acc16_lo95"] = wilson_lo(round(res["acc16"] * res["anchors"]), res["anchors"])
            run.log({f"val/{k}": v for k, v in res.items()}, step=step + 1)
            log(json.dumps({"step": step + 1, "val": res}))
            history.append({"step": step + 1, **res})
            ck = {"model": model.state_dict(), "enc": enc.state_dict(), "cfg": vars(cfg), "step": step + 1, "val": res}
            if res["ce_pred"] < best:
                best = res["ce_pred"]
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
    result.update({"best_ce_pred": best, "history": history, "model": model, "enc": enc})
    log(f"TRAIN DONE {best}")
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--utt", required=True)
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
    p.add_argument("--chance-step", type=int, default=0, help="validation step of the pre-registered chance check (0 = first >= 20 %)")
    p.add_argument("--limit", type=int, default=0, help="dry run: only this many training shard files")
    p.add_argument("--cfg", default="1024,24,16,4096,2048")
    p.add_argument("--tracking", default="on", choices=["on", "off"])
    p.add_argument("--experiment", default=EXPERIMENT)
    p.add_argument("--name", default="dy35-t")
    p.add_argument("--stop-at", type=float, default=0)
    p.add_argument("--no-last", action="store_true")
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(Path(args.utt).glob("*.npz"))
    tr_files = [f for f in files if not held_out(f.stem)]
    va_files = [f for f in files if held_out(f.stem)]
    assert va_files and not set(f.stem for f in tr_files) & set(f.stem for f in va_files)
    if args.limit:
        tr_files = tr_files[: args.limit]
    t = time.time()
    tr, va = load_corpus(tr_files), load_corpus(va_files)
    print(json.dumps({"loaded_s": round(time.time() - t, 1), "train_tokens": len(tr.stream), "train_utts": tr.n,
                      "train_anchor_utts": int(tr.ok_anchor.sum()), "val_utts": va.n, "rss_gb": round(rss_gb(), 2)}), flush=True)
    d, layers, heads, ffn, ctx = map(int, args.cfg.split(","))
    cfg = LMConfig(vocab=K + 1, d=d, layers=layers, heads=heads, ffn=ffn, ctx=ctx)
    train(args, tr, va, cfg, out, log=lambda s: print(s, flush=True))


if __name__ == "__main__":
    main()
