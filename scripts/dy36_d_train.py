"""DY-36 candidate D: training, textless checks and the dictionary (no text anywhere).

    python scripts/dy36_d_train.py synth  --out /data/dy36/d/synth            # check (i), known answer
    python scripts/dy36_d_train.py train  --dict /data/dy36/dict --pairs /data/dy36/d/pairs --out /data/dy36/d/run
    python scripts/dy36_d_train.py checks --dict /data/dy36/dict --pairs /data/dy36/d/pairs --ckpt .../last.pt --out ...
    python scripts/dy36_d_train.py dict   --dict /data/dy36/dict --ckpt .../best.pt --name D

Check (i) known answer: synthetic "words" (fixed sequences of abstract phonemes), voices = phoneme -> code maps (30% of
phonemes share one code over all voices, the rest voice-specific), random durations 2-6 frames and 10% code noise;
the same recipe trained on pairs (same word bag in two voices); test: 200 new bags in voices 0 and 1, query z(bag, v0)
among z of all bags in v1 and all other bags in v0: top-1 must be the same bag in v1 (>= 0.9), not a same-voice one.
Check (ii): 2000 held-out utterances, z(original codes) retrieves z(converted, tempo-changed copy) among the 2000
(top-1, both directions averaged); baseline = cosine of L2-normalised raw code histograms.
Check (iii): voice leakage: per held-out recording, logistic regression (standardised, 5-fold) predicting the speaker
cluster from raw z (speakers with >= 10 utterances); utterance-weighted accuracy next to the majority rate.
"""

import argparse
import json
import math
import os
import sys
import threading
import queue
import time
import uuid
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from dy36_dict import centre, load_table

from dauys.dy36_d import MASK, VDIM, ZDIM, K, ModelD, block_visible, info_nce, mask_input

EXPERIMENT = "DY-36: модель замысла — участник разговора сам выбирает, что сказать, из настоящих замыслов; без звука на входе модели и без текста"


def log(*a) -> None:
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def nparams(m) -> int:
    return sum(p.numel() for p in m.parameters())


def pad(seqs: list, value: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    L = max(len(s) for s in seqs)
    x = np.full((len(seqs), L), value, np.int64)
    for i, s in enumerate(seqs):
        x[i, : len(s)] = s
    return torch.from_numpy(x), torch.tensor([len(s) for s in seqs])


def step_loss(model, a, la, b, lb, v, args, gen=None):
    """CE on masked positions of B (z from masked A) + w * InfoNCE(z(A), z(B)); also returns parts."""
    za = model.enc(mask_input(a, la, args.mask_a, gen), la)
    vis = block_visible(b, lb, args.vis, args.block, gen)
    keep = torch.arange(b.shape[1], device=b.device)[None] < lb[:, None]
    bin_ = torch.where(vis, b, torch.full_like(b, MASK))
    lg = model.dec(bin_, lb, za, v)
    tgt = keep & ~vis
    ce = F.cross_entropy(lg[tgt].float(), b[tgt])
    if model.bag is not None and args.bag > 0:
        h = torch.zeros(len(b), K, device=b.device).scatter_add_(1, b * keep, keep.float())
        h = h * IDF.to(b.device)[None]
        h = h / h.sum(1, keepdim=True).clamp(min=1e-6)
        bag = -(h * F.log_softmax(model.bag(za).float(), -1)).sum(1).mean()
        ce = ce + args.bag * bag
    nce = torch.zeros((), device=a.device)
    if args.nce > 0:
        zb = model.enc(mask_input(b, lb, args.mask_a, gen), lb)
        nce = info_nce(za, zb)
    return ce + args.nce * nce, ce, nce


IDF = torch.ones(K)


def set_idf(T: dict, n: int = 50000) -> None:
    """Smooth idf of codes over a seeded sample of training utterances (for the bag target of variant E)."""
    global IDF
    rng = np.random.default_rng(0)
    rows = rng.choice(np.flatnonzero(~T["held"]), n, replace=False)
    df = np.zeros(K)
    for u in rows:
        df[np.unique(T["codes"][T["code_off"][u] : T["code_off"][u + 1]])] += 1
    IDF = torch.from_numpy((np.log((1 + n) / (1 + df)) + 1).astype(np.float32))


def distort(x: np.ndarray, rng, seg=(15, 30), tempo=(0.8, 1.25), p_drop=0.05, p_dup=0.05) -> np.ndarray:
    """Variant E, voice-B side: a new tempo factor every 0.3-0.6 s (nearest-frame resampling), then random frame drops
    and duplications."""
    out, i = [], 0
    while i < len(x):
        L = int(rng.integers(seg[0], seg[1] + 1))
        part = x[i : i + L]
        n2 = max(1, int(round(len(part) * rng.uniform(*tempo))))
        out.append(part[np.minimum((np.arange(n2) * len(part) / n2).astype(np.int64), len(part) - 1)])
        i += L
    y = np.concatenate(out)
    r = rng.random(len(y))
    y = np.repeat(y, np.where(r < p_drop, 0, np.where(r > 1 - p_dup, 2, 1)))
    return y if len(y) >= 10 else x


# ---------------------------------------------------------------- real pairs
class Pairs:
    def __init__(self, T: dict, d: Path, distort_b: bool = False) -> None:
        self.T, self.d = T, Path(d)
        self.distort_b, self.rng = distort_b, np.random.default_rng(12345)
        self.seen: set = set()
        self.codes, self.offs, self.vfeat = [], [], []
        self.p_sh, self.p_j, self.p_uid, self.p_v = [], [], [], []
        self.nvoice = 0

    def refresh(self) -> int:
        new = 0
        for f in sorted(self.d.glob("*.npz")):
            if f.name in self.seen or ".tmp" in f.name:
                continue
            z = np.load(f)
            sh = len(self.codes)
            self.codes.append(z["codes"])
            self.offs.append(z["off"])
            vmap = {int(s): self.nvoice + i for i, s in enumerate(z["spk"])}
            self.vfeat.append(z["vfeat"].astype(np.float32))
            self.nvoice += len(z["spk"])
            uid, tgt = z["uid"], z["tgt"]
            nv = tgt.shape[1]
            j = np.arange(len(uid) * nv)
            self.p_sh.append(np.full(len(j), sh))
            self.p_j.append(j)
            self.p_uid.append(np.repeat(uid, nv))
            self.p_v.append(np.array([vmap[int(s)] for s in tgt.ravel()]))
            self.seen.add(f.name)
            new += 1
        if new:
            self.sh, self.j = np.concatenate(self.p_sh), np.concatenate(self.p_j)
            self.uid, self.v = np.concatenate(self.p_uid), np.concatenate(self.p_v)
            self.V = np.concatenate(self.vfeat)
            self.held = self.T["held"][self.uid]
        return new

    def a(self, i: int) -> np.ndarray:
        u = self.uid[i]
        return self.T["codes"][self.T["code_off"][u] : self.T["code_off"][u + 1]].astype(np.int64)

    def b(self, i: int) -> np.ndarray:
        o = self.offs[self.sh[i]]
        return self.codes[self.sh[i]][o[self.j[i]] : o[self.j[i] + 1]].astype(np.int64)

    def batch(self, idx, train: bool = True) -> tuple:
        a, la = pad([self.a(i) for i in idx])
        bs = [self.b(i) for i in idx]
        if train and self.distort_b:
            bs = [distort(x, self.rng)[:1400] for x in bs]
        b, lb = pad(bs)
        return a, la, b, lb, torch.from_numpy(self.V[self.v[idx]])


def batches(P: Pairs, rng, budget: int, maxb: int):
    tr = np.flatnonzero(~P.held)
    idx = rng.choice(tr, 32 * maxb)
    L = P.T["len"][P.uid[idx]]
    idx = idx[np.argsort(L)]
    L = np.sort(L)
    out, s = [], 0
    while s < len(idx):
        e = s
        while e < len(idx) and (e - s + 1) * max(L[e] * 1.2, 1) <= budget and e - s < maxb:
            e += 1
        e = max(e, s + 1)
        out.append(idx[s:e])
        s = e
    rng.shuffle(out)
    return [P.batch(x) for x in out if len(x) >= 8]


def same_spk_top1(sim: np.ndarray, spk: np.ndarray) -> tuple[float, float]:
    """Query = converted copy (column), candidates = originals (rows) of the same source speaker only; top-1 accuracy
    and the mean number of candidates."""
    sim = np.asarray(sim)
    ok, nc = [], []
    for j in range(sim.shape[1]):
        c = np.flatnonzero(spk == spk[j])
        ok.append(c[sim[c, j].argmax()] == j)
        nc.append(len(c))
    return float(np.mean(ok)), float(np.mean(nc))


def hist_mat(A: list, B: list) -> np.ndarray:
    def H(xs):
        h = np.zeros((len(xs), K), np.float32)
        for i, x in enumerate(xs):
            h[i] = np.bincount(x, minlength=K)
        return h / np.linalg.norm(h, axis=1, keepdims=True)

    return H(A) @ H(B).T


def hist_retrieval(A: list, B: list) -> float:
    def H(xs):
        h = np.zeros((len(xs), K), np.float32)
        for i, x in enumerate(xs):
            h[i] = np.bincount(x, minlength=K)
        return h / np.linalg.norm(h, axis=1, keepdims=True)

    ha, hb = H(A), H(B)
    s = ha @ hb.T
    y = np.arange(len(A))
    return float(((s.argmax(1) == y).mean() + (s.argmax(0) == y).mean()) / 2)


@torch.no_grad()
def evaluate(model, P: Pairs, vidx: np.ndarray, args) -> dict:
    model.eval()
    gen = torch.Generator(device="cuda").manual_seed(1234)
    tot, tot_shuf, n = 0.0, 0.0, 0
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for s in range(0, len(vidx), 100):
            a, la, b, lb, v = [x.cuda() for x in P.batch(vidx[s : s + 100], train=False)]
            za = model.enc(a, la)
            g2 = torch.Generator(device="cuda").manual_seed(4321 + s)
            vis = block_visible(b, lb, args.vis, args.block, g2)
            keep = torch.arange(b.shape[1], device=b.device)[None] < lb[:, None]
            bin_ = torch.where(vis, b, torch.full_like(b, MASK))
            tgt = keep & ~vis
            for zz, acc in ((za, "t"), (za.roll(1, 0), "s")):
                lg = model.dec(bin_, lb, zz, v)
                ce = F.cross_entropy(lg[tgt].float(), b[tgt], reduction="sum").item()
                if acc == "t":
                    tot += ce
                else:
                    tot_shuf += ce
            n += int(tgt.sum())
        zA = model.enc.encode_many([P.a(i) for i in vidx]).float()
        zB = model.enc.encode_many([P.b(i) for i in vidx]).float()
    s = zA @ zB.T
    y = torch.arange(len(vidx), device=s.device)
    top1 = float(((s.argmax(1) == y).float().mean() + (s.argmax(0) == y).float().mean()) / 2)
    model.train()
    return {"ce": tot / n, "ce_zshuf": tot_shuf / n, "z_gain": (tot_shuf - tot) / n, "xvoice_top1": top1}


def val_index(P: Pairs, n: int = 2000) -> np.ndarray:
    h = np.flatnonzero(P.held & (P.j % 2 == 0))
    return np.random.default_rng(1234).choice(h, min(n, len(h)), replace=False)


def cmd_train(a) -> None:
    import airon

    torch.manual_seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    T = load_table(Path(a.dict))
    P = Pairs(T, Path(a.pairs), distort_b=a.distort)
    if a.bag > 0:
        set_idf(T)
    while P.refresh() == 0 or (~P.held).sum() == 0 or P.held.sum() < 2000:
        log("waiting for pairs (need held-out and training shards)")
        time.sleep(30)
    vidx = val_index(P)
    base = hist_retrieval([P.a(i) for i in vidx], [P.b(i) for i in vidx])
    model = ModelD(a.enc_layers, a.dec_layers, a.enc_type, a.bag > 0).cuda()
    if a.init:
        model.load_state_dict(torch.load(a.init, map_location="cuda", weights_only=False)["model"])
        log("initialised from", a.init)
    meta = {"run_id": str(uuid.uuid4()), "params": nparams(model), "params_encoder": nparams(model.enc),
            "params_decoder": nparams(model.dec), "args": vars(a), "git": os.environ.get("GIT_COMMIT", ""),
            "baseline_hist_xvoice_top1": base, "val_pairs": len(vidx)}
    (out / "metadata.json").write_text(json.dumps(meta, indent=1))
    log("PARAMS", json.dumps({k: meta[k] for k in ("params", "params_encoder", "params_decoder")}),
        "baseline hist xvoice top1", round(base, 4))
    run = airon.init(project="dauys", name=a.name, config=meta, experiment=a.experiment, tags=["dy36", "dy36-d"],
                     total_steps=a.steps, run_id=meta["run_id"], mode="disabled" if a.tracking == "off" else None)
    decay = [q for n, q in model.named_parameters() if q.dim() >= 2 and "emb" not in n]
    other = [q for n, q in model.named_parameters() if not (q.dim() >= 2 and "emb" not in n)]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.05}, {"params": other, "weight_decay": 0.0}],
                            lr=a.lr, betas=(0.9, 0.98), fused=True)
    rng = np.random.default_rng(a.seed)
    lock = threading.Lock()
    q: queue.Queue = queue.Queue(maxsize=64)

    def producer() -> None:
        r = np.random.default_rng(a.seed + 1)
        while True:
            with lock:  # refresh swaps the index arrays
                bts = batches(P, r, a.budget, a.maxb)
            for bt in bts:
                q.put(bt)

    for _ in range(2):
        threading.Thread(target=producer, daemon=True).start()
    hist, best, stale, t0 = [], 1e9, 0, time.time()
    acc = {"loss": 0.0, "ce": 0.0, "nce": 0.0, "n": 0, "utt": 0}
    for step in range(a.steps):
        lr = a.lr * min(1, (step + 1) / a.warmup) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1, step / a.steps))))
        for g in opt.param_groups:
            g["lr"] = lr
        x, la, b, lb, v = [t.cuda(non_blocking=True) for t in q.get()]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, ce, nce = step_loss(model, x, la, b, lb, v, a)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        acc["loss"] += loss.item(); acc["ce"] += ce.item(); acc["nce"] += nce.item(); acc["n"] += 1; acc["utt"] += len(x)
        if (step + 1) % 100 == 0:
            lg = {k: round(acc[k] / acc["n"], 4) for k in ("loss", "ce", "nce")}
            lg.update(lr=lr, utt_per_batch=acc["utt"] / acc["n"], steps_per_s=round(100 / (time.time() - t0), 2),
                      pair_shards=len(P.seen), train_pairs=int((~P.held).sum()))
            t0 = time.time()
            log(json.dumps({"step": step + 1, **lg}))
            run.log({f"train/{k}": v for k, v in lg.items()}, step=step + 1)
            acc = {"loss": 0.0, "ce": 0.0, "nce": 0.0, "n": 0, "utt": 0}
        if (step + 1) % 1000 == 0:
            with lock:
                n = P.refresh()
            if n:
                log("pairs refreshed: +", n, "shards; train pairs", int((~P.held).sum()))
        late = a.stop_at and time.time() > a.stop_at
        if (step + 1) % a.val_every == 0 or late or step + 1 == a.steps:
            r = evaluate(model, P, vidx, a)
            r["step"] = step + 1
            hist.append(r)
            log("VAL", json.dumps(r))
            run.log({f"val/{k}": v for k, v in r.items() if k != "step"}, step=step + 1)
            ck = {"model": model.state_dict(), "step": step + 1, "val": r, "args": vars(a)}
            torch.save(ck, out / "last.pt")
            score = r["ce"] - r["xvoice_top1"]  # both should improve; lower is better
            if score < best - 1e-3:
                best, stale = score, 0
                torch.save(ck, out / "best.pt")
            else:
                stale += 1
            (out / "val_history.json").write_text(json.dumps(hist, indent=1))
            if a.patience and stale >= a.patience:
                log("early stop", step + 1)
                break
            if late:
                log("stop-at reached", step + 1)
                break
    (out / "DONE").write_text(json.dumps(hist[-1]))
    run.finish()
    log("TRAIN DONE")


# ---------------------------------------------------------------- check (i): synthetic known answer
class Synth:
    def __init__(self, seed: int = 0, P: int = 40, W: int = 300, V: int = 64) -> None:
        r = np.random.default_rng(seed)
        self.words = [r.integers(0, P, size=r.integers(2, 6)) for _ in range(W)]
        shared = r.choice(K, P, replace=False)
        self.vmap = np.where(r.random((V, P)) < 0.3, shared[None], r.integers(0, K, (V, P)))
        self.vvec = r.standard_normal((V, VDIM)).astype(np.float32)
        self.W, self.V = W, V

    def render(self, r, bag, voice) -> np.ndarray:
        seq = []
        for w in bag:
            for p in self.words[w]:
                seq += [self.vmap[voice, p]] * int(r.integers(2, 7))
        x = np.array(seq)
        noise = r.random(len(x)) < 0.1
        x[noise] = r.integers(0, K, int(noise.sum()))
        return x

    def bag(self, r):
        return r.choice(self.W, int(r.integers(5, 13)), replace=False)


def cmd_synth(a) -> None:
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    S = Synth()
    r = np.random.default_rng(1)
    model = ModelD(a.enc_layers, a.dec_layers).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.05)
    t0 = time.time()
    res = {}

    def test() -> dict:
        model.eval()
        rt = np.random.default_rng(99)
        bags = [S.bag(rt) for _ in range(200)]
        x0 = [S.render(rt, b, 0) for b in bags]
        x1 = [S.render(rt, b, 1) for b in bags]
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            z0 = model.enc.encode_many(x0).float()
            z1 = model.enc.encode_many(x1).float()
        s_other = z0 @ z1.T
        s_same = z0 @ z0.T
        s_same.fill_diagonal_(-9)
        best_other, arg_other = s_other.max(1)
        best_same = s_same.max(1).values
        y = torch.arange(200, device=z0.device)
        right = ((arg_other == y) & (best_other > best_same)).float().mean().item()
        same_voice = (best_same >= best_other).float().mean().item()
        h = hist_retrieval(x0, x1)
        model.train()
        return {"top1_same_bag_other_voice": right, "top1_same_voice_other_bag": same_voice, "hist_baseline_xvoice": h}

    res["before"] = test()
    log("SYNTH before", json.dumps(res["before"]))
    for step in range(a.steps):
        bags = [S.bag(r) for _ in range(64)]
        va = r.integers(0, S.V, 64)
        vb = (va + r.integers(1, S.V, 64)) % S.V
        A, la = pad([S.render(r, b, v) for b, v in zip(bags, va)])
        B, lb = pad([S.render(r, b, v) for b, v in zip(bags, vb)])
        vv = torch.from_numpy(S.vvec[vb])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, ce, nce = step_loss(model, A.cuda(), la.cuda(), B.cuda(), lb.cuda(), vv.cuda(), a)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if (step + 1) % 250 == 0:
            t = test()
            log("SYNTH", step + 1, round(ce.item(), 3), round(nce.item(), 3), json.dumps(t))
            res[f"step{step + 1}"] = {**t, "ce": ce.item(), "nce": nce.item()}
    res["final"] = test()
    res["pass"] = res["final"]["top1_same_bag_other_voice"] >= 0.9
    res["args"] = vars(a)
    res["sec"] = time.time() - t0
    (out / f"synth_nce{a.nce}.json").write_text(json.dumps(res, indent=1))
    log("SYNTH DONE", json.dumps(res["final"]), "pass", res["pass"])


# ---------------------------------------------------------------- checks (ii), (iii) and the dictionary
def load_model(path: str):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    ar = ck["args"]
    m = ModelD(ar.get("enc_layers", 6), ar.get("dec_layers", 2), ar.get("enc_type", "global"), ar.get("bag", 0) > 0).cuda().eval()
    m.load_state_dict(ck["model"])
    return m, ck


def z_rows(model, T: dict, rows: np.ndarray) -> np.ndarray:
    co, off = T["codes"], T["code_off"]
    out = np.zeros((len(rows), ZDIM), np.float32)
    step = 20000
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for s in range(0, len(rows), step):
            us = [co[off[u] : off[u + 1]].astype(np.int64) for u in rows[s : s + step]]
            out[s : s + len(us)] = model.enc.encode_many(us, budget=int(os.environ.get("ENC_BUDGET", 16384))).float().cpu().numpy()
    return out


def probe(z: np.ndarray, T: dict, rows: np.ndarray) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    tot, acc, maj, nrec = 0, 0.0, 0.0, 0
    for e in np.unique(T["ep"][rows]):
        m = T["ep"][rows] == e
        y = T["spk"][rows][m]
        cl, cn = np.unique(y, return_counts=True)
        okc = cl[cn >= 10]
        if len(okc) < 2:
            continue
        k = np.isin(y, okc)
        zz, yy = z[m][k], y[k]
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
        s = cross_val_score(clf, zz, yy, cv=StratifiedKFold(5, shuffle=True, random_state=0)).mean()
        n = len(yy)
        tot += n
        acc += s * n
        maj += np.unique(yy, return_counts=True)[1].max()
        nrec += 1
    return {"probe_acc": round(acc / tot, 4), "majority": round(maj / tot, 4), "recordings": nrec, "utterances": tot}


def cmd_checks(a) -> None:
    T = load_table(Path(a.dict))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    model, ck = load_model(a.ckpt)
    res = {"ckpt": a.ckpt, "step": ck["step"], "val_at_ckpt": ck["val"]}
    P = Pairs(T, Path(a.pairs))
    P.refresh()
    vidx = val_index(P)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        zA = model.enc.encode_many([P.a(i) for i in vidx]).float()
        zB = model.enc.encode_many([P.b(i) for i in vidx]).float()
    s = zA @ zB.T
    y = torch.arange(len(vidx), device=s.device)
    res["ii_xvoice_top1"] = float(((s.argmax(1) == y).float().mean() + (s.argmax(0) == y).float().mean()) / 2)
    res["ii_hist_baseline"] = hist_retrieval([P.a(i) for i in vidx], [P.b(i) for i in vidx])
    res["ii_n"] = len(vidx)
    spk = T["spk"][P.uid[vidx]]
    res["ii_same_spk_top1"], res["ii_same_spk_candidates"] = same_spk_top1(s.cpu().numpy(), spk)
    res["ii_same_spk_hist_baseline"], _ = same_spk_top1(hist_mat([P.a(i) for i in vidx], [P.b(i) for i in vidx]), spk)
    held = np.flatnonzero(T["held"])
    zD = z_rows(model, T, held)
    res["iii_D"] = probe(zD, T, held)
    log("iii D", res["iii_D"])
    for nm in a.others.split(","):
        f = Path(a.dict) / f"dict_{nm}.npz"
        if f.exists():
            res[f"iii_{nm}"] = probe(np.load(f)["raw"][held], T, held)
            log("iii", nm, res[f"iii_{nm}"])
    (out / f"checks_step{ck['step']}.json").write_text(json.dumps(res, indent=1))
    log("CHECKS", json.dumps(res))


def cmd_dict(a) -> None:
    T = load_table(Path(a.dict))
    model, ck = load_model(a.ckpt)
    n = len(T["ep"])
    raw = z_rows(model, T, np.arange(n))
    th, valid = centre(raw, T["spk"])
    np.savez(Path(a.dict) / f"dict_{a.name}.npz", raw=raw, thought=th, valid=valid,
             info=np.array(json.dumps({"ckpt": a.ckpt, "step": ck["step"], "val": ck["val"], "rule": __doc__.split("\n")[0]})))
    np.savez(Path(a.dict) / f"dict_{a.name}raw.npz", raw=raw, thought=raw, valid=valid)
    log("DICT", a.name, "n", n, "valid", float(valid.mean()))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["synth", "train", "checks", "dict"])
    p.add_argument("--dict", default="/data/dy36/dict")
    p.add_argument("--pairs", default="/data/dy36/d/pairs")
    p.add_argument("--out", default="/data/dy36/d/run")
    p.add_argument("--ckpt", default="")
    p.add_argument("--init", default="", help="start from the weights of this checkpoint (new optimiser)")
    p.add_argument("--name", default="dy36-d")
    p.add_argument("--others", default="A,C2500")
    p.add_argument("--enc-layers", type=int, default=6)
    p.add_argument("--dec-layers", type=int, default=2)
    p.add_argument("--mask-a", type=float, default=0.15)
    p.add_argument("--vis", type=float, default=0.2)
    p.add_argument("--block", type=int, default=10)
    p.add_argument("--nce", type=float, default=0.5)
    p.add_argument("--enc-type", default="global", choices=["global", "local"])
    p.add_argument("--bag", type=float, default=0.0, help="weight of the tf-idf code-bag target predicted from z alone")
    p.add_argument("--distort", action="store_true", help="per-segment tempo and frame drop/dup on the voice-B side")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--budget", type=int, default=32000)
    p.add_argument("--maxb", type=int, default=256)
    p.add_argument("--val-every", type=int, default=500)
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--stop-at", type=float, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tracking", default="on", choices=["on", "off"])
    p.add_argument("--experiment", default=EXPERIMENT)
    a = p.parse_args()
    if a.cmd == "synth" and a.steps == 20000:
        a.steps = 1500
    {"synth": cmd_synth, "train": cmd_train, "checks": cmd_checks, "dict": cmd_dict}[a.cmd](a)


if __name__ == "__main__":
    main()
