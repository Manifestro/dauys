"""DY-35 checklist before training (results -> checklist.json / checklist.md).

  1 leakage   - every utterance is exactly one separately encoded chunk (all shards); the gist encoder input of a row
                is exactly its utterance's stream slice; no true utterance and no decoy overlaps the heard part of its
                anchor; decoys >= 60 s from anchor and utterance; held-out shards not in training.
  2 utterances on 100 anchors (+ the whole training table): lengths, speaker before the pause, same-speaker decoys
                by clustering and by the fixed similarity threshold, minimum time distance, usable counts.
  3 known answer, first half: synthetic marker in the heard part -> marker in the next utterance; accuracy >= 0.9;
                shuffled markers -> chance.
  4 known answer, second half: a marker inside the utterance given ONLY by its z: true z >= 0.9, swapped z not.
  5 overfit one batch (full model) and a 200-step dry run on 2 shards (full model, real code path), steps/s.

    python scripts/dy35_check.py --utt /opt/dlami/nvme/utt --out /data/t/check
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from dy21_train import held_out
from dy35_train import (EOS, GAP_S, HEARD, N_SAME, UMAX, UMIN, Builder, Corpus, encode_rows, load_corpus,
                        make_models, step_loss, thought_tables, train, validate)

from dauys.dy21_lm import LMConfig

SIM_THR = 0.5  # fixed before training: same speaker if cos(decoy vector, mean vector of the anchor speaker) >= 0.5


def check_chunks_all(files: list[Path]) -> dict:
    n_u = 0
    for f in files:
        z = np.load(f)
        ch = set(zip(z["chunk_piece"].tolist(), z["chunk_s"].tolist(), z["chunk_e"].tolist()))
        L = z["seg_e"] - z["seg_s"]
        m = (L >= UMIN) & (L <= UMAX)
        for p, s, e in zip(z["seg_piece"][m].tolist(), z["seg_s"][m].tolist(), z["seg_e"][m].tolist()):
            assert (p, s, e) in ch, (f.stem, p, s, e)
        n_u += int(m.sum())
    return {"files": len(files), "utterances_each_one_chunk": n_u}


def leak_batches(c: Corpus, T: int, n: int, seed: int) -> dict:
    """Builder output checks on n micro-batches of 16 windows."""
    b = Builder(c, T, np.random.default_rng(seed))
    rng = np.random.default_rng(seed + 1)
    mind, n_anchor, n_dec = 1e9, 0, 0
    for _ in range(n):
        starts = rng.integers(0, len(c.stream) - T - 1, size=16)
        mb = b.build(starts, pred_on=True)
        pos = np.concatenate([[0], np.cumsum(mb["lens"])])
        for r, u in enumerate(mb["rows_u"]):  # gist encoder input = exactly the utterance slice
            assert np.array_equal(mb["codes"][pos[r] : pos[r + 1]], c.stream[c.st[u] : c.en[u]])
        for (bi, p, cand) in mb["anchors"]:
            w = starts[bi]
            u = mb["rows_u"][cand[0]]
            assert c.st[u] == w + p + 1 and p >= HEARD - 1  # anchor = last frame before the utterance, >= 750 heard
            assert c.sin[u] >= HEARD  # the heard 750 frames are inside the utterance's piece
            heard = (w, w + p + 1)
            for r in cand:
                d = mb["rows_u"][r]
                assert c.en[d] <= heard[0] or c.st[d] >= heard[1], "candidate overlaps the heard part"
            for r in cand[1 : 1 + N_SAME]:
                d = mb["rows_u"][r]
                assert c.spk[d] == c.spk[u] and c.ep[d] == c.ep[u] and d != u
                dist = min(abs(c.t0[u] - c.t1[d]), abs(c.t0[d] - c.t1[u])) if not (c.t1[d] <= c.t0[u] or c.t0[d] >= c.t1[u]) else (
                    c.t0[u] - c.t1[d] if c.t1[d] <= c.t0[u] else c.t0[d] - c.t1[u])
                assert dist >= GAP_S, dist
                assert 0.5 <= c.len[d] / c.len[u] <= 2.0
                mind = min(mind, dist)
                n_dec += 1
            for r in cand[1 + N_SAME :]:
                assert c.ep[mb["rows_u"][r]] != c.ep[u]
            n_anchor += 1
    return {"micro_batches": n, "anchors": n_anchor, "same_speaker_decoys": n_dec, "min_decoy_distance_s": round(mind, 2)}


def stats_100(c: Corpus, T: int, seed: int) -> dict:
    b = Builder(c, T, np.random.default_rng(seed))
    rng = np.random.default_rng(seed + 7)
    A = []
    while len(A) < 100:
        starts = rng.integers(0, len(c.stream) - T - 1, size=16)
        mb = b.build(starts, pred_on=False)
        for bi, p, cand in mb["anchors"]:
            A.append((mb["rows_u"][cand[0]], [mb["rows_u"][r] for r in cand[1 : 1 + N_SAME]]))
    A = A[:100]
    u = np.array([a[0] for a in A])
    L = c.len[u] / 50
    diff = c.prev_spk[u] != (c.spk[u] % 64)
    has_prev = c.prev_spk[u] >= 0
    emb = c.emb.astype(np.float32)
    cent = {}
    sims, dists, sims_other = [], [], []
    for uu, ds in A:
        g = int(c.spk[uu])
        if g not in cent:
            v = emb[c.groups[c.grp[uu]]].mean(0)
            cent[g] = v / np.linalg.norm(v)
        for d in ds:
            sims.append(float(emb[d] @ cent[g]))
            dists.append(min(abs(c.t0[uu] - c.t1[d]), abs(c.t0[d] - c.t1[uu])))
        # context: similarity of the other speakers of the same episode to this speaker's mean
        others = np.flatnonzero((c.ep == c.ep[uu]) & (c.spk != c.spk[uu]))[:50]
        sims_other += (emb[others] @ cent[g]).tolist()
    sims = np.array(sims)
    Lall = c.len / 50
    return {
        "anchors": 100,
        "anchor_utt_len_s_pct10_25_50_75_90": np.percentile(L, [10, 25, 50, 75, 90]).round(2).tolist(),
        "all_utt_len_s_pct10_25_50_75_90": np.percentile(Lall, [10, 25, 50, 75, 90]).round(2).tolist(),
        "share_next_speaker_differs_from_speaker_before": round(float(diff[has_prev].mean()), 3),
        "anchors_with_a_previous_segment_in_piece": int(has_prev.sum()),
        "gap_before_utterance_s_pct10_50_90": np.percentile(c.prev_gap[u][has_prev] / 50, [10, 50, 90]).round(2).tolist(),
        "decoys_same_speaker_by_clustering": 1.0,
        "similarity_threshold_fixed": SIM_THR,
        "decoys_same_speaker_by_similarity": round(float((sims >= SIM_THR).mean()), 3),
        "decoy_similarity_pct5_50_95": np.percentile(sims, [5, 50, 95]).round(3).tolist(),
        "other_speakers_same_episode_similarity_pct50_95": np.percentile(sims_other, [50, 95]).round(3).tolist() if sims_other else None,
        "other_speakers_same_episode_above_threshold": round(float((np.array(sims_other) >= SIM_THR).mean()), 3) if sims_other else None,
        "min_decoy_distance_s": round(float(min(dists)), 2),
        "usable_utterances": int(c.n),
        "usable_anchor_utterances": int(c.ok_anchor.sum()),
    }


# ---------------------------------------------------------------- synthetic known answers
def synth(kind: str, seed: int, n_ep: int = 40, ep_frames: int = 30000) -> Corpus:
    rng = np.random.default_rng(seed)
    # zdist: 32 fixed classes (same for every corpus), each a Dirichlet(0.3) distribution over codes 1000-1199
    dist = np.random.default_rng(777).dirichlet(np.full(200, 0.3), size=32)
    cls = []
    parts, st, en, ep, spk, t0, t1, sin = [], [], [], [], [], [], [], []
    pos = 0
    for e in range(n_ep):
        x = rng.integers(0, 1000, ep_frames).astype(np.int16)
        t, k = 800, 0
        while True:
            L = int(rng.integers(120, 221))
            if t + 20 + L > ep_frames:
                break
            s = t + 20
            m = int(rng.integers(0, 400))
            if kind == "marker":
                x[t:s] = 1500 + m
                x[s : s + 5] = 1000 + m
            elif kind == "shuffled":
                x[t:s] = 1500 + int(rng.integers(0, 400))
                x[s : s + 5] = 1000 + m
            elif kind == "zmarker":  # 32 marker values, one frame at utterance frame 20
                x[t:s] = 1999
                x[s + 20] = 1000 + m % 32
            elif kind == "zdist":  # every frame of the utterance from its class distribution; heard part: no class info
                x[t:s] = 1999
                x[s : s + L] = 1000 + rng.choice(200, L, p=dist[m % 32])
            cls.append(m % 32)
            st.append(pos + s)
            en.append(pos + s + L)
            ep.append(e)
            spk.append(e * 64 + (k % 2))
            t0.append(s / 50)
            t1.append((s + L) / 50)
            sin.append(s)
            t, k = s + L, k + 1
        parts += [x, np.array([EOS], np.int16)]
        pos += ep_frames + 1
    a = lambda v: np.asarray(v)
    c = Corpus(np.concatenate(parts), a(st), a(en), a(ep), a(spk), a(t0), a(t1), a(sin), [f"synth-{kind}"])
    c.cls = np.asarray(cls)
    return c


def small_args(steps: int, out: str) -> SimpleNamespace:
    return SimpleNamespace(steps=steps, batch=32, micro=16, lr=3e-4, warmup=50, val_every=steps, val_windows=128,
                           log_every=50, ema=0.999, seed=1, pred_from=10**9, chance_step=10**9, tracking="off",
                           experiment="", name="dy35-check", stop_at=0, no_last=True, limit=0, out=out)


SMALL = LMConfig(vocab=2001, d=256, layers=4, heads=4, ffn=1024, ctx=1024)


@torch.no_grad()
def zmarker_acc(model, enc, c: Corpus, T: int, n_batches: int = 8) -> dict:
    """Accuracy of the marker frame (utterance frame 20) with the true z and with z of another utterance."""
    dev = torch.device("cuda")
    b = Builder(c, T, np.random.default_rng(99))
    rng = np.random.default_rng(98)
    hit = {"true": 0, "swap": 0, "null": 0}
    n = 0
    model.eval()
    enc.eval()
    for _ in range(n_batches):
        starts = rng.integers(0, len(c.stream) - T - 1, size=16)
        mb = b.build(starts, pred_on=False, modes=False)
        ids = torch.from_numpy(mb["ids"]).to(dev)
        tg = []
        for i, (bi, u, r, m, a) in enumerate(mb["slots"]):
            q = c.st[u] + 19 - starts[bi]
            if c.st[u] >= starts[bi] and q < T:
                tg.append((bi, int(q), int(c.stream[c.st[u] + 20])))
        if not tg:
            continue
        perm = list(range(len(mb["slots"])))
        perm = perm[1:] + perm[:1]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            Z = encode_rows(enc, mb, dev)
            for k in hit:
                if k == "null":
                    h = model.states(ids[:, :-1], None, None)
                else:
                    sl = mb["slots"] if k == "true" else [(bi, u, mb["slots"][perm[i]][2], m, a) for i, (bi, u, r, m, a) in enumerate(mb["slots"])]
                    zin, has = thought_tables({**mb, "slots": sl}, Z, None, dev)
                    h = model.states(ids[:, :-1], zin, has)
                bb = torch.tensor([t[0] for t in tg], device=dev)
                qq = torch.tensor([t[1] for t in tg], device=dev)
                pr = model.logits(h[bb, qq]).argmax(-1).cpu().numpy()
                hit[k] += int((pr == np.array([t[2] for t in tg])).sum())
        n += len(tg)
    model.train()
    enc.train()
    return {f"acc_{k}": round(v / max(n, 1), 4) for k, v in hit.items()} | {"targets": n}


def known(kind: str, steps: int, out: Path, n_ep: int = 40) -> dict:
    tr, va = synth(kind, 1, n_ep=n_ep), synth(kind, 2, n_ep=10)
    t = time.time()
    res = train(small_args(steps, str(out)), tr, va, SMALL, out, log=lambda s: None)
    r = {"kind": kind, "steps": steps, "train_s": round(time.time() - t, 1), "val": res["history"][-1]}
    if kind == "zmarker":
        r["marker"] = zmarker_acc(res["model"], res["enc"], va, SMALL.ctx)
    if kind == "zdist":
        r["class_from_z"] = class_acc(res["enc"], tr, va)
    return r


@torch.no_grad()
def class_acc(enc, tr: Corpus, va: Corpus) -> dict:
    """Class of a held-out synthetic utterance from its z: nearest class-mean z of 3000 training utterances."""
    dev = torch.device("cuda")
    enc.eval()
    rng = np.random.default_rng(3)
    enc_z = lambda c, us: enc.encode_many([torch.from_numpy(c.codes(int(u)).astype(np.int64)).to(dev) for u in us]).float()
    ut, uv = rng.choice(tr.n, 3000, replace=False), rng.choice(va.n, 1000, replace=False)
    zt, zv = enc_z(tr, ut), enc_z(va, uv)
    ct = torch.from_numpy(tr.cls[ut]).to(dev)
    mu = torch.stack([F.normalize(zt[ct == k].mean(0), dim=0) for k in range(32)])
    acc = float(((zv @ mu.T).argmax(1).cpu().numpy() == va.cls[uv]).mean())
    enc.train()
    return {"acc": round(acc, 4), "chance": 1 / 32, "n": 1000}


def overfit(tr: Corpus, cfg: LMConfig, steps: int) -> dict:
    dev = torch.device("cuda")
    torch.manual_seed(0)
    model, enc, tgt = make_models(cfg, dev)
    opt = torch.optim.AdamW(list(model.parameters()) + list(enc.parameters()), lr=3e-4, betas=(0.9, 0.95))
    b = Builder(tr, cfg.ctx, np.random.default_rng(5))
    starts = np.random.default_rng(6).integers(0, len(tr.stream) - cfg.ctx - 1, size=16)
    mb = b.build(starts, pred_on=False)
    hist = []
    for s in range(steps):
        for g in opt.param_groups:
            g["lr"] = 3e-4 * min(1, (s + 1) / 5)
        opt.zero_grad(set_to_none=True)
        total, part = step_loss(model, tgt, enc, mb, dev, False)
        total.backward()
        torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(enc.parameters()), 1.0)
        opt.step()
        hist.append({k: round(part[k], 4) for k in ("ce", "nce", "acc16", "jepa")})
    del model, enc, tgt, opt
    torch.cuda.empty_cache()
    return {"steps": steps, "anchors": len(mb["anchors"]), "first": hist[0], "last": hist[-1],
            "ce_fell": hist[-1]["ce"] < hist[0]["ce"] - 0.5, "nce_fell": hist[-1]["nce"] < hist[0]["nce"] - 0.5}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--utt", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--only", default="1,2,3,4,5")
    p.add_argument("--known-steps", type=int, default=1500)
    p.add_argument("--known-steps4", type=int, default=1500)
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    only = set(a.only.split(","))
    files = sorted(Path(a.utt).glob("*.npz"))
    tr_files = [f for f in files if not held_out(f.stem)]
    va_files = [f for f in files if held_out(f.stem)]
    R: dict = {}
    jpath = out / "checklist.json"
    if jpath.exists():
        R = json.loads(jpath.read_text())
    save = lambda: jpath.write_text(json.dumps(R, indent=1))
    cfg = LMConfig(vocab=2001, d=1024, layers=24, heads=16, ffn=4096, ctx=2048)
    if "1" in only or "2" in only or "5" in only:
        t = time.time()
        tr = load_corpus(tr_files)
        va = load_corpus(va_files)
        print("loaded", round(time.time() - t, 1), flush=True)
    if "1" in only:
        r = {"held_out_shards": [f.stem for f in va_files], "training_shards": len(tr_files),
             "held_out_in_training": sorted(set(f.stem for f in tr_files) & set(f.stem for f in va_files))}
        assert not r["held_out_in_training"] and all(held_out(f.stem) for f in va_files)
        r["chunks"] = check_chunks_all(files)
        r["builder_train"] = leak_batches(tr, cfg.ctx, 300, 11)
        r["builder_val"] = leak_batches(va, cfg.ctx, 100, 12)
        r["statement"] = ("utterances, decoys and the heard part are separate WavLM chunks (dy35_utt.py: chunks cut at "
                          "every utterance start and end, each encoded alone); the gist encoder input of every row equals "
                          "its utterance's stream slice; no candidate overlaps the heard part; held-out shards are not in "
                          "training")
        r["pass"] = True
        R["1_leakage"] = r
        save()
        print("1", json.dumps(r), flush=True)
    if "2" in only:
        r = stats_100(tr, cfg.ctx, 21)
        r["val"] = {"usable_utterances": int(va.n), "usable_anchor_utterances": int(va.ok_anchor.sum())}
        r["pass"] = r["min_decoy_distance_s"] >= GAP_S
        R["2_utterances"] = r
        save()
        print("2", json.dumps(r), flush=True)
    if "3" in only:
        a3 = known("marker", a.known_steps, out / "k3")
        b3 = known("shuffled", a.known_steps, out / "k3s")
        R["3_known_first_half"] = {"marker": a3, "shuffled": b3, "pass": a3["val"]["acc16"] >= 0.9 and b3["val"]["acc16"] <= 0.15}
        save()
        print("3", json.dumps(R["3_known_first_half"]), flush=True)
    if "4" in only:
        # the gist is a property of the whole utterance: each utterance's frames come from one of 32 code distributions
        # (overlapping, over codes 1000-1199); nothing in the heard part predicts the class. 400 episodes (12M frames),
        # so the small model cannot memorise the random background.
        r = known("zdist", a.known_steps4, out / "k4", n_ep=400)
        v = r["val"]
        r["ceu"] = {"true": v["ceu_true"], "null": v["ceu_null"], "swap": v["ceu_swap"]}
        R["4_known_second_half"] = {**r, "pass": v["ceu_true"] < v["ceu_null"] - 0.05 and v["ceu_true"] < v["ceu_swap"] - 0.05
                                    and r["class_from_z"]["acc"] >= 0.9}
        save()
        print("4", json.dumps(R["4_known_second_half"]), flush=True)
    if "5" in only:
        t = time.time()
        r = {"overfit": overfit(load_corpus(tr_files[:2]), cfg, 40), "overfit_s": round(time.time() - t, 1)}
        save()
        print("5a", json.dumps(r), flush=True)
        tr2 = load_corpus(tr_files[:2])
        args = small_args(200, str(out / "dry"))
        args.lr, args.warmup, args.val_every, args.val_windows, args.log_every = 3e-4, 50, 100, 64, 25
        args.pred_from = 100
        lines = []

        def log(s: str) -> None:
            lines.append(s)
            print(s, flush=True)

        (out / "dry").mkdir(exist_ok=True)
        res = train(args, tr2, va, cfg, out / "dry", log=log)
        sps = [json.loads(l)["steps_per_s"] for l in lines if l.startswith('{"step"') and "steps_per_s" in l]
        r["dry_run"] = {"steps": 200, "train_shards": 2, "val": res["history"], "steps_per_s_last": sps[-3:],
                        "acc16_logged": [json.loads(l)["acc16"] for l in lines if l.startswith('{"step"') and "acc16" in l][-4:]}
        r["pass"] = r["overfit"]["ce_fell"] and r["overfit"]["nce_fell"] and len(res["history"]) >= 2
        R["5_overfit_dry"] = r
        save()
        print("5", json.dumps(r), flush=True)
    lines = ["# DY-35 t: checklist before training", "",
             f"Silence/same-speaker threshold fixed before training: cos(decoy vector, mean vector of the anchor speaker) >= {SIM_THR}.", ""]
    for k, v in R.items():
        lines += [f"## {k}: {'PASS' if v.get('pass') else 'FAIL'}", "", "```json", json.dumps(v, indent=1), "```", ""]
    (out / "checklist.md").write_text("\n".join(lines))
    print("CHECK DONE", json.dumps({k: v.get("pass") for k, v in R.items()}), flush=True)


if __name__ == "__main__":
    main()
