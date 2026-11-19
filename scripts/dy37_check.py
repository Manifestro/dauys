"""DY-37 checklist before training (results -> checklist.json / checklist.md).

  1 leakage   - held-out shards not in training; dictionary rows matched to the utterances and checked on the codes;
                on 300 training + 100 held-out micro-batches: a thought reaches the input only on frames of its own
                "me" utterance (never before it starts); masked inputs are exactly the utterance frames, targets
                unchanged; roles equal a recomputation from (recording, sequence number) alone and do not change when
                the same sequence is cut at another start; candidates never overlap the heard part, decoys are the
                same "me" speaker >= 60 s away, length within x2, other-recording candidates are other recordings;
                decided thoughts (bank) never come from the target's recording.
  2 known answer, first half: marker in the heard part -> recognition of the next "me" utterance's thought >= 0.9;
                shuffled markers -> chance.
  3 known answer, second half: every frame of an utterance drawn from its class distribution, thought = class vector:
                swapped thought >= 10 % worse CE than true (and true below null), predictions follow the swap; with
                MASKED input and the true thought the class is recovered >= 0.9, with a swapped one it follows the swap.
  4 overfit one batch (full model) and a 200-step dry run on 2 shards (full model, real code path); all logged
                quantities finite; parameter count.

    python scripts/dy37_check.py --utt /opt/dlami/nvme/utt --dict D.npz --table table.npz --out /data/t/check37
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
from dy35_train import EOS, GAP_S, HEARD, N_SAME, Corpus
from dy37_train import (MASK, VOCAB, Bank, Builder, frame_roles, load_all, make_models, me_of, prepare, step_loss,
                        thought_table, train)

from dauys.dy21_lm import LMConfig
from dauys.dy37_model import ROLE_ME, param_counts

SMALL = LMConfig(vocab=VOCAB, d=256, layers=4, heads=4, ffn=1024, ctx=1024)


# ---------------------------------------------------------------- 1 leakage
def leak_batches(c: Corpus, T: int, n: int, seed: int) -> dict:
    b = Builder(c, T, np.random.default_rng(seed))
    rng = np.random.default_rng(seed + 1)
    n_anchor = n_dec = n_slot = n_mask = n_shift = 0
    mind = 1e9
    for it in range(n):
        starts = rng.integers(0, len(c.stream) - T - 1, size=16)
        seqs = it * 16 + np.arange(16)
        mb = b.build(starts, seqs, pred_on=True)
        for bi, w in enumerate(starts):
            r, me = frame_roles(c, c.fspk[w : w + T + 1], int(seqs[bi]))
            assert np.array_equal(r[:T], mb["roles"][bi]), "roles differ from the recomputation"
            # the same sequence cut at another start: every recording keeps its "me"
            w2 = int(np.clip(w + rng.integers(-T // 2, T // 2), 0, len(c.stream) - T - 2))
            _, me2 = frame_roles(c, c.fspk[w2 : w2 + T + 1], int(seqs[bi]))
            for s in me & me2:
                assert me_of(c, s // 64, int(seqs[bi])) == s
            for e in {s // 64 for s in me} & {s // 64 for s in me2}:
                assert {s for s in me if s // 64 == e} == {s for s in me2 if s // 64 == e}
                n_shift += 1
            assert np.array_equal(mb["y"][bi], c.stream[w + 1 : w + T + 1]), "targets changed"
        for i, (bi, u, mode, ap, masked) in enumerate(mb["slots"]):
            w = starts[bi]
            pos = np.flatnonzero(mb["upos"][bi] == i)
            assert len(pos) and pos.min() == max(c.st[u] - w, 0) and pos.max() == min(c.en[u] - w, T) - 1
            assert np.all(w + pos >= c.st[u]), "thought before its utterance starts"
            assert c.spk[u] in set(mb["me"][bi]) and c.zvalid[u]
            xs = mb["x"][bi, pos]
            if masked:
                assert np.all(xs == MASK)
                n_mask += 1
            else:
                assert np.array_equal(xs, c.stream[w + pos])
            n_slot += 1
        outside = (mb["upos"] < 0)
        assert not np.any(mb["x"][outside] == MASK), "mask outside a me utterance"
        for bi, p, cand in mb["anchors"]:
            w = starts[bi]
            u = cand[0]
            assert c.st[u] == w + p + 1 and p >= HEARD - 1 and c.sin[u] >= HEARD
            assert c.spk[u] in set(mb["me"][bi])
            heard = (w, w + p + 1)
            for d in cand:
                assert c.en[d] <= heard[0] or c.st[d] >= heard[1], "candidate overlaps the heard part"
            for d in cand[1 : 1 + N_SAME]:
                assert c.spk[d] == c.spk[u] and d != u and c.zvalid[d]
                dist = c.t0[u] - c.t1[d] if c.t1[d] <= c.t0[u] else c.t0[d] - c.t1[u]
                assert dist >= GAP_S, dist
                assert 0.5 <= c.len[d] / c.len[u] <= 2.0
                mind = min(mind, dist)
                n_dec += 1
            for d in cand[1 + N_SAME :]:
                assert c.ep[d] != c.ep[u]
            n_anchor += 1
    return {"micro_batches": n, "me_slots": n_slot, "masked_slots": n_mask, "masked_share": round(n_mask / max(n_slot, 1), 3),
            "anchors": n_anchor, "same_speaker_decoys": n_dec, "min_decoy_distance_s": round(float(mind), 2),
            "recordings_checked_under_shifted_cut": n_shift}


def decided_leak(tr: Corpus, n: int = 20) -> dict:
    """Decided thoughts from the bank (small model, real data): never from the target's recording, never the target."""
    dev = torch.device("cuda")
    model, tgt = make_models(SMALL, dev)
    Zg = torch.from_numpy(tr.Z).to(dev)
    ep = torch.from_numpy(tr.ep).to(dev)
    bank = Bank(tr, 100000, dev)
    b = Builder(tr, 2048, np.random.default_rng(31))
    rng = np.random.default_rng(32)
    n_dec = 0
    for it in range(n):
        mb = b.build(rng.integers(0, len(tr.stream) - 2049, size=4), it * 4 + np.arange(4), pred_on=True)
        dbg = {}
        with torch.no_grad():
            step_loss(model, tgt, mb, Zg, ep, bank, dev, True, debug=dbg)
        for u, bu in dbg.get("decided", []):
            assert bu != u and tr.ep[bu] != tr.ep[u]
            n_dec += 1
    del model, tgt
    torch.cuda.empty_cache()
    return {"decided_checked": n_dec, "bank_size": len(bank.ids), "rule": "bank = 100,000 fixed training utterances; same recording masked out"}


# ---------------------------------------------------------------- synthetic corpora
def synth(kind: str, seed: int, n_ep: int = 40, ep_frames: int = 30000) -> Corpus:
    """Two speakers alternating; 20-frame gap (no speech) before each utterance. marker: gap = 1500 + m, utterance
    starts with 5 frames 1000 + m (400 classes); shuffled: gap marker random; zdist: gap 1999, every utterance frame
    from class distribution m % 32 over codes 1000-1199. Thought = fixed unit vector of the class."""
    rng = np.random.default_rng(seed)
    dist = np.random.default_rng(777).dirichlet(np.full(200, 0.3), size=32)
    ncls = 400 if kind in ("marker", "shuffled") else 32
    cvec = np.random.default_rng(778).normal(size=(ncls, 64))
    cvec /= np.linalg.norm(cvec, axis=1, keepdims=True)
    cls, parts, st, en, ep, spk, t0, t1, sin = [], [], [], [], [], [], [], [], []
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
            elif kind == "zdist":
                m = m % 32
                x[t:s] = 1999
                x[s : s + L] = 1000 + rng.choice(200, L, p=dist[m])
            cls.append(m)
            st.append(pos + s); en.append(pos + s + L); ep.append(e); spk.append(e * 64 + (k % 2))
            t0.append(s / 50); t1.append((s + L) / 50); sin.append(s)
            t, k = s + L, k + 1
        parts += [x, np.array([EOS], np.int16)]
        pos += ep_frames + 1
    a = np.asarray
    c = Corpus(np.concatenate(parts), a(st), a(en), a(ep), a(spk), a(t0), a(t1), a(sin), [f"synth-{kind}"])
    c.cls = a(cls)
    c.Z = cvec[c.cls].astype(np.float32)
    c.zvalid = np.ones(c.n, bool)
    fs = np.full(len(c.stream), -1, np.int32)
    for u in range(c.n):
        fs[c.st[u] : c.en[u]] = c.spk[u]
    prepare(c, fs, 1)
    return c


def small_args(steps: int, out: str) -> SimpleNamespace:
    return SimpleNamespace(steps=steps, batch=32, micro=16, lr=3e-4, warmup=50, val_every=steps, val_windows=128,
                           log_every=50, ema=0.999, seed=1, pred_from=10**9, chance_step=10**9, tracking="off",
                           experiment="", name="dy37-check", stop_at=0, no_last=True, limit=0, out=out, dict="synthetic",
                           dict_key="class vector", bank_n=100000)


@torch.no_grad()
def follow(model, c: Corpus, T: int, masked: bool, n_batches: int = 16) -> dict:
    """Class implied by the model's own predictions on utterance frames 1-20 (mean predicted distribution over codes
    1000-1199; class = argmax_c sum_v p(v) log dist_c(v)) with the true and with a swapped thought."""
    dev = torch.device("cuda")
    dist = torch.from_numpy(np.random.default_rng(777).dirichlet(np.full(200, 0.3), size=32)).float().to(dev)
    ld = torch.log(dist + 1e-9)
    Zg = torch.from_numpy(c.Z).to(dev)
    b = Builder(c, T, np.random.default_rng(51))
    rng = np.random.default_rng(52)
    r = {"n": 0, "true_ok": 0, "nd": 0, "swap_swap": 0, "swap_true": 0}
    model.eval()
    for it in range(n_batches):
        starts = rng.integers(0, len(c.stream) - T - 1, size=16)
        mb = b.build(starts, it * 16 + np.arange(16), pred_on=False, val=True, p_mask=1.0 if masked else 0.0)
        x = torch.from_numpy(mb["x"]).to(dev)
        roles = torch.from_numpy(mb["roles"]).to(dev)
        items = []
        for (bi, u, mode, ap, msk), u2 in zip(mb["slots"], mb["swap_u"]):
            q0 = int(c.st[u] - starts[bi])
            if q0 >= 0 and q0 + 21 <= T and u2 >= 0:
                items.append((bi, q0, int(c.cls[u]), int(c.cls[u2])))
        if not items:
            continue
        out = {}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            for k in ("true", "swap"):
                if k == "true":
                    zin, has = thought_table(mb, Zg, dev)
                else:
                    sw = {i: Zg[u2] for i, u2 in enumerate(mb["swap_u"]) if u2 >= 0}
                    zin, has = thought_table(mb, Zg, dev, mode_over=[0 if u2 >= 0 else 2 for u2 in mb["swap_u"]], z_over=sw)
                h = model.states(x, roles, zin, has)
                pc = []
                for bi, q0, _, _ in items:
                    p = torch.softmax(model.logits(h[bi, q0 : q0 + 20]), -1)[:, 1000:1200].mean(0)
                    pc.append(int((ld @ (p / p.sum())).argmax()))
                out[k] = pc
        for j, (_, _, ct, cs) in enumerate(items):
            r["n"] += 1
            r["true_ok"] += out["true"][j] == ct
            if cs != ct:
                r["nd"] += 1
                r["swap_swap"] += out["swap"][j] == cs
                r["swap_true"] += out["swap"][j] == ct
    model.train()
    n, nd = max(r["n"], 1), max(r["nd"], 1)
    return {"masked_input": masked, "n": r["n"], "n_swap_with_other_class": r["nd"], "acc_true_thought": round(r["true_ok"] / n, 4),
            "swap_predicts_swap_class": round(r["swap_swap"] / nd, 4), "swap_predicts_true_class": round(r["swap_true"] / nd, 4)}


def known(kind: str, steps: int, out: Path, n_ep: int = 40) -> dict:
    tr, va = synth(kind, 1, n_ep=n_ep), synth(kind, 2, n_ep=10)
    t = time.time()
    res = train(small_args(steps, str(out)), tr, va, SMALL, out, log=lambda s: None)
    r = {"kind": kind, "steps": steps, "train_s": round(time.time() - t, 1), "val": res["history"][-1]}
    if kind == "zdist":
        r["follow_unmasked"] = follow(res["model"], va, SMALL.ctx, False)
        r["follow_masked"] = follow(res["model"], va, SMALL.ctx, True)
    del res
    torch.cuda.empty_cache()
    return r


def overfit(tr: Corpus, cfg: LMConfig, steps: int, B: int = 2) -> dict:
    dev = torch.device("cuda")
    torch.manual_seed(0)
    model, tgt = make_models(cfg, dev)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95))
    Zg = torch.from_numpy(tr.Z).to(dev)
    ep = torch.from_numpy(tr.ep).to(dev)
    bank = Bank(tr, 100000, dev)
    b = Builder(tr, cfg.ctx, np.random.default_rng(5))
    mb = None
    rng = np.random.default_rng(6)
    while mb is None or not mb["anchors"]:
        mb = b.build(rng.integers(0, len(tr.stream) - cfg.ctx - 1, size=B), np.arange(B), pred_on=False)
    hist = []
    for s in range(steps):
        for g in opt.param_groups:
            g["lr"] = 3e-4 * min(1, (s + 1) / 5)
        opt.zero_grad(set_to_none=True)
        total, part = step_loss(model, tgt, mb, Zg, ep, bank, dev, False)
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        hist.append({k: round(part[k], 4) for k in ("ce_w", "ce_me", "nce", "acc16", "jepa")})
    pc = param_counts(model)
    mem = torch.cuda.max_memory_allocated() / 1e9
    del model, tgt, opt, bank
    torch.cuda.empty_cache()
    return {"steps": steps, "windows": B, "anchors": len(mb["anchors"]), "first": hist[0], "last": hist[-1],
            "ce_fell": hist[-1]["ce_w"] < hist[0]["ce_w"] - 0.5, "nce_fell": hist[-1]["nce"] < hist[0]["nce"] - 0.5,
            "params": pc, "max_gpu_gb": round(mem, 1)}


def finite(v) -> bool:
    if isinstance(v, dict):
        return all(finite(x) for x in v.values())
    if isinstance(v, (list, tuple)):
        return all(finite(x) for x in v)
    if isinstance(v, float):
        return math.isfinite(v)
    return True


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--utt", required=True)
    p.add_argument("--dict", required=True)
    p.add_argument("--table", required=True)
    p.add_argument("--dict-key", default="thought")
    p.add_argument("--out", required=True)
    p.add_argument("--only", default="1,2,3,4")
    p.add_argument("--known-steps", type=int, default=1500)
    p.add_argument("--known-steps3", type=int, default=3000)
    p.add_argument("--dry-batch", type=int, default=4)
    p.add_argument("--dry-micro", type=int, default=2)
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    only = set(a.only.split(","))
    R: dict = {}
    jpath = out / "checklist.json"
    if jpath.exists():
        R = json.loads(jpath.read_text())
    save = lambda: jpath.write_text(json.dumps(R, indent=1, default=str))
    cfg = LMConfig(vocab=VOCAB, d=1024, layers=24, heads=16, ffn=4096, ctx=2048)
    files = sorted(Path(a.utt).glob("*.npz"))
    if "1" in only or "4" in only:
        t = time.time()
        tr, va, info = load_all(a.utt, a.dict, a.table, a.dict_key, limit=0 if "1" in only else 2)
        print("loaded", round(time.time() - t, 1), json.dumps(info), flush=True)
    if "1" in only:
        tr_files = [f for f in files if not held_out(f.stem)]
        va_files = [f for f in files if held_out(f.stem)]
        r = {"held_out_shards": [f.stem for f in va_files], "training_shards": len(tr_files),
             "held_out_in_training": sorted(set(f.stem for f in tr_files) & set(f.stem for f in va_files)), "dictionary": info}
        assert not r["held_out_in_training"]
        r["builder_train"] = leak_batches(tr, cfg.ctx, 300, 11)
        r["builder_val"] = leak_batches(va, cfg.ctx, 100, 12)
        r["decided_bank"] = decided_leak(tr)
        r["statement"] = ("a thought is added only on input frames of its own 'me' utterance (never before it starts); "
                          "roles are a function of the recording and the sequence number only, fixed before the window is "
                          "cut; masked inputs are exactly the utterance frames and targets are the stream; candidates never "
                          "overlap the heard part; decided thoughts never come from the target's recording; held-out shards "
                          "are not in training; dictionary rows matched by shard/piece/start/end and checked on the codes")
        r["pass"] = True
        R["1_leakage"] = r
        save()
        print("1", json.dumps(r, default=str), flush=True)
    if "2" in only:
        a2 = known("marker", a.known_steps, out / "k2")
        b2 = known("shuffled", a.known_steps, out / "k2s")
        R["2_known_first_half"] = {"marker": a2, "shuffled": b2,
                                   "pass": a2["val"]["acc16"] >= 0.9 and b2["val"]["acc16"] <= 0.15}
        save()
        print("2", json.dumps(R["2_known_first_half"]), flush=True)
    if "3" in only:
        r = known("zdist", a.known_steps3, out / "k3", n_ep=400)
        v = r["val"]
        fu, fm = r["follow_unmasked"], r["follow_masked"]
        r["rule"] = ("unmasked: CE with a swapped thought >= 10 % above CE with the true one, true below null, predictions "
                     "follow the swap (majority); masked input: class from the true thought >= 0.9 and the swapped thought "
                     "predicts the swapped class (majority)")
        r["pass"] = bool(v["ceu_swap_minus_true_rel"] >= 0.10 and v["ceu_true"] < v["ceu_null"]
                         and fu["swap_predicts_swap_class"] > 0.5 and fm["acc_true_thought"] >= 0.9
                         and fm["swap_predicts_swap_class"] > 0.5)
        R["3_known_second_half"] = r
        save()
        print("3", json.dumps(r), flush=True)
    if "4" in only:
        tr2 = tr if len(tr.names) <= 2 else None
        if tr2 is None:
            tr2, _, _ = load_all(a.utt, a.dict, a.table, a.dict_key, limit=2)
        t = time.time()
        r = {"overfit": overfit(tr2, cfg, 40), "overfit_s": round(time.time() - t, 1)}
        print("4a", json.dumps(r), flush=True)
        args = small_args(200, str(out / "dry"))
        args.batch, args.micro = a.dry_batch, a.dry_micro
        args.lr, args.warmup, args.val_every, args.val_windows, args.log_every, args.pred_from = 3e-4, 50, 100, 32, 25, 100
        args.dict, args.dict_key = a.dict, a.dict_key
        lines = []

        def log(s: str) -> None:
            lines.append(s)
            print(s, flush=True)

        (out / "dry").mkdir(exist_ok=True)
        res = train(args, tr2, va, cfg, out / "dry", log=log)
        logged = [json.loads(l) for l in lines if l.startswith('{"step"') and "steps_per_s" in l]
        r["dry_run"] = {"steps": 200, "train_shards": 2, "batch": args.batch, "micro": args.micro, "val": res["history"],
                        "logged_last": logged[-1] if logged else None, "steps_per_s_last": [x["steps_per_s"] for x in logged[-3:]],
                        "decided_logged_after_100": [x["decided"] for x in logged if x["step"] > 100]}
        r["all_finite"] = finite(logged) and finite(res["history"])
        r["params"] = r["overfit"]["params"]
        r["pass"] = r["overfit"]["ce_fell"] and r["overfit"]["nce_fell"] and len(res["history"]) >= 2 and r["all_finite"]
        R["4_overfit_dry"] = r
        save()
        print("4", json.dumps(r), flush=True)
    lines = ["# DY-37 t: checklist before training", "", f"dictionary: {a.dict} (key {a.dict_key})", ""]
    for k, v in R.items():
        lines += [f"## {k}: {'PASS' if v.get('pass') else 'FAIL'}", "", "```json", json.dumps(v, indent=1, default=str), "```", ""]
    (out / "checklist.md").write_text("\n".join(lines))
    print("CHECK DONE", json.dumps({k: v.get("pass") for k, v in R.items()}), flush=True)


if __name__ == "__main__":
    main()
