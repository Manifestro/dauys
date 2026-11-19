"""DY-36 thought model: checklist and training.

Data: /data/dy36/dict/table.npz + dict_<name>.npz (scripts/dy36_dict.py). Training = utterances of the training shards;
held-out shards (la-00030, la-00071, la-00109) only for validation / checks.
Objective: InfoNCE (tau 0.1) of the output before every "me" utterance against its dictionary thought, 15 same-speaker
same-recording utterances >= 60 s away (length within x2) and the targets of the other sequences of the batch (other
recordings). Validation: acc16 on 1000 held-out anchors that are NOT the C3 anchors (seed 7). Stop when acc16 has not
improved for --patience validations.

    python scripts/dy36_train.py check --dict /data/dy36/dict --name A --out /data/dy36/check
    python scripts/dy36_train.py train --dict /data/dy36/dict --name A --out /data/dy36/run --experiment "..."
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

sys.path.insert(0, str(Path(__file__).parent))
from dy36_common import log, wilson
from dy36_dict import load_table
from dy36_model import (CTX, N_SAME, Cfg, Data, ThoughtModel, history, last_outputs, leak_assert, make_seq, n_params,
                        nce_loss, train_batch)

EXPERIMENT = ("DY-36: модель замысла — участник разговора сам выбирает, что сказать, из настоящих замыслов; "
              "без звука на входе модели и без текста")


def data_of(T: dict, Dz, rows: np.ndarray) -> Data:
    return Data(T["ep"][rows], T["spk"][rows], T["t0"][rows], T["t1"][rows], T["len"][rows], Dz["raw"][rows],
                Dz["thought"][rows], Dz["valid"][rows])


def load(dict_dir: str, name: str):
    T = load_table(Path(dict_dir))
    Dz = np.load(Path(dict_dir) / f"dict_{name}.npz")
    Dz = {k: Dz[k] for k in ("raw", "thought", "valid")}
    tr_rows, va_rows = np.flatnonzero(~T["held"]), np.flatnonzero(T["held"])
    return T, data_of(T, Dz, tr_rows), data_of(T, Dz, va_rows), tr_rows, va_rows


def c3_anchors(T: dict, va_rows: np.ndarray, utt_dir: str = "/data/utt") -> list[tuple[int, list[int]]]:
    """The 1,200 anchors + 15 decoys of the DY-35 P1 baselines (dy35_eval.anchors(va, 1200, 2026)), mapped to rows of
    the held-out Data (index into va_rows)."""
    from dy21_train import held_out
    from dy35_eval import anchors
    from dy35_train import load_corpus

    files = sorted(f for f in Path(utt_dir).glob("*.npz") if held_out(f.stem))
    va35 = load_corpus(files)
    A = anchors(va35, 1200, 2026)
    base = json.load(open("/data/p1_baselines.json"))
    assert [x[0] for x in A] == base["anchor_ids"], "DY-35 anchors not reproduced"
    keys = []
    for f in files:
        z = np.load(f)
        L = z["seg_e"] - z["seg_s"]
        m = np.flatnonzero((L >= 100) & (L <= 1000))
        keys += [(f.stem, int(p), int(s)) for p, s in zip(z["seg_piece"][m], z["seg_s"][m])]
    assert len(keys) == va35.n
    names = json.loads(str(T["files"]))
    mine = {(names[int(T["fi"][r])], int(T["piece"][r]), int(T["s"][r])): i for i, r in enumerate(va_rows)}
    return [(mine[keys[u]], [mine[keys[d]] for d in ds]) for u, ds in A]


def val_anchors(Dv: Data, exclude: set, n: int = 1000, seed: int = 7) -> list[tuple[int, np.ndarray]]:
    rng = np.random.default_rng(seed)
    ids = np.array([u for u in np.flatnonzero(Dv.ok_target) if u not in exclude])
    ids = np.sort(rng.choice(ids, min(n, len(ids)), replace=False))
    return [(int(u), Dv.decoys(int(u), rng)) for u in ids]


def acc16(model, Dv: Data, anchors, dev, n_hist: int = CTX, seqs=None) -> dict:
    model.eval()
    if seqs is None:
        seqs = [make_seq(Dv, history(Dv, u, n_hist), int(Dv.spk[u])) for u, _ in anchors]
    o = last_outputs(model, seqs, dev)
    C = torch.from_numpy(np.stack([Dv.thought[np.r_[u, d]] for u, d in anchors])).to(dev)
    s = torch.einsum("nd,nkd->nk", o, C)
    hit = (s[:, 0] > s[:, 1:].max(1).values).cpu().numpy()
    model.train()
    return {"acc16": float(hit.mean()), "n": len(hit), "ci95": wilson(int(hit.sum()), len(hit)), "hits": hit}


def make_model(cfg: Cfg, dev):
    m = ThoughtModel(cfg).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1)
    return m, opt


def lr_at(step, steps, peak, warm):
    if step < warm:
        return peak * (step + 1) / warm
    return peak * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, (step - warm) / max(steps - warm, 1)))))


# ---------------------------------------------------------------- synthetic known answer
def synth(kind: str, seed: int, n_ep: int = 400, n_utt: int = 120) -> Data:
    """Two speakers in random turns; thoughts are unit vectors in 64 dims. Every utterance after the first two of a
    speaker = normalise(A @ last heard thought + B @ own previous thought) (A, B random orthogonal; fixed for "rule",
    drawn anew per episode for "shuffled"). raw = thought."""
    rng = np.random.default_rng(seed)
    orth = lambda: np.linalg.qr(rng.standard_normal((64, 64)))[0]
    A0, B0 = orth(), orth()
    ep, spk, th = [], [], []
    for e in range(n_ep):
        A, B = (A0, B0) if kind == "rule" else (orth(), orth())
        who = np.cumsum(rng.random(n_utt) < 0.6) % 2
        last = {0: None, 1: None}
        for i in range(n_utt):
            s = int(who[i])
            heard, own = last[1 - s], last[s]
            if heard is None or own is None:
                v = rng.standard_normal(64)
            else:
                v = A @ heard + B @ own
            v = v / np.linalg.norm(v)
            last[s] = v
            ep.append(e)
            spk.append(e * 64 + s)
            th.append(v)
    n = len(ep)
    t0 = np.tile(np.arange(n_utt) * 7.0, n_ep)
    th = np.array(th, np.float32)
    return Data(ep, spk, t0, t0 + 6.0, np.full(n, 300), th, th, np.ones(n, bool))


def fit(model, opt, D: Data, steps: int, B: int, rng, dev, fixed=None, log_every=50, tag="") -> list[dict]:
    hist = []
    for st in range(steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(st, steps, 3e-4, min(100, steps // 10))
        seqs, tg = fixed if fixed else train_batch(D, rng, B)
        leak_assert(tg)
        loss, lg = nce_loss(model, seqs, tg, D, dev)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if (st + 1) % log_every == 0:
            log(tag, json.dumps({"step": st + 1, **{k: round(v, 4) for k, v in lg.items()}}))
            hist.append({"step": st + 1, **lg})
    return hist


def synth_acc(model, D: Data, dev, n: int = 1000) -> float:
    rng = np.random.default_rng(99)
    ids = np.flatnonzero(D.ok_target & (D.t0 > 30))
    ids = rng.choice(ids, min(n, len(ids)), replace=False)
    A = [(int(u), D.decoys(int(u), rng)) for u in ids]
    return acc16(model, D, A, dev)["acc16"]


def cmd_check(a) -> None:
    dev = torch.device("cuda")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = Cfg()
    R = {}
    # 1. leakage (asserts in code + perturbation test on real data)
    T, Dt, Dv, tr_rows, va_rows = load(a.dict, a.name)
    assert not set(np.unique(T["ep"][tr_rows])) & set(np.unique(T["ep"][va_rows])), "held-out recordings in training"
    names = json.loads(str(T["files"]))
    from dy21_train import held_out
    assert all(not held_out(names[int(f)]) for f in np.unique(T["fi"][tr_rows]))
    rng = np.random.default_rng(0)
    model, opt = make_model(cfg, dev)
    model.eval()
    seqs, tg = train_batch(Dt, rng, 8)
    leak_assert(tg)
    from dy36_model import collate
    bt = collate(seqs, dev)
    with torch.no_grad():
        o1 = model(bt["x"], bt["role"], bt["dur"], bt["pause"], bt["pad"])
    # perturb the input of the utterance at the first target position p of each sequence: outputs 0..p must not move
    bt2 = {k: v.clone() for k, v in bt.items()}
    ps = []
    for i, t in enumerate(tg):
        if len(t["pos"]):
            p = int(t["pos"][0])
            bt2["x"][i, p:] = torch.randn_like(bt2["x"][i, p:])
            ps.append((i, p))
    with torch.no_grad():
        o2 = model(bt2["x"], bt2["role"], bt2["dur"], bt2["pause"], bt2["pad"])
    md = max(float((o1[i, : p + 1] - o2[i, : p + 1]).abs().max()) for i, p in ps)
    moved = max(float((o1[i, p + 1 :] - o2[i, p + 1 :]).abs().max()) for i, p in ps)
    nd = sum(len(t["dec"]) for t in tg)
    R["1_leakage"] = {"pass": bool(md < 1e-4 and moved > 1e-3), "max_change_at_or_before_target": md,
                      "max_change_after": moved, "decoy_rows_checked": nd,
                      "asserts": ["target utterance not among inputs before its output position (leak_assert)",
                                  "inputs are raw + causal speaker-centred vectors; the recording-centred thought is "
                                  "only target/candidate/bank",
                                  "held-out recordings disjoint from training; held-out shards not in training rows",
                                  "decoys same recording, same speaker, >= 60 s away (Data.decoys asserts)",
                                  "bank for choosing = training rows, moments = held-out rows (asserted in dy36_eval)"]}
    log("CHECK 1", json.dumps(R["1_leakage"]))
    model.train()
    # 2. known answer
    ka = {}
    for kind in ("rule", "shuffled"):
        Dtr, Dte = synth(kind, 1), synth(kind, 2, n_ep=60)
        if kind == "rule":
            Dte = synth("rule", 1, n_ep=460)
            keep = Dte.ep >= 400  # same rule (seed 1), unseen episodes
            Dte = Data(Dte.ep[keep], Dte.spk[keep], Dte.t0[keep], Dte.t1[keep], Dte.len[keep], Dte.raw[keep],
                       Dte.thought[keep], Dte.valid[keep])
        m, o = make_model(cfg, dev)
        fit(m, o, Dtr, a.known_steps, 32, np.random.default_rng(1), dev, tag=f"known-{kind}", log_every=200)
        ka[kind] = synth_acc(m, Dte, dev)
        log("CHECK 2", kind, ka[kind])
        del m, o
    R["2_known_answer"] = {"pass": bool(ka["rule"] >= 0.9 and ka["shuffled"] <= 0.15), "acc16_rule": ka["rule"],
                           "acc16_shuffled": ka["shuffled"], "chance": 1 / 16, "steps": a.known_steps,
                           "rule": synth.__doc__}
    # 3. overfit one batch
    m, o = make_model(Cfg(dropout=0.0), dev)
    fixed = train_batch(Dt, np.random.default_rng(5), 16)
    h = fit(m, o, Dt, 300, 16, None, dev, fixed=fixed, tag="overfit", log_every=50)
    R["3_overfit_one_batch"] = {"pass": bool(h[-1]["acc16"] >= 0.95), "first": h[0], "last": h[-1]}
    del m, o
    # 4. dry run on two shards
    two = np.unique(T["fi"][tr_rows])[:2]
    rows = tr_rows[np.isin(T["fi"][tr_rows], two)]
    Dz = np.load(Path(a.dict) / f"dict_{a.name}.npz")
    D2 = data_of(T, {k: Dz[k] for k in ("raw", "thought", "valid")}, rows)
    m, o = make_model(cfg, dev)
    h = fit(m, o, D2, 100, 32, np.random.default_rng(3), dev, tag="dry", log_every=25)
    exclude = set()
    va_a = val_anchors(Dv, exclude, n=300)
    v = acc16(m, Dv, va_a, dev)
    R["4_dry_run_two_shards"] = {"pass": bool(np.isfinite(h[-1]["loss"])), "shards": [names[int(f)] for f in two],
                                 "last": h[-1], "val_acc16_300": v["acc16"]}
    del m, o
    R["params"] = n_params(ThoughtModel(cfg))
    (out / "checklist.json").write_text(json.dumps(R, indent=1, default=float))
    lines = ["# DY-36 checklist (before training the thought model)", "", f"dictionary: {a.name}; model params {R['params']:,}", ""]
    for k, v in R.items():
        if isinstance(v, dict):
            lines.append(f"- **{k}**: {'PASS' if v['pass'] else 'FAIL'} — " +
                         json.dumps({x: y for x, y in v.items() if x not in ('pass', 'rule')}, default=float))
    (out / "checklist.md").write_text("\n".join(lines) + "\n")
    log("CHECK DONE", all(v["pass"] for v in R.values() if isinstance(v, dict)))


def cmd_train(a) -> None:
    import airon

    dev = torch.device("cuda")
    torch.manual_seed(a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    T, Dt, Dv, tr_rows, va_rows = load(a.dict, a.name)
    C3 = c3_anchors(T, va_rows)
    VA = val_anchors(Dv, {u for u, _ in C3})
    cfg = Cfg()
    model, opt = make_model(cfg, dev)
    meta = {"run_id": str(uuid.uuid4()), "dictionary": a.name, "params": n_params(model), "cfg": vars(cfg),
            "ctx_utterances": CTX, "batch_sequences": a.batch, "lr": a.lr, "max_steps": a.steps,
            "train_utterances": Dt.n, "train_targets": int(Dt.ok_target.sum()), "train_episodes": len(Dt.eb),
            "val_anchors": len(VA), "val_rule": "held-out ok targets not in the C3 anchors, seed 7",
            "git": os.environ.get("DY36_COMMIT", ""), "tau": 0.1, "same_speaker_decoys": N_SAME}
    (out / "metadata.json").write_text(json.dumps(meta, indent=1))
    log("PARAMS", meta["params"], json.dumps(meta))
    run = airon.init(project="dauys", name=a.name_run, config=meta, experiment=a.experiment, tags=["dy36", f"dict-{a.name}"],
                     total_steps=a.steps, run_id=meta["run_id"])
    rng = np.random.default_rng(a.seed)
    q: queue.Queue = queue.Queue(maxsize=8)
    stop = threading.Event()

    def producer():
        while not stop.is_set():
            q.put(train_batch(Dt, rng, a.batch))

    for _ in range(3):
        threading.Thread(target=producer, daemon=True).start()
    best, bad, hist, t0 = -1.0, 0, [], time.time()
    for st in range(a.steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(st, a.steps, a.lr, a.warmup)
        seqs, tg = q.get()
        leak_assert(tg)
        loss, lg = nce_loss(model, seqs, tg, Dt, dev)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        lg["grad_norm"] = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        opt.step()
        if (st + 1) % 50 == 0:
            lg["lr"] = opt.param_groups[0]["lr"]
            lg["steps_per_s"] = 50 / (time.time() - t0)
            t0 = time.time()
            run.log({f"train/{k}": v for k, v in lg.items()}, step=st + 1)
            log(json.dumps({"step": st + 1, **{k: round(v, 4) for k, v in lg.items()}}))
        if (st + 1) % a.val_every == 0:
            v = acc16(model, Dv, VA, dev)
            v.pop("hits")
            run.log({"val/acc16": v["acc16"], "val/acc16_lo95": v["ci95"][0]}, step=st + 1)
            hist.append({"step": st + 1, **v, "train_acc16": lg["acc16"], "train_loss": lg["loss"]})
            log("VAL", json.dumps(hist[-1]))
            (out / "val_history.json").write_text(json.dumps(hist, indent=1))
            ck = {"model": model.state_dict(), "cfg": vars(cfg), "step": st + 1, "val": v, "dictionary": a.name}
            if v["acc16"] > best:
                best, bad = v["acc16"], 0
                torch.save(ck, out / "best.pt")
            else:
                bad += 1
            if bad >= a.patience:
                log("EARLY STOP", st + 1, "best", best)
                break
    stop.set()
    run.finish()
    (out / "DONE").write_text(json.dumps({"best_acc16": best, "steps": hist[-1]["step"] if hist else 0}))
    log("TRAIN DONE", best)


def main() -> None:
    p = argparse.ArgumentParser()
    s = p.add_subparsers(dest="cmd", required=True)
    for n in ("check", "train"):
        x = s.add_parser(n)
        x.add_argument("--dict", required=True)
        x.add_argument("--name", required=True)
        x.add_argument("--out", required=True)
    s.choices["check"].add_argument("--known-steps", type=int, default=1500)
    t = s.choices["train"]
    t.add_argument("--experiment", default=EXPERIMENT)
    t.add_argument("--name-run", default="dy36-g")
    t.add_argument("--batch", type=int, default=64)
    t.add_argument("--lr", type=float, default=3e-4)
    t.add_argument("--warmup", type=int, default=200)
    t.add_argument("--steps", type=int, default=8000)
    t.add_argument("--val-every", type=int, default=200)
    t.add_argument("--patience", type=int, default=6)
    t.add_argument("--seed", type=int, default=1)
    a = p.parse_args()
    {"check": cmd_check, "train": cmd_train}[a.cmd](a)


if __name__ == "__main__":
    main()
