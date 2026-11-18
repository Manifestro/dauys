"""DY-31 evaluation: a letter LM continues held-out speech; a blind judge reads the continuation as plain text.

Prompts as DY-30: held-out pieces >= 45 s (seed 1234), prompt = audio 0-30 s, real = 30-45 s, the letters
recogniser (dy31_units.Letters) on each part separately. gen = EOS + prompt tokens continued for len(real) tokens
(temperature 0.8, top-p 0.95, EOS banned, seed 1234 + i; reference: temperature 1.0). floor = as many letters drawn
from the unigram distribution of the validation tokens. Every text comes straight from the tokens (dy31_units.Text);
there is no synthesis and no Whisper in the path (Whisper only rules the symbols in gate S).

Judge: dy30_eval.Judge (Qwen2.5-7B-Instruct, bfloat16, same template, both orders). pairs(prompts, X, Y) = D(X vs Y),
the share of pairs where X is picked as the real continuation.
  Dr = D(real vs gen);  Dc = D(gen vs real of another piece);  both also at temperature 1.0.
  Controls on real text: (a) D(real vs real of another piece); (b) D(real vs itself cut into 4-word chunks, chunks
  shuffled); (c) D(real vs floor).
Gates before the big pass (one held-out shard, 50 pieces): S = median WER of the letters text vs Whisper on the same
15 s <= 0.30; J = (a) >= 0.85, (b) >= 0.85, (c) >= 0.95. After training: M = Dc >= 0.70 and Dr <= 0.85;
"no meaning" = Dc <= 0.55; otherwise "partial".

    python scripts/dy31_eval.py gates --shard la-00030 --tokens la-00030.npz --out /data/dy31/gates
    python scripts/dy31_eval.py eval --units /opt/dlami/nvme/char-units --run /data/dy31/run --out /data/dy31/eval
"""

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from dy30_eval import (ASR, EOS, MIN_S, N_EVAL, N_GATE, N_SHOW, SEED, SR, TEMP, TEMP_REF, Judge, acc_higher, candidates,
                       cleanup, d_stats, generate, load_lm, logprob, other_index, parts, read_pieces, rep4, text_refs, wer)

S_MAX_WER, J_MIN_OTHER, J_MIN_SHUFFLE, J_MIN_FLOOR = 0.30, 0.85, 0.85, 0.95
M_MIN_DC, M_MAX_DR, NONE_MAX_DC = 0.70, 0.85, 0.55
CHUNK = 4


# ---------------------------------------------------------------- pure helpers

def shuffle_chunks(text: str, seed) -> tuple[str, bool]:
    """The text cut into 4-word chunks, chunks shuffled (fixed seed); if the shuffle keeps the order, rotate by one
    chunk. Returns the text and whether it differs from the original order (False when there is < 2 chunks)."""
    w = text.split()
    ch = [w[i : i + CHUNK] for i in range(0, len(w), CHUNK)]
    if len(ch) < 2:
        return text, False
    order = np.random.default_rng(seed).permutation(len(ch))
    if (order == np.arange(len(ch))).all():
        order = np.roll(order, 1)
    return " ".join(x for i in order for x in ch[i]), True


def verdict31(dc: float, dr: float) -> dict:
    m = bool(dc >= M_MIN_DC and dr <= M_MAX_DR)
    none = bool(dc <= NONE_MAX_DC)
    return {"Dc": dc, "Dr": dr, "M": m, "no_meaning": none, "rule": f"M: Dc >= {M_MIN_DC} and Dr <= {M_MAX_DR}; "
            f"no meaning: Dc <= {NONE_MAX_DC}", "verdict": "meaning (M passed)" if m else "no meaning" if none else "partial"}


def in_vocab_share(texts: list[str], vocab: set[str]) -> float:
    words = [w for t in texts for w in t.split()]
    return float(np.mean([w in vocab for w in words])) if words else float("nan")


def strip_p(d: dict) -> dict:
    return {k: v for k, v in d.items() if k != "p_real"}


# ---------------------------------------------------------------- recogniser / Whisper

def letters_parts(audio: list[np.ndarray], device):
    from dy31_units import Letters

    ph = Letters(device)
    P, R, Fr = [], [], []
    for w in audio:
        a, b, c = parts(w)
        P.append(ph(a))
        R.append(ph(b))
        Fr.append(ph(c) if c is not None else None)
    del ph
    cleanup()
    return P, R, Fr


def whisper_texts(audio16: list[np.ndarray], device) -> list[str]:
    import torch
    from transformers import pipeline

    asr = pipeline("automatic-speech-recognition", model=ASR, device=device, torch_dtype=torch.float16, chunk_length_s=30)
    res = asr([{"raw": w.astype(np.float32) / 32768.0, "sampling_rate": SR} for w in audio16], batch_size=16,
              generate_kwargs={"language": "en", "task": "transcribe"})
    out = [r["text"].strip() for r in res]
    del asr
    cleanup()
    return out


def controls(j: Judge, prompt_t: list[str], real_t: list[str], floor_t: list[str]) -> tuple[dict, list[str]]:
    oi = other_index(len(real_t))
    sh = [shuffle_chunks(t, [SEED, i]) for i, t in enumerate(real_t)]
    shuffled = [s for s, _ in sh]
    c = {"a_other_real": j.pairs(prompt_t, real_t, [real_t[i] for i in oi]),
         "b_shuffled_chunks": j.pairs(prompt_t, real_t, shuffled),
         "c_floor": j.pairs(prompt_t, real_t, floor_t)}
    c["b_shuffled_chunks"]["unchanged_items"] = int(sum(not d for _, d in sh))
    return c, shuffled


# ---------------------------------------------------------------- stages

def cmd_gates(a) -> None:
    import torch

    from dy21_train import held_out
    from dy31_units import Text

    device = torch.device("cuda")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(dir=out))
    assert held_out(a.shard), a.shard
    cands = candidates([a.shard], MIN_S, tmp)
    rng = np.random.default_rng(SEED)
    pick = rng.choice(len(cands), size=min(a.n, len(cands)), replace=False)
    sel = [cands[i] for i in pick]
    audio = read_pieces(sel, tmp)
    P, R, _ = letters_parts(audio, device)
    tok = np.load(a.tokens)["tok"].astype(np.int64)
    counts = np.bincount(tok).astype(np.float64)
    floor = [rng.choice(len(counts), size=len(r), p=counts / counts.sum()) for r in R]
    tx = Text()
    prompt_t, real_t, floor_t = [tx(x) for x in P], [tx(x) for x in R], [tx(x) for x in floor]
    real_o = whisper_texts([parts(w)[1] for w in audio], device)
    w = [wer(o, t) for o, t in zip(real_o, real_t)]
    med = float(np.nanmedian(w)) if np.isfinite(w).any() else float("nan")
    print("S", json.dumps({"median_wer": med, "pieces": len(w)}), flush=True)
    j = Judge(device)
    c, shuffled = controls(j, prompt_t, real_t, floor_t)
    del j
    cleanup()
    res = {"shard": a.shard, "pieces": [list(x) for x in sel], "text": tx.info,
           "S": {"median_wer": round(med, 4), "threshold": S_MAX_WER, "pass": bool(med <= S_MAX_WER),
                 "wer": [None if not np.isfinite(x) else round(x, 4) for x in w]},
           "J": {"a_other_real": {**strip_p(c["a_other_real"]), "threshold": J_MIN_OTHER, "pass": bool(c["a_other_real"]["D"] >= J_MIN_OTHER)},
                 "b_shuffled_chunks": {**strip_p(c["b_shuffled_chunks"]), "threshold": J_MIN_SHUFFLE,
                                       "pass": bool(c["b_shuffled_chunks"]["D"] >= J_MIN_SHUFFLE)},
                 "c_floor": {**strip_p(c["c_floor"]), "threshold": J_MIN_FLOOR, "pass": bool(c["c_floor"]["D"] >= J_MIN_FLOOR)}},
           "texts": [{"prompt": p, "real": r, "whisper_original": o, "shuffled": s, "floor": f}
                     for p, r, o, s, f in list(zip(prompt_t, real_t, real_o, shuffled, floor_t))[:10]],
           "p_real": {k: v["p_real"] for k, v in c.items()}}
    failed = ([] if res["S"]["pass"] else ["S"]) + [f"J.{k}" for k, v in res["J"].items() if not v["pass"]]
    res["pass"] = not failed
    (out / "gates.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print("GATE S", json.dumps({k: v for k, v in res["S"].items() if k != "wer"}), flush=True)
    for k, v in res["J"].items():
        print("GATE J", k, json.dumps(v), flush=True)
    if failed:
        print("GATES FAILED:", ", ".join(failed), flush=True)
        sys.exit(1)
    print("GATES PASS", flush=True)


def cmd_eval(a) -> None:
    import torch

    from dy21_train import held_out, load_stream
    from dy31_units import Text

    device = torch.device("cuda")
    out, run = Path(a.out), Path(a.run)
    out.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(dir=out))
    meta = json.loads((run / "metadata.json").read_text())
    val_names = meta["val_files_names"]
    if not all(held_out(s) for s in val_names):
        raise SystemExit(f"LEAK: validation files include training shards: {val_names}")
    cands = candidates(val_names, MIN_S, tmp)
    rng = np.random.default_rng(SEED)
    pick = rng.choice(len(cands), size=min(a.n, len(cands)), replace=False)
    sel = [cands[i] for i in pick]
    assert all(held_out(s) for s, *_ in sel)
    audio = read_pieces(sel, tmp)
    n = len(sel)
    print(json.dumps({"candidates": len(cands), "picked": n, "val_shards": len(val_names)}), flush=True)
    P, R, Fr = letters_parts(audio, device)
    del audio

    val = load_stream([Path(a.units) / f"{s}.npz" for s in val_names])
    counts = np.bincount(val[val != EOS].astype(np.int64)).astype(np.float64)
    floor = [rng.choice(len(counts), size=len(r), p=counts / counts.sum()) for r in R]
    tx = Text()
    vocab = set(tx(np.where(val == EOS, tx.delim_id, val)).split())
    oi = other_index(n)

    model, best = load_lm(run, device)
    t0 = time.time()
    gen = [generate(model, P[i], len(R[i]), TEMP, SEED + i, device) for i in range(n)]
    gen1 = [generate(model, P[i], len(R[i]), TEMP_REF, SEED + i, device) for i in range(n)]
    print(json.dumps({"generated": n, "seconds": round(time.time() - t0)}), flush=True)
    lp = {k: [] for k in ("true", "other", "true_n", "other_n", "far_true", "far", "far_true_n", "far_n")}
    for i in range(n):
        ctx = np.r_[EOS, P[i]]
        t, o = logprob(model, ctx, R[i], device), logprob(model, ctx, R[oi[i]], device)
        lp["true"].append(t); lp["other"].append(o)
        lp["true_n"].append(t / max(len(R[i]), 1)); lp["other_n"].append(o / max(len(R[oi[i]]), 1))
        if Fr[i] is not None:
            f = logprob(model, ctx, Fr[i], device)
            lp["far_true"].append(t); lp["far"].append(f)
            lp["far_true_n"].append(t / max(len(R[i]), 1)); lp["far_n"].append(f / max(len(Fr[i]), 1))
    textless = {"vs_other_piece": {"acc_total": acc_higher(lp["true"], lp["other"]), "acc_per_token": acc_higher(lp["true_n"], lp["other_n"]), "n": n},
                "vs_same_piece_60s_later": {"acc_total": acc_higher(lp["far_true"], lp["far"]),
                                            "acc_per_token": acc_higher(lp["far_true_n"], lp["far_n"]), "n": len(lp["far"])}}
    print("TEXTLESS", json.dumps(textless), flush=True)
    del model
    cleanup()

    prompt_t, real_t, gen_t, gen1_t, floor_t = ([tx(x) for x in X] for X in (P, R, gen, gen1, floor))
    other_t = [real_t[i] for i in oi]
    j = Judge(device)
    D = {"Dr": j.pairs(prompt_t, real_t, gen_t), "Dc": j.pairs(prompt_t, gen_t, other_t),
         "Dr_t1": j.pairs(prompt_t, real_t, gen1_t), "Dc_t1": j.pairs(prompt_t, gen1_t, other_t)}
    ctl, shuffled = controls(j, prompt_t, real_t, floor_t)
    del j
    cleanup()
    for k, v in list(D.items()) + list(ctl.items()):
        print("JUDGE", k, json.dumps(strip_p(v)), flush=True)

    refs = text_refs(device, prompt_t, {"gen": gen_t, "real": real_t, "floor": floor_t, "gen_t1": gen1_t})
    refs["summary"]["rep4_tokens"] = {k: float(np.mean([rep4(x) for x in v])) for k, v in
                                      (("gen", gen), ("real", R), ("floor", floor), ("gen_t1", gen1))}
    refs["summary"]["words_in_heldout_vocab"] = {"gen": in_vocab_share(gen_t, vocab), "real": in_vocab_share(real_t, vocab),
                                                 "gen_t1": in_vocab_share(gen1_t, vocab), "floor": in_vocab_share(floor_t, vocab),
                                                 "vocab_size": len(vocab)}
    v = verdict31(D["Dc"]["D"], D["Dr"]["D"])
    summ = {"n": n, "best": best, "verdict": v, "judge": {k: strip_p(d) for k, d in D.items()},
            "controls": {k: strip_p(d) for k, d in ctl.items()}, "textless": textless, "refs": refs["summary"],
            "text": tx.info, "tokens_median": {"prompt": float(np.median([len(x) for x in P])), "real": float(np.median([len(x) for x in R]))}}
    rows = [{"shard": s, "piece": p, "start": st, "len_s": round(ln / SR, 1), "prompt": prompt_t[i], "real": real_t[i],
             "gen": gen_t[i], "gen_t1": gen1_t[i], "floor": floor_t[i], "shuffled": shuffled[i], "other_real": other_t[i],
             "p": {k: d["p_real"][i] for k, d in list(D.items()) + list(ctl.items())}}
            for i, (s, p, st, ln) in enumerate(sel)]
    (out / "eval.json").write_text(json.dumps({"summary": summ, "rows": rows}, indent=2, ensure_ascii=False))

    lines = ["# DY-31: what the person said next vs what the model wrote", "",
             f"Dr {D['Dr']['D']} ± {D['Dr']['ci95']} (0.5 = indistinguishable from the real one); "
             f"Dc {D['Dc']['D']} ± {D['Dc']['ci95']} (0.5 = no more tied to the prompt than another piece). Verdict: {v['verdict']}.", ""]
    for i in range(min(N_SHOW, n)):
        lines += [f"## {i + 1}", "", f"**Prompt:** {prompt_t[i]}", "", f"**Person said next:** {real_t[i]}", "",
                  f"**Model wrote:** {gen_t[i]}", "",
                  f"_Judge: picked the person over the model {D['Dr']['p_real'][i]:.2f}; "
                  f"picked the model over another piece's real text {D['Dc']['p_real'][i]:.2f}_", ""]
    (out / "pairs.md").write_text("\n".join(lines))
    print("VERDICT", json.dumps(v), flush=True)
    print("SUMMARY", json.dumps(summ), flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    s = p.add_subparsers(dest="cmd", required=True)
    x = s.add_parser("gates")
    x.add_argument("--shard", required=True, help="one held-out shard")
    x.add_argument("--tokens", required=True, help="its letter tokens (.npz): unigram source for the floor")
    x.add_argument("--n", type=int, default=N_GATE)
    x.add_argument("--out", required=True)
    x = s.add_parser("eval")
    x.add_argument("--units", required=True)
    x.add_argument("--run", required=True)
    x.add_argument("--n", type=int, default=N_EVAL)
    x.add_argument("--out", required=True)
    a = p.parse_args()
    {"gates": cmd_gates, "eval": cmd_eval}[a.cmd](a)


if __name__ == "__main__":
    main()
