"""DY-36 checks after training (pre-registered thresholds; never changed).

Reply moments (200, by rule, from the held-out shards, before looking at the model): the next utterance u is by "me"
(its speaker), the previous utterance of the recording is by another speaker, u is 4-15 s, >= 10 utterances precede it;
episodes in id order, seeded order inside an episode (seed 1234), round-robin. Context audio = the last 40 s of the
piece before u (less if the piece starts later). Derangement j (seed 1234, j[i] != i).
Decision: the model output after the history (up to 256 utterances before u) is compared by cosine with the bank
(training utterances of 4-15 s with a valid thought = other recordings); one utterance is sampled among the 50 nearest
with weights exp(cos / 0.1) (seeds 0-4); reference variant: the single nearest.
C1: judge (FITS question, both orders), context text + choice for this moment vs choice for moment j[i] (seed 0).
    Pass >= 0.65. Reference rows (no threshold): real replies; ceiling (bank nearest to the true reply's thought); echo
    (bank nearest to the thought of the last heard utterance); single nearest.
C2: 5 choices per moment; mean over seeds of the C1 share >= 0.60 AND in >= half of the moments >= 3 of 5 choices differ.
C3: 1,200 DY-35 P1 anchors (same 15 decoys): acc16 by cosine for (a) true history, (b) own past utterances replaced by
    the utterances of a speaker of another held-out recording (in order), (c) heard utterances replaced by the
    utterances of another held-out recording (in order); (a) > (b) and (a) > (c) with paired 95% CI (bootstrap) > 0.
    Without thresholds: (a) with the last 1 / 4 / 16 / 64 / 256 utterances, split reply / continuation.
Listening: 8 of the 200 moments chosen on the context only (>= 25 Whisper words in the last 15 s; lowest words per
second, then highest mean corpus frequency of the words, then shortest mean word length); ids stored before choices.

    python scripts/dy36_eval.py --dict /data/dy36/dict --name A --ckpt /data/dy36/run/best.pt --out /data/dy36/eval
"""

import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from dy36_common import FITS, SR, Audio, Judge, TextCache, log, utt_job, wilson
from dy36_model import CTX, Cfg, ThoughtModel, history, last_outputs, make_seq
from dy36_train import c3_anchors, load

N, SEED, TOPK, TEMP, NSEED = 200, 1234, 50, 0.1, 5


def reply_moments(T, Dv, va_rows) -> list[int]:
    rng = np.random.default_rng(SEED)
    ok = []
    for i in range(Dv.n):
        e = Dv.ep_of[i]
        k = i - Dv.eb[e]
        if k >= 10 and Dv.spk[i - 1] != Dv.spk[i] and 200 <= Dv.len[i] <= 750 and Dv.valid[i]:
            ok.append(i)
    ok = np.array(ok)
    per = [rng.permutation(ok[Dv.ep_of[ok] == e]) for e in np.unique(Dv.ep_of[ok])]
    out, ptr = [], [0] * len(per)
    while len(out) < N and any(ptr[x] < len(per[x]) for x in range(len(per))):
        for x in range(len(per)):
            if len(out) < N and ptr[x] < len(per[x]):
                out.append(int(per[x][ptr[x]]))
                ptr[x] += 1
    return out


def derange(n: int, seed: int = SEED) -> np.ndarray:
    rng = np.random.default_rng(seed)
    while True:
        j = rng.permutation(n)
        if (j != np.arange(n)).all():
            return j


def choose(o: torch.Tensor, B: torch.Tensor, seed: int) -> np.ndarray:
    s = o @ B.T
    top = s.topk(TOPK, dim=1)
    if seed < 0:
        return top.indices[:, 0].cpu().numpy()
    g = torch.Generator(device=o.device).manual_seed(seed)
    w = torch.softmax(top.values / TEMP, 1)
    k = torch.multinomial(w, 1, generator=g)[:, 0]
    return top.indices.gather(1, k[:, None])[:, 0].cpu().numpy()


def boot_diff(a: np.ndarray, b: np.ndarray, n: int = 10000) -> dict:
    d = a.astype(float) - b.astype(float)
    rng = np.random.default_rng(0)
    bs = d[rng.integers(0, len(d), (n, len(d)))].mean(1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return {"diff": round(float(d.mean()), 4), "ci95": [round(float(lo), 4), round(float(hi), 4)], "excludes_zero": bool(lo > 0 or hi < 0)}


def mp3(path: Path, w: np.ndarray) -> None:
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "s16le", "-ar", str(SR), "-ac", "1", "-i", "-", "-b:a", "64k",
                    str(path)], input=w.astype(np.int16).tobytes(), check=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dict", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda")
    T, Dt, Dv, tr_rows, va_rows = load(a.dict, a.name)
    files = json.loads(str(T["files"]))
    audio = Audio(files)
    tc = TextCache(out / "texts.json")
    M = reply_moments(T, Dv, va_rows)
    R = va_rows[M]  # table rows of the real replies
    jj = derange(len(M))
    log("moments", len(M), "episodes", len(np.unique(Dv.ep_of[M])))
    # contexts and real replies -> texts; listening ids decided on the context only
    jobs = {}
    for i, r in enumerate(R):
        jobs[f"ctx40_{i}"] = utt_job(T, int(r), -2000, 0)
        jobs[f"ctx15_{i}"] = utt_job(T, int(r), -750, 0)
        jobs[f"u{int(r)}"] = utt_job(T, int(r))
    tc.fill(audio, jobs)
    lis_f = out / "listen_ids.json"
    if not lis_f.exists():
        words = [tc[f"ctx15_{i}"]["whisper"].lower().split() for i in range(len(M))]
        clean = [[w.strip(".,!?;:\"'()-") for w in ws] for ws in words]
        freq = Counter(w for ws in clean for w in ws if w)
        rows = []
        for i, ws in enumerate(clean):
            ws = [w for w in ws if w]
            if len(ws) >= 25:
                sec = tc[f"ctx15_{i}"]["sec"]
                rows.append((len(ws) / sec, -float(np.mean([freq[w] for w in ws])), float(np.mean([len(w) for w in ws])), i))
        rows.sort()
        lis = [r[3] for r in rows[:8]]
        lis_f.write_text(json.dumps({"rule": ">= 25 Whisper words in the last 15 s; lowest words/s, then highest mean "
                                     "corpus frequency (200 contexts), then shortest mean word length; decided before "
                                     "the model's choices", "eligible": len(rows), "ids": lis,
                                     "keys": [r[:3] for r in rows[:8]]}, indent=1))
    lis = json.loads(lis_f.read_text())["ids"]
    log("listening ids", lis)
    # model
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    model = ThoughtModel(Cfg(**ck["cfg"])).to(dev).eval()
    model.load_state_dict(ck["model"])
    bank = np.flatnonzero((Dt.len >= 200) & (Dt.len <= 750) & Dt.valid)
    bank_rows = tr_rows[bank]
    assert not T["held"][bank_rows].any() and not np.isin(T["ep"][bank_rows], T["ep"][R]).any()
    B = torch.from_numpy(Dt.thought[bank]).to(dev)
    seqs = [make_seq(Dv, history(Dv, u), int(Dv.spk[u])) for u in M]
    for s, u in zip(seqs, M):
        assert len(s["role"]) == len(history(Dv, u)) and u not in history(Dv, u)
    o = last_outputs(model, seqs, dev)
    ch = {f"s{k}": bank_rows[choose(o, B, k)] for k in range(NSEED)}
    ch["nearest"] = bank_rows[choose(o, B, -1)]
    ch["ceiling"] = bank_rows[choose(torch.from_numpy(Dv.thought[M]).to(dev), B, -1)]
    ch["echo"] = bank_rows[choose(torch.from_numpy(Dv.thought[np.array(M) - 1]).to(dev), B, -1)]
    ch["real"] = R
    cos_true = (o * torch.from_numpy(Dv.thought[M]).to(dev)).sum(1).cpu().numpy()
    tc.fill(audio, {f"u{int(r)}": utt_job(T, int(r)) for v in ch.values() for r in v})
    del model
    torch.cuda.empty_cache()
    j = Judge()
    res = {"rule": __doc__, "dictionary": a.name, "ckpt_step": ck["step"], "n": len(M), "moment_rows": R.tolist(),
           "derangement": jj.tolist(), "choices": {k: v.tolist() for k, v in ch.items()},
           "cos_model_to_true_reply_mean": float(cos_true.mean())}
    rows = {}
    for k in ["s0", "real", "ceiling", "echo", "nearest", "s1", "s2", "s3", "s4"]:
        for row in ("letters", "whisper"):
            ctx = [tc[f"ctx40_{i}"][row] for i in range(len(M))]
            x = [tc[f"u{int(ch[k][i])}"][row] for i in range(len(M))]
            y = [tc[f"u{int(ch[k][jj[i]])}"][row] for i in range(len(M))]
            rows[f"{k}/{row}"] = j.pairs(FITS, ctx, x, y)
            log("FIT", k, row, json.dumps({q: v for q, v in rows[f"{k}/{row}"].items() if q != "per_item"}))
    res["fit"] = rows
    del j
    torch.cuda.empty_cache()
    res["C1"] = {"D_letters": rows["s0/letters"]["D"], "ci95": rows["s0/letters"]["ci95"],
                 "D_whisper": rows["s0/whisper"]["D"], "pass": bool(rows["s0/letters"]["D"] >= 0.65),
                 "reference_rows_letters": {k: rows[f"{k}/letters"]["D"] for k in ("real", "ceiling", "echo", "nearest")},
                 "reference_rows_whisper": {k: rows[f"{k}/whisper"]["D"] for k in ("real", "ceiling", "echo", "nearest")}}
    S = np.stack([ch[f"s{k}"] for k in range(NSEED)], 1)
    distinct = np.array([len(set(r)) for r in S])
    fit_mean = float(np.mean([rows[f"s{k}/letters"]["D"] for k in range(NSEED)]))
    res["C2"] = {"mean_fit_letters": round(fit_mean, 4), "per_seed": [rows[f"s{k}/letters"]["D"] for k in range(NSEED)],
                 "mean_fit_whisper": round(float(np.mean([rows[f"s{k}/whisper"]["D"] for k in range(NSEED)])), 4),
                 "share_moments_ge3_distinct": float((distinct >= 3).mean()),
                 "distinct_hist": np.bincount(distinct, minlength=6)[1:].tolist(),
                 "pass": bool(fit_mean >= 0.60 and (distinct >= 3).mean() >= 0.5)}
    log("C1", json.dumps(res["C1"]))
    log("C2", json.dumps(res["C2"]))
    (out / "c1c2.json").write_text(json.dumps(res, indent=1))
    # listening set
    L = out / "listen"
    L.mkdir(exist_ok=True)
    t = np.arange(int(0.3 * SR)) / SR
    beep = (0.3 * 32767 * np.sin(2 * np.pi * 880 * t) * np.hanning(len(t))).astype(np.int16)
    tr = {}
    for i in lis:
        r = int(R[i])
        ctx15 = audio.many([utt_job(T, r, -750, 0)])[0]
        ctx5 = ctx15[-5 * SR :]
        real, mine = audio.many([utt_job(T, r), utt_job(T, int(ch["s0"][i]))])
        nn = f"{i:03d}"
        mp3(L / f"ctx__{nn}.mp3", ctx15)
        mp3(L / f"real__{nn}.mp3", np.concatenate([ctx5, np.zeros(SR // 5, np.int16), beep, np.zeros(SR // 5, np.int16), real]))
        mp3(L / f"model__{nn}.mp3", np.concatenate([ctx5, np.zeros(SR // 5, np.int16), beep, np.zeros(SR // 5, np.int16), mine]))
        tr[nn] = {"context_40s": tc[f"ctx40_{i}"], "context_15s": tc[f"ctx15_{i}"], "real_reply": tc[f"u{r}"],
                  "model_choice": tc[f"u{int(ch['s0'][i])}"],
                  "other_sampled_choices": [tc[f"u{int(ch[f's{k}'][i])}"] for k in range(1, NSEED)],
                  "fit_p_model_vs_other_moment_letters": rows["s0/letters"]["per_item"].get(i),
                  "table_row_real": r, "table_row_choice": int(ch["s0"][i]), "shard": files[int(T["fi"][r])]}
    (L / "transcripts.json").write_text(json.dumps(tr, indent=1))
    log("C1C2 DONE")


def c3(a) -> None:
    dev = torch.device("cuda")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    T, Dt, Dv, tr_rows, va_rows = load(a.dict, a.name)
    A = c3_anchors(T, va_rows)
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    model = ThoughtModel(Cfg(**ck["cfg"])).to(dev).eval()
    model.load_state_dict(ck["model"])
    rng = np.random.default_rng(2026)
    C = torch.from_numpy(np.stack([Dv.thought[np.r_[u, d]] for u, d in A])).to(dev)

    def acc(seqs):
        o = last_outputs(model, seqs, dev)
        s = torch.einsum("nd,nkd->nk", o, C)
        return (s[:, 0] > s[:, 1:].max(1).values).cpu().numpy()

    big = [g for g in Dv.groups if len(g) >= 20]
    sa, sb, sc = [], [], []
    for u, _ in A:
        h = history(Dv, u)
        me = int(Dv.spk[u])
        sa.append(make_seq(Dv, h, me))
        mine = np.flatnonzero(Dv.spk[h] == me)
        heard = np.flatnonzero(Dv.spk[h] != me)
        # (b) own utterances <- a speaker of another recording, in order
        while True:
            g = big[rng.integers(len(big))]
            if Dv.ep[g[0]] != Dv.ep[u]:
                break
        st = rng.integers(0, max(1, len(g) - len(mine)))
        don = np.resize(g[st:], max(len(mine), 1))
        sb.append(make_seq(Dv, h, me, {"raw": {int(p): Dv.raw[d] for p, d in zip(mine, don)},
                                       "spk": {int(p): int(Dv.spk[d]) for p, d in zip(mine, don)}}))
        # (c) heard utterances <- consecutive utterances of another recording, in order
        while True:
            e = rng.integers(len(Dv.eb))
            if e != Dv.ep_of[u] and Dv.ee[e] - Dv.eb[e] >= 20:
                break
        idx = np.arange(Dv.eb[e], Dv.ee[e])
        st = rng.integers(0, max(1, len(idx) - len(heard)))
        don = np.resize(idx[st:], max(len(heard), 1))
        sc.append(make_seq(Dv, h, me, {"raw": {int(p): Dv.raw[d] for p, d in zip(heard, don)},
                                       "spk": {int(p): int(Dv.spk[d]) + 10**9 for p, d in zip(heard, don)}}))
    ha, hb, hc = acc(sa), acc(sb), acc(sc)
    res = {"n": len(A), "ckpt_step": ck["step"], "baselines": json.load(open("/data/p1_baselines.json")).get("acoustic_15s"),
           "a_true": {"acc16": float(ha.mean()), "ci95": wilson(int(ha.sum()), len(ha))},
           "b_own_replaced": {"acc16": float(hb.mean()), "ci95": wilson(int(hb.sum()), len(hb))},
           "c_heard_replaced": {"acc16": float(hc.mean()), "ci95": wilson(int(hc.sum()), len(hc))},
           "a_minus_b": boot_diff(ha, hb), "a_minus_c": boot_diff(ha, hc)}
    res["pass"] = bool(res["a_minus_b"]["diff"] > 0 and res["a_minus_b"]["excludes_zero"] and res["a_minus_c"]["diff"] > 0
                       and res["a_minus_c"]["excludes_zero"])
    prev_other = np.array([len(history(Dv, u)) > 0 and Dv.spk[u - 1] != Dv.spk[u] for u, _ in A])
    prev_same = np.array([len(history(Dv, u)) > 0 and Dv.spk[u - 1] == Dv.spk[u] for u, _ in A])
    curve = {}
    for n in (1, 4, 16, 64, 256):
        hn = acc([make_seq(Dv, history(Dv, u, n), int(Dv.spk[u])) for u, _ in A])
        curve[str(n)] = {"all": round(float(hn.mean()), 4),
                         "reply": {"acc16": round(float(hn[prev_other].mean()), 4), "n": int(prev_other.sum())},
                         "continuation": {"acc16": round(float(hn[prev_same].mean()), 4), "n": int(prev_same.sum())}}
    res["context_curve"] = curve
    res["a_true_by_kind"] = {"reply": float(ha[prev_other].mean()), "continuation": float(ha[prev_same].mean())}
    res["history_len_mean"] = float(np.mean([len(s["role"]) for s in sa]))
    (out / "c3.json").write_text(json.dumps(res, indent=1))
    log("C3", json.dumps(res))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "c3":
        sys.argv.pop(1)
        p = argparse.ArgumentParser()
        for k in ("--dict", "--name", "--ckpt", "--out"):
            p.add_argument(k, required=True)
        c3(p.parse_args())
    else:
        main()
