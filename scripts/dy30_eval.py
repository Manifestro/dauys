"""DY-30 evaluation: does a phoneme LM continue speech with meaning? Blind judge on real vs model continuations.

Prompts: held-out pieces (shards never trained on) with >= 45 s of audio, picked with seed 1234; prompt = audio
0-30 s, real continuation = 30-45 s; the oracle (phon_units.Phonemes, as for the training data) runs on each part
separately. ``gen``: the LM continues EOS + prompt tokens for exactly len(real) tokens (temperature 0.8, top-p 0.95,
EOS banned, no repetition penalty; reference: the same at temperature 1.0). ``floor``: as many phonemes drawn from
the unigram distribution of the validation tokens.

Path to text, the same for prompt, gen, real and floor: phoneme ids -> Piper (one fixed en_US voice; the voice's
ONNX model is fed phoneme ids directly: every character of an oracle token, NFD-normalised, through the voice's
phoneme_id_map, with Piper's BOS / pad-after-each-symbol / EOS; no word boundaries are added) -> 16 kHz -> Whisper
large-v3-turbo -> text. Reference: Whisper on the original audio of the prompt and of the continuation.

Judge: Qwen2.5-7B-Instruct (fp16, a ruler only) sees the prompt text and two continuations A and B and is asked
which one was said next; the answer is read from the next-token logits of "A" vs "B", both orders; a pair is right
when the mean probability of the real one is > 0.5 (ties 0.5). D = share of right pairs, 95 % binomial CI.

    python scripts/dy30_eval.py gates --shard la-00030 --tokens la-00030.npz --dy22 dy22_eval.json \
        --dy23 dy23_eval.json --out /data/dy30/gates
    python scripts/dy30_eval.py eval --units /opt/dlami/nvme/phon --run /data/dy30/run --gates gates.json --out /data/dy30/eval
"""

import argparse
import json
import math
import subprocess
import sys
import tempfile
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

SR = 16000
SRC = "s3://manifestro-dataset-876108973178-us-east-1-an/outputs/duplexchat/en-16k"
ORACLE = "facebook/wav2vec2-lv-60-espeak-cv-ft"
PIPER_REPO = "rhasspy/piper-voices"
VOICE = "en/en_US/lessac/medium/en_US-lessac-medium"
ASR = "openai/whisper-large-v3-turbo"
JUDGE = "Qwen/Qwen2.5-7B-Instruct"
EOS = 8192
PROMPT_S, CONT_S, MIN_S, FAR_GAP_S = 30, 15, 45, 60
N_EVAL, N_GATE, N_SHOW = 200, 50, 20
SEED, TEMP, TOP_P, TEMP_REF = 1234, 0.8, 0.95, 1.0
P_MAX_WER, K_MIN, FLOOR_MIN = 0.35, 0.70, 0.95
TEMPLATE = (
    "Below is the beginning of a transcribed conversation, followed by two candidate continuations, A and B. "
    "One of them is what was actually said next; the other is not.\n\n"
    "Beginning:\n{prompt}\n\nContinuation A:\n{a}\n\nContinuation B:\n{b}\n\n"
    "Which continuation was actually said next? Answer with a single letter, A or B."
)


# ---------------------------------------------------------------- pure helpers (numpy only)

def norm_text(t: str) -> str:
    t = "".join(c if c.isalnum() or c == "'" else " " for c in t.lower())
    return " ".join(t.split())


def rep4(u: np.ndarray) -> float:
    g = [tuple(u[i : i + 4]) for i in range(len(u) - 3)]
    return 1 - len(set(g)) / len(g) if g else 0.0


def other_index(n: int, seed: int = 7) -> np.ndarray:
    """For every item another item (never itself), as in the dy23 topic test."""
    if n < 2:
        return np.zeros(n, np.int64)
    r = np.random.default_rng(seed)
    return np.array([(i + r.integers(1, n)) % n for i in range(n)], np.int64)


def d_stats(p_real: np.ndarray) -> dict:
    p = np.asarray(p_real, np.float64)
    right = np.where(p > 0.5, 1.0, np.where(p == 0.5, 0.5, 0.0))
    d = float(right.mean()) if len(p) else float("nan")
    return {"D": round(d, 4), "ci95": round(1.96 * math.sqrt(d * (1 - d) / max(len(p), 1)), 4), "n": int(len(p)),
            "mean_p_real": round(float(p.mean()), 4) if len(p) else float("nan")}


def verdict(d22: float, dp: float) -> dict:
    m_thr, none_thr = 0.5 + 0.5 * (d22 - 0.5), 0.5 + 0.8 * (d22 - 0.5)
    v = "meaning (M passed)" if dp <= m_thr else "no meaning" if dp >= none_thr else "partial"
    return {"D_dy22": d22, "D_phon": dp, "K": bool(d22 >= K_MIN), "M_threshold": round(m_thr, 4),
            "no_meaning_threshold": round(none_thr, 4), "M": bool(dp <= m_thr), "no_meaning": bool(dp >= none_thr),
            "verdict": v if d22 >= K_MIN else f"invalid: K failed ({v})"}


def phoneme_map(vocab: dict[str, int], id_map: dict[str, list[int]]) -> tuple[dict[int, list[int]], dict]:
    """Oracle id -> Piper ids (each NFD character of the token through phoneme_id_map, followed by the pad symbol,
    as piper's phonemes_to_ids does); special tokens (<...>) map to nothing. Also a report of unmapped symbols."""
    pad = id_map["_"]
    out, bad_tokens, bad_chars, special = {}, [], {}, []
    for tok, i in vocab.items():
        if tok.startswith("<") and tok.endswith(">"):
            special.append(tok)
            out[i] = []
            continue
        ids, miss = [], []
        for ch in unicodedata.normalize("NFD", tok):
            if ch in id_map:
                ids += list(id_map[ch]) + list(pad)
            else:
                miss.append(ch)
                bad_chars[ch] = bad_chars.get(ch, 0) + 1
        out[i] = ids
        if miss:
            bad_tokens.append({"token": tok, "id": i, "unmapped": "".join(miss), "lost_all": not ids})
    rep = {"vocab": len(vocab), "special": special, "tokens_with_unmapped_chars": len(bad_tokens),
           "tokens_lost_entirely": sum(b["lost_all"] for b in bad_tokens), "unmapped_chars": bad_chars,
           "unmapped_tokens": bad_tokens}
    return out, rep


def wer(ref: str, hyp: str) -> float:
    import jiwer

    r, h = norm_text(ref), norm_text(hyp)
    if not r:
        return float("nan")
    return float(jiwer.wer(r, h)) if h else 1.0


def acc_higher(a: list[float], b: list[float]) -> float:
    x, y = np.asarray(a), np.asarray(b)
    return float(np.mean(np.where(x > y, 1.0, np.where(x == y, 0.5, 0.0)))) if len(x) else float("nan")


# ---------------------------------------------------------------- audio from S3

def sh(cmd: str) -> str:
    return subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True).stdout


def candidates(shards: list[str], min_s: float, tmp: Path) -> list[tuple[str, int, int, int]]:
    """(shard, piece number, start, length) of every piece >= min_s in these shards, shards in the given order."""
    out = []
    for s in shards:
        f = tmp / f"{s}_index.npy"
        if not f.exists():
            sh(f"aws s3 cp {SRC}/{s}/index.npy {f} --only-show-errors")
        for p, (st, ln) in enumerate(np.load(f)):
            if ln >= min_s * SR:
                out.append((s, p, int(st), int(ln)))
    return out


def read_pieces(sel: list[tuple[str, int, int, int]], tmp: Path) -> list[np.ndarray]:
    """int16 audio of each selected piece by byte-range reads of segs.bin (no whole-shard download)."""
    bucket, prefix = SRC[5:].split("/", 1)

    def one(j_item):
        j, (s, _, st, ln) = j_item
        f = tmp / f"piece_{j}.bin"
        sh(f"aws s3api get-object --bucket {bucket} --key {prefix}/{s}/segs.bin --range bytes={2 * st}-{2 * (st + ln) - 1} {f}")
        w = np.fromfile(f, dtype=np.int16)
        f.unlink()
        assert len(w) == ln, (s, st, ln, len(w))
        return w

    with ThreadPoolExecutor(16) as ex:
        return list(ex.map(one, enumerate(sel)))


def parts(w: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    a, b = PROMPT_S * SR, (PROMPT_S + CONT_S) * SR
    f0 = (PROMPT_S + FAR_GAP_S) * SR
    far = w[f0 : f0 + CONT_S * SR] if len(w) >= f0 + CONT_S * SR else None
    return w[:a], w[a:b], far


# ---------------------------------------------------------------- path: phonemes -> Piper -> Whisper -> text

class Speech:
    def __init__(self, device, whisper: bool = True) -> None:
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download

        onnx = hf_hub_download(PIPER_REPO, f"{VOICE}.onnx")
        self.cfg = json.loads(Path(hf_hub_download(PIPER_REPO, f"{VOICE}.onnx.json")).read_text())
        self.sess = ort.InferenceSession(onnx, providers=["CPUExecutionProvider"])
        self.inputs = [i.name for i in self.sess.get_inputs()]
        vocab = json.loads(Path(hf_hub_download(ORACLE, "vocab.json")).read_text())
        id_map = self.cfg["phoneme_id_map"]
        self.map, self.report = phoneme_map(vocab, id_map)
        self.bos, self.eos = list(id_map["^"]), list(id_map["$"])
        self.sr = int(self.cfg["audio"]["sample_rate"])
        inf = self.cfg.get("inference", {})
        self.scales = np.array([inf.get("noise_scale", 0.667), inf.get("length_scale", 1.0), inf.get("noise_w", 0.8)], np.float32)
        self.lost = {i for i, ids in self.map.items() if not ids}
        self.asr = None
        if whisper:
            import torch
            from transformers import pipeline

            self.asr = pipeline("automatic-speech-recognition", model=ASR, device=device, torch_dtype=torch.float16,
                                chunk_length_s=30)

    def ids(self, toks: np.ndarray) -> list[int]:
        body = [x for t in toks for x in self.map.get(int(t), [])]
        return self.bos + body + self.eos if body else []

    def wav(self, toks: np.ndarray) -> np.ndarray:
        """16 kHz float32 Piper audio of a phoneme sequence (empty if nothing maps)."""
        from scipy.signal import resample_poly

        ids = self.ids(toks)
        if not ids:
            return np.zeros(0, np.float32)
        feed = {"input": np.asarray([ids], np.int64), "input_lengths": np.asarray([len(ids)], np.int64), "scales": self.scales}
        if "sid" in self.inputs:
            feed["sid"] = np.zeros(1, np.int64)
        a = np.asarray(self.sess.run(None, feed)[0], np.float32).reshape(-1)
        return resample_poly(a, SR, self.sr).astype(np.float32)

    def texts(self, waves: list[np.ndarray]) -> list[str]:
        out = [""] * len(waves)
        idx = [i for i, w in enumerate(waves) if len(w) >= SR // 10]
        if idx:
            res = self.asr([{"raw": waves[i], "sampling_rate": SR} for i in idx], batch_size=16,
                           generate_kwargs={"language": "en", "task": "transcribe"})
            for i, r in zip(idx, res):
                out[i] = r["text"].strip()
        return out

    def path(self, seqs: list[np.ndarray], workers: int = 8) -> tuple[list[str], list[np.ndarray]]:
        with ThreadPoolExecutor(workers) as ex:
            waves = list(ex.map(self.wav, seqs))
        return self.texts(waves), waves

    def original(self, audio: list[np.ndarray]) -> list[str]:
        return self.texts([w.astype(np.float32) / 32768.0 for w in audio])

    def lost_share(self, seqs: list[np.ndarray]) -> float:
        """Share of token occurrences whose oracle symbol has no Piper mapping at all (dropped from the speech)."""
        n = sum(len(s) for s in seqs)
        return float(sum(int(t) in self.lost for s in seqs for t in s) / max(n, 1))


# ---------------------------------------------------------------- judge

class Judge:
    def __init__(self, device) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch, self.device = torch, device
        self.tok = AutoTokenizer.from_pretrained(JUDGE)
        self.tok.padding_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(JUDGE, torch_dtype=torch.float16).to(device).eval()
        ab = [self.tok.encode(x, add_special_tokens=False) for x in ("A", "B")]
        assert all(len(x) == 1 for x in ab), ab
        self.ab = [ab[0][0], ab[1][0]]

    def p_first(self, prompts: list[str], first: list[str], second: list[str], batch: int = 8) -> np.ndarray:
        """P(answer "A") renormalised over A and B, where A = first, B = second."""
        torch = self.torch
        texts = [self.tok.apply_chat_template([{"role": "user", "content": TEMPLATE.format(prompt=p, a=a, b=b)}],
                                              tokenize=False, add_generation_prompt=True)
                 for p, a, b in zip(prompts, first, second)]
        out = []
        with torch.no_grad():
            for i in range(0, len(texts), batch):
                enc = self.tok(texts[i : i + batch], return_tensors="pt", padding=True).to(self.device)
                lg = self.model(**enc).logits[:, -1, self.ab].float()
                if not torch.isfinite(lg).all():
                    raise RuntimeError("judge logits not finite (fp16 overflow?)")
                out.append(torch.softmax(lg, -1)[:, 0].cpu().numpy())
        return np.concatenate(out) if out else np.zeros(0)

    def pairs(self, prompts: list[str], reals: list[str], others: list[str]) -> dict:
        """D for (prompt, real, other) triples: both orders, mean probability of the real one."""
        pa = self.p_first(prompts, reals, others)
        pb = 1.0 - self.p_first(prompts, others, reals)
        p = (pa + pb) / 2
        return {**d_stats(p), "p_real": [round(float(x), 4) for x in p],
                "p_real_shown_first": round(float(pa.mean()), 4), "p_real_shown_second": round(float(pb.mean()), 4)}


# ---------------------------------------------------------------- LM

def load_lm(run: Path, device):
    import torch

    from dauys.dy21_lm import LMConfig, UnitLM

    ck = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
    m = UnitLM(LMConfig(**ck["cfg"])).to(device).eval()
    m.load_state_dict(ck["model"])
    return m, {"step": ck.get("step"), "val": ck.get("val")}


def generate(model, prompt: np.ndarray, n: int, temp: float, seed: int, device) -> np.ndarray:
    import torch

    from dauys.dy16_model import sample_next

    g = torch.Generator(device=device).manual_seed(seed)
    ids = torch.from_numpy(np.r_[EOS, prompt].astype(np.int64))[None].to(device)
    with torch.no_grad():
        for _ in range(n):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                h = model.states(ids[:, -model.cfg.ctx :])
            lg = model.logits(h[:, -1])
            lg[:, EOS] = -float("inf")
            ids = torch.cat([ids, sample_next(lg, temp, TOP_P, g)[:, None]], 1)
    return ids[0, ids.shape[1] - n :].cpu().numpy() if n else np.zeros(0, np.int64)


def logprob(model, ctx: np.ndarray, cont: np.ndarray, device) -> float:
    """Total log p(cont | ctx) (bench_slm.logprob without the unit filter)."""
    import torch

    if len(cont) == 0:
        return 0.0
    ids = torch.from_numpy(np.concatenate([ctx, cont]).astype(np.int64))[None].to(device)
    ids = ids[:, -(model.cfg.ctx + 1) :]
    with torch.no_grad():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h = model.states(ids[:, :-1])
        lp = torch.log_softmax(model.logits(h), -1)[0]
    tgt = ids[0, 1:]
    t = lp.gather(1, tgt[:, None])[:, 0]
    return float(t[-min(len(cont), len(tgt)) :].sum())


# ---------------------------------------------------------------- text rulers (reference only)

def text_refs(device, prompt_text: list[str], kinds: dict[str, list[str]]) -> dict:
    import torch
    import torch.nn.functional as F
    from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

    tk = AutoTokenizer.from_pretrained("gpt2-medium")
    lm = AutoModelForCausalLM.from_pretrained("gpt2-medium").to(device).eval()

    @torch.no_grad()
    def logppl(t: str) -> float:
        ids = tk(t, return_tensors="pt").input_ids[:, :1024].to(device)
        return float("nan") if ids.shape[1] < 3 else float(lm(ids, labels=ids).loss)

    lp = {k: [logppl(t) for t in v] for k, v in kinds.items()}
    del lm
    etk = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    emb = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2").to(device).eval()

    @torch.no_grad()
    def vec(texts):
        b = etk([t or "." for t in texts], padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
        h = emb(**b).last_hidden_state
        m = b["attention_mask"].unsqueeze(-1).float()
        return F.normalize((h * m).sum(1) / m.sum(1).clamp(min=1), dim=-1)

    pv = vec(prompt_text)
    oi = torch.from_numpy(other_index(len(prompt_text))).to(device)
    summ = {}
    for k, v in kinds.items():
        cv = vec(v)
        own, oth = (cv * pv).sum(-1), (cv * pv[oi]).sum(-1)
        summ[k] = {"median_logppl": float(np.nanmedian(lp[k])) if np.isfinite(lp[k]).any() else float("nan"),
                   "no_text_share": float(np.mean([not np.isfinite(x) for x in lp[k]])),
                   "words_median": float(np.median([len(t.split()) for t in v])),
                   "topic_acc": float(((own > oth).float() + 0.5 * (own == oth).float()).mean()),
                   "sim_own": float(own.mean()), "sim_other": float(oth.mean())}
    if all(k in summ for k in ("gen", "real", "floor")):
        g, r, f = (summ[k]["median_logppl"] for k in ("gen", "real", "floor"))
        summ["gap_closed"] = (f - g) / (f - r) if f != r else float("nan")
    return {"summary": summ, "logppl": lp}


# ---------------------------------------------------------------- stages

def cleanup() -> None:
    """Return GPU memory after the caller dropped its references (del x; cleanup())."""
    import gc

    import torch

    gc.collect()
    torch.cuda.empty_cache()


def oracle_parts(audio: list[np.ndarray], device):
    from phon_units import Phonemes

    ph = Phonemes(device)
    P, R, Fr = [], [], []
    for w in audio:
        a, b, c = parts(w)
        P.append(ph(a))
        R.append(ph(b))
        Fr.append(ph(c) if c is not None else None)
    del ph
    cleanup()
    return P, R, Fr


def cmd_gates(a) -> None:
    """Cheap gates before the big data pass: path check P, judge calibration K (DY-22) and known answers."""
    import torch

    from dy21_train import held_out

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
    P, R, _ = oracle_parts(audio, device)
    tok = np.load(a.tokens)["tok"].astype(np.int64)
    counts = np.bincount(tok).astype(np.float64)
    floor = [rng.choice(len(counts), size=len(r), p=counts / counts.sum()) for r in R]
    t0 = time.time()
    sp = Speech(device)
    prompt_t, _ = sp.path(P)
    real_t, _ = sp.path(R)
    floor_t, _ = sp.path(floor)
    real_o = sp.original([parts(w)[1] for w in audio])
    lost = sp.lost_share(P + R)
    report = sp.report
    del sp
    cleanup()
    w = [wer(o, t) for o, t in zip(real_o, real_t)]
    med = float(np.nanmedian(w)) if np.isfinite(w).any() else float("nan")
    print("PATH", json.dumps({"median_wer": med, "pieces": len(w), "seconds": round(time.time() - t0)}), flush=True)

    j = Judge(device)
    d22 = json.loads(Path(a.dy22).read_text())["rows"]
    d23 = json.loads(Path(a.dy23).read_text())["prompt_text"]
    n = min(len(d22["gen"]), len(d22["real"]), len(d23))
    cal = j.pairs(d23[:n], [x["text"] for x in d22["real"][:n]], [x["text"] for x in d22["gen"][:n]])
    k_floor = j.pairs(prompt_t, real_t, floor_t)
    oi = other_index(len(real_t))
    k_other = j.pairs(prompt_t, real_t, [real_t[i] for i in oi])
    del j
    cleanup()
    res = {
        "shard": a.shard, "pieces": [list(x) for x in sel],
        "P": {"median_wer": round(med, 4), "pass": bool(med <= P_MAX_WER), "threshold": P_MAX_WER,
              "wer": [None if not np.isfinite(x) else round(x, 4) for x in w]},
        "K": {**{k: v for k, v in cal.items() if k != "p_real"}, "pass": bool(cal["D"] >= K_MIN), "threshold": K_MIN},
        "known_floor": {**{k: v for k, v in k_floor.items() if k != "p_real"}, "pass": bool(k_floor["D"] >= FLOOR_MIN)},
        "known_other_real": {k: v for k, v in k_other.items() if k != "p_real"},
        "piper": {k: v for k, v in report.items() if k != "unmapped_tokens"} | {"lost_token_share": round(lost, 4),
                                                                              "unmapped_tokens": report["unmapped_tokens"]},
        "texts": [{"prompt": p, "real": r, "real_original": o, "floor": f} for p, r, o, f in zip(prompt_t, real_t, real_o, floor_t)],
        "calibration_p_real": cal["p_real"],
    }
    res["pass"] = bool(res["P"]["pass"] and res["K"]["pass"] and res["known_floor"]["pass"])
    (out / "gates.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
    for g in ("P", "K", "known_floor"):
        print("GATE", g, json.dumps({k: v for k, v in res[g].items() if k != "wer"}), flush=True)
    print("KNOWN other_real", json.dumps(res["known_other_real"]), flush=True)
    print("PIPER", json.dumps({k: v for k, v in res["piper"].items() if k != "unmapped_tokens"}, ensure_ascii=False), flush=True)
    if not res["pass"]:
        why = [g for g in ("P", "K", "known_floor") if not res[g]["pass"]]
        print("GATES FAILED:", ", ".join(why), flush=True)
        sys.exit(1)
    print("GATES PASS", flush=True)


def cmd_eval(a) -> None:
    import soundfile as sf
    import torch

    from dy21_train import held_out, load_stream

    device = torch.device("cuda")
    out, run = Path(a.out), Path(a.run)
    (out / "listen").mkdir(parents=True, exist_ok=True)
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
    P, R, Fr = oracle_parts(audio, device)

    val = load_stream([Path(a.units) / f"{s}.npz" for s in val_names])
    counts = np.bincount(val[val != EOS].astype(np.int64)).astype(np.float64)
    floor = [rng.choice(len(counts), size=len(r), p=counts / counts.sum()) for r in R]
    oi = other_index(n)

    model, best = load_lm(run, device)
    t0 = time.time()
    gen = [generate(model, P[i], len(R[i]), TEMP, SEED + i, device) for i in range(n)]
    gen1 = [generate(model, P[i], len(R[i]), TEMP_REF, SEED + i, device) for i in range(n)]
    print(json.dumps({"generated": n, "seconds": round(time.time() - t0)}), flush=True)
    lp = {"true": [], "other": [], "true_n": [], "other_n": [], "far_true": [], "far": [], "far_true_n": [], "far_n": []}
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

    sp = Speech(device)
    prompt_t, _ = sp.path(P)
    real_t, _ = sp.path(R)
    gen_t, gen_w = sp.path(gen)
    gen1_t, _ = sp.path(gen1)
    floor_t, _ = sp.path(floor)
    prompt_o = sp.original([parts(w)[0] for w in audio])
    real_o = sp.original([parts(w)[1] for w in audio])
    piper = {k: v for k, v in sp.report.items() if k != "unmapped_tokens"}
    piper["lost_token_share"] = {"prompt_real": round(sp.lost_share(P + R), 4), "gen": round(sp.lost_share(gen), 4)}
    del sp
    cleanup()
    w = [wer(o, t) for o, t in zip(real_o, real_t)]

    j = Judge(device)
    D = {"phon": j.pairs(prompt_t, real_t, gen_t),
         "phon_t1": j.pairs(prompt_t, real_t, gen1_t),
         "phon_prompt_original": j.pairs(prompt_o, real_t, gen_t),
         "known_floor": j.pairs(prompt_t, real_t, floor_t),
         "known_other_real": j.pairs(prompt_t, real_t, [real_t[i] for i in oi])}
    del j
    cleanup()
    for k, v in D.items():
        print("JUDGE", k, json.dumps({x: y for x, y in v.items() if x != "p_real"}), flush=True)

    refs = text_refs(device, prompt_t, {"gen": gen_t, "real": real_t, "floor": floor_t, "gen_t1": gen1_t})
    refs["summary"]["rep4_tokens"] = {k: float(np.mean([rep4(x) for x in v])) for k, v in
                                      (("gen", gen), ("real", R), ("floor", floor), ("gen_t1", gen1))}
    gates = json.loads(Path(a.gates).read_text()) if a.gates else {}
    d22 = gates.get("K", {}).get("D", float("nan"))
    v = verdict(d22, D["phon"]["D"])
    summ = {"n": n, "best": best, "verdict": v, "judge": {k: {x: y for x, y in d.items() if x != "p_real"} for k, d in D.items()},
            "textless": textless, "path_wer_median_200": float(np.nanmedian(w)) if np.isfinite(w).any() else float("nan"),
            "piper": piper, "refs": refs["summary"],
            "tokens_median": {"prompt": float(np.median([len(x) for x in P])), "real": float(np.median([len(x) for x in R]))}}
    rows = [{"shard": s, "piece": p, "start": st, "len_s": round(ln / SR, 1), "prompt_tokens": len(P[i]), "real_tokens": len(R[i]),
             "prompt_text": prompt_t[i], "prompt_original": prompt_o[i], "real_text": real_t[i], "real_original": real_o[i],
             "gen_text": gen_t[i], "gen_t1_text": gen1_t[i], "floor_text": floor_t[i], "wer_real_path": w[i],
             "p_real": {k: d["p_real"][i] for k, d in D.items()},
             "gen_tokens": gen[i].tolist(), "real_tokens_ids": R[i].tolist()}
            for i, (s, p, st, ln) in enumerate(sel)]
    (out / "eval.json").write_text(json.dumps({"summary": summ, "rows": rows}, indent=2, ensure_ascii=False))

    lines = ["# DY-30: 20 continuations (all texts through the same path: phonemes -> Piper -> Whisper)", ""]
    for i in range(min(N_SHOW, n)):
        s, p, st, ln = sel[i]
        lines += [f"## {i:02d} — {s}, piece {p}", "", f"**Prompt (0–30 s):** {prompt_t[i]}", "",
                  f"**Real continuation (30–45 s):** {real_t[i]}", "", f"**Model continuation:** {gen_t[i]}", "",
                  f"*Whisper on the original audio — prompt:* {prompt_o[i]}", "", f"*— continuation:* {real_o[i]}", "",
                  f"*Judge P(real) for this pair:* {D['phon']['p_real'][i]}", ""]
        tail = parts(audio[i])[0][-5 * SR :].astype(np.float32) / 32768.0
        g = gen_w[i]
        pk = lambda x: 0.9 * x / max(float(np.abs(x).max()), 1e-6) if len(x) else x
        sf.write(out / "listen" / f"{i:02d}_prompt5s_then_model.wav", np.concatenate([pk(tail), np.zeros(SR * 3 // 10, np.float32), pk(g)]), SR)
    (out / "pairs.md").write_text("\n".join(lines))
    print("VERDICT", json.dumps(v), flush=True)
    print("SUMMARY", json.dumps(summ), flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    s = p.add_subparsers(dest="cmd", required=True)
    x = s.add_parser("gates")
    x.add_argument("--shard", required=True, help="one held-out shard")
    x.add_argument("--tokens", required=True, help="its oracle tokens (.npz): unigram source for the floor")
    x.add_argument("--dy22", required=True, help="HF dy22/eval.json")
    x.add_argument("--dy23", required=True, help="HF dy23/eval.json (prompt_text)")
    x.add_argument("--n", type=int, default=N_GATE)
    x.add_argument("--out", required=True)
    x = s.add_parser("eval")
    x.add_argument("--units", required=True)
    x.add_argument("--run", required=True)
    x.add_argument("--gates", default="", help="gates.json (for D(DY-22) in the gate arithmetic)")
    x.add_argument("--n", type=int, default=N_EVAL)
    x.add_argument("--out", required=True)
    a = p.parse_args()
    {"gates": cmd_gates, "eval": cmd_eval}[a.cmd](a)


if __name__ == "__main__":
    main()
