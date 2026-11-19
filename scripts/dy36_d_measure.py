"""DY-36 dictionaries, judge-free measure (text only measures finished audio; nothing here trains anything).

Utterances are recognised by Whisper large-v3-turbo. Text similarity of two utterances:
  words  Jaccard overlap of content-word sets (lower-cased [a-z']+ tokens minus the fixed sklearn English stop list);
  emb    cosine of sentence embeddings (sentence-transformers/all-MiniLM-L6-v2).
Random pairs: 3000 pairs of held-out utterances from different recordings (seed 7), valid in every dictionary;
  Spearman correlation between cosine(z_i, z_j) and each text similarity.
Neighbours: the 500 G0 references (dy36_g0.select: held-out refs, training bank = other recordings, one shared random
  candidate per reference); per dictionary the bank utterance with the highest thought cosine; mean text similarity
  reference<->neighbour and reference<->random with 95% CI (normal approximation) and of their paired difference.
Pairs with an empty transcript on either side are dropped (counts reported).

    python scripts/dy36_d_measure.py --dict /data/dy36/dict --dicts D,A,C2500 --out /data/dy36/d/measure
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from dy36_common import ASR, SR, Audio, log, utt_job
from dy36_dict import load_table
from dy36_g0 import neighbours, select


def whisper(wavs: list) -> list[str]:
    import torch
    from transformers import pipeline

    asr = pipeline("automatic-speech-recognition", model=ASR, device=torch.device("cuda"), torch_dtype=torch.float16,
                   chunk_length_s=30)
    out = [""] * len(wavs)
    idx = [i for i, w in enumerate(wavs) if len(w) >= SR // 10]
    res = asr([{"raw": wavs[i].astype(np.float32) / 32768.0, "sampling_rate": SR} for i in idx], batch_size=16,
              generate_kwargs={"language": "en", "task": "transcribe"})
    for i, r in zip(idx, res):
        out[i] = r["text"].strip()
    return out


def content_words(t: str) -> set:
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

    return {w for w in re.findall(r"[a-z']+", t.lower()) if w.strip("'") and w not in ENGLISH_STOP_WORDS}


def jacc(a: str, b: str) -> float:
    x, y = content_words(a), content_words(b)
    return len(x & y) / len(x | y) if x | y else 0.0


def ci(x: np.ndarray) -> list:
    m, h = float(x.mean()), 1.96 * float(x.std(ddof=1)) / np.sqrt(len(x))
    return [round(m, 4), round(m - h, 4), round(m + h, 4)]


def main() -> None:
    from scipy.stats import spearmanr

    p = argparse.ArgumentParser()
    p.add_argument("--dict", required=True)
    p.add_argument("--dicts", default="D,A,C2500")
    p.add_argument("--out", required=True)
    p.add_argument("--n-pairs", type=int, default=3000)
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    T = load_table(Path(a.dict))
    names = a.dicts.split(",")
    Dd = {n: np.load(Path(a.dict) / f"dict_{n}.npz") for n in names}
    Z = {n: (d["thought"], d["raw"], d["valid"]) for n, d in Dd.items()}
    valid_all = np.logical_and.reduce([Z[n][2] for n in names])
    held = np.flatnonzero(T["held"] & valid_all)
    rng = np.random.default_rng(7)
    pi, pj = [], []
    while len(pi) < a.n_pairs:
        i, j = rng.choice(held, 2, replace=False)
        if T["ep"][i] != T["ep"][j]:
            pi.append(int(i))
            pj.append(int(j))
    pi, pj = np.array(pi), np.array(pj)
    refs, bank, rand = select(T)
    nb = {n: neighbours(Z[n][0], Z[n][2], refs, bank) for n in names}
    need = set(pi.tolist()) | set(pj.tolist()) | set(refs.tolist()) | set(rand.tolist())
    for n in names:
        need |= set(int(u) for u in nb[n])
    tf = out / "texts.json"
    texts = json.loads(tf.read_text()) if tf.exists() else {}
    todo = sorted(u for u in need if str(u) not in texts)
    if todo:
        log("audio", len(todo))
        au = Audio(json.loads(str(T["files"])))
        wavs = au.many([utt_job(T, u) for u in todo])
        log("whisper", len(todo))
        for u, t in zip(todo, whisper(wavs)):
            texts[str(u)] = t
        tf.write_text(json.dumps(texts))
    from sentence_transformers import SentenceTransformer

    st = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cuda")
    keys = sorted(texts)
    E = st.encode([texts[k] for k in keys], batch_size=256, normalize_embeddings=True, convert_to_numpy=True)
    emb = {k: E[i] for i, k in enumerate(keys)}

    def sims(x: np.ndarray, y: np.ndarray):
        ok = np.array([bool(texts[str(u)].strip()) and bool(texts[str(v)].strip()) for u, v in zip(x, y)])
        w = np.array([jacc(texts[str(u)], texts[str(v)]) for u, v in zip(x, y)])
        e = np.array([float(emb[str(u)] @ emb[str(v)]) for u, v in zip(x, y)])
        return w, e, ok

    res = {"rule": __doc__, "n_pairs": len(pi), "n_refs": len(refs)}
    w, e, ok = sims(pi, pj)
    res["random_pairs_used"] = int(ok.sum())
    rw, re_, rok = sims(refs, rand)
    for n in names:
        th, raw, _ = Z[n]
        r = {}
        for nm, M in (("thought", th), ("raw", raw)):
            c = (M[pi] * M[pj]).sum(1)
            if nm == "raw":
                c = c / np.maximum(np.linalg.norm(M[pi], axis=1) * np.linalg.norm(M[pj], axis=1), 1e-12)
            r[f"spearman_{nm}_words"] = round(float(spearmanr(c[ok], w[ok]).statistic), 4)
            r[f"spearman_{nm}_emb"] = round(float(spearmanr(c[ok], e[ok]).statistic), 4)
        nw, ne, nok = sims(refs, nb[n])
        k = nok & rok
        r["nb_n"] = int(k.sum())
        r["words_nb"], r["words_rand"], r["words_diff"] = ci(nw[k]), ci(rw[k]), ci(nw[k] - rw[k])
        r["emb_nb"], r["emb_rand"], r["emb_diff"] = ci(ne[k]), ci(re_[k]), ci(ne[k] - re_[k])
        r["examples"] = [{"ref": texts[str(int(refs[i]))], "nb": texts[str(int(nb[n][i]))]} for i in range(8)]
        res[n] = r
        log(n, json.dumps({x: y for x, y in r.items() if x != "examples"}))
    (out / "measure.json").write_text(json.dumps(res, indent=1))
    log("MEASURE DONE")


if __name__ == "__main__":
    main()
