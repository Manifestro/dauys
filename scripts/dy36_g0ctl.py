"""DY-36 G0 positive control (validates the measurement only; text never enters a dictionary or a model).

Subset: 20,000 bank utterances (the G0 bank, 4-15 s) drawn at random, seed 1234. Random candidate per reference:
uniform from the subset among lengths within x2 (seed 1235). The 500 G0 references are unchanged.
T / T10 / T100: the 1st / 10th / 100th nearest subset utterance by cosine of text sentence embeddings
(sentence-transformers/all-MiniLM-L6-v2, mean pooling over tokens, L2-normalised) of the Whisper large-v3-turbo transcripts.
Dictionary rows: nearest subset utterance by thought cosine (valid thoughts only).
Judge as G0 (CLOSER, both orders), letters and Whisper rows, plus the share of items whose answer flips with the order.
Judge-free reading: mean text-embedding cosine reference-neighbour vs reference-random (paired), 95% CI.

    python scripts/dy36_g0ctl.py --dict /data/dy36/dict --dicts A,C2500 --out /data/dy36/g0ctl
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from dy36_common import CLOSER, SR, Audio, Judge, log, transcribe, utt_job
from dy36_dict import load_table
from dy36_g0 import select

NSUB, SEED = 20000, 1234
EMB = "sentence-transformers/all-MiniLM-L6-v2"


def embed(texts: list[str]) -> np.ndarray:
    import torch
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(EMB)
    m = AutoModel.from_pretrained(EMB).cuda().eval()
    out = []
    with torch.no_grad():
        for b in range(0, len(texts), 256):
            enc = tok(texts[b : b + 256], padding=True, truncation=True, max_length=256, return_tensors="pt").to("cuda")
            h = m(**enc).last_hidden_state
            msk = enc["attention_mask"][..., None].float()
            v = (h * msk).sum(1) / msk.sum(1).clamp(min=1)
            out.append(torch.nn.functional.normalize(v, dim=-1).cpu().numpy())
    del m
    torch.cuda.empty_cache()
    return np.concatenate(out)


def mean_ci(x: np.ndarray) -> dict:
    return {"mean": round(float(x.mean()), 4), "ci95": [round(float(x.mean() - 1.96 * x.std(ddof=1) / np.sqrt(len(x))), 4),
                                                       round(float(x.mean() + 1.96 * x.std(ddof=1) / np.sqrt(len(x))), 4)]}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dict", required=True)
    p.add_argument("--dicts", default="A,C2500")
    p.add_argument("--out", required=True)
    p.add_argument("--g0texts", default="/data/dy36/g0/texts.json")
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    T = load_table(Path(a.dict))
    files = json.loads(str(T["files"]))
    refs, bank, _ = select(T)
    sub = np.sort(np.random.default_rng(SEED).choice(bank, NSUB, replace=False))
    r2 = np.random.default_rng(SEED + 1)
    rand = []
    for u in refs:
        r = T["len"][sub] / T["len"][u]
        el = sub[(r >= 0.5) & (r <= 2.0)]
        rand.append(int(el[r2.integers(len(el))]))
    rand = np.array(rand)
    (out / "subset.json").write_text(json.dumps({"rule": __doc__, "subset_rows": sub.tolist(), "ref_rows": refs.tolist(),
                                                 "rand_rows": rand.tolist()}))
    audio = Audio(files)
    # Whisper of the subset (and letters of the candidates later); reference texts from the G0 cache
    g0t = json.loads(Path(a.g0texts).read_text())
    wf = out / "whisper_subset.json"
    W = json.loads(wf.read_text()) if wf.exists() else {}
    todo = [int(u) for u in sub if str(int(u)) not in W]
    for b in range(0, len(todo), 4000):
        part = todo[b : b + 4000]
        wavs = audio.many([utt_job(T, u) for u in part])
        import torch
        from transformers import pipeline

        asr = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3-turbo", device="cuda",
                       torch_dtype=torch.float16, chunk_length_s=30)
        res = asr([{"raw": w.astype(np.float32) / 32768.0, "sampling_rate": SR} for w in wavs], batch_size=32,
                  generate_kwargs={"language": "en", "task": "transcribe"})
        for u, r in zip(part, res):
            W[str(u)] = r["text"].strip()
        del asr
        torch.cuda.empty_cache()
        wf.write_text(json.dumps(W))
        log("whisper subset", len(W))
    ref_w = [g0t[str(int(u))]["whisper"] for u in refs]
    E = embed(ref_w + [W[str(int(u))] for u in sub])
    Er, Es = E[: len(refs)], E[len(refs) :]
    pos = {int(u): i for i, u in enumerate(sub)}
    S = Er @ Es.T
    rk = np.argsort(-S, 1)
    nb = {"T": sub[rk[:, 0]], "T10": sub[rk[:, 9]], "T100": sub[rk[:, 99]]}
    for name in a.dicts.split(","):
        z = np.load(Path(a.dict) / f"dict_{name}.npz")
        th, valid = z["thought"], z["valid"]
        s2 = sub[valid[sub]]
        nb[name] = s2[(th[refs] @ th[s2].T).argmax(1)]
    for k, v in nb.items():
        assert np.isin(v, sub).all() and not np.isin(T["ep"][v], T["ep"][refs]).any()
    # letters for every candidate and reference
    need = sorted({int(u) for v in list(nb.values()) + [rand] for u in v})
    lf = out / "letters.json"
    Lt = json.loads(lf.read_text()) if lf.exists() else {}
    todo = [u for u in need if str(u) not in Lt]
    if todo:
        le, _ = transcribe(audio.many([utt_job(T, u) for u in todo]), whisper=False)
        Lt.update({str(u): t for u, t in zip(todo, le)})
        lf.write_text(json.dumps(Lt))
    txt = lambda u, row: (Lt[str(int(u))] if row == "letters" else W[str(int(u))])
    ref_t = {"letters": [g0t[str(int(u))]["letters"] for u in refs], "whisper": ref_w}
    cos_rand = np.array([Er[i] @ Es[pos[int(rand[i])]] for i in range(len(refs))])
    import subprocess

    # the judge (bf16 ~15 GB) does not fit next to the B feature pass on a 24 GB GPU: pause it (it resumes per shard)
    paused = subprocess.run("pkill -f '[d]y36_featB.py pass'", shell=True).returncode == 0
    if paused:
        import time

        time.sleep(15)
    j = Judge()
    res = {"rule": __doc__, "n_refs": len(refs), "subset": NSUB, "embedding_model": EMB}
    for name, v in nb.items():
        r = {}
        for row in ("letters", "whisper"):
            r[row] = j.pairs(CLOSER, ref_t[row], [txt(u, row) for u in v], [txt(u, row) for u in rand])
            log("CTL", name, row, json.dumps({k: x for k, x in r[row].items() if k != "per_item"}))
        cn = np.array([Er[i] @ Es[pos[int(v[i])]] for i in range(len(refs))])
        r["text_cos_neighbour"] = mean_ci(cn)
        r["text_cos_random"] = mean_ci(cos_rand)
        r["text_cos_diff"] = mean_ci(cn - cos_rand)
        r["nb_rows"] = [int(u) for u in v]
        r["examples"] = [{"ref": ref_w[i], "neighbour": W[str(int(v[i]))], "random": W[str(int(rand[i]))]} for i in range(5)]
        res[name] = r
        (out / "g0ctl.json").write_text(json.dumps(res, indent=1))
    del j
    import torch

    torch.cuda.empty_cache()
    if paused:
        subprocess.run("nohup /data/run.sh python scripts/dy36_featB.py pass --dict /data/dy36/dict --work /data/w "
                       ">> /data/dy36/featB.log 2>&1 < /dev/null &", shell=True)
    summ = {n: {"G0_letters": res[n]["letters"]["D"], "ci95": res[n]["letters"]["ci95"],
                "flip_letters": res[n]["letters"]["order_flip_share"], "G0_whisper": res[n]["whisper"]["D"],
                "ci95_whisper": res[n]["whisper"]["ci95"], "flip_whisper": res[n]["whisper"]["order_flip_share"],
                "text_cos_nb": res[n]["text_cos_neighbour"], "text_cos_rand": res[n]["text_cos_random"]["mean"],
                "text_cos_diff": res[n]["text_cos_diff"]} for n in nb}
    (out / "g0ctl_summary.json").write_text(json.dumps(summ, indent=1))
    log("CTL DONE", json.dumps(summ))


if __name__ == "__main__":
    main()
