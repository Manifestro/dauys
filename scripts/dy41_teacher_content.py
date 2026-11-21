"""Teachers trained with text: do their features carry the content of an utterance? DY-36's yardstick (as
dy41_mimi_content.py). Candidates (only as training targets of a thought vector later; text never enters our model):
  whL   Whisper large-v3 encoder (Apache-2.0), output of encoder layer L in 8, 16, 24, 32 (50 Hz frames)
  w2vL  wav2vec2-large-960h-lv60-self (Apache-2.0, CTC on LibriSpeech transcripts), hidden state L in 12, 18, 24
Every utterance alone, mean over its own frames. Variants a, b (minus the speaker's mean), c*, bc* as dy36_sweep.
References on the same yardstick: text +0.41, WavLM-L18 +0.096, Mimi <= +0.054, DY-41 LM <= +0.054; content >= +0.10.

    python scripts/dy41_teacher_content.py --out /data/tc
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import dy36_common
from dy36_common import log, utt_job
from dy36_dict import load_table
from dy36_sweep import Yard, pc_remove, utt_list
from dy40_mimi import BUCKET, SR

WH_LAYERS = [8, 16, 24, 32]
W2V_LAYERS = [12, 18, 24]


def s3get(key: str, dst: Path) -> None:
    if not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(f"aws s3 cp {BUCKET}/dauys/{key} {dst} --only-show-errors", shell=True, check=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="/data/tc")
    p.add_argument("--batch", type=int, default=16)
    a = p.parse_args()
    out = Path(a.out)
    dev = torch.device("cuda")
    s3get("dy36/dict/table.npz", out / "dict/table.npz")
    for f in ("subset.json", "emb.npz"):
        s3get(f"dy36/d/ctl_inputs/{f}", out / f"ctl/{f}")
    T = load_table(out / "dict")
    Y = Yard(T, out / "ctl")
    dy36_common.IDX_DIR = out / "idx"
    dy36_common.HELD_DIR = out / "held_none"
    rows, ncore = utt_list(T, Y)
    wavs = dy36_common.Audio(json.loads(str(T["files"]))).many([utt_job(T, int(u)) for u in rows])
    wavs = [np.asarray(w, np.float32) / 32768.0 for w in wavs]
    log("utterances", len(rows), "core", ncore, "audio h", round(sum(len(w) for w in wavs) / SR / 3600, 2))
    order = np.argsort(-np.array([len(w) for w in wavs]))
    pooled: dict[str, np.ndarray] = {}

    def put(k: str, i: int, v: torch.Tensor) -> None:
        pooled.setdefault(k, np.zeros((len(rows), v.numel()), np.float32))[i] = v.float().cpu().numpy()

    # Whisper large-v3 encoder (30 s window; pool over the utterance's own 50 Hz frames)
    from transformers import WhisperFeatureExtractor, WhisperModel

    fe = WhisperFeatureExtractor.from_pretrained("openai/whisper-large-v3")
    enc = WhisperModel.from_pretrained("openai/whisper-large-v3", torch_dtype=torch.float16).encoder.to(dev).eval()
    t0 = time.time()
    with torch.inference_mode():
        for b in range(0, len(order), a.batch):
            idx = order[b : b + a.batch]
            feats = fe([wavs[i][: 30 * SR] for i in idx], sampling_rate=SR, return_tensors="pt").input_features
            hs = enc(feats.to(dev, torch.float16), output_hidden_states=True).hidden_states
            for r, i in enumerate(idx):
                nf = max(1, min(1500, int(np.ceil(len(wavs[i]) / 320))))
                for L in WH_LAYERS:
                    put(f"wh{L}", i, hs[L][r, :nf].float().mean(0))
            if (b // a.batch) % 100 == 0:
                log("whisper", b + len(idx), "of", len(order), round(time.time() - t0), "s")
    del enc
    torch.cuda.empty_cache()

    # wav2vec2-large-960h-lv60-self (input normalised per utterance, as its feature extractor)
    from transformers import Wav2Vec2ForCTC

    w2v = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h-lv60-self").wav2vec2.to(dev).eval()
    t0 = time.time()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        for b in range(0, len(order), a.batch):
            idx = order[b : b + a.batch]
            n = max(len(wavs[i]) for i in idx)
            x = torch.zeros(len(idx), n)
            am = torch.zeros(len(idx), n, dtype=torch.long)
            for r, i in enumerate(idx):
                w = torch.from_numpy(wavs[i])
                x[r, : len(w)] = (w - w.mean()) / (w.std() + 1e-7)
                am[r, : len(w)] = 1
            hs = w2v(x.to(dev), attention_mask=am.to(dev), output_hidden_states=True).hidden_states
            for r, i in enumerate(idx):
                nf = max(1, (len(wavs[i]) - 400) // 320 + 1)
                for L in W2V_LAYERS:
                    put(f"w2v{L}", i, hs[L][r, :nf].float().mean(0))
            if (b // a.batch) % 100 == 0:
                log("wav2vec2", b + len(idx), "of", len(order), round(time.time() - t0), "s")
    np.savez(out / "pooled.npz", **pooled)

    nsub, nref = len(Y.sub), len(Y.refs)
    _, inv = np.unique(T["spk"][rows], return_inverse=True)
    cnt = np.bincount(inv)
    res = []
    for k, X in pooled.items():
        S = np.zeros((cnt.size, X.shape[1]))
        np.add.at(S, inv, X)
        C = X - (S / cnt[:, None])[inv]
        core = slice(0, nsub + nref)
        variants = {"a": X[core], "b": C[core]}
        for pc in (1, 3, 10):
            variants[f"c{pc}"] = pc_remove(X[core], X[:nsub], pc)
            variants[f"bc{pc}"] = pc_remove(C[core], C[:nsub], pc)
        for v, Mx in variants.items():
            r = Y.score(Mx[nsub:], Mx[:nsub])
            r["cand"] = f"{k}-{v}"
            res.append(r)
    res.sort(key=lambda r: -r["diff"])
    best: dict[str, dict] = {}
    for r in res:
        best.setdefault(r["cand"].split("-")[0], r)
    lines = ["# Content in teachers trained with text, by DY-36's yardstick", "",
             "References: text +0.41, WavLM-L18 +0.096, Mimi and DY-41 <= +0.054; 'carries content' from +0.10.", "",
             "| representation | best variant | reading | 95% CI |", "|---|---|---|---|"]
    for k, r in best.items():
        lines.append(f"| {k} | {r['cand']} | {r['diff']:+.3f} | {r['ci95'][0]:+.3f} … {r['ci95'][1]:+.3f} |")
    (out / "teacher_content.json").write_text(json.dumps({"best": best, "all": res}, indent=1))
    (out / "teacher_content.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)

    import airon

    run = airon.init()
    m = {f"eval/content_{k}": r["diff"] for k, r in best.items()}
    m.update({f"eval/content_{k}_ci_lo": r["ci95"][0] for k, r in best.items()})
    run.log(m)
    subprocess.run(f"aws s3 cp {out}/teacher_content.json {BUCKET}/dauys/dy41/teacher_content/ --only-show-errors"
                   f" && aws s3 cp {out}/teacher_content.md {BUCKET}/dauys/dy41/teacher_content/ --only-show-errors",
                   shell=True, check=True)
    log("TEACHER CONTENT DONE")


if __name__ == "__main__":
    main()
