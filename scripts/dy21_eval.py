"""DY-21 evaluation: are the model's continuations made of words?

200 held-out pieces (shards never trained on) with at least 210 units; prompt = the first 150 units
(~30 s of speech), continuation = 60 units. Three continuations per prompt, each rendered with the same
fixed voice and timing (a unit then one "continue" frame: 5 units/s; synthesis = DY-14 unit means ->
Sylber flow matching -> SPARC): ``gen`` (the model, temperature 0.8, top-p 0.95), ``real`` (the true next
60 units, the ceiling of this pipeline) and ``floor`` (60 units drawn from the unigram distribution).
Whisper large-v3-turbo transcribes each; GPT-2 medium gives the transcript's per-token log-perplexity
(text models are rulers only, as Whisper was before). Gate W: median log-PPL of gen closes at least half
the gap from floor to real, and the 4-gram repeat share of gen units is at most twice the real one.

    python scripts/dy21_eval.py --units /opt/dlami/nvme/units --run /data/dy21/run --means unit_means.pt \
        --config sylber_resynthesis.yaml --ref ref.wav --out /data/dy21/eval
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).parent))
from dy16_voice import SR, load_synth, render, unit_frames
from dy21_train import EOS, K, load_stream

from dauys.dy16_model import sample_next
from dauys.dy21_lm import LMConfig, UnitLM

PROMPT, CONT, N = 150, 60, 200


def to_grid(units: np.ndarray) -> np.ndarray:
    g = np.full(2 * len(units), K, np.int64)
    g[0::2] = units
    return g


@torch.no_grad()
def generate(model: UnitLM, prompts: np.ndarray, device, seed: int) -> np.ndarray:
    gen = torch.Generator(device=device).manual_seed(seed)
    ids = torch.from_numpy(prompts.astype(np.int64)).to(device)
    model.eval()
    for _ in range(CONT):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h = model.states(ids[:, -model.cfg.ctx :])
        logits = model.logits(h[:, -1])
        logits[:, EOS] = -float("inf")
        ids = torch.cat([ids, sample_next(logits, 0.8, 0.95, gen)[:, None]], 1)
    return ids[:, -CONT:].cpu().numpy()


def rep4(u: np.ndarray) -> float:
    g = [tuple(u[i : i + 4]) for i in range(len(u) - 3)]
    return 1 - len(set(g)) / len(g) if g else 0.0


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--units", required=True)
    p.add_argument("--run", required=True)
    p.add_argument("--means", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--ref", required=True, help="10 s wav of one held-out voice, used for every rendering")
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=N)
    args = p.parse_args()
    out = Path(args.out)
    (out / "listen").mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    meta = json.loads((Path(args.run) / "metadata.json").read_text())
    val = load_stream([Path(args.units) / f"{n}.npz" for n in meta["val_files_names"]])
    pieces = np.split(val, np.flatnonzero(val == EOS) + 1)
    pieces = [x[x != EOS] for x in pieces if (x != EOS).sum() >= PROMPT + CONT]
    rng = np.random.default_rng(1234)
    pick = rng.choice(len(pieces), size=min(args.n, len(pieces)), replace=False)
    prompts = np.stack([pieces[i][:PROMPT] for i in pick])
    real = np.stack([pieces[i][PROMPT : PROMPT + CONT] for i in pick])
    counts = np.bincount(val[val != EOS].astype(np.int64), minlength=K).astype(np.float64)
    floor = rng.choice(K, size=real.shape, p=counts / counts.sum())

    ck = torch.load(Path(args.run) / "best.pt", map_location="cpu", weights_only=False)
    model = UnitLM(LMConfig(**ck["cfg"])).to(device)
    model.load_state_dict(ck["model"])
    gen = np.concatenate([generate(model, prompts[i : i + 50], device, 1234 + i) for i in range(0, len(prompts), 50)])
    del model
    torch.cuda.empty_cache()

    from sparc import load_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

    means = torch.load(args.means)["means"].numpy()
    synth = load_synth(args.config, device)
    coder = load_model("en", device="cuda", use_penn=False)
    ref, _ = sf.read(args.ref, dtype="float32")
    asr = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3-turbo", device=0,
                   torch_dtype=torch.float16)
    tok = AutoTokenizer.from_pretrained("gpt2-medium")
    lm = AutoModelForCausalLM.from_pretrained("gpt2-medium").to(device).eval()

    def wav_of(units: np.ndarray) -> np.ndarray:
        return render(synth, coder, unit_frames(to_grid(units), means, K), ref, device)

    def text(w: np.ndarray) -> str:
        return asr({"raw": w.astype(np.float32), "sampling_rate": SR}, generate_kwargs={"language": "en"})["text"].strip()

    @torch.no_grad()
    def logppl(t: str) -> float:
        ids = tok(t, return_tensors="pt").input_ids.to(device)
        if ids.shape[1] < 3:
            return float("nan")
        return float(lm(ids, labels=ids).loss)

    rows = {}
    for kind, arr in (("gen", gen), ("real", real), ("floor", floor)):
        r = []
        for i, u in enumerate(arr):
            t = text(wav_of(u))
            r.append({"text": t, "logppl": logppl(t), "words": len(t.split()), "rep4": rep4(u)})
        rows[kind] = r
        print(kind, json.dumps(r[:3]), flush=True)
    for i in range(min(10, len(gen))):
        for kind, arr in (("gen", gen), ("real", real)):
            w = wav_of(np.concatenate([prompts[i], arr[i]]))
            sf.write(out / "listen" / f"{i:02d}_{kind}.wav", w, SR)

    def med(kind: str) -> float:
        v = np.array([x["logppl"] for x in rows[kind]])
        return float(np.nanmedian(v))

    summ = {
        kind: {
            "median_logppl": med(kind),
            "no_text_share": float(np.mean([np.isnan(x["logppl"]) for x in rows[kind]])),
            "words_median": float(np.median([x["words"] for x in rows[kind]])),
            "rep4": float(np.mean([x["rep4"] for x in rows[kind]])),
        }
        for kind in rows
    }
    g, r, f = summ["gen"]["median_logppl"], summ["real"]["median_logppl"], summ["floor"]["median_logppl"]
    summ["gap_closed"] = (f - g) / (f - r) if f != r else float("nan")
    summ["W"] = bool(summ["gap_closed"] >= 0.5 and summ["gen"]["rep4"] <= 2 * max(summ["real"]["rep4"], 1e-3))
    summ["n"] = len(gen)
    (out / "eval.json").write_text(json.dumps({"summary": summ, "rows": rows}, indent=2, ensure_ascii=False))
    print(json.dumps(summ), flush=True)


if __name__ == "__main__":
    main()
