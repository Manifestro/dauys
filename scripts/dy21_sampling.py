"""DY-21b: can sampling alone remove the repeats, and is the voice path or the model the reason the audio is unclear?

Same 200 held-out prompts as dy21_eval.py (same pieces, same seed), the saved DY-21 checkpoint, no training.
Generation settings (60 units each): base (T 0.8, top-p 0.95, as in DY-21), repetition penalty 1.2 over the
last 60 units, a ban on repeating any 4-gram of units already in the context, and both with T 0.7 / T 1.0.
Each continuation is rendered with the fixed DY-21 timing (one unit then one "continue" frame), Whisper
large-v3-turbo transcribes it and GPT-2 medium gives the transcript's log-perplexity (rulers only).

Timing check: the true next 60 units are rendered twice, with the fixed timing (as in DY-21) and with their
real timing and pauses from the 10 Hz grid. If the real-timed version gives many more words and a lower
log-PPL, the unclear audio comes from the fixed timing (the model predicts no durations), not from the units.

    python scripts/dy21_sampling.py --units /opt/dlami/nvme/units --run /data/dy21/run --means unit_means.pt \
        --config sylber_resynthesis.yaml --ref ref.wav --out /data/dy21b
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
from dy21_eval import CONT, N, PROMPT, rep4, to_grid
from dy21_train import EOS, K

from dauys.dy16_model import sample_next
from dauys.dy21_lm import LMConfig, UnitLM

SETTINGS = {
    "base": dict(t=0.8, p=0.95, rp=1.0, ban4=False),
    "rp1.2": dict(t=0.8, p=0.95, rp=1.2, ban4=False),
    "ban4": dict(t=0.8, p=0.95, rp=1.0, ban4=True),
    "rp1.2_ban4_t0.7": dict(t=0.7, p=0.95, rp=1.2, ban4=True),
    "rp1.2_ban4_t1.0": dict(t=1.0, p=0.95, rp=1.2, ban4=True),
}


def grid_pieces(files: list[Path]) -> list[np.ndarray]:
    """10 Hz grid of every piece, in the same order and with the same filter as dy21_train.load_stream."""
    out = []
    for f in files:
        z = np.load(f)
        tok, index = z["tok"], z["index"]
        for st, n in index:
            g = tok[st : st + n].astype(np.int64)
            if (g < K).sum() >= 16:
                out.append(g)
    return out


@torch.no_grad()
def generate(model: UnitLM, prompts: np.ndarray, device, seed: int, s: dict) -> np.ndarray:
    gen = torch.Generator(device=device).manual_seed(seed)
    ids = torch.from_numpy(prompts.astype(np.int64)).to(device)
    model.eval()
    for _ in range(CONT):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h = model.states(ids[:, -model.cfg.ctx :])
        logits = model.logits(h[:, -1])
        logits[:, EOS] = -float("inf")
        if s["rp"] != 1.0:
            recent = ids[:, -CONT:]
            v = logits.gather(1, recent)
            logits.scatter_(1, recent, torch.where(v > 0, v / s["rp"], v * s["rp"]))
        if s["ban4"]:
            hist = ids.cpu().numpy()
            for b, row in enumerate(hist):
                last = tuple(row[-3:])
                banned = [row[i + 3] for i in range(len(row) - 3) if tuple(row[i : i + 3]) == last]
                if banned:
                    logits[b, torch.tensor(banned, device=device)] = -float("inf")
        ids = torch.cat([ids, sample_next(logits, s["t"], s["p"], gen)[:, None]], 1)
    return ids[:, -CONT:].cpu().numpy()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--units", required=True)
    p.add_argument("--run", required=True)
    p.add_argument("--means", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--ref", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=N)
    args = p.parse_args()
    out = Path(args.out)
    (out / "listen").mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    meta = json.loads((Path(args.run) / "metadata.json").read_text())
    grids = grid_pieces([Path(args.units) / f"{n}.npz" for n in meta["val_files_names"]])
    grids = [g for g in grids if (g < K).sum() >= PROMPT + CONT]
    rng = np.random.default_rng(1234)  # same draws as dy21_eval.py
    pick = rng.choice(len(grids), size=min(args.n, len(grids)), replace=False)
    units = [grids[i][grids[i] < K] for i in pick]
    prompts = np.stack([u[:PROMPT] for u in units])
    real = np.stack([u[PROMPT : PROMPT + CONT] for u in units])
    real_timed = []
    for i in pick:
        pos = np.flatnonzero(grids[i] < K)
        real_timed.append(grids[i][pos[PROMPT] : pos[PROMPT + CONT - 1] + 2])

    ck = torch.load(Path(args.run) / "best.pt", map_location="cpu", weights_only=False)
    model = UnitLM(LMConfig(**ck["cfg"])).to(device)
    model.load_state_dict(ck["model"])
    gens = {}
    for name, s in SETTINGS.items():
        gens[name] = np.concatenate(
            [generate(model, prompts[i : i + 50], device, 1234 + i, s) for i in range(0, len(prompts), 50)])
        print("generated", name, "rep4", round(float(np.mean([rep4(u) for u in gens[name]])), 4), flush=True)
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

    def wav_of_grid(g: np.ndarray) -> np.ndarray:
        return render(synth, coder, unit_frames(g, means, K), ref, device)

    def text(w: np.ndarray) -> str:
        return asr({"raw": w.astype(np.float32), "sampling_rate": SR}, generate_kwargs={"language": "en"})["text"].strip()

    @torch.no_grad()
    def logppl(t: str) -> float:
        ids = tok(t, return_tensors="pt").input_ids.to(device)
        if ids.shape[1] < 3:
            return float("nan")
        return float(lm(ids, labels=ids).loss)

    kinds = {f"gen_{k}": [to_grid(u) for u in v] for k, v in gens.items()}
    kinds["real_fixed"] = [to_grid(u) for u in real]
    kinds["real_timed"] = real_timed
    rows = {}
    for kind, arr in kinds.items():
        r = []
        for g in arr:
            t = text(wav_of_grid(g))
            r.append({"text": t, "logppl": logppl(t), "words": len(t.split()), "rep4": rep4(g[g < K])})
        rows[kind] = r
        v = np.array([x["logppl"] for x in r])
        print(kind, json.dumps({"median_logppl": float(np.nanmedian(v)), "words": float(np.median([x["words"] for x in r])),
                                "rep4": float(np.mean([x["rep4"] for x in r]))}), flush=True)

    summ = {}
    for kind, r in rows.items():
        v = np.array([x["logppl"] for x in r])
        summ[kind] = {"median_logppl": float(np.nanmedian(v)), "no_text_share": float(np.mean(np.isnan(v))),
                      "words_median": float(np.median([x["words"] for x in r])),
                      "rep4": float(np.mean([x["rep4"] for x in r])),
                      "rep4_any_share": float(np.mean([x["rep4"] > 0 for x in r]))}
    lim = 2 * max(summ["real_fixed"]["rep4"], 1e-3)
    ok = [k for k in summ if k.startswith("gen_") and summ[k]["rep4"] <= lim]
    summ["rep4_limit"] = lim
    summ["passing_rep4"] = ok
    summ["n"] = len(prompts)
    (out / "eval.json").write_text(json.dumps({"summary": summ, "rows": rows}, indent=2, ensure_ascii=False))
    print("SUMMARY", json.dumps(summ), flush=True)

    best = min(ok, key=lambda k: summ[k]["median_logppl"]) if ok else "gen_base"
    best_units = gens[best.removeprefix("gen_")]
    for i in range(min(5, len(prompts))):
        ctx = to_grid(prompts[i][-25:])  # ~5 s of context, so most of each file is the continuation
        for kind, g in ((best, to_grid(best_units[i])), ("real_timed", real_timed[i])):
            sf.write(out / "listen" / f"{i:02d}_{kind}.wav", wav_of_grid(np.concatenate([ctx, g])), SR)


if __name__ == "__main__":
    main()
