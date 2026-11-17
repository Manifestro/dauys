"""DY-22 evaluation: does the model speak with a human tempo, pauses and intonation, and still in words?

200 held-out pieces (pauses cut to 2 s as in training); prompt = first 300 grid frames (30 s), continuation =
150 frames (15 s). ``gen``: the model with the DY-21b sampler (repetition penalty 1.2 on recent units, no
repeated unit 4-gram, T 0.7, top-p 0.95), its own timing and its own sampled F0/energy/voicing.
``real``: the true next 150 frames with their timing and F0. ``floor``: the real frames with every unit
replaced by a unigram draw (real timing). All rendered with one fixed voice: DY-14 unit means -> Sylber
flow matching -> SPARC, with the pitch track set from the (model's or real) F0 of each voiced unit.

Rulers: Whisper large-v3-turbo transcript -> GPT-2 medium log-PPL; syllable rate (units per second of
non-pause), pause share (pause frames / all), F0 spread (std of F0 semitones over voiced units).
Gate L (all three):
  L1 words: median log-PPL of gen closes >= 0.5 of the floor-to-real gap, and gen rep4 <= 2 x max(real, 1e-3);
  L2 tempo: median syllable rate of gen within 15 % of real, and median pause share within 30 % of real;
  L3 intonation: median F0 spread of gen within 30 % of real.

    python scripts/dy22_eval.py --units /opt/dlami/nvme/units --run /data/dy22/run --means unit_means.pt \
        --config sylber_resynthesis.yaml --ref ref.wav --out /data/dy22/eval
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.ndimage import uniform_filter1d
from scipy.signal import resample_poly

sys.path.insert(0, str(Path(__file__).parent))
from dy16_voice import SR, load_synth, unit_frames
from dy21_eval import rep4
from dy22_train import load_grid

from dauys.dy14_units import GRID
from dauys.dy16_model import sample_next
from dauys.dy21_lm import LMConfig
from dauys.dy22_lm import CONT, EOS, K, SIL, GridLM, e_of_bin, f0_of_bin

PROMPT, GEN, N = 300, 150, 200
T_UNIT, TOP_P, RP = 0.7, 0.95, 1.2


@torch.no_grad()
def generate(model: GridLM, prompts: np.ndarray, device, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Grid tokens [B, GEN] and prosody [B, GEN, 3] (f0 st, energy dB, voiced), zeros at non-units."""
    g = torch.Generator(device=device).manual_seed(seed)
    ids = torch.from_numpy(prompts.astype(np.int64)).to(device)
    b = len(ids)
    pros = torch.zeros(b, GEN + 1, 3, device=device)
    model.eval()
    for step in range(GEN + 1):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h = model.states(ids[:, -model.cfg.ctx :])[:, -1]
        if step:
            pf, pe, pv = model.prosody(h)
            v = torch.multinomial(torch.softmax(pv, -1), 1, generator=g)[:, 0]
            f = f0_of_bin(torch.multinomial(torch.softmax(pf, -1), 1, generator=g)[:, 0])
            e = e_of_bin(torch.multinomial(torch.softmax(pe, -1), 1, generator=g)[:, 0])
            unit = ids[:, -1] < K
            pros[:, step - 1] = torch.stack([f * v, e, v.float()], -1) * unit[:, None]
        if step == GEN:
            break
        logits = model.logits(h)
        logits[:, EOS] = -float("inf")
        hist = ids.cpu().numpy()
        for r, row in enumerate(hist):
            u = row[row < K]
            recent = torch.from_numpy(np.unique(u[-60:])).to(device)
            val = logits[r, recent]
            logits[r, recent] = torch.where(val > 0, val / RP, val * RP)
            last = tuple(u[-3:])
            banned = [u[i + 3] for i in range(len(u) - 3) if tuple(u[i : i + 3]) == last]
            if banned:
                logits[r, torch.tensor(banned, device=device)] = -float("inf")
        ids = torch.cat([ids, sample_next(logits, T_UNIT, TOP_P, g)[:, None]], 1)
    return ids[:, -GEN:].cpu().numpy(), pros[:, :GEN].cpu().numpy()


def stats(tok: np.ndarray, pros: np.ndarray) -> dict:
    units = tok < K
    active = (tok != SIL).sum()
    st = pros[units & (pros[:, 2] > 0.5), 0]
    return {"rate": float(units.sum() / max(active / 10, 0.1)), "pause": float((tok == SIL).mean()),
            "f0_std": float(st.std()) if len(st) > 2 else float("nan"), "rep4": rep4(tok[units])}


@torch.no_grad()
def render(synth, coder, tok: np.ndarray, pros: np.ndarray, means: np.ndarray, ref: np.ndarray, device) -> np.ndarray:
    frames = unit_frames(tok, means, K)
    feats = torch.from_numpy(frames).float().to(device)[None]
    art = synth.resynthesize(features=feats, steps=50, rand_scale=0.0)[0][0].cpu().numpy()
    # pitch from F0 of each voiced unit, held from its onset to the next event, smoothed over 100 ms
    lp = np.full(len(tok) * GRID, np.nan, np.float32)
    cur = np.nan
    for i, t in enumerate(tok):
        if t < K:
            cur = pros[i, 0] * np.log(2) / 12 if pros[i, 2] > 0.5 else np.nan
        elif t == SIL:
            cur = np.nan
        lp[i * GRID : (i + 1) * GRID] = cur
    n = min(len(art), len(lp))
    m = ~np.isnan(lp[:n])
    if m.any():
        sm = uniform_filter1d(np.where(m, lp[:n], 0), 5) / np.maximum(uniform_filter1d(m.astype(np.float32), 5), 1e-3)
        art[:n, 12] = np.where(m, sm, art[:n, 12])
    sf.write("/tmp/dy22_ref.wav", ref, SR)
    sp = coder.encode("/tmp/dy22_ref.wav")
    pitch = np.exp(art[:, 12:13]) * sp["pitch_stats"][0]
    wav = np.asarray(coder.decode(art[:, :12], pitch, art[:, 13:], sp["spk_emb"]), np.float32).reshape(-1)
    return resample_poly(wav, SR, coder.sr) if coder.sr != SR else wav


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
    tok, pb = load_grid([Path(args.units) / f"{n}.npz" for n in meta["val_files_names"]])
    cuts = np.flatnonzero(tok == EOS)
    bounds = list(zip(np.r_[0, cuts[:-1] + 1], cuts))
    bounds = [(s, e) for s, e in bounds if e - s >= PROMPT + GEN]
    rng = np.random.default_rng(1234)
    pick = rng.choice(len(bounds), size=min(args.n, len(bounds)), replace=False)
    prompts = np.stack([tok[bounds[i][0] : bounds[i][0] + PROMPT] for i in pick]).astype(np.int64)
    real = np.stack([tok[bounds[i][0] + PROMPT : bounds[i][0] + PROMPT + GEN] for i in pick]).astype(np.int64)
    # real prosody back from bins
    def unbin(x: np.ndarray) -> np.ndarray:
        t = torch.from_numpy(x.astype(np.int64))
        return np.stack([(f0_of_bin(t[..., 0]) * t[..., 2]).numpy(), e_of_bin(t[..., 1]).numpy(),
                         t[..., 2].float().numpy()], -1)
    real_p = np.stack([unbin(pb[bounds[i][0] + PROMPT : bounds[i][0] + PROMPT + GEN]) for i in pick])
    prompt_p = np.stack([unbin(pb[bounds[i][0] : bounds[i][0] + PROMPT]) for i in pick])
    units_all = tok[tok < K].astype(np.int64)
    counts = np.bincount(units_all, minlength=K).astype(np.float64)
    floor = real.copy()
    mask = floor < K
    floor[mask] = rng.choice(K, size=int(mask.sum()), p=counts / counts.sum())

    ck = torch.load(Path(args.run) / "best.pt", map_location="cpu", weights_only=False)
    model = GridLM(LMConfig(**ck["cfg"])).to(device)
    model.load_state_dict(ck["model"])
    gen, gen_p = [], []
    for i in range(0, len(prompts), 50):
        a, b = generate(model, prompts[i : i + 50], device, 1234 + i)
        gen.append(a)
        gen_p.append(b)
    gen, gen_p = np.concatenate(gen), np.concatenate(gen_p)
    del model
    torch.cuda.empty_cache()
    print("generated", len(gen), flush=True)

    from sparc import load_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

    means = torch.load(args.means)["means"].numpy()
    synth = load_synth(args.config, device)
    coder = load_model("en", device="cuda", use_penn=False)
    ref, _ = sf.read(args.ref, dtype="float32")
    asr = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3-turbo", device=0,
                   torch_dtype=torch.float16)
    tk = AutoTokenizer.from_pretrained("gpt2-medium")
    lm = AutoModelForCausalLM.from_pretrained("gpt2-medium").to(device).eval()

    def text(w: np.ndarray) -> str:
        return asr({"raw": w.astype(np.float32), "sampling_rate": SR}, generate_kwargs={"language": "en"})["text"].strip()

    @torch.no_grad()
    def logppl(t: str) -> float:
        ids = tk(t, return_tensors="pt").input_ids.to(device)
        return float("nan") if ids.shape[1] < 3 else float(lm(ids, labels=ids).loss)

    rows = {}
    for kind, arr, pr in (("gen", gen, gen_p), ("real", real, real_p), ("floor", floor, real_p)):
        r = []
        for g, q in zip(arr, pr):
            t = text(render(synth, coder, g, q, means, ref, device))
            r.append({"text": t, "logppl": logppl(t), "words": len(t.split()), **stats(g, q)})
        rows[kind] = r
        print(kind, json.dumps(r[:2]), flush=True)
    for i in range(min(10, len(gen))):
        for kind, arr, pr in (("gen", gen, gen_p), ("real", real, real_p)):
            g = np.concatenate([prompts[i][-50:], arr[i]])  # 5 s of context, then 15 s of continuation
            q = np.concatenate([prompt_p[i][-50:], pr[i]])
            sf.write(out / "listen" / f"{i:02d}_{kind}.wav", render(synth, coder, g, q, means, ref, device), SR)

    def med(kind: str, key: str) -> float:
        return float(np.nanmedian([x[key] for x in rows[kind]]))

    summ = {kind: {"median_logppl": med(kind, "logppl"), "words_median": med(kind, "words"),
                   "no_text_share": float(np.mean([np.isnan(x["logppl"]) for x in rows[kind]])),
                   "rep4": float(np.mean([x["rep4"] for x in rows[kind]])), "rate": med(kind, "rate"),
                   "pause": med(kind, "pause"), "f0_std": med(kind, "f0_std")} for kind in rows}
    g, r, f = (summ[k]["median_logppl"] for k in ("gen", "real", "floor"))
    summ["gap_closed"] = (f - g) / (f - r) if f != r else float("nan")
    G, R = summ["gen"], summ["real"]
    summ["L1"] = bool(summ["gap_closed"] >= 0.5 and G["rep4"] <= 2 * max(R["rep4"], 1e-3))
    summ["L2"] = bool(abs(G["rate"] / R["rate"] - 1) <= 0.15 and abs(G["pause"] - R["pause"]) <= 0.3 * R["pause"])
    summ["L3"] = bool(abs(G["f0_std"] / R["f0_std"] - 1) <= 0.3)
    summ["L"] = summ["L1"] and summ["L2"] and summ["L3"]
    summ["n"] = len(gen)
    (out / "eval.json").write_text(json.dumps({"summary": summ, "rows": rows}, indent=2, ensure_ascii=False))
    print("SUMMARY", json.dumps(summ), flush=True)


if __name__ == "__main__":
    main()
