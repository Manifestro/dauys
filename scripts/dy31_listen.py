"""DY-31 interim listening samples from a checkpoint copy, while training runs (not a result, no verdict).

Pieces: the final eval's selection (held-out pieces >= 45 s, rng 1234 over the validation shards of the run), kept
if >= 60 s, one per shard first. Prompt = audio 0-30 s -> letters (dy31_units.Letters, on the part as in eval);
the model continues EOS + prompt with the eval sampler (temperature 0.8, top-p 0.95, EOS banned, seed 1234 + i)
for about --gen-s seconds of letters at the prompt's own letter rate, never past the context. Real continuation =
letters of 30 s + the same duration. Texts via dy31_units.Text. Audio per sample: the last 10 s of the real prompt,
a 0.3 s beep, then the model text read by Piper through its normal text path (a separate Python with piper-tts).
The GPU memory of this process is capped (--gpu-gb) so a running training job keeps its memory.

    nice -n 19 python scripts/dy31_listen.py --ckpt ckpt.pt --meta metadata.json --piper-python pvenv/bin/python --out out
"""

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from dy30_eval import EOS, MIN_S, N_EVAL, SEED, SR, TEMP, TOP_P, candidates, cleanup, generate, parts, read_pieces

PIPER_REPO = "rhasspy/piper-voices"
VOICE = "en/en_US/lessac/medium/en_US-lessac-medium"
PROMPT_S, TAIL_S, BEEP_S, LONG_S = 30, 10, 0.3, 60


def pick(cands: list, n: int) -> list:
    """The eval's selection order, pieces >= LONG_S only, one per shard first, then the rest."""
    rng = np.random.default_rng(SEED)
    order = rng.choice(len(cands), size=min(N_EVAL, len(cands)), replace=False)
    long = [cands[i] for i in order if cands[i][3] >= LONG_S * SR]
    first, seen = [], set()
    for c in long:
        if c[0] not in seen:
            first.append(c)
            seen.add(c[0])
    rest = [c for c in long if c not in first]
    return (first + rest)[:n]


def beep() -> np.ndarray:
    t = np.arange(int(BEEP_S * SR)) / SR
    return (0.3 * np.sin(2 * np.pi * 880 * t) * np.hanning(len(t))).astype(np.float32)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--meta", required=True, help="the run's metadata.json (validation shard names)")
    p.add_argument("--piper-python", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=6)
    p.add_argument("--gen-s", type=float, default=30.0)
    p.add_argument("--gpu-gb", type=float, default=8.0)
    p.add_argument("--kbps", type=int, default=64)
    a = p.parse_args()
    import torch
    from huggingface_hub import hf_hub_download
    from scipy.signal import resample_poly

    from dauys.dy21_lm import LMConfig, UnitLM
    from dy21_train import held_out
    from dy31_units import Letters, Text

    device = torch.device("cuda")
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.0, a.gpu_gb * 2**30 / total), 0)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(dir=out))
    val_names = json.loads(Path(a.meta).read_text())["val_files_names"]
    assert val_names and all(held_out(s) for s in val_names), val_names
    sel = pick(candidates(val_names, MIN_S, tmp), a.n)
    assert all(held_out(s) for s, *_ in sel)
    audio = read_pieces(sel, tmp)

    ph = Letters(device)
    P = [ph(parts(w)[0]) for w in audio]
    rates = [len(x) / PROMPT_S for x in P]
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    ctx = LMConfig(**ck["cfg"]).ctx
    n_gen = [max(0, min(int(round(r * a.gen_s)), ctx - 1 - len(x))) for r, x in zip(rates, P)]
    real_s = [n / max(r, 1e-6) for n, r in zip(n_gen, rates)]
    R = [ph(w[PROMPT_S * SR : int((PROMPT_S + s) * SR)]) for w, s in zip(audio, real_s)]
    del ph
    cleanup()

    model = UnitLM(LMConfig(**ck["cfg"])).to(device).eval()
    model.load_state_dict(ck["model"])
    step, val = ck.get("step"), ck.get("val", {})
    del ck
    t0 = time.time()
    G = [generate(model, P[i], n_gen[i], TEMP, SEED + i, device) for i in range(len(P))]
    gen_s = round(time.time() - t0)
    peak = torch.cuda.max_memory_allocated() / 2**30
    del model
    cleanup()
    tx = Text()
    prompt_t, gen_t, real_t = [tx(x) for x in P], [tx(x) for x in G], [tx(x) for x in R]

    onnx = hf_hub_download(PIPER_REPO, f"{VOICE}.onnx")
    hf_hub_download(PIPER_REPO, f"{VOICE}.onnx.json")
    try:
        card = Path(hf_hub_download(PIPER_REPO, str(Path(VOICE).parent / "MODEL_CARD"))).read_text()
    except Exception as e:  # the voice still works without the card; say so in the notes
        card = f"(MODEL_CARD not found: {e})"
    has_ffmpeg = shutil.which("ffmpeg") is not None
    files = []
    for i, ((s, pc, st, ln), w) in enumerate(zip(sel, audio)):
        tail = parts(w)[0][-TAIL_S * SR :].astype(np.float32) / 32768.0
        speech = np.zeros(0, np.float32)
        if gen_t[i]:
            raw = tmp / f"piper_{i}.wav"
            subprocess.run([a.piper_python, "-m", "piper", "-m", onnx, "-f", str(raw)], input=gen_t[i].encode(),
                           check=True, capture_output=True)
            import soundfile as sf

            x, sr = sf.read(raw, dtype="float32", always_2d=True)
            speech = resample_poly(x.mean(1), SR, sr).astype(np.float32)
        pk = lambda v: 0.9 * v / max(float(np.abs(v).max()), 1e-6) if len(v) else v
        y = np.concatenate([pk(tail), beep(), pk(speech)])
        import soundfile as sf

        name = f"{i + 1:02d}_{s}_p{pc}"
        wav = out / f"{name}.wav"
        sf.write(wav, y, SR, subtype="PCM_16")
        if has_ffmpeg:
            mp3 = out / f"{name}.mp3"
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(wav), "-ac", "1", "-b:a", f"{a.kbps}k", str(mp3)], check=True)
            wav.unlink()
            f = mp3
        else:
            f = wav
        files.append({"file": f.name, "seconds": round(len(y) / SR, 1), "model_speech_s": round(len(speech) / SR, 1)})

    lines = ["# DY-31 interim listening samples (not a result)", "",
             f"Checkpoint: step {step}, val CE {val.get('ce')} (copy of best.pt taken while training runs).",
             f"Sampling: temperature {TEMP}, top-p {TOP_P}, EOS banned, seed {SEED} + sample index; continuation "
             f"length about {a.gen_s:g} s of letters at the prompt's letter rate, capped by the {ctx}-token context.",
             f"Each file: last {TAIL_S} s of the real prompt audio, a {BEEP_S} s beep, then the model's text read by "
             f"Piper voice {Path(VOICE).name} (rhasspy/piper-voices, normal text input).",
             "Pieces: held-out shards only (the final eval's selection, pieces >= 60 s, one per shard first).", "",
             "Voice model card (licence):", "", "```", card.strip(), "```", ""]
    for i, (s, pc, st, ln) in enumerate(sel):
        lines += [f"## {i + 1}. {files[i]['file']} ({files[i]['seconds']} s; {s}, piece {pc})", "",
                  f"**Prompt (0–30 s, letters):** {prompt_t[i]}", "",
                  f"**Model continuation ({n_gen[i]} letters):** {gen_t[i]}", "",
                  f"**Real continuation (30–{PROMPT_S + real_s[i]:.0f} s, letters):** {real_t[i]}", ""]
    (out / "samples.md").write_text("\n".join(lines))
    info = {"step": step, "val": val, "files": files, "pieces": [list(x) for x in sel], "n_gen": n_gen,
            "prompt_tokens": [len(x) for x in P], "letters_per_s": [round(r, 2) for r in rates], "gen_seconds": gen_s,
            "gpu_peak_gb": round(peak, 2), "ffmpeg": has_ffmpeg, "voice": VOICE}
    (out / "info.json").write_text(json.dumps(info, indent=2))
    shutil.rmtree(tmp, ignore_errors=True)
    print("LISTEN DONE", json.dumps(info), flush=True)


if __name__ == "__main__":
    main()
