"""DY-32 track A, job 3: one evaluation for every continuation folder gen/<name>/XX.wav (50 prompts of the pack).

As dy31_eval, but every text comes from audio through the same recogniser (letters CTC of dy31, greedy):
prompt text = recogniser on prompt_XX.wav, real text = on real_XX.wav, gen text = on the continuation XX.wav.
  Judge (dy30_eval.Judge, Qwen2.5-7B-Instruct, both orders): Dr = D(real vs gen); Dc = D(gen vs real of ANOTHER
  piece, dy30_eval.other_index). Controls on real text (dy31_eval.controls): (a) real vs another real, (b) real vs
  its own 4-word chunks shuffled, (c) real vs floor (letters drawn from the unigram of the held-out letter tokens).
  Reading (pre-registered): Dc >= 0.70 and Dr <= 0.85 = meaning; Dc <= 0.55 = no meaning; else partial.
  Also: share of gen words in the held-out vocabulary (vocabulary of the held-out letter tokens, as dy31; bar 0.80),
  GPT-2-medium log-perplexity (dy30_eval.text_refs), 4-gram repetition on words and on letters, Whisper text
  (second opinion; only reported). Idea 3: gen/d-cont beats gen/d-disc on Dc by >= 0.05.
Per folder: eval.json, pairs.md (20 pairs), listen/XX.wav for the first 10 prompts (last 5 s of the prompt, 0.3 s
beep, the continuation). summary.md / summary.json over all folders.

    python scripts/dy32_eval.py --round early      # every complete folder in S3 gen/
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent))
from dy30_eval import EOS, SEED, SR, Judge, cleanup, other_index, rep4, text_refs
from dy31_eval import controls, in_vocab_share, strip_p, verdict31

S3 = "s3://manifestro-dataset-876108973178-us-east-1-an/outputs/dauys/dy32"
CHAR = "s3://manifestro-dataset-876108973178-us-east-1-an/outputs/dauys/char-units"
W = Path("/data/dy32/a")
PACK = Path("/data/dy32/prompts")
N, N_SHOW, N_LISTEN = 50, 20, 10
VOCAB_BAR, IDEA3_GAP = 0.80, 0.05


def sh(cmd: str) -> str:
    return subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True).stdout


def log(*a) -> None:
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def load(f: Path) -> np.ndarray:
    import torch
    import torchaudio

    w, sr = sf.read(f, dtype="float32")
    if w.ndim > 1:
        w = w.mean(1)
    if sr != SR:
        w = torchaudio.functional.resample(torch.from_numpy(w), sr, SR).numpy()
    return w


def word_rep4(t: str) -> float:
    w = t.split()
    g = [tuple(w[i : i + 4]) for i in range(len(w) - 3)]
    return 1 - len(set(g)) / len(g) if g else 0.0


def base(device) -> dict:
    """Recogniser texts of the pack, the held-out vocabulary and the floor (cached)."""
    f = W / "eval_base.json"
    if f.exists():
        return json.loads(f.read_text())
    from dy21_train import load_stream
    from dy31_units import Letters, Text

    man = json.loads((PACK / "manifest.json").read_text())
    tx, L = Text(), Letters(device)
    P = [L(load(PACK / f"prompt_{i:02d}.wav")) for i in range(N)]
    R = [L(load(PACK / f"real_{i:02d}.wav")) for i in range(N)]
    del L
    cleanup()
    cu = W / "char-units"
    cu.mkdir(parents=True, exist_ok=True)
    for s in man["held_out_shards"]:
        if not (cu / f"{s}.npz").exists():
            sh(f"aws s3 cp {CHAR}/{s}.npz {cu}/{s}.npz --only-show-errors")
    val = load_stream([cu / f"{s}.npz" for s in man["held_out_shards"]])
    counts = np.bincount(val[val != EOS].astype(np.int64)).astype(np.float64)
    rng = np.random.default_rng(SEED)
    floor = [rng.choice(len(counts), size=len(r), p=counts / counts.sum()) for r in R]
    vocab = sorted(set(tx(np.where(val == EOS, tx.delim_id, val)).split()))
    b = {"prompt": [tx(x) for x in P], "real": [tx(x) for x in R], "floor": [tx(x) for x in floor],
         "real_letters_rep4": float(np.mean([rep4(x) for x in R])), "vocab": vocab, "text": tx.info}
    f.write_text(json.dumps(b))
    return b


def transcribe(device, folders: dict[str, Path]) -> dict:
    import torch
    from transformers import pipeline

    from dy30_eval import ASR
    from dy31_units import Letters, Text

    tx, L = Text(), Letters(device)
    asr = pipeline("automatic-speech-recognition", model=ASR, device=device, torch_dtype=torch.float16, chunk_length_s=30)
    out = {}
    for name, d in folders.items():
        waves = [load(d / f"{i:02d}.wav") for i in range(N)]
        ids = [L(w) for w in waves]
        idx = [i for i, w in enumerate(waves) if len(w) >= SR // 10]
        wt = [""] * N
        for i, r in zip(idx, asr([{"raw": waves[i], "sampling_rate": SR} for i in idx], batch_size=16,
                                 generate_kwargs={"language": "en", "task": "transcribe"})):
            wt[i] = r["text"].strip()
        out[name] = {"gen": [tx(x) for x in ids], "gen_rep4_letters": float(np.mean([rep4(x) for x in ids])),
                     "whisper": wt, "seconds": [round(len(w) / SR, 2) for w in waves]}
        log("transcribed", name)
    del L, asr
    cleanup()
    return out


def listen(name: str, d: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    t = np.arange(int(0.3 * SR)) / SR
    beep = (0.3 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)
    pk = lambda x: 0.9 * x / max(float(np.abs(x).max()), 1e-6) if len(x) else x
    for i in range(N_LISTEN):
        p = load(PACK / f"prompt_{i:02d}.wav")[-5 * SR :]
        g = load(d / f"{i:02d}.wav")
        sf.write(out / f"{i:02d}_prompt5s_beep_{name}.wav", np.concatenate([pk(p), beep, pk(g)]), SR, subtype="PCM_16")


def main() -> None:
    import torch

    p = argparse.ArgumentParser()
    p.add_argument("--round", default="early")
    p.add_argument("--only", default="", help="comma list of folder names (default: all complete folders in S3 gen/)")
    p.add_argument("--extra", nargs="*", default=[], help="name=folder of XX.wav evaluated as well (e.g. an anchor)")
    p.add_argument("--hf", action="store_true")
    a = p.parse_args()
    device = torch.device("cuda")
    E = W / "eval" / a.round
    G = W / "gen"
    E.mkdir(parents=True, exist_ok=True)
    G.mkdir(parents=True, exist_ok=True)
    sh(f"aws s3 sync {S3}/gen/ {G}/ --only-show-errors --exclude '*.pt'")
    names = sorted(x.name for x in G.iterdir() if x.is_dir())
    if a.only:
        names = [n for n in names if n in a.only.split(",")]
    folders = {}
    for n in names:
        miss = [i for i in range(N) if not (G / n / f"{i:02d}.wav").exists()]
        if miss:
            log("SKIP incomplete", n, "missing", len(miss))
        else:
            folders[n] = G / n
    for spec in a.extra:
        n, d = spec.split("=", 1)
        folders[n] = Path(d)
    log("folders", list(folders))
    b = base(device)
    tr = transcribe(device, folders)
    prompt_t, real_t, floor_t = b["prompt"], b["real"], b["floor"]
    oi = other_index(N)
    other_t = [real_t[i] for i in oi]
    vocab = set(b["vocab"])

    j = Judge(device)
    cf = W / "eval_controls.json"
    if cf.exists():
        ctl = json.loads(cf.read_text())
    else:
        ctl, _ = controls(j, prompt_t, real_t, floor_t)
        cf.write_text(json.dumps(ctl))
    D = {}
    for n in folders:
        g = tr[n]["gen"]
        D[n] = {"Dr": j.pairs(prompt_t, real_t, g), "Dc": j.pairs(prompt_t, g, other_t)}
        log("JUDGE", n, json.dumps({k: strip_p(v) for k, v in D[n].items()}))
    del j
    cleanup()
    refs = text_refs(device, prompt_t, {"real": real_t, "floor": floor_t, **{n: tr[n]["gen"] for n in folders}}) if folders else {"summary": {}, "logppl": {}}

    summary = {"round": a.round, "time_utc": time.strftime("%Y-%m-%d %H:%M"), "n": N,
               "controls": {k: strip_p(v) for k, v in ctl.items()},
               "real": {"words_in_heldout_vocab": in_vocab_share(real_t, vocab), "rep4_words": float(np.mean([word_rep4(t) for t in real_t])),
                        "rep4_letters": b["real_letters_rep4"], **refs["summary"].get("real", {})},
               "floor": refs["summary"].get("floor", {}), "folders": {}}
    for n, d in folders.items():
        g = tr[n]["gen"]
        info = {}
        if (d / "info.json").exists():
            try:
                info = json.loads((d / "info.json").read_text())
            except Exception:
                info = {"error": "info.json unreadable"}
        v = verdict31(D[n]["Dc"]["D"], D[n]["Dr"]["D"])
        s = {"verdict": v, "judge": {k: strip_p(x) for k, x in D[n].items()},
             "words_in_heldout_vocab": in_vocab_share(g, vocab), "continues_speech_bar": VOCAB_BAR,
             "continues_speech": bool(in_vocab_share(g, vocab) >= VOCAB_BAR),
             "words_median": float(np.median([len(t.split()) for t in g])), "empty_share": float(np.mean([not t for t in g])),
             "rep4_words": float(np.mean([word_rep4(t) for t in g])), "rep4_letters": tr[n]["gen_rep4_letters"],
             "gpt2": refs["summary"].get(n, {}), "gen_seconds_median": float(np.median(tr[n]["seconds"])),
             "whisper_words_in_heldout_vocab": in_vocab_share([" ".join(x for x in t.lower().split()) for t in tr[n]["whisper"]], vocab),
             "info": info}
        summary["folders"][n] = s
        out = E / n
        out.mkdir(parents=True, exist_ok=True)
        rows = [{"id": f"{i:02d}", "prompt": prompt_t[i], "real": real_t[i], "gen": g[i], "gen_whisper": tr[n]["whisper"][i],
                 "other_real": other_t[i], "p_Dr": D[n]["Dr"]["p_real"][i], "p_Dc": D[n]["Dc"]["p_real"][i]} for i in range(N)]
        (out / "eval.json").write_text(json.dumps({"summary": s, "controls": summary["controls"], "rows": rows}, indent=2, ensure_ascii=False))
        L = [f"# DY-32 eval ({a.round}): {n}", "",
             f"Dc {D[n]['Dc']['D']} ± {D[n]['Dc']['ci95']} (gen vs another piece's real; 0.5 = not tied to the prompt); "
             f"Dr {D[n]['Dr']['D']} ± {D[n]['Dr']['ci95']} (real vs gen; 0.5 = indistinguishable). Reading: {v['verdict']}. "
             f"Words in held-out vocabulary {s['words_in_heldout_vocab']:.3f} (bar {VOCAB_BAR}). All texts: letters recogniser on audio.", ""]
        for r in rows[:N_SHOW]:
            L += [f"## {r['id']}", "", f"**Prompt (15 s):** {r['prompt']}", "", f"**Person said next (10 s):** {r['real']}", "",
                  f"**Model:** {r['gen']}", "", f"*Whisper on the model audio:* {r['gen_whisper']}", "",
                  f"_Judge: real over model {r['p_Dr']:.2f}; model over another piece's real {r['p_Dc']:.2f}_", ""]
        (out / "pairs.md").write_text("\n".join(L))
        listen(n, d, out / "listen")

    if "d-cont" in summary["folders"] and "d-disc" in summary["folders"]:
        gap = summary["folders"]["d-cont"]["judge"]["Dc"]["D"] - summary["folders"]["d-disc"]["judge"]["Dc"]["D"]
        summary["idea3"] = {"Dc_gap_cont_minus_disc": round(gap, 4), "threshold": IDEA3_GAP, "confirmed": bool(gap >= IDEA3_GAP)}
    (E / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    c = summary["controls"]
    L = [f"# DY-32 evaluation, round {a.round} ({summary['time_utc']} UTC)", "",
         "50 held-out prompts (15 s); texts from the letters recogniser on audio; judge Qwen2.5-7B-Instruct, both orders.",
         "Reading: Dc >= 0.70 and Dr <= 0.85 = meaning; Dc <= 0.55 = no meaning; else partial. 'Continues speech' bar: "
         f"words in held-out vocabulary >= {VOCAB_BAR}.", "",
         f"Judge controls on real text: (a) real vs another real {c['a_other_real']['D']}, (b) real vs shuffled chunks "
         f"{c['b_shuffled_chunks']['D']}, (c) real vs floor {c['c_floor']['D']}.", "",
         f"Real continuation: in-vocab {summary['real']['words_in_heldout_vocab']:.3f}, GPT-2 logppl "
         f"{summary['real'].get('median_logppl', float('nan')):.2f}, rep4 words {summary['real']['rep4_words']:.3f}. "
         f"Floor: GPT-2 logppl {summary['floor'].get('median_logppl', float('nan')):.2f}.", "",
         "| folder | Dc | Dr | reading | in-vocab | GPT-2 logppl | rep4 words | words median | empty | gen s |", "|---|---|---|---|---|---|---|---|---|---|"]
    for n, s in summary["folders"].items():
        L.append(f"| {n} | {s['judge']['Dc']['D']:.2f} ± {s['judge']['Dc']['ci95']:.2f} | {s['judge']['Dr']['D']:.2f} | {s['verdict']['verdict']} | "
                 f"{s['words_in_heldout_vocab']:.3f} | {s['gpt2'].get('median_logppl', float('nan')):.2f} | {s['rep4_words']:.3f} | "
                 f"{s['words_median']:.0f} | {s['empty_share']:.2f} | {s['gen_seconds_median']:.1f} |")
    pairs_ = [("idea 2 downstream (one voice vs corpus codes)", "g-lj", "b"), ("idea 1 downstream (several streams vs one)", "c", "b"),
              ("idea 3 (continuous vs discrete input)", "d-cont", "d-disc"), ("codebook size (8000 vs 2000)", "h-k8000", "b")]
    pairs_ += [(f"{t} (early)", x + "-early", y + "-early") for t, x, y in pairs_]
    fo = summary["folders"]
    cmp_ = []
    for title, x, y in pairs_:
        if x in fo and y in fo:
            g = fo[x]["judge"]["Dc"]["D"] - fo[y]["judge"]["Dc"]["D"]
            cmp_.append({"pair": title, "x": x, "y": y, "Dc_x": fo[x]["judge"]["Dc"]["D"], "Dc_y": fo[y]["judge"]["Dc"]["D"],
                         "Dc_gap": round(g, 4), "invocab_x": fo[x]["words_in_heldout_vocab"], "invocab_y": fo[y]["words_in_heldout_vocab"]})
    others = [n for n in fo if not n.startswith("e-") and n != "anchor-real"]
    for e in [n for n in fo if n.startswith("e-")]:
        for o in others:
            cmp_.append({"pair": "GSLM2 vs other", "x": e, "y": o, "Dc_x": fo[e]["judge"]["Dc"]["D"], "Dc_y": fo[o]["judge"]["Dc"]["D"],
                         "Dc_gap": round(fo[e]["judge"]["Dc"]["D"] - fo[o]["judge"]["Dc"]["D"], 4),
                         "invocab_x": fo[e]["words_in_heldout_vocab"], "invocab_y": fo[o]["words_in_heldout_vocab"]})
    summary["comparisons"] = cmp_
    (E / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    if cmp_:
        L += ["", "## Paired comparisons (Dc; the 95 % CI of one Dc on 50 pairs is about ±0.13, so gaps below ~0.1 are within noise)", "",
              "| comparison | x | y | Dc x | Dc y | gap x-y | in-vocab x | in-vocab y |", "|---|---|---|---|---|---|---|---|"]
        for c in cmp_:
            L.append(f"| {c['pair']} | {c['x']} | {c['y']} | {c['Dc_x']:.2f} | {c['Dc_y']:.2f} | {c['Dc_gap']:+.2f} | "
                     f"{c['invocab_x']:.3f} | {c['invocab_y']:.3f} |")
    if "idea3" in summary:
        i3 = summary["idea3"]
        L += ["", f"Idea 3 (d-cont Dc - d-disc Dc >= {IDEA3_GAP}): gap {i3['Dc_gap_cont_minus_disc']:+.3f} -> "
              f"{'CONFIRMED' if i3['confirmed'] else 'not confirmed'}"]
    (E / "summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L), flush=True)
    sh(f"aws s3 sync {E}/ {S3}/eval/{a.round}/ --only-show-errors")
    if a.hf:
        import os

        from huggingface_hub import HfApi

        api = HfApi(token=os.environ["HF_TOKEN"])
        assert api.repo_info("manifestro/dauys-runs").private, "not private"
        api.upload_folder(folder_path=str(E), path_in_repo=f"dy32/eval/{a.round}", repo_id="manifestro/dauys-runs")
        log("HF uploaded", f"dy32/eval/{a.round}")
    log("EVAL DONE", a.round)


if __name__ == "__main__":
    main()
