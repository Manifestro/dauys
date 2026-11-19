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
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent))
from dy30_eval import EOS, SEED, SR, Judge, cleanup, other_index, rep4, text_refs
from dy31_eval import controls, in_vocab_share, strip_p, verdict31

S3 = os.environ.get("DY_S3", "s3://manifestro-dataset-876108973178-us-east-1-an/outputs/dauys/dy32")  # DY-33 reuses this code
HF_PREFIX = os.environ.get("DY_HF", "dy32")
CHAR = "s3://manifestro-dataset-876108973178-us-east-1-an/outputs/dauys/char-units"
W = Path("/data/dy32/a")
PACK = Path("/data/dy32/prompts")
PACKS = "s3://manifestro-dataset-876108973178-us-east-1-an/outputs/dauys/dy32"  # prompt packs stay where DY-32 made them
JC = W / "judge_cache"
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


def speech_seconds(w: np.ndarray) -> float:
    """Simple energy VAD: 20 ms frames whose RMS is above -50 dBFS and within 30 dB of the clip's 95th-percentile frame."""
    f = 320
    n = len(w) // f
    if n == 0:
        return 0.0
    db = 20 * np.log10(np.sqrt((w[: n * f].reshape(n, f).astype(np.float64) ** 2).mean(1)) + 1e-10)
    thr = max(-50.0, float(np.percentile(db, 95)) - 30.0)
    return float((db > thr).sum() * f / SR)


def derange(n: int, seed: int = 1234) -> tuple[np.ndarray, np.ndarray]:
    """j[i] != i (another prompt's generation) and k[i] not in {i, j[i]} (another piece's real), fixed seed."""
    rng = np.random.default_rng(seed)
    while True:
        j = rng.permutation(n)
        if (j != np.arange(n)).all():
            break
    k = np.array([rng.choice([x for x in range(n) if x != i and x != j[i]]) for i in range(n)])
    return j, k


def self_pairs(j, prompts: list[str], g: list[str], jj: np.ndarray, min_words: int) -> dict:
    """Dself: the model's text for prompt i vs its own text for prompt jj[i], judged against prompt i."""
    idx = [i for i in range(len(g)) if len(g[i].split()) >= max(min_words, 1) and len(g[jj[i]].split()) >= max(min_words, 1)]
    if not idx:
        return {"D": float("nan"), "ci95": float("nan"), "n": 0, "skipped": len(g), "min_words": min_words}
    r = j.pairs([prompts[i] for i in idx], [g[i] for i in idx], [g[jj[i]] for i in idx])
    return {**strip_p(r), "skipped": len(g) - len(idx), "min_words": min_words}


def transcribe(device, folders: dict[str, list[Path]]) -> dict:
    import torch
    from transformers import pipeline

    from dy30_eval import ASR
    from dy31_units import Letters, Text

    tx, L = Text(), Letters(device)
    asr = pipeline("automatic-speech-recognition", model=ASR, device=device, torch_dtype=torch.float16, chunk_length_s=30)
    out = {}
    for name, d in folders.items():
        waves = [load(f) for f in d]
        ids = [L(w) for w in waves]
        idx = [i for i, w in enumerate(waves) if len(w) >= SR // 10]
        wt = [""] * len(waves)
        for i, r in zip(idx, asr([{"raw": waves[i], "sampling_rate": SR} for i in idx], batch_size=16,
                                 generate_kwargs={"language": "en", "task": "transcribe"})):
            wt[i] = r["text"].strip()
        out[name] = {"gen": [tx(x) for x in ids], "gen_rep4_letters": float(np.mean([rep4(x) for x in ids])),
                     "whisper": wt, "seconds": [round(len(w) / SR, 2) for w in waves],
                     "speech_seconds": [round(speech_seconds(w), 2) for w in waves]}
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
    p.add_argument("--skip", default="", help="comma list of folder names left out")
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
    names = [n for n in names if n not in a.skip.split(",") and n.removesuffix("-x") not in a.skip.split(",")]
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
    folders = {n: d for n, d in folders.items() if not n.endswith("-x")}
    log("folders", list(folders))
    b = base(device)
    # extension pack (prompts_x, pieces 050-199): union controls for folders with a "<name>-x" twin
    PX = Path("/data/dy32/prompts_x")
    sh(f"aws s3 sync {PACKS}/prompts_x/ {PX}/ --only-show-errors")
    ext_ids = [i for i in range(50, 200) if (PX / f"prompt_{i:03d}.wav").exists() and (PX / f"real_{i:03d}.wav").exists()]
    twins = {}
    for n, d in folders.items():
        tw = Path(str(d) + "-x") if n != "anchor-real" else PX
        pat = "real_{:03d}.wav" if n == "anchor-real" else "{:03d}.wav"
        ids = [i for i in ext_ids if (tw / pat.format(i)).exists()]
        if ids:
            twins[n] = (tw, pat, ids)
    jobs = {n: [d / f"{i:02d}.wav" for i in range(N)] for n, d in folders.items()}
    bx = {}
    if twins:
        fx = W / "eval_base_x.json"
        bx = json.loads(fx.read_text()) if fx.exists() else {}
        if sorted(map(int, bx.get("ids", []))) != ext_ids:
            jobs["_px"] = [PX / f"prompt_{i:03d}.wav" for i in ext_ids]
            jobs["_rx"] = [PX / f"real_{i:03d}.wav" for i in ext_ids]
        for n, (tw, pat, ids) in twins.items():
            jobs[n + "@x"] = [tw / pat.format(i) for i in ids]
    log("twins", {n: len(v[2]) for n, v in twins.items()})
    # cache: transcripts and judge results of a folder are reused while its files (names, sizes) are unchanged
    JC.mkdir(parents=True, exist_ok=True)
    sig = {n: [[f.name, f.stat().st_size] for f in jobs[n] + jobs.get(n + "@x", [])] for n in folders}
    cached = {}
    for n in folders:
        f = JC / f"{n}.json"
        if f.exists():
            c = json.loads(f.read_text())
            if c.get("sig") == sig[n] and (n not in twins) == ("U" not in c):
                cached[n] = c
    log("cached (reused)", list(cached))
    tr = transcribe(device, {k: v for k, v in jobs.items() if k.split("@")[0] not in cached})
    for n, c in cached.items():
        tr[n] = c["tr"]
        if "tr_x" in c:
            tr[n + "@x"] = c["tr_x"]
    if "_px" in tr:
        bx = {"ids": ext_ids, "prompt": dict(zip(map(str, ext_ids), tr["_px"]["gen"])), "real": dict(zip(map(str, ext_ids), tr["_rx"]["gen"]))}
        (W / "eval_base_x.json").write_text(json.dumps(bx))
    prompt_t, real_t, floor_t = b["prompt"], b["real"], b["floor"]
    oi = other_index(N)
    other_t = [real_t[i] for i in oi]
    dj, dk = derange(N)
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
        if n in cached:
            D[n] = cached[n]["D"]
            continue
        g = tr[n]["gen"]
        D[n] = {"Dr": j.pairs(prompt_t, real_t, g), "Dc": j.pairs(prompt_t, g, other_t),
                "Dc_mis": j.pairs(prompt_t, [g[x] for x in dj], [real_t[x] for x in dk])}
        D[n]["Dself"] = self_pairs(j, prompt_t, g, dj, 1)
        D[n]["Dself_5w"] = self_pairs(j, prompt_t, g, dj, 5)
        log("JUDGE", n, json.dumps({k: strip_p(v) for k, v in D[n].items()}))
    U = {}
    for n, (tw, pat, ids) in twins.items():
        if n in cached:
            U[n] = cached[n]["U"]
            continue
        pu = b["prompt"] + [bx["prompt"][str(i)] for i in ids]
        ru = b["real"] + [bx["real"][str(i)] for i in ids]
        gu = tr[n]["gen"] + tr[n + "@x"]["gen"]
        uj, uk = derange(len(gu))
        U[n] = {"n": len(gu), "Dc_mis": strip_p(j.pairs(pu, [gu[x] for x in uj], [ru[x] for x in uk])),
                "Dself": self_pairs(j, pu, gu, uj, 1), "Dself_5w": self_pairs(j, pu, gu, uj, 5),
                "speech_seconds_median": float(np.median(tr[n]["speech_seconds"] + tr[n + "@x"]["speech_seconds"])),
                "words_median": float(np.median([len(t.split()) for t in gu])),
                "whisper_words_median": float(np.median([len(t.split()) for t in tr[n]["whisper"] + tr[n + "@x"]["whisper"]])),
                "empty_share": float(np.mean([not t for t in gu]))}
        log("UNION", n, json.dumps(U[n]))
    for n in folders:
        if n not in cached:
            c = {"sig": sig[n], "tr": tr[n], "D": D[n]}
            if n in twins:
                c.update({"tr_x": tr[n + "@x"], "U": U[n]})
            (JC / f"{n}.json").write_text(json.dumps(c))
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
             "speech_seconds_median": float(np.median(tr[n]["speech_seconds"])),
             "whisper_words_median": float(np.median([len(t.split()) for t in tr[n]["whisper"]])),
             "whisper_words_in_heldout_vocab": in_vocab_share([" ".join(x for x in t.lower().split()) for t in tr[n]["whisper"]], vocab),
             "info": info}
        if n in U:
            s["union"] = U[n]
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
         "Added controls (same judge/template/both orders; derangement seed 1234): Dc_mis = another prompt's generation vs a third "
         "piece's real, judged against prompt i (≈ Dc means the judge prefers the model's style, not its relation to the prompt; "
         "≈0.5 means Dc reflects the prompt). Dself = the model's text for prompt i vs its own text for another prompt (style "
         "identical on both sides; > 0.5 = relation to the prompt); pairs with an empty text skipped (n = pairs used); Dself5 = "
         "only pairs with >= 5 words on both sides. Speech s = energy-VAD seconds of speech in the continuation.", "",
         "| folder | Dc | Dr | reading | Dc_mis | Dself (n) | Dself5 (n) | in-vocab | GPT-2 logppl | rep4 words | words median (letters / Whisper) | empty | speech s / gen s |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for n, s in summary["folders"].items():
        jd = s["judge"]
        L.append(f"| {n} | {jd['Dc']['D']:.2f} ± {jd['Dc']['ci95']:.2f} | {jd['Dr']['D']:.2f} | {s['verdict']['verdict']} | "
                 f"{jd['Dc_mis']['D']:.2f} | {jd['Dself']['D']:.2f} ± {jd['Dself']['ci95']:.2f} ({jd['Dself']['n']}) | "
                 f"{jd['Dself_5w']['D']:.2f} ({jd['Dself_5w']['n']}) | "
                 f"{s['words_in_heldout_vocab']:.3f} | {s['gpt2'].get('median_logppl', float('nan')):.2f} | {s['rep4_words']:.3f} | "
                 f"{s['words_median']:.0f} / {s['whisper_words_median']:.0f} | {s['empty_share']:.2f} | "
                 f"{s['speech_seconds_median']:.1f} / {s['gen_seconds_median']:.1f} |")
    if U:
        L += ["", "## Union controls (first 50 + extension pieces 050-199 where the track wrote `<name>-x`; Dc/Dr stay on the first 50)", "",
              "| folder | n | Dc_mis | Dself ± 95% (n used, skipped) | Dself5 ± 95% (n used) | words median (letters / Whisper) | empty | speech s |",
              "|---|---|---|---|---|---|---|---|"]
        for n, u in U.items():
            L.append(f"| {n} | {u['n']} | {u['Dc_mis']['D']:.2f} ± {u['Dc_mis']['ci95']:.2f} | {u['Dself']['D']:.2f} ± {u['Dself']['ci95']:.2f} "
                     f"({u['Dself']['n']}, {u['Dself']['skipped']}) | {u['Dself_5w']['D']:.2f} ± {u['Dself_5w']['ci95']:.2f} ({u['Dself_5w']['n']}) | "
                     f"{u['words_median']:.0f} / {u['whisper_words_median']:.0f} | {u['empty_share']:.2f} | {u['speech_seconds_median']:.1f} |")
    pairs_ = [("idea 2 downstream (one voice vs corpus codes)", "g-lj", "b"), ("idea 1 downstream (several streams vs one)", "c", "b"),
              ("idea 3 (continuous vs discrete input)", "d-cont", "d-disc"), ("codebook size (8000 vs 2000)", "h-k8000", "b")]
    pairs_ += [(f"{t} ({suf[1:]})", x + suf, y + suf) for suf in ("-early", "-t10") for t, x, y in pairs_]
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
        api.upload_folder(folder_path=str(E), path_in_repo=f"{HF_PREFIX}/eval/{a.round}", repo_id="manifestro/dauys-runs")
        log("HF uploaded", f"{HF_PREFIX}/eval/{a.round}")
    log("EVAL DONE", a.round)


if __name__ == "__main__":
    main()
