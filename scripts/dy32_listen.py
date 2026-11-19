"""DY-32 track A: listening set for the owner (phone-friendly mp3, 16 kHz mono).

gen/<name>/   -> <name>__XX.mp3 for prompts 00-05: last 5 s of prompt_XX + 0.3 s beep + the model's continuation.
resynth/<name>/ (other tracks) and the channel-lab candidates -> <name>__XX.mp3 for pieces 00-02: the original
               real_XX + 0.3 s beep + the same piece through that channel.
listen.md: one line per folder (what it is) and, for gen folders, the texts of the 6 prompts (prompt, model, real) from
the evaluation round. Uploaded to S3 dy32/eval/listen/ (and HF dy32/eval/listen/ with --hf).

    python scripts/dy32_listen.py --round early [--hf]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent))
from dy32_channel import S3, SR, W, sh
from dy32_eval import PACK, load

N_GEN, N_RES = 6, 3


def beep() -> np.ndarray:
    t = np.arange(int(0.3 * SR)) / SR
    return (0.3 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)


def pk(x: np.ndarray) -> np.ndarray:
    return (0.9 * x / max(float(np.abs(x).max()), 1e-6)).astype(np.float32) if len(x) else x


def save(f: Path, w: np.ndarray) -> None:
    try:
        sf.write(f.with_suffix(".mp3"), w, SR, format="MP3")
    except Exception:
        sf.write(f.with_suffix(".wav"), w, SR, subtype="PCM_16")


def complete(d: Path, n: int) -> bool:
    return all((d / f"{i:02d}.wav").exists() for i in range(n))


def describe(d: Path) -> str:
    f = d / "info.json"
    if not f.exists():
        return "(no info.json)"
    try:
        j = json.loads(f.read_text())
    except Exception:
        return "(info.json unreadable)"
    keys = ["track", "units", "decoder", "step", "training_step", "val_loss", "validation_loss", "prompt_seen_s", "sampling", "desc", "description"]
    got = {k: j[k] for k in keys if k in j}
    return json.dumps(got or j, ensure_ascii=False)[:600]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--round", default="early")
    p.add_argument("--hf", action="store_true")
    a = p.parse_args()
    out = W / "listen" / a.round
    out.mkdir(parents=True, exist_ok=True)
    G, X = W / "gen", W / "ext_resynth"
    sh(f"aws s3 sync {S3}/gen/ {G}/ --only-show-errors --exclude '*.pt'")
    sh(f"aws s3 sync {S3}/resynth/ {X}/ --only-show-errors")
    b = beep()
    L = [f"# DY-32 listening set ({a.round}, {time.strftime('%Y-%m-%d %H:%M')} UTC)", "",
         "Continuations: `<name>__XX` = last 5 s of the prompt, beep, then the model (prompts 00-05, same for every folder). "
         "Channels: `<name>__XX` = the original 10 s piece, beep, then the same piece through that channel (pieces 00-02). "
         "Texts below come from the letters recogniser on audio.", "", "## Continuations (gen/)", ""]
    tab = json.loads((W / "channel.json").read_text()) if (W / "channel.json").exists() else {}
    for d in sorted(x for x in G.iterdir() if x.is_dir() and not x.name.endswith("-x")) if G.exists() else []:
        if not complete(d, N_GEN):
            L += [f"- **{d.name}**: incomplete, skipped", ""]
            continue
        for i in range(N_GEN):
            tail = load(PACK / f"prompt_{i:02d}.wav")[-5 * SR :]
            save(out / f"{d.name}__{i:02d}", np.concatenate([pk(tail), b, pk(load(d / f"{i:02d}.wav"))]))
        L += [f"### {d.name}", "", f"What: {describe(d)}", ""]
        ev = W / "eval" / a.round / d.name / "eval.json"
        if ev.exists():
            e = json.loads(ev.read_text())
            s = e["summary"]
            jd = s["judge"]
            L += [f"Dc {jd['Dc']['D']}, Dr {jd['Dr']['D']} ({s['verdict']['verdict']}); controls: Dc_mis {jd['Dc_mis']['D']}, "
                  f"Dself {jd['Dself']['D']} (n {jd['Dself']['n']}), Dself5 {jd['Dself_5w']['D']} (n {jd['Dself_5w']['n']}); "
                  f"speech {s['speech_seconds_median']:.1f} s of {s['gen_seconds_median']:.1f} s; words {s['words_median']:.0f} "
                  f"(Whisper {s['whisper_words_median']:.0f}); in held-out vocabulary {s['words_in_heldout_vocab']:.3f}", ""]
            if "union" in s:
                u = s["union"]
                L += [f"Union ({u['n']} prompts): Dc_mis {u['Dc_mis']['D']}, Dself {u['Dself']['D']} ± {u['Dself']['ci95']} "
                      f"(n {u['Dself']['n']}), Dself5 {u['Dself_5w']['D']} (n {u['Dself_5w']['n']}), speech {u['speech_seconds_median']:.1f} s", ""]
            for r in e["rows"][:N_GEN]:
                L += [f"- {r['id']} prompt: {r['prompt'][-300:]}", f"  - model: {r['gen']}", f"  - real: {r['real']}"]
            L.append("")
        else:
            L += ["(not evaluated in this round)", ""]
    L += ["## Channels (real piece through units + decoder, no model)", ""]
    folders = [(x.name, x) for x in sorted(X.iterdir()) if x.is_dir()] if X.exists() else []
    folders += [(x.name, x) for x in sorted((W / "resynth").iterdir()) if x.is_dir()] if (W / "resynth").exists() else []
    for name, d in folders:
        if not complete(d, N_RES):
            L.append(f"- **{name}**: incomplete, skipped")
            continue
        for i in range(N_RES):
            save(out / f"{name}__{i:02d}", np.concatenate([pk(load(PACK / f"real_{i:02d}.wav")), b, pk(load(d / f"{i:02d}.wav"))]))
        row = tab.get(name) or tab.get(f"trk-{name}") or {}
        what = row.get("desc") or describe(d)
        wer = f"letters WER median {row['letters_wer']['median']:.3f}, Whisper {row['whisper_wer']['median']:.3f}" if row else "not measured yet"
        L.append(f"- **{name}** ({'track output' if d.parent == X else 'track A channel lab'}): {what}; {wer}")
    (out / "listen.md").write_text("\n".join(L) + "\n")
    sh(f"aws s3 sync {out}/ {S3}/eval/{a.round}/listen/ --only-show-errors")
    sh(f"aws s3 sync {out}/ {S3}/eval/listen/ --only-show-errors")
    if a.hf:
        import os

        from huggingface_hub import HfApi

        api = HfApi(token=os.environ["HF_TOKEN"])
        assert api.repo_info("manifestro/dauys-runs").private, "not private"
        api.upload_folder(folder_path=str(out), path_in_repo="dy32/eval/listen", repo_id="manifestro/dauys-runs")
    print("LISTEN DONE", out, len(list(out.glob("*.mp3"))), "mp3", flush=True)


if __name__ == "__main__":
    main()
