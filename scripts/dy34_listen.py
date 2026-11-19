"""DY-34 / DY-33 listening set chosen by how simple the PROMPT is (never by the model's answers).

Rule, on the 200 test prompts, using only the recognised (letters) text of the prompt and of the real continuation:
keep prompts with >= 25 recognised words in the 15 s and an English, intelligible transcript (proxy: >= 95 % of the
prompt's words are in the held-out vocabulary of the letter tokens); rank by (a) lowest words per second, then
(b) highest share of words among the 1,000 most frequent words of the 200 real-continuation transcripts, then
(c) shortest mean word length; take the top 8.

select: writes the ids (before any model output is looked at).
build:  for each chosen prompt: <folder>__XXX.mp3 = last 5 s of the prompt + 0.3 s beep + 10 s of the model, for every
        given folder, and real__XXX.mp3 = the same with the real continuation; listen.md with the full prompt, the real
        continuation and each folder's transcript.

    python scripts/dy34_listen.py select --out /data/dy34/eval/listen_ids.json
    python scripts/dy34_listen.py build --ids /data/dy34/eval/listen_ids.json --folders b-t10,b-g2 --out DIR
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from dy32_eval import SR, load

W = Path("/data/dy32/a")
MIN_WORDS, MIN_VOCAB, TOP = 25, 0.95, 8
RULE = __doc__.split("\n\n")[1].strip()


def texts() -> tuple[dict, dict, set]:
    b = json.loads((W / "eval_base.json").read_text())
    bx = json.loads((W / "eval_base_x.json").read_text())
    P = {i: t for i, t in enumerate(b["prompt"])} | {int(k): v for k, v in bx["prompt"].items()}
    R = {i: t for i, t in enumerate(b["real"])} | {int(k): v for k, v in bx["real"].items()}
    return P, R, set(b["vocab"])


def select(a) -> None:
    P, R, vocab = texts()
    top = {w for w, _ in Counter(w for t in R.values() for w in t.split()).most_common(1000)}
    rows = []
    for i, t in P.items():
        w = t.split()
        if len(w) < MIN_WORDS or np.mean([x in vocab for x in w]) < MIN_VOCAB:
            continue
        rows.append((len(w) / 15.0, -float(np.mean([x in top for x in w])), float(np.mean([len(x) for x in w])), i))
    rows.sort()
    ids = [r[3] for r in rows[:TOP]]
    res = {"rule": RULE, "eligible": len(rows), "ids": ids,
           "ranked": [{"id": r[3], "words_per_s": round(r[0], 3), "top1000_share": round(-r[1], 3), "mean_word_len": round(r[2], 3)}
                      for r in rows[:TOP]]}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res), flush=True)


def gen_file(folder: str, i: int) -> Path:
    return W / "gen" / folder / f"{i:02d}.wav" if i < 50 else W / "gen" / f"{folder}-x" / f"{i:03d}.wav"


def gen_text(folder: str, i: int) -> str:
    c = json.loads((W / "judge_cache" / f"{folder}.json").read_text())
    return c["tr"]["gen"][i] if i < 50 else c["tr_x"]["gen"][i - 50]


def build(a) -> None:
    import soundfile as sf

    sel = json.loads(Path(a.ids).read_text())
    P, R, _ = texts()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t = np.arange(int(0.3 * SR)) / SR
    beep = (0.3 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)
    pk = lambda x: (0.9 * x / max(float(np.abs(x).max()), 1e-6)).astype(np.float32) if len(x) else x
    folders = a.folders.split(",")
    L = ["# Listening set (prompts chosen by prompt simplicity only)", "", "Rule: " + " ".join(sel["rule"].split()), "",
         f"Eligible prompts: {sel['eligible']}. Chosen ids (rank order): {', '.join(f'{i:03d}' for i in sel['ids'])}.", "",
         "Files: `<folder>__XXX.mp3` = last 5 s of the prompt, beep, then 10 s of the model; `real__XXX.mp3` = the same with "
         "the real continuation. Folders: " + ", ".join(f"{f} ({a.labels.split(',')[k] if a.labels else f})" for k, f in enumerate(folders)) + ".", ""]
    for r in sel["ranked"]:
        i = r["id"]
        pdir = Path("/data/dy32/prompts") if i < 50 else Path("/data/dy32/prompts_x")
        pf = pdir / (f"prompt_{i:02d}.wav" if i < 50 else f"prompt_{i:03d}.wav")
        rf = pdir / (f"real_{i:02d}.wav" if i < 50 else f"real_{i:03d}.wav")
        tail = load(pf)[-5 * SR :]
        sf.write(out / f"real__{i:03d}.mp3", np.concatenate([pk(tail), beep, pk(load(rf))]), SR, format="MP3")
        L += [f"## {i:03d}  ({r['words_per_s']} words/s, top-1000 share {r['top1000_share']}, mean word length {r['mean_word_len']})", "",
              f"**Prompt (15 s):** {P[i]}", "", f"**Real continuation:** {R[i]}", ""]
        for k, f in enumerate(folders):
            sf.write(out / f"{f}__{i:03d}.mp3", np.concatenate([pk(tail), beep, pk(load(gen_file(f, i)))]), SR, format="MP3")
            lab = a.labels.split(",")[k] if a.labels else f
            L += [f"**{lab} ({f}):** {gen_text(f, i)}", ""]
    (out / "listen.md").write_text("\n".join(L) + "\n")
    print("LISTEN BUILT", out, len(list(out.glob("*.mp3"))), flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    s = p.add_subparsers(dest="cmd", required=True)
    x = s.add_parser("select")
    x.add_argument("--out", required=True)
    x = s.add_parser("build")
    x.add_argument("--ids", required=True)
    x.add_argument("--folders", required=True)
    x.add_argument("--labels", default="")
    x.add_argument("--out", required=True)
    a = p.parse_args()
    {"select": select, "build": build}[a.cmd](a)


if __name__ == "__main__":
    main()
