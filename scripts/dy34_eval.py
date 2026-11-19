"""DY-34 track M: guidance weight selection on the dev pack, and the "late Dself" extra for the test evaluation.

dev:  folders s3://.../dy34/gen/b-g{1,2,4,8}-dev/200.wav..299.wav. Per folder: letters-recogniser texts, Dself on the
      100 dev prompts (dy32_eval.self_pairs: same judge, both orders, derangement seed 1234), words median, share of
      words in the held-out vocabulary, speech seconds (energy VAD). Pre-registered rule: among w with words >= 20 AND
      in-vocab >= 0.80 the highest Dself; tie -> the smaller w; none -> w = 1 ("none qualified").
late: Dself on the 200-prompt union computed on transcripts of seconds 3-10 only (first 3 s of audio cut before
      recognition), for the given folders (+ -x twins) and anchor-real; reported only.

    python scripts/dy34_eval.py dev
    python scripts/dy34_eval.py late --round final --folders b-g2,b-t10
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from dy30_eval import SR, Judge, cleanup
from dy31_eval import in_vocab_share
from dy32_eval import derange, load, self_pairs, sh, speech_seconds, transcribe

B = "s3://manifestro-dataset-876108973178-us-east-1-an/outputs/dauys"
W = Path("/data/dy32/a")
D = Path("/data/dy34")
WS = (1, 2, 4, 8)
MIN_WORDS, MIN_VOCAB = 20, 0.80


def log(*a) -> None:
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def cut_dir(src: list[Path], dst: Path, cut_s: float = 3.0) -> list[Path]:
    import soundfile as sf

    dst.mkdir(parents=True, exist_ok=True)
    out = []
    for f in src:
        g = dst / f.name
        if not g.exists():
            sf.write(g, load(f)[int(cut_s * SR) :], SR, subtype="PCM_16")
        out.append(g)
    return out


def dev(a) -> None:
    import torch

    device = torch.device("cuda")
    PD = D / "prompts_dev"
    sh(f"aws s3 sync {B}/dy34/prompts_dev/ {PD}/ --only-show-errors")
    G = D / "gen"
    ids = list(range(200, 300))
    folders = {}
    for w in WS:
        n = f"b-g{w}-dev"
        sh(f"aws s3 sync {B}/dy34/gen/{n}/ {G / n}/ --only-show-errors")
        if all((G / n / f"{i}.wav").exists() for i in ids):
            folders[w] = G / n
    if not folders:
        raise SystemExit("no complete dev folder")
    jobs = {f"g{w}": [d / f"{i}.wav" for i in ids] for w, d in folders.items()}
    cf = D / "dev_prompt_texts.json"
    if not cf.exists():
        jobs["_p"] = [PD / f"prompt_{i}.wav" for i in ids]
    tr = transcribe(device, jobs)
    if "_p" in tr:
        cf.write_text(json.dumps(tr["_p"]["gen"]))
    prompts = json.loads(cf.read_text())
    vocab = set(json.loads((W / "eval_base.json").read_text())["vocab"])
    dj, _ = derange(len(ids))
    j = Judge(device)
    rows = {}
    for w in folders:
        g = tr[f"g{w}"]["gen"]
        ds = self_pairs(j, prompts, g, dj, 1)
        d5 = self_pairs(j, prompts, g, dj, 5)
        rows[w] = {"folder": f"b-g{w}-dev", "Dself": ds, "Dself_5w": d5,
                   "words_median": float(np.median([len(t.split()) for t in g])), "in_vocab": in_vocab_share(g, vocab),
                   "speech_seconds_median": float(np.median(tr[f"g{w}"]["speech_seconds"])),
                   "whisper_words_median": float(np.median([len(t.split()) for t in tr[f"g{w}"]["whisper"]])),
                   "texts_first3": g[:3]}
        rows[w]["qualifies"] = bool(rows[w]["words_median"] >= MIN_WORDS and rows[w]["in_vocab"] >= MIN_VOCAB)
        log("DEV", w, json.dumps({k: v for k, v in rows[w].items() if k != "texts_first3"}))
    del j
    cleanup()
    q = [w for w in sorted(rows) if rows[w]["qualifies"]]
    if q:
        best = max(rows[w]["Dself"]["D"] for w in q)
        chosen, note = min(w for w in q if rows[w]["Dself"]["D"] == best), "highest Dself among qualifying w; tie -> smaller w"
    else:
        chosen, note = 1, "none qualified"
    missing = [w for w in WS if w not in rows]
    res = {"w": chosen, "note": note, "rule": f"qualify: words median >= {MIN_WORDS} and in-vocab >= {MIN_VOCAB}; "
           "highest Dself; tie -> smaller w; none -> w = 1", "missing_folders": missing,
           "table": {str(w): {k: v for k, v in r.items() if k != "texts_first3"} for w, r in rows.items()},
           "time_utc": time.strftime("%Y-%m-%d %H:%M")}
    E = D / "eval"
    E.mkdir(parents=True, exist_ok=True)
    (E / "chosen_w.json").write_text(json.dumps(res, indent=1))
    L = ["# DY-34 dev selection (100 dev prompts 200-299)", "", res["rule"] + ".", "",
         "| w | Dself ± 95% (n) | Dself >= 5 words (n) | words median (letters / Whisper) | in-vocab | speech s | qualifies |",
         "|---|---|---|---|---|---|---|"]
    for w, r in sorted(rows.items()):
        L.append(f"| {w} | {r['Dself']['D']:.2f} ± {r['Dself']['ci95']:.2f} ({r['Dself']['n']}) | {r['Dself_5w']['D']:.2f} "
                 f"({r['Dself_5w']['n']}) | {r['words_median']:.0f} / {r['whisper_words_median']:.0f} | {r['in_vocab']:.3f} | "
                 f"{r['speech_seconds_median']:.1f} | {'yes' if r['qualifies'] else 'no'} |")
    L += ["", f"**Chosen w = {chosen}** ({note})." + (f" Missing folders: {missing}." if missing else "")]
    (E / "dev.md").write_text("\n".join(L) + "\n")
    sh(f"aws s3 cp {E}/chosen_w.json {B}/dy34/eval/chosen_w.json --only-show-errors && aws s3 cp {E}/dev.md {B}/dy34/eval/dev.md --only-show-errors")
    print("\n".join(L), flush=True)


def late(a) -> None:
    import torch

    device = torch.device("cuda")
    base = json.loads((W / "eval_base.json").read_text())
    bx = json.loads((W / "eval_base_x.json").read_text())
    PX = Path("/data/dy32/prompts_x")
    ext = sorted(int(i) for i in bx["ids"])
    prompts = base["prompt"] + [bx["prompt"][str(i)] for i in ext]
    G = W / "gen"
    jobs = {}
    for n in a.folders.split(","):
        files = [G / n / f"{i:02d}.wav" for i in range(50)] + [G / f"{n}-x" / f"{i:03d}.wav" for i in ext]
        jobs[n] = files
    jobs["anchor-real"] = [W / "anchor_real" / f"{i:02d}.wav" for i in range(50)] + [PX / f"real_{i:03d}.wav" for i in ext]
    for n, files in list(jobs.items()):
        if not all(f.exists() for f in files):
            log("SKIP incomplete for late Dself", n)
            del jobs[n]
            continue
        jobs[n] = cut_dir(files[:50], D / "late" / n, 3.0) + cut_dir(files[50:], D / "late" / f"{n}-x", 3.0)
    tr = transcribe(device, jobs)
    dj, _ = derange(len(prompts))
    j = Judge(device)
    res = {}
    for n in jobs:
        g = tr[n]["gen"]
        res[n] = {"n": len(g), "late_Dself": self_pairs(j, prompts, g, dj, 1), "late_Dself_5w": self_pairs(j, prompts, g, dj, 5),
                  "late_words_median": float(np.median([len(t.split()) for t in g]))}
        log("LATE", n, json.dumps(res[n]))
    del j
    cleanup()
    E = W / "eval" / a.round
    E.mkdir(parents=True, exist_ok=True)
    (E / "late_dself.json").write_text(json.dumps(res, indent=1))
    L = ["", "## Late Dself (reported only): transcripts of seconds 3-10 of each continuation (first 3 s cut before recognition), "
         "200-prompt union, same derangement", "", "| folder | late Dself ± 95% (n) | late Dself >= 5 words (n) | late words median |",
         "|---|---|---|---|"]
    for n, r in res.items():
        L.append(f"| {n} | {r['late_Dself']['D']:.2f} ± {r['late_Dself']['ci95']:.2f} ({r['late_Dself']['n']}) | "
                 f"{r['late_Dself_5w']['D']:.2f} ({r['late_Dself_5w']['n']}) | {r['late_words_median']:.0f} |")
    sm = E / "summary.md"
    if sm.exists():
        sm.write_text(sm.read_text() + "\n".join(L) + "\n")
    sj = E / "summary.json"
    if sj.exists():
        s = json.loads(sj.read_text())
        s["late_dself"] = res
        sj.write_text(json.dumps(s, indent=2, ensure_ascii=False))
    sh(f"aws s3 sync {E}/ {a.s3}/eval/{a.round}/ --only-show-errors --exclude '*' --include 'summary.*' --include 'late_dself.json'")
    print("\n".join(L), flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    s = p.add_subparsers(dest="cmd", required=True)
    s.add_parser("dev")
    x = s.add_parser("late")
    x.add_argument("--round", required=True)
    x.add_argument("--folders", required=True)
    x.add_argument("--s3", default=f"{B}/dy34")
    a = p.parse_args()
    {"dev": dev, "late": late}[a.cmd](a)


if __name__ == "__main__":
    main()
