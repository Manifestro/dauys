"""DY-33 track M, final round: copy measure, paired preference and the combined summary (extras reported only).

extras:  for the given folders (+ -x twins, 200-prompt union; transcripts from the judge cache of dy32_eval):
         copy measure as in dy34_extras (share of continuation words / content words / word 3- and 4-grams found in the
         own prompt's transcript vs another prompt's, derangement seed 1234), letters and Whisper transcripts;
         paired preference x vs y per prompt with dy30_eval.Judge (same template, both orders) on the full 10 s and on
         seconds 3-10 (cut files written by dy34_eval late), letters transcripts; empty pairs skipped.
summary: summary.md / summary.json from the dy32_eval round (channel_summary.json), late_dself.json and extras.json,
         with the pre-registered threshold line: Dself(union) of the better of S1 (s-t10) and S2 (s-ra08) >= 0.65 AND
         >= model b under the same sampling + 0.05.

    python scripts/dy33_extras.py extras --round final --folders s-t10,s-ra08,b-t10,b-ra08,anchor-real \
        --pairs s-t10:b-t10,s-ra08:b-ra08
    python scripts/dy33_extras.py summary --round final
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from dy30_eval import Judge, cleanup
from dy32_eval import strip_p, transcribe
from dy34_extras import FUNCTION_WORDS, copy_table

W = Path("/data/dy32/a")
LATE = Path("/data/dy34/late")
TH_ABS, TH_GAP = 0.65, 0.05
UNVERIFIED = {"s-g4": "unverified implementation: the w = 0 check failed (12 of 50 pieces identical to S1, 63 % code "
                      "agreement, distributions within 6e-8)"}


def tag(n: str) -> str:
    return f"{n} (unverified implementation)" if n.split(":")[0] in UNVERIFIED else n
import os
DECLARED = {"S1 (temperature 1.0)": ("s-t10", "b-t10"), "S2 (repeat-aware sharpening)": ("s-ra08", "b-ra08")}
if os.environ.get("DY_DECLARED"):  # DY-40: {"t10": ["m-t10", "b-t10"], ...}
    DECLARED = {k: tuple(v) for k, v in json.loads(os.environ["DY_DECLARED"]).items()}


def log(*a) -> None:
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def union_texts(n: str) -> dict:
    c = json.loads((W / "judge_cache" / f"{n}.json").read_text())
    return {"letters": c["tr"]["gen"] + c["tr_x"]["gen"], "whisper": c["tr"]["whisper"] + c["tr_x"]["whisper"]}


def prompt_texts(device) -> dict:
    base = json.loads((W / "eval_base.json").read_text())
    bx = json.loads((W / "eval_base_x.json").read_text())
    ext = sorted(int(i) for i in bx["ids"])
    f = W / "prompt_whisper.json"
    if not f.exists():
        PX, P0 = Path("/data/dy32/prompts_x"), Path("/data/dy32/prompts")
        files = [P0 / f"prompt_{i:02d}.wav" for i in range(50)] + [PX / f"prompt_{i:03d}.wav" for i in ext]
        f.write_text(json.dumps(transcribe(device, {"_p": files})["_p"]["whisper"]))
    return {"letters": base["prompt"] + [bx["prompt"][str(i)] for i in ext], "whisper": json.loads(f.read_text())}


def late_texts(device, names: list[str]) -> dict:
    out, jobs = {}, {}
    for n in names:
        f = W / f"late_letters_{n}.json"
        files = sorted((LATE / n).glob("*.wav")) + sorted((LATE / f"{n}-x").glob("*.wav"))
        if f.exists() and json.loads(f.read_text())["files"] == [x.name for x in files]:
            out[n] = json.loads(f.read_text())["letters"]
        elif len(files) == 200:
            jobs[n] = files
        else:
            log("late files missing for", n, len(files))
    if jobs:
        tr = transcribe(device, jobs)
        for n, files in jobs.items():
            out[n] = tr[n]["gen"]
            (W / f"late_letters_{n}.json").write_text(json.dumps({"files": [x.name for x in files], "letters": out[n]}))
    return out


def extras(a) -> None:
    import torch

    device = torch.device("cuda")
    E = W / "eval" / a.round
    E.mkdir(parents=True, exist_ok=True)
    folders = a.folders.split(",")
    pairs = [tuple(p.split(":")) for p in a.pairs.split(",") if p]
    prompts = prompt_texts(device)
    test = {n: union_texts(n) for n in folders}
    late = late_texts(device, sorted({x for p in pairs for x in p}))
    res = {"function_words": sorted(FUNCTION_WORDS), "copy": {}, "paired": {}, "empty": {}, "time_utc": time.strftime("%Y-%m-%d %H:%M")}
    for rec in ("letters", "whisper"):
        res["copy"][rec] = {n: copy_table(test[n][rec], prompts[rec]) for n in folders}
    for n in folders:
        res["empty"][n] = {rec: int(sum(not t.strip() for t in test[n][rec])) for rec in ("letters", "whisper")}
        if n in late:
            res["empty"][n]["late_letters"] = int(sum(not t.strip() for t in late[n]))
    j = Judge(device)
    pl = prompts["letters"]
    for x, y in pairs:
        r = {}
        for part, gx, gy in (("full", test[x]["letters"], test[y]["letters"]), ("late_3_10s", late.get(x), late.get(y))):
            if gx is None or gy is None:
                continue
            idx = [i for i in range(len(pl)) if gx[i].strip() and gy[i].strip()]
            d = j.pairs([pl[i] for i in idx], [gx[i] for i in idx], [gy[i] for i in idx])
            r[part] = {**strip_p(d), "skipped_empty": len(pl) - len(idx),
                       "per_prompt": {str(i): d["p_real"][k] for k, i in enumerate(idx)}}
            log("PAIRED", x, y, part, json.dumps(strip_p(d)))
        res["paired"][f"{x}:{y}"] = r
    res["paired_question"] = ("dy30_eval.Judge template ('which continuation was actually said next'), both orders, per prompt; "
                              "D = share of prompts where the first folder's text is preferred (letters transcripts; "
                              "per_prompt = mean P(first) over the two orders; prompt index 0-199 = ids 00-49, 050-199)")
    del j
    cleanup()
    (E / "extras.json").write_text(json.dumps(res, indent=1))
    log("EXTRAS DONE")


def summary(a) -> None:
    E = W / "eval" / a.round
    S = json.loads((E / "channel_summary.json").read_text())
    late = json.loads((E / "late_dself.json").read_text()) if (E / "late_dself.json").exists() else {}
    X = json.loads((E / "extras.json").read_text()) if (E / "extras.json").exists() else {}
    fo = S["folders"]
    rows = {}
    for n, s in fo.items():
        u = s.get("union")
        jd = s["judge"]
        r = {"n": u["n"] if u else 50, "scope": "union 200" if u else "first 50 only",
             "Dself": u["Dself"] if u else jd["Dself"], "Dself_5w": u["Dself_5w"] if u else jd["Dself_5w"],
             "Dc_mis": u["Dc_mis"] if u else jd["Dc_mis"], "Dc_first50": jd["Dc"], "Dr_first50": jd["Dr"],
             "words_median_letters": u["words_median"] if u else s["words_median"],
             "words_median_whisper": u["whisper_words_median"] if u else s["whisper_words_median"],
             "speech_seconds_median": u["speech_seconds_median"] if u else s["speech_seconds_median"],
             "in_vocab_first50": s["words_in_heldout_vocab"], "empty_share_letters": u["empty_share"] if u else s["empty_share"],
             "late": late.get(n), "empty_counts": X.get("empty", {}).get(n)}
        rows[n] = r
    th = []
    for samp, (sn, bn) in DECLARED.items():
        if sn in rows and bn in rows and rows[sn]["scope"] == "union 200":
            th.append((samp, sn, rows[sn]["Dself"]["D"], bn, rows[bn]["Dself"]["D"]))
    thr = None
    if th:
        samp, sn, ds, bn, db = max(th, key=lambda t: t[2])
        ok = bool(ds >= TH_ABS and ds >= db + TH_GAP)
        thr = {"rule": f"Dself(union) of the better declared sampling >= {TH_ABS} AND >= model b under the same sampling + {TH_GAP}",
               "better_sampling": samp, "folder": sn, "Dself": ds, "b_folder": bn, "b_Dself": db, "pass": ok,
               "candidates": [{"sampling": t[0], "folder": t[1], "Dself": t[2], "b_folder": t[3], "b_Dself": t[4]} for t in th]}
    out = {"round": a.round, "time_utc": time.strftime("%Y-%m-%d %H:%M"), "threshold": thr, "table": rows,
           "controls": S["controls"], "copy": X.get("copy"), "paired": X.get("paired"), "paired_question": X.get("paired_question"),
           "infos": {n: s.get("info") for n, s in fo.items()},
           "unverified": {n: v for n, v in UNVERIFIED.items() if n in fo}}
    for n in out["unverified"]:
        rows[n]["status"] = UNVERIFIED[n]
    (E / "summary.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    ci = lambda d: f"{d['D']:.2f} ± {d['ci95']:.2f} ({d['n']})"
    L = [f"# {os.environ.get('DY_TITLE', 'DY-33')} final evaluation ({out['time_utc']} UTC)", "",
         os.environ.get("DY_MODELS", "Model s = DY-33 slow-step model (12.5 steps/s x 4 codes), best.pt step 5000; model b = DY-32 track B;") + " anchor-real = "
         "the true continuations. Texts from audio: letters recogniser wav2vec2-large-960h-lv60-self (measuring), Whisper "
         "large-v3-turbo (second opinion); judge Qwen2.5-7B-Instruct, both orders. Union = prompts 00-49 + 050-199 (n = 200); "
         "Dc / Dr / in-vocab are on the first 50 only (as in DY-32/33). Dself n = pairs used (pairs with an empty text skipped). "
         "Late = transcripts of seconds 3-10 only. Early folders have 50 prompts, so their Dself is on 50.", ""]
    if thr:
        L += [f"**Threshold** ({thr['rule']}): better declared sampling = {thr['better_sampling']} ({thr['folder']}), Dself "
              f"{thr['Dself']:.4f} vs {thr['b_folder']} {thr['b_Dself']:.4f} -> **{'PASS' if thr['pass'] else 'FAIL'}**.", ""]
    L += ["| folder | scope | Dself ± 95% (n) | Dself ≥5 words (n) | late Dself (n) | Dc_mis | Dc (50) | Dr (50) | words median letters / Whisper | speech s | in-vocab (50) | empty letters / Whisper |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for n, r in rows.items():
        lt = ci(r["late"]["late_Dself"]) if r["late"] else "-"
        ec = r["empty_counts"]
        L.append(f"| {tag(n)} | {r['scope']} | {ci(r['Dself'])} | {ci(r['Dself_5w'])} | {lt} | {r['Dc_mis']['D']:.2f} ± {r['Dc_mis']['ci95']:.2f} | "
                 f"{r['Dc_first50']['D']:.2f} | {r['Dr_first50']['D']:.2f} | {r['words_median_letters']:.0f} / {r['words_median_whisper']:.0f} | "
                 f"{r['speech_seconds_median']:.1f} | {r['in_vocab_first50']:.3f} | "
                 + (f"{ec['letters']} / {ec['whisper']}" if ec else "-") + " |")
    for n, v in out["unverified"].items():
        L += ["", f"**{n}: {v}.** Its numbers are reported as measured."]
    c = S["controls"]
    L += ["", f"Judge controls on real text (first 50): real vs another real {c['a_other_real']['D']}, vs shuffled chunks "
          f"{c['b_shuffled_chunks']['D']}, vs floor {c['c_floor']['D']}.", ""]
    if X.get("copy"):
        f = lambda d: f"{d['mean']:.3f} ± {d['ci95']:.3f}" if d["mean"] is not None else "-"
        L += ["## Copy measure (200-prompt union; own prompt / another prompt = chance level; mean ± 95% CI over items)", ""]
        for rec in ("letters", "whisper"):
            L += [f"### {rec} transcripts", "", "| folder | words own / other | content words own / other | 3-grams own / other | 4-grams own / other |",
                  "|---|---|---|---|---|"]
            for n, t in X["copy"][rec].items():
                L.append(f"| {tag(n)} | " + " | ".join(f"{f(t[k]['own_prompt'])} / {f(t[k]['other_prompt'])}"
                                                  for k in ("words", "content_words", "3grams", "4grams")) + " |")
            L.append("")
    if X.get("paired"):
        L += ["## Paired preference (same judge and template, both orders, per prompt; D = share of prompts where x is preferred over y)", "",
              "| x vs y | part | D(x over y) ± 95% | n | skipped (empty) | mean P(x) |", "|---|---|---|---|---|---|"]
        for k, r in X["paired"].items():
            for part, d in r.items():
                L.append(f"| {tag(k.split(':')[0]) + ' vs ' + tag(k.split(':')[1])} | {part} | {d['D']:.2f} ± {d['ci95']:.2f} | {d['n']} | {d['skipped_empty']} | {d['mean_p_real']:.3f} |")
        L += ["", "Question: " + X.get("paired_question", ""), ""]
    (E / "summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L), flush=True)


def mark_listen(a) -> None:
    """Marks the unverified folders in the listening set (transcripts.json and listen.md)."""
    d = W / "eval" / a.round / "listen"
    tj = json.loads((d / "transcripts.json").read_text())
    for item in tj.values():
        for v in item.values():
            if isinstance(v, dict) and v.get("folder") in UNVERIFIED:
                v["status"] = UNVERIFIED[v["folder"]]
    tj["_notes"] = {n: v for n, v in UNVERIFIED.items()}
    (d / "transcripts.json").write_text(json.dumps(tj, indent=1, ensure_ascii=False))
    md = (d / "listen.md").read_text()
    for n, v in UNVERIFIED.items():
        md = md.replace(f"({n}):**", f"({n}, {v}):**")
    (d / "listen.md").write_text(md)


def main() -> None:
    p = argparse.ArgumentParser()
    s = p.add_subparsers(dest="cmd", required=True)
    x = s.add_parser("extras")
    x.add_argument("--round", required=True)
    x.add_argument("--folders", required=True)
    x.add_argument("--pairs", default="")
    for c in ("summary", "mark_listen"):
        x = s.add_parser(c)
        x.add_argument("--round", required=True)
    a = p.parse_args()
    {"extras": extras, "summary": summary, "mark_listen": mark_listen}[a.cmd](a)


if __name__ == "__main__":
    main()
