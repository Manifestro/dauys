"""DY-34 extras (track M, reported only; the pre-registered reading is not touched).

1. Copy measure, from the letters and the Whisper transcripts: share of continuation words found in the prompt's
   transcript (all words; content words = words not in FUNCTION_WORDS below), share of continuation word 3-grams and
   4-grams found in the prompt's transcript; the same against ANOTHER prompt's transcript (derangement seed 1234) as the
   chance level. Per item shares, mean with 95 % CI (1.96 x standard error); items without words are left out.
   Test: b-g4, b-t10, anchor-real on the 200 test prompts; dose curve: b-g{1,2,4,8}-dev on the 100 dev prompts.
2. Paired preference: dy30_eval.Judge (same template "which continuation was actually said next", both orders) on
   each test prompt i: guided b-g4_i vs unguided b-t10_i -> D(guided over unguided); also on the late part only
   (transcripts of seconds 3-10).

    python scripts/dy34_extras.py
"""

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from dy30_eval import Judge, cleanup
from dy32_eval import derange, sh, strip_p, transcribe

B = "s3://manifestro-dataset-876108973178-us-east-1-an/outputs/dauys"
W = Path("/data/dy32/a")
D = Path("/data/dy34")
E = W / "eval" / "final"
FUNCTION_WORDS = set("""a an the and or but if so as of to in on at by for from with about into over under than then
that this these those there here it its it's i i'm me my we us our you your he him his she her they them their what
which who whom whose when where why how is am are was were be been being do does did have has had will would can could
shall should may might must not no yes just very too also all any some such only own same up down out off again once
oh uh um yeah like well okay""".split())


def norm(t: str) -> list[str]:
    return "".join(c if c.isalnum() or c == "'" else " " for c in t.lower()).split()


def grams(w: list[str], n: int) -> list[tuple]:
    return [tuple(w[i : i + n]) for i in range(len(w) - n + 1)]


def shares(cont: str, prompt: str) -> dict:
    c, p = norm(cont), norm(prompt)
    ps = set(p)
    out = {"words": np.mean([x in ps for x in c]) if c else np.nan}
    cc = [x for x in c if x not in FUNCTION_WORDS]
    out["content_words"] = np.mean([x in ps for x in cc]) if cc else np.nan
    for n in (3, 4):
        g, pg = grams(c, n), set(grams(p, n))
        out[f"{n}grams"] = np.mean([x in pg for x in g]) if g else np.nan
    return out


def mean_ci(x: list[float]) -> dict:
    a = np.array([v for v in x if np.isfinite(v)], float)
    if not len(a):
        return {"mean": None, "ci95": None, "n": 0}
    return {"mean": round(float(a.mean()), 4), "ci95": round(float(1.96 * a.std(ddof=1) / math.sqrt(len(a))) if len(a) > 1 else 0.0, 4),
            "n": int(len(a))}


def copy_table(conts: list[str], prompts: list[str]) -> dict:
    dj, _ = derange(len(prompts))
    own = [shares(c, p) for c, p in zip(conts, prompts)]
    oth = [shares(c, prompts[dj[i]]) for i, c in enumerate(conts)]
    keys = ["words", "content_words", "3grams", "4grams"]
    return {k: {"own_prompt": mean_ci([o[k] for o in own]), "other_prompt": mean_ci([o[k] for o in oth])} for k in keys}


def main() -> None:
    import torch

    device = torch.device("cuda")
    base = json.loads((W / "eval_base.json").read_text())
    bx = json.loads((W / "eval_base_x.json").read_text())
    ext = sorted(int(i) for i in bx["ids"])
    PX, P0 = Path("/data/dy32/prompts_x"), Path("/data/dy32/prompts")
    prompt_files = [P0 / f"prompt_{i:02d}.wav" for i in range(50)] + [PX / f"prompt_{i:03d}.wav" for i in ext]
    dev_ids = list(range(200, 300))
    jobs = {"_prompts": prompt_files, "_devprompts": [D / "prompts_dev" / f"prompt_{i}.wav" for i in dev_ids]}
    for w in (1, 2, 4, 8):
        jobs[f"dev{w}"] = [D / "gen" / f"b-g{w}-dev" / f"{i}.wav" for i in dev_ids]
    for n in ("b-g4", "b-t10", "anchor-real"):
        jobs[f"late_{n}"] = sorted((D / "late" / n).glob("*.wav")) + sorted((D / "late" / f"{n}-x").glob("*.wav"))
    tr = transcribe(device, jobs)
    test = {}
    for n in ("b-g4", "b-t10", "anchor-real"):
        c = json.loads((W / "judge_cache" / f"{n}.json").read_text())
        test[n] = {"letters": c["tr"]["gen"] + c["tr_x"]["gen"], "whisper": c["tr"]["whisper"] + c["tr_x"]["whisper"]}
    prompts = {"letters": base["prompt"] + [bx["prompt"][str(i)] for i in ext], "whisper": tr["_prompts"]["whisper"]}
    devp = {"letters": tr["_devprompts"]["gen"], "whisper": tr["_devprompts"]["whisper"]}
    res = {"function_words": sorted(FUNCTION_WORDS), "copy": {}, "copy_dev": {}, "time_utc": time.strftime("%Y-%m-%d %H:%M")}
    for rec in ("letters", "whisper"):
        res["copy"][rec] = {n: copy_table(test[n][rec], prompts[rec]) for n in test}
        res["copy_dev"][rec] = {f"b-g{w}-dev": copy_table(tr[f"dev{w}"]["gen" if rec == "letters" else "whisper"], devp[rec]) for w in (1, 2, 4, 8)}
    j = Judge(device)
    g, u = test["b-g4"]["letters"], test["b-t10"]["letters"]
    lg, lu = tr["late_b-g4"]["gen"], tr["late_b-t10"]["gen"]
    pl = prompts["letters"]
    idx = [i for i in range(len(pl)) if g[i].strip() and u[i].strip()]
    lidx = [i for i in range(len(pl)) if lg[i].strip() and lu[i].strip()]
    res["paired"] = {
        "full": {**strip_p(j.pairs([pl[i] for i in idx], [g[i] for i in idx], [u[i] for i in idx])), "skipped_empty": len(pl) - len(idx)},
        "late_3_10s": {**strip_p(j.pairs([pl[i] for i in lidx], [lg[i] for i in lidx], [lu[i] for i in lidx])),
                       "skipped_empty": len(pl) - len(lidx)},
        "question": "dy30_eval.Judge template (which continuation was actually said next), both orders; D = share of "
                    "prompts where the guided b-g4 text is preferred over the unguided b-t10 text (letters transcripts)"}
    del j
    cleanup()
    E.mkdir(parents=True, exist_ok=True)
    (E / "extras.json").write_text(json.dumps(res, indent=1))
    f = lambda d: f"{d['mean']:.3f} ± {d['ci95']:.3f}" if d["mean"] is not None else "-"
    L = ["# DY-34 extras (reported only; the pre-registered reading is not touched)", "",
         "## 1. Copying from the prompt", "",
         "Share of continuation words (all / content words = not in the fixed list of "
         f"{len(FUNCTION_WORDS)} function words in extras.json) and word 3-/4-grams that occur in the prompt's transcript; "
         "'other' = against another prompt's transcript (derangement seed 1234) = chance level. Mean ± 95 % CI over items.", ""]
    for key, title in (("copy", "Test, 200 prompts"), ("copy_dev", "Dev dose curve, 100 prompts")):
        for rec in ("letters", "whisper"):
            L += [f"### {title} — {rec} transcripts", "",
                  "| folder | words own / other | content words own / other | 3-grams own / other | 4-grams own / other |",
                  "|---|---|---|---|---|"]
            for n, t in res[key][rec].items():
                L.append(f"| {n} | " + " | ".join(f"{f(t[k]['own_prompt'])} / {f(t[k]['other_prompt'])}"
                                                  for k in ("words", "content_words", "3grams", "4grams")) + " |")
            L.append("")
    p = res["paired"]
    L += ["## 2. Paired preference, guided b-g4 vs unguided b-t10 (same judge, both orders, 200 test prompts)", "",
          "| part | D(guided over unguided) ± 95% | n | skipped (empty) |", "|---|---|---|---|",
          f"| full 10 s | {p['full']['D']:.2f} ± {p['full']['ci95']:.2f} | {p['full']['n']} | {p['full']['skipped_empty']} |",
          f"| late, seconds 3-10 | {p['late_3_10s']['D']:.2f} ± {p['late_3_10s']['ci95']:.2f} | {p['late_3_10s']['n']} | {p['late_3_10s']['skipped_empty']} |"]
    (E / "extras.md").write_text("\n".join(L) + "\n")
    sh(f"aws s3 cp {E}/extras.md {B}/dy34/eval/final/extras.md --only-show-errors && aws s3 cp {E}/extras.json {B}/dy34/eval/final/extras.json --only-show-errors")
    import os

    from huggingface_hub import HfApi

    api = HfApi(token=os.environ["HF_TOKEN"])
    for x in ("extras.md", "extras.json"):
        api.upload_file(path_or_fileobj=str(E / x), path_in_repo=f"dy34/eval/final/{x}", repo_id="manifestro/dauys-runs")
    print("\n".join(L), flush=True)


if __name__ == "__main__":
    main()
