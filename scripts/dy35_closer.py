"""DY-35 track M: evaluation on the "before an utterance" pack (dy35/prompts_turn/: prompt_NNN.wav 15 s, real_NNN.wav the
next utterance cut at 10 s, swap_NNN.wav another utterance of the same speaker, NNN = 000-199).

texts:   letters recogniser (measuring) and Whisper texts of prompt / real / swap and of any folder of NNN.wav (cached
         per file list in <work>/texts_<name>.json).
control: judge sanity control for the "closer in content" question (CLOSER below, Qwen2.5-7B-Instruct, both orders):
         reference = real utterance, A/B = real utterance vs another real utterance of the same speaker (swap).
         C0: all three texts from the letters recogniser (reference and candidate are the same text);
         C1: reference letters, candidates Whisper; C2: reference Whisper, candidates letters. Required: >= 0.90.
closer:  reference = real utterance (letters), candidates = folder x vs folder y per item -> D(x closer than y);
         empty pairs skipped; per-item P stored.
dself:   Dself on this pack (dy32_eval.self_pairs: dy30_eval.Judge template, both orders, derangement seed 1234) for
         folders of NNN.wav and for the real utterances (anchor).

    python scripts/dy35_closer.py control --pack /data/dy35/prompts_turn --work /data/dy35/eval
    python scripts/dy35_closer.py closer --pack ... --work ... --x true=DIR --y swapped=DIR
    python scripts/dy35_closer.py dself --pack ... --work ... --folders b=DIR,t=DIR
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from dy30_eval import Judge, cleanup
from dy32_eval import derange, self_pairs, strip_p, transcribe

N = 200
REQ = 0.90
CLOSER = (
    "Below is a reference transcript of an utterance, followed by two candidate transcripts, A and B.\n\n"
    "Reference:\n{prompt}\n\nCandidate A:\n{a}\n\nCandidate B:\n{b}\n\n"
    "Which candidate is closer in content to the reference? Answer with a single letter, A or B."
)


def log(*a) -> None:
    print(time.strftime("%H:%M:%S"), *a, flush=True)


class CloserJudge(Judge):
    """dy30_eval.Judge with the CLOSER question; pairs(refs, x, y) -> D(x closer than y), both orders."""

    def p_first(self, prompts, first, second, batch: int = 8) -> np.ndarray:
        torch = self.torch
        texts = [self.tok.apply_chat_template([{"role": "user", "content": CLOSER.format(prompt=p, a=a, b=b)}],
                                              tokenize=False, add_generation_prompt=True)
                 for p, a, b in zip(prompts, first, second)]
        out = []
        with torch.no_grad():
            for i in range(0, len(texts), batch):
                enc = self.tok(texts[i : i + batch], return_tensors="pt", padding=True).to(self.device)
                lg = self.model(**enc).logits[:, -1, self.ab].float()
                if not torch.isfinite(lg).all():
                    raise RuntimeError("judge logits not finite")
                out.append(torch.softmax(lg, -1)[:, 0].cpu().numpy())
        return np.concatenate(out) if out else np.zeros(0)


def texts(device, work: Path, sets: dict[str, list[Path]]) -> dict:
    out, jobs = {}, {}
    for n, files in sets.items():
        f = work / f"texts_{n}.json"
        sig = [[x.name, x.stat().st_size] for x in files]
        if f.exists() and json.loads(f.read_text())["sig"] == sig:
            out[n] = json.loads(f.read_text())
        else:
            jobs[n] = files
    if jobs:
        tr = transcribe(device, jobs)
        for n, files in jobs.items():
            out[n] = {"sig": [[x.name, x.stat().st_size] for x in files], "letters": tr[n]["gen"], "whisper": tr[n]["whisper"],
                      "speech_seconds": tr[n]["speech_seconds"]}
            (work / f"texts_{n}.json").write_text(json.dumps(out[n]))
    return out


def pack_files(pack: Path, kind: str) -> list[Path]:
    return [pack / f"{kind}_{i:03d}.wav" for i in range(N)]


def folder_files(d: Path) -> list[Path]:
    return [d / f"{i:03d}.wav" for i in range(N)]


def judged(j, refs, x, y) -> dict:
    idx = [i for i in range(len(refs)) if refs[i].strip() and x[i].strip() and y[i].strip()]
    d = j.pairs([refs[i] for i in idx], [x[i] for i in idx], [y[i] for i in idx])
    return {**strip_p(d), "skipped_empty": len(refs) - len(idx), "per_item": {f"{i:03d}": d["p_real"][k] for k, i in enumerate(idx)}}


def control(a) -> None:
    import torch

    device = torch.device("cuda")
    pack, work = Path(a.pack), Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    T = texts(device, work, {"real": pack_files(pack, "real"), "swap": pack_files(pack, "swap")})
    j = CloserJudge(device)
    R, S = T["real"], T["swap"]
    res = {"question": CLOSER, "required": REQ,
           "C0_letters_all": judged(j, R["letters"], R["letters"], S["letters"]),
           "C1_ref_letters_cand_whisper": judged(j, R["letters"], R["whisper"], S["whisper"]),
           "C2_ref_whisper_cand_letters": judged(j, R["whisper"], R["letters"], S["letters"]),
           "empty": {k: {r: int(sum(not t.strip() for t in T[k][r])) for r in ("letters", "whisper")} for k in T},
           "time_utc": time.strftime("%Y-%m-%d %H:%M")}
    for k in ("C0_letters_all", "C1_ref_letters_cand_whisper", "C2_ref_whisper_cand_letters"):
        res[k]["meets_required"] = bool(res[k]["D"] >= REQ)
        log("CONTROL", k, json.dumps({x: v for x, v in res[k].items() if x != "per_item"}))
    del j
    cleanup()
    (work / "closer_control.json").write_text(json.dumps(res, indent=1))


def closer(a) -> None:
    import torch

    device = torch.device("cuda")
    pack, work = Path(a.pack), Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    (xn, xd), (yn, yd) = a.x.split("=", 1), a.y.split("=", 1)
    T = texts(device, work, {"real": pack_files(pack, "real"), xn: folder_files(Path(xd)), yn: folder_files(Path(yd))})
    j = CloserJudge(device)
    res = {"question": CLOSER, "x": xn, "y": yn, "letters": judged(j, T["real"]["letters"], T[xn]["letters"], T[yn]["letters"]),
           "whisper": judged(j, T["real"]["whisper"], T[xn]["whisper"], T[yn]["whisper"]), "time_utc": time.strftime("%Y-%m-%d %H:%M")}
    del j
    cleanup()
    for r in ("letters", "whisper"):
        log("CLOSER", xn, yn, r, json.dumps({k: v for k, v in res[r].items() if k != "per_item"}))
    (work / f"closer_{xn}_vs_{yn}.json").write_text(json.dumps(res, indent=1))


def dself(a) -> None:
    import torch

    device = torch.device("cuda")
    pack, work = Path(a.pack), Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    F = {n: folder_files(Path(d)) for n, d in (x.split("=", 1) for x in a.folders.split(",") if x)}
    T = texts(device, work, {"prompt": pack_files(pack, "prompt"), "real": pack_files(pack, "real"), **F})
    dj, _ = derange(N)
    j = Judge(device)
    p = T["prompt"]["letters"]
    res = {}
    for n in ["real", *F]:
        g = T[n]["letters"]
        res[n] = {"Dself": self_pairs(j, p, g, dj, 1), "Dself_5w": self_pairs(j, p, g, dj, 5),
                  "words_median_letters": float(np.median([len(t.split()) for t in g])),
                  "words_median_whisper": float(np.median([len(t.split()) for t in T[n]["whisper"]])),
                  "speech_seconds_median": float(np.median(T[n]["speech_seconds"])),
                  "empty_letters": int(sum(not t.strip() for t in g)), "empty_whisper": int(sum(not t.strip() for t in T[n]["whisper"]))}
        log("DSELF", n, json.dumps(res[n]))
    del j
    cleanup()
    (work / "dself_turn.json").write_text(json.dumps(res, indent=1))


def main() -> None:
    p = argparse.ArgumentParser()
    s = p.add_subparsers(dest="cmd", required=True)
    for c in ("control", "closer", "dself"):
        x = s.add_parser(c)
        x.add_argument("--pack", required=True)
        x.add_argument("--work", required=True)
        if c == "closer":
            x.add_argument("--x", required=True)
            x.add_argument("--y", required=True)
        if c == "dself":
            x.add_argument("--folders", default="")
    a = p.parse_args()
    {"control": control, "closer": closer, "dself": dself}[a.cmd](a)


if __name__ == "__main__":
    main()
