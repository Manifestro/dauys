"""DY-36 gate G0 "the dictionary is usable" (pre-registered).

References: 500 held-out utterances of 4-15 s (200-750 frames; their speaker has >= 2 utterances in the recording),
episodes in id order, inside an episode a seeded order (seed 1234), round-robin over episodes.
Bank: training utterances of 4-15 s (speaker has >= 2 utterances in the recording) = other recordings by construction.
Random candidate (one per reference, shared by all dictionaries): uniform from the bank among lengths within x2 (seed 1235).
Neighbour (per dictionary): the bank utterance with the highest cosine of thoughts (valid thoughts only).
The three audios are recognised (letters = main row, Whisper = robustness row); the judge (CLOSER question, both orders)
says which candidate is closer in content to the reference. G0 = share choosing the neighbour; pass >= 0.65.

    python scripts/dy36_g0.py --dict /data/dy36/dict --dicts A,C2400,C2400raw --out /data/dy36/g0
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from dy36_common import CLOSER, Audio, Judge, TextCache, log, utt_job, wilson
from dy36_dict import load_table

N, SEED, PASS = 500, 1234, 0.65


def groups_ok(T: dict) -> np.ndarray:
    _, inv, cnt = np.unique(T["spk"], return_inverse=True, return_counts=True)
    return cnt[inv] >= 2


def select(T: dict):
    ok = groups_ok(T) & (T["len"] >= 200) & (T["len"] <= 750)
    held, train = T["held"], ~T["held"]
    rng = np.random.default_rng(SEED)
    cand = np.flatnonzero(held & ok)
    eps = np.unique(T["ep"][cand])
    per = [rng.permutation(cand[T["ep"][cand] == e]) for e in eps]
    refs, ptr = [], [0] * len(per)
    while len(refs) < N and any(ptr[i] < len(per[i]) for i in range(len(per))):
        for i in range(len(per)):
            if len(refs) < N and ptr[i] < len(per[i]):
                refs.append(int(per[i][ptr[i]]))
                ptr[i] += 1
    bank = np.flatnonzero(train & ok)
    r2 = np.random.default_rng(SEED + 1)
    rand = []
    for u in refs:
        r = T["len"][bank] / T["len"][u]
        el = bank[(r >= 0.5) & (r <= 2.0)]
        rand.append(int(el[r2.integers(len(el))]))
    return np.array(refs), bank, np.array(rand)


def neighbours(th: np.ndarray, valid: np.ndarray, refs: np.ndarray, bank: np.ndarray) -> np.ndarray:
    import torch

    b = bank[valid[bank]]
    B = torch.from_numpy(th[b]).cuda()
    R = torch.from_numpy(th[refs]).cuda()
    return b[(R @ B.T).argmax(1).cpu().numpy()]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dict", required=True)
    p.add_argument("--dicts", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    T = load_table(Path(a.dict))
    files = json.loads(str(T["files"]))
    refs, bank, rand = select(T)
    assert not T["held"][bank].any() and T["held"][refs].all()
    assert not np.isin(T["ep"][rand], T["ep"][refs]).any()
    log("refs", len(refs), "episodes", len(np.unique(T["ep"][refs])), "bank", len(bank))
    D = {}
    for name in a.dicts.split(","):
        z = np.load(Path(a.dict) / f"dict_{name}.npz")
        th, valid = z["thought"], z["valid"]
        nb = neighbours(th, valid, refs, bank)
        assert not np.isin(T["ep"][nb], T["ep"][refs]).any()
        cn = (th[refs] * th[nb]).sum(1)
        cr = (th[refs] * th[rand]).sum(1)
        D[name] = {"nb": nb, "cos_nb": cn, "cos_rand": cr, "ref_valid": valid[refs]}
    audio = Audio(files)
    tc = TextCache(out / "texts.json")
    jobs = {str(int(u)): utt_job(T, int(u)) for u in set(refs.tolist()) | set(rand.tolist())
            | {int(u) for d in D.values() for u in d["nb"]}}
    tc.fill(audio, jobs)
    j = Judge()
    res = {"rule": __doc__, "n_refs": len(refs), "pass": PASS, "ref_ids": refs.tolist(), "rand_ids": rand.tolist()}
    for name, d in D.items():
        r = {"cos_nb_mean": round(float(d["cos_nb"].mean()), 4), "cos_rand_mean": round(float(d["cos_rand"].mean()), 4),
             "ref_valid_share": float(d["ref_valid"].mean()), "nb_ids": d["nb"].tolist()}
        for row in ("letters", "whisper"):
            ref = [tc[str(u)][row] for u in refs]
            x = [tc[str(int(u))][row] for u in d["nb"]]
            y = [tc[str(int(u))][row] for u in rand]
            r[row] = j.pairs(CLOSER, ref, x, y)
            log("G0", name, row, json.dumps({k: v for k, v in r[row].items() if k != "per_item"}))
        r["G0"] = r["letters"]["D"]
        r["passed"] = bool(r["G0"] >= PASS)
        r["examples"] = [{"ref": tc[str(int(refs[i]))]["whisper"], "neighbour": tc[str(int(d["nb"][i]))]["whisper"],
                          "random": tc[str(int(rand[i]))]["whisper"], "ref_letters": tc[str(int(refs[i]))]["letters"],
                          "cos_nb": round(float(d["cos_nb"][i]), 3)} for i in range(10)]
        res[name] = r
        (out / "g0.json").write_text(json.dumps(res, indent=1))
    summ = {n: {"G0_letters": res[n]["letters"]["D"], "ci95": res[n]["letters"]["ci95"], "n": res[n]["letters"]["n"],
                "G0_whisper": res[n]["whisper"]["D"], "ci95_whisper": res[n]["whisper"]["ci95"],
                "cos_nb": res[n]["cos_nb_mean"], "cos_rand": res[n]["cos_rand_mean"], "passed": res[n]["passed"]}
            for n in D}
    (out / "g0_summary.json").write_text(json.dumps(summ, indent=1))
    log("G0 DONE", json.dumps(summ))


if __name__ == "__main__":
    main()
