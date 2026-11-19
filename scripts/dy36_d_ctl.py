"""DY-36 d: judge-free row on the gate's fixed control set (text only measures finished audio).

Same 8,000-utterance subset, 500 references and random candidates as dy36_g0ctl.py (S3 dy36/g0/control/subset.json),
same subset Whisper texts (whisper_subset.json) and the same text embedding (dy36_g0ctl.embed, all-MiniLM-L6-v2 mean
pooling). Reference texts: Whisper large-v3-turbo of the 500 references (cached, transcribed once here).
Dictionary D from a checkpoint: z of every utterance of the speakers of the subset and the references (original codes),
thought = z minus the speaker's mean in the recording, normalised (dy36_dict.centre); "Draw" = z itself.
Neighbour = the subset utterance with the highest thought cosine (valid thoughts only).
Reading: mean text cosine reference-neighbour minus reference-random (paired), 95% CI. A and C2500 are recomputed from
their dictionary files as a check that the scale matches the gate's numbers.

    python scripts/dy36_d_ctl.py --ckpt /data/dy36/d/run/last.pt --out /data/dy36/d/ctl
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from dy36_common import Audio, log, utt_job
from dy36_dict import centre, load_table


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dict", default="/data/dy36/dict")
    p.add_argument("--ctl", default="/data/ctl")
    p.add_argument("--ckpt", default="")
    p.add_argument("--others", default="A,C2500")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    T = load_table(Path(a.dict))
    S = json.loads((Path(a.ctl) / "subset.json").read_text())
    sub, refs, rand = (np.array(S[k], np.int64) for k in ("subset_rows", "ref_rows", "rand_rows"))
    W = json.loads((Path(a.ctl) / "whisper_subset.json").read_text())
    rf = Path(a.ctl) / "whisper_refs.json"
    if rf.exists():
        RW = json.loads(rf.read_text())
    else:
        from dy36_d_measure import whisper

        au = Audio(json.loads(str(T["files"])))
        txt = whisper(au.many([utt_job(T, int(u)) for u in refs]))
        RW = {str(int(u)): t for u, t in zip(refs, txt)}
        rf.write_text(json.dumps(RW))
    ef = Path(a.ctl) / "emb.npz"
    if ef.exists():
        E = np.load(ef)
        Er, Es = E["Er"], E["Es"]
    else:
        from dy36_g0ctl import embed

        E = embed([RW[str(int(u))] for u in refs] + [W[str(int(u))] for u in sub])
        Er, Es = E[: len(refs)], E[len(refs) :]
        np.savez(ef, Er=Er, Es=Es)
    pos = {int(u): i for i, u in enumerate(sub)}
    cos_rand = np.array([Er[i] @ Es[pos[int(rand[i])]] for i in range(len(refs))])

    def row(th: np.ndarray, valid: np.ndarray) -> dict:
        s2 = sub[valid[sub]]
        nb = s2[(th[refs] @ th[s2].T).argmax(1)]
        assert not np.isin(T["ep"][nb], T["ep"][refs]).any()
        cn = np.array([Er[i] @ Es[pos[int(nb[i])]] for i in range(len(refs))])
        d = cn - cos_rand
        h = 1.96 * d.std(ddof=1) / np.sqrt(len(d))
        return {"text_cos_nb": round(float(cn.mean()), 4), "text_cos_rand": round(float(cos_rand.mean()), 4),
                "diff": round(float(d.mean()), 4), "ci95": [round(float(d.mean() - h), 4), round(float(d.mean() + h), 4)],
                "examples": [{"ref": RW[str(int(refs[i]))], "nb": W[str(int(nb[i]))]} for i in range(5)]}

    res = {"rule": __doc__}
    for n in [x for x in a.others.split(",") if x]:
        z = np.load(Path(a.dict) / f"dict_{n}.npz")
        res[n] = row(z["thought"], z["valid"])
        log(n, json.dumps({k: v for k, v in res[n].items() if k != "examples"}))
    if a.ckpt:
        import torch

        from dy36_d_train import load_model, z_rows

        model, ck = load_model(a.ckpt)
        spk = np.unique(T["spk"][np.concatenate([sub, refs])])
        rows = np.flatnonzero(np.isin(T["spk"], spk))
        log("encoding", len(rows), "utterances of", len(spk), "speakers")
        raw = np.zeros((len(T["ep"]), 64), np.float32)
        raw[rows] = z_rows(model, T, rows)
        th, valid = centre(raw[rows], T["spk"][rows])
        TH = np.zeros_like(raw)
        V = np.zeros(len(raw), bool)
        TH[rows], V[rows] = th, valid
        res["ckpt"], res["step"] = a.ckpt, ck["step"]
        res["D"] = row(TH, V)
        res["Draw"] = row(raw, V)
        for n in ("D", "Draw"):
            log(n, "step", ck["step"], json.dumps({k: v for k, v in res[n].items() if k != "examples"}))
    tag = f"step{res.get('step', 0)}"
    (out / f"ctl_{tag}.json").write_text(json.dumps(res, indent=1))
    log("CTL DONE", tag)


if __name__ == "__main__":
    main()
