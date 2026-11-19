"""DY-36 sweep: where in the signal does content similarity live? (text only measures; no candidate sees text)

Yardstick (as dy36_g0ctl.py / dy36_d_ctl.py): the gate's 8,000-utterance subset, 500 references and random candidates;
Whisper texts of the subset (gate) and references (dy36_d_ctl); text embeddings all-MiniLM-L6-v2 (gate's embed()).
Neighbour = subset utterance with the highest cosine of the candidate vectors; reading = mean text cosine
reference-neighbour minus reference-random (paired), 95% CI.

feats:  every utterance of subset + references, encoded ALONE (+80 zero samples), fp16 autocast:
        WavLM-Large (kNN-VC hub copy; layer L = output after transformer layer L) layers 6..24 and HuBERT-Large
        (facebook/hubert-large-ll60k, self-supervised only; hidden_states[L]) layers 12, 18, 24.
        Pooled means for all of them; frame features (fp16) of subset + references for later unit passes.
        For speaker centring also up to MATES other utterances of each speaker (same recording) - pooled only.
block1: per layer: (a) mean-pooled; (b) minus the mean of the speaker's pooled features (own + up to MATES mates);
        (c) (a) and (b) with the top 1/3/10 principal components (fitted on the subset) removed; (d) (a) z-scored per dim.

    python scripts/dy36_sweep.py feats  --out /data/d36/sweep
    python scripts/dy36_sweep.py block1 --out /data/d36/sweep
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import dy36_common
from dy36_common import log, utt_job
from dy36_dict import load_table

WL = [6, 9, 12, 15, 18, 21, 24]
HL = [12, 18, 24]
MATES = 8


class Yard:
    """The gate's control yardstick."""

    def __init__(self, T: dict, ctl: Path) -> None:
        S = json.loads((ctl / "subset.json").read_text())
        self.sub, self.refs, self.rand = (np.array(S[k], np.int64) for k in ("subset_rows", "ref_rows", "rand_rows"))
        E = np.load(ctl / "emb.npz")
        self.Er, self.Es = E["Er"], E["Es"]
        self.pos = {int(u): i for i, u in enumerate(self.sub)}
        self.cos_rand = np.array([self.Er[i] @ self.Es[self.pos[int(self.rand[i])]] for i in range(len(self.refs))])
        self.T = T

    def score(self, vr: np.ndarray, vs: np.ndarray) -> dict:
        """vr [500, d] reference vectors, vs [8000, d] subset vectors (any scale; cosine)."""
        vr = vr / np.maximum(np.linalg.norm(vr, axis=1, keepdims=True), 1e-12)
        vs = vs / np.maximum(np.linalg.norm(vs, axis=1, keepdims=True), 1e-12)
        nb = (vr @ vs.T).argmax(1)
        assert not np.isin(self.T["ep"][self.sub[nb]], self.T["ep"][self.refs]).any()
        cn = np.array([self.Er[i] @ self.Es[nb[i]] for i in range(len(self.refs))])
        d = cn - self.cos_rand
        h = 1.96 * d.std(ddof=1) / np.sqrt(len(d))
        return {"diff": round(float(d.mean()), 4), "ci95": [round(float(d.mean() - h), 4), round(float(d.mean() + h), 4)],
                "text_cos_nb": round(float(cn.mean()), 4)}


def utt_list(T: dict, Y: Yard, seed: int = 0) -> tuple[np.ndarray, int]:
    """subset + refs first (their count returned), then up to MATES other utterances of each of their speakers."""
    core = np.concatenate([Y.sub, Y.refs])
    rng = np.random.default_rng(seed)
    cs = set(core.tolist())
    extra = []
    for s in np.unique(T["spk"][core]):
        rows = [r for r in np.flatnonzero(T["spk"] == s) if r not in cs]
        extra += [int(x) for x in rng.permutation(rows)[:MATES]]
    return np.concatenate([core, np.array(extra, np.int64)]), len(core)


def cmd_feats(a) -> None:
    import torch
    from transformers import HubertModel

    T = load_table(Path(a.dict))
    Y = Yard(T, Path(a.ctl))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dy36_common.HELD_DIR = Path(a.held)
    dy36_common.IDX_DIR = out / "idx"
    rows, ncore = utt_list(T, Y)
    np.save(out / "rows.npy", rows)
    log("utterances", len(rows), "core", ncore, "speakers", len(np.unique(T["spk"][rows[:ncore]])))
    au = dy36_common.Audio(json.loads(str(T["files"])))
    wavs = au.many([utt_job(T, int(u)) for u in rows])
    log("audio h", round(sum(len(w) for w in wavs) / 16000 / 3600, 2))
    nfr = np.array([max(0, (len(w) + 80 - 400) // 320 + 1) for w in wavs])
    off = np.concatenate([[0], np.cumsum(nfr[:ncore])])
    np.save(out / "frame_off.npy", off)
    knn = torch.hub.load("bshall/knn-vc", "knn_vc", prematched=True, trust_repo=True, pretrained=True, device="cuda",
                         verbose=False)
    wavlm = knn.wavlm.eval()
    hub = HubertModel.from_pretrained("facebook/hubert-large-ll60k").cuda().eval()
    models = [("W", WL), ("H", HL)]
    pooled = {f"{m}{L}": np.zeros((len(rows), 1024), np.float32) for m, Ls in models for L in Ls}
    frames = {k: np.lib.format.open_memmap(out / f"frames_{k}.npy", "w+", np.float16, (int(off[-1]), 1024)) for k in pooled}
    order = np.argsort(-np.array([len(w) for w in wavs]))
    b = 0
    t0 = time.time()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        while b < len(order):
            L0 = len(wavs[order[b]])
            nb = max(1, int(a.batch_sec * 16000 // L0))
            idx = order[b : b + nb]
            b += nb
            x = torch.zeros(len(idx), L0 + 80)
            pad = torch.ones(len(idx), L0 + 80, dtype=torch.bool)
            for j, i in enumerate(idx):
                x[j, : len(wavs[i])] = torch.from_numpy(wavs[i].astype(np.float32) / 32768.0)
                pad[j, : len(wavs[i]) + 80] = False
            x, pad = x.cuda(), pad.cuda()
            (_, lr), _ = wavlm.extract_features(x, padding_mask=pad if pad.any() else None, output_layer=24,
                                                ret_layer_results=True)
            hw = {L: lr[L][0].transpose(0, 1) for L in WL}
            am = (~pad).long()
            # HuBERT-Large expects each waveform normalised to zero mean, unit variance (its feature extractor does so)
            xn = torch.zeros_like(x)
            for j, i in enumerate(idx):
                n = len(wavs[i])
                w = x[j, :n]
                xn[j, :n] = (w - w.mean()) / (w.std() + 1e-7)
            hh = hub(xn, attention_mask=am, output_hidden_states=True).hidden_states
            hh = {L: hh[L] for L in HL}
            for m, H in (("W", hw), ("H", hh)):
                for L, f in H.items():
                    k = f"{m}{L}"
                    for j, i in enumerate(idx):
                        n = int(nfr[i])
                        g = f[j, :n].float()
                        pooled[k][i] = g.mean(0).cpu().numpy()
                        if i < ncore:
                            frames[k][off[i] : off[i] + n] = g.half().cpu().numpy()[: off[i + 1] - off[i]]
            if (b // nb) % 50 == 0:
                log("done", b, "of", len(order), round(time.time() - t0), "s")
    for f in frames.values():
        f.flush()
    np.savez(out / "pooled.npz", **pooled)
    # sanity: WavLM layer 6 frames re-quantised by the K2000 codebook vs the stored codes
    cent = np.load(a.cb).astype(np.float32)
    cent /= np.linalg.norm(cent, axis=1, keepdims=True)
    agree, tot = 0, 0
    for i in range(0, ncore, 40):
        u = rows[i]
        f = np.asarray(frames["W6"][off[i] : off[i + 1]], np.float32)
        c = (f / np.linalg.norm(f, axis=1, keepdims=True) @ cent.T).argmax(1)
        ref = T["codes"][T["code_off"][u] : T["code_off"][u + 1]]
        n = min(len(c), len(ref))
        agree += int((c[:n] == ref[:n]).sum())
        tot += max(len(c), len(ref))
    log("FEATS DONE", "W6 requant agreement with stored codes", round(agree / tot, 4), "sec", round(time.time() - t0))


def pc_remove(X: np.ndarray, fit: np.ndarray, k: int) -> np.ndarray:
    mu = fit.mean(0)
    _, _, Vt = np.linalg.svd(fit - mu, full_matrices=False)
    V = Vt[:k].T
    Xc = X - mu
    return Xc - (Xc @ V) @ V.T


def cmd_block1(a) -> None:
    T = load_table(Path(a.dict))
    Y = Yard(T, Path(a.ctl))
    out = Path(a.out)
    rows = np.load(out / "rows.npy")
    P = np.load(out / "pooled.npz")
    nsub, nref = len(Y.sub), len(Y.refs)
    spk = T["spk"][rows]
    _, inv = np.unique(spk, return_inverse=True)
    cnt = np.bincount(inv)
    log("speaker group sizes for centring (core utterances): median", float(np.median(cnt[inv[: nsub + nref]])),
        "share with >= 3", float((cnt[inv[: nsub + nref]] >= 3).mean()))
    res = []
    for k in P.files:
        X = P[k]
        S = np.zeros((cnt.size, X.shape[1]))
        np.add.at(S, inv, X)
        C = X - (S / cnt[:, None])[inv]
        core = slice(0, nsub + nref)
        variants = {"a": X[core], "b": C[core], "d": (X[core] - X[:nsub].mean(0)) / (X[:nsub].std(0) + 1e-6)}
        for pc in (1, 3, 10):
            variants[f"c{pc}"] = pc_remove(X[core], X[:nsub], pc)
            variants[f"bc{pc}"] = pc_remove(C[core], C[:nsub], pc)
        for v, M in variants.items():
            r = Y.score(M[nsub:], M[:nsub])
            r["cand"] = f"{k}-{v}"
            res.append(r)
            log(json.dumps(r))
    res.sort(key=lambda r: -r["diff"])
    (out / "block1.json").write_text(json.dumps(res, indent=1))
    log("BLOCK1 DONE")
    for r in res[:15]:
        print(r["cand"], r["diff"], r["ci95"])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["feats", "block1"])
    p.add_argument("--dict", default="/data/d36/dict")
    p.add_argument("--ctl", default="/data/d36/ctl")
    p.add_argument("--held", default="/opt/dlami/nvme/held")
    p.add_argument("--cb", default="/data/cb/km-corpus-K2000.npy")
    p.add_argument("--out", default="/data/d36/sweep")
    p.add_argument("--batch-sec", type=float, default=400.0)
    a = p.parse_args()
    {"feats": cmd_feats, "block1": cmd_block1}[a.cmd](a)


if __name__ == "__main__":
    main()
