"""Does Mimi carry the content of an utterance? DY-36's yardstick on Mimi's own representations (and the DY-41 LM).

Yardstick (dy36_sweep.Yard, unchanged): 8,000 training utterances (subset) and 500 held-out references with random
candidates; Whisper texts and all-MiniLM-L6-v2 embeddings made in DY-36 (S3 dy36/d/ctl_inputs/). For a candidate
representation, every utterance -> one vector; neighbour of a reference = subset utterance with the highest cosine;
reading = mean text cosine reference-neighbour minus reference-random, 95% CI. Text only measures. DY-36 readings for
the same yardstick: text +0.41, WavLM-L18 mean +0.096, WavLM-L6 +0.005...+0.024. DY-37's start rule for "carries
content": >= +0.10.

Every utterance is encoded ALONE (16 kHz -> 24 kHz, Mimi rev 89091b3e), pooled over its frames:
  enc    output of Mimi's encoder transformer after downsampling (12.5 Hz, everything the codes are made from)
  sem    semantic latent before quantisation (input projection of the semantic quantiser; distilled from WavLM)
  c0emb  the quantised semantic vector (code 0's codebook entry)
  c0bag  bag of code-0 values (consecutive repeats merged), tf-idf (idf from the subset), L2
  lmL    DY-41 LM temporal layer L (8, 12, 16, 20, 24) on [EOS + the utterance's delayed codes] (with --lm)
Variants as dy36_sweep block1: a = mean, b = minus the speaker's mean (own + up to 8 mates of the recording),
c1/c3/c10 = top principal components removed, bc* = both.

    python scripts/dy41_mimi_content.py --out /data/mc --lm s3://airon-checkpoints-.../best.pt
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import dy36_common
from dy36_common import log, utt_job
from dy36_dict import load_table
from dy36_sweep import Yard, pc_remove, utt_list
from dy40_mimi import BUCKET, HOP, MSR, SR, Mimi

LM_LAYERS = [8, 12, 16, 20, 24]


def s3get(key: str, dst: Path) -> None:
    if not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(f"aws s3 cp {BUCKET}/dauys/{key} {dst} --only-show-errors", shell=True, check=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="/data/mc")
    p.add_argument("--lm", default="", help="Airon checkpoint address of a DY-41 best.pt (weights)")
    p.add_argument("--batch", type=int, default=32)
    a = p.parse_args()
    out = Path(a.out)
    dev = torch.device("cuda")
    s3get("dy36/dict/table.npz", out / "dict/table.npz")
    for f in ("subset.json", "emb.npz"):
        s3get(f"dy36/d/ctl_inputs/{f}", out / f"ctl/{f}")
    T = load_table(out / "dict")
    Y = Yard(T, out / "ctl")
    dy36_common.IDX_DIR = out / "idx"
    dy36_common.HELD_DIR = out / "held_none"  # every shard by byte range from S3
    rows, ncore = utt_list(T, Y)
    log("utterances", len(rows), "core", ncore)
    wavs = dy36_common.Audio(json.loads(str(T["files"]))).many([utt_job(T, int(u)) for u in rows])
    log("audio h", round(sum(len(w) for w in wavs) / SR / 3600, 2))

    mm = Mimi(dev)
    M = mm.m
    grab: dict[str, torch.Tensor] = {}
    M.downsample.register_forward_hook(lambda m, i, o: grab.__setitem__("enc", o))
    sq = M.quantizer.semantic_residual_vector_quantizer
    if getattr(sq, "input_proj", None) is not None:
        sq.input_proj.register_forward_hook(lambda m, i, o: grab.__setitem__("sem", o))
    cb0 = sq.layers[0].codebook

    lm = None
    if a.lm:
        import airon

        from dauys.dy40_lm import DLConfig, DepthLM, delay

        ck = torch.load(airon.fetch(a.lm), map_location="cpu", weights_only=False)
        lm = DepthLM(DLConfig(**ck["cfg"])).to(dev).eval()
        lm.load_state_dict(ck["model"])
        log("lm step", ck.get("step", ck.get("state", {}).get("step")))  # best.pt or a full last.pt
        for L in LM_LAYERS:
            lm.blocks[L - 1].register_forward_hook(lambda m, i, o, L=L: grab.__setitem__(f"lm{L}", o))

    pooled: dict[str, np.ndarray] = {}
    codes0: list[np.ndarray] = [None] * len(rows)  # type: ignore
    order = np.argsort(-np.array([len(w) for w in wavs]))
    t0 = time.time()
    with torch.inference_mode():
        for b in range(0, len(order), a.batch):
            idx = order[b : b + a.batch]
            xs = [mm.up(torch.from_numpy(wavs[i].astype(np.float32) / 32768.0).to(dev)) for i in idx]
            n = -(-max(len(x) for x in xs) // HOP) * HOP
            x = torch.zeros(len(idx), 1, n, device=dev)
            for r, xi in enumerate(xs):
                x[r, 0, : len(xi)] = xi
            grab.clear()
            codes = M.encode(x, num_quantizers=8).audio_codes  # [B, 8, F]
            feats = {"enc": grab["enc"]}  # [B, D, F]
            if "sem" in grab:
                feats["sem"] = grab["sem"]
            c0 = codes[:, 0]
            feats["c0emb"] = cb0.decode(c0).transpose(1, 2) if hasattr(cb0, "decode") else None
            if feats["c0emb"] is None:
                del feats["c0emb"]
            for r, i in enumerate(idx):
                nf = -(-len(xs[r]) // HOP)
                codes0[i] = c0[r, :nf].cpu().numpy()
                for k, f in feats.items():
                    v = f[r, :, :nf].float().mean(1).cpu().numpy()
                    pooled.setdefault(k, np.zeros((len(rows), v.size), np.float32))[i] = v
                if lm is not None:
                    fr = codes[r, :, :nf].T.cpu().numpy().astype(np.int64)
                    ids = np.r_[np.full((1, 8), lm.cfg.k), delay(fr, lm.cfg.k + 1)][: lm.cfg.ctx]
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        lm.states(torch.from_numpy(ids)[None].to(dev))
                    for L in LM_LAYERS:
                        h = grab[f"lm{L}"][0, 1:].float().mean(0).cpu().numpy()  # skip the EOS position
                        pooled.setdefault(f"lm{L}", np.zeros((len(rows), h.size), np.float32))[i] = h
            if (b // a.batch) % 50 == 0:
                log("encoded", b + len(idx), "of", len(order), round(time.time() - t0), "s")
    # bag of code-0 values, tf-idf with idf from the subset
    K = 2048
    bag = np.zeros((len(rows), K), np.float32)
    for i, c in enumerate(codes0):
        c = c[np.r_[True, c[1:] != c[:-1]]] if len(c) else c
        np.add.at(bag[i], c, 1.0)
    nsub = len(Y.sub)
    df = (bag[:nsub] > 0).sum(0)
    idf = np.log((1 + nsub) / (1 + df)) + 1
    bag = bag * idf
    bag /= np.maximum(np.linalg.norm(bag, axis=1, keepdims=True), 1e-12)
    pooled["c0bag"] = bag
    np.savez(out / "pooled.npz", **pooled)

    nref = len(Y.refs)
    spk = T["spk"][rows]
    _, inv = np.unique(spk, return_inverse=True)
    cnt = np.bincount(inv)
    res = []
    for k, X in pooled.items():
        S = np.zeros((cnt.size, X.shape[1]))
        np.add.at(S, inv, X)
        C = X - (S / cnt[:, None])[inv]
        core = slice(0, nsub + nref)
        variants = {"a": X[core], "b": C[core]}
        for pc in (1, 3, 10):
            variants[f"c{pc}"] = pc_remove(X[core], X[:nsub], pc)
            variants[f"bc{pc}"] = pc_remove(C[core], C[:nsub], pc)
        for v, Mx in variants.items():
            r = Y.score(Mx[nsub:], Mx[:nsub])
            r["cand"] = f"{k}-{v}"
            res.append(r)
    res.sort(key=lambda r: -r["diff"])
    summary = {"yardstick": "DY-36 (text +0.41, WavLM-L18 +0.096, WavLM-L6 +0.005..+0.024; carries content >= +0.10)",
               "best_per_representation": {}, "all": res}
    for r in res:
        k = r["cand"].split("-")[0]
        summary["best_per_representation"].setdefault(k, r)
    (out / "mimi_content.json").write_text(json.dumps(summary, indent=1))
    lines = ["# Content in Mimi (and DY-41) by DY-36's yardstick", "",
             "Reading = text closeness of the neighbour by this vector minus of a random utterance (95% CI).",
             "References: text +0.41, WavLM-L18 +0.096, WavLM-L6 +0.005..+0.024; 'carries content' from +0.10.", "",
             "| representation | best variant | reading | 95% CI |", "|---|---|---|---|"]
    for k, r in summary["best_per_representation"].items():
        lines.append(f"| {k} | {r['cand']} | {r['diff']:+.3f} | {r['ci95'][0]:+.3f} … {r['ci95'][1]:+.3f} |")
    (out / "mimi_content.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)

    import airon

    run = airon.init()
    run.log({f"eval/content_{k}": r["diff"] for k, r in summary["best_per_representation"].items()})
    subprocess.run(f"aws s3 cp {out}/mimi_content.json {BUCKET}/dauys/dy41/mimi_content/mimi_content.json --only-show-errors"
                   f" && aws s3 cp {out}/mimi_content.md {BUCKET}/dauys/dy41/mimi_content/mimi_content.md --only-show-errors",
                   shell=True, check=True)
    log("MIMI CONTENT DONE")


if __name__ == "__main__":
    main()
