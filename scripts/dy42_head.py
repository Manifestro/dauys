"""DY-42: the phrase head of dy41_phrase_vec.py trained on stored frames (dataset dy42-w2v-l12-frames, in $AIRON_DATA)
and transcripts (dataset dy42-whisper-texts), all ~800k eligible utterances.

Same head, target, loss and schedule as DY-41/35 (4 layers, all-mpnet-base-v2 of the Whisper transcript, symmetric
InfoNCE at 0.05, AdamW 3e-4, warmup 200, cosine, batch 256); only the number of pairs changes. Frames stay on disk
(memory-mapped), batches are gathered by a prefetch thread. Every --ckpt-every steps: held-back fit (cos_true, top1),
DY-36's yardstick on the core frames (eval/content_phrase_head, the primary), a checkpoint; resumes via airon.resume().

    python scripts/dy42_head.py --texts <dir with texts/*.json> --steps 16000
"""

import argparse
import json
import os
import queue
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from dy36_common import log
from dy36_dict import load_table
from dy36_sweep import Yard
from dy41_phrase_vec import GATE, LAYER, TARGET, Head, embed, s3get


class Frames:
    """Memory-mapped frame files; utterance i is (file, start, end)."""

    def __init__(self, paths: list[Path]) -> None:
        self.mm, rows = [], []
        for f, p in enumerate(sorted(paths)):
            z = np.load(p.with_suffix(".npz"))
            self.mm.append(np.load(p.with_suffix(".npy"), mmap_mode="r"))
            off = z["off"]
            rows += [(int(q), f, int(off[j]), int(off[j + 1])) for j, q in enumerate(z["pos"])]
        rows.sort()
        self.pos = np.array([r[0] for r in rows], np.int64)
        self.loc = np.array([r[1:] for r in rows], np.int64)

    def batch(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        segs = [self.mm[f][s:e] for f, s, e in self.loc[idx]]
        n = max(len(x) for x in segs)
        x = np.zeros((len(segs), n, segs[0].shape[1]), np.float16)
        pad = np.ones((len(segs), n), bool)
        for r, sg in enumerate(segs):
            x[r, : len(sg)] = sg
            pad[r, : len(sg)] = False
        return torch.from_numpy(x).pin_memory(), torch.from_numpy(pad).pin_memory()


@torch.no_grad()
def vectors(head: Head, fr: Frames, idx: np.ndarray, batch: int = 512) -> np.ndarray:
    head.eval()
    out = []
    for b in range(0, len(idx), batch):
        x, pad = fr.batch(idx[b : b + batch])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out.append(head(x.cuda().float(), pad.cuda()).float().cpu().numpy())
    head.train()
    return np.concatenate(out)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=os.environ.get("AIRON_DATA", "/data/dy42/frames_ds"))
    p.add_argument("--texts", required=True, help="folder with texts/*.json ({position: transcript})")
    p.add_argument("--out", default="/data/dy42/head")
    p.add_argument("--steps", type=int, default=16000)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--ckpt-every", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--limit", type=int, default=0, help="dry run: use only this many pairs")
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    data = Path(a.data)
    torch.manual_seed(a.seed)

    s3get("dy36/dict/table.npz", out / "dict/table.npz")
    for f in ("subset.json", "emb.npz"):
        s3get(f"dy36/d/ctl_inputs/{f}", out / f"ctl/{f}")
    T = load_table(out / "dict")
    Y = Yard(T, out / "ctl")
    core = Frames([data / "core/core.npy"])
    want = np.concatenate([Y.sub, Y.refs])
    assert np.array_equal(core.pos, np.sort(want)), "core frames do not match the yardstick"
    core_idx = np.searchsorted(core.pos, want)  # rows of the table -> core rows, in yardstick order

    fr = Frames(list(data.glob("frames/*.npy")))
    texts: dict[int, str] = {}
    for f in sorted(Path(a.texts).glob("texts/*.json")):
        texts.update({int(k): v for k, v in json.loads(f.read_text()).items()})
    have = np.array([q in texts and len(texts[q].split()) >= 3 for q in fr.pos])
    log("utterances with frames", len(fr.pos), "with a transcript of >= 3 words", int(have.sum()), "texts", len(texts))
    assert have.mean() > 0.8, "most utterances must have frames and a usable transcript"
    ids = np.flatnonzero(have)
    if a.limit:  # dry run: a slice of the pairs (same code path, minutes instead of the full embedding)
        ids = ids[: a.limit]
    E = embed([texts[int(q)] for q in fr.pos[ids]], TARGET)  # row j of E is utterance ids[j]
    Et = torch.from_numpy(E).cuda()

    rng = np.random.default_rng(a.seed)
    perm = rng.permutation(len(ids))
    nval = min(len(ids) // 20, 2000)
    val, trn = perm[:nval], perm[nval:]
    log("train pairs", len(trn), "held back", nval, "passes over the data", round(a.steps * a.batch / len(trn), 2))

    head = Head(layers=a.layers).cuda()
    nparam = sum(x.numel() for x in head.parameters())
    opt = torch.optim.AdamW(head.parameters(), lr=3e-4, weight_decay=0.01)
    import airon

    start = 0
    ck = airon.resume()
    if ck:
        st = torch.load(ck, map_location="cuda", weights_only=False)
        head.load_state_dict(st["head"])
        opt.load_state_dict(st["opt"])
        start = st["step"]
        log("resumed from", ck, "step", start)
    run = airon.init()
    run.config.update({"steps": a.steps, "layers": a.layers, "batch": a.batch, "seed": a.seed, "lr": 3e-4,
                       "temperature": 0.05, "target": TARGET, "input": f"wav2vec2-960h layer {LAYER}, 12.5 Hz",
                       "train_pairs": int(len(trn)), "head_params": int(nparam)})

    # batches drawn from a generator seeded by the step, so a resume continues the same sequence
    q: queue.Queue = queue.Queue(maxsize=8)

    def feeder() -> None:
        for step in range(start + 1, a.steps + 1):
            bi = np.random.default_rng([a.seed, step]).choice(trn, size=a.batch, replace=False)
            q.put((step, bi, *fr.batch(ids[bi])))

    threading.Thread(target=feeder, daemon=True).start()

    def evaluate(step: int) -> dict:
        vv = vectors(head, fr, ids[val])
        cos_true = float((vv * E[val]).sum(1).mean())
        top1 = float(((vv @ E[val].T).argmax(1) == np.arange(len(val))).mean())
        V = vectors(head, core, core_idx)
        r = Y.score(V[len(Y.sub) :], V[: len(Y.sub)])
        m = {"eval/cos_true": cos_true, "eval/top1_of_val": top1, "eval/content_phrase_head": r["diff"],
             "eval/content_phrase_head_ci_lo": r["ci95"][0], "eval/content_phrase_head_ci_hi": r["ci95"][1]}
        run.log(m, step=step)
        log("eval step", step, json.dumps(m))
        return {"reading": r, "cos_true": cos_true, "top1": top1}

    t0 = time.time()
    res = None
    for _ in range(start + 1, a.steps + 1):
        step, bi, x, pad = q.get()
        lr = 3e-4 * min(1.0, step / 200) * (0.5 * (1 + np.cos(np.pi * min(1.0, step / a.steps))))
        for g in opt.param_groups:
            g["lr"] = lr
        with torch.autocast("cuda", dtype=torch.bfloat16):
            v = head(x.cuda(non_blocking=True).float(), pad.cuda(non_blocking=True))
        logits = v.float() @ Et[bi].T / 0.05
        lab = torch.arange(len(bi), device="cuda")
        loss = (F.cross_entropy(logits, lab) + F.cross_entropy(logits.T, lab)) / 2
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        opt.step()
        if step % 50 == 0:
            run.log({"train/loss": float(loss), "train/lr": lr, "train/grad_norm": float(gn)}, step=step,
                    samples=a.batch)
        if step % 500 == 0:
            log("step", step, "loss", round(float(loss), 3), round(time.time() - t0), "s")
        if step % a.ckpt_every == 0 or step == a.steps:
            res = evaluate(step)
            path = out / "state.pt"
            torch.save({"head": head.state_dict(), "opt": opt.state_dict(), "step": step}, path)
            run.checkpoint(str(path), step=step)

    r = res["reading"]
    meta = {"reading": r, "gate": GATE, "passed": bool(r["diff"] >= GATE), "n_train": int(len(trn)), "steps": a.steps,
            "head_layers": a.layers, "head_params": int(nparam), "cos_true": res["cos_true"], "top1_of_val": res["top1"],
            "references": "DY-41/35 (200k pairs, 16k steps) +0.232 [+0.218, +0.246]; text +0.41; wav2vec2-L12 mean +0.104"}
    log("PHRASE VECTOR", json.dumps(meta))
    torch.save({"head": head.state_dict(), "meta": meta}, out / "phrase_head.pt")
    run.checkpoint(str(out / "phrase_head.pt"), step=a.steps, label="best")
    (out / "phrase_vec.json").write_text(json.dumps(meta, indent=1))
    log("DY42 HEAD DONE")


if __name__ == "__main__":
    main()
