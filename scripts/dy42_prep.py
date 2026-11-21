"""DY-42 data for the phrase vectoriser, computed once and kept as datasets (airon type: run jobs).

Utterances: every eligible one of the en-16k training shards (2-15 s, recordings outside DY-36's yardstick), ordered
so that the first 200,000 are exactly DY-41/34-35's draw (their Whisper transcripts are reused), the rest in a fixed
permutation. A position in this order is the utterance's id in every DY-42 file.

    frames   this machine's part (position % AIRON_SHARDS == AIRON_SHARD) -> wav2vec2-960h hidden state 12 at 12.5 Hz,
             fp16: $AIRON_OUTPUT/frames/<shard>_<chunk>.npy [sum T, 1024] + .npz (pos, off); shard 0 also writes the
             yardstick's 8,500 utterances (core/)
    texts    Whisper large-v3-turbo transcripts of positions >= 200,000 (this machine's part), plus the reused 200,000
             -> $AIRON_OUTPUT/texts/<shard>.json {pos: text}

    python scripts/dy42_prep.py frames|texts [--limit N]
"""

import argparse
import json
import os
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
from dy36_sweep import Yard
from dy40_mimi import BUCKET, SR
from dy41_phrase_vec import LAYER, POOL, whisper

N_REUSED = 200000


def s3get(key: str, dst: Path) -> None:
    if not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(f"aws s3 cp {BUCKET}/dauys/{key} {dst} --only-show-errors", shell=True, check=True)


def setup(work: Path):
    """table, yardstick, audio reader and the full utterance order (rows of the table)."""
    s3get("dy36/dict/table.npz", work / "dict/table.npz")
    for f in ("subset.json", "emb.npz"):
        s3get(f"dy36/d/ctl_inputs/{f}", work / f"ctl/{f}")
    T = load_table(work / "dict")
    Y = Yard(T, work / "ctl")
    dy36_common.IDX_DIR = work / "idx"
    dy36_common.HELD_DIR = work / "held_none"
    au = dy36_common.Audio(json.loads(str(T["files"])))
    # the same eligibility and draw as dy41_phrase_vec.py (seed 0), so positions < 200,000 are DY-41/34-35's pairs
    used_eps = set(T["ep"][np.concatenate([Y.sub, Y.refs, Y.rand])].tolist())
    ok = (~T["held"]) & (T["len"] >= 100) & (T["len"] <= 750) & ~np.isin(T["ep"], list(used_eps))
    elig = np.flatnonzero(ok)
    first = np.random.default_rng(0).choice(elig, size=N_REUSED, replace=False)
    rest = np.random.default_rng(1).permutation(np.setdiff1d(elig, first))
    order = np.concatenate([first, rest])
    assert len(np.unique(order)) == len(order) == len(elig)
    assert not np.isin(T["ep"][order], list(used_eps)).any(), "a training utterance shares a recording with the yardstick"
    log("eligible utterances", len(order), "recordings", len(np.unique(T["ep"][order])))
    return T, Y, au, order


def w2v_model():
    from transformers import Wav2Vec2ForCTC

    m = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h-lv60-self").wav2vec2
    # hidden_states[LAYER] is the input of layer LAYER+1 (stable layer norm: the final norm only touches the last
    # state), so layers past LAYER+1 are never needed: keep LAYER+1 of them, same numbers, ~45 % less compute
    m.encoder.layers = m.encoder.layers[: LAYER + 1]
    return m.cuda().eval()


def frames(m, wavs: list[np.ndarray], batch: int) -> list[np.ndarray]:
    out: list = [None] * len(wavs)
    order = np.argsort(-np.array([len(w) for w in wavs]))
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        for b in range(0, len(order), batch):
            idx = order[b : b + batch]
            n = max(len(wavs[i]) for i in idx)
            x = torch.zeros(len(idx), n)
            am = torch.zeros(len(idx), n, dtype=torch.long)
            for r, i in enumerate(idx):
                w = torch.from_numpy(wavs[i].astype(np.float32) / 32768.0)
                x[r, : len(w)] = (w - w.mean()) / (w.std() + 1e-7)
                am[r, : len(w)] = 1
            h = m(x.cuda(), attention_mask=am.cuda(), output_hidden_states=True).hidden_states[LAYER]
            for r, i in enumerate(idx):
                nf = max(POOL, (len(wavs[i]) - 400) // 320 + 1)
                out[i] = h[r, : nf // POOL * POOL].float().reshape(-1, POOL, h.shape[-1]).mean(1).half().cpu().numpy()
    return out


def save_frames(path: Path, pos: np.ndarray, fr: list[np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path.with_suffix(".npy"), np.concatenate(fr))
    off = np.concatenate([[0], np.cumsum([len(f) for f in fr])]).astype(np.int64)
    np.savez(path.with_suffix(".npz"), pos=pos.astype(np.int64), off=off)


def cmd_frames(a, T, Y, au, order, out: Path) -> None:
    k, n = int(os.environ.get("AIRON_SHARD", 0)), int(os.environ.get("AIRON_SHARDS", 1))
    mine = np.arange(k, len(order), n)
    if a.limit:
        mine = mine[: a.limit]
    m = w2v_model()
    if k == 0:  # the yardstick's subset + references (rows of the table, not positions)
        core = np.concatenate([Y.sub, Y.refs])
        lim = core[: a.limit] if a.limit else core
        save_frames(out / "core/core", lim, frames(m, au.many([utt_job(T, int(u)) for u in lim]), a.batch))
        log("core frames", len(lim))
    t0, h = time.time(), 0.0
    nch = -(-len(mine) // a.chunk)
    for c in range(nch):
        pos = mine[c * a.chunk : (c + 1) * a.chunk]
        wav = au.many([utt_job(T, int(order[p])) for p in pos])
        h += sum(len(w) for w in wav) / SR / 3600
        save_frames(out / f"frames/{k:02d}_{c:03d}", pos, frames(m, wav, a.batch))
        done = min((c + 1) * a.chunk, len(mine))
        print(f"::airon progress name=frames done={done} total={len(mine)} unit=utts", flush=True)
        log("chunk", c + 1, "of", nch, "utts", done, "audio h", round(h, 1), round(time.time() - t0), "s")
    (out / f"frames/{k:02d}_DONE.json").write_text(json.dumps({"shard": k, "shards": n, "utts": int(len(mine)),
                                                                "audio_h": round(h, 2), "eligible": int(len(order))}))


def cmd_texts(a, T, Y, au, order, out: Path) -> None:
    k, n = int(os.environ.get("AIRON_SHARD", 0)), int(os.environ.get("AIRON_SHARDS", 1))
    res: dict[int, str] = {}
    if k == 0:  # DY-41/34's transcripts of the same 200,000 utterances, checked by count
        s3get("dy41/phrase_vec/train_texts.json", out / "reuse.json")
        reuse = json.loads((out / "reuse.json").read_text())
        (out / "reuse.json").unlink()
        assert len(reuse) == N_REUSED, len(reuse)
        res.update({i: t for i, t in enumerate(reuse)})
    mine = np.arange(N_REUSED + k, len(order), n)
    if a.limit:
        mine = mine[: a.limit]
    t0 = time.time()
    nch = -(-len(mine) // a.chunk)
    for c in range(nch):
        pos = mine[c * a.chunk : (c + 1) * a.chunk]
        res.update(zip(pos.tolist(), whisper(au.many([utt_job(T, int(order[p])) for p in pos]))))
        done = min((c + 1) * a.chunk, len(mine))
        print(f"::airon progress name=whisper done={done} total={len(mine)} unit=utts", flush=True)
        log("chunk", c + 1, "of", nch, "utts", done, round(time.time() - t0), "s")
    (out / "texts").mkdir(parents=True, exist_ok=True)
    (out / f"texts/{k:02d}.json").write_text(json.dumps({str(p): t for p, t in res.items()}))
    log("texts", len(res))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["frames", "texts"])
    p.add_argument("--limit", type=int, default=0, help="dry run: this many utterances per machine")
    p.add_argument("--chunk", type=int, default=10000)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--work", default="/data/dy42")
    a = p.parse_args()
    out = Path(os.environ.get("AIRON_OUTPUT", "/data/dy42/out"))
    T, Y, au, order = setup(Path(a.work))
    out.mkdir(parents=True, exist_ok=True)
    (cmd_frames if a.cmd == "frames" else cmd_texts)(a, T, Y, au, order, out)
    (out / "summary.md").write_text(f"DY-42 {a.cmd}: shard {os.environ.get('AIRON_SHARD', 0)} of "
                                    f"{os.environ.get('AIRON_SHARDS', 1)}, eligible utterances {len(order)}\n")
    log("DY42", a.cmd, "DONE")


if __name__ == "__main__":
    main()
