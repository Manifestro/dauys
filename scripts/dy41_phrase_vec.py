"""Phrase vectoriser: can a small head turn speech features of a whole utterance into a vector of its content?

Teacher signal (training target only; text never enters the dauys model): all-mpnet-base-v2 sentence embedding of the
Whisper large-v3-turbo transcript of the utterance. The yardstick uses another text model (all-MiniLM-L6-v2), so the
head is not fitted to the measuring model itself.
Input: wav2vec2-large-960h-lv60-self hidden state 12 (best frame feature on the yardstick, +0.104 as a plain mean),
50 Hz averaged to 12.5 Hz. Head: linear 1024 -> 512, 2 transformer layers (8 heads), attention pooling with a learned
query, linear -> 768, L2. Loss: symmetric InfoNCE between speech and text vectors of a batch (temperature 0.05).
Training utterances: N_TRAIN from training shards, 2-15 s, from recordings outside the yardstick's subset and
references (no recording shared with the measurement). 5 % of them held back for a sanity check of the fit.
Reading: DY-36's yardstick (dy36_sweep.Yard) on the head's vectors of the 500 references and the 8,000 subset
utterances; references: text +0.41, best speech feature so far +0.104; start rule for a thought: >= +0.25.

    python scripts/dy41_phrase_vec.py --out /data/pv
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, str(Path(__file__).parent))
import dy36_common
from dy36_common import log, utt_job
from dy36_dict import load_table
from dy36_sweep import Yard
from dy40_mimi import BUCKET, SR

N_TRAIN, LAYER, POOL = 24000, 12, 4
TARGET = "sentence-transformers/all-mpnet-base-v2"
ASR = "openai/whisper-large-v3-turbo"
GATE = 0.25


def s3get(key: str, dst: Path) -> None:
    if not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(f"aws s3 cp {BUCKET}/dauys/{key} {dst} --only-show-errors", shell=True, check=True)


def whisper(wavs: list[np.ndarray]) -> list[str]:
    from transformers import pipeline

    asr = pipeline("automatic-speech-recognition", model=ASR, device=torch.device("cuda"), torch_dtype=torch.float16,
                   chunk_length_s=30)
    res = asr([{"raw": w.astype(np.float32) / 32768.0, "sampling_rate": SR} for w in wavs], batch_size=96,
              generate_kwargs={"language": "en", "task": "transcribe"})
    del asr
    torch.cuda.empty_cache()
    return [r["text"].strip() for r in res]


def embed(texts: list[str], name: str) -> np.ndarray:
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name)
    m = AutoModel.from_pretrained(name).cuda().eval()
    out = []
    with torch.no_grad():
        for b in range(0, len(texts), 256):
            enc = tok(texts[b : b + 256], padding=True, truncation=True, max_length=256, return_tensors="pt").to("cuda")
            h = m(**enc).last_hidden_state
            msk = enc["attention_mask"][..., None].float()
            out.append(F.normalize((h * msk).sum(1) / msk.sum(1).clamp(min=1), dim=-1).cpu().numpy())
    del m
    torch.cuda.empty_cache()
    return np.concatenate(out)


def w2v_frames(wavs: list[np.ndarray], batch: int = 48) -> list[torch.Tensor]:
    """hidden state LAYER of wav2vec2-960h, 50 Hz -> 12.5 Hz (mean of POOL frames), fp16 on CPU."""
    from transformers import Wav2Vec2ForCTC

    m = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h-lv60-self").wav2vec2.cuda().eval()
    out: list[torch.Tensor] = [None] * len(wavs)  # type: ignore
    order = np.argsort(-np.array([len(w) for w in wavs]))
    t0 = time.time()
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
                f = h[r, : nf // POOL * POOL].float().reshape(-1, POOL, h.shape[-1]).mean(1)
                out[i] = f.half().cpu()
            if (b // batch) % 200 == 0:
                log("w2v", b + len(idx), "of", len(order), round(time.time() - t0), "s")
    del m
    torch.cuda.empty_cache()
    return out


class Head(nn.Module):
    def __init__(self, din: int = 1024, d: int = 512, dout: int = 768, layers: int = 2) -> None:
        super().__init__()
        self.inp = nn.Linear(din, d)
        layer = nn.TransformerEncoderLayer(d, 8, 4 * d, dropout=0.1, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers)
        self.q = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.att = nn.MultiheadAttention(d, 8, batch_first=True)
        self.out = nn.Linear(d, dout)

    def forward(self, x: torch.Tensor, pad: torch.Tensor) -> torch.Tensor:
        h = self.enc(self.inp(x), src_key_padding_mask=pad)
        z, _ = self.att(self.q.expand(len(x), -1, -1), h, h, key_padding_mask=pad)
        return F.normalize(self.out(z[:, 0]), dim=-1)


def batchify(frames: list[torch.Tensor], idx) -> tuple[torch.Tensor, torch.Tensor]:
    n = max(len(frames[i]) for i in idx)
    x = torch.zeros(len(idx), n, frames[idx[0]].shape[1])
    pad = torch.ones(len(idx), n, dtype=torch.bool)
    for r, i in enumerate(idx):
        x[r, : len(frames[i])] = frames[i].float()
        pad[r, : len(frames[i])] = False
    return x.cuda(), pad.cuda()


@torch.no_grad()
def vectors(head: Head, frames: list[torch.Tensor], batch: int = 256) -> np.ndarray:
    head.eval()
    out = []
    for b in range(0, len(frames), batch):
        x, pad = batchify(frames, list(range(b, min(b + batch, len(frames)))))
        out.append(head(x, pad).cpu().numpy())
    head.train()
    return np.concatenate(out)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="/data/pv")
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--n-train", type=int, default=N_TRAIN)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--reuse-texts", action="store_true", help="take the transcripts from S3 dy41/phrase_vec/ instead of Whisper")
    p.add_argument("--chunk", type=int, default=20000)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    torch.manual_seed(a.seed)
    out = Path(a.out)
    s3get("dy36/dict/table.npz", out / "dict/table.npz")
    for f in ("subset.json", "emb.npz"):
        s3get(f"dy36/d/ctl_inputs/{f}", out / f"ctl/{f}")
    T = load_table(out / "dict")
    Y = Yard(T, out / "ctl")
    dy36_common.IDX_DIR = out / "idx"
    dy36_common.HELD_DIR = out / "held_none"
    au = dy36_common.Audio(json.loads(str(T["files"])))

    # training utterances: training shards, 2-15 s, recordings outside the yardstick (subset, references, randoms)
    used_eps = set(T["ep"][np.concatenate([Y.sub, Y.refs, Y.rand])].tolist())
    ok = (~T["held"]) & (T["len"] >= 100) & (T["len"] <= 750) & ~np.isin(T["ep"], list(used_eps))
    rng = np.random.default_rng(a.seed)
    tr = rng.choice(np.flatnonzero(ok), size=min(a.n_train, int(ok.sum())), replace=False)
    assert not np.isin(T["ep"][tr], list(used_eps)).any()
    log("training utterances", len(tr), "from", int(ok.sum()), "eligible; recordings", len(np.unique(T["ep"][tr])))
    core = np.concatenate([Y.sub, Y.refs])
    # in chunks, so that only the 12.5 Hz frames (not the audio) of all training utterances stay in memory
    texts: list[str] = []
    reuse: list[str] | None = None
    if a.reuse_texts:  # transcripts of the same draw (same seed and size) from an earlier run
        s3get("dy41/phrase_vec/train_texts.json", out / "reuse_texts.json")
        reuse = json.loads((out / "reuse_texts.json").read_text())
        assert len(reuse) == len(tr), (len(reuse), len(tr))
    fr_tr: list[torch.Tensor] = []
    h_tr = 0.0
    for c in range(0, len(tr), a.chunk):
        wav = au.many([utt_job(T, int(u)) for u in tr[c : c + a.chunk]])
        h_tr += sum(len(w) for w in wav) / SR / 3600
        texts += reuse[c : c + a.chunk] if reuse is not None else whisper(wav)
        fr_tr += w2v_frames(wav)
        del wav
        log("chunk", c // a.chunk + 1, "of", -(-len(tr) // a.chunk), "train audio h so far", round(h_tr, 1))
    wav_core = au.many([utt_job(T, int(u)) for u in core])
    fr_core = w2v_frames(wav_core)
    del wav_core
    txt_f = out / "train_texts.json"
    txt_f.write_text(json.dumps(texts))
    keep = np.array([len(t.split()) >= 3 for t in texts])
    log("transcripts with >= 3 words", int(keep.sum()), "of", len(texts), "train audio h", round(h_tr, 1))
    E = embed(texts, TARGET)

    ids = np.flatnonzero(keep)
    rng.shuffle(ids)
    nval = min(len(ids) // 20, 2000)
    val, trn = ids[:nval], ids[nval:]
    head = Head(layers=a.layers).cuda()
    opt = torch.optim.AdamW(head.parameters(), lr=3e-4, weight_decay=0.01)
    Et = torch.from_numpy(E).cuda()
    import airon

    run = airon.init()
    t0 = time.time()
    for step in range(1, a.steps + 1):
        lr = 3e-4 * min(1.0, step / 200) * (0.5 * (1 + np.cos(np.pi * min(1.0, step / a.steps))))
        for g in opt.param_groups:
            g["lr"] = lr
        bi = rng.choice(trn, size=a.batch, replace=False)
        x, pad = batchify(fr_tr, list(bi))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            v = head(x, pad)
        logits = v.float() @ Et[bi].T / 0.05
        lab = torch.arange(len(bi), device="cuda")
        loss = (F.cross_entropy(logits, lab) + F.cross_entropy(logits.T, lab)) / 2
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        opt.step()
        if step % 100 == 0 or step == a.steps:
            vv = vectors(head, [fr_tr[i] for i in val])
            cos_true = float((vv * E[val]).sum(1).mean())
            top1 = float(((vv @ E[val].T).argmax(1) == np.arange(len(val))).mean())
            run.log({"train/loss": float(loss), "train/lr": lr, "train/grad_norm": float(gn),
                     "eval/cos_true": cos_true, "eval/top1_of_val": top1}, step=step)
            log("step", step, "loss", round(float(loss), 3), "val cos", round(cos_true, 3), "val top1",
                round(top1, 3), f"(of {len(val)})", round(time.time() - t0), "s")

    V = vectors(head, fr_core)
    nsub = len(Y.sub)
    r = Y.score(V[nsub:], V[:nsub])
    res = {"reading": r, "gate": GATE, "passed": bool(r["diff"] >= GATE), "n_train": int(len(trn)),
           "target": TARGET, "input": f"wav2vec2-960h layer {LAYER}, 12.5 Hz", "steps": a.steps, "head_layers": a.layers,
           "references": "text +0.41, wav2vec2-L12 mean +0.104, WavLM-L18 +0.096, Mimi/DY-41 <= +0.054"}
    log("PHRASE VECTOR", json.dumps(res))
    run.log({"eval/content_phrase_head": r["diff"], "eval/content_phrase_head_ci_lo": r["ci95"][0],
             "eval/content_phrase_head_ci_hi": r["ci95"][1]}, step=a.steps)
    (out / "phrase_vec.json").write_text(json.dumps(res, indent=1))
    torch.save({"head": head.state_dict(), "meta": res}, out / "phrase_head.pt")
    subprocess.run(f"aws s3 cp {out}/phrase_vec.json {BUCKET}/dauys/dy41/phrase_vec/ --only-show-errors && "
                   f"aws s3 cp {out}/phrase_head.pt {BUCKET}/dauys/dy41/phrase_vec/ --only-show-errors && "
                   f"aws s3 cp {txt_f} {BUCKET}/dauys/dy41/phrase_vec/ --only-show-errors", shell=True, check=True)
    log("PHRASE VECTOR DONE")


if __name__ == "__main__":
    main()
