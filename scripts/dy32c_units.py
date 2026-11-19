"""DY-32 track C units: six streams of WavLM-Large k-means codes per 20 ms frame (SpeechBrain DASB recipe).

Streams = WavLM-Large (microsoft/wavlm-large) hidden states of layers 1, 3, 7, 12, 18, 23 (hidden_states[L], the
waveform layer-normalised first, no output norm, as speechbrain DiscreteSSL/WavLM), each assigned to the nearest of
1000 centroids of SpeechBrain's LibriSpeech-960 k-means for that layer (speechbrain/hifigan-wavlm-k1000-LibriTTS,
kmeans/LibriSpeech_wavlm_k1000_L<L>.pt). These codes are decoded by that repo's discrete HiFi-GAN (all six, or any
subset). 50 Hz, no deduplication.

Pieces >= 5 s of every DuplexChat shard are cut into 20 s windows (+80 samples of look-ahead so a full window gives
exactly 1000 frames); the windows of a batch are padded to one length with an attention mask. Output ``<shard>.npz``:
``tok`` int16 [n_frames, 6] (layer order 1, 3, 7, 12, 18, 23), ``index`` [n_pieces, 2] (start, frames).

    python scripts/dy32c_units.py run --part 0/3 --max-shards 50 --work /opt/dlami/nvme/w0
    python scripts/dy32c_units.py known --shard la-00030 --work /opt/dlami/nvme/known
"""

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from dy21_train import held_out
from dy30_units import SRC, fetch, listing, sh

BUCKET = "s3://manifestro-dataset-876108973178-us-east-1-an/outputs"
DST = f"{BUCKET}/dauys/dy32/c/units"
SR = 16000
WIN = 20 * SR
LOOK = 80  # 400-sample receptive field, hop 320: WIN + 80 samples -> exactly WIN / 320 frames
HOP = 320
LAYERS = (1, 3, 7, 12, 18, 23)
K = 1000
VOC_REPO = "speechbrain/hifigan-wavlm-k1000-LibriTTS"
SSL = "microsoft/wavlm-large"
BATCH = 24


def n_frames(n_samples: int) -> int:
    return max((n_samples - 400) // HOP + 1, 0)


def centroids() -> torch.Tensor:
    """[6, 1000, 1024] float32 cluster centres of the SpeechBrain k-means (sklearn MiniBatchKMeans pickles)."""
    import joblib
    from huggingface_hub import hf_hub_download

    cs = []
    for L in LAYERS:
        km = joblib.load(hf_hub_download(VOC_REPO, f"kmeans/LibriSpeech_wavlm_k1000_L{L}.pt"))
        cs.append(torch.from_numpy(np.asarray(km.cluster_centers_, np.float32)))
    return torch.stack(cs)


class Units:
    def __init__(self, device, dtype=torch.float16) -> None:
        from transformers import WavLMModel

        self.device, self.dtype = device, dtype
        self.ssl = WavLMModel.from_pretrained(SSL).to(device).eval()
        self.C = centroids().to(device)  # [6, K, D]
        self.C2 = (self.C ** 2).sum(-1)  # [6, K]

    @torch.no_grad()
    def feats(self, wavs: list[np.ndarray]) -> tuple[list[torch.Tensor], list[int]]:
        """Hidden states of LAYERS for a batch of float32 waves: list of [B, T, D] (fp32), frames per wave."""
        n = [n_frames(len(w)) for w in wavs]
        T = max(len(w) for w in wavs)
        x = torch.zeros(len(wavs), T)
        mask = torch.zeros(len(wavs), T, dtype=torch.long)
        for i, w in enumerate(wavs):
            t = torch.from_numpy(w)
            x[i, : len(w)] = (t - t.mean()) / torch.sqrt(t.var(unbiased=False) + 1e-5)
            mask[i, : len(w)] = 1
        x, mask = x.to(self.device), mask.to(self.device)
        with torch.autocast("cuda", dtype=self.dtype, enabled=self.dtype != torch.float32):
            out = self.ssl(x, attention_mask=mask, output_hidden_states=True)
        return [out.hidden_states[L].float() for L in LAYERS], n

    @torch.no_grad()
    def codes(self, wavs: list[np.ndarray]) -> list[np.ndarray]:
        hs, n = self.feats(wavs)
        res = []
        for j, h in enumerate(hs):
            d = self.C2[j][None, None] - 2 * h @ self.C[j].T  # ||x||^2 is constant per row
            res.append(d.argmin(-1))
        tok = torch.stack(res, -1).cpu().numpy().astype(np.int16)  # [B, T, 6]
        return [tok[i, : n[i]] for i in range(len(wavs))]


def piece_windows(w: np.ndarray) -> list[np.ndarray]:
    """20 s windows (+ look-ahead); a tail shorter than 1 s is dropped (its few frames are lost)."""
    return [w[i : i + WIN + LOOK] for i in range(0, len(w), WIN) if len(w) - i >= SR]


def piece_codes(u: Units, pieces: list[np.ndarray], batch: int = BATCH) -> list[np.ndarray]:
    """Codes per piece [frames, 6]; each 20 s window gives WIN / HOP frames (last one: what fits)."""
    wins = [(k, j, x) for k, w in enumerate(pieces) for j, x in enumerate(piece_windows(w))]
    order = sorted(range(len(wins)), key=lambda i: -len(wins[i][2]))  # similar lengths together
    out: dict[int, np.ndarray] = {}
    for b in range(0, len(order), batch):
        part = order[b : b + batch]
        res = u.codes([wins[i][2].astype(np.float32) / 32768.0 for i in part])
        for i, r in zip(part, res):
            out[i] = r[: WIN // HOP]
    per = [[] for _ in pieces]
    for i, (k, j, _) in enumerate(wins):
        per[k].append(out[i])
    return [np.concatenate(r) if r else np.zeros((0, len(LAYERS)), np.int16) for r in per]


def save(path: Path, toks: list[np.ndarray]) -> int:
    index, pos = [], 0
    for t in toks:
        index.append((pos, len(t)))
        pos += len(t)
    tok = np.concatenate(toks) if toks else np.zeros((0, len(LAYERS)), np.int16)
    np.savez(path, tok=tok, index=np.asarray(index, np.int64).reshape(-1, 2))
    return pos


def cmd_run(a) -> None:
    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    u = Units(torch.device("cuda"))
    done = set(listing(a.dst, dirs=False))
    if a.shards:
        mine = [s for s in a.shards.split(",") if s not in done]
    else:
        k, n = map(int, a.part.split("/"))
        mine = [s for s in listing(a.src, dirs=True)[k::n] if s not in done and (a.held or not held_out(s))]
    if a.max_shards:
        mine = mine[: a.max_shards]
    print(json.dumps({"part": a.part, "todo": len(mine), "already_in_dst": len(done)}), flush=True)
    dl, up = ThreadPoolExecutor(1), ThreadPoolExecutor(2)
    uploads, failed = [], []
    lock = threading.Lock()

    def upload(f: Path, s: str) -> None:
        try:
            sh(f"aws s3 cp {f} {a.dst}/{s}.npz --only-show-errors")
            if not a.keep:
                f.unlink()
        except Exception as e:
            with lock:
                failed.append((s, str(e)[-200:]))

    nxt = dl.submit(fetch, a.src, mine[0], work) if mine else None
    t_all, h_all = time.time(), 0.0
    for j in range(len(mine)):
        t0 = time.time()
        s, pieces, secs = nxt.result()
        wait = time.time() - t0
        nxt = dl.submit(fetch, a.src, mine[j + 1], work) if j + 1 < len(mine) else None
        toks = piece_codes(u, pieces)
        out = (Path(a.keep) if a.keep else work) / f"{s}.npz"
        out.parent.mkdir(parents=True, exist_ok=True)
        nf = save(out, toks)
        del pieces, toks
        uploads.append(up.submit(upload, out, s))
        el = time.time() - t0
        h_all += secs / 3600
        print(json.dumps({"part": a.part, "shard": s, "held": held_out(s), "done": j + 1, "of": len(mine),
                          "hours": round(secs / 3600, 2), "frames": nf, "x_realtime": round(secs / el, 1),
                          "waited_for_download_s": round(wait, 1),
                          "x_realtime_total": round(h_all * 3600 / (time.time() - t_all), 1)}), flush=True)
    for f in uploads:
        f.result()
    if failed:
        print("UPLOAD FAILED", json.dumps(failed), flush=True)
        sys.exit(1)
    print(f"C UNITS DONE part {a.part}", round(h_all, 1), "h", flush=True)


def cmd_known(a) -> None:
    """Batched fp16 path vs a one-piece-at-a-time fp32 path (whole piece, no windows) on the first pieces of a shard:
    share of identical codes per stream. Window cuts and fp16 change a few codes; the bar is >= 0.85 per stream."""
    dev = torch.device("cuda")
    u = Units(dev)
    _, pieces, _ = fetch(a.src, a.shard, Path(a.work))
    use = [w[: 50 * SR] for w in pieces[: a.pieces]]
    t0 = time.time()
    new = piece_codes(u, use)
    torch.cuda.synchronize()
    print("batched x_realtime", round(sum(len(w) for w in use) / SR / (time.time() - t0), 1), flush=True)
    u.dtype = torch.float32
    ref = [u.codes([w.astype(np.float32) / 32768.0])[0] for w in use]
    agree = np.zeros(len(LAYERS))
    nf = 0
    for r, x in zip(ref, new):
        m = min(len(r), len(x))
        agree += (r[:m] == x[:m]).sum(0)
        nf += m
    agree = agree / max(nf, 1)
    lens = [(len(r), len(x)) for r, x in zip(ref, new)]
    res = {"shard": a.shard, "pieces": len(use), "frames": nf, "agree_per_stream": dict(zip(map(str, LAYERS), agree.round(4).tolist())),
           "len_ref_new_first": lens[:5], "pass": bool(agree.min() >= 0.85)}
    print("KNOWN", json.dumps(res), flush=True)
    sys.exit(0 if res["pass"] else 1)


def main() -> None:
    p = argparse.ArgumentParser()
    s = p.add_subparsers(dest="cmd", required=True)
    x = s.add_parser("run")
    x.add_argument("--part", default="0/1")
    x.add_argument("--shards", default="")
    x.add_argument("--work", required=True)
    x.add_argument("--src", default=SRC)
    x.add_argument("--dst", default=DST)
    x.add_argument("--max-shards", type=int, default=0)
    x.add_argument("--held", action="store_true", help="include held-out shards in --part")
    x.add_argument("--keep", default="", help="also keep the npz in this dir")
    x = s.add_parser("known")
    x.add_argument("--shard", required=True)
    x.add_argument("--work", required=True)
    x.add_argument("--src", default=SRC)
    x.add_argument("--pieces", type=int, default=20)
    a = p.parse_args()
    {"run": cmd_run, "known": cmd_known}[a.cmd](a)


if __name__ == "__main__":
    main()
