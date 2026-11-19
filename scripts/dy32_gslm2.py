"""DY-32 track E: GSLM2 (arXiv 2611.01143, github.com/facebookresearch/gslm2, CC BY-NC 4.0, research use only).

Runs in the release's LM environment (Python 3.12, PyTorch 2.9, fairseq2 0.7) with the gslm2 package installed;
everything model-side is the release's own code (SpidR 2.0 encoder, multi-stream OPT + Depformer). Nothing here is
shared with other DY-32 tracks.

Units: SpidR 2.0 prediction heads of layers 5..10, argmax per frame (50 Hz, 320-sample hop), raw codes 0..255, streams
in layer order 5,6,7,8,9,10. "dedup" = the release's cross-stream deduplication (a frame is dropped only if all six
streams equal the previous frame; durations are not kept). The LM sees raw code + 2 (ids 0/1 are reserved, 1 = pad).

    wavs   --in DIR --glob 'prompt_*.wav' --out DIR      units of whole wav files (full-rate and dedup JSONL)
    gen    --lm DIR --prompts DIR --out DIR             continue every prompt (batched, no padding)
    shards --part k/n --work DIR --dst S3               full-rate units of DuplexChat training shards (npz per shard)
    score  --units DIR --lms A,B --out F                 held-out loss of LMs on our units (train's validation windows)
    train  --units DIR --init DIR --out DIR             fine-tune a released LM on our units (Airon-tracked)
"""

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

SR = 16000
HOP = 320
LAYERS = [5, 6, 7, 8, 9, 10]
OFFSET = 2
BUCKET = "s3://manifestro-dataset-876108973178-us-east-1-an/outputs"
SRC = f"{BUCKET}/duplexchat/en-16k"
EXPERIMENT = "DY-32: канал, через который слова выживают, без текста — три идеи и повтор GSLM параллельными ветками"


def held_out(name: str) -> bool:  # identical to scripts/dy21_train.held_out (not imported: that module needs dauys)
    return int(hashlib.sha1(name.encode()).hexdigest(), 16) % 50 == 0


def sh(cmd: str) -> str:
    return subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True).stdout


def fmt(streams: np.ndarray) -> str:
    return "|".join(" ".join(map(str, s.tolist())) for s in streams)


def dedup(codes: np.ndarray) -> np.ndarray:
    """Cross-stream dedup of (6, T): keep frame t if any stream differs from frame t-1 (release rule)."""
    if codes.shape[1] == 0:
        return codes
    keep = np.ones(codes.shape[1], bool)
    keep[1:] = (codes[:, 1:] != codes[:, :-1]).any(0)
    return codes[:, keep]


# ---------------------------------------------------------------- encoder
class Encoder:
    def __init__(self, ckpt: str, device: torch.device):
        from gslm2.encoder.config import SpidRConfig
        from gslm2.encoder.models import build_model

        self.model = build_model(checkpoint=Path(ckpt), model_type="spidr").to(device).eval()
        self.conv = SpidRConfig().extractor_conv_layer_config
        self.device = device

    @torch.no_grad()
    def codes(self, wavs: list[np.ndarray]) -> list[np.ndarray]:
        """Full-rate (6, T) uint8 codes per waveform (float32 in [-1, 1]); one padded batch, as the release does."""
        from gslm2.encoder.data.collators import conv_length
        from gslm2.encoder.data.masking.speech import build_attention_mask
        from gslm2.shared.audio_io import normalize_waveform

        xs = [normalize_waveform(torch.from_numpy(np.ascontiguousarray(w, dtype=np.float32))) for w in wavs]
        lens = torch.tensor([len(x) for x in xs])
        x = torch.nn.utils.rnn.pad_sequence(xs, batch_first=True).to(self.device)
        cl = conv_length(self.conv, lens.to(self.device))
        mask = build_attention_mask(cl)
        outs = self.model.get_codebooks(x, attention_mask=mask, layers=LAYERS, onehot=False)
        per = [outs[l - 1].reshape(len(xs), -1, outs[l - 1].shape[-1]).argmax(-1) for l in LAYERS]
        st = torch.stack(per, 1).to(torch.uint8).cpu().numpy()  # (B, 6, T)
        return [st[b, :, : int(cl[b])] for b in range(len(xs))]


def read_wav(p: Path) -> np.ndarray:
    import soundfile as sf

    w, sr = sf.read(str(p), dtype="float32", always_2d=True)
    assert sr == SR, (p, sr)
    return w.mean(1)


def cmd_wavs(a) -> None:
    enc = Encoder(a.encoder, torch.device("cuda"))
    files = sorted(Path(a.inp).glob(a.glob))
    out = Path(a.out)
    for sub in ("fullrate", "dedup"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    stats, allf, alld = [], [], []
    for i in range(0, len(files), a.batch):
        part = files[i : i + a.batch]
        for f, c in zip(part, enc.codes([read_wav(f) for f in part])):
            d = dedup(c.astype(np.int64))
            key = f.stem.split("_")[-1]
            base = {"fileid": f.stem, "audio": f"{key}.wav", "source_wav": f.name, "unit_offset": 0,
                    "streams": "SpidR 2.0 layers 5|6|7|8|9|10", "frame_rate_hz": 50}
            rf = {**base, "units": fmt(c), "audio_repr": fmt(c), "file_name": f.name, "frames": int(c.shape[1]),
                  "deduplicated": False}
            rd = {**base, "units": fmt(d), "audio_repr": fmt(d), "file_name": f.name, "frames": int(d.shape[1]),
                  "fullrate_frames": int(c.shape[1]), "deduplicated": True}
            (out / "fullrate" / f"{key}.jsonl").write_text(json.dumps(rf) + "\n")
            (out / "dedup" / f"{key}.jsonl").write_text(json.dumps(rd) + "\n")
            allf.append(rf)
            alld.append(rd)
            stats.append((c.shape[1], d.shape[1]))
    (out / "fullrate" / "all.jsonl").write_text("".join(json.dumps(r) + "\n" for r in allf))
    (out / "dedup" / "all.jsonl").write_text("".join(json.dumps(r) + "\n" for r in alld))
    s = np.array(stats)
    print(json.dumps({"files": len(files), "fullrate_frames_mean": float(s[:, 0].mean()),
                      "dedup_ratio": float(s[:, 1].sum() / s[:, 0].sum())}), flush=True)


# ---------------------------------------------------------------- generation
def load_lm(lm_dir: str, device: torch.device, dtype=torch.bfloat16):
    from gslm2.lm.inference import load_opt_model

    d = Path(lm_dir)
    loaded = load_opt_model(d / "model.pt", device, model_config=d / "config.yaml")
    loaded.model.to(dtype)
    return loaded


@torch.inference_mode()
def decode_batch(loaded, seqs: torch.Tensor, steps: int, temperature: float, top_p: float, anneal: float) -> torch.Tensor:
    """seqs: (B, 6, T) LM ids (raw + offset), equal lengths. Appends ``steps`` sampled frames; returns (B, 6, T+steps).
    Same per-step computation as gslm2.lm.inference.decode_tokens (full recompute, Depformer sampling), batched."""
    from fairseq2.nn import BatchLayout

    B, S, _ = seqs.shape
    m = loaded.model
    for _ in range(steps):
        flat = seqs.reshape(B * S, -1)
        layout = BatchLayout(shape=flat.shape, seq_lens=[flat.shape[-1]] * flat.shape[0], packed=False, device=flat.device)
        h = m.forward(flat, layout, return_decoder_outputs=True)
        cands, _ = m.forward_depformer_inference(h, temperature, top_p, temp_anneal=anneal)
        nxt = torch.stack(cands, 1)[:, :, 0, 0]  # (B, 6)
        seqs = torch.cat([seqs, nxt[:, :, None]], -1)
    return seqs


def cmd_gen(a) -> None:
    device = torch.device("cuda")
    t0 = time.time()
    loaded = load_lm(a.lm, device)
    n_params = sum(p.numel() for p in loaded.model.parameters())
    rows = [json.loads(l) for l in (Path(a.prompts) / "all.jsonl").read_text().splitlines()]
    if a.limit:
        rows = rows[: a.limit]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cont_frames = int(round(a.cont_s * 50))
    items = []
    for r in rows:
        full = np.array([[int(t) for t in s.split()] for s in r["units"].split("|")], np.int64)
        d = dedup(full)
        ratio = d.shape[1] / full.shape[1]
        steps = min(int(cont_frames * ratio * 1.5), cont_frames)  # release heuristic (lm/generation/speech.py)
        items.append({"row": r, "d": d, "ratio": ratio, "steps": steps})
    order = sorted(range(len(items)), key=lambda i: items[i]["d"].shape[1])
    print(json.dumps({"lm": a.lm, "params": n_params, "prompts": len(items), "load_s": round(time.time() - t0, 1)}), flush=True)
    for g in range(0, len(order), a.group):
        idx = order[g : g + a.group]
        L = min(items[i]["d"].shape[1] for i in idx)
        if a.max_ctx:
            L = min(L, a.max_ctx)
        steps = max(items[i]["steps"] for i in idx)
        seqs = torch.tensor(np.stack([items[i]["d"][:, -L:] for i in idx]) + OFFSET, device=device)
        torch.manual_seed(a.seed + g)
        t1 = time.time()
        res = decode_batch(loaded, seqs, steps, a.temperature, a.top_p, a.anneal).cpu().numpy() - OFFSET
        for j, i in enumerate(idx):
            it = items[i]
            cont = res[j, :, L : L + it["steps"]]
            bad = int((cont < 0).sum())
            key = it["row"]["audio"].removesuffix(".wav")
            rec = {"fileid": key, "audio": f"{key}.wav", "units": fmt(np.clip(cont, 0, 255)),
                   "context_units": fmt(it["d"][:, -L:]), "unit_offset": 0, "deduplicated": True,
                   "streams": "SpidR 2.0 layers 5|6|7|8|9|10", "continuation_positions": int(cont.shape[1]),
                   "context_positions": int(L), "prompt_dedup_positions": int(it["d"].shape[1]),
                   "prompt_fullrate_frames": int(it["row"]["frames"]), "prompt_dedup_ratio": round(it["ratio"], 4),
                   "context_seconds_approx": round(L / it["d"].shape[1] * it["row"]["frames"] / 50, 2),
                   "special_ids_clipped": bad}
            (out / f"{key}.jsonl").write_text(json.dumps(rec) + "\n")
        print(json.dumps({"group": g // a.group, "n": len(idx), "ctx": L, "steps": steps,
                          "s": round(time.time() - t1, 1)}), flush=True)
    recs = [json.loads((out / f"{items[i]['row']['audio'].removesuffix('.wav')}.jsonl").read_text()) for i in range(len(items))]
    (out / "all.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
    info = {"lm": a.lm, "params": n_params, "sampling": {"temperature": a.temperature, "top_p": a.top_p,
            "depformer_temperature_annealing": a.anneal, "seed": a.seed},
            "continuation_seconds": a.cont_s, "steps_rule": "min(int(500 * dedup_ratio_of_prompt * 1.5), 500)",
            "prompts": len(items), "group": a.group, "max_ctx": a.max_ctx, "seconds": round(time.time() - t0, 1)}
    (out / "info.json").write_text(json.dumps(info, indent=2))
    print("GEN DONE", json.dumps(info), flush=True)


# ---------------------------------------------------------------- training-shard units
def fetch(s: str, work: Path, min_s: float) -> tuple[str, list[np.ndarray], float]:
    d = work / s
    d.mkdir(parents=True, exist_ok=True)
    for fn in ("index.npy", "segs.bin"):
        sh(f"aws s3 cp {SRC}/{s}/{fn} {d}/{fn} --only-show-errors")
    idx = np.load(d / "index.npy")
    segs = np.fromfile(d / "segs.bin", dtype=np.int16)
    sh(f"rm -rf {d}")
    pieces = [segs[st : st + ln] for st, ln in idx if ln >= min_s * SR]
    return s, pieces, sum(len(w) for w in pieces) / SR


def shard_codes(enc: Encoder, pieces: list[np.ndarray], seg_s: float, budget_s: float) -> list[np.ndarray]:
    """Full-rate codes per piece: pieces cut into <= seg_s windows (whole windows through the encoder, as the
    release does with whole files), batched by length, codes of a piece's windows joined in order."""
    seg = int(seg_s * SR)
    wins = [(k, i) for k, w in enumerate(pieces) for i in range(0, len(w), seg) if len(w) - i >= HOP * 4]
    wins.sort(key=lambda ki: -min(seg, len(pieces[ki[0]]) - ki[1]))
    per: dict[tuple[int, int], np.ndarray] = {}
    b = 0
    while b < len(wins):
        n = max(1, int(budget_s * SR // min(seg, len(pieces[wins[b][0]]) - wins[b][1])))
        part = wins[b : b + n]
        res = enc.codes([pieces[k][i : i + seg].astype(np.float32) / 32768.0 for k, i in part])
        for ki, r in zip(part, res):
            per[ki] = r
        b += n
    out = [[] for _ in pieces]
    for (k, i) in sorted(per):
        out[k].append(per[(k, i)])
    return [np.concatenate(o, 1) if o else np.zeros((6, 0), np.uint8) for o in out]


def cmd_shards(a) -> None:
    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    enc = Encoder(a.encoder, torch.device("cuda"))
    names = sorted(l.split()[-1].rstrip("/") for l in sh(f"aws s3 ls {SRC}/").splitlines() if l.strip().startswith("PRE"))
    done = {l.split()[-1].removesuffix(".npz") for l in sh(f"aws s3 ls {a.dst}/ || true").splitlines() if l.endswith(".npz")}
    if a.shards:
        mine = a.shards.split(",")
    else:
        k, n = map(int, a.part.split("/"))
        mine = [s for s in names if (held_out(s) == a.held)][k::n]
    mine = [s for s in mine if s not in done][: a.max_shards or None]
    print(json.dumps({"todo": len(mine), "already": len(done), "held": a.held}), flush=True)
    dl, up = ThreadPoolExecutor(1), ThreadPoolExecutor(2)
    failed, lock, ups = [], threading.Lock(), []

    def upload(f: Path, s: str) -> None:
        try:
            sh(f"aws s3 cp {f} {a.dst}/{s}.npz --only-show-errors")
            if not a.keep:
                f.unlink()
        except Exception as e:
            with lock:
                failed.append((s, str(e)[-200:]))

    nxt = dl.submit(fetch, mine[0], work, a.min_s) if mine else None
    t_all, h_all = time.time(), 0.0
    for j in range(len(mine)):
        t0 = time.time()
        s, pieces, secs = nxt.result()
        wait = time.time() - t0
        nxt = dl.submit(fetch, mine[j + 1], work, a.min_s) if j + 1 < len(mine) else None
        codes = shard_codes(enc, pieces, a.seg_s, a.budget_s)
        lens = [c.shape[1] for c in codes]
        index = np.stack([np.cumsum([0] + lens[:-1]), lens], 1).astype(np.int64)
        f = work / f"{s}.npz"
        np.savez(f, codes=np.concatenate(codes, 1), index=index)
        ups.append(up.submit(upload, f, s))
        el = time.time() - t0
        h_all += secs / 3600
        print(json.dumps({"shard": s, "held": held_out(s), "done": j + 1, "of": len(mine), "hours": round(secs / 3600, 2),
                          "frames": int(sum(lens)), "x_realtime": round(secs / el, 1), "waited_s": round(wait, 1),
                          "hours_total": round(h_all, 1),
                          "x_realtime_total": round(h_all * 3600 / (time.time() - t_all), 1)}), flush=True)
        del pieces, codes
        if a.stop_file and Path(a.stop_file).exists():
            print("STOP FILE seen", flush=True)
            break
    for u in ups:
        u.result()
    if failed:
        print("UPLOAD FAILED", json.dumps(failed), flush=True)
    print("SHARDS DONE", round(h_all, 1), "h", flush=True)


# ---------------------------------------------------------------- training
def load_pieces(files: list[Path], T: int) -> list[np.ndarray]:
    """Dedup (6, N) int16 LM ids per piece, only pieces with at least T + 1 positions."""
    out = []
    for f in files:
        z = np.load(f)
        c, index = z["codes"], z["index"]
        for st, n in index:
            d = dedup(c[:, st : st + n].astype(np.int16))
            if d.shape[1] >= T + 1:
                out.append(d + OFFSET)
    return out


def windows(pieces: list[np.ndarray], cum: np.ndarray, T: int, n: int, rng) -> np.ndarray:
    """n random windows of T + 1 positions, a piece drawn in proportion to its length."""
    ks = np.searchsorted(cum, rng.random(n) * cum[-1], side="right")
    res = []
    for k in ks:
        p = pieces[k]
        s = rng.integers(0, p.shape[1] - T)
        res.append(p[:, s : s + T + 1])
    return np.stack(res)


def lm_loss(model, ids: torch.Tensor) -> tuple[torch.Tensor, dict]:
    """ids (B, 6, T+1) -> the release's training loss (as_auto_regressive: inputs [:-1], targets [1:])."""
    from fairseq2.nn import BatchLayout

    B, S, T1 = ids.shape
    x = ids[:, :, :-1].reshape(B * S, T1 - 1)
    y = ids[:, :, 1:].reshape(B * S, T1 - 1)
    layout = BatchLayout(shape=x.shape, seq_lens=[T1 - 1] * (B * S), packed=False, device=x.device)
    total = model(x, layout, targets=y, reduction="mean")
    return total, {"stream_specific_loss": float(model.last_stream_specific_loss), "gen_loss": float(model.last_gen_loss)}


@torch.no_grad()
def cmd_score(a) -> None:
    """The training loss (eval mode, same fixed held-out windows as train's validation) of released/fine-tuned LMs."""
    device = torch.device("cuda")
    va_files = [f for f in sorted(Path(a.units).glob("*.npz")) if held_out(f.stem)]
    val = load_pieces(va_files, a.ctx)
    vcum = np.cumsum([p.shape[1] - a.ctx for p in val]).astype(np.float64)
    vbatch = torch.from_numpy(windows(val, vcum, a.ctx, a.val_windows, np.random.default_rng(1234)).astype(np.int64))
    res = {}
    for lm in a.lms.split(","):
        loaded = load_lm(lm, device)
        loaded.model.eval()
        torch.manual_seed(999)
        tot, tss, tg, n = 0.0, 0.0, 0.0, 0
        for i in range(0, len(vbatch), a.micro):
            l, part = lm_loss(loaded.model, vbatch[i : i + a.micro].to(device))
            tot, tss, tg, n = tot + l.item(), tss + part["stream_specific_loss"], tg + part["gen_loss"], n + 1
        res[lm] = {"loss": tot / n, "stream_specific_loss": tss / n, "gen_loss": tg / n}
        print(json.dumps({lm: res[lm]}), flush=True)
        del loaded
        torch.cuda.empty_cache()
    Path(a.out).write_text(json.dumps({"val_files": [f.stem for f in va_files], "ctx": a.ctx,
                                       "windows": a.val_windows, "scores": res}, indent=2))


def cmd_train(a) -> None:
    import airon

    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    device = torch.device("cuda")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(Path(a.units).glob("*.npz"))
    tr_files = [f for f in files if not held_out(f.stem)]
    va_files = [f for f in files if held_out(f.stem)]
    assert va_files, "no held-out shard units for validation"
    T = a.ctx
    train, val = load_pieces(tr_files, T), load_pieces(va_files, T)
    cum = np.cumsum([p.shape[1] - T for p in train]).astype(np.float64)
    vcum = np.cumsum([p.shape[1] - T for p in val]).astype(np.float64)
    vbatch = torch.from_numpy(windows(val, vcum, T, a.val_windows, np.random.default_rng(1234)).astype(np.int64))

    from gslm2.lm.inference import load_opt_model

    init = Path(a.init)
    loaded = load_opt_model(init / "model.pt", device, model_config=init / "config.yaml")
    model = loaded.model.float().train()
    n_params = sum(p.numel() for p in model.parameters())
    decay = [q for n, q in model.named_parameters() if q.dim() >= 2]
    other = [q for n, q in model.named_parameters() if q.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1}, {"params": other, "weight_decay": 0.0}],
                            lr=a.lr, betas=(0.9, 0.95), eps=1e-8, fused=True)
    meta = {"run_id": str(uuid.uuid4()), "params": n_params, "init": str(init), "args": vars(a),
            "train_files": len(tr_files), "val_files": len(va_files),
            "train_positions": int(sum(p.shape[1] for p in train)), "val_positions": int(sum(p.shape[1] for p in val)),
            "train_pieces": len(train), "val_files_names": [f.stem for f in va_files], "git": os.environ.get("DY32_COMMIT", ""),
            "licence": "GSLM2 code and checkpoints CC BY-NC 4.0 (research use only)"}
    (out / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps({k: v for k, v in meta.items() if k != "val_files_names"}), flush=True)
    run = airon.init(project="dauys", name=a.name, config=meta, experiment=a.experiment,
                     tags=[a.name.split("-")[0], f"seed-{a.seed}"], total_steps=a.steps, run_id=meta["run_id"],
                     mode="disabled" if a.tracking == "off" else None)

    def lr_at(step: int) -> float:
        if step < a.warmup:
            return a.lr * (step + 1) / a.warmup
        return a.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (step - a.warmup) / max(a.steps - a.warmup, 1))))

    @torch.no_grad()
    def validate() -> dict:
        model.eval()
        st = torch.random.get_rng_state()
        torch.manual_seed(999)  # the stream whose loss is taken is drawn at random inside the model
        tot, tss, tg, n = 0.0, 0.0, 0.0, 0
        for i in range(0, len(vbatch), a.micro):
            ids = vbatch[i : i + a.micro].to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                l, part = lm_loss(model, ids)
            tot += float(l)
            tss += part["stream_specific_loss"]
            tg += part["gen_loss"]
            n += 1
        torch.random.set_rng_state(st)
        model.train()
        return {"loss": tot / n, "stream_specific_loss": tss / n, "gen_loss": tg / n}

    def save(name: str, step: int, res: dict) -> None:
        d = out / name
        d.mkdir(exist_ok=True)
        torch.save({k: v.to(torch.bfloat16) if v.is_floating_point() else v for k, v in model.state_dict().items()},
                   d / "model.pt.tmp")
        os.replace(d / "model.pt.tmp", d / "model.pt")
        shutil.copy(init / "config.yaml", d / "config.yaml")
        (d / "step.json").write_text(json.dumps({"step": step, "val": res}))

    res0 = validate()
    run.log({f"val/{k}": v for k, v in res0.items()}, step=0)
    print(json.dumps({"step": 0, "val": res0}), flush=True)
    best, t0 = res0["loss"], time.time()
    step, t_rate = 0, None
    while step < a.steps:
        if a.until and step == 25:
            t_rate = time.time()
        if a.until and step == 125:  # fit the cosine schedule to the time left before the deadline (epoch seconds)
            rate = 100 / (time.time() - t_rate)
            a.steps = 125 + int(rate * (a.until - time.time() - 300) * 0.97)
            print(json.dumps({"schedule_steps": a.steps, "steps_per_s": round(rate, 3)}), flush=True)
            run.log({"train/schedule_steps": a.steps}, step=step)
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        ids = torch.from_numpy(windows(train, cum, T, a.batch, rng).astype(np.int64)).to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        chunks = ids.split(a.micro)
        logs: dict = {}
        for c in chunks:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                total, part = lm_loss(model, c)
            (total / len(chunks)).backward()
            part["loss"] = total.item()
            for k, v in part.items():
                logs[k] = logs.get(k, 0.0) + v / len(chunks)
        logs["grad_norm"] = float(torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip))
        opt.step()
        if (step + 1) % 25 == 0:
            logs["lr"] = opt.param_groups[0]["lr"]
            logs["steps_per_s"] = 25 / (time.time() - t0)
            t0 = time.time()
            run.log({f"train/{k}": v for k, v in logs.items()}, step=step + 1)
            print(json.dumps({"step": step + 1, **{k: round(v, 4) for k, v in logs.items()}}), flush=True)
        if (step + 1) % a.val_every == 0 or step + 1 == a.steps or Path(a.stop_file).exists():
            res = validate()
            run.log({f"val/{k}": v for k, v in res.items()}, step=step + 1)
            print(json.dumps({"step": step + 1, "val": res}), flush=True)
            save("last", step + 1, res)
            if res["loss"] < best:
                best = res["loss"]
                save("best", step + 1, res)
            if Path(a.stop_file).exists():
                print("STOP FILE seen", flush=True)
                break
        step += 1
    (out / "DONE").write_text(f"{best}\n")
    run.finish()
    print("TRAIN DONE", best, flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    s = p.add_subparsers(dest="cmd", required=True)
    enc = "/opt/dlami/nvme/ckpt/spidr-2.0/final.pt"
    x = s.add_parser("wavs")
    x.add_argument("--in", dest="inp", required=True)
    x.add_argument("--glob", default="*.wav")
    x.add_argument("--out", required=True)
    x.add_argument("--encoder", default=enc)
    x.add_argument("--batch", type=int, default=8)
    x = s.add_parser("gen")
    x.add_argument("--lm", required=True)
    x.add_argument("--prompts", required=True, help="dir with all.jsonl of full-rate prompt units (wavs fullrate/)")
    x.add_argument("--out", required=True)
    x.add_argument("--temperature", type=float, default=1.58)  # release test config for 350M GSLM2
    x.add_argument("--top-p", type=float, default=0.9)
    x.add_argument("--anneal", type=float, default=0.5)
    x.add_argument("--seed", type=int, default=0)
    x.add_argument("--cont-s", type=float, default=10.0)
    x.add_argument("--group", type=int, default=10)
    x.add_argument("--max-ctx", type=int, default=0)
    x.add_argument("--limit", type=int, default=0)
    x = s.add_parser("shards")
    x.add_argument("--part", default="0/1")
    x.add_argument("--shards", default="")
    x.add_argument("--held", action="store_true", help="process held-out shards (for validation) instead")
    x.add_argument("--work", required=True)
    x.add_argument("--dst", required=True)
    x.add_argument("--encoder", default=enc)
    x.add_argument("--min-s", type=float, default=2.0)
    x.add_argument("--seg-s", type=float, default=60.0)
    x.add_argument("--budget-s", type=float, default=480.0, help="audio seconds per encoder batch")
    x.add_argument("--max-shards", type=int, default=0)
    x.add_argument("--keep", action="store_true", help="keep local npz after upload")
    x.add_argument("--stop-file", default="")
    x = s.add_parser("score")
    x.add_argument("--units", required=True)
    x.add_argument("--lms", required=True, help="comma list of LM dirs (model.pt + config.yaml)")
    x.add_argument("--out", required=True)
    x.add_argument("--ctx", type=int, default=1024)
    x.add_argument("--val-windows", type=int, default=128)
    x.add_argument("--micro", type=int, default=8)
    x = s.add_parser("train")
    x.add_argument("--units", required=True)
    x.add_argument("--init", required=True)
    x.add_argument("--out", required=True)
    x.add_argument("--steps", type=int, default=6000)
    x.add_argument("--ctx", type=int, default=1024)
    x.add_argument("--batch", type=int, default=32)
    x.add_argument("--micro", type=int, default=16)
    x.add_argument("--lr", type=float, default=5e-4)
    x.add_argument("--warmup", type=int, default=200)
    x.add_argument("--clip", type=float, default=0.5)
    x.add_argument("--val-every", type=int, default=250)
    x.add_argument("--val-windows", type=int, default=128)
    x.add_argument("--seed", type=int, default=0)
    x.add_argument("--stop-file", default="/opt/dlami/nvme/STOP_TRAIN")
    x.add_argument("--until", type=float, default=0, help="deadline (epoch s): steps are refit at step 125 to end then")
    x.add_argument("--tracking", default="on", choices=["on", "off"])
    x.add_argument("--experiment", default=EXPERIMENT)
    x.add_argument("--name", default="dy32-e")
    a = p.parse_args()
    {"wavs": cmd_wavs, "gen": cmd_gen, "shards": cmd_shards, "score": cmd_score, "train": cmd_train}[a.cmd](a)


if __name__ == "__main__":
    main()
