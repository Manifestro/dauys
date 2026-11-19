"""DY-32 F: decoder to audio for the GSLM2 reproduction (SpidR 2.0 six-stream units -> unit HiFi-GAN, 16 kHz).

Runs in the gslm2 "native" environment (PyTorch 2.13, torchcodec + FFmpeg 8). Uses the gslm2 package for unit
extraction, the duration predictor and the vocoder; this file only wires them to S3 and measures.

  extract  --wavs a.wav b.wav ... --out units.jsonl      full-rate six-stream codes (gslm2 extractor, one pass)
  decode   --manifest m.jsonl --out dir [--duration ckpt] full-rate rows -> vocoder; deduplicated rows -> durations first
  wer      --wavs ... --refs refs.json                    Whisper word error rate (measuring only)
  round                                                   one decoding round of everything in S3 with the latest vocoder
  track                                                   follow the trainers' metrics files and log them to Airon

Unit rows (JSONL): {"id": "XX", "units": "<s0 frames>|<s1 frames>|...|<s5 frames>"} or "units": [[...] x 6];
"file_name"/"audio_repr" (gslm2 extractor output) are accepted too. Codes are raw 0..255 unless "unit_offset" says so.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

W = Path(os.environ.get("DY32F_WORK", "/opt/dlami/nvme/w"))
SPIDR = W / "ckpt/spidr/final.pt"
VOC_DIR = W / "runs/vocoder"
DUR_DIR = W / "runs/duration"
BUCKET = "s3://manifestro-dataset-876108973178-us-east-1-an"
S3 = f"{BUCKET}/outputs/dauys/dy32"
SR = 16000
EXPERIMENT = "DY-32: канал, через который слова выживают, без текста — три идеи и повтор GSLM параллельными ветками"


def sh(cmd: str, check: bool = True) -> str:
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if check and r.returncode:
        raise RuntimeError(f"{cmd}\n{r.stderr[-2000:]}")
    return r.stdout


# ---------------------------------------------------------------- units
def extract(wavs: list[Path], out: Path) -> Path:
    """Full-rate six-stream codes (layers 5-10) of 16 kHz mono WAVs via the gslm2 extractor; one JSONL row per WAV."""
    import soundfile as sf
    import yaml

    tmp = Path(tempfile.mkdtemp(prefix="dy32f_ext_", dir=W))
    stem = out.stem
    csv = tmp / f"{stem}.csv"
    with csv.open("w") as f:
        f.write("fileid,path,num_frames\n")
        for w in wavs:
            f.write(f"{w.stem},{w.resolve()},{sf.info(str(w)).frames}\n")
    cfg = {
        "checkpoint": str(SPIDR), "model_type": "spidr", "layers": [5, 6, 7, 8, 9, 10], "clustering": "codes",
        "vocab_size": [256] * 6, "deduplicate": False, "cross_stream_deduplicate": False, "chunk_size": 2048,
        "overwrite_folder": True, "targets": {"unit_vocoder": {"manifests": [str(csv)]}},
        "output_folder": str(tmp / "units"), "chunk": True, "batch_size": 8, "encoder_dtype": "float32",
        "distributed_port": 58600 + os.getpid() % 300,
    }
    (tmp / "cfg.yaml").write_text(yaml.safe_dump(cfg))
    sh(f"gslm2-extract-units --config {tmp / 'cfg.yaml'}")
    src = tmp / "units/5/unit_vocoder" / f"{stem}.jsonl"
    rows = [json.loads(l) for l in src.read_text().splitlines() if l.strip()]
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for r in sorted(rows, key=lambda r: r["file_name"]):
            f.write(json.dumps({"id": Path(r["file_name"]).stem, "units": r["audio_repr"], "full_rate": True}) + "\n")
    return out


def parse_rows(path: Path) -> list[dict]:
    """Read unit rows in any of the accepted layouts -> [{"id", "streams": [[int]*T]*6, "row": raw}]."""
    out = []
    for i, line in enumerate(path.read_text().splitlines()):
        if not line.strip():
            continue
        r = json.loads(line)
        u = r.get("units", r.get("audio_repr", r.get("codes")))
        if isinstance(u, str):
            streams = [[int(x) for x in s.split()] for s in u.split("|")]
        elif u and isinstance(u[0], list):
            streams = [[int(x) for x in s] for s in u]
            if len(streams) != 6 and all(len(s) == 6 for s in streams):  # frames x streams
                streams = [list(s) for s in zip(*streams)]
        else:
            raise ValueError(f"{path}:{i + 1}: cannot read units")
        if len(streams) != 6 or len({len(s) for s in streams}) != 1:
            raise ValueError(f"{path}:{i + 1}: need six aligned streams, got {[len(s) for s in streams]}")
        off = int(r.get("unit_offset", 0))
        if off:
            keep = [t for t in range(len(streams[0])) if all(s[t] >= off for s in streams)]
            streams = [[s[t] - off for t in keep] for s in streams]
        if any(x < 0 or x > 255 for s in streams for x in s):
            raise ValueError(f"{path}:{i + 1}: codes outside 0..255")
        rid = str(r.get("id", r.get("audio", r.get("file_name", i))))
        rid = Path(rid).stem if "/" in rid or rid.endswith(".wav") else rid
        out.append({"id": rid, "streams": streams, "row": r})
    return out


def cross_dedup(streams: list[list[int]]) -> tuple[list[list[int]], list[int]]:
    """Drop a frame only when all six streams repeat the previous frame (GSLM2 cross-stream dedup)."""
    keep, durs = [], []
    T = len(streams[0])
    for t in range(T):
        if t and all(s[t] == s[t - 1] for s in streams):
            durs[-1] += 1
        else:
            keep.append(t)
            durs.append(1)
    return [[s[t] for t in keep] for s in streams], durs


def is_dedup(streams: list[list[int]]) -> bool:
    return all(any(s[t] != s[t - 1] for s in streams) for t in range(1, len(streams[0])))


# ---------------------------------------------------------------- checkpoints
def latest_vocoder() -> Path:
    ck = sorted(VOC_DIR.glob("vocoder-*"), key=lambda p: int(p.name.split("-")[1]))
    ck = [c for c in ck if (c / "model.safetensors").exists()]
    if not ck:
        raise FileNotFoundError("no vocoder checkpoint yet")
    return ck[-1]


def best_duration() -> Path | None:
    p = DUR_DIR / "best.pt"
    return p if p.exists() else None


# ---------------------------------------------------------------- decoding
class Decoder:
    def __init__(self, vocoder: Path, duration: Path | None):
        import torch
        from gslm2.vocoder.model import UnitVocoder

        self.torch = torch
        self.dev = torch.device("cuda")
        self.vocoder_path, self.duration_path = vocoder, duration
        self.voc = UnitVocoder.from_pretrained(vocoder).eval().to(self.dev)
        self.dur = self.dur_cfg = None
        if duration is not None:
            from gslm2.duration.predict import load_duration_predictor

            self.dur, self.dur_cfg = load_duration_predictor(duration, self.dev)

    def expand(self, streams: list[list[int]]) -> list[list[int]]:
        from gslm2.duration.predict import dedup_to_full

        if self.dur is None:
            raise RuntimeError("deduplicated units need a duration predictor")
        with self.torch.inference_mode():
            return dedup_to_full(self.dur, streams, log_duration=bool(self.dur_cfg.get("log_duration", True)))

    def vocode(self, streams: list[list[int]]):
        with self.torch.inference_mode():
            u = self.torch.tensor(streams, dtype=self.torch.long, device=self.dev).unsqueeze(0)
            return self.voc.generate(u).squeeze(0).float().cpu().numpy()

    def info(self) -> dict:
        return {
            "vocoder": "gslm2 unit HiFi-GAN (13.4M params), six 64-dim code embeddings -> LayerNorm+Linear -> HiFi-GAN V1, "
                       "trained here on full-rate six-stream SpidR 2.0 (L5-L10) codes of LJ Speech train (12,600 utt.), 16 kHz",
            "vocoder_checkpoint": self.vocoder_path.name,
            "vocoder_step": int(self.vocoder_path.name.split("-")[1]),
            "duration_predictor": (f"gslm2 conformer duration predictor (classes 1-50 frames), trained here on full-rate "
                                   f"codes of Libri-Light small; {self.duration_path.name} step {self.dur_step()}")
            if self.duration_path else None,
        }

    def dur_step(self):
        try:
            return int(self.torch.load(self.duration_path, map_location="cpu", weights_only=True).get("step", -1))
        except Exception:
            return None


def _streams(u) -> list[list[int]]:
    return [[int(x) for x in s.split()] for s in u.split("|")] if isinstance(u, str) else [list(map(int, s)) for s in u]


def decode_gen(dec: Decoder, manifest: Path, out: Path, prompt_full: dict[str, list[list[int]]],
               max_frames: int = 500, voc_ctx: int = 100) -> list[dict]:
    """Release recipe for LM continuations (gslm2.synthesis_continuations): durations are predicted on the
    deduplicated context + continuation, only the continuation is expanded and capped at 10 s (500 frames); the vocoder
    sees the last 2 s of the full-rate prompt before it and that part is cut from the output (continuation only)."""
    import numpy as np
    import soundfile as sf
    from gslm2.duration.predict import expand_streams, predict_durations

    out.mkdir(parents=True, exist_ok=True)
    log = []
    for line in manifest.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        rid = Path(str(r.get("audio", r.get("fileid")))).stem
        cont = _streams(r["units"])
        ctx = _streams(r["context_units"]) if r.get("context_units") else [[] for _ in range(6)]
        if int(r.get("unit_offset", 0)) != 0:
            raise ValueError(f"{manifest}: unit_offset != 0 not handled for continuations")
        full_in = [c + s for c, s in zip(ctx, cont)]
        with dec.torch.inference_mode():
            d = predict_durations(dec.dur, full_in, log_duration=bool(dec.dur_cfg.get("log_duration", True)))
        n = len(ctx[0])
        exp = [s[:max_frames] for s in expand_streams(cont, d[n:])]
        pf = prompt_full.get(rid)
        left = [s[-voc_ctx:] for s in pf] if pf else [[] for _ in range(6)]
        k = len(left[0])
        a = dec.vocode([l + e for l, e in zip(left, exp)])[k * 320:]
        a = 0.95 * a / max(float(np.abs(a).max()), 1e-6)
        sf.write(out / f"{rid}.wav", a, SR)
        log.append({"id": rid, "context_positions": n, "continuation_positions": len(cont[0]),
                    "expanded_frames": len(exp[0]), "vocoder_left_context_frames": k, "seconds": round(len(a) / SR, 2)})
    return log


def decode_manifest(dec: Decoder, manifest: Path, out: Path, mode: str = "auto") -> list[dict]:
    """mode: full (rows are full-rate), dedup (cross-dedup full-rate rows first, then predict durations), auto."""
    import soundfile as sf

    out.mkdir(parents=True, exist_ok=True)
    log = []
    for r in parse_rows(manifest):
        s = r["streams"]
        rate = "full" if mode == "full" else "dedup" if mode == "dedup" else ("dedup" if is_dedup(s) else "full")
        n_in = len(s[0])
        if rate == "dedup":
            s = cross_dedup(s)[0]
            s = dec.expand(s)
        a = dec.vocode(s)
        sf.write(out / f"{r['id']}.wav", a, SR)
        log.append({"id": r["id"], "input_frames": n_in, "rate": rate, "output_frames": len(s[0]),
                    "seconds": round(len(a) / SR, 2)})
    return log


# ---------------------------------------------------------------- measuring (text only here, on finished audio)
_ASR = None


def asr_model():
    global _ASR
    if _ASR is None:
        import whisper

        _ASR = whisper.load_model("large-v3", device="cuda")
    return _ASR


def transcribe(paths: list[Path]) -> dict[str, str]:
    m = asr_model()
    return {p.stem: m.transcribe(str(p), language="en", fp16=True, temperature=0.0,
                                      condition_on_previous_text=False)["text"].strip() for p in paths}


def wer(hyps: dict[str, str], refs: dict[str, str]) -> float:
    import jiwer
    from whisper.normalizers import EnglishTextNormalizer

    n = EnglishTextNormalizer()
    keys = [k for k in refs if k in hyps and n(refs[k]).strip()]
    return jiwer.wer([n(refs[k]) for k in keys], [n(hyps[k]) or "<empty>" for k in keys])


# ---------------------------------------------------------------- S3 round
def s3_ls(prefix: str) -> list[str]:
    out = sh(f"aws s3 ls --recursive {prefix}", check=False)
    return [l.split(maxsplit=3)[3] for l in out.splitlines() if len(l.split(maxsplit=3)) == 4]


def upload_dir(local: Path, dest: str) -> None:
    sh(f"aws s3 cp --recursive {local} {dest} --only-show-errors")


def round_(args) -> None:
    """Decode everything available with the latest vocoder (and best duration predictor) and upload."""
    voc = Path(args.vocoder) if args.vocoder else latest_vocoder()
    dur = Path(args.duration) if args.duration else best_duration()
    step = int(voc.name.split("-")[1])
    R = W / f"rounds/step{step}"
    R.mkdir(parents=True, exist_ok=True)
    dec = Decoder(voc, dur)
    base = dec.info()
    report = {"step": step, "time_utc": time.strftime("%H:%M", time.gmtime()), **base, "outputs": {}}
    dests = []

    # 1) own channel: real_XX.wav -> our extraction -> full rate -> vocoder (and -> dedup -> durations -> vocoder)
    own = W / "own/real_units.jsonl"
    if own.exists():
        for name, mode, need in (("f", "full", False), ("f-dedup", "dedup", True)):
            if need and dur is None:
                continue
            d = R / "resynth" / name
            log = decode_manifest(dec, own, d, mode)
            json.dump({**base, "input": "real_XX.wav of the common pack -> SpidR 2.0 L5-L10 codes (extracted by F)"
                       + (", cross-stream deduplicated, durations predicted" if mode == "dedup" else ", full rate"),
                       "items": log}, open(d / "info.json", "w"), indent=1)
            dests.append((d, f"resynth/{name}"))

    # 2) track E unit sequences
    tmp = W / "e_in"
    tmp.mkdir(exist_ok=True)
    keys = s3_ls(f"{S3}/e/")
    for k in keys:
        rel = k.split("outputs/dauys/dy32/e/", 1)[1]
        if rel.startswith(("resynth_units/", "gen_units/", "prompt_units/fullrate/")) and not rel.endswith("all.jsonl"):
            dst = tmp / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            sh(f"aws s3 cp {BUCKET}/{k} {dst} --only-show-errors")

    def cat(files: list[Path], name: str) -> Path:
        m = tmp / f"_{name}.jsonl"
        m.write_text("".join(f.read_text().strip() + "\n" for f in files))
        return m

    for kind, name in (("fullrate", "e-fullrate"), ("dedup", "e-dedup")):
        files = sorted((tmp / "resynth_units" / kind).glob("[0-9]*.jsonl"))
        if not files or (kind == "dedup" and dur is None):
            continue
        d = R / "resynth" / name
        log = decode_manifest(dec, cat(files, name), d, "full" if kind == "fullrate" else "dedup")
        json.dump({**base, "input": f"track E units resynth_units/{kind}/XX.jsonl of real_XX.wav"
                   + ("" if kind == "fullrate" else "; durations predicted (none kept)"), "items": log},
                  open(d / "info.json", "w"), indent=1)
        dests.append((d, f"resynth/{name}"))
    prompt_full = {}
    for f in sorted((tmp / "prompt_units/fullrate").glob("[0-9]*.jsonl")):
        r = json.loads(f.read_text().splitlines()[0])
        prompt_full[Path(str(r.get("audio", f.stem))).stem] = _streams(r.get("units", r.get("audio_repr")))
    gu = tmp / "gen_units"
    for vdir in sorted(p for p in gu.iterdir() if p.is_dir()) if gu.exists() else []:
        files = sorted(vdir.glob("[0-9]*.jsonl"))
        if not files or dur is None or (args.only and vdir.name not in args.only):
            continue
        d = R / "gen" / f"e-{vdir.name}"
        logs = decode_gen(dec, cat(files, f"gen_{vdir.name}"), d, prompt_full)
        e_info = vdir / "info.json"
        if not e_info.exists():
            sh(f"aws s3 cp {S3}/e/gen_units/{vdir.name}/info.json {e_info} --only-show-errors", check=False)
        json.dump({**base, "lm": json.loads(e_info.read_text()) if e_info.exists() else f"track E variant {vdir.name}",
                   "decoding": "release recipe: durations predicted on dedup context+continuation, continuation expanded "
                               "and capped at 500 frames (10 s); vocoder sees the last 2 s of the full-rate prompt, cut "
                               "from the output (continuation only)",
                   "rough": "durations predicted per six-tuple (LM output carries none); single-speaker LJ Speech voice; "
                            "prosody not preserved by design", "items": logs}, open(d / "info.json", "w"), indent=1)
        dests.append((d, f"gen/e-{vdir.name}"))

    json.dump(report, open(R / "round.json", "w"), indent=1)
    for d, rel in dests:  # audio first, measuring after
        upload_dir(d, f"{S3}/{rel}/")
        upload_dir(d, f"{S3}/f/step{step}/{rel}/")
    sh(f"aws s3 cp {R / 'round.json'} {S3}/f/step{step}/round.json --only-show-errors")
    sh(f"aws s3 cp {R / 'round.json'} {S3}/f/latest_round.json --only-show-errors")
    print("UPLOADED", step, [r for _, r in dests], flush=True)

    # 3) measure: Whisper WER on LJ test (vs LJ text) and on resynth/f (vs Whisper on the original real_XX.wav)
    if not args.no_wer:
        mres = {}
        lj = W / "own/lj_test_units.jsonl"
        if lj.exists():
            d = R / "lj_test"
            decode_manifest(dec, lj, d, "full")
            refs = json.load(open(W / "own/lj_test_refs.json"))
            hyp = transcribe(sorted(d.glob("*.wav")))
            mres["lj_test_wer"] = wer(hyp, refs)
        rr = W / "own/real_refs.json"
        if rr.exists():
            refs = json.load(open(rr))
            for name in ("f", "f-dedup"):  # e-* use the same units (checked bit-identical)
                d = R / "resynth" / name
                if d.exists():
                    mres[f"real_{name}_wer"] = wer(transcribe(sorted(d.glob("*.wav"))), refs)
        report["wer"] = mres
        print("WER", step, json.dumps(mres), flush=True)
        with open(W / "wer_log.jsonl", "a") as f:
            f.write(json.dumps({"step": step, **mres}) + "\n")

    if not args.no_wer:
        json.dump(report, open(R / "round.json", "w"), indent=1)
        sh(f"aws s3 cp {R / 'round.json'} {S3}/f/step{step}/round.json --only-show-errors")
        sh(f"aws s3 cp {R / 'round.json'} {S3}/f/latest_round.json --only-show-errors")
    print("ROUND DONE", step, [r for _, r in dests], flush=True)


# ---------------------------------------------------------------- Airon tracking of the gslm2 trainers
def track(args) -> None:
    import airon

    os.environ["AIRON_URL"] = os.environ.get("AIRON_URL", "").removesuffix("/v1")
    runs = {}
    ids_file, seen_file = W / "airon_runs.json", W / "airon_seen.json"
    seen = json.loads(seen_file.read_text()) if seen_file.exists() else {}
    files = {"vocoder": VOC_DIR / "metrics-0.jsonl", "duration": None, "wer": W / "wer_log.jsonl"}
    while True:
        for name in ("vocoder", "duration", "wer"):
            f = files[name]
            if name == "duration":
                cands = sorted(DUR_DIR.glob("*.jsonl")) if DUR_DIR.exists() else []
                f = cands[0] if cands else None
            if f is None or not f.exists():
                continue
            if name not in runs:
                key = "vocoder" if name == "wer" else name
                if key not in runs:
                    cfg = {"commit": os.environ.get("DY21_COMMIT", ""), "part": key,
                           "params": {"vocoder": 13406594}.get(key)}
                    ids = json.loads(ids_file.read_text()) if ids_file.exists() else {}
                    kw = {"run_id": ids[key], "resume": "allow"} if key in ids else {}
                    runs[key] = airon.init(project="dauys", name=f"dy32-f-{key}", config=cfg, experiment=EXPERIMENT,
                                           tags=["dy32", "dy32-f", key], **kw)
                    ids[key] = runs[key].id
                    ids_file.write_text(json.dumps(ids))
                runs[name] = runs[key]
            lines = f.read_text().splitlines()
            for l in lines[seen.get(name, 0):]:
                try:
                    r = json.loads(l)
                except json.JSONDecodeError:
                    break
                step = int(r.pop("step", 0))
                vals = {(k if name != "wer" else f"wer/{k}"): v for k, v in r.items() if isinstance(v, int | float)}
                if name == "duration":
                    vals = {f"duration/{r.get('type', 'train')}/{k}": v for k, v in vals.items()}
                if vals:
                    runs[name].log(vals, step=step)
                seen[name] = seen.get(name, 0) + 1
        seen_file.write_text(json.dumps(seen))
        if args.once:
            break
        time.sleep(60)


def main() -> None:
    p = argparse.ArgumentParser()
    sp = p.add_subparsers(dest="cmd", required=True)
    a = sp.add_parser("extract"); a.add_argument("--wavs", nargs="+", type=Path); a.add_argument("--out", type=Path)
    a = sp.add_parser("decode"); a.add_argument("--manifest", type=Path); a.add_argument("--out", type=Path)
    a.add_argument("--vocoder"); a.add_argument("--duration"); a.add_argument("--mode", default="auto")
    a = sp.add_parser("wer"); a.add_argument("--wavs", nargs="+", type=Path); a.add_argument("--refs", type=Path)
    a = sp.add_parser("transcribe"); a.add_argument("--wavs", nargs="+", type=Path); a.add_argument("--out", type=Path)
    a = sp.add_parser("round"); a.add_argument("--vocoder"); a.add_argument("--duration"); a.add_argument("--no-wer", action="store_true")
    a.add_argument("--only", nargs="*", help="only these gen variants")
    a = sp.add_parser("track"); a.add_argument("--once", action="store_true")
    args = p.parse_args()
    if args.cmd == "extract":
        extract(args.wavs, args.out)
    elif args.cmd == "decode":
        voc = Path(args.vocoder) if args.vocoder else latest_vocoder()
        dur = Path(args.duration) if args.duration else best_duration()
        print(json.dumps(decode_manifest(Decoder(voc, dur), args.manifest, args.out, args.mode)))
    elif args.cmd == "wer":
        print(wer(transcribe(args.wavs), json.load(open(args.refs))))
    elif args.cmd == "transcribe":
        json.dump(transcribe(args.wavs), open(args.out, "w"), indent=1)
    elif args.cmd == "round":
        round_(args)
    elif args.cmd == "track":
        track(args)


if __name__ == "__main__":
    sys.exit(main())
