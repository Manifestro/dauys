"""DY-40: Mimi codec (kyutai/mimi, 8 codebooks x 2048, 12.5 Hz, 24 kHz) over DuplexChat en-16k, and its round trip.

All audio is 16 kHz int16 in S3; it is raised to 24 kHz with torchaudio's sinc resampler on the GPU (the same call
everywhere: units, prompts, heard-content test). Mimi is causal, so right padding never changes earlier frames.

pass:     shards -> pieces >= 1 s -> RUNS (consecutive pieces of one episode, gap <= 0.05 s, as DY-33) -> each run's
          audio joined, encoded in chunks of --chunk-s seconds -> <shard>.npz: tok int16 [steps, 8], index [runs, 2]
          (start step, steps), meta (JSON string: episode, start, end, pieces). Uploaded to S3 dy40/units/ at once;
          shards already there are skipped. Shard list: the first --n-train training shards of en-16k by name
          (held-out sha1 % 50 == 0 excluded) + the held-out la-00030, la-00071 (validation).
resynth:  the 50 DY-32 prompts' real continuations (real_XX.wav) -> Mimi 8 codes -> decoder -> 16 kHz wav in
          S3 dy40/resynth/m8/ (known answer: DY-32 measured WER 0.167 for this channel).

    python scripts/dy40_mimi.py pass --part 0/4 --out /opt/dlami/nvme/units --work /opt/dlami/nvme/w0
    python scripts/dy40_mimi.py resynth --out /data/dy40/resynth
"""

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from dy21_train import held_out

BUCKET = "s3://manifestro-dataset-876108973178-us-east-1-an/outputs"
SRC = f"{BUCKET}/duplexchat/en-16k"
S3 = f"{BUCKET}/dauys/dy40"
DST = f"{S3}/units"
MIMI = "kyutai/mimi"
MIMI_REV = "89091b3e466eb6a9d11e537bf26b144f194978f7"
SR, MSR, HOP = 16000, 24000, 1920  # Mimi frame = 1920 samples at 24 kHz = 80 ms = 1280 samples at 16 kHz
LEVELS, HELD = 8, ["la-00030", "la-00071"]


def sh(cmd: str) -> str:
    import subprocess

    return subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True).stdout


def listing(prefix: str) -> list[str]:
    import subprocess

    out = subprocess.run(f"aws s3 ls {prefix}/", shell=True, capture_output=True, text=True).stdout  # empty prefix: exit 1
    return sorted(l.split()[-1].rstrip("/").removesuffix(".npz") for l in out.splitlines() if l.strip())


class Mimi:
    def __init__(self, dev: torch.device) -> None:
        from transformers import MimiModel

        self.dev = dev
        self.m = MimiModel.from_pretrained(MIMI, revision=MIMI_REV).to(dev).eval()

    def up(self, w: torch.Tensor) -> torch.Tensor:
        import torchaudio

        return torchaudio.functional.resample(w, SR, MSR)

    @torch.no_grad()
    def encode(self, waves: list[np.ndarray], chunk_s: float = 120.0, batch: int = 16) -> list[np.ndarray]:
        """16 kHz waves (int16 or float in [-1, 1]) -> codes int16 [ceil(len24 / 1920), 8] each. Long waves are cut into
        chunks of chunk_s seconds (each chunk starts the causal encoder afresh); chunks of all waves are batched."""
        cl = int(chunk_s * MSR) // HOP * HOP
        jobs = []  # (wave id, chunk no, 24 kHz tensor)
        for i, w in enumerate(waves):
            x = torch.from_numpy(np.asarray(w, np.float32) / (32768.0 if np.asarray(w).dtype == np.int16 else 1.0))
            x = self.up(x.to(self.dev))
            for j in range(0, len(x), cl):
                jobs.append((i, j // cl, x[j : j + cl]))
        order = sorted(range(len(jobs)), key=lambda q: -len(jobs[q][2]))
        res: dict[tuple[int, int], np.ndarray] = {}
        for b in range(0, len(order), batch):
            part = [jobs[q] for q in order[b : b + batch]]
            n = max(len(c) for _, _, c in part)
            n = -(-n // HOP) * HOP
            x = torch.zeros(len(part), 1, n, device=self.dev)
            for r, (_, _, c) in enumerate(part):
                x[r, 0, : len(c)] = c
            codes = self.m.encode(x, num_quantizers=LEVELS).audio_codes  # [B, 8, frames]
            for r, (i, j, c) in enumerate(part):
                res[(i, j)] = codes[r, :, : -(-len(c) // HOP)].T.cpu().numpy().astype(np.int16)
        out = []
        for i in range(len(waves)):
            parts = [res[k] for k in sorted(k for k in res if k[0] == i)]
            out.append(np.concatenate(parts) if parts else np.zeros((0, LEVELS), np.int16))
        return out

    @torch.no_grad()
    def decode(self, codes: np.ndarray) -> np.ndarray:
        """[steps, 8] -> 24 kHz float32 wave."""
        c = torch.from_numpy(np.asarray(codes, np.int64).T[None]).to(self.dev)
        return self.m.decode(c).audio_values[0, 0].float().cpu().numpy()


def to16(w24: np.ndarray) -> np.ndarray:
    import torchaudio

    return torchaudio.functional.resample(torch.from_numpy(w24), MSR, SR).numpy()


def fetch(s: str, work: Path):
    d = work / s
    d.mkdir(parents=True, exist_ok=True)
    for fn in ("index.npy", "segs.bin", "pieces.jsonl"):
        sh(f"aws s3 cp {SRC}/{s}/{fn} {d}/{fn} --only-show-errors")
    idx = np.load(d / "index.npy")
    segs = np.fromfile(d / "segs.bin", dtype=np.int16)
    meta = [json.loads(l) for l in open(d / "pieces.jsonl")]
    sh(f"rm -rf {d}")
    keep = [(p, segs[st : st + ln], meta[p]) for p, (st, ln) in enumerate(idx) if ln >= SR]
    return s, keep, sum(len(w) for _, w, _ in keep) / SR


def runs_of(meta: list[dict], max_gap: float) -> list[list[int]]:
    """Same rule as dy33s_units.runs_of."""
    runs, cur = [], []
    for i, m in enumerate(meta):
        if cur:
            p = meta[cur[-1]]
            if m["ep"] == p["ep"] and abs(float(m["start"]) - float(p["end"])) <= max_gap:
                cur.append(i)
                continue
            runs.append(cur)
        cur = [i]
    if cur:
        runs.append(cur)
    return runs


def shard_list(n_train: int) -> list[str]:
    import re

    allp = [s for s in listing(SRC) if re.fullmatch(r"(la|lb|lbig)-\d{5}|s\d{5}", s)]
    train = [s for s in allp if not held_out(s)][:n_train]
    return train + HELD


def cmd_pass(a) -> None:
    torch.backends.cuda.matmul.allow_tf32 = True
    out, work = Path(a.out), Path(a.work)
    out.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    todo = a.shards.split(",") if a.shards else shard_list(a.n_train)
    k, n = map(int, a.part.split("/"))
    todo = todo[k::n]
    if a.reverse:  # helper machines walk a part from its end, meeting the forward worker in the middle
        todo = todo[::-1]
    have = set(listing(DST)) if not a.no_skip else set()
    todo = [s for s in todo if s not in have and not (out / f"{s}.npz").exists()]
    print(json.dumps({"part": a.part, "todo": len(todo)}), flush=True)
    if not todo:
        print("PASS DONE", a.part, 0, "h", flush=True)
        return
    mm = Mimi(torch.device("cuda"))
    dl = ThreadPoolExecutor(2)
    futs = [dl.submit(fetch, s, work) for s in todo[:2]]
    t_all, h_all = time.time(), 0.0
    def in_s3(s: str) -> bool:
        import subprocess

        return subprocess.run(f"aws s3 ls {DST}/{s}.npz", shell=True, capture_output=True).returncode == 0

    for j, s in enumerate(todo):
        t0 = time.time()
        _, keep, secs = futs[j].result()
        futs[j] = None  # drop the shard's audio held by the finished future (memory grew ~2 GB per shard)
        if j + 2 < len(todo):
            futs.append(dl.submit(fetch, todo[j + 2], work))
        if not a.no_skip and in_s3(s):  # another worker finished it meanwhile
            print(json.dumps({"shard": s, "skipped": "already in S3"}), flush=True)
            continue
        meta = [m for _, _, m in keep]
        runs = runs_of(meta, a.max_gap)
        waves = [np.concatenate([keep[i][1] for i in r]) for r in runs]
        codes = mm.encode(waves, a.chunk_s, a.batch)
        tok, index, rmeta, pos = [], [], [], 0
        for r, c in zip(runs, codes):
            tok.append(c)
            index.append((pos, len(c)))
            pos += len(c)
            rmeta.append({"ep": meta[r[0]]["ep"], "start": meta[r[0]]["start"], "end": meta[r[-1]]["end"],
                          "pieces": len(r)})
        del keep, waves
        f = out / f"{s}.npz"
        np.savez(f.with_suffix(".tmp.npz"), tok=np.concatenate(tok), index=np.asarray(index, np.int64),
                 meta=np.asarray(json.dumps(rmeta)))
        f.with_suffix(".tmp.npz").rename(f)
        if not a.no_upload:
            sh(f"aws s3 cp {f} {DST}/{s}.npz --only-show-errors")
        h_all += secs / 3600
        lens = np.array([x[1] for x in index])
        print(json.dumps({"shard": s, "held": held_out(s), "done": j + 1, "of": len(todo), "steps": pos,
                          "runs": len(runs), "hours": round(secs / 3600, 2),
                          "hours_in_runs_ge_2049": round(float(lens[lens >= 2049].sum()) / 12.5 / 3600, 2),
                          "x_realtime": round(secs / (time.time() - t0), 1),
                          "x_realtime_total": round(h_all * 3600 / (time.time() - t_all), 1),
                          "utc": time.strftime("%H:%M", time.gmtime())}), flush=True)
    print("PASS DONE", a.part, round(h_all, 1), "h", flush=True)


def read_pack(dst: Path, kind: str, src: str):
    import soundfile as sf

    dst.mkdir(parents=True, exist_ok=True)
    sh(f"aws s3 sync {BUCKET}/dauys/dy32/{src}/ {dst}/ --only-show-errors")
    names = sorted(p.stem.split("_")[1] for p in dst.glob(f"{kind}_*.wav"))
    waves = []
    for nm in names:
        w, sr = sf.read(dst / f"{kind}_{nm}.wav", dtype="float32")
        assert sr == SR, sr
        waves.append(w if w.ndim == 1 else w.mean(1))
    return names, waves


def cmd_resynth(a) -> None:
    import soundfile as sf

    mm = Mimi(torch.device("cuda"))
    names, waves = read_pack(Path(a.pack), "real", "prompts")
    out = Path(a.out) / "m8"
    out.mkdir(parents=True, exist_ok=True)
    for nm, c in zip(names, mm.encode(waves)):
        sf.write(out / f"{nm}.wav", to16(mm.decode(c)), SR)
    if not a.no_upload:
        sh(f"aws s3 cp {out}/ {S3}/resynth/m8/ --recursive --only-show-errors")
    print("RESYNTH DONE", len(names), flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    sp = p.add_subparsers(dest="cmd", required=True)
    q = sp.add_parser("pass")
    q.add_argument("--part", default="0/1")
    q.add_argument("--shards", default="")
    q.add_argument("--n-train", type=int, default=600)
    q.add_argument("--out", required=True)
    q.add_argument("--work", required=True)
    q.add_argument("--chunk-s", type=float, default=120.0)
    q.add_argument("--batch", type=int, default=8)
    q.add_argument("--max-gap", type=float, default=0.05)
    q.add_argument("--no-upload", action="store_true")
    q.add_argument("--no-skip", action="store_true")
    q.add_argument("--reverse", action="store_true")
    r = sp.add_parser("resynth")
    r.add_argument("--pack", default="/data/dy32/prompts")
    r.add_argument("--out", required=True)
    r.add_argument("--no-upload", action="store_true")
    a = p.parse_args()
    {"pass": cmd_pass, "resynth": cmd_resynth}[a.cmd](a)


if __name__ == "__main__":
    main()
