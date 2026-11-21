"""DY-36 shared tools: utterance audio (held-out shards local, training shards by S3 byte range), texts of finished audio
(letters recogniser = main row, Whisper = robustness row; text only measures audio), the judge (Qwen2.5-7B-Instruct,
both orders), Wilson intervals."""

import json
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

SR, HOP = 16000, 320
BUCKET = "manifestro-dataset-876108973178-us-east-1-an"
SRC_KEY = "outputs/duplexchat/en-16k"
HELD_DIR = Path("/data/held")
IDX_DIR = Path("/data/idx")
JUDGE = "Qwen/Qwen2.5-7B-Instruct"
ASR = "openai/whisper-large-v3-turbo"

CLOSER = (
    "Below is a reference transcript of an utterance, followed by two candidate transcripts, A and B.\n\n"
    "Reference:\n{prompt}\n\nCandidate A:\n{a}\n\nCandidate B:\n{b}\n\n"
    "Which candidate is closer in content to the reference? Answer with a single letter, A or B."
)
FITS = (
    "Below is a transcript of the last part of a conversation, followed by two candidate transcripts, A and B, of what "
    "could be said next.\n\nConversation:\n{prompt}\n\nCandidate A:\n{a}\n\nCandidate B:\n{b}\n\n"
    "Which candidate fits better as the next thing said in this conversation? Answer with a single letter, A or B."
)


def log(*a) -> None:
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def wilson(k: float, n: int) -> list[float]:
    if n == 0:
        return [0.0, 0.0]
    p, z = k / n, 1.96
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 4), round(c + h, 4)]


class Audio:
    """Audio (int16) of frames [s, e) of a piece of a shard; held-out shards from local folders, others by byte range."""

    def __init__(self, files: list[str]) -> None:
        import boto3

        self.files = files
        self.s3 = boto3.client("s3", region_name="us-east-1")
        self.idx, self.mm = {}, {}
        IDX_DIR.mkdir(parents=True, exist_ok=True)

    def index(self, shard: str) -> np.ndarray:
        if shard not in self.idx:
            if (HELD_DIR / shard / "index.npy").exists():
                self.idx[shard] = np.load(HELD_DIR / shard / "index.npy")
                self.mm[shard] = np.memmap(HELD_DIR / shard / "segs.bin", dtype=np.int16, mode="r")
            else:
                f = IDX_DIR / f"{shard}.npy"
                if not f.exists():
                    self.s3.download_file(BUCKET, f"{SRC_KEY}/{shard}/index.npy", str(f))
                self.idx[shard] = np.load(f)
        return self.idx[shard]

    def get(self, file: int, piece_orig: int, s: int, e: int) -> np.ndarray:
        shard = self.files[file]
        st = int(self.index(shard)[piece_orig][0])
        a, b = st + s * HOP, st + e * HOP
        if shard in self.mm:
            return np.asarray(self.mm[shard][a:b], np.int16)
        r = self.s3.get_object(Bucket=BUCKET, Key=f"{SRC_KEY}/{shard}/segs.bin", Range=f"bytes={2 * a}-{2 * b - 1}")
        return np.frombuffer(r["Body"].read(), np.int16)

    def many(self, jobs: list[tuple]) -> list[np.ndarray]:
        for j in jobs:
            self.index(self.files[j[0]])
        with ThreadPoolExecutor(32) as ex:
            return list(ex.map(lambda j: self.get(*j), jobs))


def utt_job(T: dict, u: int, ds: int = 0, de=None) -> tuple:
    """Audio job for frames [s + ds, e or s + de) of utterance u (clipped to the piece start)."""
    s = int(T["s"][u]) + ds
    e = int(T["e"][u]) if de is None else int(T["s"][u]) + de
    return (int(T["fi"][u]), int(T["piece_orig"][u]), max(0, s), e)


def transcribe(wavs: list[np.ndarray], whisper: bool = True) -> tuple[list[str], list[str]]:
    """Letters recogniser (greedy CTC on 20 s windows, as dy31_units.Letters) and Whisper large-v3-turbo texts."""
    import torch
    from transformers import pipeline

    from dy30_eval import cleanup
    from dy31_units import Letters, Text
    from phon_units import WIN

    dev = torch.device("cuda")
    tx, L = Text(), Letters(dev)
    wins, owner = [], []
    for i, w in enumerate(wavs):
        a = w.astype(np.float32) / 32768.0
        for k in range(0, max(len(a), 1), WIN):
            if len(a[k : k + WIN]) >= 400:
                wins.append(a[k : k + WIN])
                owner.append(i)
    order = sorted(range(len(wins)), key=lambda i: -len(wins[i]))
    ids = [None] * len(wins)
    for b in range(0, len(order), 32):
        part = order[b : b + 32]
        for i, r in zip(part, L.windows([wins[i] for i in part], batch=32)):
            ids[i] = r
    lt = [[] for _ in wavs]
    for i, o in enumerate(owner):
        lt[o].append(ids[i])
    letters = [tx(np.concatenate(x)) if x else "" for x in lt]
    del L
    cleanup()
    wt = [""] * len(wavs)
    if whisper:
        asr = pipeline("automatic-speech-recognition", model=ASR, device=dev, torch_dtype=torch.float16, chunk_length_s=30)
        idx = [i for i, w in enumerate(wavs) if len(w) >= SR // 10]
        res = asr([{"raw": wavs[i].astype(np.float32) / 32768.0, "sampling_rate": SR} for i in idx], batch_size=16,
                  generate_kwargs={"language": "en", "task": "transcribe"})
        for i, r in zip(idx, res):
            wt[i] = r["text"].strip()
        del asr
        cleanup()
    return letters, wt


class TextCache:
    """key -> {"letters", "whisper"} in a json file; fills missing keys from audio jobs."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.d = json.loads(self.path.read_text()) if self.path.exists() else {}

    def fill(self, audio: Audio, jobs: dict[str, tuple]) -> None:
        todo = [k for k in jobs if k not in self.d]
        if not todo:
            return
        log("audio", len(todo))
        wavs = audio.many([jobs[k] for k in todo])
        log("transcribe", len(todo))
        le, wh = transcribe(wavs)
        for k, a, b, w in zip(todo, le, wh, wavs):
            self.d[k] = {"letters": a, "whisper": b, "sec": round(len(w) / SR, 2)}
        self.path.write_text(json.dumps(self.d))

    def __getitem__(self, k: str) -> dict:
        return self.d[k]


class Judge:
    """Qwen2.5-7B-Instruct, P("A") renormalised over A/B; pairs() asks both orders and averages P(x preferred)."""

    def __init__(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(JUDGE)
        self.tok.padding_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(JUDGE, torch_dtype=torch.bfloat16).cuda().eval()
        ab = [self.tok.encode(x, add_special_tokens=False) for x in ("A", "B")]
        assert all(len(x) == 1 for x in ab), ab
        self.ab = [ab[0][0], ab[1][0]]

    def p_first(self, tmpl: str, prompts, first, second, batch: int = 8) -> np.ndarray:
        torch = self.torch
        texts = [self.tok.apply_chat_template([{"role": "user", "content": tmpl.format(prompt=p, a=a, b=b)}],
                                              tokenize=False, add_generation_prompt=True)
                 for p, a, b in zip(prompts, first, second)]
        order = sorted(range(len(texts)), key=lambda i: -len(texts[i]))
        out = np.zeros(len(texts))
        with torch.no_grad():
            for i in range(0, len(order), batch):
                part = order[i : i + batch]
                enc = self.tok([texts[j] for j in part], return_tensors="pt", padding=True).to("cuda")
                lg = self.model(**enc).logits[:, -1, self.ab].float()
                if not torch.isfinite(lg).all():
                    raise RuntimeError("judge logits not finite")
                out[part] = torch.softmax(lg, -1)[:, 0].cpu().numpy()
        return out

    def pairs(self, tmpl: str, prompts, x, y) -> dict:
        """Share of items where x is preferred (mean of both orders > 0.5; tie = 0.5), Wilson CI, per-item P."""
        idx = [i for i in range(len(prompts)) if prompts[i].strip() and x[i].strip() and y[i].strip()]
        P = [prompts[i] for i in idx]
        X = [x[i] for i in idx]
        Y = [y[i] for i in idx]
        pa = self.p_first(tmpl, P, X, Y)
        pb = 1.0 - self.p_first(tmpl, P, Y, X)
        p = (pa + pb) / 2
        right = np.where(p > 0.5, 1.0, np.where(p == 0.5, 0.5, 0.0))
        n = len(p)
        return {"D": round(float(right.mean()), 4) if n else float("nan"), "ci95": wilson(float(right.sum()), n), "n": n,
                "skipped_empty": len(prompts) - n, "mean_p": round(float(p.mean()), 4) if n else float("nan"),
                "p_first_order": round(float(pa.mean()), 4) if n else float("nan"),
                "p_second_order": round(float(pb.mean()), 4) if n else float("nan"),
                # the answer flips with the order = the judge picks the same position both times
                "order_flip_share": round(float(((pa > 0.5) != (pb > 0.5)).mean()), 4) if n else float("nan"),
                "consistent_x_share": round(float(((pa > 0.5) & (pb > 0.5)).mean()), 4) if n else float("nan"),
                "consistent_y_share": round(float(((pa < 0.5) & (pb < 0.5)).mean()), 4) if n else float("nan"),
                "per_item": {int(i): round(float(q), 4) for i, q in zip(idx, p)}}
