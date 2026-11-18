"""DY-31 data: letters (with the word delimiter) from a character CTC recogniser over every DuplexChat en-16k shard.

The recogniser is facebook/wav2vec2-large-960h-lv60-self, decoded exactly as phon_units.Phonemes decodes phonemes
(fp16, per-window normalisation with attention mask, greedy argmax per 20 ms frame, repeats collapsed, blank
dropped, 20 s windows, batches of 32): ``Letters`` subclasses Phonemes and only swaps the model. Every non-blank id
is a token, the word delimiter included. The pass (pieces >= 5 s, shard batching, prefetch, background upload,
skip-what-is-in-S3, ``--part k/n``) is dy30_units.py's, imported unchanged; output ``<shard>.npz`` (tok, index) to
S3 outputs/dauys/char-units/. ``Text`` turns ids into text: vocab symbol per id, delimiter -> space, special tokens
dropped, lowercase, spaces collapsed. Vocabulary, blank and delimiter are read from the model files at run time.

    python scripts/dy31_units.py run --part 0/8 --work /opt/dlami/nvme/w0
    python scripts/dy31_units.py known --shard la-00030 --work /opt/dlami/nvme/known --out known.json
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

sys.path.insert(0, str(Path(__file__).parent))
from dy21_train import held_out
from dy30_units import BATCH, SRC, fetch, listing, save, sh, shard_tokens
from phon_units import WIN, Phonemes

CHARS = "facebook/wav2vec2-large-960h-lv60-self"
DST = "s3://manifestro-dataset-876108973178-us-east-1-an/outputs/dauys/char-units"


class Letters(Phonemes):
    """phon_units.Phonemes with the character model; windows() / __call__ are inherited unchanged."""

    def __init__(self, device) -> None:
        from transformers import Wav2Vec2ForCTC

        self.device = device
        self.model = Wav2Vec2ForCTC.from_pretrained(CHARS).to(device).half().eval()
        self.blank = self.model.config.pad_token_id


class Text:
    """Token ids -> text, from the recogniser's own vocabulary (read at run time)."""

    def __init__(self) -> None:
        from transformers import Wav2Vec2CTCTokenizer

        tk = Wav2Vec2CTCTokenizer.from_pretrained(CHARS)
        self.delim = tk.word_delimiter_token
        special = set(tk.all_special_tokens)
        inv = {i: s for s, i in tk.get_vocab().items()}
        self.sym = {i: (" " if s == self.delim else "" if s in special else s.lower()) for i, s in inv.items()}
        self.delim_id = tk.convert_tokens_to_ids(self.delim)
        self.info = {"vocab": len(inv), "delimiter": self.delim, "delimiter_id": int(self.delim_id),
                     "special": sorted(special), "pad_token": tk.pad_token}

    def __call__(self, ids) -> str:
        return " ".join("".join(self.sym.get(int(i), "") for i in ids).split())


def cmd_run(a) -> None:
    """dy30_units.cmd_run with the letters recogniser and the char-units destination."""
    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    ph = Letters(torch.device("cuda"))
    done = set(listing(a.dst, dirs=False))
    if a.shards:
        mine = [s for s in a.shards.split(",") if s not in done]
    else:
        k, n = map(int, a.part.split("/"))
        mine = [s for s in listing(a.src, dirs=True)[k::n] if s not in done]
    if a.max_shards:
        mine = mine[: a.max_shards]
    print(json.dumps({"part": a.part, "todo": len(mine), "already_in_dst": len(done), "blank": int(ph.blank)}), flush=True)
    dl, up = ThreadPoolExecutor(1), ThreadPoolExecutor(2)
    uploads, failed, lock = [], [], threading.Lock()

    def upload(f: Path, s: str) -> None:
        try:
            sh(f"aws s3 cp {f} {a.dst}/{s}.npz --only-show-errors")
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
        toks = shard_tokens(ph, pieces)
        out = work / f"{s}.npz"
        n_tok = save(out, toks)
        del pieces, toks
        uploads.append(up.submit(upload, out, s))
        el = time.time() - t0
        h_all += secs / 3600
        print(json.dumps({"part": a.part, "shard": s, "held": held_out(s), "done": j + 1, "of": len(mine),
                          "hours": round(secs / 3600, 2), "letters": n_tok, "per_s": round(n_tok / max(secs, 1), 1),
                          "x_realtime": round(secs / el, 1), "waited_for_download_s": round(wait, 1),
                          "x_realtime_total": round(h_all * 3600 / (time.time() - t_all), 1)}), flush=True)
    for f in uploads:
        f.result()
    if failed:
        print("UPLOAD FAILED", json.dumps(failed), flush=True)
        sys.exit(1)
    print(f"CHAR UNITS DONE part {a.part}", round(h_all, 1), "h", flush=True)


def cmd_known(a) -> None:
    """The pass's batching vs the direct call (all windows of the first pieces in one Letters.windows list),
    on >= 200 windows: must be identical piece by piece. Also prints a few decoded texts."""
    ph = Letters(torch.device("cuda"))
    _, pieces, _ = fetch(a.src, a.shard, Path(a.work))
    use, n_win = [], 0
    for w in pieces:
        use.append(w)
        n_win += -(-len(w) // WIN)
        if n_win >= a.windows:
            break
    wins, owner = [], []
    for k, w in enumerate(use):
        x = w.astype(np.float32) / 32768.0
        for i in range(0, len(x), WIN):
            wins.append(x[i : i + WIN])
            owner.append(k)
    ref = ph.windows(wins, batch=BATCH)
    ref_piece = [np.concatenate([r for o, r in zip(owner, ref) if o == k]) for k in range(len(use))]
    new_piece = shard_tokens(ph, use)
    same = [bool(np.array_equal(x, y)) for x, y in zip(ref_piece, new_piece)]
    tx = Text()
    res = {"shard": a.shard, "windows": len(wins), "pieces": len(use), "letters": int(sum(len(x) for x in ref_piece)),
           "identical_pieces": round(float(np.mean(same)), 4), "pass": bool(all(same) and len(wins) >= a.windows),
           "blank": int(ph.blank), "text": tx.info, "sample": [tx(x)[:200] for x in ref_piece[:3]]}
    Path(a.out).write_text(json.dumps(res, indent=2))
    print("KNOWN TOKENS", json.dumps({k: v for k, v in res.items() if k != "sample"}), flush=True)
    print("SAMPLE", json.dumps(res["sample"]), flush=True)
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
    x = s.add_parser("known")
    x.add_argument("--shard", required=True)
    x.add_argument("--work", required=True)
    x.add_argument("--src", default=SRC)
    x.add_argument("--windows", type=int, default=200)
    x.add_argument("--out", required=True)
    a = p.parse_args()
    {"run": cmd_run, "known": cmd_known}[a.cmd](a)


if __name__ == "__main__":
    main()
