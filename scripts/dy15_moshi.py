"""DY-15: Moshi on the DY-14 task — the partner's next 2 s among 16 same-voice windows.

Moshi's own stream holds the partner (the side whose reply is chosen), its user stream
holds the "self" channel, as in DY-14. For an anchor at 10 Hz frame ``t`` the context is
the last 20 s of both tracks up to ``t``; each candidate is 2 s of the partner's audio
starting 0.4 s after the candidate's own frame, placed right after the context in Moshi's
stream, with Mimi silence in the user stream (the real self channel after ``t`` would leak
reactions to the true reply). Score: sum of log p over Moshi's 8 audio codebooks on the
candidate frames. Moshi's text monologue is not given: a first pass with no text input
predicts it frame by frame (argmax), a second pass scores the audio with that text.

Every candidate is scored with the real self channel and with the self channel of the
next test conversation (other people), as in DY-14. Correct = the true window strictly
above all 15 others.

    python scripts/dy15_moshi.py --anchors test_anchors.json --oy /data/oy16 --out /data/dy15 --n 2000
    python scripts/dy15_moshi.py ... --known 50   # known answer: the true window is contiguous
"""

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.signal import resample_poly

from dauys.dy14_eval import bootstrap

FPS = 12.5
CTX = 250  # 20 s of Mimi frames
CAND = 25  # 2 s
GAP_S = 0.4


def sample_onset(anchors: dict, n: int, seed: int = 0) -> list[dict]:
    """Up to ``n`` onset anchors, spread evenly over conversations (round robin)."""
    rng = np.random.default_rng(seed)
    by = defaultdict(list)
    for it in anchors["items"]:
        if it["onset"]:
            by[it["conv"]].append(it)
    for v in by.values():
        rng.shuffle(v)
    out, k = [], 0
    convs = sorted(by)
    while len(out) < n and any(len(by[c]) > k for c in convs):
        for c in convs:
            if len(by[c]) > k and len(out) < n:
                out.append(by[c][k])
        k += 1
    return out


@torch.no_grad()
def encode_track(mimi, wav16: np.ndarray, device) -> np.ndarray:
    x = resample_poly(wav16.astype(np.float32) / 32768.0, 3, 2).astype(np.float32)
    step = 1920 * 12 * 25 * 10  # 240 s
    codes = []
    for i in range(0, len(x), step):
        piece = x[i : i + step]
        pad = (-len(piece)) % 1920
        piece = np.pad(piece, (0, pad))
        pcm = torch.from_numpy(piece)[None, None].to(device)
        codes.append(mimi.encode(pcm)[0, :8].cpu().numpy())
    return np.concatenate(codes, 1).astype(np.int64)


def frame_of(t10: int, extra_s: float = 0.0) -> int:
    return round((t10 * 0.1 + extra_s) * FPS)


def build_rows(
    item, convs_codes, names, swap: bool, known: bool, silence: np.ndarray
) -> list[np.ndarray]:
    ci, ch = item["conv"], item["ch"]
    own = convs_codes[names[ci]][1 - ch]  # partner -> Moshi stream
    src = (
        convs_codes[names[(ci + 1) % len(names)]][ch]
        if swap
        else convs_codes[names[ci]][ch]
    )
    end = frame_of(item["t"]) if not known else frame_of(item["t"], GAP_S)
    start = max(0, end - CTX)
    idx = np.arange(start, end) % src.shape[1]
    rows = []
    for k, c in enumerate([item["t"]] + item["neg"]):
        cs = frame_of(c, GAP_S)
        cand = own[:, cs : cs + CAND]
        if cand.shape[1] < CAND:
            cand = np.pad(cand, ((0, 0), (0, CAND - cand.shape[1])), constant_values=-1)
        seq = np.full((17, end - start + CAND), -1, np.int64)
        seq[1:9, : end - start] = own[:, start:end]
        seq[1:9, end - start :] = cand
        seq[9:17, : end - start] = src[:, idx]
        seq[9:17, end - start :] = silence[:, None]
        rows.append(seq)
    return rows


@torch.no_grad()
def score(lm, rows: list[np.ndarray], device) -> np.ndarray:
    codes = torch.from_numpy(np.stack(rows)).to(device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        first = lm(codes)
        text = first.text_logits[:, 0].float()
        text = torch.nan_to_num(text, nan=-1e9).argmax(-1)
        codes[:, 0] = text
        out = lm(codes)
    logits = out.logits.float()  # [B, 8, T, card]
    target = codes[:, 1:9]
    valid = out.mask & (target >= 0)
    lp = torch.log_softmax(torch.nan_to_num(logits, nan=0.0), -1)
    lp = lp.gather(-1, target.clamp(min=0)[..., None])[..., 0]
    lp = torch.where(valid, lp, torch.zeros_like(lp))
    return lp[:, :, -CAND:].sum((1, 2)).cpu().numpy()


def top1(s: np.ndarray) -> bool:
    return bool(s[0] > s[1:].max())


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--anchors", required=True)
    p.add_argument("--oy", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=2000)
    p.add_argument("--known", type=int, default=0)
    p.add_argument("--tracking", default="on", choices=["on", "off"])
    args = p.parse_args()
    out = Path(args.out)
    (out / "mimi").mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")

    anchors = json.loads(Path(args.anchors).read_text())
    names = anchors["names"]
    items = sample_onset(anchors, args.known or args.n)
    (out / ("known_items.json" if args.known else "items.json")).write_text(
        json.dumps(items)
    )

    from moshi.models import loaders

    info = loaders.CheckpointInfo.from_hf_repo("kyutai/moshiko-pytorch-bf16")
    mimi = info.get_mimi(device=device)
    mimi.set_num_codebooks(8)
    lm = info.get_moshi(device=device, dtype=torch.bfloat16).eval()
    print(
        json.dumps({"delays": list(lm.delays), "n_q": lm.n_q, "dep_q": lm.dep_q}),
        flush=True,
    )
    silence = encode_track(mimi, np.zeros(16000 * 4, np.int16), device)[:, 10]

    needed = sorted(
        {names[it["conv"]] for it in items}
        | {names[(it["conv"] + 1) % len(names)] for it in items}
    )
    codes = {}
    for name in needed:
        cache = out / "mimi" / f"{name}.npz"
        if not cache.exists():
            a = np.load(Path(args.oy) / name / "a.npy")
            b = np.load(Path(args.oy) / name / "b.npy")
            np.savez(
                cache, a=encode_track(mimi, a, device), b=encode_track(mimi, b, device)
            )
        z = np.load(cache)
        codes[name] = (z["a"], z["b"])
    print(json.dumps({"anchors": len(items), "convs_encoded": len(needed)}), flush=True)

    run = None
    if args.tracking == "on" and not args.known:
        import airon

        run = airon.init(
            project="dauys",
            name="dy15-moshi",
            experiment=EXPERIMENT,
            tags=["dy15", "moshi"],
            total_steps=len(items),
        )
    rec = []
    t0 = time.time()
    for i, it in enumerate(items):
        real = score(
            lm, build_rows(it, codes, names, False, bool(args.known), silence), device
        )
        swap = score(
            lm, build_rows(it, codes, names, True, bool(args.known), silence), device
        )
        rec.append(
            {
                "conv": it["conv"],
                "real": top1(real),
                "swap": top1(swap),
                "pmi": top1(real - swap),
                "real_scores": real.tolist(),
                "swap_scores": swap.tolist(),
            }
        )
        if (i + 1) % 50 == 0:
            acc = np.mean([r["real"] for r in rec])
            sw = np.mean([r["swap"] for r in rec])
            logs = {
                "acc_onset": acc,
                "acc_onset_swap": sw,
                "delta": acc - sw,
                "acc_pmi": np.mean([r["pmi"] for r in rec]),
                "anchors_per_s": 50 / (time.time() - t0),
            }
            t0 = time.time()
            print(
                json.dumps(
                    {"done": i + 1, **{k: round(float(v), 4) for k, v in logs.items()}}
                ),
                flush=True,
            )
            if run:
                run.log({f"eval/{k}": float(v) for k, v in logs.items()}, step=i + 1)
    conv = np.array([r["conv"] for r in rec])
    real = np.array([r["real"] for r in rec], float)
    swap = np.array([r["swap"] for r in rec], float)
    pmi = np.array([r["pmi"] for r in rec], float)
    res = {"n": len(rec), "chance": 1 / 16}
    res["acc_onset"], res["acc_onset_lo"], res["acc_onset_hi"] = bootstrap(real, conv)
    res["acc_onset_swap"], _, _ = bootstrap(swap, conv)
    res["delta_onset"], res["delta_onset_lo"], res["delta_onset_hi"] = bootstrap(
        real - swap, conv
    )
    res["acc_pmi"], res["acc_pmi_lo"], res["acc_pmi_hi"] = bootstrap(pmi, conv)
    name = "known_result.json" if args.known else "moshi_result.json"
    (out / name).write_text(json.dumps({"summary": res, "per_anchor": rec}))
    print(json.dumps({"MOSHI" if not args.known else "KNOWN": res}), flush=True)
    if run:
        run.finish()


EXPERIMENT = "Moshi на задаче DY-14: выбор ответа партнёра среди 16 окон того же голоса"

if __name__ == "__main__":
    main()
