"""DY-40 checklist on the machine, before training (results -> --out JSON, also printed).

leak:    no training shard is held out (sha1 % 50 == 0); the validation shards are the held-out la-00030, la-00071;
         the DY-32 prompt packs come from held-out shards only (their info.json / names are checked when present).
layout:  delay / undelay round trip; generation vs teacher forcing: for a random small model, every drawn code's
         log-prob during generate() equals the teacher-forced log-prob of the same position and level (|diff| < 5e-2
         in bf16), so the delayed layout is the same in training, generation and the heard test.
known:   the 50 DY-32 real continuations through Mimi 8 codes and back: Whisper large-v3-turbo WER against the
         original's transcript, median (DY-32 measured 0.167 for this channel with its recogniser).

    python scripts/dy40_checklist.py --units /opt/dlami/nvme/units --out /data/dy40/checklist.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from dauys.dy40_lm import DLConfig, DepthLM, delay, generate, undelay
from dy21_train import held_out
from dy40_mimi import HELD, SR, Mimi, read_pack, to16


def check_leak(units: Path) -> dict:
    names = sorted(f.stem for f in units.glob("*.npz"))
    train = [n for n in names if not held_out(n)]
    val = [n for n in names if held_out(n)]
    return {"units_shards": len(names), "train": len(train), "val": val,
            "ok": bool(train) and sorted(val) == sorted(HELD) and not any(held_out(n) for n in train)}


@torch.no_grad()
def check_layout(dev) -> dict:
    rng = np.random.default_rng(0)
    fr = rng.integers(0, 2048, size=(37, 8))
    d = delay(fr, 2049)
    rt = bool(np.array_equal(undelay(d, 37), fr)) and d.shape == (38, 8) and (d[0, 1:] == 2049).all() and d[-1, 0] == 2049
    torch.manual_seed(0)
    cfg = DLConfig(d=128, layers=2, heads=4, ffn=256, ctx=512, dd=64, dlayers=2, dheads=4, dffn=128)
    m = DepthLM(cfg).to(dev).eval()
    prompts = np.concatenate([np.full((3, 1, 8), cfg.k), rng.integers(0, 2048, size=(3, 20, 8))], 1)
    tr: list = []
    gen = generate(m, prompts, 6, "t10", 7, dev, trace=tr)
    ids = tr[-1][1]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        h = m.states(ids[:, :-1])
        lg = m.depth_logits(h.reshape(-1, h.shape[-1]), ids[:, 1:].reshape(-1, 8)).view(3, ids.shape[1] - 1, 8, -1)
    lg[..., cfg.k :] = -float("inf")
    lp = torch.log_softmax(lg, -1)
    diffs = []
    for p, k, tok, lpg in tr[:-1]:
        lpt = lp[:, p - 1, k].gather(1, tok[:, None])[:, 0]
        diffs.append(float((lpt - lpg).abs().max()))
    # the generated frames are what ids hold after the prompt, read back through undelay
    back = np.stack([undelay(x, 6, start=20) for x in ids[:, 1:].cpu().numpy()])
    same = bool(np.array_equal(back, gen))
    prompt_kept = bool(np.array_equal(undelay(ids[:, 1:].cpu().numpy()[0], 20), prompts[0, 1:]))
    return {"delay_roundtrip": rt, "draws": len(diffs), "max_abs_logprob_diff": max(diffs), "undelay_matches_output": same,
            "prompt_kept": prompt_kept, "ok": rt and same and prompt_kept and max(diffs) < 5e-2 and len(diffs) == 6 * 8}


def check_known(dev, pack: Path) -> dict:
    import jiwer
    from transformers import pipeline

    mm = Mimi(dev)
    names, waves = read_pack(pack, "real", "prompts")
    codes = mm.encode(waves)
    rs = [to16(mm.decode(c)) for c in codes]
    asr = pipeline("automatic-speech-recognition", "openai/whisper-large-v3-turbo", torch_dtype=torch.float16, device=dev)
    kw = {"generate_kwargs": {"language": "en", "task": "transcribe"}, "batch_size": 16}
    norm = jiwer.Compose([jiwer.ToLowerCase(), jiwer.RemovePunctuation(), jiwer.RemoveMultipleSpaces(), jiwer.Strip()])
    ref = [r["text"] for r in asr([{"raw": w, "sampling_rate": SR} for w in waves], **kw)]
    hyp = [r["text"] for r in asr([{"raw": w, "sampling_rate": SR} for w in rs], **kw)]
    wers = [jiwer.wer(norm(r), norm(h)) if norm(r) else 0.0 for r, h in zip(ref, hyp)]
    med = float(np.median(wers))
    return {"pieces": len(names), "frames_per_10s": round(float(np.mean([len(c) for c in codes])) / (len(waves[0]) / SR) * 10, 2),
            "wer_median": round(med, 4), "wer_mean": round(float(np.mean(wers)), 4), "dy32_reference": 0.167,
            "ok": med <= 0.25}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--units", required=True)
    p.add_argument("--pack", default="/data/dy32/prompts")
    p.add_argument("--out", required=True)
    p.add_argument("--only", default="leak,layout,known")
    a = p.parse_args()
    dev = torch.device("cuda")
    res = {}
    if "leak" in a.only:
        res["leak"] = check_leak(Path(a.units))
    if "layout" in a.only:
        res["layout"] = check_layout(dev)
    if "known" in a.only:
        res["known"] = check_known(dev, Path(a.pack))
    res["ok"] = all(v["ok"] for v in res.values())
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1), flush=True)
    print("CHECKLIST", "PASS" if res["ok"] else "FAIL", flush=True)


if __name__ == "__main__":
    main()
