"""DY-20: DuplexChat pieces separated by Kezek v1 (S3 outputs/duplexchat/kezek-v1-1000h) to the Open Yap layout.

Each shard dir holds a.bin / b.bin (int16 16 kHz, persons A and B, pieces concatenated), index.npy
(start, length) and pieces.jsonl. A piece becomes ``OUT/dc_<shard>_<i>/{a,b}.npy`` when it is at least
``--min-s`` long, Kezek found speech of both people (each >= ``--min-speech-s``) and had no error.
Writes ``OUT/dc_names.json`` (kept names, hours) and deletes each shard's .bin after converting it.

    python scripts/dy20_dc_prepare.py --src /data/dcraw --out /data/dc
"""

import argparse
import json
from pathlib import Path

import numpy as np

SR = 16_000


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--min-s", type=float, default=80.0)
    p.add_argument("--min-speech-s", type=float, default=10.0)
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    names, kept_s, all_s, n_all = [], 0.0, 0.0, 0
    for d in sorted(Path(args.src).iterdir()):
        if not (d / "index.npy").exists():
            continue
        idx = np.load(d / "index.npy")
        rows = [json.loads(line) for line in open(d / "pieces.jsonl")]
        a = np.memmap(d / "a.bin", dtype=np.int16, mode="r")
        b = np.memmap(d / "b.bin", dtype=np.int16, mode="r")
        for i, ((st, n), r) in enumerate(zip(idx, rows)):
            n_all += 1
            all_s += n / SR
            if (
                n / SR < args.min_s
                or "error" in r
                or r["speech_A_s"] < args.min_speech_s
                or r["speech_B_s"] < args.min_speech_s
            ):
                continue
            name = f"dc_{d.name}_{i:05d}"
            (out / name).mkdir(exist_ok=True)
            np.save(out / name / "a.npy", np.asarray(a[st : st + n]))
            np.save(out / name / "b.npy", np.asarray(b[st : st + n]))
            names.append(name)
            kept_s += n / SR
        del a, b
        for f in ("a.bin", "b.bin"):
            (d / f).unlink()
    info = {
        "pieces": n_all,
        "hours": all_s / 3600,
        "kept": len(names),
        "kept_hours": kept_s / 3600,
        "min_s": args.min_s,
        "min_speech_s": args.min_speech_s,
        "names": names,
    }
    (out / "dc_names.json").write_text(json.dumps(info))
    print(json.dumps({k: v for k, v in info.items() if k != "names"}), flush=True)


if __name__ == "__main__":
    main()
