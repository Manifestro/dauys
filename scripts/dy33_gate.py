"""DY-33 track M: gate service. Every ~2 min: new or changed folders under s3://.../dy33/resynth/<name>/ (the 50
held-out real_XX.wav through track S's slow codes) are measured exactly as the DY-32 channel measure
(dy32_channel: letters recogniser + Whisper on the resynthesised audio vs the same recogniser on the original;
median letters WER is the gate number, pre-registered threshold 0.30) and gate.json / gate.md are rewritten in
s3://.../dy33/eval/. The recognisers stay loaded between polls.

    python scripts/dy33_gate.py --until 14:45
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from dy32_channel import W, Rec, Table, load_wavs, log, score

S3_33 = "s3://manifestro-dataset-876108973178-us-east-1-an/outputs/dauys/dy33"
LOCAL = Path("/data/dy33/resynth")
THRESHOLD = 0.30
REFS = {"km-corpus-K2000 (DY-32 single stream, 50 Hz)": 0.185, "decoder ceiling knn-cont (DY-32)": 0.138,
        "Mimi q4 (DY-32)": 0.235}


def sh(cmd: str) -> str:
    return subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True).stdout


def listing() -> dict[str, tuple]:
    """folder -> signature (sorted (file, size, timestamp)) of its wavs in S3."""
    try:
        out = sh(f"aws s3 ls {S3_33}/resynth/ --recursive")
    except subprocess.CalledProcessError:
        return {}
    sig: dict[str, list] = {}
    for line in out.splitlines():
        p = line.split()
        if len(p) < 4 or not p[3].endswith(".wav"):
            continue
        key = p[3].split("/resynth/", 1)[1]
        if "/" not in key:
            continue
        folder, f = key.rsplit("/", 1)
        sig.setdefault(folder, []).append((f, p[2], p[0] + p[1]))
    return {k: tuple(sorted(v)) for k, v in sig.items()}


def write(state: dict) -> None:
    gj = {"updated_utc": time.strftime("%Y-%m-%d %H:%M:%S"), "threshold_median_letters_wer": THRESHOLD,
          "measure": "dy32_channel measure: letters recogniser (wav2vec2-large-960h-lv60-self, greedy) WER vs the same "
                     "recogniser on the original real_XX.wav; Whisper large-v3-turbo as second opinion",
          "references": REFS, "folders": state}
    (W / "gate.json").write_text(json.dumps(gj, indent=1))
    L = ["# DY-33 gate (track M)", "", f"Updated {gj['updated_utc']} UTC. Gate: median letters WER <= {THRESHOLD} on the 50 "
         "held-out real_XX.wav through the slow codes and the kNN-VC vocoder.", "",
         "| folder | n | letters WER median | mean | Whisper WER median | <= 0.30 | measured (UTC) |", "|---|---|---|---|---|---|---|"]
    for n, r in sorted(state.items(), key=lambda x: x[1].get("median", 9)):
        if "median" in r:
            L.append(f"| {n} | {r['n']} | {r['median']:.3f} | {r['mean']:.3f} | {r['whisper_median']:.3f} | "
                     f"{'yes' if r['pass'] else 'no'} | {r['measured_utc']} |")
        else:
            L.append(f"| {n} | {r.get('n', 0)} | - | - | - | - | {r.get('status', '')} |")
    L += ["", "References (DY-32, same measure): " + "; ".join(f"{k} {v}" for k, v in REFS.items())]
    (W / "gate.md").write_text("\n".join(L) + "\n")
    sh(f"aws s3 cp {W}/gate.json {S3_33}/eval/gate.json --only-show-errors && aws s3 cp {W}/gate.md {S3_33}/eval/gate.md --only-show-errors")


def main() -> None:
    import torch

    p = argparse.ArgumentParser()
    p.add_argument("--until", default="14:45", help="UTC hh:mm today")
    p.add_argument("--every", type=int, default=120)
    a = p.parse_args()
    end = time.mktime(time.strptime(time.strftime("%Y-%m-%d ") + a.until, "%Y-%m-%d %H:%M")) - time.timezone
    device = torch.device("cuda")
    ref = json.loads((W / "ref.json").read_text())
    rec = Rec(device)
    f = W / "gate.json"
    state = json.loads(f.read_text())["folders"] if f.exists() else {}
    seen = {k: tuple(map(tuple, v.get("sig", []))) for k, v in state.items()}
    write(state)
    log("gate service up; until", a.until, "UTC")
    while time.time() < end and not (W / "STOP_GATE").exists():
        for name, sig in sorted(listing().items()):
            if seen.get(name) == sig:
                continue
            wavs = [x for x in sig if x[0][:2].isdigit() and x[0].endswith(".wav")]
            if len(wavs) < 50:
                state[name] = {"n": len(wavs), "status": f"incomplete ({len(wavs)} of 50 wavs)"}
                write(state)
                continue
            sh(f"aws s3 sync {S3_33}/resynth/{name}/ {LOCAL}/{name}/ --only-show-errors --exclude '*' --include '*.wav'")
            waves = load_wavs(LOCAL / name)
            if waves is None:
                continue
            seen[name] = sig
            h = rec(waves)
            sc = score(ref, h)
            state[name] = {"n": 50, "median": sc["letters_wer"]["median"], "mean": sc["letters_wer"]["mean"],
                           "whisper_median": sc["whisper_wer"]["median"], "cer_median": sc["letters_cer"]["median"],
                           "pass": bool(sc["letters_wer"]["median"] <= THRESHOLD), "measured_utc": time.strftime("%H:%M:%S"),
                           "sig": [list(x) for x in sig], "texts_first3": h["letters"][:3]}
            Table().add(f"dy33-{name}", f"DY-33 slow codes, track S folder resynth/{name}", "12.5 steps/s x 4 codes", sc, h)
            write(state)
            log("GATE", name, state[name]["median"], state[name]["whisper_median"], "PASS" if state[name]["pass"] else "")
        write(state)  # refresh updated_utc every poll so readers see the service is alive
        time.sleep(a.every)
    log("GATE SERVICE END")


if __name__ == "__main__":
    main()
