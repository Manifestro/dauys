"""Export a DY-41 checkpoint as a self-contained private HF model repo: weights (safetensors), config, the model code
(dauys/dy21_lm.py, dauys/dy40_lm.py), inference.py (wav -> Mimi -> continuation -> wav) and a README; then a check
that loads the uploaded files fresh and speaks 10 s after a real prompt (sample.wav in the repo).

    python scripts/dy41_export.py --uri s3://airon-checkpoints-.../last.pt --repo manifestro/dauys-dy41-mimi-lm
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
REPO_ROOT = Path(__file__).resolve().parents[1]

INFERENCE = '''"""Continue speech with the DY-41 model: a 16/24 kHz wav prompt -> Mimi codes -> N seconds more -> wav.

    pip install torch torchaudio "transformers>=4.46" safetensors soundfile huggingface_hub
    python inference.py prompt.wav out.wav --seconds 10 --sampling ra08
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, soundfile as sf, torch, torchaudio
from safetensors.torch import load_file
from transformers import MimiModel
sys.path.insert(0, str(Path(__file__).parent))
from dauys.dy40_lm import DLConfig, DepthLM, generate

MIMI, MIMI_REV, MSR, HOP = "kyutai/mimi", "89091b3e466eb6a9d11e537bf26b144f194978f7", 24000, 1920


def load_lm(d: Path, dev):
    cfg = json.loads((d / "config.json").read_text())["model"]
    m = DepthLM(DLConfig(**cfg)).to(dev).eval()
    m.load_state_dict({k: v.float() for k, v in load_file(d / "model.safetensors").items()})
    return m


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("prompt"); p.add_argument("out")
    p.add_argument("--seconds", type=float, default=10.0)
    p.add_argument("--sampling", default="ra08", choices=["ra08", "t10"])
    p.add_argument("--prompt-seconds", type=float, default=15.0, help="use the last N s of the prompt")
    p.add_argument("--seed", type=int, default=1234)
    a = p.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = Path(__file__).parent
    lm = load_lm(d, dev)
    mimi = MimiModel.from_pretrained(MIMI, revision=MIMI_REV).to(dev).eval()
    w, sr = sf.read(a.prompt, dtype="float32")
    w = torch.from_numpy(w if w.ndim == 1 else w.mean(1))
    w = torchaudio.functional.resample(w, sr, MSR)[-int(a.prompt_seconds * MSR):].to(dev)
    n = -(-len(w) // HOP) * HOP
    x = torch.zeros(1, 1, n, device=dev); x[0, 0, : len(w)] = w
    codes = mimi.encode(x, num_quantizers=8).audio_codes[0].T.cpu().numpy()  # [frames, 8]
    prompt = np.r_[np.full((1, 8), lm.cfg.k), codes][None].astype(np.int64)  # EOS frame + prompt frames
    new = generate(lm, prompt, int(round(a.seconds * 12.5)), a.sampling, a.seed, dev)[0]
    y = mimi.decode(torch.from_numpy(new.T[None].astype(np.int64)).to(dev)).audio_values[0, 0].cpu().numpy()
    sf.write(a.out, y, MSR)
    print("wrote", a.out, len(y) / MSR, "s")


if __name__ == "__main__":
    main()
'''


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--uri", required=True)
    p.add_argument("--repo", default="manifestro/dauys-dy41-mimi-lm")
    p.add_argument("--out", default="/data/export")
    a = p.parse_args()
    import airon
    from huggingface_hub import HfApi
    from safetensors.torch import save_file

    out = Path(a.out)
    shutil.rmtree(out, ignore_errors=True)
    (out / "dauys").mkdir(parents=True)
    ck = torch.load(airon.fetch(a.uri), map_location="cpu", weights_only=False)
    if ck.get("best_model") is not None:  # a full last.pt: take the best weights it carries
        sd, step, val = ck["best_model"], ck["best_step"], ck["best_val"]
    else:
        sd, step, val = ck["model"], ck["step"], ck["val"]
    sd = {k: v.detach().to(torch.bfloat16).contiguous() for k, v in sd.items()}
    save_file(sd, out / "model.safetensors")
    nparams = sum(v.numel() for v in sd.values())
    cfg = ck["cfg"]
    meta = {"model": cfg, "params": nparams, "step": int(step), "val": {k: float(v) for k, v in val.items()},
            "dtype": "bfloat16", "source": f"DY-41 (Airon), job DY-41/24, {a.uri}",
            "codec": "kyutai/mimi rev 89091b3e466eb6a9d11e537bf26b144f194978f7, 8 codebooks x 2048, 12.5 Hz, 24 kHz",
            "layout": "EOS frame, then frames; acoustic codes (1..7) one frame after code 0 (dy40_lm.delay)",
            "training": "DuplexChat en-16k, ~14.2k h usable (runs >= 164 s), loss (10*CE0 + sum CE1..7)/17, no text"}
    (out / "config.json").write_text(json.dumps(meta, indent=2))
    for f in ("dy21_lm.py", "dy40_lm.py"):
        shutil.copy(REPO_ROOT / "src/dauys" / f, out / "dauys" / f)
    (out / "dauys/__init__.py").write_text("")
    (out / "inference.py").write_text(INFERENCE)
    (out / "README.md").write_text(f"""# dauys DY-41: speech LM on Mimi codes (private)

Textless spoken-language model: {nparams / 1e6:.1f}M parameters (temporal transformer 24 x 1024 over 12.5 Hz frames +
depth transformer 4 x 768 over the 8 Mimi codebooks, Moshi-style). Trained without any text on ~14.2k hours of
conversational English (DuplexChat en-16k). Weights = best validation point at step {int(step)} (val CE
{float(val['ce']):.3f}), taken from the running DY-41 job; the final DY-41 checkpoint will replace it.

It speaks fluent, natural-sounding English with human tempo, but its continuations do not follow the topic of the
prompt (DY-41 mid-run evaluation: Dself 0.58 vs 0.92 for real speech).

## Use

```bash
pip install torch torchaudio "transformers>=4.46" safetensors soundfile huggingface_hub
huggingface-cli download {a.repo} --local-dir dy41
python dy41/inference.py prompt.wav continuation.wav --seconds 10 --sampling ra08
```

- `sampling`: `ra08` (keeps the previous frame's code probability, the rest at temperature 0.8; steadier) or `t10`
  (temperature 1.0 on every code).
- In Python: `from dauys.dy40_lm import DLConfig, DepthLM, generate` with `config.json["model"]` and
  `model.safetensors` (bfloat16; cast to float for inference); see `inference.py`.
- Codec: `kyutai/mimi` at revision 89091b3e (CC-BY 4.0). Layout: an EOS frame (all codes = 2048), then the prompt's
  frames; acoustic codes are one frame after code 0 (handled inside `generate`).

`sample.wav`: the last 3 s of a real held-out prompt, then 10 s generated by this checkpoint (ra08).
""")
    api = HfApi(token=os.environ["HF_TOKEN"])
    api.create_repo(a.repo, private=True, exist_ok=True)
    assert api.repo_info(a.repo).private
    api.upload_folder(folder_path=str(out), repo_id=a.repo)
    print("UPLOADED", a.repo, "step", int(step), flush=True)

    # check: fresh download of the repo, load, speak after a real prompt
    import subprocess

    from huggingface_hub import snapshot_download

    d = Path(snapshot_download(a.repo, token=os.environ["HF_TOKEN"], local_dir="/data/export_check"))
    prompt = Path("/data/dy32/prompts/prompt_47.wav")
    subprocess.run(f"aws s3 cp s3://manifestro-dataset-876108973178-us-east-1-an/outputs/dauys/dy32/prompts/prompt_47.wav {prompt} --only-show-errors",
                   shell=True, check=True)
    subprocess.run([sys.executable, str(d / "inference.py"), str(prompt), "/data/export_check/cont.wav", "--seconds", "10"],
                   check=True)
    import soundfile as sf
    import torchaudio

    y, sr = sf.read("/data/export_check/cont.wav", dtype="float32")
    assert sr == 24000 and 9.5 < len(y) / sr < 10.5 and np.isfinite(y).all() and np.abs(y).max() > 1e-3, (sr, len(y))
    pw, psr = sf.read(prompt, dtype="float32")
    tail = torchaudio.functional.resample(torch.from_numpy(pw[-3 * psr:]), psr, 24000).numpy()
    sf.write("/data/export_check/sample.wav", np.concatenate([tail, np.zeros(7200, np.float32), y]), 24000)
    api.upload_file(path_or_fileobj="/data/export_check/sample.wav", path_in_repo="sample.wav", repo_id=a.repo)
    print("CHECK OK: fresh download loads and speaks", round(len(y) / sr, 2), "s", flush=True)


if __name__ == "__main__":
    main()
