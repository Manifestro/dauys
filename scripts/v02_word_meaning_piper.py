"""A second synthesizer for the word-meaning test: Piper, never used in training.

Every word is spoken in its held-out frame by several Piper voices, in three
versions: clean, with pink noise at 15 dB SNR, and band-limited to a phone
line (8 kHz, 300–3400 Hz). Layout matches the live recordings
(``<output>/<variant>/<voice>/<topic>-<word>.wav``, 24 kHz), so
``v02_word_meaning_live.py encode --all-words`` turns each variant into a split.
Runs in its own environment (``pip install piper-tts soundfile scipy
huggingface_hub``); it does not import dauys.
"""

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile
from huggingface_hub import hf_hub_download
from scipy.signal import butter, resample_poly, sosfilt

VOICES = (
    "en/en_US/lessac/medium/en_US-lessac-medium",
    "en/en_US/amy/medium/en_US-amy-medium",
    "en/en_US/ryan/high/en_US-ryan-high",
    "en/en_US/joe/medium/en_US-joe-medium",
    "en/en_GB/alan/medium/en_GB-alan-medium",
    "en/en_GB/jenny_dioco/medium/en_GB-jenny_dioco-medium",
)


def phrases():
    sys.path.insert(0, str(Path(__file__).parent))
    from v02_word_meaning_text import TOPICS

    for topic, spec in TOPICS.items():
        frames = spec["frames"]
        for index, word in enumerate(spec["words"]):
            yield topic, word, frames[index % len(frames)].format(w=word)


def pink_noise(length: int, rng: np.random.Generator) -> np.ndarray:
    spectrum = np.fft.rfft(rng.standard_normal(length))
    spectrum /= np.sqrt(np.arange(1, len(spectrum) + 1))
    noise = np.fft.irfft(spectrum, n=length)
    return noise / (np.std(noise) + 1e-9)


def variants(audio: np.ndarray, rng: np.random.Generator) -> dict[str, np.ndarray]:
    power = np.sqrt(np.mean(audio**2)) + 1e-9
    noisy = audio + pink_noise(len(audio), rng) * power / 10 ** (15 / 20)
    narrow = resample_poly(audio, 1, 3)
    band = butter(4, (300, 3400), btype="bandpass", fs=8000, output="sos")
    phone = resample_poly(sosfilt(band, narrow), 3, 1)[: len(audio)]
    return {"clean": audio, "noise": noisy, "phone": phone}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rng = np.random.default_rng(0)
    for path in VOICES:
        model = hf_hub_download("rhasspy/piper-voices", f"{path}.onnx")
        hf_hub_download("rhasspy/piper-voices", f"{path}.onnx.json")
        voice = Path(path).name
        for topic, word, text in phrases():
            raw = args.output / "raw" / f"{voice}-{topic}-{word}.wav"
            raw.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                [sys.executable, "-m", "piper", "-m", model, "-f", str(raw)],
                input=text.encode(),
                check=True,
                capture_output=True,
            )
            audio, rate = soundfile.read(raw, dtype="float32", always_2d=True)
            audio = resample_poly(audio.mean(1), 24_000, rate).astype(np.float32)
            for name, version in variants(audio, rng).items():
                target = args.output / name / voice / f"{topic}-{word}.wav"
                target.parent.mkdir(parents=True, exist_ok=True)
                peak = max(1.0, float(np.abs(version).max()))
                soundfile.write(target, (version / peak).astype(np.float32), 24_000)
        print(f"voice done {voice}", flush=True)


if __name__ == "__main__":
    main()
