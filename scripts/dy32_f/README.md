# DY-32 track F — decoder to audio for the GSLM2 reproduction

Licence: built with the GSLM2 code base (github.com/facebookresearch/gslm2, commit e3020fe) and its SpidR 2.0
encoder checkpoint, both CC BY-NC 4.0 — research use only. The vocoder and duration checkpoints here are trained by us
with that code and inherit the same non-commercial restriction.

## What it is
- **Units**: SpidR 2.0 (~90M, self-supervised, LibriSpeech 960 h audio only; release `final.pt`), argmax codes of
  the prediction heads of layers 5..10, six streams x 256 codes, 50 Hz (320-sample hop at 16 kHz).
- **Vocoder** (`vocoder-<step>/`: `config.json` + `model.safetensors`, standalone unit-HiFi-GAN format, 13,406,594
  params): gslm2 `UnitVocoder` — six 64-dim code embeddings, concatenated -> LayerNorm + Linear(384->128) ->
  HiFi-GAN V1 generator, 16 kHz. Trained with the gslm2 trainer (`python -m gslm2.vocoder`, adversarial MPD+MSD,
  mel loss) on **full-rate** (non-deduplicated) six-stream codes of LJ Speech 1.1 train (12,600 utterances, ~23 h,
  one female speaker, the fixed gslm2 split), batch 64 x 8,960 samples, bf16, lr 2e-4, one RTX PRO 6000.
  The paper trained on 8 GPUs; ours sees fewer samples — see the step in each output's `info.json`.
- **Duration predictor** (`duration_best.pt`): gslm2 conformer (8 layers, 192 dim), one duration class (1..50
  frames) per cross-stream-deduplicated six-tuple, trained with `gslm2-train-duration` on full-rate codes of
  Libri-Light small (60 s VAD segments, ~5% of books held out for validation; the paper validates on a Libri-Light
  large sample instead).
- Single speaker by construction (LJ voice for every input); prosody/speaker of the input are not preserved — as stated
  by the GSLM2 authors.

## Entry point
`scripts/dy32_f_decode.py` (dauys branch `dy32-f`), in the gslm2 native environment (`scripts/dy32_f_pod.sh setup`):
- `extract --wavs *.wav --out units.jsonl` — full-rate six-stream codes (the release extractor).
- `decode --manifest m.jsonl --out dir --vocoder vocoder-N [--duration duration_best.pt] [--mode full|dedup|auto]`
  — rows `{"id"|"audio", "units": "s5|s6|s7|s8|s9|s10"}` (raw codes 0..255); full-rate rows go straight to the
  vocoder, deduplicated rows get predicted durations first.
- `round` — one decoding round of everything track E has put in S3, with the newest vocoder checkpoint.
- LM continuations follow the release recipe (`gslm2.synthesis_continuations`): durations are predicted on the
  deduplicated prompt context + continuation, the continuation alone is expanded and capped at 500 frames (10 s);
  the vocoder is run on the last 2 s of the full-rate prompt followed by the continuation, and the prompt part is
  cut away, so the WAV is the continuation only.

## Where
S3 `outputs/dauys/dy32/`:
- `resynth/f/XX.wav` — real_XX.wav -> our extraction -> full rate -> vocoder (pure channel check).
- `resynth/f-dedup/XX.wav` — same units, cross-stream deduplicated -> duration predictor -> vocoder.
- `resynth/e-fullrate/`, `resynth/e-dedup/` — the same from track E's units.
- `gen/e-<variant>/XX.wav` + `info.json` — track E's LM continuations.
- `f/step<N>/...` — every round kept by vocoder step; `f/latest_round.json` — the last round's settings and WER.
- `f/checkpoints/` — vocoder and duration checkpoints; HF `manifestro/dauys-runs` `dy32/f/` holds the same.
