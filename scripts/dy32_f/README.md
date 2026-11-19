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

## Final state (19 Nov 2026)
- Vocoder: `vocoder-114141` (stopped 07:11 UTC; ~6.9 h on one GPU, batch 64). Validation mel L1 (16 LJ valid
  utterances): 0.786 at 6k, 0.774 at 30k, 0.776 at 114k — flat after ~10k steps.
- Duration predictor: `duration_best.pt` = step 6000 of a shortened run (batch 32 x 6,000 steps instead of
  64 x 25,000; it shared the GPU with the vocoder). Held-out Libri-Light small: val loss 0.292, MAE 0.14 frames,
  duration correlation 0.886, total-length ratio 0.986.

## Channel check (measuring only; Whisper large-v3, English, greedy, whisper text normaliser)
`lj_test` = 40 LJ Speech test utterances through units -> vocoder vs the LJ transcripts.
`real f` = the 50 pack `real_XX.wav` through units -> vocoder (full rate) vs Whisper on the original `real_XX.wav`
(so it measures words lost in the channel, not recogniser errors; one item, real_02, is not English).
`real f-dedup` = same, but cross-stream deduplicated and re-timed by the duration predictor.

| vocoder step | lj_test WER | real f WER | real f-dedup WER |
|---:|---:|---:|---:|
| 4,000 | 3.6% | 28.0% | – |
| 6,000 | 3.6% | 25.8% | 26.9% (dur. step 500) |
| 12,000 | 4.2% | 25.1% | 25.0% |
| 24,000 | 3.2% | 24.3% | 25.1% |
| 48,000 | 3.0% | 23.4% | 24.4% |
| 72,000 | 2.7% | 23.7% | 25.2% |
| 96,000 | 3.3% | 23.9% | 24.3% |
| 114,141 | 2.7% | 23.7% | 23.1% |

Every step's WER is in `wer_log.jsonl`. Run-to-run noise is about +-1.5 points (one round gave 31% for f-dedup).
The vocoder plateaus early: the ~24% word loss on conversational speech is in the units (SpidR 2.0 trained on
read LibriSpeech), not in the vocoder. Track E's full-rate units of real_XX are identical to ours (49/50 items
bit-identical, 1 code of 2,994 differs in real_36).
