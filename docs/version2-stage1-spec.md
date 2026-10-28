# Dauys v0.2 — Stage 1 specification

Stage 1 trains the full model on about 1,000 hours of unlabeled single-speaker
speech. The ear learns what was said; the dialogue tower learns to continue
from what the ear heard. Stage 2 (dialogue) starts only after Stage 1 passes
Gate A below.

This document fixes the recipe, the measurements and the decisions in advance.
A run ends in exactly one of three outcomes: **GO**, **FIX** or **STOP**. There
is no open-ended tuning.

## 1. Why this stage changes

Lessons from the 2026-10-27/28 runs:

- On the previous Stage 1, channel A was always silence. The cross-attention
  path from the ear was never used (fusion-off ΔNLL −0.01). Listening was
  learned only from ~3k Stage 2 dialogues.
- All losses and metrics counted silence frames. Held-out CB0 accuracy read 74%,
  but on frames where the model actually speaks it was 37.8%.
- At 100 hours, a 94M model overfit by step 5,000.
- Intent as regression onto the next second plateaued early (cosine 0.66) and
  most likely learned voice, not content.

Stage 1 now: A→B continuation on ~1,000 hours, losses on speech frames,
intent as a contrastive choice among candidates from the same speaker.

## 2. Data

| Item | Value |
| --- | --- |
| Corpus | LibriSpeech (train.100 + train.360 + train.other.500) + LibriLight subset, ≈ 1,000 h |
| Format | mono 24 kHz → Mimi codes (8 codebooks × 2,048, 12.5 frames/s), one fixed encoder for all splits (TF32 off) |
| Held-out | LibriSpeech dev-clean + dev-other. Speakers disjoint from training. |
| Leakage check (before training) | No held-out speaker ID may appear in any training split, LibriLight included. The run does not start until this check passes. |

### A→B layout

Each training item is cut from one recording of one speaker:

1. Find pauses: runs of ≥ 3 frames (≥ 240 ms) with no speech activity.
2. Choose a cut pause so that ≥ 3 s of speech lies before it and ≥ 2 s after it.
3. **Channel A** = frames before the cut, then Mimi silence codes.
   **Channel B** = Mimi silence codes up to the cut, then the frames after it.
   The natural pause stays in the timeline: B starts after it, like a reply.
4. A 256-frame crop (≈ 20 s) is placed around the cut, with ≥ 3 s of A before it.

B is the future of A and never appears in A. The ear hears A only. The dialogue
tower sees A through cross-attention and its own B history.

## 3. Model

| Part | Setting |
| --- | --- |
| Base | `dauys-hybrid-v0.2-screen.yaml` (dialogue 768, d_state 128, depth decoder 4 × 512) |
| Ear (listening tower) | **widened to 512 × 6 layers** (was 384 × 4); trains from random weights |
| Dialogue tower | 6 Mamba-2 blocks initialized from `state-spaces/mamba2-130m` (residual blocks only, no embeddings, no text head); fusion blocks start as identities |
| Size | ≈ 100M trainable parameters (exact count printed at launch) |
| Self-hearing | off |

## 4. Objectives

Every item trains three objectives at once.

### 4.1 A→B continuation (dialogue tower)

- CB0 and depth (CB1–7) cross-entropy on B **speech frames only**.
- Activity cross-entropy on **all valid frames**. The model must learn when
  not to speak.
- C4 acoustic dropout on A, label smoothing 0.05 on CB0.

### 4.2 Contrastive intent (dialogue tower)

- At the last frame before B starts, the intent head maps the dialogue state to
  a vector ẑ.
- Candidates: the true B continuation and K − 1 = 7 other continuations by the
  **same speaker** from the same batch. Each candidate is encoded by an EMA copy
  of the ear over its first 2.5 s (32 frames) and mean-pooled.
- Loss: InfoNCE, temperature 0.1. All candidates share one voice, so only the
  content tells them apart.
- Batches are built as 16 speakers × 4 items, which supplies the same-speaker
  negatives (missing slots are filled from the speaker's other items in the
  corpus).

### 4.3 Next-frame listening (ear)

- C1: the ear predicts the next A CB0 code on A speech frames, weight 1.0 → 0.3
  over training (existing schedule).

Weights: continuation 1.0 (CB0) / 0.5 (depth) / 0.2 (activity); intent 0.5;
C1 as above.

## 5. Training recipe

| Setting | Value |
| --- | --- |
| Steps | 20,000 (≈ 7 epochs at ≈ 2,800 steps/epoch) |
| Batch | 64 (16 speakers × 4 items), crop 256 frames |
| Optimizer | AdamW, LR 2e-4, cosine, warmup 3%, weight decay 0.05 |
| Precision | bfloat16 autocast, gradient clip 1.0 |
| Regularization | dropout 0.1, C4 acoustic dropout, label smoothing 0.05 |
| Checkpoints | every 2,000 steps; best kept by held-out speech-frame CB0 NLL |
| Seed | 42 |
| Hardware / budget | one A40; ≤ 10 h and ≤ $7 per run |

## 6. Measurements

All on the held-out speakers, at every checkpoint unless marked *final*.

| ID | Metric | How |
| --- | --- | --- |
| M1 | Speech-frame CB0 NLL | B speech frames only, teacher forced |
| M2 | Bigram reference | count-based CB0 bigram fitted on training tokens, scored on the same frames (computed once before training) |
| M3 | Future selection | the intent head picks the true continuation among 8 same-speaker candidates; chance 12.5% |
| M4 | Fusion dependence | M1 with all cross-attention blocks off minus M1 with them on |
| M5 | Empty continuations (*final*) | 100 held-out items, generate 75 frames of B with the C6 decoder; empty = ASR returns no words **or** RMS < 0.003 |
| M6 | Overfit gap | M1 on a training sample minus M1 held-out, as CB0 accuracy points |
| M7 | Speaker probe (*final*, report only) | linear probe for speaker ID on frozen ear states, held-out speakers split by utterance |
| M8 | Listening (*final*) | 8 B continuations, model channel only, ≥ 4 s each, rated by the owner |

## 7. Gate A: decisions

Each metric falls into one band. Bands are fixed before the run.

| Metric | GO | FIX | FAIL |
| --- | --- | --- | --- |
| M1 − M2 (speech-frame NLL below bigram) | ≤ −0.5 nat | −0.5 … −0.2 | > −0.2 |
| M3 future selection | ≥ 70% | 40–70% | < 40% |
| M4 fusion dependence | ≥ 0.30 nat | 0.05–0.30 | < 0.05 |
| M5 empty continuations | < 30% | 30–50% | > 50% |
| M6 overfit gap | < 10 pts | 10–15 | > 15 |
| M8 listening | ≥ 3 of 8 contain words the owner recognises | 1–2 of 8 | 0 of 8 |

M7 has no band. It is recorded for Stage 2 and the voice work deferred to v0.3.

**Outcome of a run** (taken at its best checkpoint):

- **GO**: every metric is in GO. → Stage 2 starts from this checkpoint.
- **FIX**: no metric in FAIL and at least one in FIX. → One fix iteration (§8).
- **STOP**: any metric in FAIL. → No tuning. Write the diagnosis (§9) and
  revise this document before any new run.

### Early abort during a run

The run is stopped early, and counts as STOP, if:

- at step 4,000, M1 is not below M2 at all; or
- at step 8,000, M3 < 25%; or
- at any checkpoint, M4 < 0.01 nat after step 6,000 (the ear is ignored); or
- the loss turns NaN, or intent embeddings collapse (spread < 0.05).

## 8. Fix iterations

At most **two** fix iterations follow the first run. Each changes only the
knob tied to the failing metric, chosen from this table. A fix may not add new
objectives, data or architecture.

| Metric in FIX | Allowed change |
| --- | --- |
| M1 | +50% steps (to 30,000), or LR 3e-4 |
| M3 | intent weight 0.5 → 1.0, or temperature 0.1 → 0.07 |
| M4 | C1 weight halved; A acoustic dropout 0.5 → 0.3 |
| M5 | none in training; re-score with decoder θ_on grid {0.25, 0.35, 0.5} |
| M6 | dropout 0.1 → 0.2, or stop at the best earlier checkpoint |
| M8 | follows M1/M5; no separate change |

If a run is still FIX after the second iteration, the outcome is **STOP**.

Total Stage 1 budget: 3 runs, ≤ $21 GPU. When it is spent, the outcome is
whatever the last run gave.

## 9. Diagnosis on STOP

Write a short report with these checks before changing the design:

1. Oracle check: does Mimi round-trip of held-out B sound clean (ceiling)?
2. Does M1 drop on training items but not held-out (memorization) or on
   neither (capacity/optimization)?
3. M4 ≈ 0: is A actually present in the batches (inspect 5 items as audio)?
4. M3 near chance: are negatives truly same-speaker; does the EMA ear collapse?
5. M5 high with good M1: decoding, not training — re-score with other C6
   settings first.

## 10. Deliverables

- Best checkpoint (base model under `model`), its metrics JSON, the run's W&B
  link.
- Gate A table with the band of each metric and the outcome.
- 8 listening samples (model channel only).
- On STOP: the diagnosis from §9.

## 11. Not in Stage 1 (deferred)

- Voice-invariance positives (TTS re-rendering, formant shift): v0.3, only if
  M7 shows speaker identity dominating the ear.
- Content probe on word minimal pairs and intonation probe: v0.3.
- Self-hearing, inner monologue, whole-conversation training: after Stage 2.
- Question→answer, relevance, turn-taking, backchannels, yielding, memory:
  Stage 2 (Gate B).

## 12. Before the first run

1. Encode all splits with one encoder; build the A→B manifest with cut points.
2. Run the speaker leakage check (§2).
3. Compute M2 (bigram) on the held-out speech frames.
4. Re-encode hohere for Stage 2 if the RunPod volume was deleted.
5. Unit tests: A→B layout (no B frame in A), speech-frame masks, same-speaker
   batching, InfoNCE on a toy batch.
