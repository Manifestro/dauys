# Listening bench v1: Moshi

Run 2026-10-30/31 on one A40. Kept here for later comparison with Dauys v0.2; not a report.

- Bench code: branch `bench/duplex-listening`, directory `bench/`.
  - Run 1 at `fe8749b`: sWUGGY.
  - Run 2 at `8b0cffb`: contingency with the timed swap, sBLIMP and tStoryCloze on the fixed pairing, generation.
  - Run 1's sBLIMP (400 pairs, wrong grouping) and contingency are superseded by run 2.
- Model: `kyutai/moshiko-pytorch-bf16`, text stream = Moshi's own greedy text (two passes), seed 0.
- Data:
  - hohere-bench v0.2.0 E1, 267 conversations, first 1,500 frames for scoring and 750 frames (60 s) for generation;
  - sLM21 dev: sWUGGY 40,000 pairs, sBLIMP 25,200 pairs;
  - tStoryCloze: 1,871 stories.
- Everything, including the encoded test sets and Moshi's generated codes: HF `manifestro/dauys-runs` `bench/listening-v1/`.

## Contingency (teacher-forced NLL of Moshi's own channel on its speech frames)

| Partner channel | CB0 | CB1–7 mean |
| --- | ---: | ---: |
| true | 2.026 | 3.609 |
| true, spliced into its own speech frames | 2.041 | 3.654 |
| silence | 2.347 | 3.697 |
| another partner's speech in the true partner's speech frames (timed swap) | 2.378 | 3.676 |
| another conversation's partner channel (plain swap) | 2.823 | 3.888 |

Gaps, 95% bootstrap over conversations:

| Gap | CB0 | CB1–7 |
| --- | --- | --- |
| plain swap − true | +0.796 [0.772, 0.822] | +0.279 [0.263, 0.296] |
| silence − true | +0.320 [0.309, 0.332] | +0.089 [0.084, 0.093] |
| timed swap − true spliced (content and voice) | **+0.337 [0.324, 0.350]** | +0.022 [0.016, 0.027] |

- About 0.46 of the 0.80 plain-swap gap is turn timing; 0.34 is what the partner said and how they sounded.
- Wrong words at the right times cost Moshi as much as hearing nothing.
- The effect lives in the semantic stream; the acoustic codebooks barely depend on partner content.
- The timed swap changes both words and voice; separating them needs a same-voice swap.

## Generation (Moshi hears the partner channel and speaks freely, 60 s)

| Measure | Value |
| --- | --- |
| Moshi speech per conversation | 17.5 s [16.7, 18.2] |
| Echo: share of Moshi's CB0 4-grams heard from its partner | 0.43% [0.33, 0.55] |
| Echo with a swapped partner | 0.10% [0.06, 0.14] |
| Excess echo | **+0.34% [0.23, 0.46]**: real but tiny; Moshi does not repeat its partner |
| Voice cosine to its partner | 0.659 |
| Voice cosine to other partners | 0.646 |
| Voice copy (partner − others) | **+0.013 [−0.015, +0.040]**: no voice copying |
| Voice cosine to the dataset's assistant voice | 0.366 (Moshi keeps its own voice) |

## Semantic stream (item on Moshi's own channel, silence on the partner channel)

| Test | Pairs | Accuracy, summed CB0 log-likelihood | Mean-normalized |
| --- | ---: | --- | ---: |
| sWUGGY | 40,000 | **60.9%** [60.4, 61.3] | 60.5% |
| sBLIMP | 25,200 | **57.0%** [56.4, 57.6] | 52.2% |
| tStoryCloze | 1,871 | **65.4%** [63.3, 67.5] | 79.3% |

Chance is 50% for all three. Moshi normally reads its own text stream while it speaks; here that stream is its own greedy guess, so these numbers are Moshi's audio stream, not Moshi with reference text.
