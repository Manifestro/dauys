# Dauys Hybrid v0.1

This is the Stage 1 candidate architecture. The dialogue model is trained from random weights; the pretrained Mimi codec remains frozen.

## Timeline and codec

Each channel is mono audio at 24 kHz. Mimi produces eight codebooks of 2,048 values at 12.5 frames per second, or one frame per 80 ms. Channel A is always the user and channel B is always the model-side speaker. Both channels remain synchronized through silence and overlapping speech.

Each codebook has a separate trainable 2,048 by 256 embedding table, shared between the two channels. For each frame, the eight selected embeddings are summed and divided by the square root of eight. A trainable role embedding distinguishes the channels. Separate projections map this frame representation to the listening and dialogue widths.

The eight codebooks occupy one backbone time step per audio frame. They are not expanded into eight sequential backbone steps.

Attention uses RoPE with the same absolute frame positions in both streams. Self-attention and cross-attention retain at most 512 frames of context. Mamba-2 state persists throughout a conversation and resets only when a new conversation begins.

## Listening tower

The listening tower applies a 256-to-384 projection followed by four causal Mamba-2 layers at width 384. Each layer uses `d_state=64`, `d_conv=4`, `expand=2`, `headdim=64`, and `ngroups=1`.

## Dialogue tower

The dialogue tower applies a 256-to-640 projection followed by ten blocks at width 640:

| Block | Type | Listening fusion |
| --- | --- | --- |
| 1 | Mamba-2 | |
| 2 | Mamba-2 | |
| 3 | Transformer | Causal cross-attention |
| 4 | Mamba-2 | |
| 5 | Mamba-2 | |
| 6 | Transformer | Causal cross-attention |
| 7 | Mamba-2 | |
| 8 | Transformer | Causal cross-attention |
| 9 | Mamba-2 | |
| 10 | Transformer | Causal cross-attention |

The Mamba-2 layers use the same state, convolution, expansion, head, and group settings as the listening tower. Transformer layers use causal self-attention with 10 heads, pre-RMSNorm, residual connections, and a SwiGLU feed-forward width of 1,728. Each cross-attention module uses dialogue states as queries and listening states, projected from width 384 to 640, as keys and values. Cross-attention never sees future listening frames.

At frame `t`, the listening tower may consume user frame `A_t`. The dialogue tower consumes the previous model-side frame `B_(t-1)` and predicts `B_t`. A configured input delay of one frame restricts listening access to `A_(<=t-1)`. The Stage 1 configuration uses zero extra input-delay frames.

## Token prediction

The dialogue state predicts codebook 0 through a 640-to-2,048 output head. A two-layer causal depth Transformer at width 384 with six heads predicts codebooks 1 through 7 in order within the same frame, conditioned on the dialogue state and earlier codebooks. It uses learned codebook-position embeddings and a separate output head for each target codebook. It does not consume future audio frames. Training uses teacher forcing for the depth sequence.

A separate 640-to-2 speech activity head predicts silence or speaking. Silence remains represented by valid Mimi codes rather than an arbitrary token index. The output scheduler retains every frame in the timeline, including silent model-side frames.

## Training objective

The initial loss weights are 1.0 for codebook 0 cross-entropy, 0.5 for the mean cross-entropy over codebooks 1 through 7, and 0.2 for binary speech activity. Valid-frame masks apply to token losses.

The configuration caps all trainable embeddings, towers, fusion modules, and heads at 65 million parameters. The cap must be checked by direct parameter counting before training.

## Required pre-training checks

1. Changing future input audio cannot alter earlier predictions.
2. Incremental and batched forward passes produce matching logits within numerical tolerance.
3. Offline and streaming Mimi tokenization agree, and the resulting codes decode successfully.
4. Incoming frames are processed while output frames are generated without resetting state.
5. Direct counting confirms the trainable parameter cap.
