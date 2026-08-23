# Chapter 3: Full GPT Architecture

**Date:** 2026-01-20
**Files:** `config.py`, `layer_norm.py`, `feed_forward.py`, `transformer_block.py`, `gpt_model.py`, `generate.py`

## What was built

Assembled Chapter 2's attention mechanism plus several supporting pieces into a complete, working GPT model that can run the full text generation loop end to end.

### Config dictionary (`config.py`)
- Single dict controls model size: `vocab_size`, `context_length`, `embed_dim`, `num_heads`, `num_layers`, `dropout`, `qkv_bias`
- Same `GPTModel` class scales from a tiny 6.5M-param CPU-friendly config up to GPT-2 small (124M/163M) with zero code changes -- just different numbers
- Added `GPT_CONFIG_TINY` (embed_dim=64, 2 layers, context_length=128) for fast local iteration; `GPT_CONFIG_124M` matches the book's real GPT-2 small spec

### LayerNorm (`layer_norm.py`)
- Normalizes each token's feature vector independently to mean=0, variance=1 (verified numerically: mean ~0, var exactly 1.0 even with a deliberately skewed input containing an outlier feature)
- Two learned parameters, `scale` and `shift`, initialized to 1s/0s so LayerNorm is a no-op transform-wise at initialization, but can adapt during training
- Key distinction from BatchNorm: normalizes over the embedding dimension (last dim), per token, independent of other tokens in the batch/sequence -- important for variable-length sequence models

### GELU + FeedForward (`feed_forward.py`)
- GELU implemented via the tanh approximation (same formula GPT-2 uses)
- Verified numerically against ReLU: GELU passes small negative values through with a small negative output (e.g. -0.16 at x=-0.6) instead of ReLU's hard 0 cutoff -- confirms the "smoother" claim isn't just a description, it's numerically visible
- FeedForward: `embed_dim -> 4*embed_dim -> GELU -> embed_dim`, operates independently per token (no cross-token mixing, that's attention's job)

### Transformer block (`transformer_block.py`)
- Combines: `LayerNorm -> MultiHeadAttention -> +shortcut`, then `LayerNorm -> FeedForward -> +shortcut`
- This is GPT-2's "pre-norm" arrangement (LayerNorm before the sub-layer, not after)
- **Reused the actual Chapter 2 `MultiHeadAttention` class** via a `sys.path` insert rather than duplicating it -- keeps chapters self-contained folders while avoiding a copy-paste fork
- Built a concrete gradient-magnitude comparison: stacked 5 blocks WITH shortcuts vs WITHOUT, measured `|gradient|` at the very first input. With shortcuts: 0.0129. Without: 0.0019 -- roughly 7x larger gradient signal reaches the input layer with shortcuts, even at only 5 layers. This is the vanishing-gradient problem shortcuts solve, made concrete rather than just asserted.

### Full GPT model (`gpt_model.py`)
- `token_embedding + pos_embedding -> dropout -> N x TransformerBlock -> final LayerNorm -> output_head (Linear, no bias)`
- Output is raw **logits** `[batch, seq_len, vocab_size]`, not probabilities -- softmax happens later, during generation
- Verified against the book's own numbers on GPT_CONFIG_124M:
  - As-coded (separate output_head weights): **163,009,536** params -- matches book exactly
  - With weight tying (`output_head.weight = token_embedding.weight`): **124,412,160** params -- matches the commonly-cited "124M" figure exactly
  - Estimated float32 memory: **621.8 MB** -- matches the book's reported ~622 MB
- This confirms the implementation's numbers aren't just plausible, they reproduce the book's actual reference figures precisely

### Text generation loop (`generate.py`)
- Loop: forward pass → take only the LAST position's logits → softmax → argmax (greedy) → append token → repeat, truncating the input window to `context_length` each step
- Ran on the untrained tiny model: output was gibberish ("Hello, I am obscerva Kennybassitizenschery mids glimpse rate TDs") -- exactly as expected, since weights are random
- Verified every intermediate shape and invariant explicitly: probabilities sum to exactly 1.0, argmax always returns a valid token ID, sequence grows by exactly 1 token per iteration
- This is the chapter's correctness checkpoint: the MECHANICS are proven correct, not the output quality (training comes in Chapter 4/Phase 2)

## Key bugs / gotchas hit

- None broke during implementation. The main design decision was reusing Chapter 2's `MultiHeadAttention` via `sys.path.insert` rather than copy-pasting the class into `ch03_gpt_model/` -- keeps the codebase DRY and means any future bugfix to attention automatically applies here too.

## Verification performed

- LayerNorm: numerically confirmed mean=0, var=1 output on skewed input
- GELU: confirmed smooth negative-value passthrough vs ReLU's hard cutoff, at specific x values
- FeedForward: confirmed 4x expansion then compression, shape preserved
- TransformerBlock: confirmed output shape == input shape (stackability); confirmed shortcut connections produce measurably larger gradients at depth
- GPTModel: confirmed output shape `[batch, seq_len, vocab_size]`; confirmed parameter counts match the book's reported 163M/124M and ~622MB exactly
- generate.py: confirmed full generation loop runs correctly end to end on an untrained model; softmax sums to 1, argmax valid, sequence grows correctly

## Chapter 3 complete! 🎉

We now have a complete, structurally correct GPT model -- the exact architecture used by real GPT-2, just untrained. Every number checked against the book's reference figures matches exactly.

## Next step

Phase 2: Pretraining. Build the training loop (loss tracking, train/val split), pretrain the tiny model on a small corpus, and watch the generation loop actually start producing coherent text instead of gibberish.
