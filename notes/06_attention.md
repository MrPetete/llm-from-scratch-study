# Chapter 2: Self-Attention (4 stages)

**Date:** 2026-01-20
**Files:** `simple_attention.py`, `trainable_attention.py`, `causal_attention.py`, `multihead_attention.py`

## What was built

The complete attention mechanism used inside every GPT transformer block, built in 4 incremental stages.

### Stage 1 — Simplified self-attention (no learned params)
- Attention score = dot product of query vector against every token's embedding (similarity measure)
- Softmax turns raw scores into weights summing to 1
- Context vector = weighted sum of all token embeddings (the "value" here is just the raw embedding)
- Vectorized the whole-sequence version (`inputs @ inputs.T`) and confirmed it matches the manual single-query loop exactly (`torch.allclose` sanity check passed)

### Stage 2 — Trainable Q/K/V attention
- Replaced raw embeddings with 3 learned projections: `W_query`, `W_key`, `W_value`
- `attn_scores = Q @ K.T`, scaled by `1/sqrt(d_k)` before softmax, then `context = attn_weights @ V`
- Implemented two versions: `SelfAttentionV1` (explicit `nn.Parameter` matrices, mirrors the math 1:1) and `SelfAttentionV2` (`nn.Linear` projections, the practical version used going forward)
- Demonstrated the scaling effect directly: unscaled softmax is measurably more peaked than scaled — with GPT-2's real `d_k=64` per head this effect is much sharper and is what scaled dot-product attention is specifically designed to prevent (vanishing gradients from a near one-hot softmax)

### Stage 3 — Causal (masked) attention
- Precomputed a boolean upper-triangular mask once via `register_buffer` (moves with `.to(device)`, not a trainable param)
- `attn_scores.masked_fill_(mask, -inf)` before softmax → those positions become exactly 0 after softmax
- Verified: each row of the weight matrix still sums to 1 (softmax renormalizes over only the *visible* tokens) — token 0 can only attend to itself, token 5 (last) can attend to all 6
- Extended to batched input `[batch, seq_len, d_in]` — this is the actual shape produced by the Step 4 DataLoader
- Demonstrated dropout on attention weights: PyTorch's `nn.Dropout` zeroes ~p of the weights and rescales survivors by `1/(1-p)` automatically; disabled at inference via `model.eval()`

### Stage 4 — Multi-head attention
- Split `d_out` into `num_heads x head_dim` and reshaped so heads become an extra batch-like dimension: `[batch, seq_len, d_out] → [batch, seq_len, num_heads, head_dim] → [batch, num_heads, seq_len, head_dim]`
- This lets one batched matmul compute all heads in parallel — no Python loop over heads
- Verified the two heads produce genuinely different attention weight distributions on the same input (`torch.allclose` check confirms they differ) — each head has its own Q/K slice
- Concatenated heads back (`transpose` + `.view`) and mixed through a final `out_proj` linear layer
- This module is the literal attention sub-layer that will sit inside each transformer block in Chapter 3

## Key bugs / gotchas hit

- None functionally broke, but had to be careful with `.contiguous()` before `.view()` after `.transpose()` in multi-head attention — `transpose` returns a non-contiguous view, and `.view()` requires contiguous memory. Using `.contiguous().view(...)` avoids a runtime error.
- `masked_fill_` (in-place, trailing underscore) mutates `attn_scores` directly — need to be careful this doesn't accidentally corrupt a tensor that's reused elsewhere (not an issue here since scores are recomputed each forward pass, but worth flagging for future code).

## Verification performed

- Stage 1: single-query manual computation matches vectorized full-matrix computation exactly
- Stage 2: both `SelfAttentionV1` and `SelfAttentionV2` produce correctly-shaped output; scaling demo shows measurably softer softmax distribution
- Stage 3: causal mask zeroes exactly the upper triangle; weight rows still sum to 1.0; dropout demo shows correct rescaling behavior
- Stage 4: output shape `[batch, seq_len, d_out]` correct; shape walkthrough printed at every reshape step; confirmed heads learn genuinely different attention patterns

## Next step

Chapter 3: GPT architecture — LayerNorm, GELU, feed-forward block, assembling the full transformer block (attention + feed-forward + residual connections + layer norm placement), then stacking blocks into the complete GPT model.
