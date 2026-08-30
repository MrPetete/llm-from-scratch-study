# Chapter 4 (Step 7): Loading Pretrained GPT-2 Weights from OpenAI

**Date:** 2026-01-20
**File:** `load_openai_weights.py`

## What was built

The final piece of the Chapter 5 roadmap diagram (step 7, "Pretrained weights from OpenAI"). Everything before this trained OUR model on a tiny 20K-character dataset -- enough to prove the pipeline works, but nowhere near enough data for genuinely fluent text. This step loads OpenAI's actual GPT-2-small weights (trained on ~40GB of web text) into our own `GPTModel` class.

## Why this matters

This is the strongest possible test of whether the architecture we built (Chapters 1-3) is a genuinely correct, faithful reimplementation of GPT-2 -- not just "structurally similar," but tensor-for-tensor compatible. If our layer shapes, attention mechanism, or transformer block wiring had any bug, loading real trained weights would either fail with a shape mismatch or (worse) silently produce garbage, since a single misplaced tensor among ~150 would corrupt the whole forward pass.

## Weight source: HuggingFace instead of OpenAI's raw checkpoint

The book downloads OpenAI's original TensorFlow checkpoint via a custom `gpt_download.py` helper. That format is fragile to reproduce today (TF checkpoint readers, specific folder layout). Instead, pulled the equivalent weights from `openai-community/gpt2` on HuggingFace in `.safetensors` format -- same numbers, modern/safe format, no pickle or TensorFlow dependency. Installed `safetensors` + `huggingface_hub` (not the full `transformers` package, to keep the install light) via `uv pip install`, redirected `HF_HOME=D:/hf-cache` to keep the ~500MB download off the nearly-full C: drive.

## The hard part: key name and format mismatches

HuggingFace's GPT-2 tensors don't map 1:1 onto our attribute names:

1. **Naming convention**: OpenAI uses `wte`/`wpe`/`h.{i}.attn.c_attn`/`ln_1`/`mlp.c_fc`, ours uses `token_embedding`/`pos_embedding`/`transformer_blocks[i].attention...`/`norm1`/`feed_forward.layers[0]`
2. **Fused Q/K/V**: OpenAI stores query, key, and value projections as ONE tensor per block (`c_attn`, shape `[768, 2304]`), while our `MultiHeadAttention` (Chapter 2) uses three separate `nn.Linear` layers. Fixed with `tensor.chunk(3, dim=-1)`.
3. **Conv1D vs Linear convention**: OpenAI's original GPT-2 code used a `Conv1D` layer with weight shape `[in_features, out_features]` -- the OPPOSITE of PyTorch's `nn.Linear`, which stores `[out_features, in_features]`. Every attention and feed-forward weight needed a `.T` transpose before assignment.
4. **Weight tying**: `output_head.weight` is assigned the SAME tensor as `token_embedding.weight` (`wte.weight`), matching GPT-2's actual trained configuration (this is also why the "124M" parameter count is quoted with tying, vs our 163M as-coded without it in Chapter 3).

Used a small `assign()` helper (shape-checked, raises immediately on mismatch) for every one of the ~150 tensor assignments, so any wrong mapping would fail loudly rather than silently.

## Verification: byte-identical to the book's own output

With `qkv_bias=True` (OpenAI's GPT-2 used bias vectors, unlike modern LLMs), `context_length=1024`, seed 123, prompt `"Every effort moves you"`, temperature=1.5, top_k=50:

```
Generated: 'Every effort moves you toward finding an ideal new way to practice
something! What makes us want to be on top of that?'
```

This is **word-for-word identical** to the reference output quoted in the book itself. Same seed, same weights, same architecture, same sampling parameters, same result -- about the strongest verification signal possible short of a formal test suite.

## Key bugs / gotchas hit

- None on the first attempt -- the shape-checked `assign()` helper meant any mapping error would have been caught immediately rather than debugged after the fact.
- Disk discipline mattered here: C: was at 1.7GB free, so `HF_HOME` had to be explicitly redirected to `D:/hf-cache` before the download, not left at its default (which would land on C:).

## Verification performed

- All 160 tensors in the checkpoint accounted for and assigned without a shape mismatch
- Generated text is grammatically fluent (a qualitative leap from our own tiny-dataset-trained model's repetitive output)
- Output matches the book's own published reference text exactly, character for character

## Chapter 5 (per the book's own diagram) is now fully complete

All 7 steps from the roadmap figure: text generation, text evaluation, training/validation losses, LLM training function, text generation strategies, weight saving/loading, and pretrained weights from OpenAI.
