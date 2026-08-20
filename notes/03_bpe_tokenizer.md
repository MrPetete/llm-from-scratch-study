# Step 3: Byte Pair Encoding (BPE) Tokenizer

**Date:** 2026-01-20  
**File:** `ch01_tokenizer/bpe_tokenizer.py`

## What was built

Implemented BPE tokenization using OpenAI's `tiktoken` library (the same tokenizer used in GPT-2/GPT-3/ChatGPT). BPE solves the fundamental problem with word-level tokenizers: unknown words.

### Key concepts learned

1. **BPE breaks words into subwords** — instead of `<|unk|>` for "someunknownPlace", BPE splits it into `["some", "unknown", "Place"]` (3 tokens). Any word can be represented, even if it wasn't in training data.

2. **How BPE works (conceptual):**
   - Start with individual characters as the base vocab
   - Iteratively merge the most frequent character pairs into subwords
   - Example: "d" + "e" → "de" (common in "define", "depend", "made")
   - Final vocab size: 50,257 tokens for GPT-2

3. **Special token handling:**
   - `<|endoftext|>` (ID 50256) is the largest token ID
   - Used to separate unrelated documents during training
   - `tiktoken` requires explicit `allowed_special={"<|endoftext|>"}` to encode it as a single token
   - Without that flag, it treats it as regular text and splits it into 10+ tokens

4. **Multilingual support:** BPE works on UTF-8 bytes, so it handles any Unicode text (Chinese, emoji, code) without special handling.

### Tests run

- ✅ Basic encode/decode round-trip
- ✅ Unknown word breaks into subwords (`someunknownPlace` → 3 tokens)
- ✅ Special token `<|endoftext|>` encoding (with/without `allowed_special`)
- ✅ Vocab size comparison (50,257 vs. 1,132 for SimpleTokenizerV2)
- ✅ Multilingual text (Chinese + English)

### Bug encountered

Initially tried to encode `<|endoftext|>` without `allowed_special` — `tiktoken` threw `ValueError` by design. This is a safety feature to prevent accidentally treating special tokens as regular text. Fixed by:
- Using `disallowed_special=()` to treat it as text
- Using `allowed_special={"<|endoftext|>"}` to treat it as a special token

### Why we used `tiktoken` instead of implementing BPE from scratch

The book itself uses `tiktoken` because "implementing BPE can be relatively complicated." Since the goal is to understand GPT architecture (attention, transformers), not reinvent tokenization algorithms, using the production tokenizer is the right call. We understand BPE conceptually (merge frequent pairs), which is enough for this project.

### Next step

Step 4: Build a PyTorch `Dataset` and `DataLoader` that uses BPE tokenization to create (input, target) pairs for next-token prediction training.
