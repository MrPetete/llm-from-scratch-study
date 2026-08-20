# Step 2: Special Tokens (`SimpleTokenizerV2`)

## What we built
Added two special tokens to handle real-world cases:
- `<|unk|>` (ID 0): replacement for unknown words not in vocabulary
- `<|endoftext|>` (ID 1): document boundary marker for training on multiple texts

## Key implementation details

### Vocab construction
Special tokens are prepended to the vocabulary so they get stable, low IDs:
```python
all_tokens = ["<|unk|>", "<|endoftext|>"] + sorted(regular_tokens)
```

### Unknown word handling
In `encode()`, when a word isn't in the vocab, emit `<|unk|>` ID instead of crashing:
```python
ids = [self.str_to_int.get(token, self.str_to_int["<|unk|>"]) for token in preprocessed]
```

### Special token preservation
The `allowed_special` parameter controls whether special tokens in the input text are:
- Treated as special tokens (single token with known ID)
- Or split like regular text (punctuation breakdown → unknown tokens)

**Implementation challenge:** special tokens like `<|endoftext|>` contain punctuation that the regex would normally split. Solution: replace them with space-padded placeholders before splitting, then restore them after:

```python
# Before split: "<|endoftext|>" → " SPECIALTOKEN0 "
# After split: "SPECIALTOKEN0" → "<|endoftext|>"
```

The space padding ensures the placeholder is always isolated during the regex split, even when the special token sits directly between two words with no surrounding whitespace.

## Test results (on "The Verdict" vocab, 1,132 tokens)

1. **Known words:** Perfect round-trip, same as V1
2. **Unknown word ("Hello"):** Mapped to `<|unk|>` (ID 0) instead of crashing ✓
3. **Special token handling:**
   - Without `allowed_special`: `<|endoftext|>` split into punctuation → all unknown
   - With `allowed_special={"<|endoftext|>"}`: preserved as single token (ID 1) ✓

## Why this matters for LLM training
- `<|unk|>` lets the model train on any text, even with words outside the original vocab
- `<|endoftext|>` teaches the model document boundaries, preventing it from learning spurious connections between the end of one document and the start of the next

## Next: Byte Pair Encoding (BPE)
Word-level tokenization wastes vocab space (each form of a word needs its own entry) and can't handle misspellings or rare words well. BPE solves this by building subword units, which is what GPT-2 and modern LLMs actually use.
