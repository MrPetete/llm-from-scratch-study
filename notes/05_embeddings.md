# Step 5: Token & Positional Embeddings

**Date:** 2026-01-20  
**File:** `ch01_tokenizer/embeddings.py`

## What was built

Implemented the embedding layer that converts token IDs (integers) into dense vectors with positional information. This is the first layer of the GPT model — the input that feeds into transformer blocks.

### Key concepts learned

1. **Token embeddings (`nn.Embedding`):**
   - Converts token IDs → dense vectors
   - Shape: `[batch_size, seq_len]` → `[batch_size, seq_len, embed_dim]`
   - Learned during training (not hand-crafted)
   - Example: token ID 42 → 256-dimensional vector like `[-2.47, -1.00, -1.65, ...]`
   - Semantically similar tokens (e.g., "cat"/"dog") learn similar vectors

2. **Positional embeddings:**
   - Adds position information to each token
   - **Why needed?** Self-attention is permutation-invariant — without position info, "dog bites man" = "man bites dog"
   - GPT uses **learned** positional embeddings (not sinusoidal like original Transformer)
   - Shape: `[context_length, embed_dim]` — one vector per position
   - Same token at different positions → different final vectors

3. **Combining embeddings:**
   - Final embedding = `token_embedding + positional_embedding`
   - Element-wise addition (broadcasting handles batch dimension)
   - Formula: `embed[b, t, d] = token_emb[vocab[t], d] + pos_emb[t, d]`

4. **Dropout for regularization:**
   - Applied after combining embeddings
   - Randomly zeros out some dimensions during training (prevents overfitting)
   - Typical value: 0.1 (10% of dimensions dropped)

### Test results

**Test 1: Token embeddings**
- Input: `[2, 8]` (2 sequences, 8 tokens each)
- Output: `[2, 8, 256]` (each token → 256D vector)
- ✅ Shape correct

**Test 2: Positional embeddings**
- Input: `[2, 8, 256]` (token embeddings)
- Output: `[2, 8, 256]` (with position info added)
- ✅ Broadcasting works correctly

**Test 3: Complete GPT embedding**
- Input: `[2, 8]` (token IDs)
- Output: `[2, 8, 256]` (dense vectors ready for transformer)
- ✅ Single forward pass combines both embeddings

**Test 4: Position matters**
- Same token ID 42 at position 0 vs position 1
- Position 0 first 5 dims: `[-2.47, -1.00, -1.65, 1.21, -1.45]`
- Position 1 first 5 dims: `[-0.59, 1.89, -0.30, 2.97, 0.90]`
- ✅ Vectors differ — position information successfully added

**Test 5: Real text**
- Text: "The quick brown fox jumps"
- Token IDs: `[464, 2068, 7586, 21831, 18045]`
- Output: `[1, 5, 256]` (5 tokens → 5 vectors with position info)
- ✅ Ready to feed into transformer blocks

### Implementation details

**Three classes:**
1. `TokenEmbedding` — wraps `nn.Embedding(vocab_size, embed_dim)`
2. `PositionalEmbedding` — wraps `nn.Embedding(context_length, embed_dim)` + position indexing
3. `GPTEmbedding` — combines both + dropout (production-ready)

**GPT-2 parameters:**
- `vocab_size = 50257` (BPE vocabulary)
- `embed_dim = 768` (GPT-2 base), scaled to 256 for demo
- `context_length = 1024` (max sequence length)
- `dropout = 0.1` (regularization)

### Why learned positional embeddings?

The original Transformer paper (Vaswani et al. 2017) used **sinusoidal** positional encodings (fixed, not learned). GPT uses **learned** embeddings because:
- More flexible — can learn task-specific position patterns
- Empirically performs as well or better
- Simpler to implement (just another `nn.Embedding`)

Trade-off: learned embeddings can't extrapolate to longer sequences than `context_length` during training (but GPT doesn't need to since context is fixed at 1024/2048).

### Data flow summary

```
Token IDs [batch, seq_len]
    ↓
Token Embedding [batch, seq_len, embed_dim]
    ↓
+ Positional Embedding [seq_len, embed_dim]  (broadcast across batch)
    ↓
Dropout (regularization)
    ↓
Final Embeddings [batch, seq_len, embed_dim]
    ↓
Feed into Transformer Blocks (next chapter)
```

### Chapter 1 complete! 🎉

We've built the complete data pipeline:
1. ✅ Tokenization (BPE via tiktoken)
2. ✅ DataLoader (sliding window, batching)
3. ✅ Embeddings (token + positional)

**Next chapter:** Attention mechanisms — the core of transformers.
