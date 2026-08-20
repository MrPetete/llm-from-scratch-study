# Step 4: Sliding-Window DataLoader for Next-Token Prediction

**Date:** 2026-01-20  
**File:** `ch01_tokenizer/dataloader.py`

## What was built

Implemented a PyTorch `Dataset` and `DataLoader` that creates (input, target) pairs for GPT training using a sliding window over tokenized text. This is the data pipeline that feeds the model during training.

### Key concepts learned

1. **Next-token prediction:** GPT is trained to predict the next token given previous tokens.
   - Input: `["I", "HAD", "always", "thought"]` (tokens 0-3)
   - Target: `["HAD", "always", "thought", "Jack"]` (tokens 1-4)
   - The target is the input shifted by 1 position

2. **Sliding window with stride:**
   - `context_length=8`: each training example has 8 tokens
   - `stride=4`: slide the window by 4 positions (50% overlap)
   - Overlapping windows = more training examples from the same text
   - Example: 5,145 tokens → 1,284 windows with stride=4

3. **Why overlap?** 
   - Stride=1 (max overlap): every possible sequence becomes a training example
   - Stride=context_length (no overlap): fastest but wastes data
   - Stride=context_length/2 (50% overlap): good balance for efficiency

4. **Batching:**
   - DataLoader groups multiple windows into batches
   - Shape: `[batch_size, context_length]`
   - Example: batch_size=2, context_length=8 → shape `[2, 8]`
   - GPU processes entire batch in parallel

### Test results

Using "The Verdict" text (20,479 characters, 5,145 tokens):
- Configuration: context_length=8, stride=4, batch_size=2
- ✅ 1,284 windows extracted
- ✅ 642 batches created
- ✅ Target correctly shifted by 1 position
- ✅ Decode verification: input/target text matches expected sequences

**Sample window:**
```
Input:  "I HAD always thought Jack Gis"
Target: " HAD always thought Jack Gisburn"
```

Notice the target is shifted: the model sees "I HAD always thought Jack Gis" and must predict "born" (next token).

### Implementation details

**`GPTDatasetV1` class:**
- Tokenizes text once in `__init__` (efficient)
- `__len__`: number of complete windows = `(total_tokens - context_length) // stride`
- `__getitem__`: extracts window at position `idx * stride`

**`create_dataloader_v1` function:**
- Wraps dataset creation + DataLoader setup
- Default settings: shuffle=True, drop_last=True (recommended for training)
- Returns batches of shape `[batch_size, context_length]`

### Why this matters for training

During training, the model will:
1. Receive an input batch `[batch_size, context_length]` of token IDs
2. Convert to embeddings (next step)
3. Process through transformer blocks
4. Output predictions `[batch_size, context_length, vocab_size]`
5. Compare predictions against target batch using cross-entropy loss
6. Backpropagate and update weights

Every position in the sequence learns to predict the next token independently — this is why GPT can generate text one token at a time after training.

### Next step

Step 5: Token embeddings (`nn.Embedding`) + positional embeddings — convert token IDs to dense vectors that the transformer can process.
