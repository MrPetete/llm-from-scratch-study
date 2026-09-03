# Roadmap

Working solo, own implementation for every section. Reference material is read, then
closed, then implemented from understanding — compared against the reference afterward
to find gaps, not copied upfront.

## Phase 1 — Foundations (target: summer, before semester starts)

### Section 1 — Tokenization & data (`ch01_tokenizer/`)
- [x] Simple word-level tokenizer: split raw text, build vocab, encode/decode round-trip
- [x] Special tokens (`<|unk|>`, `<|endoftext|>`) — `SimpleTokenizerV2`
- [x] Byte Pair Encoding (BPE) via `tiktoken`, understand the merge algorithm conceptually
- [x] Sliding-window dataset + DataLoader producing (input, target) pairs for next-token prediction
- [x] Token embeddings (`nn.Embedding`) + positional embeddings — the actual input tensor for Ch3
- **Checkpoint:** encode → decode round-trip is lossless; DataLoader yields correctly shaped batches; embeddings ready for transformer ✅

### Section 2 — Attention (`ch02_attention/`)
- [x] Simplified self-attention (no trainable weights) — raw dot-product attention on toy vectors
- [x] Scaled dot-product self-attention with trainable Q/K/V weight matrices
- [x] Causal mask (prevent attending to future positions)
- [x] Multi-head attention (split into heads, concat, project)
- **Checkpoint:** print the attention weight matrix on a toy sequence, visually confirm the causal mask zeroes out future positions; verify output shape end to end ✅

### Section 3 — GPT architecture (`ch03_gpt_model/`)
- [x] LayerNorm, GELU, feed-forward block
- [x] Transformer block (attention + feed-forward + residuals + norm placement)
- [x] Full GPT model: token embeddings + positional embeddings + stacked blocks + output head
- [x] Text generation loop (forward pass → sample next token → append → repeat)
- **Checkpoint:** forward pass on random input returns `[batch, seq_len, vocab_size]`; untrained model can run the generation loop (output will be gibberish — confirming the mechanics work, not the quality) ✅

**Model size for Phase 1:** small by design (e.g. embedding dim 64–128, 2–4 layers, 2–4 heads)
— CPU-only hardware, the point here is correctness and understanding, not scale.

## Phase 2 — Pretraining & fine-tuning (target: late summer)

### Section 4 — Pretraining (`ch04_pretraining/`)
- [x] Cross-entropy loss — manual walkthrough verified against `F.cross_entropy`, untrained-model loss confirmed close to the ln(vocab_size) random baseline
- [x] Training loop (AdamW, backward/step, train/val loss tracking) — pretrained tiny GPT on "The Verdict" (20K chars), loss dropped 11.02 → 5.37 (train), overfitting visible in the val loss curve as expected
- [x] Decoding strategies (temperature scaling, top-k sampling) — fixed greedy decoding's repetition-loop failure mode
- [x] Save/load model weights (`state_dict` + optimizer state) — round-trip verified bit-for-bit identical
- [x] Load pretrained GPT-2 weights from OpenAI (via HuggingFace safetensors) — proves the architecture is a faithful GPT-2 reimplementation
- **Checkpoint:** model trained on a small corpus produces grammatically-plausible (if overfit/repetitive) text, clearly improved over Chapter 3's random gibberish ✅
- **Checkpoint 2:** loading OpenAI's real GPT-2-small weights into our own `GPTModel` class produces output text WORD-FOR-WORD IDENTICAL to the book's reference output ("Every effort moves you toward finding an ideal new way to practice something!...") — strongest possible confirmation the architecture is correct ✅

### Section 5 — Classification fine-tuning (`ch05_finetuning/`)
- [x] Dataset preparation: downloaded SMS Spam Collection (5,572 messages), balanced to 747+747, split 70/10/20, PyTorch dataloaders with padding to max_length=120
- [x] Model setup: loaded OpenAI's GPT-2-small weights, replaced output layer (768→50257 becomes 768→2), froze all except last transformer block + final_norm + new head (5.70% trainable params)
- [x] Training loop for classification: AdamW fine-tuning on spam detection task, tracking both loss and accuracy
- [x] Inference function (`classify_review`) for new messages
- **Checkpoint:** pipeline verified — model loads, trains, and improves accuracy from ~50% baseline (random guessing) to 57.5% after just 10 training batches; full 5-epoch training achieves ~97% train/val, ~96% test accuracy per the book ✅

- [ ] Instruction fine-tuning on a small instruction dataset
- **Deliverable:** documented experiment log — configs tried, loss curves, what broke and why

## Phase 3 — Applied differentiator: time-series adaptation (target: into semester)
- [ ] Adapt the architecture for numeric time-series input (patch-based tokenization, similar in spirit to PatchTST)
- [ ] Benchmark against a classical baseline (ARIMA) and an LSTM baseline on a real dataset
- [ ] Write up the comparison — this is the centerpiece for CV/interview discussion, not just "built a GPT"

## Notes convention

Each completed task gets a short entry in `notes/` — a few sentences: what was built,
one bug hit and how it was fixed. This becomes the raw material for a CV bullet and for
answering "walk me through what you built" in an interview.
