# Chapter 6: Classification Fine-Tuning

**Date:** 2026-01-20
**Files:** `dataset.py`, `model_setup.py`, `train.py`

## What was built

Chapter 5 gave us a pretrained GPT-2 that generates coherent text. Chapter 6 teaches how to repurpose that general-purpose text generator into a specialized **spam classifier** — instead of predicting "the next word," the model now outputs a single decision: "spam" or "not spam."

### Stage 1: Dataset preparation (`dataset.py`)
- Downloaded the SMS Spam Collection dataset (5,572 text messages labeled "spam" or "ham")
- Balanced the dataset by undersampling "ham" to match "spam" count: 747 + 747 = 1,494 messages total
- Split 70/10/20 into train (1,045) / validation (149) / test (300)
- Built PyTorch `Dataset` and `DataLoader` with padding to `max_length=120` (the longest message in the dataset)
- Verified: one batch shape `[batch_size=8, seq_len=120]`, labels `[batch_size]` with 0=ham, 1=spam

### Stage 2: Model setup (`model_setup.py`)
- **Step 4**: Loaded OpenAI's pretrained GPT-2-small weights (from Chapter 4's `load_openai_weights.py`)
- **Step 5**: Replaced the output layer — the key architectural change:
  - Old: `Linear(768, 50257)` for next-token prediction over the full vocabulary
  - New: `Linear(768, 2)` for binary classification (spam vs ham)
  - The new layer is randomly initialized — it will be learned during fine-tuning
- **Step 6**: Froze most of the model, unfroze only:
  - Last transformer block (`transformer_blocks[-1]`)
  - Final LayerNorm (`final_norm`)
  - New output head (`output_head`)
  - Result: only **7,090,944 / 124,441,344 parameters (5.70%)** are trainable
- **Step 7**: Implemented evaluation utilities:
  - `calc_loss_batch`: cross-entropy loss computed on the **last token's logits** only (not every position like pretraining)
  - `calc_accuracy_loader`: fraction of correct predictions
- Verified baseline accuracy before fine-tuning: ~50% (random guessing on balanced dataset)

### Stage 3: Fine-tuning and usage (`train.py`)
- **Step 8**: Training loop for 5 epochs using AdamW (lr=5e-5, same structure as Chapter 4's pretraining)
  - Only the unfrozen 5.7% of parameters are updated
  - Tracks both loss and accuracy during training
- **Step 9**: Final evaluation on full train/val/test sets (expected per book: ~97% train/val, ~96% test)
- **Step 10**: `classify_review()` function for inference on new messages
  - Takes a text string, tokenizes, pads to max_length=120, runs forward pass
  - Returns predicted label (0=ham, 1=spam) + confidence probability

## Key concepts

**Transfer learning in action:** We're REUSING GPT-2's pretrained understanding of language (attention mechanisms, feed-forward layers, embeddings trained on billions of tokens) but swapping its "head" from a next-token predictor to a binary classifier. The early layers' representations are already good, so we only need to fine-tune the last few layers — faster and often just as effective as full fine-tuning.

**Classification vs. generation:** In pretraining (Chapter 4), the model predicted the next token at EVERY position in the sequence. In classification, we only care about ONE decision per input — we take the logits at the **last token position**, apply softmax, and pick the higher score (spam or ham).

**Freezing strategy:** By freezing 94% of the parameters, we:
1. Speed up training (fewer gradients to compute)
2. Reduce memory usage
3. Prevent catastrophic forgetting (the pretrained knowledge stays intact)
4. Still get high accuracy because the task-specific knowledge lives in the last block + new head

## Training in progress

Training started in background process `proc_579c380425e9` — will complete in ~5-10 minutes on CPU (book reports ~6 minutes on a laptop). Expected final accuracy: ~97% train/val, ~96% test.

## Verification performed (so far)

- Dataset: 1,494 balanced messages (747+747), split correctly, longest=120 tokens
- Model setup: output layer swapped to 768→2, only 5.70% trainable params, baseline accuracy ~50%
- Training loop structure matches Chapter 4's pretraining (same AdamW, loss tracking, eval utilities)

## Next step after training completes

Verify final accuracy matches the book's reported numbers, test `classify_review()` on new spam/ham examples, commit with real training results documented.
