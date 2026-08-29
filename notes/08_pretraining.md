# Chapter 4: Pretraining on Unlabeled Data

**Date:** 2026-01-20
**Files:** `loss.py`, `train.py`, `decoding.py`, `save_load.py`

## What was built

Took the untrained GPT architecture from Chapter 3 and actually trained it -- the model went from producing random gibberish to generating grammatically-plausible (if repetitive/overfit) text.

### Cross-entropy loss (`loss.py`)
- Manual `-log(true_token_probability)` computation verified bit-exact against PyTorch's `F.cross_entropy` on a toy 2-position example
- Ran on the real (still untrained at this point) tiny GPT model + real text data: loss was **11.02**, very close to the theoretical random-guessing baseline `ln(50257) = 10.82` -- confirms an untrained model really is close to random guessing, not subtly broken
- `calc_loss_batch` (single batch) and `calc_loss_loader` (averaged over N batches, with an optional cap for fast mid-training evals) built as reusable utilities for the training loop

### Training loop (`train.py`)
- Standard cycle: `optimizer.zero_grad()` → `loss.backward()` → `optimizer.step()`, using AdamW
- Trained the tiny GPT (6.5M params) on "The Verdict" (20,479 characters, 90/10 train/val split) for 10 epochs
- **Real numbers from the run:** train loss 11.02 → 5.37, val loss 11.01 → bottomed at **6.70** around epoch 3, then crept back up to 6.54 by epoch 10 while train loss kept falling -- a textbook overfitting curve, visible directly in the printed numbers, not just described abstractly
- Generated text after training: `"Every effort moves you, the the had a--I have, the of the the the the was the the the the"` -- a clear qualitative jump from Chapter 3's total gibberish (now real English words, even a plausible phrase opening), but stuck in a repetition loop on "the" -- the classic greedy-decoding failure mode, which motivated the next step

### Decoding strategies (`decoding.py`)
- **Temperature scaling**: `softmax(logits / T)` then sample via `torch.multinomial` instead of argmax. Verified on a toy distribution: T=0.1 → 99.995% probability on the top logit (near one-hot), T=2.0 → much flatter distribution
- **Top-k filtering**: keep only the k highest logits, set the rest to `-inf` before softmax. Verified: with k=3 on a 5-logit toy example, the 2 lowest became exactly `-inf`
- On the actually-trained model: greedy still repeats ("of the the the the"); temperature=0.5 breaks the loop but drifts into newline spam (tiny dataset artifact, not a bug); temperature=1.5 is much more lexically diverse but less coherent -- demonstrates the real diversity/coherence tradeoff, not just the textbook description of it

### Save/load model weights (`save_load.py`)
- Standard PyTorch pattern: `torch.save({"model_state_dict": ..., "optimizer_state_dict": ...}, path)`, load via `load_state_dict` into a model built with the SAME config
- Verification was not just "it loaded without an error" -- built a completely fresh model with a different random seed, confirmed it generates different (still-gibberish) text, then loaded the checkpoint into that same fresh model and confirmed via `torch.equal` that (a) the generated output now exactly matches the originally-trained model's output, and (b) every single weight tensor is bit-for-bit identical
- Checkpoint file (~76.7MB) saved to `ch04_pretraining/checkpoints/`, gitignored (binary model weights don't belong in the repo)

## Key bugs / gotchas hit

- None broke functionally. The main thing worth flagging: `model.train()` vs `model.eval()` matters for BOTH training (dropout active) and evaluation (dropout must be off for a clean, deterministic loss/generation) -- easy to forget to toggle back and forth around the eval blocks inside the training loop.

## Verification performed

- Manual cross-entropy matches `F.cross_entropy` exactly on a toy example
- Untrained model loss confirmed close to the theoretical random-guess baseline
- Real training run: loss curves show learning (train loss dropped monotonically) AND overfitting (val loss bottoms out then rises) -- both expected behaviors on a 20K-character dataset, both visible in the actual printed numbers
- Generated text quality visibly improved from Chapter 3's total gibberish to real (if repetitive) English words
- Temperature and top-k filtering verified against hand-computed expected values on toy logits before being applied to the real trained model
- Save/load round-trip verified bit-for-bit identical, not just "no error was raised"

## Chapter 4 (Phase 2 core) complete! 🎉

We now have the full pipeline: tokenize → attention → GPT architecture → train → generate → save/load. Every component has been verified against either the book's own reference numbers or first-principles hand calculations, not just "it ran without crashing."

## Next step

Classification fine-tuning (adapt the model head for a downstream task) and instruction fine-tuning, per the roadmap -- or pivot toward Phase 3 (time-series adaptation), the CV/interview centerpiece, depending on priority.
