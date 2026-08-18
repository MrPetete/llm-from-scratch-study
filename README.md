# LLM From Scratch — Study & Build Project

A hands-on implementation of a GPT-style large language model, built from the ground up
to understand what actually happens inside a transformer — tokenization, attention,
architecture, pretraining, and fine-tuning — rather than treating it as a black box.

This project follows the structure of Sebastian Raschka's *Build a Large Language Model
(From Scratch)* (Manning, 2024) as a conceptual reference. **No code in this repo is
copied from the book or its companion repo** ([rasbt/LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch)).
The workflow for every section is: read/watch the reference material, close it, implement
my own version, then compare against the reference to find and fix gaps in understanding.

## Goal

1. Understand transformer internals deeply enough to explain and defend design choices
   in an interview or research context — not just "I followed a tutorial."
2. Build something extendable: once the core GPT works, adapt it to a second domain
   (planned: time-series forecasting, patch-based tokenization à la PatchTST) as the
   differentiator beyond the base exercise.
3. Produce a clean, documented, incrementally-committed repo usable as a CV/portfolio piece.

## Status

🚧 Phase 1 in progress — see [ROADMAP.md](./ROADMAP.md) for the full plan and task breakdown.

## Project structure

```
llm-from-scratch-study/
├── README.md
├── ROADMAP.md            # phase-by-phase plan and checkpoints
├── requirements.txt
├── .venv/                # local virtual env (not committed)
├── ch01_tokenizer/        # simple + BPE tokenization, dataset/dataloader
├── ch02_attention/         # self-attention → causal → multi-head
├── ch03_gpt_model/         # transformer block, full GPT architecture
├── ch04_pretraining/       # training loop, loss tracking, text generation
├── ch05_finetuning/        # classification + instruction fine-tuning
└── notes/                # short markdown writeups per section (what I built, bugs hit/fixed)
```

## Environment

- Python 3.11, managed with `uv`
- PyTorch (CPU build — no dedicated GPU on this machine; models kept small enough to
  train on CPU locally, larger runs done on free-tier Colab/Kaggle GPU when needed)
- `tiktoken` for GPT-2 BPE tokenizer reference/comparison
- Setup:
  ```bash
  uv venv --python 3.11
  uv pip install -r requirements.txt
  ```

## Reference material

- Sebastian Raschka, *Build a Large Language Model (From Scratch)*, Manning, 2024
- [rasbt/LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch) — official companion code (read-only reference, not copied from)
