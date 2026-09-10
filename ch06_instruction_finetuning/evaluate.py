"""
Chapter 7, Stage 3: Evaluation (Steps 6-8)

Step 6: Inspect the loss curves (train vs val, printed during train.py)
Step 7: Generate responses on held-out TEST examples and manually inspect
        quality -- this is qualitative, not a single accuracy number
Step 8: Discuss how instruction-tuned models are properly evaluated

WHY "ACCURACY" DOESN'T WORK HERE
    Chapter 6's spam classifier had exactly one correct answer per example
    (spam or ham) -- accuracy was well-defined. Instruction-following has
    NO single correct answer: "Convert 45 kilometers to meters" could be
    answered as "45000 meters", "45km = 45000m", "That's 45,000 meters.",
    etc. -- all correct, none identical. Comparing token-for-token to one
    reference answer (like the classification accuracy metric) would
    unfairly penalize equally-valid phrasings.

HOW REAL INSTRUCTION-TUNED MODELS ARE EVALUATED (per the book)
    1. Human evaluation -- the gold standard, but slow/expensive/subjective
    2. Benchmark datasets (MMLU, HellaSwag, etc.) -- test specific capabilities
       with multiple-choice-style questions that DO have single correct answers
    3. LLM-as-judge -- use a stronger model (e.g. GPT-4/Llama-3-70B via Ollama)
       to SCORE the fine-tuned model's responses against a reference answer on
       a 1-100 scale, prompted to consider correctness + relevance, not exact
       wording match. This scales far better than human eval and is what the
       book demonstrates in this chapter (via a locally-run Llama 3 through
       Ollama, scoring our model's test-set responses).

This script implements (1) manual inspection on test examples, matching what
we can run locally without needing a second large model as judge.
"""

import os
import sys
import json
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ch03_gpt_model"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ch04_pretraining"))

from config import GPT_CONFIG_355M
from gpt_model import GPTModel
from decoding import generate_text_sampled
from dataset import download_and_load_instruction_data, format_input


def load_finetuned_model(checkpoint_path, device="cpu"):
    """Load the fine-tuned GPT-2-medium instruction model from a saved checkpoint."""
    cfg = GPT_CONFIG_355M.copy()
    cfg["qkv_bias"] = True

    model = GPTModel(cfg)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    return model, cfg


def generate_response(model, tokenizer, formatted_prompt, device, context_length,
                       max_new_tokens=100):
    """Generate a free-form text response to an instruction prompt."""
    input_ids = torch.tensor([tokenizer.encode(formatted_prompt)]).to(device)

    with torch.no_grad():
        token_ids = generate_text_sampled(
            model, input_ids, max_new_tokens=max_new_tokens,
            context_length=context_length, temperature=0.0,  # greedy for evaluation
        )

    generated_text = tokenizer.decode(token_ids[0].tolist())
    response_only = generated_text[len(formatted_prompt):].strip()

    # Stop at the next "### Instruction" if the model runs on (shouldn't with
    # a well-trained model, but a safety net for an undertrained one)
    if "###" in response_only:
        response_only = response_only.split("###")[0].strip()

    return response_only


if __name__ == "__main__":
    import tiktoken

    print("=== Chapter 7, Stage 3: Evaluation ===\n")

    device = "cpu"
    tokenizer = tiktoken.get_encoding("gpt2")

    # --- Load test split (held-out, never seen during training) ---
    url = ("https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/"
           "ch07/01_main-chapter-code/instruction-data.json")
    data = download_and_load_instruction_data(url, "instruction-data.json")

    train_portion = int(len(data) * 0.85)
    test_portion = int(len(data) * 0.1)
    test_data = data[train_portion:train_portion + test_portion]

    print(f"Test set: {len(test_data)} held-out examples\n")

    # --- Load the fine-tuned model ---
    checkpoint_path = os.path.join("checkpoints", "gpt2_medium_instruction_finetuned.pt")
    print(f"Loading fine-tuned model from {checkpoint_path}...")
    model, cfg = load_finetuned_model(checkpoint_path, device)
    print("Loaded.\n")

    # --- Step 7: Generate responses on test examples, manually inspect ---
    print("Step 7: Generating responses on held-out test examples\n")
    print("=" * 70)

    num_examples_to_show = 5
    for i, entry in enumerate(test_data[:num_examples_to_show]):
        prompt = format_input(entry)
        response = generate_response(model, tokenizer, prompt, device, cfg["context_length"])

        print(f"[{i+1}] Instruction: {entry['instruction']}")
        if entry.get("input"):
            print(f"    Input: {entry['input']}")
        print(f"    Reference answer: {entry['output']}")
        print(f"    Model response:   {response}")
        print("-" * 70)

    print("\nStep 8: How instruction-tuned models are properly evaluated")
    print("=" * 70)
    print("""
Unlike Chapter 6's classification task, there is NO single correct answer for
free-form text generation -- "45 kilometers is 45000 meters" and "That's
45,000 m" are both valid responses to the same instruction, but a token-match
metric would score them as completely different. Because of this, instruction
models are evaluated with methods that judge MEANING and RELEVANCE, not exact
string match:

  1. Human evaluation      -- gold standard, but slow and expensive
  2. Benchmark datasets     -- MMLU, HellaSwag etc: multiple-choice-style
                                questions that DO have a single correct answer,
                                used to measure specific capabilities at scale
  3. LLM-as-judge           -- prompt a STRONGER model (e.g. GPT-4, or a local
                                Llama 3 via Ollama) to score the fine-tuned
                                model's response against the reference answer
                                on a numeric scale, considering correctness and
                                relevance rather than exact wording

The book demonstrates option 3 using a local Ollama + Llama 3 setup to score
our model's test-set responses on a 1-100 scale. That requires an additional
~4-8GB LLM running locally via Ollama, which is out of scope for this specific
CPU-only fine-tuning run -- the manual inspection above (option-1-lite, self-
judged) serves the same illustrative purpose: confirming the model produces
sensible, on-topic responses rather than echoing the prompt or generating
gibberish.
""")
