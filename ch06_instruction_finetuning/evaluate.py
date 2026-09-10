"""Generate responses on held-out test examples for manual quality inspection of the instruction-tuned model."""

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
    full_prompt = formatted_prompt + "\n\n### Response:\n"
    input_ids = torch.tensor([tokenizer.encode(full_prompt)]).to(device)

    with torch.no_grad():
        token_ids = generate_text_sampled(
            model, input_ids, max_new_tokens=max_new_tokens,
            context_length=context_length, temperature=0.0,  # greedy for evaluation
        )

    generated_text = tokenizer.decode(token_ids[0].tolist())
    response_only = generated_text[len(full_prompt):].strip()

    # Trim off anything past the response (model may keep generating past it)
    if "###" in response_only:
        response_only = response_only.split("###")[0].strip()
    if "<|endoftext|>" in response_only:
        response_only = response_only.split("<|endoftext|>")[0].strip()

    return response_only


if __name__ == "__main__":
    import tiktoken

    print("=== Chapter 7, Stage 3: Evaluation ===\n")

    device = "cpu"
    tokenizer = tiktoken.get_encoding("gpt2")

    url = ("https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/"
           "ch07/01_main-chapter-code/instruction-data.json")
    data = download_and_load_instruction_data(url, "instruction-data.json")

    train_portion = int(len(data) * 0.85)
    test_portion = int(len(data) * 0.1)
    test_data = data[train_portion:train_portion + test_portion]

    print(f"Test set: {len(test_data)} held-out examples\n")

    checkpoint_path = os.path.join("checkpoints", "gpt2_medium_instruction_finetuned.pt")
    print(f"Loading fine-tuned model from {checkpoint_path}...")
    model, cfg = load_finetuned_model(checkpoint_path, device)
    print("Loaded.\n")

    print("Generating responses on held-out test examples\n")
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

    print("\nHow instruction-tuned models are properly evaluated")
    print("=" * 70)
    print("""
Unlike classification, free-form generation has no single correct answer --
"45 kilometers is 45000 meters" and "That's 45,000 m" are both valid, but a
token-match metric would score them as completely different. Real evaluation
options: human review (gold standard but slow), benchmark datasets like MMLU
(multiple-choice, so scorable), or LLM-as-judge (a stronger model scores
responses for correctness/relevance on a numeric scale -- e.g. Llama 3 via
Ollama, as the book demonstrates). This script does manual inspection instead,
since running a second large model as judge is out of scope here -- it's
enough to confirm the model gives sensible, on-topic answers.
""")
