"""
Instruction fine-tuning training loop.

Loads GPT-2-medium, fine-tunes all parameters on instruction-response pairs.
"""

import os
import sys
import time
import torch
from functools import partial
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ch03_gpt_model"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ch04_pretraining"))

from config import GPT_CONFIG_355M
from gpt_model import GPTModel
from load_openai_weights import download_gpt2_state_dict, load_openai_weights_into_gpt
from decoding import generate_text_sampled
from dataset import (
    download_and_load_instruction_data,
    InstructionDataset,
    custom_collate_fn,
    format_input,
)


def calc_loss_batch(input_batch, target_batch, model, device):
    """Cross-entropy loss over all positions (next-token prediction)."""
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), target_batch.flatten()
    )
    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    """Average loss over a dataloader."""
    total_loss, count = 0.0, 0
    model.eval()
    with torch.no_grad():
        for i, (input_batch, target_batch) in enumerate(data_loader):
            if num_batches is not None and i >= num_batches:
                break
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
            count += 1
    return total_loss / count if count > 0 else 0.0


def train_model(model, train_loader, val_loader, optimizer, device, num_epochs,
                 eval_freq, eval_iter, start_context, tokenizer):
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, 0

    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()

            tokens_seen += input_batch.numel()
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
                val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Epoch {epoch+1} (step {global_step:04d}): "
                      f"train loss {train_loss:.3f}, val loss {val_loss:.3f}")
                model.train()

        print(f"\n--- Sample generation after epoch {epoch+1} ---")
        generate_and_print_sample(model, tokenizer, device, start_context)
        print()

    return train_losses, val_losses, track_tokens_seen


def generate_and_print_sample(model, tokenizer, device, formatted_prompt, context_length=1024):
    model.eval()
    input_ids = torch.tensor([tokenizer.encode(formatted_prompt)]).to(device)

    with torch.no_grad():
        token_ids = generate_text_sampled(
            model, input_ids, max_new_tokens=50, context_length=context_length,
            temperature=0.0,
        )

    generated_text = tokenizer.decode(token_ids[0].tolist())
    response_only = generated_text[len(formatted_prompt):].replace("### Response:", "").strip()
    print(response_only[:200])
    model.train()


if __name__ == "__main__":
    import tiktoken

    print("=== Chapter 7, Stage 2: Fine-tuning ===\n")

    device = "cpu"
    torch.manual_seed(123)

    url = ("https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/"
           "ch07/01_main-chapter-code/instruction-data.json")
    data = download_and_load_instruction_data(url, "instruction-data.json")

    train_portion = int(len(data) * 0.85)
    test_portion = int(len(data) * 0.1)
    train_data = data[:train_portion]
    test_data = data[train_portion:train_portion + test_portion]
    val_data = data[train_portion + test_portion:]

    print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}\n")

    tokenizer = tiktoken.get_encoding("gpt2")
    train_dataset = InstructionDataset(train_data, tokenizer)
    val_dataset = InstructionDataset(val_data, tokenizer)

    customized_collate_fn = partial(custom_collate_fn, device=device, allowed_max_length=1024)

    train_loader = DataLoader(
        train_dataset, batch_size=8, shuffle=True, drop_last=True,
        collate_fn=customized_collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=8, shuffle=False, drop_last=False,
        collate_fn=customized_collate_fn
    )

    print("Step 4: Loading pretrained GPT-2-medium (355M params) from OpenAI...")
    cfg = GPT_CONFIG_355M.copy()
    cfg["qkv_bias"] = True

    model = GPTModel(cfg)
    sd = download_gpt2_state_dict("openai-community/gpt2-medium")
    load_openai_weights_into_gpt(model, sd, num_layers=cfg["num_layers"])
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded. Total params: {total_params:,}\n")

    print("--- Before fine-tuning: model's response to an instruction ---")
    sample_prompt = format_input(val_data[0])
    print(f"Prompt: {val_data[0]['instruction']}")
    generate_and_print_sample(model, tokenizer, device, sample_prompt, cfg["context_length"])
    print("(Expect: echoing the prompt back or unrelated text -- no real instruction-following yet)\n")

    print("Step 5: Fine-tuning for 2 epochs...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)

    start_time = time.time()
    train_losses, val_losses, tokens_seen = train_model(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=2, eval_freq=5, eval_iter=5,
        start_context=sample_prompt, tokenizer=tokenizer,
    )
    end_time = time.time()

    print(f"\nTraining completed in {(end_time - start_time)/60:.2f} minutes.")

    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    save_path = os.path.join(checkpoint_dir, "gpt2_medium_instruction_finetuned.pt")
    torch.save(model.state_dict(), save_path)
    print(f"Saved fine-tuned model to {save_path}")

    print("\n=== Stage 2 complete ===")
    print(f"Fine-tuned GPT-2-medium on {len(train_data)} instruction-response pairs.")
    print(f"Final train loss: {train_losses[-1]:.3f}, final val loss: {val_losses[-1]:.3f}")
