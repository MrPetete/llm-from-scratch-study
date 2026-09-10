"""Training loop: forward pass, backward pass, optimizer step, periodic eval."""

import os
import sys
import math
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ch03_gpt_model"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ch01_tokenizer"))

from config import GPT_CONFIG_TINY
from gpt_model import GPTModel
from dataloader import create_dataloader_v1
from loss import calc_loss_batch, calc_loss_loader


def train_model(model, train_loader, val_loader, optimizer, device,
                 num_epochs, eval_freq, eval_iter):
    """
    Args:
        model: GPTModel instance
        train_loader, val_loader: DataLoaders yielding (input, target) batches
        optimizer: e.g. torch.optim.AdamW
        device: "cpu" or "cuda"
        num_epochs: how many full passes over train_loader
        eval_freq: evaluate train/val loss every N training steps
        eval_iter: how many batches to average over when evaluating

    Returns:
        train_losses, val_losses, track_tokens_seen: lists for plotting/inspection
    """
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
                model.eval()
                train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
                val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
                model.train()

                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)

                print(f"Epoch {epoch+1} (step {global_step:04d}): "
                      f"train loss {train_loss:.3f}, val loss {val_loss:.3f}")

    return train_losses, val_losses, track_tokens_seen


if __name__ == "__main__":
    torch.manual_seed(123)

    print("=== Training Loop: Pretraining the Tiny GPT Model ===\n")

    device = "cpu"   # no CUDA available; tiny model keeps CPU training fast
    print(f"Device: {device}\n")

    data_path = os.path.join(os.path.dirname(__file__), "..", "ch01_tokenizer", "data", "the-verdict.txt")
    with open(data_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    split_idx = int(len(raw_text) * 0.9)
    train_text = raw_text[:split_idx]
    val_text = raw_text[split_idx:]
    print(f"Total characters: {len(raw_text)}  (train: {len(train_text)}, val: {len(val_text)})")

    context_length = GPT_CONFIG_TINY["context_length"] // 4
    train_loader = create_dataloader_v1(
        train_text, batch_size=2, context_length=context_length,
        stride=context_length, shuffle=True, drop_last=True
    )
    val_loader = create_dataloader_v1(
        val_text, batch_size=2, context_length=context_length,
        stride=context_length, shuffle=False, drop_last=True
    )
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}\n")

    model = GPTModel(GPT_CONFIG_TINY).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.1)

    model.eval()
    initial_train_loss = calc_loss_loader(train_loader, model, device)
    initial_val_loss = calc_loss_loader(val_loader, model, device)
    print(f"Before training -- train loss: {initial_train_loss:.4f}, val loss: {initial_val_loss:.4f}")
    print(f"(Random-guess baseline: ln({GPT_CONFIG_TINY['vocab_size']}) = {math.log(GPT_CONFIG_TINY['vocab_size']):.4f})\n")

    print("--- Training for 10 epochs ---")
    train_losses, val_losses, tokens_seen = train_model(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=10, eval_freq=5, eval_iter=5
    )

    print(f"\nFinal train loss: {train_losses[-1]:.4f}")
    print(f"Final val loss:   {val_losses[-1]:.4f}")
    print(f"Loss dropped from {initial_train_loss:.4f} -> {train_losses[-1]:.4f} (train)")

    print("\n--- Generated text after training ---")
    import tiktoken
    from generate import generate_text

    tokenizer = tiktoken.get_encoding("gpt2")
    start_text = "Every effort moves you"
    start_ids = torch.tensor([tokenizer.encode(start_text)])

    generated_ids = generate_text(model, start_ids, max_new_tokens=20, context_length=context_length)
    generated_text = tokenizer.decode(generated_ids[0].tolist())
    print(f"Prompt: {repr(start_text)}")
    print(f"Generated: {repr(generated_text)}")

    print("\n=== Key observations ===")
    print("1. loss.backward() computes gradients; optimizer.step() applies them")
    print("2. model.train()/model.eval() toggles dropout for training vs eval")
    print("3. Evaluating on a subset of batches (eval_iter) keeps checks fast")
    print("4. Train vs val loss gap reveals overfitting")
    print("5. With only ~20K characters of training text, this model will overfit quickly")
