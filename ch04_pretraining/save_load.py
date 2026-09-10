"""Save/load model + optimizer state so training doesn't need to restart from scratch."""

import os
import sys
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ch03_gpt_model"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ch01_tokenizer"))


def save_checkpoint(model, optimizer, path):
    """Save both model weights and optimizer state (allows exact training resume)."""
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, path)


def load_checkpoint(path, model, optimizer=None, map_location="cpu"):
    """Load a checkpoint into an already-constructed model (must use the same config)."""
    checkpoint = torch.load(path, map_location=map_location)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return model, optimizer


if __name__ == "__main__":
    from config import GPT_CONFIG_TINY
    from gpt_model import GPTModel
    from dataloader import create_dataloader_v1
    from loss import calc_loss_batch
    import tiktoken

    torch.manual_seed(123)

    print("=== Saving and Loading Model Weights ===\n")

    # Train briefly so we have real (non-random) weights to save
    data_path = os.path.join(os.path.dirname(__file__), "..", "ch01_tokenizer", "data", "the-verdict.txt")
    with open(data_path, "r", encoding="utf-8") as f:
        raw_text = f.read()
    split_idx = int(len(raw_text) * 0.9)
    train_text = raw_text[:split_idx]

    context_length = GPT_CONFIG_TINY["context_length"] // 4
    train_loader = create_dataloader_v1(
        train_text, batch_size=2, context_length=context_length,
        stride=context_length, shuffle=True, drop_last=True
    )

    model = GPTModel(GPT_CONFIG_TINY)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.1)

    print("Training briefly (5 epochs)...")
    for epoch in range(5):
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model)
            loss.backward()
            optimizer.step()
    print(f"Done. Final training loss: {loss.item():.4f}\n")

    checkpoint_dir = os.path.join(os.path.dirname(__file__), "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, "tiny_gpt_verdict.pt")

    save_checkpoint(model, optimizer, checkpoint_path)
    file_size_kb = os.path.getsize(checkpoint_path) / 1024
    print(f"Saved checkpoint to: {checkpoint_path}")
    print(f"File size: {file_size_kb:.1f} KB\n")

    print("--- Verifying save/load round-trip ---")
    tokenizer = tiktoken.get_encoding("gpt2")
    start_text = "Every effort moves you"
    start_ids = torch.tensor([tokenizer.encode(start_text)])

    from generate import generate_text

    torch.manual_seed(999)  # different seed -- fresh random init
    fresh_model = GPTModel(GPT_CONFIG_TINY)
    fresh_ids = generate_text(fresh_model, start_ids, max_new_tokens=15, context_length=context_length)
    print(f"Fresh (random, untrained) model output: {repr(tokenizer.decode(fresh_ids[0].tolist()))}")

    trained_ids = generate_text(model, start_ids, max_new_tokens=15, context_length=context_length)
    print(f"Originally trained model output:        {repr(tokenizer.decode(trained_ids[0].tolist()))}")

    fresh_optimizer = torch.optim.AdamW(fresh_model.parameters(), lr=5e-4, weight_decay=0.1)
    load_checkpoint(checkpoint_path, fresh_model, fresh_optimizer)

    loaded_ids = generate_text(fresh_model, start_ids, max_new_tokens=15, context_length=context_length)
    print(f"Fresh model AFTER loading checkpoint:   {repr(tokenizer.decode(loaded_ids[0].tolist()))}")

    match = torch.equal(trained_ids, loaded_ids)
    print(f"\nLoaded model's output EXACTLY matches the original trained model: {match}")

    weights_match = all(
        torch.equal(p1, p2)
        for p1, p2 in zip(model.parameters(), fresh_model.parameters())
    )
    print(f"All model weights are bit-for-bit identical after load: {weights_match}")

    print("\n=== Key observations ===")
    print("1. state_dict() saves only the WEIGHTS, not the model class -- architecture must be")
    print("   reconstructed with the same config before loading")
    print("2. Saving optimizer state too allows resuming training with momentum intact, not just")
    print("   restarting from scratch with fresh Adam moment estimates")
    print("3. Round-trip verified: fresh model's output DIFFERS before loading, then EXACTLY")
    print("   matches the original trained model's output after loading the checkpoint")
    print("4. This is how we avoid retraining from scratch every session going forward")
