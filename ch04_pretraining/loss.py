"""
Chapter 4, Step 1: Cross-Entropy Loss

The model outputs logits [batch, seq_len, vocab_size] -- one score per
vocabulary token at every sequence position. To train, we need a single
scalar that measures "how wrong" these predictions are against the actual
next tokens (the targets from our sliding-window DataLoader).

Cross-entropy loss:
    1. softmax(logits) -> probabilities over the vocabulary at each position
    2. Look up the probability the model assigned to the TRUE target token
    3. loss = -log(that probability)
       - true prob = 1.0 -> loss = 0 (perfect prediction)
       - true prob -> 0   -> loss -> infinity (terrible prediction)
    4. Average this over all positions and all sequences in the batch

PyTorch's nn.functional.cross_entropy does steps 1-3 in one numerically
stable call (it works directly on logits, not pre-softmaxed probabilities,
to avoid float precision issues with very small/large numbers).
"""

import torch
import torch.nn.functional as F


def calc_loss_batch(input_batch, target_batch, model, device="cpu"):
    """
    Cross-entropy loss for one batch of (input, target) token ID pairs.

    Args:
        input_batch: [batch, seq_len] token IDs
        target_batch: [batch, seq_len] token IDs (input shifted by 1, from GPTDatasetV1)
        model: a GPTModel instance
        device: "cpu" or "cuda"

    Returns:
        scalar loss tensor
    """
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)   # [batch, seq_len, vocab_size]

    # cross_entropy expects [N, num_classes] and [N] -- flatten batch and seq_len together
    loss = F.cross_entropy(
        logits.flatten(0, 1),      # [batch*seq_len, vocab_size]
        target_batch.flatten(),    # [batch*seq_len]
    )
    return loss


def calc_loss_loader(dataloader, model, device="cpu", num_batches=None):
    """
    Average cross-entropy loss over multiple batches from a DataLoader.

    Args:
        dataloader: yields (input_batch, target_batch) pairs
        model: a GPTModel instance
        device: "cpu" or "cuda"
        num_batches: cap on how many batches to average over (None = all of them)

    Returns:
        average loss (float), or nan if the dataloader is empty
    """
    total_loss = 0.0
    if len(dataloader) == 0:
        return float("nan")

    if num_batches is None:
        num_batches = len(dataloader)
    else:
        num_batches = min(num_batches, len(dataloader))

    for i, (input_batch, target_batch) in enumerate(dataloader):
        if i >= num_batches:
            break
        loss = calc_loss_batch(input_batch, target_batch, model, device)
        total_loss += loss.item()

    return total_loss / num_batches


if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ch03_gpt_model"))
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ch01_tokenizer"))

    from config import GPT_CONFIG_TINY
    from gpt_model import GPTModel
    from dataloader import create_dataloader_v1

    torch.manual_seed(123)

    print("=== Cross-Entropy Loss ===\n")

    # --- Manual walkthrough on a tiny toy example first ---
    print("--- Manual walkthrough: 2 positions, tiny vocab ---")
    # Pretend vocab_size=5, 2 positions in the sequence
    toy_logits = torch.tensor([
        [2.0, 1.0, 0.1, 0.1, 0.1],   # position 0's logits over 5-token vocab
        [0.1, 0.1, 0.1, 3.0, 0.1],   # position 1's logits
    ])
    toy_targets = torch.tensor([0, 3])   # true next token at each position

    probs = torch.softmax(toy_logits, dim=-1)
    print(f"Logits:\n{toy_logits}")
    print(f"Probabilities (softmax):\n{probs}")
    print(f"True targets: {toy_targets}")

    true_token_probs = probs[torch.arange(2), toy_targets]
    print(f"Probability assigned to the TRUE target at each position: {true_token_probs}")

    manual_loss = -torch.log(true_token_probs).mean()
    print(f"Manual loss (-log(true_prob), averaged): {manual_loss:.4f}")

    pytorch_loss = torch.nn.functional.cross_entropy(toy_logits, toy_targets)
    print(f"PyTorch F.cross_entropy result:          {pytorch_loss:.4f}")
    print(f"Match: {torch.allclose(manual_loss, pytorch_loss)}\n")

    # --- Real model, real data ---
    print("--- Loss on the untrained tiny GPT model + real text ---")
    model = GPTModel(GPT_CONFIG_TINY)

    data_path = os.path.join(os.path.dirname(__file__), "..", "ch01_tokenizer", "data", "the-verdict.txt")
    with open(data_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    # Split 90/10 train/val (this file, this loss check now -- full train/val loop next step)
    split_idx = int(len(raw_text) * 0.9)
    train_text = raw_text[:split_idx]
    val_text = raw_text[split_idx:]

    train_loader = create_dataloader_v1(
        train_text, batch_size=2, context_length=GPT_CONFIG_TINY["context_length"] // 4,
        stride=GPT_CONFIG_TINY["context_length"] // 4, shuffle=False, drop_last=True
    )
    val_loader = create_dataloader_v1(
        val_text, batch_size=2, context_length=GPT_CONFIG_TINY["context_length"] // 4,
        stride=GPT_CONFIG_TINY["context_length"] // 4, shuffle=False, drop_last=True
    )

    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    train_loss = calc_loss_loader(train_loader, model)
    val_loss = calc_loss_loader(val_loader, model)
    print(f"\nUntrained model -- train loss: {train_loss:.4f}")
    print(f"Untrained model -- val loss:   {val_loss:.4f}")

    import math
    random_guess_loss = math.log(GPT_CONFIG_TINY["vocab_size"])
    print(f"\nFor comparison, loss of a UNIFORM RANDOM guesser: ln({GPT_CONFIG_TINY['vocab_size']}) = {random_guess_loss:.4f}")
    print("An untrained model's loss should be close to this -- it hasn't learned")
    print("anything yet, so its predictions are close to random guessing.")

    print("\n=== Key observations ===")
    print("1. Manual -log(true_prob) computation matches F.cross_entropy exactly")
    print("2. F.cross_entropy works directly on logits (numerically stable, no separate softmax needed)")
    print("3. Untrained model's loss is close to ln(vocab_size) -- the 'random guessing' baseline")
    print("4. Lower loss = model assigns higher probability to the actual next token")
    print("5. This loss is what we'll minimize during training -- next step: the training loop")
