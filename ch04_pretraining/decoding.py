"""Decoding strategies: temperature scaling and top-k sampling, as alternatives to greedy argmax."""

import os
import sys
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ch03_gpt_model"))


def generate_text_sampled(model, token_ids, max_new_tokens, context_length,
                           temperature=1.0, top_k=None, eos_id=None):
    """
    Text generation with temperature scaling and optional top-k sampling.

    Args:
        model: a GPTModel instance
        token_ids: [batch, seq_len] starting token IDs
        max_new_tokens: how many new tokens to generate
        context_length: model's max sequence length (sliding window truncation)
        temperature: softmax temperature. 1.0 = unscaled. 0.0 falls back to
            greedy argmax (dividing by 0 otherwise).
        top_k: if set, restrict sampling to the top_k highest-logit tokens
        eos_id: if set, stop generation early once this token ID is produced

    Returns:
        token_ids: [batch, seq_len + up to max_new_tokens]
    """
    model.eval()
    for _ in range(max_new_tokens):
        input_window = token_ids[:, -context_length:]

        with torch.no_grad():
            logits = model(input_window)

        last_logits = logits[:, -1, :]

        if top_k is not None:
            top_logits, _ = torch.topk(last_logits, top_k)
            min_val = top_logits[:, -1]
            last_logits = torch.where(
                last_logits < min_val.unsqueeze(-1),
                torch.tensor(-torch.inf, device=last_logits.device),
                last_logits,
            )

        if temperature > 0.0:
            scaled_logits = last_logits / temperature
            probs = torch.softmax(scaled_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            next_token = torch.argmax(last_logits, dim=-1, keepdim=True)

        if eos_id is not None and (next_token == eos_id).all():
            break

        token_ids = torch.cat([token_ids, next_token], dim=1)

    return token_ids


if __name__ == "__main__":
    import tiktoken

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ch01_tokenizer"))
    from config import GPT_CONFIG_TINY
    from gpt_model import GPTModel

    torch.manual_seed(123)

    print("=== Decoding Strategies: Temperature and Top-k ===\n")

    print("--- Temperature scaling on a toy logit distribution ---")
    toy_logits = torch.tensor([1.0, 2.0, 3.0, 0.5, 0.1])
    for temp in [0.1, 1.0, 2.0]:
        probs = torch.softmax(toy_logits / temp, dim=-1)
        print(f"T={temp}: probs = {probs}")
    print("Lower T -> sharper distribution. Higher T -> flatter distribution.\n")

    print("--- Top-k filtering (k=3) on the same toy logits ---")
    top_k = 3
    top_logits, top_idx = torch.topk(toy_logits, top_k)
    min_val = top_logits[-1]
    filtered = torch.where(toy_logits < min_val, torch.tensor(-torch.inf), toy_logits)
    print(f"Original logits: {toy_logits}")
    print(f"After top-{top_k} filter: {filtered}\n")

    print("--- Comparing decoding strategies on a briefly-trained model ---")
    print("(Retraining here for a self-contained demo -- see train.py for the full run)\n")

    from dataloader import create_dataloader_v1
    from loss import calc_loss_batch

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
    for epoch in range(10):
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model)
            loss.backward()
            optimizer.step()

    tokenizer = tiktoken.get_encoding("gpt2")
    start_text = "Every effort moves you"
    start_ids = torch.tensor([tokenizer.encode(start_text)])

    print(f"Prompt: {repr(start_text)}\n")

    torch.manual_seed(123)
    from generate import generate_text
    greedy_ids = generate_text(model, start_ids, max_new_tokens=20, context_length=context_length)
    print(f"Greedy (argmax):              {repr(tokenizer.decode(greedy_ids[0].tolist()))}")

    torch.manual_seed(123)
    low_temp_ids = generate_text_sampled(model, start_ids, max_new_tokens=20, context_length=context_length,
                                          temperature=0.5, top_k=None)
    print(f"Temperature=0.5, no top-k:    {repr(tokenizer.decode(low_temp_ids[0].tolist()))}")

    torch.manual_seed(123)
    high_temp_ids = generate_text_sampled(model, start_ids, max_new_tokens=20, context_length=context_length,
                                           temperature=1.5, top_k=None)
    print(f"Temperature=1.5, no top-k:    {repr(tokenizer.decode(high_temp_ids[0].tolist()))}")

    torch.manual_seed(123)
    topk_ids = generate_text_sampled(model, start_ids, max_new_tokens=20, context_length=context_length,
                                      temperature=1.0, top_k=5)
    print(f"Temperature=1.0, top_k=5:     {repr(tokenizer.decode(topk_ids[0].tolist()))}")

    print("\n=== Key observations ===")
    print("1. Greedy decoding is deterministic and prone to repetition loops")
    print("2. temperature > 1 increases diversity but risks incoherent output")
    print("3. temperature < 1 stays closer to greedy but with some variation")
    print("4. top_k caps the candidate pool, preventing wildly implausible picks")
    print("5. Real-world generation typically combines top_k/top_p with temperature")
