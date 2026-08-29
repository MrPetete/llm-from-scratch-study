"""
Chapter 4, Step 3: Decoding Strategies (Temperature, Top-k)

Greedy decoding (argmax, from Chapter 3) always picks the single
highest-probability token. This is deterministic but tends to get stuck in
repetitive loops -- exactly what we just saw ("the the the the...") after
training. Two techniques fix this by adding controlled randomness:

TEMPERATURE SCALING
    Divide logits by a temperature T before softmax:
        probs = softmax(logits / T)
    - T < 1.0: sharpens the distribution (more confident, closer to greedy)
    - T = 1.0: unchanged
    - T > 1.0: flattens the distribution (more random, more diverse/risky)
    Then SAMPLE from this distribution (torch.multinomial) instead of argmax.

TOP-K SAMPLING
    Before sampling, zero out (set to -inf) every token EXCEPT the k highest
    -logit ones. This prevents temperature from ever sampling a wildly
    inappropriate low-probability token -- we only ever sample among the
    k most plausible candidates, just with controlled randomness about
    WHICH of those k gets picked.

Combining both: top-k narrows the candidate pool, temperature controls how
sharply we prefer the top of that pool vs spreading weight across it.
"""

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
            greedy argmax (temperature=0 would divide by zero otherwise).
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

        last_logits = logits[:, -1, :]   # [batch, vocab_size]

        # --- Top-k filtering: keep only the k highest logits, -inf out the rest ---
        if top_k is not None:
            top_logits, _ = torch.topk(last_logits, top_k)
            min_val = top_logits[:, -1]                      # kth highest logit, per batch row
            last_logits = torch.where(
                last_logits < min_val.unsqueeze(-1),
                torch.tensor(-torch.inf, device=last_logits.device),
                last_logits,
            )

        # --- Sample (temperature > 0) or greedy argmax (temperature == 0) ---
        if temperature > 0.0:
            scaled_logits = last_logits / temperature
            probs = torch.softmax(scaled_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)   # [batch, 1] -- actual random sampling
        else:
            next_token = torch.argmax(last_logits, dim=-1, keepdim=True)   # deterministic fallback

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

    # --- Demonstrate temperature's effect on a toy distribution first ---
    print("--- Temperature scaling on a toy logit distribution ---")
    toy_logits = torch.tensor([1.0, 2.0, 3.0, 0.5, 0.1])
    for temp in [0.1, 1.0, 2.0]:
        probs = torch.softmax(toy_logits / temp, dim=-1)
        print(f"T={temp}: probs = {probs}")
    print("Lower T -> sharper (closer to one-hot). Higher T -> flatter (closer to uniform).\n")

    # --- Demonstrate top-k filtering on the same toy logits ---
    print("--- Top-k filtering (k=3) on the same toy logits ---")
    top_k = 3
    top_logits, top_idx = torch.topk(toy_logits, top_k)
    min_val = top_logits[-1]
    filtered = torch.where(toy_logits < min_val, torch.tensor(-torch.inf), toy_logits)
    print(f"Original logits: {toy_logits}")
    print(f"After top-{top_k} filter: {filtered}")
    print("The 2 lowest logits became -inf -- softmax will assign them exactly 0 probability.\n")

    # --- Use the model we just trained -- reload via a fresh run for a clean comparison ---
    print("--- Comparing decoding strategies on the trained model ---")
    print("(Retraining briefly here for a self-contained demo -- see train.py for the full run)\n")

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
    print("4. top_k caps the candidate pool -- prevents sampling a wildly implausible token")
    print("5. Real-world generation (e.g. GPT-2/3/4 APIs) typically combines both: top_k or")
    print("   top_p (nucleus sampling) PLUS a temperature setting")
