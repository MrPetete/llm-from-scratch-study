"""Greedy text generation loop (argmax decoding)."""

import torch

from config import GPT_CONFIG_TINY
from gpt_model import GPTModel


def generate_text(model, token_ids, max_new_tokens, context_length):
    """
    Greedy (argmax) text generation.

    Args:
        model: a GPTModel instance
        token_ids: [batch, seq_len] starting token IDs
        max_new_tokens: how many new tokens to generate
        context_length: model's max sequence length -- input is truncated to this
            each step so the model never sees more context than it was built for

    Returns:
        token_ids: [batch, seq_len + max_new_tokens]
    """
    model.eval()
    for _ in range(max_new_tokens):
        input_window = token_ids[:, -context_length:]

        with torch.no_grad():
            logits = model(input_window)

        last_logits = logits[:, -1, :]
        probs = torch.softmax(last_logits, dim=-1)
        next_token = torch.argmax(probs, dim=-1, keepdim=True)

        token_ids = torch.cat([token_ids, next_token], dim=1)

    return token_ids


if __name__ == "__main__":
    import tiktoken

    torch.manual_seed(123)

    print("=== Text Generation Loop (untrained model -- gibberish expected) ===\n")

    tokenizer = tiktoken.get_encoding("gpt2")
    model = GPTModel(GPT_CONFIG_TINY)

    start_text = "Hello, I am"
    start_ids = torch.tensor([tokenizer.encode(start_text)])
    print(f"Starting text: '{start_text}'")
    print(f"Starting token IDs: {start_ids.tolist()}\n")

    print("--- Single step, expanded ---")
    model.eval()
    with torch.no_grad():
        logits = model(start_ids)
    print(f"1. Logits shape: {logits.shape}")

    last_logits = logits[:, -1, :]
    print(f"2. Last-position logits shape: {last_logits.shape}")

    probs = torch.softmax(last_logits, dim=-1)
    print(f"3. Probabilities sum to 1: {probs.sum(dim=-1)}")

    next_token = torch.argmax(probs, dim=-1, keepdim=True)
    next_token_text = tokenizer.decode([next_token.item()])
    print(f"4. Argmax next token ID: {next_token.item()}  -> decodes to: {repr(next_token_text)}")

    top5_probs, top5_ids = torch.topk(probs, 5, dim=-1)
    print(f"\nTop-5 candidate tokens:")
    for prob, tid in zip(top5_probs[0], top5_ids[0]):
        print(f"   {repr(tokenizer.decode([tid.item()]))}: {prob.item():.4f}")

    print("\n--- Full generation loop (10 new tokens) ---")
    generated_ids = generate_text(
        model, start_ids, max_new_tokens=10,
        context_length=GPT_CONFIG_TINY["context_length"]
    )
    generated_text = tokenizer.decode(generated_ids[0].tolist())

    print(f"Generated token IDs: {generated_ids.tolist()}")
    print(f"Generated text: {repr(generated_text)}")
    print("\nGibberish is expected -- random, untrained weights. Point is the loop")
    print("runs correctly end to end: shapes match, sequence grows by 1 token per step.\n")

    print("=== Key observations ===")
    print("1. Only the last position's logits matter for generating the next token")
    print("2. softmax -> argmax is the full greedy decoding pipeline")
    print("3. The growing sequence is truncated to context_length each step")
    print("4. model.eval() disables dropout for deterministic generation")
    print("5. Gibberish confirms mechanics are correct; Chapter 5 adds real training")
