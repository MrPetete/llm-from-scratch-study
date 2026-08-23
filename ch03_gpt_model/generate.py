"""
Text Generation Loop.

The model produces logits for every position in the input sequence, but we
only care about the LAST position's logits when generating -- that's the
prediction for "what comes next" given everything seen so far.

Loop:
    1. Forward pass: logits = model(input_ids)                [batch, seq_len, vocab_size]
    2. Take only the last position: logits[:, -1, :]           [batch, vocab_size]
    3. softmax -> probabilities over the vocabulary
    4. argmax (greedy) -> pick the single highest-probability token ID
    5. Append that token ID to the input sequence
    6. Repeat, feeding the EXTENDED sequence back in

On an UNTRAINED model (random weights), this loop still runs correctly end
to end -- the output text will be gibberish, because the model hasn't
learned anything yet. That's the correctness checkpoint for this chapter:
verify the MECHANICS work, not the quality of the output.
"""

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
        context_length: the model's max sequence length -- older tokens are
            dropped from the input once the sequence exceeds this, so the
            model never sees more context than it was built to handle

    Returns:
        token_ids: [batch, seq_len + max_new_tokens]
    """
    model.eval()   # disable dropout for generation
    for _ in range(max_new_tokens):
        # Truncate to the last `context_length` tokens if the sequence has grown too long
        input_window = token_ids[:, -context_length:]

        with torch.no_grad():
            logits = model(input_window)               # [batch, seq_len, vocab_size]

        last_logits = logits[:, -1, :]                  # [batch, vocab_size] -- only the newest position
        probs = torch.softmax(last_logits, dim=-1)       # [batch, vocab_size]
        next_token = torch.argmax(probs, dim=-1, keepdim=True)  # [batch, 1] -- greedy pick

        token_ids = torch.cat([token_ids, next_token], dim=1)   # append and feed back in

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

    # --- Walk through ONE generation step manually, showing every intermediate tensor ---
    print("--- Single step, expanded ---")
    model.eval()
    with torch.no_grad():
        logits = model(start_ids)
    print(f"1. Logits shape: {logits.shape}  (batch, seq_len, vocab_size)")

    last_logits = logits[:, -1, :]
    print(f"2. Last-position logits shape: {last_logits.shape}  (only care about newest position)")

    probs = torch.softmax(last_logits, dim=-1)
    print(f"3. Probabilities sum to 1: {probs.sum(dim=-1)}")

    next_token = torch.argmax(probs, dim=-1, keepdim=True)
    next_token_text = tokenizer.decode([next_token.item()])
    print(f"4. Argmax next token ID: {next_token.item()}  -> decodes to: {repr(next_token_text)}")

    top5_probs, top5_ids = torch.topk(probs, 5, dim=-1)
    print(f"\nTop-5 candidate tokens (for context -- greedy just picks #1):")
    for prob, tid in zip(top5_probs[0], top5_ids[0]):
        print(f"   {repr(tokenizer.decode([tid.item()]))}: {prob.item():.4f}")

    # --- Full loop: generate several tokens ---
    print("\n--- Full generation loop (10 new tokens) ---")
    generated_ids = generate_text(
        model, start_ids, max_new_tokens=10,
        context_length=GPT_CONFIG_TINY["context_length"]
    )
    generated_text = tokenizer.decode(generated_ids[0].tolist())

    print(f"Generated token IDs: {generated_ids.tolist()}")
    print(f"Generated text: {repr(generated_text)}")
    print("\nThis is gibberish, as expected -- the model has RANDOM, untrained weights.")
    print("The point of this checkpoint is that the loop runs correctly end to end:")
    print("shapes match, softmax normalizes properly, argmax picks a valid token ID,")
    print("and the sequence grows by exactly 1 token per iteration.\n")

    print("=== Key observations ===")
    print("1. Only the LAST position's logits matter for generating the next token")
    print("2. softmax -> probs -> argmax is the full 'pick next word' pipeline (greedy decoding)")
    print("3. The growing sequence is truncated to context_length on each step (sliding window)")
    print("4. model.eval() disables dropout -- generation should be deterministic given weights")
    print("5. Gibberish output confirms MECHANICS are correct; Chapter 5 will add real training")
