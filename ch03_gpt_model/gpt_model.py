"""
Full GPT model: token+positional embeddings -> stack of TransformerBlocks ->
final LayerNorm -> output head -> logits over the vocabulary.
"""

import torch
import torch.nn as nn

from config import GPT_CONFIG_124M, GPT_CONFIG_TINY
from layer_norm import LayerNorm
from transformer_block import TransformerBlock


class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.token_embedding = nn.Embedding(cfg["vocab_size"], cfg["embed_dim"])
        self.pos_embedding = nn.Embedding(cfg["context_length"], cfg["embed_dim"])
        self.dropout_embed = nn.Dropout(cfg["dropout"])

        self.transformer_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["num_layers"])]
        )

        self.final_norm = LayerNorm(cfg["embed_dim"])
        # Separate weight matrix from token_embedding (no weight tying) -- that's why
        # the as-coded param count (163M) is higher than the commonly-cited 124M.
        self.output_head = nn.Linear(cfg["embed_dim"], cfg["vocab_size"], bias=False)

    def forward(self, token_ids):
        batch_size, seq_len = token_ids.shape

        token_embeds = self.token_embedding(token_ids)
        positions = torch.arange(seq_len, device=token_ids.device)
        pos_embeds = self.pos_embedding(positions)

        x = token_embeds + pos_embeds
        x = self.dropout_embed(x)
        x = self.transformer_blocks(x)
        x = self.final_norm(x)
        logits = self.output_head(x)
        return logits


def count_parameters(model, tie_weights=False):
    """Count trainable parameters. tie_weights=True mimics sharing output_head
    with token_embedding (GPT-2's reported 124M assumes this tying)."""
    total = sum(p.numel() for p in model.parameters())
    if tie_weights:
        total -= model.output_head.weight.numel()
    return total


if __name__ == "__main__":
    torch.manual_seed(123)

    print("=== Full GPT Model ===\n")

    print("--- Tiny config (for fast CPU experimentation) ---")
    model_tiny = GPTModel(GPT_CONFIG_TINY)
    print(f"Config: {GPT_CONFIG_TINY}")

    import tiktoken
    tokenizer = tiktoken.get_encoding("gpt2")

    text = "Hello, I am"
    token_ids = torch.tensor([tokenizer.encode(text)])
    print(f"\nInput text: '{text}'")
    print(f"Token IDs: {token_ids.tolist()}  shape={token_ids.shape}")

    with torch.no_grad():
        logits = model_tiny(token_ids)
    print(f"\nOutput logits shape: {logits.shape}")
    print(f"Expected: [1, {token_ids.shape[1]}, {GPT_CONFIG_TINY['vocab_size']}]")
    assert logits.shape == (1, token_ids.shape[1], GPT_CONFIG_TINY["vocab_size"]), "Shape mismatch!"
    print("Shape check passed.\n")

    n_params_tiny = count_parameters(model_tiny)
    print(f"Tiny model parameter count: {n_params_tiny:,}\n")

    print("--- GPT-2 small config (124M, structural comparison only) ---")
    print(f"Config: {GPT_CONFIG_124M}")
    model_124m = GPTModel(GPT_CONFIG_124M)

    n_params_untied = count_parameters(model_124m)
    n_params_tied = count_parameters(model_124m, tie_weights=True)
    print(f"\nParameter count as coded (separate output_head weights): {n_params_untied:,}")
    print(f"Parameter count with weight tying:                        {n_params_tied:,}")
    print("Both describe the same architecture -- just whether output_head.weight is")
    print("a separate matrix or the same tensor as token_embedding.weight.\n")

    size_mb = n_params_untied * 4 / (1024 ** 2)
    print(f"Approx. size at float32: {size_mb:.1f} MB\n")

    print("=== Key observations ===")
    print("1. Model output is logits, not probabilities -- softmax happens during generation")
    print("2. Same GPTModel class scales from tiny to GPT-2 small/XL via cfg dict")
    print("3. Untrained model still produces correctly-shaped output")
    print("4. Weight tying saves real memory at GPT-2 scale (~38.6M params)")
