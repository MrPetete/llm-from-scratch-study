"""
Full GPT Model.

Assembles every piece built so far into one model:

    token IDs
      -> token embeddings + positional embeddings (Chapter 1)
      -> dropout
      -> stack of N TransformerBlocks (attention + feed-forward + shortcuts)
      -> final LayerNorm
      -> linear output head -> logits over the vocabulary [vocab_size]

The output is NOT probabilities yet -- it's raw logits, one score per
vocabulary token, for every position in the sequence. Softmax (applied during
generation, not inside the model) turns these into a probability distribution.
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
        # Output head: projects back to vocab_size logits. Note this is a
        # SEPARATE weight matrix from token_embedding here (no weight tying) --
        # that's why the book's as-coded param count (163M) is higher than the
        # widely-quoted 124M, which assumes tying output.weight = token_embedding.weight.
        self.output_head = nn.Linear(cfg["embed_dim"], cfg["vocab_size"], bias=False)

    def forward(self, token_ids):
        batch_size, seq_len = token_ids.shape

        token_embeds = self.token_embedding(token_ids)                 # [batch, seq_len, embed_dim]
        positions = torch.arange(seq_len, device=token_ids.device)
        pos_embeds = self.pos_embedding(positions)                     # [seq_len, embed_dim]

        x = token_embeds + pos_embeds
        x = self.dropout_embed(x)
        x = self.transformer_blocks(x)
        x = self.final_norm(x)
        logits = self.output_head(x)                                   # [batch, seq_len, vocab_size]
        return logits


def count_parameters(model, tie_weights=False):
    """
    Count total trainable parameters. If tie_weights=True, count as if
    output_head shares its weights with token_embedding (GPT-2's actual
    reported parameter count assumes this tying; 124M vs 163M as-coded).
    """
    total = sum(p.numel() for p in model.parameters())
    if tie_weights:
        # output_head.weight would no longer be counted separately
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
    print(f"\nOutput logits shape: {logits.shape}  (batch, seq_len, vocab_size)")
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
    print(f"Parameter count with output/token-embedding weight TYING:  {n_params_tied:,}")
    print("The book reports ~163M as-coded vs the commonly-cited 124M with tying --")
    print("both describe the SAME architecture, just whether output_head.weight is")
    print("a separate matrix or literally the same tensor as token_embedding.weight.\n")

    # Rough memory footprint at float32 (4 bytes/param)
    size_mb = n_params_untied * 4 / (1024 ** 2)
    print(f"Approx. size at float32 (4 bytes/param): {size_mb:.1f} MB")
    print("(Book reports ~622 MB for the as-coded 163M-param version -- matches this estimate.)\n")

    print("=== Key observations ===")
    print("1. Model output is LOGITS, not probabilities -- softmax happens during generation")
    print("2. Same GPTModel class, just different cfg dict, scales from tiny to GPT-2 small/XL")
    print("3. Untrained model still produces correctly-SHAPED output -- weights are just random")
    print("4. Weight tying (sharing token_embedding and output_head weights) saves real memory")
    print("   at GPT-2 scale (768 x 50257 = ~38.6M parameters saved)")
