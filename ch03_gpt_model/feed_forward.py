"""
GELU activation + Feed-Forward network.

GELU (Gaussian Error Linear Unit) is a smoother alternative to ReLU. Unlike
ReLU's hard cutoff at 0 (f(x) = max(0, x)), GELU has a smooth curve that
still passes small negative values through with a small negative output,
weighted by how "Gaussian-likely" that input is. This smoothness helps
gradients behave better during training -- no sharp kink at x=0.

The feed-forward network is a small per-token MLP: expand embed_dim -> 4x,
apply GELU, then compress back down to embed_dim. It runs independently on
each token's vector (no mixing across tokens -- that's attention's job).
This gives the model extra capacity to transform each token's representation
after attention has gathered context from other tokens.
"""

import torch
import torch.nn as nn


class GELU(nn.Module):
    """
    GELU activation, using the tanh approximation from the original GPT-2 paper
    (exact erf-based GELU exists too, but this approximation is what GPT-2 uses).
    """
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) * (x + 0.044715 * x**3)
        ))


class FeedForward(nn.Module):
    """
    Per-token MLP: embed_dim -> 4*embed_dim -> GELU -> embed_dim.

    Args:
        cfg: config dict with "embed_dim" key
    """
    def __init__(self, cfg):
        super().__init__()
        embed_dim = cfg["embed_dim"]
        self.layers = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            GELU(),
            nn.Linear(4 * embed_dim, embed_dim),
        )

    def forward(self, x):
        return self.layers(x)


if __name__ == "__main__":
    torch.manual_seed(123)

    print("=== GELU vs ReLU ===\n")

    x = torch.linspace(-3, 3, 11)
    gelu = GELU()
    relu = nn.ReLU()

    print(f"Input:  {x}")
    print(f"GELU:   {gelu(x)}")
    print(f"ReLU:   {relu(x)}")
    print("\nNotice: ReLU is exactly 0 for all negative inputs (hard cutoff).")
    print("GELU passes small negative values through with a small negative output")
    print("(e.g. at x=-0.6, GELU is slightly negative, not exactly 0) -- smoother curve,")
    print("no sharp kink in the gradient at x=0.\n")

    print("=== Feed-Forward Network ===\n")

    from config import GPT_CONFIG_TINY

    ff = FeedForward(GPT_CONFIG_TINY)
    embed_dim = GPT_CONFIG_TINY["embed_dim"]

    # Toy batch: [batch_size=2, seq_len=3, embed_dim]
    batch = torch.rand(2, 3, embed_dim)
    print(f"Input shape:  {batch.shape}  (batch, seq_len, embed_dim={embed_dim})")

    out = ff(batch)
    print(f"Output shape: {out.shape}  (same as input -- FFN preserves shape)")

    # Show the intermediate expansion explicitly
    hidden = ff.layers[0](batch)
    print(f"\nIntermediate hidden shape (after first Linear, before GELU): {hidden.shape}")
    print(f"Expanded from {embed_dim} -> {4 * embed_dim} dims (4x expansion), then compressed back")

    print("\n=== Key observations ===")
    print("1. FeedForward operates independently on each token -- no cross-token mixing")
    print("2. 4x expansion gives the model more capacity to transform each representation")
    print("3. GELU's smooth curve (vs ReLU's hard 0 cutoff) is used throughout GPT-2")
    print("4. Output shape always matches input shape -- this block is stackable/composable")
