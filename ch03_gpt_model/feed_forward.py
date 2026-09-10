"""GELU activation and the per-token feed-forward MLP."""

import torch
import torch.nn as nn


class GELU(nn.Module):
    """GELU activation, tanh approximation (same as used in GPT-2)."""
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) * (x + 0.044715 * x**3)
        ))


class FeedForward(nn.Module):
    """Per-token MLP: embed_dim -> 4*embed_dim -> GELU -> embed_dim."""
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
    print("\nGELU passes small negative values through with a small negative output,")
    print("unlike ReLU's hard cutoff at 0.\n")

    print("=== Feed-Forward Network ===\n")

    from config import GPT_CONFIG_TINY

    ff = FeedForward(GPT_CONFIG_TINY)
    embed_dim = GPT_CONFIG_TINY["embed_dim"]

    batch = torch.rand(2, 3, embed_dim)
    print(f"Input shape:  {batch.shape}")

    out = ff(batch)
    print(f"Output shape: {out.shape}")

    hidden = ff.layers[0](batch)
    print(f"\nIntermediate hidden shape (after first Linear): {hidden.shape}")
    print(f"Expanded {embed_dim} -> {4 * embed_dim}, then compressed back")

    print("\n=== Key observations ===")
    print("1. FeedForward operates independently on each token, no cross-token mixing")
    print("2. 4x expansion gives extra capacity per token")
    print("3. Output shape always matches input shape")
