"""Layer normalization -- normalizes each token's feature vector to mean 0, var 1."""

import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    """
    Args:
        embed_dim: size of the last dimension to normalize over
        eps: small constant to avoid division by zero
    """
    def __init__(self, embed_dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(embed_dim))
        self.shift = nn.Parameter(torch.zeros(embed_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)  # biased variance, matches GPT-2's original impl
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


if __name__ == "__main__":
    torch.manual_seed(123)

    print("=== Layer Normalization ===\n")

    batch = torch.tensor([
        [-0.5, 3.2, 10.1, -8.4, 1.0],
        [2.0, 2.1, 1.9, 2.05, 100.0],
    ])
    print(f"Input:\n{batch}")
    print(f"Input mean per row: {batch.mean(dim=-1)}")
    print(f"Input var per row:  {batch.var(dim=-1, unbiased=False)}\n")

    ln = LayerNorm(embed_dim=5)
    normed = ln(batch)

    print(f"Output:\n{normed}")
    print(f"Output mean per row (should be ~0): {normed.mean(dim=-1)}")
    print(f"Output var per row (should be ~1):  {normed.var(dim=-1, unbiased=False)}\n")

    print(f"scale (init to 1s): {ln.scale}")
    print(f"shift (init to 0s): {ln.shift}")

    print("\n=== Key observations ===")
    print("1. Normalization happens per token, not across the batch")
    print("2. Output mean ~0 and variance ~1 regardless of input scale")
    print("3. scale and shift are trainable")
    print("4. eps prevents division by zero")
