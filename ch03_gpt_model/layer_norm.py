"""
Layer Normalization.

Rescales each token's feature vector independently to mean=0, variance=1,
then applies two LEARNED parameters (scale, shift) so the network can still
adjust the normalized output during training. Keeps activations stable as
they flow through many stacked layers -- without this, deep stacks tend to
have activations that explode or vanish in scale.

Note: LayerNorm normalizes across the embedding dimension (last dim), per
token, per sample -- unlike BatchNorm which normalizes across the batch.
This matters for sequence models: each token's normalization doesn't depend
on other tokens in the batch or sequence.
"""

import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    """
    Args:
        embed_dim: size of the last dimension to normalize over
        eps: small constant added to variance to avoid division by zero
    """
    def __init__(self, embed_dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(embed_dim))   # learned, init to 1 (no-op at start)
        self.shift = nn.Parameter(torch.zeros(embed_dim))  # learned, init to 0 (no-op at start)

    def forward(self, x):
        # x: [..., embed_dim] -- normalize over the last dimension
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)  # biased variance (matches GPT-2's original impl)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


if __name__ == "__main__":
    torch.manual_seed(123)

    print("=== Layer Normalization ===\n")

    # Toy batch: 2 samples, 5 features each, deliberately unnormalized (different scales)
    batch = torch.tensor([
        [-0.5, 3.2, 10.1, -8.4, 1.0],
        [2.0, 2.1, 1.9, 2.05, 100.0],   # one huge outlier feature
    ])
    print(f"Input:\n{batch}")
    print(f"Input mean per row: {batch.mean(dim=-1)}")
    print(f"Input var per row:  {batch.var(dim=-1, unbiased=False)}\n")

    ln = LayerNorm(embed_dim=5)
    normed = ln(batch)

    print(f"Output:\n{normed}")
    print(f"Output mean per row (should be ~0): {normed.mean(dim=-1)}")
    print(f"Output var per row (should be ~1):  {normed.var(dim=-1, unbiased=False)}\n")

    print("--- Effect of scale/shift being learnable ---")
    print(f"scale (init to 1s): {ln.scale}")
    print(f"shift (init to 0s): {ln.shift}")
    print("At init, LayerNorm is a pure normalize-to-(mean=0,var=1) operation.")
    print("During training, scale/shift are updated by gradient descent, letting")
    print("the network learn to re-widen or re-shift the distribution if useful.\n")

    print("=== Key observations ===")
    print("1. Normalization happens PER TOKEN (per row here), not across the batch")
    print("2. Output mean ~0 and variance ~1 regardless of how skewed the input was")
    print("3. scale and shift are trainable -- LayerNorm isn't purely fixed math")
    print("4. eps prevents division by zero when a token's variance is exactly 0")
