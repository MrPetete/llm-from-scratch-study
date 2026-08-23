"""
Transformer Block: the core repeating unit of GPT.

Combines multi-head attention (Chapter 2) and the feed-forward network,
each wrapped in its own LayerNorm and shortcut (residual) connection:

    x -> LayerNorm -> MultiHeadAttention -> (+x)  [shortcut 1]
      -> LayerNorm -> FeedForward        -> (+x)  [shortcut 2]

This is GPT-2's "pre-norm" arrangement: LayerNorm is applied BEFORE each
sub-layer (attention / feed-forward), not after. The shortcut connections add
each sub-layer's INPUT back onto its OUTPUT -- this is what keeps gradients
alive through a deep stack (12 blocks in GPT-2 small, 48 in the largest).
Without shortcuts, gradients would have to flow backward through every
LayerNorm and nonlinearity in every block, and tend to vanish in early layers
of a deep network.

This block is stacked num_layers times to build the full GPT model.
"""

import os
import sys
import torch
import torch.nn as nn

# Reuse the actual Chapter 2 MultiHeadAttention implementation rather than
# duplicating it -- keeps each chapter folder self-contained but avoids a
# copy-paste fork of the same class.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ch02_attention"))
from multihead_attention import MultiHeadAttention  # noqa: E402

from layer_norm import LayerNorm
from feed_forward import FeedForward


class TransformerBlock(nn.Module):
    """
    Args:
        cfg: config dict with embed_dim, num_heads, context_length, dropout, qkv_bias
    """
    def __init__(self, cfg):
        super().__init__()
        self.attention = MultiHeadAttention(
            d_in=cfg["embed_dim"],
            d_out=cfg["embed_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["num_heads"],
            dropout=cfg["dropout"],
            qkv_bias=cfg["qkv_bias"],
        )
        self.feed_forward = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["embed_dim"])
        self.norm2 = LayerNorm(cfg["embed_dim"])
        self.dropout_shortcut = nn.Dropout(cfg["dropout"])

    def forward(self, x):
        # --- Attention sub-layer, with shortcut connection ---
        shortcut = x
        x = self.norm1(x)
        x = self.attention(x)
        x = self.dropout_shortcut(x)
        x = x + shortcut   # residual connection 1: add the block's INPUT back

        # --- Feed-forward sub-layer, with shortcut connection ---
        shortcut = x
        x = self.norm2(x)
        x = self.feed_forward(x)
        x = self.dropout_shortcut(x)
        x = x + shortcut   # residual connection 2

        return x


if __name__ == "__main__":
    torch.manual_seed(123)

    from config import GPT_CONFIG_TINY

    print("=== Transformer Block ===\n")

    block = TransformerBlock(GPT_CONFIG_TINY)
    embed_dim = GPT_CONFIG_TINY["embed_dim"]

    # Toy batch: [batch_size=2, seq_len=4, embed_dim]
    batch = torch.rand(2, 4, embed_dim)
    print(f"Input shape:  {batch.shape}  (batch, seq_len, embed_dim={embed_dim})")

    out = block(batch)
    print(f"Output shape: {out.shape}  (identical to input -- blocks are stackable)\n")

    # --- Demonstrate WHY shortcut connections matter: gradient magnitude ---
    print("--- Shortcut connections and gradient flow ---")
    print("Comparing gradient magnitude at the input, WITH vs WITHOUT shortcuts,")
    print("through a deeper stack of blocks (5 layers, no shortcuts vs with).\n")

    class BlockNoShortcut(nn.Module):
        """Same sub-layers, but shortcuts removed -- for comparison only."""
        def __init__(self, cfg):
            super().__init__()
            self.attention = MultiHeadAttention(
                d_in=cfg["embed_dim"], d_out=cfg["embed_dim"],
                context_length=cfg["context_length"], num_heads=cfg["num_heads"],
                dropout=cfg["dropout"], qkv_bias=cfg["qkv_bias"],
            )
            self.feed_forward = FeedForward(cfg)
            self.norm1 = LayerNorm(cfg["embed_dim"])
            self.norm2 = LayerNorm(cfg["embed_dim"])

        def forward(self, x):
            x = self.attention(self.norm1(x))   # no "+ shortcut"
            x = self.feed_forward(self.norm2(x))  # no "+ shortcut"
            return x

    def measure_input_grad(model_cls, num_layers=5):
        torch.manual_seed(123)
        layers = nn.ModuleList([model_cls(GPT_CONFIG_TINY) for _ in range(num_layers)])
        x = torch.rand(1, 4, embed_dim, requires_grad=True)
        out = x
        for layer in layers:
            out = layer(out)
        loss = out.mean()
        loss.backward()
        return x.grad.abs().mean().item()

    grad_with_shortcut = measure_input_grad(TransformerBlock)
    grad_without_shortcut = measure_input_grad(BlockNoShortcut)

    print(f"Mean |gradient| at input, WITH shortcuts:    {grad_with_shortcut:.8f}")
    print(f"Mean |gradient| at input, WITHOUT shortcuts: {grad_without_shortcut:.8f}")
    print("With randomly initialized weights the gap isn't always dramatic at only 5 layers,")
    print("but the WITHOUT-shortcut gradient shrinks much faster as depth increases (try num_layers=20).")
    print("This is the vanishing-gradient problem shortcuts are designed to prevent.\n")

    print("=== Key observations ===")
    print("1. Pre-norm: LayerNorm is applied BEFORE attention/feed-forward, not after")
    print("2. Each sub-layer's output is added back to its input (x = x + shortcut)")
    print("3. Output shape == input shape -- this is why blocks can be stacked N times")
    print("4. Shortcuts give gradients a direct path backward, bypassing the nonlinearities")
    print("5. Real MultiHeadAttention from Chapter 2 is reused here, not reimplemented")
