"""
Transformer block: attention + feed-forward, each with LayerNorm and a residual
shortcut (GPT-2's pre-norm arrangement). Stacked num_layers times to build the model.
"""

import os
import sys
import torch
import torch.nn as nn

# Reuse the Chapter 2 MultiHeadAttention implementation instead of duplicating it
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
        shortcut = x
        x = self.norm1(x)
        x = self.attention(x)
        x = self.dropout_shortcut(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.feed_forward(x)
        x = self.dropout_shortcut(x)
        x = x + shortcut

        return x


if __name__ == "__main__":
    torch.manual_seed(123)

    from config import GPT_CONFIG_TINY

    print("=== Transformer Block ===\n")

    block = TransformerBlock(GPT_CONFIG_TINY)
    embed_dim = GPT_CONFIG_TINY["embed_dim"]

    batch = torch.rand(2, 4, embed_dim)
    print(f"Input shape:  {batch.shape}")

    out = block(batch)
    print(f"Output shape: {out.shape}\n")

    print("--- Shortcut connections and gradient flow ---")
    print("Comparing gradient magnitude at the input, with vs without shortcuts,")
    print("through a 5-layer stack.\n")

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
            x = self.attention(self.norm1(x))
            x = self.feed_forward(self.norm2(x))
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
    print("The gap grows with depth -- this is the vanishing-gradient problem shortcuts prevent.\n")

    print("=== Key observations ===")
    print("1. Pre-norm: LayerNorm applied before attention/feed-forward, not after")
    print("2. Each sub-layer's output is added back to its input")
    print("3. Output shape == input shape, so blocks can be stacked N times")
    print("4. Shortcuts give gradients a direct path backward")
