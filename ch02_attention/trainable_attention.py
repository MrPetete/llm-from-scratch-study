"""Self-attention with trainable Q/K/V projections -- the mechanism actually used in GPT."""

import torch
import torch.nn as nn


class SelfAttentionV1(nn.Module):
    """Q/K/V as raw nn.Parameter matrices, so every multiplication is explicit."""
    def __init__(self, d_in, d_out):
        super().__init__()
        self.W_query = nn.Parameter(torch.rand(d_in, d_out))
        self.W_key = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))

    def forward(self, x):
        queries = x @ self.W_query
        keys = x @ self.W_key
        values = x @ self.W_value

        attn_scores = queries @ keys.T
        d_k = keys.shape[-1]
        attn_weights = torch.softmax(attn_scores / d_k**0.5, dim=-1)

        context_vec = attn_weights @ values
        return context_vec


class SelfAttentionV2(nn.Module):
    """Same computation as V1, using nn.Linear for the projections (proper init, optional bias)."""
    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x):
        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)

        attn_scores = queries @ keys.T
        d_k = keys.shape[-1]
        attn_weights = torch.softmax(attn_scores / d_k**0.5, dim=-1)

        context_vec = attn_weights @ values
        return context_vec


if __name__ == "__main__":
    torch.manual_seed(123)

    inputs = torch.tensor([
        [0.43, 0.15, 0.89],  # Your
        [0.55, 0.87, 0.66],  # journey
        [0.57, 0.85, 0.64],  # starts
        [0.22, 0.58, 0.33],  # with
        [0.77, 0.25, 0.10],  # one
        [0.05, 0.80, 0.55],  # step
    ])
    tokens = ["Your", "journey", "starts", "with", "one", "step"]

    d_in = inputs.shape[1]
    d_out = 2

    print("=== Self-Attention with Trainable Weights (Q, K, V) ===\n")
    print(f"Input shape: {inputs.shape}  (seq_len=6, d_in={d_in})")
    print(f"Projecting to d_out={d_out}\n")

    print("--- SelfAttentionV1 (raw nn.Parameter W matrices) ---")
    torch.manual_seed(123)
    sa_v1 = SelfAttentionV1(d_in, d_out)
    context_v1 = sa_v1(inputs)
    print(f"W_query shape: {sa_v1.W_query.shape}")
    print(f"Context vectors shape: {context_v1.shape}")
    print(f"Context vectors:\n{context_v1}\n")

    print("--- SelfAttentionV2 (nn.Linear projections) ---")
    torch.manual_seed(123)
    sa_v2 = SelfAttentionV2(d_in, d_out)
    context_v2 = sa_v2(inputs)
    print(f"Context vectors shape: {context_v2.shape}")
    print(f"Context vectors:\n{context_v2}\n")

    print("--- Why scale by 1/sqrt(d_k)? ---")
    queries = sa_v2.W_query(inputs)
    keys = sa_v2.W_key(inputs)
    raw_scores = queries @ keys.T
    d_k = keys.shape[-1]

    unscaled_weights = torch.softmax(raw_scores, dim=-1)
    scaled_weights = torch.softmax(raw_scores / d_k**0.5, dim=-1)

    print(f"d_k (key dimension) = {d_k}")
    print(f"Raw scores range: [{raw_scores.min():.3f}, {raw_scores.max():.3f}]")
    print(f"Unscaled softmax row 0 (more peaked): {unscaled_weights[0]}")
    print(f"Scaled softmax row 0   (softer):       {scaled_weights[0]}")
    print("Unscaled distribution is sharper (closer to one-hot) -- with larger")
    print("d_k and real embedding sizes (e.g. 768), this effect is much more extreme,")
    print("causing near-zero gradients for all but the top-scoring token.\n")

    print("=== Key observations ===")
    print("1. Q, K, V are three separate learned projections of the same input")
    print("2. attn_scores = Q @ K.T  -- how well each query matches each key")
    print("3. Scaling by 1/sqrt(d_k) keeps softmax gradients healthy as d_k grows")
    print("4. context_vec = attn_weights @ V  -- retrieve values weighted by relevance")
    print("5. Because W_query/W_key/W_value are learned, the model adapts what 'relevant' means")
