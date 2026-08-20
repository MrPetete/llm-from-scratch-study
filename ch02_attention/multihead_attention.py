"""
Chapter 2, Stage 4: Multi-Head Attention

Instead of one Q/K/V projection producing one attention pattern, split the
output dimension into multiple parallel "heads," each with its own smaller
Q/K/V. Each head can specialize in a different kind of relationship (e.g.
one head might track syntactic dependency, another topical relevance). Their
outputs are concatenated and passed through a final linear layer (out_proj).

Practically: d_out is split into num_heads x head_dim. Attention is computed
per head in parallel (via an extra tensor dimension, not a Python loop), then
heads are combined back into a single vector per token.

This module is literally what feeds into the transformer block in Chapter 3.
"""

import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    """
    Multi-head causal self-attention, computed efficiently via reshaping
    instead of looping over separate single-head modules.

    Args:
        d_in: input embedding dimension
        d_out: total output dimension across all heads (must be divisible by num_heads)
        context_length: max sequence length (for the causal mask buffer)
        num_heads: number of parallel attention heads
        dropout: dropout probability on attention weights
        qkv_bias: whether Q/K/V linear layers have a bias term
    """
    def __init__(self, d_in, d_out, context_length, num_heads, dropout=0.0, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads   # dimension of each individual head

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

        # Combines the concatenated head outputs back into one d_out-dim vector.
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)

        mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
        self.register_buffer("mask", mask.bool())

    def forward(self, x):
        batch_size, seq_len, d_in = x.shape

        queries = self.W_query(x)   # [batch, seq_len, d_out]
        keys = self.W_key(x)
        values = self.W_value(x)

        # Split d_out into (num_heads, head_dim), then move num_heads before seq_len
        # so each head is processed as an independent batch dimension.
        # [batch, seq_len, d_out] -> [batch, seq_len, num_heads, head_dim] -> [batch, num_heads, seq_len, head_dim]
        queries = queries.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        keys = keys.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        values = values.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention scores per head: [batch, num_heads, seq_len, seq_len]
        attn_scores = queries @ keys.transpose(2, 3)
        attn_scores.masked_fill_(self.mask[:seq_len, :seq_len], -torch.inf)

        attn_weights = torch.softmax(attn_scores / self.head_dim**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # [batch, num_heads, seq_len, seq_len] @ [batch, num_heads, seq_len, head_dim]
        # -> [batch, num_heads, seq_len, head_dim]
        context_vec = attn_weights @ values

        # Recombine heads: [batch, num_heads, seq_len, head_dim] -> [batch, seq_len, num_heads, head_dim]
        # -> [batch, seq_len, d_out] (concatenate heads back into one vector per token)
        context_vec = context_vec.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_out)

        context_vec = self.out_proj(context_vec)   # final linear mix of the concatenated heads
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
    batch = torch.stack([inputs, inputs], dim=0)   # [2, 6, 3]

    d_in = inputs.shape[1]
    d_out = 4          # total output dim across all heads
    num_heads = 2       # so each head has head_dim = 2
    context_length = batch.shape[1]

    print("=== Multi-Head Attention ===\n")
    print(f"Batch shape: {batch.shape}  (batch_size=2, seq_len=6, d_in={d_in})")
    print(f"d_out={d_out}, num_heads={num_heads} -> head_dim={d_out // num_heads}\n")

    torch.manual_seed(123)
    mha = MultiHeadAttention(d_in, d_out, context_length, num_heads, dropout=0.0)
    context_vecs = mha(batch)

    print(f"Output shape: {context_vecs.shape}  (batch, seq_len, d_out)")
    print("Output:")
    print(context_vecs)

    # --- Show the shape transformations explicitly, step by step ---
    print("\n--- Shape walkthrough ---")
    with torch.no_grad():
        q = mha.W_query(batch)
        print(f"1. Q after linear projection:        {q.shape}  (batch, seq_len, d_out)")
        q_split = q.view(2, 6, num_heads, mha.head_dim)
        print(f"2. Q split into heads:                {q_split.shape}  (batch, seq_len, num_heads, head_dim)")
        q_transposed = q_split.transpose(1, 2)
        print(f"3. Q transposed (heads as batch dim): {q_transposed.shape}  (batch, num_heads, seq_len, head_dim)")
        print("   -> attention now computed independently per head, in parallel")

    # --- Confirm each head produces a genuinely different attention pattern ---
    print("\n--- Do the two heads attend differently? ---")
    with torch.no_grad():
        queries = mha.W_query(batch).view(2, 6, num_heads, mha.head_dim).transpose(1, 2)
        keys = mha.W_key(batch).view(2, 6, num_heads, mha.head_dim).transpose(1, 2)
        attn_scores = queries @ keys.transpose(2, 3)
        attn_scores.masked_fill_(mha.mask[:6, :6], -torch.inf)
        attn_weights = torch.softmax(attn_scores / mha.head_dim**0.5, dim=-1)

    print(f"Head 0 attention weights (seq 0, last row -- token 'step' attending to all 6):")
    print(attn_weights[0, 0, -1])
    print(f"Head 1 attention weights (seq 0, last row):")
    print(attn_weights[0, 1, -1])
    print(f"Heads differ: {not torch.allclose(attn_weights[0, 0], attn_weights[0, 1])}")
    print("(Expected -- each head has its own W_query/W_key slice, so different Q/K -> different scores)")

    print("\n=== Key observations ===")
    print("1. d_out is split into num_heads x head_dim (here: 4 = 2 heads x 2 dims each)")
    print("2. transpose(1, 2) moves num_heads next to batch -- lets one matmul do all heads at once")
    print("3. Causal mask + scaling applied identically per head, just like single-head attention")
    print("4. Heads are concatenated back (.view after transpose) then mixed by out_proj")
    print("5. Different heads learn different attention patterns from the same input")
    print("6. This IS the attention sub-layer used inside each transformer block in Chapter 3")
