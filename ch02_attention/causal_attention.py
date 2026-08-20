"""
Chapter 2, Stage 3: Causal (Masked) Self-Attention

GPT generates text left-to-right, one token at a time. A token must never
"see" future tokens during training -- that would leak information the model
hasn't generated yet and wouldn't have at inference time.

Fix: mask out the upper triangle of the attention score matrix (positions
after the current token) with -inf BEFORE softmax. Since softmax(-inf) = 0,
those future positions get zero weight -- each token only attends to itself
and prior tokens. Hence "causal" -- attention respects the temporal order of
generation.

Also covered: dropout on attention weights during training (randomly zeroing
some weights, then PyTorch rescales survivors by 1/(1-p) automatically) --
standard regularization, disabled at inference (model.eval()).
"""

import torch
import torch.nn as nn


class CausalSelfAttention(nn.Module):
    """
    Scaled dot-product self-attention with a causal mask and dropout.

    Args:
        d_in: input embedding dimension
        d_out: output (context vector) dimension
        context_length: max sequence length (used to precompute the mask buffer)
        dropout: dropout probability applied to attention weights
        qkv_bias: whether Q/K/V linear layers have a bias term
    """
    def __init__(self, d_in, d_out, context_length, dropout=0.0, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)

        # Precompute the causal mask once. register_buffer -> moves with .to(device)
        # automatically but isn't a trainable parameter.
        # torch.triu with diagonal=1 gives a strict upper triangle of 1s (future positions).
        mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
        self.register_buffer("mask", mask.bool())

    def forward(self, x):
        # x: [batch_size, seq_len, d_in]  (batched version, unlike Stage 1/2 which were unbatched)
        batch_size, seq_len, d_in = x.shape

        queries = self.W_query(x)   # [batch, seq_len, d_out]
        keys = self.W_key(x)
        values = self.W_value(x)

        attn_scores = queries @ keys.transpose(1, 2)   # [batch, seq_len, seq_len]

        # Apply causal mask: set future positions to -inf before softmax.
        # Slice the precomputed mask to the actual sequence length (handles seq_len < context_length).
        attn_scores.masked_fill_(self.mask[:seq_len, :seq_len], -torch.inf)

        d_k = keys.shape[-1]
        attn_weights = torch.softmax(attn_scores / d_k**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)   # regularization (no-op in eval mode)

        context_vec = attn_weights @ values          # [batch, seq_len, d_out]
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

    # Simulate a batch of 2 identical sequences (as the book does, to show batching works)
    batch = torch.stack([inputs, inputs], dim=0)   # [2, 6, 3]

    d_in = inputs.shape[1]
    d_out = 2
    context_length = batch.shape[1]

    print("=== Causal (Masked) Self-Attention ===\n")
    print(f"Batch shape: {batch.shape}  (batch_size=2, seq_len=6, d_in={d_in})\n")

    # --- Show the mask itself first, standalone ---
    print("--- The causal mask ---")
    mask = torch.triu(torch.ones(context_length, context_length), diagonal=1).bool()
    print("True = masked out (future position, set to -inf before softmax):")
    print(mask)
    print()

    # --- Run causal attention, dropout disabled (eval-like: dropout=0.0) ---
    print("--- CausalSelfAttention forward pass (dropout=0.0) ---")
    torch.manual_seed(123)
    causal_attn = CausalSelfAttention(d_in, d_out, context_length, dropout=0.0)
    context_vecs = causal_attn(batch)
    print(f"Output shape: {context_vecs.shape}  (batch, seq_len, d_out)")

    # Recompute the attention weights manually to display them (forward() doesn't return them)
    with torch.no_grad():
        queries = causal_attn.W_query(batch)
        keys = causal_attn.W_key(batch)
        attn_scores = queries @ keys.transpose(1, 2)
        attn_scores.masked_fill_(causal_attn.mask[:context_length, :context_length], -torch.inf)
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)

    print("\nAttention weights for sequence 0 (rows=query token, cols=key token):")
    print(attn_weights[0])
    print("\nNotice: upper triangle is exactly 0.0 -- no attention paid to future tokens.")
    print("Row sums (should still be 1.0 -- masked softmax renormalizes over visible tokens):")
    print(attn_weights[0].sum(dim=-1))

    # --- Demonstrate dropout's effect on attention weights ---
    print("\n--- Dropout on attention weights (training-time regularization) ---")
    torch.manual_seed(123)
    dropout_layer = nn.Dropout(0.5)   # 50% for a visible effect in this small example
    dropout_layer.train()  # ensure training mode (dropout is a no-op in eval mode)
    example_weights = torch.ones(6, 6) * (1 / 6)  # uniform 1/6 weights, sums to 1 per row
    dropped = dropout_layer(example_weights)
    print(f"Before dropout (uniform, each = 1/6 = {1/6:.4f}):\n{example_weights[0]}")
    print(f"After dropout(p=0.5) -- survivors rescaled by 1/(1-0.5)=2x, ~half zeroed:\n{dropped[0]}")
    print("At inference (model.eval()), dropout is disabled -- all weights pass through unchanged.")

    print("\n=== Key observations ===")
    print("1. Mask is precomputed once via register_buffer -- reused every forward pass")
    print("2. masked_fill_(-inf) BEFORE softmax -> those positions become exactly 0 after softmax")
    print("3. Each row still sums to 1 -- softmax renormalizes over only the VISIBLE tokens")
    print("4. Token 0 ('Your') can only attend to itself; token 5 ('step') can attend to all 6")
    print("5. Dropout is applied to attn_weights during training only, disabled at inference")
    print("6. This module is batched ([batch, seq_len, d_in]) -- ready for real DataLoader batches")
