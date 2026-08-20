"""
Chapter 2, Stage 1: Simplified Self-Attention (no trainable weights)

The core intuition before any learned parameters are involved. For a chosen
"query" token, we score it against every other token via dot product, turn
those scores into weights via softmax, and take a weighted sum of the value
vectors (here, just the raw embeddings) to get a context vector.

Doing this for every token as query simultaneously gives a full attention
matrix and a full set of context vectors -- one enriched representation per
input token.
"""

import torch


def compute_attention_scores_single_query(inputs, query_idx):
    """
    Attention scores for ONE query token against every token (incl. itself).

    Args:
        inputs: [seq_len, embed_dim] -- embeddings for the whole sequence
        query_idx: index of the token acting as the query

    Returns:
        scores: [seq_len] -- raw dot-product similarity to every token
    """
    query = inputs[query_idx]          # [embed_dim]
    scores = torch.empty(inputs.shape[0])
    for i, x_i in enumerate(inputs):
        scores[i] = torch.dot(query, x_i)   # similarity measure
    return scores


def compute_all_attention_scores(inputs):
    """
    Vectorized version: attention scores for EVERY token as query, at once.

    Args:
        inputs: [seq_len, embed_dim]

    Returns:
        scores: [seq_len, seq_len] -- scores[i, j] = similarity of token i (query) to token j
    """
    # [seq_len, embed_dim] @ [embed_dim, seq_len] -> [seq_len, seq_len]
    return inputs @ inputs.T


def compute_context_vectors(inputs):
    """
    Full simplified self-attention: scores -> softmax weights -> weighted sum.

    Args:
        inputs: [seq_len, embed_dim]

    Returns:
        context_vectors: [seq_len, embed_dim] -- one enriched vector per input token
        attn_weights: [seq_len, seq_len] -- the normalized attention weight matrix
    """
    attn_scores = compute_all_attention_scores(inputs)          # [seq_len, seq_len]
    attn_weights = torch.softmax(attn_scores, dim=-1)           # normalize each row to sum to 1
    context_vectors = attn_weights @ inputs                     # [seq_len, seq_len] @ [seq_len, embed_dim]
    return context_vectors, attn_weights


if __name__ == "__main__":
    torch.manual_seed(123)

    # Toy sentence, 3-dim embeddings (book uses this exact size to hand-check the math)
    # Sentence: "Your journey starts with one step"
    inputs = torch.tensor([
        [0.43, 0.15, 0.89],  # Your
        [0.55, 0.87, 0.66],  # journey
        [0.57, 0.85, 0.64],  # starts
        [0.22, 0.58, 0.33],  # with
        [0.77, 0.25, 0.10],  # one
        [0.05, 0.80, 0.55],  # step
    ])
    tokens = ["Your", "journey", "starts", "with", "one", "step"]

    print("=== Simplified Self-Attention (no trainable weights) ===\n")
    print(f"Input embeddings shape: {inputs.shape}  (seq_len=6, embed_dim=3)\n")

    # --- Single query walkthrough: "journey" (index 1) ---
    query_idx = 1
    print(f"--- Single query walkthrough: token '{tokens[query_idx]}' (idx {query_idx}) ---")
    scores = compute_attention_scores_single_query(inputs, query_idx)
    print(f"Raw attention scores (dot products): {scores}")

    weights = torch.softmax(scores, dim=0)
    print(f"Attention weights (softmax, sum={weights.sum():.4f}): {weights}")

    context = weights @ inputs
    print(f"Context vector for '{tokens[query_idx]}': {context}\n")

    # --- Full matrix: every token as query simultaneously ---
    print("--- Full attention matrix (every token as query) ---")
    context_vectors, attn_weights = compute_context_vectors(inputs)

    print("Attention weights matrix (row i = weights when token i is the query):")
    print(attn_weights)
    print(f"\nEach row sums to 1: {attn_weights.sum(dim=-1)}")

    print(f"\nContext vectors shape: {context_vectors.shape}  (one context vector per token)")
    print("Context vectors:")
    print(context_vectors)

    # Sanity check: single-query result should match row 1 of the full matrix
    print(f"\nSanity check -- single-query context for 'journey' matches full-matrix row 1: "
          f"{torch.allclose(context, context_vectors[query_idx])}")

    print("\n=== Key observations ===")
    print("1. Attention weights are a probability distribution over tokens (softmax, sum=1)")
    print("2. A token attends to itself too (diagonal isn't zero) -- often a large weight")
    print("3. Context vector = weighted blend of ALL token embeddings, weighted by relevance")
    print("4. No learned parameters yet -- relevance is purely raw embedding similarity")
    print("5. Next stage: replace raw embeddings with learned Q/K/V projections")
