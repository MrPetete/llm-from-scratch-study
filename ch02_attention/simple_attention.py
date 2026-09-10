"""Simplified self-attention: dot-product scores + softmax + weighted sum, no trainable weights yet."""

import torch


def compute_attention_scores_single_query(inputs, query_idx):
    """Dot-product score of one query token against every token (incl. itself). Returns [seq_len]."""
    query = inputs[query_idx]
    scores = torch.empty(inputs.shape[0])
    for i, x_i in enumerate(inputs):
        scores[i] = torch.dot(query, x_i)
    return scores


def compute_all_attention_scores(inputs):
    """Vectorized scores for every token as query at once. scores[i, j] = similarity of token i to token j."""
    return inputs @ inputs.T


def compute_context_vectors(inputs):
    """Full simplified self-attention: scores -> softmax weights -> weighted sum of inputs."""
    attn_scores = compute_all_attention_scores(inputs)
    attn_weights = torch.softmax(attn_scores, dim=-1)
    context_vectors = attn_weights @ inputs
    return context_vectors, attn_weights


if __name__ == "__main__":
    torch.manual_seed(123)

    # "Your journey starts with one step", 3-dim embeddings (book uses this size for hand-checking)
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

    query_idx = 1
    print(f"--- Single query walkthrough: token '{tokens[query_idx]}' (idx {query_idx}) ---")
    scores = compute_attention_scores_single_query(inputs, query_idx)
    print(f"Raw attention scores (dot products): {scores}")

    weights = torch.softmax(scores, dim=0)
    print(f"Attention weights (softmax, sum={weights.sum():.4f}): {weights}")

    context = weights @ inputs
    print(f"Context vector for '{tokens[query_idx]}': {context}\n")

    print("--- Full attention matrix (every token as query) ---")
    context_vectors, attn_weights = compute_context_vectors(inputs)

    print("Attention weights matrix (row i = weights when token i is the query):")
    print(attn_weights)
    print(f"\nEach row sums to 1: {attn_weights.sum(dim=-1)}")

    print(f"\nContext vectors shape: {context_vectors.shape}  (one context vector per token)")
    print("Context vectors:")
    print(context_vectors)

    print(f"\nSanity check -- single-query context for 'journey' matches full-matrix row 1: "
          f"{torch.allclose(context, context_vectors[query_idx])}")

    print("\n=== Key observations ===")
    print("1. Attention weights are a probability distribution over tokens (softmax, sum=1)")
    print("2. A token attends to itself too (diagonal isn't zero) -- often a large weight")
    print("3. Context vector = weighted blend of ALL token embeddings, weighted by relevance")
    print("4. No learned parameters yet -- relevance is purely raw embedding similarity")
    print("5. Next stage: replace raw embeddings with learned Q/K/V projections")
