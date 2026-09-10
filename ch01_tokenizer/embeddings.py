"""Token + positional embeddings: converts token IDs into dense vectors for the transformer."""

import torch
import torch.nn as nn
import tiktoken


class TokenEmbedding(nn.Module):
    """Token ID -> dense vector. [batch, seq_len] -> [batch, seq_len, embed_dim]."""
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)

    def forward(self, token_ids):
        return self.embedding(token_ids)


class PositionalEmbedding(nn.Module):
    """Learned (not sinusoidal) position embeddings, added to token embeddings."""
    def __init__(self, context_length, embed_dim):
        super().__init__()
        self.pos_embedding = nn.Embedding(context_length, embed_dim)

    def forward(self, token_embeddings):
        batch_size, seq_len, embed_dim = token_embeddings.shape
        positions = torch.arange(seq_len, device=token_embeddings.device)
        pos_embeds = self.pos_embedding(positions)
        return token_embeddings + pos_embeds  # broadcasts over batch dim


class GPTEmbedding(nn.Module):
    """Combined token + positional embedding layer, the first layer of the GPT model."""
    def __init__(self, vocab_size, embed_dim, context_length, dropout=0.1):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Embedding(context_length, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, token_ids):
        batch_size, seq_len = token_ids.shape
        token_embeds = self.token_embedding(token_ids)
        positions = torch.arange(seq_len, device=token_ids.device)
        pos_embeds = self.pos_embedding(positions)
        embeddings = token_embeds + pos_embeds
        embeddings = self.dropout(embeddings)
        return embeddings


if __name__ == "__main__":
    print("=== Token & Positional Embeddings ===\n")

    vocab_size = 50257      # GPT-2 BPE vocab
    embed_dim = 256         # 768 in real GPT-2 base, 256 for this demo
    context_length = 1024
    batch_size = 2
    seq_len = 8

    token_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    print(f"Input token IDs shape: {token_ids.shape}")
    print(f"Sample IDs:\n{token_ids}\n")

    print("=== Test 1: Token Embeddings ===")
    token_emb = TokenEmbedding(vocab_size, embed_dim)
    token_vectors = token_emb(token_ids)
    print(f"Output shape: {token_vectors.shape}")
    print(f"Each token → {embed_dim}-dimensional vector\n")

    print("=== Test 2: Positional Embeddings ===")
    pos_emb = PositionalEmbedding(context_length, embed_dim)
    embeddings_with_pos = pos_emb(token_vectors)
    print(f"Output shape: {embeddings_with_pos.shape}")
    print(f"Position information added to each token\n")

    print("=== Test 3: Complete GPT Embedding ===")
    gpt_embedding = GPTEmbedding(vocab_size, embed_dim, context_length, dropout=0.1)
    final_embeddings = gpt_embedding(token_ids)
    print(f"Input:  {token_ids.shape} (token IDs)")
    print(f"Output: {final_embeddings.shape} (dense vectors with position info)")
    print(f"\nThis output is ready to be fed into transformer blocks!\n")

    print("=== Test 4: Position Matters ===")
    same_token = torch.tensor([[42, 42]])  # same token, two positions
    emb = GPTEmbedding(vocab_size, embed_dim, context_length, dropout=0.0)

    with torch.no_grad():  # disable dropout for this demo
        result = emb.token_embedding(same_token) + emb.pos_embedding(torch.arange(2))

    pos_0_vector = result[0, 0, :5]
    pos_1_vector = result[0, 1, :5]

    print(f"Token ID 42 at position 0 (first 5 dims): {pos_0_vector}")
    print(f"Token ID 42 at position 1 (first 5 dims): {pos_1_vector}")
    print(f"Vectors differ because position embeddings are different\n")

    print("=== Test 5: Real Text Example ===")
    tokenizer = tiktoken.get_encoding("gpt2")
    text = "The quick brown fox jumps"
    token_ids_real = tokenizer.encode(text)
    token_ids_tensor = torch.tensor([token_ids_real])

    print(f"Text: '{text}'")
    print(f"Token IDs: {token_ids_real}")
    print(f"Tensor shape: {token_ids_tensor.shape}")

    gpt_emb = GPTEmbedding(vocab_size, embed_dim, context_length, dropout=0.0)
    embedded = gpt_emb(token_ids_tensor)

    print(f"\nEmbedded shape: {embedded.shape}")
    print(f"Each of the {len(token_ids_real)} tokens → {embed_dim}D vector with position info")
    print(f"\nFirst token 'The' embedding (first 10 dims):")
    print(f"{embedded[0, 0, :10]}")

    print("\n=== Key Takeaways ===")
    print("1. Token embeddings convert IDs → semantic vectors (learned)")
    print("2. Positional embeddings add position info (learned, not sinusoidal)")
    print("3. Final embedding = token_emb + pos_emb")
    print("4. Same token at different positions → different final vectors")
    print("5. Output shape [batch, seq_len, embed_dim] feeds into transformer blocks")
