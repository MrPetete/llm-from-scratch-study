"""
Step 5: Token Embeddings + Positional Embeddings

This module converts token IDs (integers) into dense vectors that transformers process.
Two types of embeddings are combined:

1. Token embeddings: Learn a vector representation for each token in the vocabulary
   - "cat" and "dog" should have similar vectors (semantic similarity)
   - "cat" and "computer" should be far apart
   
2. Positional embeddings: Add information about the token's position in the sequence
   - Position 0 vs position 50 should be treated differently
   - Word order matters: "dog bites man" ≠ "man bites dog"

Final input to transformer = token_embedding + positional_embedding

GPT uses learned positional embeddings (not sinusoidal like original Transformer).
"""

import torch
import torch.nn as nn
import tiktoken


class TokenEmbedding(nn.Module):
    """
    Converts token IDs to dense vectors.
    
    Args:
        vocab_size: Size of the vocabulary (50257 for GPT-2)
        embed_dim: Dimension of embedding vectors (768 for GPT-2 base)
    
    Shape:
        Input: [batch_size, seq_len] (token IDs)
        Output: [batch_size, seq_len, embed_dim] (dense vectors)
    """
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
    
    def forward(self, token_ids):
        # token_ids: [batch_size, seq_len]
        # output: [batch_size, seq_len, embed_dim]
        return self.embedding(token_ids)


class PositionalEmbedding(nn.Module):
    """
    Adds learned positional information to embeddings.
    
    Args:
        context_length: Maximum sequence length (1024 for GPT-2)
        embed_dim: Dimension of embedding vectors (must match token embeddings)
    
    Shape:
        Input: [batch_size, seq_len, embed_dim]
        Output: [batch_size, seq_len, embed_dim] (with position info added)
    """
    def __init__(self, context_length, embed_dim):
        super().__init__()
        # Learnable position embeddings for each position [0, context_length)
        self.pos_embedding = nn.Embedding(context_length, embed_dim)
    
    def forward(self, token_embeddings):
        # token_embeddings: [batch_size, seq_len, embed_dim]
        batch_size, seq_len, embed_dim = token_embeddings.shape
        
        # Create position indices [0, 1, 2, ..., seq_len-1]
        positions = torch.arange(seq_len, device=token_embeddings.device)
        
        # Get position embeddings: [seq_len, embed_dim]
        pos_embeds = self.pos_embedding(positions)
        
        # Add to token embeddings (broadcasting handles batch dimension)
        # token_embeddings: [batch_size, seq_len, embed_dim]
        # pos_embeds:       [seq_len, embed_dim]
        # result:           [batch_size, seq_len, embed_dim]
        return token_embeddings + pos_embeds


class GPTEmbedding(nn.Module):
    """
    Complete embedding layer for GPT: token embeddings + positional embeddings.
    
    This is the first layer of the GPT model. It converts token IDs to dense vectors
    with position information, ready to be processed by transformer blocks.
    
    Args:
        vocab_size: Size of the vocabulary
        embed_dim: Dimension of embedding vectors
        context_length: Maximum sequence length
        dropout: Dropout rate (regularization)
    """
    def __init__(self, vocab_size, embed_dim, context_length, dropout=0.1):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Embedding(context_length, embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, token_ids):
        # token_ids: [batch_size, seq_len]
        batch_size, seq_len = token_ids.shape
        
        # Token embeddings: [batch_size, seq_len, embed_dim]
        token_embeds = self.token_embedding(token_ids)
        
        # Position indices: [seq_len]
        positions = torch.arange(seq_len, device=token_ids.device)
        
        # Position embeddings: [seq_len, embed_dim]
        pos_embeds = self.pos_embedding(positions)
        
        # Combine (broadcasting adds pos_embeds to each sequence in batch)
        # [batch_size, seq_len, embed_dim] + [seq_len, embed_dim]
        embeddings = token_embeds + pos_embeds
        
        # Apply dropout (regularization during training)
        embeddings = self.dropout(embeddings)
        
        return embeddings


if __name__ == "__main__":
    print("=== Token & Positional Embeddings ===\n")
    
    # GPT-2 configuration (scaled down for demo)
    vocab_size = 50257      # GPT-2 BPE vocabulary
    embed_dim = 256         # 768 in GPT-2 base, using 256 for demo
    context_length = 1024   # Max sequence length
    batch_size = 2
    seq_len = 8
    
    # Create sample token IDs
    token_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    print(f"Input token IDs shape: {token_ids.shape}")
    print(f"Sample IDs:\n{token_ids}\n")
    
    # Test 1: Token embeddings only
    print("=== Test 1: Token Embeddings ===")
    token_emb = TokenEmbedding(vocab_size, embed_dim)
    token_vectors = token_emb(token_ids)
    print(f"Output shape: {token_vectors.shape}")
    print(f"Each token → {embed_dim}-dimensional vector\n")
    
    # Test 2: Add positional embeddings
    print("=== Test 2: Positional Embeddings ===")
    pos_emb = PositionalEmbedding(context_length, embed_dim)
    embeddings_with_pos = pos_emb(token_vectors)
    print(f"Output shape: {embeddings_with_pos.shape}")
    print(f"Position information added to each token\n")
    
    # Test 3: Complete GPT embedding layer
    print("=== Test 3: Complete GPT Embedding ===")
    gpt_embedding = GPTEmbedding(vocab_size, embed_dim, context_length, dropout=0.1)
    final_embeddings = gpt_embedding(token_ids)
    print(f"Input:  {token_ids.shape} (token IDs)")
    print(f"Output: {final_embeddings.shape} (dense vectors with position info)")
    print(f"\nThis output is ready to be fed into transformer blocks!\n")
    
    # Test 4: Demonstrate position matters
    print("=== Test 4: Position Matters ===")
    # Same token at different positions gets different final vectors
    same_token = torch.tensor([[42, 42]])  # Same token repeated
    emb = GPTEmbedding(vocab_size, embed_dim, context_length, dropout=0.0)
    
    with torch.no_grad():  # Disable dropout for this demo
        result = emb.token_embedding(same_token) + emb.pos_embedding(torch.arange(2))
    
    pos_0_vector = result[0, 0, :5]  # First 5 dims of position 0
    pos_1_vector = result[0, 1, :5]  # First 5 dims of position 1
    
    print(f"Token ID 42 at position 0 (first 5 dims): {pos_0_vector}")
    print(f"Token ID 42 at position 1 (first 5 dims): {pos_1_vector}")
    print(f"Vectors differ because position embeddings are different\n")
    
    # Test 5: Real text example
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
