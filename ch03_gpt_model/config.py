"""
GPT model configuration.

The same architecture scales from GPT-2 small (124M params) up to XL just by
changing these numbers -- no structural code changes needed.
"""

GPT_CONFIG_124M = {
    "vocab_size": 50257,     # BPE vocabulary size (tiktoken gpt2 encoding)
    "context_length": 1024,  # max sequence length the model can process
    "embed_dim": 768,        # embedding dimension
    "num_heads": 12,         # number of attention heads
    "num_layers": 12,        # number of transformer blocks stacked
    "dropout": 0.1,          # dropout rate (embeddings, attention, feed-forward)
    "qkv_bias": False,       # whether Q/K/V linear layers have a bias term
}

# A much smaller config for fast local experimentation on CPU -- same code,
# just tiny numbers so forward/backward passes run in milliseconds.
GPT_CONFIG_TINY = {
    "vocab_size": 50257,
    "context_length": 128,
    "embed_dim": 64,
    "num_heads": 4,
    "num_layers": 2,
    "dropout": 0.1,
    "qkv_bias": False,
}
