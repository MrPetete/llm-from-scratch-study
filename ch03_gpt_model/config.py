"""GPT model configurations -- same architecture, different sizes."""

GPT_CONFIG_124M = {
    "vocab_size": 50257,     # tiktoken gpt2 BPE vocab
    "context_length": 1024,
    "embed_dim": 768,
    "num_heads": 12,
    "num_layers": 12,
    "dropout": 0.1,
    "qkv_bias": False,
}

# GPT-2 medium (355M) -- used in Chapter 7 for instruction fine-tuning
GPT_CONFIG_355M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "embed_dim": 1024,
    "num_heads": 16,
    "num_layers": 24,
    "dropout": 0.1,
    "qkv_bias": False,
}

# Small config for fast local CPU experimentation
GPT_CONFIG_TINY = {
    "vocab_size": 50257,
    "context_length": 128,
    "embed_dim": 64,
    "num_heads": 4,
    "num_layers": 2,
    "dropout": 0.1,
    "qkv_bias": False,
}
