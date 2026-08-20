"""
Step 4: Sliding-window Dataset and DataLoader for next-token prediction training

This module creates (input, target) pairs from tokenized text using a sliding window.
For a GPT model, we train on sequences where the model sees the first N tokens
and must predict the N+1th token.

Example with context_length=4:
  Text: "The quick brown fox jumps over the lazy dog"
  Tokenized: [0, 1, 2, 3, 4, 5, 6, 7, 8]
  
  Window 1: input=[0,1,2,3], target=[1,2,3,4]
  Window 2: input=[1,2,3,4], target=[2,3,4,5]  (stride=1)
  ...
  
  Note: target is shifted by 1 position (next-token prediction).
  Each position in the input predicts the next token.
"""

import torch
from torch.utils.data import Dataset, DataLoader


class GPTDatasetV1(Dataset):
    """
    Sliding-window dataset for GPT training.
    
    Args:
        text: Raw input text
        tokenizer: A tokenizer with .encode() method (e.g., tiktoken tokenizer)
        context_length: Max sequence length (GPT's context window)
        stride: How many positions to slide the window (stride=1 means overlapping windows)
    """
    def __init__(self, text, tokenizer, context_length, stride):
        self.tokenizer = tokenizer
        self.context_length = context_length
        self.stride = stride
        
        # Tokenize the entire text once
        self.token_ids = tokenizer.encode(text)
        
    def __len__(self):
        # How many complete windows can we extract?
        # We need context_length tokens for input + 1 more for target
        return (len(self.token_ids) - self.context_length) // self.stride
    
    def __getitem__(self, idx):
        # Starting position of this window
        start_idx = idx * self.stride
        end_idx = start_idx + self.context_length
        
        # Input: tokens [start:end]
        # Target: tokens [start+1:end+1] (shifted by 1)
        input_chunk = self.token_ids[start_idx:end_idx]
        target_chunk = self.token_ids[start_idx + 1:end_idx + 1]
        
        return torch.tensor(input_chunk), torch.tensor(target_chunk)


def create_dataloader_v1(text, batch_size=4, context_length=256, 
                         stride=128, shuffle=True, drop_last=True, 
                         num_workers=0):
    """
    Create a DataLoader for GPT training.
    
    Args:
        text: Raw input text
        batch_size: Number of sequences per batch
        context_length: Max sequence length (GPT's context window)
        stride: Sliding window stride
        shuffle: Whether to shuffle batches (typically True for training)
        drop_last: Drop the last incomplete batch (recommended for training)
        num_workers: Number of worker processes (0 = main process only)
    
    Returns:
        DataLoader yielding (input_batch, target_batch) of shape [batch_size, context_length]
    """
    import tiktoken
    tokenizer = tiktoken.get_encoding("gpt2")
    
    dataset = GPTDatasetV1(text, tokenizer, context_length, stride)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )
    
    return dataloader


if __name__ == "__main__":
    import tiktoken
    
    # Test with a short story excerpt
    with open("ch01_tokenizer/data/the-verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()
    
    print("=== GPT DataLoader (Sliding Window) ===\n")
    
    # Configuration
    context_length = 8  # Small for demo (real GPT-2 uses 1024)
    batch_size = 2
    stride = 4  # Overlapping windows
    
    tokenizer = tiktoken.get_encoding("gpt2")
    
    # Create dataset
    dataset = GPTDatasetV1(raw_text, tokenizer, context_length, stride)
    
    print(f"Text length: {len(raw_text)} characters")
    print(f"Token count: {len(dataset.token_ids)} tokens")
    print(f"Context length: {context_length} tokens")
    print(f"Stride: {stride} tokens")
    print(f"Total windows: {len(dataset)}\n")
    
    # Show first 3 windows
    print("First 3 windows (input → target):\n")
    for i in range(3):
        input_ids, target_ids = dataset[i]
        print(f"Window {i}:")
        print(f"  Input:  {input_ids.tolist()}")
        print(f"  Target: {target_ids.tolist()}")
        
        # Decode to show what text these IDs represent
        input_text = tokenizer.decode(input_ids.tolist())
        target_text = tokenizer.decode(target_ids.tolist())
        print(f"  Input text:  {repr(input_text)}")
        print(f"  Target text: {repr(target_text)}")
        print()
    
    # Create DataLoader and show batches
    dataloader = create_dataloader_v1(
        raw_text,
        batch_size=batch_size,
        context_length=context_length,
        stride=stride,
        shuffle=False,  # Don't shuffle for demo (want to see sequential windows)
        drop_last=False
    )
    
    print(f"\n=== DataLoader batches (batch_size={batch_size}) ===\n")
    
    # Show first 2 batches
    for batch_idx, (input_batch, target_batch) in enumerate(dataloader):
        if batch_idx >= 2:
            break
        
        print(f"Batch {batch_idx}:")
        print(f"  Input shape:  {input_batch.shape}  (batch_size, context_length)")
        print(f"  Target shape: {target_batch.shape}")
        print(f"  Input batch:\n{input_batch}")
        print(f"  Target batch:\n{target_batch}")
        print()
    
    print("\n=== Key observations ===")
    print("1. Target is input shifted by 1 position (next-token prediction)")
    print("2. Each token in input predicts the next token in target")
    print("3. With stride < context_length, windows overlap (data efficiency)")
    print("4. DataLoader batches multiple windows together for parallel training")
    print(f"\nTotal batches: {len(dataloader)}")
    print(f"Tokens per batch: {batch_size * context_length}")
