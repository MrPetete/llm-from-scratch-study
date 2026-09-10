"""Sliding-window Dataset/DataLoader that turns tokenized text into (input, target) pairs for next-token prediction."""

import torch
from torch.utils.data import Dataset, DataLoader


class GPTDatasetV1(Dataset):
    """Sliding-window dataset for GPT training. stride=1 gives overlapping windows."""
    def __init__(self, text, tokenizer, context_length, stride):
        self.tokenizer = tokenizer
        self.context_length = context_length
        self.stride = stride
        self.token_ids = tokenizer.encode(text)

    def __len__(self):
        return (len(self.token_ids) - self.context_length) // self.stride

    def __getitem__(self, idx):
        start_idx = idx * self.stride
        end_idx = start_idx + self.context_length

        # target is the input shifted by one position (next-token prediction)
        input_chunk = self.token_ids[start_idx:end_idx]
        target_chunk = self.token_ids[start_idx + 1:end_idx + 1]

        return torch.tensor(input_chunk), torch.tensor(target_chunk)


def create_dataloader_v1(text, batch_size=4, context_length=256,
                         stride=128, shuffle=True, drop_last=True,
                         num_workers=0):
    """Build a GPT-2-tokenized DataLoader yielding (input_batch, target_batch) of shape [batch_size, context_length]."""
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

    with open("ch01_tokenizer/data/the-verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()

    print("=== GPT DataLoader (Sliding Window) ===\n")

    context_length = 8  # small for demo, real GPT-2 uses 1024
    batch_size = 2
    stride = 4  # overlapping windows

    tokenizer = tiktoken.get_encoding("gpt2")

    dataset = GPTDatasetV1(raw_text, tokenizer, context_length, stride)

    print(f"Text length: {len(raw_text)} characters")
    print(f"Token count: {len(dataset.token_ids)} tokens")
    print(f"Context length: {context_length} tokens")
    print(f"Stride: {stride} tokens")
    print(f"Total windows: {len(dataset)}\n")

    print("First 3 windows (input → target):\n")
    for i in range(3):
        input_ids, target_ids = dataset[i]
        print(f"Window {i}:")
        print(f"  Input:  {input_ids.tolist()}")
        print(f"  Target: {target_ids.tolist()}")

        input_text = tokenizer.decode(input_ids.tolist())
        target_text = tokenizer.decode(target_ids.tolist())
        print(f"  Input text:  {repr(input_text)}")
        print(f"  Target text: {repr(target_text)}")
        print()

    dataloader = create_dataloader_v1(
        raw_text,
        batch_size=batch_size,
        context_length=context_length,
        stride=stride,
        shuffle=False,  # keep windows sequential for the demo
        drop_last=False
    )

    print(f"\n=== DataLoader batches (batch_size={batch_size}) ===\n")

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
