"""
Instruction fine-tuning dataset preparation.

Downloads 1,100 instruction-response pairs, formats with Alpaca template,
custom collate handles padding and loss masking.
"""

import json
import os
import urllib.request
import torch
from torch.utils.data import Dataset


def download_and_load_instruction_data(url, file_path):
    if not os.path.exists(file_path):
        print(f"Downloading {url}...")
        with urllib.request.urlopen(url) as response:
            text_data = response.read().decode("utf-8")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(text_data)
    else:
        print(f"{file_path} already exists. Skipping download.")

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data


def format_input(entry):
    """Build Alpaca-style prompt from instruction + optional input."""
    instruction_text = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )

    input_text = f"\n\n### Input:\n{entry['input']}" if entry.get("input") else ""

    return instruction_text + input_text


def format_target(entry):
    return f"\n\n### Response:\n{entry['output']}"


class InstructionDataset(Dataset):
    """Pre-tokenize instruction-response pairs."""

    def __init__(self, data, tokenizer):
        self.data = data
        self.encoded_texts = []

        for entry in data:
            full_text = format_input(entry) + format_target(entry)
            self.encoded_texts.append(tokenizer.encode(full_text))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.encoded_texts[idx]


def custom_collate_fn(batch, pad_token_id=50256, ignore_index=-100,
                       allowed_max_length=None, device="cpu"):
    """
    Pad variable-length sequences, create shifted targets, mask padding.
    
    Masks all but the first padding token with -100 so cross_entropy ignores them.
    """
    batch_max_length = max(len(item) + 1 for item in batch)

    inputs_lst, targets_lst = [], []

    for item in batch:
        new_item = item.copy()
        new_item += [pad_token_id]

        padded = new_item + [pad_token_id] * (batch_max_length - len(new_item))

        inputs = torch.tensor(padded[:-1])
        targets = torch.tensor(padded[1:])

        # Mask all but the first padding token
        mask = targets == pad_token_id
        indices = torch.nonzero(mask).squeeze()
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index

        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]

        inputs_lst.append(inputs)
        targets_lst.append(targets)

    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)

    return inputs_tensor, targets_tensor


if __name__ == "__main__":
    import tiktoken
    from functools import partial
    from torch.utils.data import DataLoader

    print("=== Chapter 7, Stage 1: Dataset Preparation ===\n")

    url = ("https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/"
           "ch07/01_main-chapter-code/instruction-data.json")
    file_path = "instruction-data.json"

    data = download_and_load_instruction_data(url, file_path)
    print(f"Number of entries: {len(data)}\n")

    print("--- Example entry (raw) ---")
    print(json.dumps(data[50], indent=2))
    print()

    print("--- Example entry (formatted with Alpaca template) ---")
    formatted = format_input(data[50]) + format_target(data[50])
    print(formatted)
    print()

    no_input_entries = [e for e in data if not e.get("input")]
    print(f"Entries with no separate 'input' field: {len(no_input_entries)} / {len(data)}")
    print("--- Example (instruction-only, no ### Input: section) ---")
    print(format_input(no_input_entries[0]) + format_target(no_input_entries[0]))
    print()

    # Split 85/5/10
    train_portion = int(len(data) * 0.85)
    test_portion = int(len(data) * 0.1)
    val_portion = len(data) - train_portion - test_portion

    train_data = data[:train_portion]
    test_data = data[train_portion:train_portion + test_portion]
    val_data = data[train_portion + test_portion:]

    print(f"Split: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}\n")

    tokenizer = tiktoken.get_encoding("gpt2")

    train_dataset = InstructionDataset(train_data, tokenizer)
    val_dataset = InstructionDataset(val_data, tokenizer)

    device = "cpu"
    customized_collate_fn = partial(custom_collate_fn, device=device, allowed_max_length=1024)

    train_loader = DataLoader(
        train_dataset, batch_size=8, shuffle=True, drop_last=True,
        collate_fn=customized_collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=8, shuffle=False, drop_last=False,
        collate_fn=customized_collate_fn
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches:   {len(val_loader)}\n")

    print("--- Sample batch from train_loader ---")
    inputs, targets = next(iter(train_loader))
    print(f"Input shape:  {inputs.shape}")
    print(f"Target shape: {targets.shape}")

    print(f"\nFirst input (first 30 tokens):  {inputs[0, :30].tolist()}")
    print(f"First target (first 30 tokens): {targets[0, :30].tolist()}")
    print(f"\nDecoded first input (first 100 chars):")
    print(repr(tokenizer.decode(inputs[0].tolist())[:100]))

    num_masked = (targets[0] == -100).sum().item()
    print(f"\nNumber of masked (-100) positions in first target: {num_masked}")

    print("\n=== Stage 1 complete ===")
    print(f"Downloaded {len(data)} instruction-response pairs, formatted with Alpaca template,")
    print("and built dataloaders with custom collate/masking.")
