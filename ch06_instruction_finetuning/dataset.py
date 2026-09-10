"""
Chapter 7, Stage 1: Dataset Preparation (Steps 1-3)

Step 1: Download the instruction dataset (1,100 instruction->response pairs,
        purpose-built for the book)
Step 2: Format each entry with the Alpaca prompt template:
        ### Instruction: / ### Input: (optional) / ### Response:
Step 3: Custom collate function -- pads variable-length sequences, creates
        shifted targets, masks padding with -100 so it's ignored by the loss

This is fundamentally different from Chapter 6's classification task: instead
of a single fixed label (spam/ham), the model now needs to generate a
FREE-FORM TEXT RESPONSE to an arbitrary natural-language instruction. That
means we're back to next-token-prediction loss (like Chapter 4's pretraining),
but now trained specifically on instruction-following examples.
"""

import json
import os
import urllib.request
import torch
from torch.utils.data import Dataset


def download_and_load_instruction_data(url, file_path):
    """Download the instruction dataset (JSON) if not already present."""
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
    """
    Build the Alpaca-style prompt from an instruction (+ optional input).

    Format:
        Below is an instruction that describes a task. Write a response
        that appropriately completes the request.

        ### Instruction:
        <instruction text>

        ### Input:          <-- only included if entry["input"] is non-empty
        <input text>

        ### Response:
    """
    instruction_text = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )

    input_text = f"\n\n### Input:\n{entry['input']}" if entry.get("input") else ""

    return instruction_text + input_text


def format_target(entry):
    """The response section, appended after format_input()."""
    return f"\n\n### Response:\n{entry['output']}"


class InstructionDataset(Dataset):
    """
    Pre-tokenizes each (instruction, response) pair into a single sequence:
        format_input(entry) + format_target(entry)

    The collate function (below) handles padding, target shifting, and loss
    masking at batch-construction time rather than here, since padding
    length varies per batch.
    """

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
    Custom batching function for instruction fine-tuning.

    For each sequence in the batch:
      1. Pad to the length of the LONGEST sequence in this batch (+1, since
         we need room for a shift).
      2. Build inputs = sequence[:-1], targets = sequence[1:]  (next-token
         prediction, same idea as Chapter 1's dataloader, but per-example
         here since sequences have very different natural lengths).
      3. Mask ALL but the first padding token in targets with -100
         (PyTorch's cross_entropy ignore_index) -- this stops the loss from
         being computed on padding, so the model isn't "punished" for not
         predicting more <|endoftext|> tokens than necessary.

    Why mask all but the first pad token? The first padding token right after
    real content still carries signal (the model should learn to eventually
    stop / emit <|endoftext|>) -- book's chosen convention, kept here.

    Returns:
        inputs_tensor:  [batch_size, max_len-1]
        targets_tensor: [batch_size, max_len-1] (with -100 for masked positions)
    """
    batch_max_length = max(len(item) + 1 for item in batch)  # +1 for the shift

    inputs_lst, targets_lst = [], []

    for item in batch:
        new_item = item.copy()
        new_item += [pad_token_id]  # ensure at least one pad token exists

        padded = new_item + [pad_token_id] * (batch_max_length - len(new_item))

        inputs = torch.tensor(padded[:-1])   # everything except the last token
        targets = torch.tensor(padded[1:])   # shifted by one (next-token targets)

        # Mask all but the FIRST padding token in targets
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

    # --- Step 1: Download ---
    url = ("https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/"
           "ch07/01_main-chapter-code/instruction-data.json")
    file_path = "instruction-data.json"

    data = download_and_load_instruction_data(url, file_path)
    print(f"Number of entries: {len(data)}\n")

    print("--- Example entry (raw) ---")
    print(json.dumps(data[50], indent=2))
    print()

    # --- Step 2: Format with Alpaca template ---
    print("--- Example entry (formatted with Alpaca template) ---")
    formatted = format_input(data[50]) + format_target(data[50])
    print(formatted)
    print()

    # Example with empty "input" field (instruction only, no separate input)
    no_input_entries = [e for e in data if not e.get("input")]
    print(f"Entries with no separate 'input' field: {len(no_input_entries)} / {len(data)}")
    print("--- Example (instruction-only, no ### Input: section) ---")
    print(format_input(no_input_entries[0]) + format_target(no_input_entries[0]))
    print()

    # --- Split 85% / 5% / 10% (book's convention for this dataset) ---
    train_portion = int(len(data) * 0.85)
    test_portion = int(len(data) * 0.1)
    val_portion = len(data) - train_portion - test_portion

    train_data = data[:train_portion]
    test_data = data[train_portion:train_portion + test_portion]
    val_data = data[train_portion + test_portion:]

    print(f"Split: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}\n")

    # --- Step 3: Dataset + custom collate function ---
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

    # --- Inspect one batch ---
    print("--- Sample batch from train_loader ---")
    inputs, targets = next(iter(train_loader))
    print(f"Input shape:  {inputs.shape}")
    print(f"Target shape: {targets.shape}")

    print(f"\nFirst input (first 30 tokens):  {inputs[0, :30].tolist()}")
    print(f"First target (first 30 tokens): {targets[0, :30].tolist()}")
    print(f"\nDecoded first input (first 100 chars):")
    print(repr(tokenizer.decode(inputs[0].tolist())[:100]))

    # Verify masking: count -100 tokens in first target row
    num_masked = (targets[0] == -100).sum().item()
    print(f"\nNumber of masked (-100) positions in first target: {num_masked}")
    print("(These are padding tokens beyond the first one -- excluded from loss)")

    print("\n=== Stage 1 complete ===")
    print(f"Downloaded {len(data)} instruction-response pairs, formatted with the")
    print("Alpaca template, and built dataloaders with custom collate/masking.")
