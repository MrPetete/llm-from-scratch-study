"""
Chapter 6, Stage 1: Dataset Preparation (Steps 1-3)

Step 1: Download the SMS Spam Collection dataset
Step 2: Balance, split (train/val/test 70/10/20)
Step 3: Create PyTorch dataloaders with padding to max length

The dataset: 5,572 SMS text messages labeled "spam" or "ham" (not spam).
Originally imbalanced (4,825 ham vs 747 spam) -- we undersample to 747 of each
class for simplicity, giving a clean 50/50 balanced dataset.
"""

import urllib.request
import zipfile
import os
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


def download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path):
    """Download and extract the SMS Spam Collection dataset."""
    if data_file_path.exists():
        print(f"{data_file_path} already exists. Skipping download.")
        return
    
    print(f"Downloading {url}...")
    with urllib.request.urlopen(url) as response:
        with open(zip_path, "wb") as out_file:
            out_file.write(response.read())
    
    print(f"Unzipping to {extracted_path}...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extracted_path)
    
    # Rename to add .tsv extension
    original_file_path = Path(extracted_path) / "SMSSpamCollection"
    os.rename(original_file_path, data_file_path)
    print(f"File downloaded and saved as {data_file_path}")


def load_and_balance_dataset(data_file_path):
    """
    Load the dataset, undersample to balance classes, return a pandas DataFrame.
    
    Returns:
        df: balanced DataFrame with columns ["Label", "Text"]
    """
    df = pd.read_csv(data_file_path, sep="\t", header=None, names=["Label", "Text"])
    print(f"\nOriginal dataset: {len(df)} rows")
    print(f"Label distribution:\n{df['Label'].value_counts()}\n")
    
    # Undersample "ham" to match "spam" count (747 of each)
    num_spam = df[df["Label"] == "spam"].shape[0]
    ham_subset = df[df["Label"] == "ham"].sample(n=num_spam, random_state=123)
    balanced_df = pd.concat([ham_subset, df[df["Label"] == "spam"]], axis=0)
    
    print(f"Balanced dataset: {len(balanced_df)} rows")
    print(f"Label distribution:\n{balanced_df['Label'].value_counts()}\n")
    
    return balanced_df


def random_split(df, train_frac, val_frac):
    """
    Split DataFrame into train/val/test with given fractions.
    
    Args:
        df: input DataFrame
        train_frac: fraction for training (e.g. 0.7)
        val_frac: fraction for validation (e.g. 0.1)
        
    Returns:
        train_df, val_df, test_df
    """
    df = df.sample(frac=1, random_state=123).reset_index(drop=True)  # shuffle
    
    train_end = int(len(df) * train_frac)
    val_end = train_end + int(len(df) * val_frac)
    
    train_df = df[:train_end]
    val_df = df[train_end:val_end]
    test_df = df[val_end:]
    
    return train_df, val_df, test_df


class SpamDataset(Dataset):
    """
    PyTorch Dataset for spam classification.
    
    Args:
        texts: list of strings (SMS messages)
        labels: list of 0/1 (0=ham, 1=spam)
        tokenizer: tiktoken tokenizer
        max_length: pad/truncate all sequences to this length
        pad_token_id: token ID to use for padding
    """
    def __init__(self, texts, labels, tokenizer, max_length=None, pad_token_id=50256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_token_id = pad_token_id  # <|endoftext|> token for GPT-2
        
        # Encode all texts once at init
        self.encoded_texts = [tokenizer.encode(text) for text in texts]
        
        # If max_length not provided, use the longest sequence in this dataset
        if self.max_length is None:
            self.max_length = max(len(enc) for enc in self.encoded_texts)
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        encoded = self.encoded_texts[idx]
        label = self.labels[idx]
        
        # Truncate if longer than max_length
        if len(encoded) > self.max_length:
            encoded = encoded[:self.max_length]
        
        # Pad if shorter than max_length
        padded = encoded + [self.pad_token_id] * (self.max_length - len(encoded))
        
        return torch.tensor(padded, dtype=torch.long), torch.tensor(label, dtype=torch.long)


if __name__ == "__main__":
    import tiktoken
    
    print("=== Chapter 6, Stage 1: Dataset Preparation ===\n")
    
    # --- Step 1: Download ---
    url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
    zip_path = "sms_spam_collection.zip"
    extracted_path = "sms_spam_collection"
    data_file_path = Path(extracted_path) / "SMSSpamCollection.tsv"
    
    download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path)
    
    # --- Step 2: Balance and split ---
    df = load_and_balance_dataset(data_file_path)
    
    train_df, val_df, test_df = random_split(df, train_frac=0.7, val_frac=0.1)
    print(f"Split: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}\n")
    
    # Convert labels: "ham" -> 0, "spam" -> 1
    train_df["Label"] = train_df["Label"].map({"ham": 0, "spam": 1})
    val_df["Label"] = val_df["Label"].map({"ham": 0, "spam": 1})
    test_df["Label"] = test_df["Label"].map({"ham": 0, "spam": 1})
    
    # --- Step 3: Create dataloaders ---
    tokenizer = tiktoken.get_encoding("gpt2")
    
    # Find the longest message in the ENTIRE dataset (train+val+test) to set max_length
    all_texts = pd.concat([train_df["Text"], val_df["Text"], test_df["Text"]])
    all_lengths = [len(tokenizer.encode(text)) for text in all_texts]
    max_length = max(all_lengths)
    print(f"Longest message in dataset: {max_length} tokens")
    print(f"Using max_length={max_length} for all dataloaders (pad shorter, truncate longer)\n")
    
    train_dataset = SpamDataset(
        train_df["Text"].tolist(), train_df["Label"].tolist(),
        tokenizer, max_length=max_length
    )
    val_dataset = SpamDataset(
        val_df["Text"].tolist(), val_df["Label"].tolist(),
        tokenizer, max_length=max_length
    )
    test_dataset = SpamDataset(
        test_df["Text"].tolist(), test_df["Label"].tolist(),
        tokenizer, max_length=max_length
    )
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches:   {len(val_loader)}")
    print(f"Test batches:  {len(test_loader)}\n")
    
    # --- Inspect one batch ---
    print("--- Sample batch from train_loader ---")
    inputs, targets = next(iter(train_loader))
    print(f"Input shape:  {inputs.shape}  (batch_size, max_length={max_length})")
    print(f"Target shape: {targets.shape}  (batch_size,)")
    print(f"First input (token IDs, truncated to first 20): {inputs[0, :20].tolist()}")
    print(f"First target (label): {targets[0].item()}  (0=ham, 1=spam)")
    print(f"Decoded first input (truncated to first 50 chars): {repr(tokenizer.decode(inputs[0].tolist())[:50])}")
    
    print("\n=== Stage 1 complete ===")
    print("Dataset downloaded, balanced (747 spam + 747 ham), split 70/10/20,")
    print(f"and wrapped in PyTorch dataloaders with padding to max_length={max_length}.")
