"""Turn a pretrained GPT-2 into a spam/ham classifier: swap the output head and freeze most layers."""

import os
import sys
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ch03_gpt_model"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ch04_pretraining"))

from config import GPT_CONFIG_124M
from gpt_model import GPTModel
from load_openai_weights import download_gpt2_small_state_dict, load_openai_weights_into_gpt


def replace_output_layer_for_classification(model, num_classes=2):
    """Replace model.output_head (768->50257) with a fresh 768->num_classes layer."""
    embed_dim = model.token_embedding.embedding_dim
    model.output_head = nn.Linear(embed_dim, num_classes, bias=False)
    return model


def freeze_model_except_last_block_and_head(model):
    """Freeze all params except the last transformer block, final_norm, and output_head."""
    for param in model.parameters():
        param.requires_grad = False

    for param in model.transformer_blocks[-1].parameters():
        param.requires_grad = True

    for param in model.final_norm.parameters():
        param.requires_grad = True

    for param in model.output_head.parameters():
        param.requires_grad = True


def calc_loss_batch(input_batch, target_batch, model, device):
    """Cross-entropy loss using only the last token position's logits (the classification decision)."""
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)

    logits = model(input_batch)
    logits = logits[:, -1, :]

    loss = nn.functional.cross_entropy(logits, target_batch)
    return loss


def calc_accuracy_loader(data_loader, model, device, num_batches=None):
    """Classification accuracy over a dataloader (optionally capped to num_batches for speed)."""
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for i, (input_batch, target_batch) in enumerate(data_loader):
            if num_batches is not None and i >= num_batches:
                break

            input_batch, target_batch = input_batch.to(device), target_batch.to(device)

            logits = model(input_batch)
            logits = logits[:, -1, :]
            predicted = torch.argmax(logits, dim=-1)

            correct += (predicted == target_batch).sum().item()
            total += target_batch.size(0)

    return correct / total if total > 0 else 0.0


if __name__ == "__main__":
    import tiktoken
    from dataset import download_and_unzip_spam_data, load_and_balance_dataset, random_split, SpamDataset
    from torch.utils.data import DataLoader
    from pathlib import Path

    print("=== Chapter 6, Stage 2: Model Setup ===\n")

    device = "cpu"
    print(f"Device: {device}\n")

    print("Loading pretrained GPT-2-small weights from OpenAI...")
    cfg = GPT_CONFIG_124M.copy()
    cfg["qkv_bias"] = True

    model = GPTModel(cfg)
    model.eval()

    sd = download_gpt2_small_state_dict()
    load_openai_weights_into_gpt(model, sd, num_layers=cfg["num_layers"])
    model.to(device)
    print("Loaded.\n")

    print("Replacing output layer (768->50257) with classification head (768->2)...")
    replace_output_layer_for_classification(model, num_classes=2)
    print(f"New output_head: {model.output_head}\n")

    print("Freezing all layers except last transformer block + final_norm + output_head...")
    freeze_model_except_last_block_and_head(model)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)\n")

    print("Testing evaluation utilities on the spam dataset...")

    url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
    zip_path = "sms_spam_collection.zip"
    extracted_path = "sms_spam_collection"
    data_file_path = Path(extracted_path) / "SMSSpamCollection.tsv"

    download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path)
    df = load_and_balance_dataset(data_file_path)
    train_df, val_df, test_df = random_split(df, train_frac=0.7, val_frac=0.1)

    train_df["Label"] = train_df["Label"].map({"ham": 0, "spam": 1})
    val_df["Label"] = val_df["Label"].map({"ham": 0, "spam": 1})

    tokenizer = tiktoken.get_encoding("gpt2")
    max_length = 120

    train_dataset = SpamDataset(train_df["Text"].tolist(), train_df["Label"].tolist(), tokenizer, max_length)
    val_dataset = SpamDataset(val_df["Text"].tolist(), val_df["Label"].tolist(), tokenizer, max_length)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    train_acc = calc_accuracy_loader(train_loader, model, device, num_batches=5)
    val_acc = calc_accuracy_loader(val_loader, model, device, num_batches=5)
    print(f"Accuracy BEFORE fine-tuning (random output_head, first 5 batches only):")
    print(f"  Train: {train_acc:.4f}")
    print(f"  Val:   {val_acc:.4f}")
    print("(Should be close to 0.50 -- random guessing on a balanced 50/50 dataset)\n")

    input_batch, target_batch = next(iter(train_loader))
    loss = calc_loss_batch(input_batch, target_batch, model, device)
    print(f"Sample batch loss: {loss.item():.4f}")
    print("(Cross-entropy loss for binary classification, before any training)\n")

    print("=== Stage 2 complete ===")
    print("Model loaded with OpenAI weights, output layer replaced with 768->2,")
    print("only last block + final_norm + output_head unfrozen for fine-tuning.")
