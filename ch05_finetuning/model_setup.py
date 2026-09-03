"""
Chapter 6, Stage 2: Model Setup (Steps 4-7)

Step 4: Initialize a pretrained GPT-2 model (load OpenAI weights from Chapter 4)
Step 5: Replace the output layer: 768->50257 (next-token prediction) becomes
        768->2 (spam vs ham classification)
Step 6: Freeze most of the model, unfreeze only:
        - last transformer block
        - final LayerNorm
        - new output layer
Step 7: Implement evaluation utilities (accuracy + loss)

The key architectural change: we're REUSING the pretrained GPT's understanding
of language (attention, feed-forward, embeddings) but swapping its "head" from
a next-token predictor to a binary classifier. The output is now a single
decision per INPUT SEQUENCE (not per token) -- we take the logits at the LAST
token position, pass them through softmax, and pick "spam" or "ham" based on
which score is higher.
"""

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
    """
    Replace the model's output_head with a new layer: embed_dim -> num_classes.
    
    The pretrained output_head is [embed_dim=768, vocab_size=50257]. For binary
    classification, we only need [768, 2] -- one score for "ham", one for "spam".
    
    Args:
        model: a GPTModel instance
        num_classes: 2 for binary (ham/spam)
    
    Returns:
        model with the new output_head (randomly initialized)
    """
    # Get the embedding dimension from the existing token_embedding layer
    embed_dim = model.token_embedding.embedding_dim
    
    # Replace output_head with a fresh, randomly-initialized layer
    model.output_head = nn.Linear(embed_dim, num_classes, bias=False)
    
    # The new layer's weights are NOT pretrained -- they'll be learned during
    # fine-tuning. Everything else (embeddings, attention, feed-forward) keeps
    # OpenAI's pretrained weights.
    
    return model


def freeze_model_except_last_block_and_head(model):
    """
    Freeze most of the model, unfreeze only:
      - transformer_blocks[-1] (the last transformer block)
      - final_norm
      - output_head
    
    Freezing means setting requires_grad=False -- those parameters won't be
    updated during training. This is faster (fewer gradients to compute) and
    often works just as well as full fine-tuning, since the early layers'
    pretrained representations are already good.
    """
    # Freeze everything first
    for param in model.parameters():
        param.requires_grad = False
    
    # Unfreeze the last transformer block
    for param in model.transformer_blocks[-1].parameters():
        param.requires_grad = True
    
    # Unfreeze final LayerNorm
    for param in model.final_norm.parameters():
        param.requires_grad = True
    
    # Unfreeze output_head (the new classification layer)
    for param in model.output_head.parameters():
        param.requires_grad = True


def calc_loss_batch(input_batch, target_batch, model, device):
    """
    Compute cross-entropy loss for a classification batch.
    
    Unlike pretraining (where we predict the next token at EVERY position),
    here we only care about the logits at the LAST token position -- that's
    where the model's classification decision lives.
    
    Args:
        input_batch: [batch_size, seq_len] token IDs
        target_batch: [batch_size] labels (0 or 1)
        model: GPTModel with 2-class output_head
        device: "cpu" or "cuda"
    
    Returns:
        loss: scalar cross-entropy loss
    """
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    
    logits = model(input_batch)   # [batch_size, seq_len, num_classes=2]
    logits = logits[:, -1, :]      # Take only the LAST token's logits: [batch_size, 2]
    
    loss = nn.functional.cross_entropy(logits, target_batch)
    return loss


def calc_accuracy_loader(data_loader, model, device, num_batches=None):
    """
    Compute classification accuracy over a dataloader.
    
    Args:
        data_loader: PyTorch DataLoader
        model: GPTModel with 2-class output_head
        device: "cpu" or "cuda"
        num_batches: if set, only evaluate this many batches (for speed)
    
    Returns:
        accuracy: fraction of correct predictions (0.0 to 1.0)
    """
    model.eval()
    correct, total = 0, 0
    
    with torch.no_grad():
        for i, (input_batch, target_batch) in enumerate(data_loader):
            if num_batches is not None and i >= num_batches:
                break
            
            input_batch, target_batch = input_batch.to(device), target_batch.to(device)
            
            logits = model(input_batch)
            logits = logits[:, -1, :]   # [batch_size, 2]
            predicted = torch.argmax(logits, dim=-1)  # [batch_size]
            
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
    
    # --- Step 4: Load pretrained GPT-2 ---
    print("Step 4: Loading pretrained GPT-2-small weights from OpenAI...")
    cfg = GPT_CONFIG_124M.copy()
    cfg["qkv_bias"] = True  # OpenAI's GPT-2 used bias in Q/K/V
    
    model = GPTModel(cfg)
    model.eval()
    
    sd = download_gpt2_small_state_dict()
    load_openai_weights_into_gpt(model, sd, num_layers=cfg["num_layers"])
    model.to(device)
    print("Loaded.\n")
    
    # --- Step 5: Replace output layer ---
    print("Step 5: Replacing output layer (768->50257) with classification head (768->2)...")
    replace_output_layer_for_classification(model, num_classes=2)
    print(f"New output_head: {model.output_head}\n")
    
    # --- Step 6: Freeze most of the model ---
    print("Step 6: Freezing all layers except last transformer block + final_norm + output_head...")
    freeze_model_except_last_block_and_head(model)
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)\n")
    
    # --- Step 7: Evaluation utilities ---
    print("Step 7: Testing evaluation utilities on the spam dataset...")
    
    # Quick dataset setup (reuse from dataset.py)
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
    
    # Accuracy BEFORE fine-tuning (random weights in the new output_head)
    train_acc = calc_accuracy_loader(train_loader, model, device, num_batches=5)
    val_acc = calc_accuracy_loader(val_loader, model, device, num_batches=5)
    print(f"Accuracy BEFORE fine-tuning (random output_head, first 5 batches only):")
    print(f"  Train: {train_acc:.4f}")
    print(f"  Val:   {val_acc:.4f}")
    print("(Should be close to 0.50 — random guessing on a balanced 50/50 dataset)\n")
    
    # Loss on one batch
    input_batch, target_batch = next(iter(train_loader))
    loss = calc_loss_batch(input_batch, target_batch, model, device)
    print(f"Sample batch loss: {loss.item():.4f}")
    print("(Cross-entropy loss for binary classification, before any training)\n")
    
    print("=== Stage 2 complete ===")
    print("Model loaded with OpenAI weights, output layer replaced with 768->2,")
    print("only last block + final_norm + output_head unfrozen for fine-tuning.")
