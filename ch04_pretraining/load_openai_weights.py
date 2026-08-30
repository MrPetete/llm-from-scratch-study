"""
Chapter 4, Step 7: Loading Pretrained Weights from OpenAI

Everything up to this point trained OUR model on a tiny 20K-character
dataset -- useful for proving the pipeline works, but nowhere near enough
data to produce genuinely fluent text. OpenAI trained GPT-2 on ~40GB of
web text. This step proves our architecture is a REAL, faithful GPT-2
implementation by loading OpenAI's actual trained weights into it and
confirming the SAME model class suddenly produces fluent, coherent text.

WEIGHT SOURCE
    The book's original code downloads OpenAI's own TensorFlow checkpoint
    format via a custom `gpt_download.py` helper. That format is fragile
    to reproduce today. Instead, we pull the equivalent weights from
    HuggingFace's `openai-community/gpt2` repo in `.safetensors` format --
    same numbers, modern/safe format, no pickle/tensorflow dependency.

WHY THE KEYS DON'T MATCH DIRECTLY
    HuggingFace's GPT-2 uses OpenAI's original naming (wte, wpe, h.{i}.attn.c_attn,
    ln_1, mlp.c_fc, ...) and its Q/K/V projections are stored as a SINGLE fused
    "Conv1D" layer per block (c_attn: combines all three), with weight shape
    [in_features, out_features] -- the OPPOSITE convention from PyTorch's
    nn.Linear, which stores [out_features, in_features]. Our MultiHeadAttention
    (Chapter 2) uses three SEPARATE nn.Linear layers. So loading requires:
      1. Splitting the fused c_attn tensor into three equal chunks (Q, K, V)
      2. Transposing each chunk (Conv1D -> nn.Linear convention)
      3. Mapping OpenAI's block-index naming (h.{i}....) onto our
         transformer_blocks[i]... attribute names

CONFIDENCE CHECK
    If any shape or mapping is wrong, either loading raises a shape-mismatch
    error immediately, or (worse, silently) the model produces gibberish
    despite "successfully" loading. Producing FLUENT text is itself the
    proof that every single one of ~150 tensor assignments landed correctly --
    a single wrong mapping among hundreds would show up as garbage output.
"""

import os
import sys
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ch03_gpt_model"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ch01_tokenizer"))

from config import GPT_CONFIG_124M
from gpt_model import GPTModel


def assign(left: torch.nn.Parameter, right: torch.Tensor) -> torch.nn.Parameter:
    """Sanity-checked weight assignment: shapes must match exactly."""
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
    return torch.nn.Parameter(right.clone())


def load_openai_weights_into_gpt(gpt: GPTModel, sd: dict, num_layers: int):
    """
    Args:
        gpt: a GPTModel instance built with qkv_bias=True and the matching
             context_length/embed_dim/num_heads/num_layers for the checkpoint
        sd: HuggingFace GPT-2 safetensors state dict (raw tensor names)
        num_layers: number of transformer blocks (12 for gpt2-small)
    """
    # --- Embeddings ---
    gpt.pos_embedding.weight = assign(gpt.pos_embedding.weight, sd["wpe.weight"])
    gpt.token_embedding.weight = assign(gpt.token_embedding.weight, sd["wte.weight"])

    for b in range(num_layers):
        prefix = f"h.{b}."
        block = gpt.transformer_blocks[b]

        # --- Attention: fused c_attn -> split into Q/K/V, transpose Conv1D->Linear ---
        c_attn_w = sd[prefix + "attn.c_attn.weight"]   # [embed_dim, 3*embed_dim]
        c_attn_b = sd[prefix + "attn.c_attn.bias"]     # [3*embed_dim]
        q_w, k_w, v_w = c_attn_w.chunk(3, dim=-1)      # each [embed_dim, embed_dim]
        q_b, k_b, v_b = c_attn_b.chunk(3, dim=-1)      # each [embed_dim]

        block.attention.W_query.weight = assign(block.attention.W_query.weight, q_w.T)
        block.attention.W_key.weight = assign(block.attention.W_key.weight, k_w.T)
        block.attention.W_value.weight = assign(block.attention.W_value.weight, v_w.T)
        block.attention.W_query.bias = assign(block.attention.W_query.bias, q_b)
        block.attention.W_key.bias = assign(block.attention.W_key.bias, k_b)
        block.attention.W_value.bias = assign(block.attention.W_value.bias, v_b)

        # --- Attention output projection ---
        block.attention.out_proj.weight = assign(
            block.attention.out_proj.weight, sd[prefix + "attn.c_proj.weight"].T
        )
        block.attention.out_proj.bias = assign(
            block.attention.out_proj.bias, sd[prefix + "attn.c_proj.bias"]
        )

        # --- Feed-forward (mlp.c_fc = expand, mlp.c_proj = compress) ---
        block.feed_forward.layers[0].weight = assign(
            block.feed_forward.layers[0].weight, sd[prefix + "mlp.c_fc.weight"].T
        )
        block.feed_forward.layers[0].bias = assign(
            block.feed_forward.layers[0].bias, sd[prefix + "mlp.c_fc.bias"]
        )
        block.feed_forward.layers[2].weight = assign(
            block.feed_forward.layers[2].weight, sd[prefix + "mlp.c_proj.weight"].T
        )
        block.feed_forward.layers[2].bias = assign(
            block.feed_forward.layers[2].bias, sd[prefix + "mlp.c_proj.bias"]
        )

        # --- LayerNorms (OpenAI: ln_1 before attn, ln_2 before mlp -- matches our norm1/norm2) ---
        block.norm1.scale = assign(block.norm1.scale, sd[prefix + "ln_1.weight"])
        block.norm1.shift = assign(block.norm1.shift, sd[prefix + "ln_1.bias"])
        block.norm2.scale = assign(block.norm2.scale, sd[prefix + "ln_2.weight"])
        block.norm2.shift = assign(block.norm2.shift, sd[prefix + "ln_2.bias"])

    # --- Final LayerNorm + output head (weight-tied to token embedding, per GPT-2) ---
    gpt.final_norm.scale = assign(gpt.final_norm.scale, sd["ln_f.weight"])
    gpt.final_norm.shift = assign(gpt.final_norm.shift, sd["ln_f.bias"])
    gpt.output_head.weight = assign(gpt.output_head.weight, sd["wte.weight"])

    return gpt


def download_gpt2_small_state_dict(hf_home: str = "D:/hf-cache") -> dict:
    """Download (or reuse cached) GPT-2 small weights from HuggingFace, redirected to D:."""
    os.environ.setdefault("HF_HOME", hf_home)
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    weights_path = hf_hub_download(repo_id="openai-community/gpt2", filename="model.safetensors")
    return load_file(weights_path)


if __name__ == "__main__":
    import tiktoken

    print("=== Loading Pretrained GPT-2 Weights from OpenAI (via HuggingFace) ===\n")

    print("Downloading/loading cached GPT-2 small weights (~500MB, cached to D:\\hf-cache)...")
    sd = download_gpt2_small_state_dict()
    print(f"Loaded {len(sd)} tensors from the checkpoint.\n")

    # --- Build OUR model with matching config: GPT-2 small used qkv_bias=True ---
    cfg = GPT_CONFIG_124M.copy()
    cfg["qkv_bias"] = True   # OpenAI's original GPT-2 used bias in Q/K/V linear layers
    print(f"Config: {cfg}\n")

    torch.manual_seed(123)
    gpt = GPTModel(cfg)
    gpt.eval()

    print("Loading OpenAI's weights into our GPTModel instance...")
    load_openai_weights_into_gpt(gpt, sd, num_layers=cfg["num_layers"])
    print("Done. All tensor shapes matched -- no ValueError raised.\n")

    # --- The real confidence check: does it produce coherent text? ---
    print("--- Generating text with OpenAI's real weights, loaded into OUR architecture ---")
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ch04_pretraining"))
    from decoding import generate_text_sampled

    tokenizer = tiktoken.get_encoding("gpt2")
    start_text = "Every effort moves you"
    start_ids = torch.tensor([tokenizer.encode(start_text)])

    torch.manual_seed(123)
    output_ids = generate_text_sampled(
        gpt, start_ids, max_new_tokens=25, context_length=cfg["context_length"],
        temperature=1.5, top_k=50,
    )
    generated_text = tokenizer.decode(output_ids[0].tolist())

    print(f"Prompt: {repr(start_text)}")
    print(f"Generated: {repr(generated_text)}\n")

    print("=== Key observations ===")
    print("1. SAME GPTModel class as Chapters 3-4, only the WEIGHTS changed -- this is the proof")
    print("   our architecture is a faithful, structurally correct GPT-2 reimplementation")
    print("2. HuggingFace's GPT-2 stores Q/K/V fused in one 'c_attn' tensor, and uses the")
    print("   Conv1D convention ([in, out]) instead of nn.Linear's ([out, in]) -- required a")
    print("   chunk() + transpose() for every attention weight, not a direct copy")
    print("3. Fluent, grammatically coherent output (vs our own tiny-dataset model's repetition)")
    print("   confirms ALL ~150 tensor mappings landed correctly -- a single wrong mapping")
    print("   among hundreds would show up immediately as garbage output, not a subtle bug")
    print("4. This model was trained by OpenAI on far more data than we could ever gather")
    print("   locally -- loading it is how real-world LLM projects bootstrap from pretraining")
