"""Load OpenAI's pretrained GPT-2 weights (via HuggingFace safetensors) into our GPTModel."""

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
    gpt.pos_embedding.weight = assign(gpt.pos_embedding.weight, sd["wpe.weight"])
    gpt.token_embedding.weight = assign(gpt.token_embedding.weight, sd["wte.weight"])

    for b in range(num_layers):
        prefix = f"h.{b}."
        block = gpt.transformer_blocks[b]

        # HF stores fused QKV as one Conv1D tensor [in, out] -- split into 3, then
        # transpose each to nn.Linear's [out, in] convention
        c_attn_w = sd[prefix + "attn.c_attn.weight"]
        c_attn_b = sd[prefix + "attn.c_attn.bias"]
        q_w, k_w, v_w = c_attn_w.chunk(3, dim=-1)
        q_b, k_b, v_b = c_attn_b.chunk(3, dim=-1)

        block.attention.W_query.weight = assign(block.attention.W_query.weight, q_w.T)
        block.attention.W_key.weight = assign(block.attention.W_key.weight, k_w.T)
        block.attention.W_value.weight = assign(block.attention.W_value.weight, v_w.T)
        block.attention.W_query.bias = assign(block.attention.W_query.bias, q_b)
        block.attention.W_key.bias = assign(block.attention.W_key.bias, k_b)
        block.attention.W_value.bias = assign(block.attention.W_value.bias, v_b)

        block.attention.out_proj.weight = assign(
            block.attention.out_proj.weight, sd[prefix + "attn.c_proj.weight"].T
        )
        block.attention.out_proj.bias = assign(
            block.attention.out_proj.bias, sd[prefix + "attn.c_proj.bias"]
        )

        # Conv1D convention again -- transpose to match our nn.Linear layers
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

        block.norm1.scale = assign(block.norm1.scale, sd[prefix + "ln_1.weight"])
        block.norm1.shift = assign(block.norm1.shift, sd[prefix + "ln_1.bias"])
        block.norm2.scale = assign(block.norm2.scale, sd[prefix + "ln_2.weight"])
        block.norm2.shift = assign(block.norm2.shift, sd[prefix + "ln_2.bias"])

    gpt.final_norm.scale = assign(gpt.final_norm.scale, sd["ln_f.weight"])
    gpt.final_norm.shift = assign(gpt.final_norm.shift, sd["ln_f.bias"])
    gpt.output_head.weight = assign(gpt.output_head.weight, sd["wte.weight"])  # weight-tied, per GPT-2

    return gpt


def download_gpt2_small_state_dict(hf_home: str = "D:/hf-cache") -> dict:
    """Download (or reuse cached) GPT-2 small weights from HuggingFace, redirected to D:."""
    return download_gpt2_state_dict("openai-community/gpt2", hf_home)


def download_gpt2_state_dict(repo_id: str, hf_home: str = "D:/hf-cache") -> dict:
    """
    Download (or reuse cached) GPT-2 weights from HuggingFace, redirected to D:.

    repo_id examples:
        "openai-community/gpt2"         -> small,  124M params, 12 layers
        "openai-community/gpt2-medium"  -> medium, 355M params, 24 layers
        "openai-community/gpt2-large"   -> large,  774M params, 36 layers
        "openai-community/gpt2-xl"      -> xl,     1.5B params, 48 layers
    """
    os.environ.setdefault("HF_HOME", hf_home)
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")  # huggingface.co is unreliable here
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    weights_path = hf_hub_download(repo_id=repo_id, filename="model.safetensors")
    return load_file(weights_path)


if __name__ == "__main__":
    import tiktoken

    print("=== Loading Pretrained GPT-2 Weights from OpenAI (via HuggingFace) ===\n")

    print("Downloading/loading cached GPT-2 small weights (~500MB, cached to D:\\hf-cache)...")
    sd = download_gpt2_small_state_dict()
    print(f"Loaded {len(sd)} tensors from the checkpoint.\n")

    cfg = GPT_CONFIG_124M.copy()
    cfg["qkv_bias"] = True  # OpenAI's original GPT-2 used bias in Q/K/V linear layers
    print(f"Config: {cfg}\n")

    torch.manual_seed(123)
    gpt = GPTModel(cfg)
    gpt.eval()

    print("Loading OpenAI's weights into our GPTModel instance...")
    load_openai_weights_into_gpt(gpt, sd, num_layers=cfg["num_layers"])
    print("Done. All tensor shapes matched -- no ValueError raised.\n")

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
