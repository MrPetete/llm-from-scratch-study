"""
Step 3: Byte Pair Encoding (BPE) Tokenizer using tiktoken

BPE is the tokenization scheme used by GPT-2, GPT-3, and ChatGPT.
Instead of treating words as atomic units, BPE breaks them into subword units,
which allows it to:
1. Handle unknown words without <|unk|> tokens
2. Represent rare words efficiently
3. Work with any Unicode text

We use OpenAI's tiktoken library, which implements BPE efficiently in Rust.
"""

import tiktoken

def main():
    # Load the GPT-2 BPE tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")
    
    print("=== BPE Tokenizer (GPT-2) ===\n")
    
    # Test 1: Basic encoding/decoding
    print("Test 1: Basic text")
    text1 = "Hello, do you like tea?"
    ids1 = tokenizer.encode(text1)
    decoded1 = tokenizer.decode(ids1)
    print(f"  Original: {text1}")
    print(f"  IDs: {ids1}")
    print(f"  Decoded: {decoded1}")
    print(f"  Match: {text1 == decoded1}")
    print()
    
    # Test 2: Unknown/rare words - BPE breaks them into subwords
    print("Test 2: Unknown word (breaks into subword units)")
    text2 = "someunknownPlace"
    ids2 = tokenizer.encode(text2)
    decoded2 = tokenizer.decode(ids2)
    print(f"  Original: {text2}")
    print(f"  IDs: {ids2}")
    print(f"  Decoded: {decoded2}")
    print(f"  Match: {text2 == decoded2}")
    print(f"  Note: The word was split into {len(ids2)} tokens")
    
    # Show each subword token
    print("  Subword breakdown:")
    for token_id in ids2:
        token_bytes = tokenizer.decode_single_token_bytes(token_id)
        token_str = token_bytes.decode('utf-8', errors='replace')
        print(f"    ID {token_id} -> '{token_str}'")
    print()
    
    # Test 3: Special token <|endoftext|>
    print("Test 3: <|endoftext|> handling")
    text3 = "Document one.<|endoftext|>Document two."
    
    # Without allowed_special: we must explicitly disable the check to treat it as text
    # (tiktoken by default refuses to encode special tokens as regular text)
    ids3_no_special = tokenizer.encode(text3, disallowed_special=())
    print(f"  Original: {text3}")
    print(f"  IDs (no special, treated as text): {ids3_no_special}")
    print(f"  Decoded: {tokenizer.decode(ids3_no_special)}")
    print()
    
    # With allowed_special: <|endoftext|> treated as single token (ID 50256)
    ids3_with_special = tokenizer.encode(text3, allowed_special={"<|endoftext|>"})
    print(f"  IDs (with special): {ids3_with_special}")
    print(f"  Decoded: {tokenizer.decode(ids3_with_special)}")
    print(f"  Note: <|endoftext|> has ID 50256 (largest in GPT-2 vocab)")
    print()
    
    # Test 4: Compare vocab sizes
    print("Test 4: Vocabulary size")
    print(f"  GPT-2 BPE vocab size: {tokenizer.n_vocab}")
    print(f"  SimpleTokenizerV1 vocab: 1,130 (from 'The Verdict')")
    print(f"  SimpleTokenizerV2 vocab: 1,132 (added <|unk|> and <|endoftext|>)")
    print()
    print(f"  BPE vocab is {tokenizer.n_vocab // 1132}x larger, but can handle ANY text")
    print(f"  without unknown tokens, including code, multilingual text, etc.")
    print()
    
    # Test 5: Multilingual and special characters
    print("Test 5: Multilingual text (Chinese + English)")
    text5 = "你好世界! Hello world!"
    ids5 = tokenizer.encode(text5)
    decoded5 = tokenizer.decode(ids5)
    print(f"  Original: {text5}")
    print(f"  IDs: {ids5}")
    print(f"  Token count: {len(ids5)}")
    print(f"  Decoded: {decoded5}")
    print(f"  Match: {text5 == decoded5}")

if __name__ == "__main__":
    main()
