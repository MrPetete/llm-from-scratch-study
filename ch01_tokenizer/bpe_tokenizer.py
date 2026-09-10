"""BPE tokenizer (GPT-2's scheme) via tiktoken -- subword units instead of whole-word vocab, so no <|unk|> needed."""

import tiktoken

def main():
    tokenizer = tiktoken.get_encoding("gpt2")

    print("=== BPE Tokenizer (GPT-2) ===\n")

    # Test 1: basic encode/decode
    print("Test 1: Basic text")
    text1 = "Hello, do you like tea?"
    ids1 = tokenizer.encode(text1)
    decoded1 = tokenizer.decode(ids1)
    print(f"  Original: {text1}")
    print(f"  IDs: {ids1}")
    print(f"  Decoded: {decoded1}")
    print(f"  Match: {text1 == decoded1}")
    print()

    # Test 2: unknown word gets split into subwords
    print("Test 2: Unknown word (breaks into subword units)")
    text2 = "someunknownPlace"
    ids2 = tokenizer.encode(text2)
    decoded2 = tokenizer.decode(ids2)
    print(f"  Original: {text2}")
    print(f"  IDs: {ids2}")
    print(f"  Decoded: {decoded2}")
    print(f"  Match: {text2 == decoded2}")
    print(f"  Note: The word was split into {len(ids2)} tokens")

    print("  Subword breakdown:")
    for token_id in ids2:
        token_bytes = tokenizer.decode_single_token_bytes(token_id)
        token_str = token_bytes.decode('utf-8', errors='replace')
        print(f"    ID {token_id} -> '{token_str}'")
    print()

    # Test 3: <|endoftext|> special token
    print("Test 3: <|endoftext|> handling")
    text3 = "Document one.<|endoftext|>Document two."

    # tiktoken refuses special tokens as plain text unless explicitly allowed
    ids3_no_special = tokenizer.encode(text3, disallowed_special=())
    print(f"  Original: {text3}")
    print(f"  IDs (no special, treated as text): {ids3_no_special}")
    print(f"  Decoded: {tokenizer.decode(ids3_no_special)}")
    print()

    ids3_with_special = tokenizer.encode(text3, allowed_special={"<|endoftext|>"})
    print(f"  IDs (with special): {ids3_with_special}")
    print(f"  Decoded: {tokenizer.decode(ids3_with_special)}")
    print(f"  Note: <|endoftext|> has ID 50256 (largest in GPT-2 vocab)")
    print()

    # Test 4: vocab size comparison
    print("Test 4: Vocabulary size")
    print(f"  GPT-2 BPE vocab size: {tokenizer.n_vocab}")
    print(f"  SimpleTokenizerV1 vocab: 1,130 (from 'The Verdict')")
    print(f"  SimpleTokenizerV2 vocab: 1,132 (added <|unk|> and <|endoftext|>)")
    print()
    print(f"  BPE vocab is {tokenizer.n_vocab // 1132}x larger, but can handle ANY text")
    print(f"  without unknown tokens, including code, multilingual text, etc.")
    print()

    # Test 5: multilingual text
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
