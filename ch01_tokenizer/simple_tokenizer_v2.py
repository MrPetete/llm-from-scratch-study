"""Word-level tokenizer with <|unk|> for unknown words and <|endoftext|> as a doc boundary marker."""
import re


class SimpleTokenizerV2:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}

    def encode(self, text, allowed_special=None):
        """Encode text to token IDs. Pass allowed_special (e.g. {'<|endoftext|>'}) to keep those tokens intact."""
        if allowed_special is None:
            allowed_special = set()

        # swap special tokens for placeholders first so the regex split doesn't break them apart
        special_placeholder_map = {}
        for i, token in enumerate(allowed_special):
            if token in text:
                placeholder = f" SPECIALTOKEN{i} "
                special_placeholder_map[placeholder.strip()] = token
                text = text.replace(token, placeholder)

        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]

        preprocessed = [
            special_placeholder_map.get(token, token) for token in preprocessed
        ]

        ids = [
            self.str_to_int.get(token, self.str_to_int["<|unk|>"])
            for token in preprocessed
        ]

        return ids

    def decode(self, ids):
        """Convert token IDs back to text."""
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.:;?!"()\'])', r'\1', text)
        return text


def build_vocab_v2(raw_text):
    """Build vocab from text, with <|unk|> and <|endoftext|> prepended so they get stable IDs 0/1."""
    preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
    preprocessed = [item.strip() for item in preprocessed if item.strip()]

    all_tokens = sorted(set(preprocessed))
    all_tokens = ["<|unk|>", "<|endoftext|>"] + all_tokens

    vocab = {token: idx for idx, token in enumerate(all_tokens)}
    return vocab


if __name__ == "__main__":
    with open("ch01_tokenizer/data/the-verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()

    vocab = build_vocab_v2(raw_text)
    tokenizer = SimpleTokenizerV2(vocab)

    print(f"Vocabulary size: {len(vocab):,}")
    print(f"First 10 tokens in vocab: {list(vocab.keys())[:10]}")
    print(f"<|unk|> ID: {vocab['<|unk|>']}")
    print(f"<|endoftext|> ID: {vocab['<|endoftext|>']}")
    print()

    # Test 1: known words
    text1 = "It was not till three years later that I heard the truth."
    ids1 = tokenizer.encode(text1)
    decoded1 = tokenizer.decode(ids1)
    print(f"Test 1 (known words):")
    print(f"  Original : {text1}")
    print(f"  IDs      : {ids1}")
    print(f"  Decoded  : {decoded1}")
    print(f"  Match    : {text1 == decoded1}")
    print()

    # Test 2: unknown word
    text2 = "Hello, do you like tea?"
    ids2 = tokenizer.encode(text2)
    decoded2 = tokenizer.decode(ids2)
    print(f"Test 2 (unknown word 'Hello'):")
    print(f"  Original : {text2}")
    print(f"  IDs      : {ids2}")
    print(f"  Decoded  : {decoded2}")
    print(f"  Match    : {text2 == decoded2}")
    print(f"  Note     : 'Hello' → '<|unk|>' (ID {vocab['<|unk|>']})")
    print()

    # Test 3: <|endoftext|> as a boundary marker
    text3 = "This is document one.<|endoftext|>This is document two."
    ids3_without_special = tokenizer.encode(text3)
    ids3_with_special = tokenizer.encode(text3, allowed_special={"<|endoftext|>"})
    print(f"Test 3 (<|endoftext|> handling):")
    print(f"  Original              : {text3}")
    print(f"  IDs (not allowed)     : {ids3_without_special}")
    print(f"  IDs (allowed special) : {ids3_with_special}")
    print(f"  Decoded (allowed)     : {tokenizer.decode(ids3_with_special)}")
