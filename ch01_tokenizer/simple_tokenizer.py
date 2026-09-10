"""Word-level tokenizer: regex split + vocab lookup. No <unk>/<endoftext> handling yet."""

import re


def split_text(text):
    """Split text into words and punctuation tokens, e.g. "Hello, world." -> ["Hello", ",", "world", "."]."""
    pieces = re.split(r'([,.:;?_!"()\']|--|\s)', text)
    tokens = [p.strip() for p in pieces if p.strip()]
    return tokens


class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}

    @classmethod
    def build_vocab(cls, text):
        tokens = split_text(text)
        unique_tokens = sorted(set(tokens))
        vocab = {token: i for i, token in enumerate(unique_tokens)}
        return vocab

    def encode(self, text):
        tokens = split_text(text)
        ids = [self.str_to_int[token] for token in tokens]
        return ids

    def decode(self, ids):
        tokens = [self.int_to_str[i] for i in ids]
        text = " ".join(tokens)
        # drop the space before punctuation: "world ." -> "world."
        text = re.sub(r'\s+([,.:;?_!"()\'])', r'\1', text)
        return text


if __name__ == "__main__":
    with open("data/the-verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()

    print("Total characters:", len(raw_text))

    tokens = split_text(raw_text)
    print("Total tokens:", len(tokens))
    print("First 10 tokens:", tokens[:10])

    vocab = SimpleTokenizerV1.build_vocab(raw_text)
    print("Vocab size:", len(vocab))
    print("First 10 vocab entries:", list(vocab.items())[:10])

    tokenizer = SimpleTokenizerV1(vocab)

    sample = "It was not till three years later that I heard the truth."
    ids = tokenizer.encode(sample)
    decoded = tokenizer.decode(ids)

    print("\nSample text:", sample)
    print("Encoded IDs:", ids)
    print("Decoded text:", decoded)
    print("Round-trip match:", sample == decoded)
