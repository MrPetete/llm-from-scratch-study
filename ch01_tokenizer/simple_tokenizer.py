"""
Section 1 — simple word-level tokenizer, built from understanding of chapter 2's
regex-splitting + vocab-lookup approach. No <unk>/<endoftext> handling yet —
that's the next task (special tokens).
"""

import re


def split_text(text):
    """Split raw text into tokens: words, and punctuation as separate tokens.

    Splits on whitespace and a set of punctuation marks, keeping the punctuation
    itself as tokens (so "Hello, world." -> ["Hello", ",", "world", "."]).
    """
    # capture group on the delimiters so re.split keeps them in the output
    pieces = re.split(r'([,.:;?_!"()\']|--|\s)', text)
    # drop empty strings and pure-whitespace pieces (whitespace itself isn't a token here)
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
        # remove the space introduced before punctuation, e.g. "world ." -> "world."
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
