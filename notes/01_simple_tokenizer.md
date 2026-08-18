# Section 1, Task 1 — Simple word-level tokenizer

**Built:** `SimpleTokenizerV1` — regex-based split (words + punctuation as separate
tokens, whitespace dropped), vocab built from sorted unique tokens, `encode`/`decode`
via dict lookup in both directions.

**Data:** "The Verdict" by Edith Wharton (public domain short story, same sample text
used in the book/official repo) — 20,479 characters, 4,690 tokens, 1,130 unique tokens.

**Verified:** round-trip encode → decode on a test sentence returns the exact original
string, including correct punctuation spacing (handled via a regex that strips the
space `split_text` inserts before punctuation during decode).

**Bug hit (deliberately, to understand the next problem):** encoding any word not
in the training vocab raises `KeyError` — e.g. `tokenizer.encode("Hello, do you like tea?")`
fails on `'Hello'` since that word never appears in the source text. This is a fixed
vocabulary's fundamental limitation, and it's the direct motivation for the next task:
special tokens (`<|unk|>`, `<|endoftext|>`), and eventually BPE, which sidesteps this
problem entirely by falling back to subword/character-level pieces for anything unseen.

**Next:** add special token handling (`SimpleTokenizerV2` with `<|unk|>`/`<|endoftext|>`),
then move to BPE via `tiktoken` and compare.
