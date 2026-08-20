"""
SimpleTokenizerV2: word-level tokenizer with special tokens
Adds <|unk|> for unknown words and <|endoftext|> as a document boundary marker.
"""
import re


class SimpleTokenizerV2:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}
        
    def encode(self, text, allowed_special=None):
        """
        Convert text to token IDs.
        
        Args:
            text: input string
            allowed_special: set of special tokens to preserve (e.g. {'<|endoftext|>'})
                            If None, special tokens in text are split like regular words
        
        Returns:
            list of token IDs
        """
        if allowed_special is None:
            allowed_special = set()
        
        # Protect special tokens BEFORE splitting by replacing them with safe placeholders
        # Wrap placeholders in spaces so they're always isolated during split
        special_placeholder_map = {}
        for i, token in enumerate(allowed_special):
            if token in text:
                # Use a placeholder that won't be split by the regex, with spaces for isolation
                placeholder = f" SPECIALTOKEN{i} "
                special_placeholder_map[placeholder.strip()] = token
                text = text.replace(token, placeholder)
        
        # Split on word boundaries (words + punctuation)
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        
        # Replace placeholders back with original special tokens
        preprocessed = [
            special_placeholder_map.get(token, token) for token in preprocessed
        ]
        
        # Convert to IDs, using <|unk|> for unknown words
        ids = [
            self.str_to_int.get(token, self.str_to_int["<|unk|>"]) 
            for token in preprocessed
        ]
        
        return ids
    
    def decode(self, ids):
        """Convert token IDs back to text."""
        text = " ".join([self.int_to_str[i] for i in ids])
        # Clean up spacing around punctuation
        text = re.sub(r'\s+([,.:;?!"()\'])', r'\1', text)
        return text


def build_vocab_v2(raw_text):
    """
    Build vocabulary from raw text with special tokens prepended.
    
    Special tokens get IDs 0 and 1, so they're stable across different datasets.
    """
    # Split text into tokens
    preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
    preprocessed = [item.strip() for item in preprocessed if item.strip()]
    
    # Get all unique tokens, sorted
    all_tokens = sorted(set(preprocessed))
    
    # Prepend special tokens FIRST so they get low, stable IDs
    all_tokens = ["<|unk|>", "<|endoftext|>"] + all_tokens
    
    vocab = {token: idx for idx, token in enumerate(all_tokens)}
    return vocab


if __name__ == "__main__":
    # Load the same sample text
    with open("ch01_tokenizer/data/the-verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()
    
    # Build vocab with special tokens
    vocab = build_vocab_v2(raw_text)
    tokenizer = SimpleTokenizerV2(vocab)
    
    print(f"Vocabulary size: {len(vocab):,}")
    print(f"First 10 tokens in vocab: {list(vocab.keys())[:10]}")
    print(f"<|unk|> ID: {vocab['<|unk|>']}")
    print(f"<|endoftext|> ID: {vocab['<|endoftext|>']}")
    print()
    
    # Test 1: sentence from the training text (should work same as V1)
    text1 = "It was not till three years later that I heard the truth."
    ids1 = tokenizer.encode(text1)
    decoded1 = tokenizer.decode(ids1)
    print(f"Test 1 (known words):")
    print(f"  Original : {text1}")
    print(f"  IDs      : {ids1}")
    print(f"  Decoded  : {decoded1}")
    print(f"  Match    : {text1 == decoded1}")
    print()
    
    # Test 2: sentence with unknown word (the crash case from V1)
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
    
    # Test 3: using <|endoftext|> as a boundary marker
    text3 = "This is document one.<|endoftext|>This is document two."
    ids3_without_special = tokenizer.encode(text3)
    ids3_with_special = tokenizer.encode(text3, allowed_special={"<|endoftext|>"})
    print(f"Test 3 (<|endoftext|> handling):")
    print(f"  Original              : {text3}")
    print(f"  IDs (not allowed)     : {ids3_without_special}")
    print(f"  IDs (allowed special) : {ids3_with_special}")
    print(f"  Decoded (allowed)     : {tokenizer.decode(ids3_with_special)}")
