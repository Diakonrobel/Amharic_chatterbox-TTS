"""
Simple Tokenizer for Merged Vocabulary JSON
Loads merged Chatterbox + Amharic tokenizer
"""

import json
from pathlib import Path
from typing import List, Dict


class MergedTokenizer:
    """
    Simple tokenizer that loads from merged vocab JSON
    (Chatterbox multilingual + Amharic)
    """
    
    def __init__(self, vocab_path: str):
        """
        Load merged tokenizer from JSON
        
        Args:
            vocab_path: Path to merged vocab JSON file
        """
        with open(vocab_path, 'r', encoding='utf-8') as f:
            self.vocab = json.load(f)
        
        # Create reverse mapping
        self.id_to_token = {v: k for k, v in self.vocab.items()}
        
        # Special token IDs
        self.pad_id = self.vocab.get('<pad>', 0)
        self.unk_id = self.vocab.get('<unk>', 1)
        self.bos_id = self.vocab.get('<s>', 2)
        self.eos_id = self.vocab.get('</s>', 3)
    
    def encode(self, text: str, use_phonemes: bool = False) -> List[int]:
        """
        Encode text to token IDs using character-level tokenization
        
        Args:
            text: Input text
            use_phonemes: Ignored (for compatibility)
            
        Returns:
            List of token IDs
        """
        import unicodedata
        
        # Normalize Unicode
        text = unicodedata.normalize('NFC', text)
        
        tokens = []
        for char in text:
            if char in self.vocab:
                tokens.append(self.vocab[char])
            else:
                # Try to find similar character or use UNK
                tokens.append(self.unk_id)
        
        return tokens
    
    def decode(self, ids: List[int]) -> str:
        """
        Decode token IDs to text
        
        Args:
            ids: List of token IDs
            
        Returns:
            Decoded text
        """
        chars = []
        for id in ids:
            if id in self.id_to_token:
                chars.append(self.id_to_token[id])
            else:
                chars.append('�')  # Replacement character
        
        return ''.join(chars)
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size"""
        return len(self.vocab)
    
    @classmethod
    def load(cls, vocab_path: str, **kwargs):
        """
        Load tokenizer from vocab JSON
        
        Args:
            vocab_path: Path to merged vocab JSON
            **kwargs: Ignored (for compatibility)
            
        Returns:
            MergedTokenizer instance
        """
        return cls(vocab_path)


def load_merged_tokenizer(vocab_path: str = "models/tokenizer/Am_tokenizer_merged.json"):
    """
    Helper function to load merged tokenizer
    
    Args:
        vocab_path: Path to merged vocab JSON
        
    Returns:
        MergedTokenizer instance
    """
    return MergedTokenizer(vocab_path)
