"""
Amharic Tokenizer for Chatterbox TTS
Extends Chatterbox's base tokenizer with Amharic tokens
Based on practical experience from multilingual training
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Optional
import sentencepiece as spm
from tokenizers import Tokenizer, models, pre_tokenizers, trainers, processors


class AmharicTokenizer:
    """
    Custom tokenizer for Amharic TTS
    Supports extending Chatterbox's base tokenizer
    """
    
    def __init__(self, g2p=None):
        """
        Initialize tokenizer
        
        Args:
            g2p: AmharicG2P instance for phoneme conversion
        """
        self.g2p = g2p
        self.char_tokenizer = None
        self.vocab = {}
        self.id_to_token = {}
    
    def build_vocab_from_texts(self, texts: List[str]) -> Dict:
        """
        Build vocabulary from Amharic texts
        
        Args:
            texts: List of Amharic texts
            
        Returns:
            Dictionary mapping tokens to IDs
        """
        # Collect unique characters and phonemes
        char_vocab = set()
        phoneme_vocab = set()
        
        for text in texts:
            # Add characters
            for char in text:
                if char.strip():
                    char_vocab.add(char)
            
            # Add phonemes if G2P is available
            if self.g2p:
                phonemes = self.g2p.text_to_sequence(text)
                phoneme_vocab.update(phonemes)
        
        # Combine vocabularies
        all_tokens = sorted(char_vocab.union(phoneme_vocab))
        
        # Add special tokens
        special_tokens = [
            "<PAD>",   # Padding
            "<BOS>",   # Begin of sequence
            "<EOS>",   # End of sequence
            "<UNK>",   # Unknown
            "<MASK>",  # Masking
        ]
        
        vocab = {token: idx for idx, token in enumerate(special_tokens)}
        for token in all_tokens:
            if token not in vocab:
                vocab[token] = len(vocab)
        
        self.vocab = vocab
        self.id_to_token = {v: k for k, v in vocab.items()}
        
        return vocab
    
    def train_sentencepiece(self, texts: List[str], model_prefix: str, 
                           vocab_size: int = 500, 
                           exclude_tokens: Optional[set] = None):
        """
        Train SentencePiece BPE model for Amharic
        
        Args:
            texts: Training texts
            model_prefix: Output model prefix
            vocab_size: Target vocabulary size
            exclude_tokens: Tokens to exclude (for avoiding overlap)
        """
        # Save texts to temporary file
        temp_file = f"{model_prefix}_temp.txt"
        with open(temp_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(texts))
        
        # Train SentencePiece with enhanced settings for Amharic
        # Using only parameters compatible with all SentencePiece versions
        spm.SentencePieceTrainer.train(
            input=temp_file,
            model_prefix=model_prefix,
            vocab_size=vocab_size,
            character_coverage=1.0,  # Critical for Ethiopic script coverage
            model_type='bpe',
            pad_id=0,
            unk_id=1,
            bos_id=2,
            eos_id=3,
            user_defined_symbols=['<MASK>'],
            normalization_rule_name='nfkc',
            # Basic training settings for compatibility
            input_sentence_size=200000,
            shuffle_input_sentence=True,
            num_threads=4,
            split_by_whitespace=True,
            split_by_unicode_script=True,
            split_by_number=True,
            split_digits=True
        )
        
        # Load trained model
        self.char_tokenizer = spm.SentencePieceProcessor()
        self.char_tokenizer.load(f"{model_prefix}.model")
        
        # Clean up
        os.remove(temp_file)
        
        print(f"✓ Trained SentencePiece model")
        print(f"  Vocabulary size: {self.char_tokenizer.get_piece_size()}")
    
    def encode(self, text: str, use_phonemes: bool = True) -> List[int]:
        """
        Encode text to token IDs
        
        Args:
            text: Input text
            use_phonemes: Whether to use phoneme representation
            
        Returns:
            List of token IDs
        """
        if use_phonemes and self.g2p:
            # Convert to phonemes first
            phoneme_text = self.g2p.grapheme_to_phoneme(text)
            
            if self.char_tokenizer:
                return self.char_tokenizer.encode(phoneme_text, out_type=int)
            else:
                # Get UNK token ID with fallback
                unk_id = self.vocab.get("<UNK>", 1)  # Default to 1 if not found
                return [self.vocab.get(p, unk_id) for p in phoneme_text]
        else:
            if self.char_tokenizer:
                return self.char_tokenizer.encode(text, out_type=int)
            else:
                # Get UNK token ID with fallback
                unk_id = self.vocab.get("<UNK>", 1)  # Default to 1 if not found
                return [self.vocab.get(c, unk_id) for c in text]
    
    def decode(self, ids: List[int]) -> str:
        """
        Decode token IDs to text
        
        Args:
            ids: List of token IDs
            
        Returns:
            Decoded text
        """
        if self.char_tokenizer:
            return self.char_tokenizer.decode(ids)
        else:
            return ''.join([self.id_to_token.get(id, "<UNK>") for id in ids])
    
    def save(self, save_dir: str):
        """
        Save tokenizer components
        
        Args:
            save_dir: Directory to save tokenizer
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save vocabulary
        with open(save_dir / "vocab.json", 'w', encoding='utf-8') as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)
        
        # Save config
        config = {
            "vocab_size": len(self.vocab),
            "has_sentencepiece": self.char_tokenizer is not None,
            "language": "amharic"
        }
        
        with open(save_dir / "config.json", 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"✓ Tokenizer saved to: {save_dir}")
    
    @classmethod
    def load(cls, load_dir: str, g2p=None):
        """
        Load tokenizer from directory
        
        Args:
            load_dir: Directory containing tokenizer
            g2p: G2P instance
            
        Returns:
            Loaded tokenizer instance
        """
        load_dir = Path(load_dir)
        
        tokenizer = cls(g2p=g2p)
        
        # Load vocabulary
        vocab_file = load_dir / "vocab.json"
        if vocab_file.exists():
            with open(vocab_file, 'r', encoding='utf-8') as f:
                tokenizer.vocab = json.load(f)
                tokenizer.id_to_token = {v: k for k, v in tokenizer.vocab.items()}
        
        # Load SentencePiece if exists
        sp_model = load_dir / "sentencepiece.model"
        if sp_model.exists():
            tokenizer.char_tokenizer = spm.SentencePieceProcessor()
            tokenizer.char_tokenizer.load(str(sp_model))
        
        return tokenizer
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size"""
        if self.char_tokenizer:
            return self.char_tokenizer.get_piece_size()
        return len(self.vocab)


# Training function
def train_amharic_tokenizer(data_file: str, output_dir: str, vocab_size: int = 500):
    """
    Complete tokenizer training pipeline
    
    Args:
        data_file: CSV file with 'text' column
        output_dir: Output directory for tokenizer
        vocab_size: Target vocabulary size
    """
    import pandas as pd
    from ..g2p.amharic_g2p import AmharicG2P
    
    # Load data
    print("Loading data...")
    
    # Try to load with different formats
    try:
        # First try pipe-delimited (LJSpeech format: filename|text|normalized_text)
        df = pd.read_csv(data_file, sep='|', header=None, names=['filename', 'text', 'normalized_text'])
        texts = df['text'].tolist()
        print(f"✓ Loaded LJSpeech format (pipe-delimited)")
    except:
        try:
            # Try comma-delimited with 'text' column
            df = pd.read_csv(data_file)
            if 'text' in df.columns:
                texts = df['text'].tolist()
            elif 'normalized_text' in df.columns:
                texts = df['normalized_text'].tolist()
            else:
                # Use first text column found
                text_cols = [col for col in df.columns if 'text' in col.lower()]
                if text_cols:
                    texts = df[text_cols[0]].tolist()
                else:
                    raise ValueError(f"No 'text' column found. Available columns: {df.columns.tolist()}")
            print(f"✓ Loaded CSV format")
        except Exception as e:
            raise ValueError(f"Could not load data file. Error: {e}")
    
    # Validate data
    if not texts:
        raise ValueError("No text data found in the file!")
    
    # Remove empty texts
    texts = [t for t in texts if t and str(t).strip()]
    
    if not texts:
        raise ValueError("All text entries are empty!")
    
    print(f"Training tokenizer on {len(texts)} samples...")
    
    # Warn if dataset is too small
    if len(texts) < 1000:
        print(f"\n⚠️  WARNING: Dataset is small ({len(texts)} samples)")
        print(f"   Recommended: >1000 samples for good tokenizer quality")
        print(f"   Current quality may be suboptimal.\n")
    
    print(f"Sample texts:")
    for i, text in enumerate(texts[:3]):
        print(f"  {i+1}. {text[:100]}...")
    
    # Initialize
    g2p = AmharicG2P()
    tokenizer = AmharicTokenizer(g2p=g2p)
    
    # Build vocabulary
    print("Building vocabulary...")
    vocab = tokenizer.build_vocab_from_texts(texts)
    print(f"✓ Vocabulary size: {len(vocab)}")
    
    # Train SentencePiece (on plain text, not phonemes)
    print("Training SentencePiece model...")
    tokenizer.train_sentencepiece(
        texts,
        model_prefix=f"{output_dir}/sentencepiece",
        vocab_size=vocab_size
    )
    
    # Save tokenizer
    tokenizer.save(output_dir)
    
    # Test encoding/decoding
    test_text = "ሰላም ለዓለም"
    
    print(f"\n{'='*50}")
    print("Testing tokenizer...")
    print(f"\n1. Direct encoding (graphemes):")
    encoded_direct = tokenizer.encode(test_text, use_phonemes=False)
    decoded_direct = tokenizer.decode(encoded_direct)
    print(f"   Original:  {test_text}")
    print(f"   Encoded:   {encoded_direct[:20]}{'...' if len(encoded_direct) > 20 else ''}")
    print(f"   Decoded:   {decoded_direct}")
    print(f"   Match: {'✓' if test_text in decoded_direct else '✗'}")
    
    print(f"\n2. With phoneme conversion:")
    phoneme_text = g2p.grapheme_to_phoneme(test_text)
    print(f"   Phonemes:  {phoneme_text}")
    encoded_phoneme = tokenizer.encode(test_text, use_phonemes=True)
    decoded_phoneme = tokenizer.decode(encoded_phoneme)
    print(f"   Encoded:   {encoded_phoneme[:20]}{'...' if len(encoded_phoneme) > 20 else ''}")
    print(f"   Decoded:   {decoded_phoneme}")
    
    print(f"{'='*50}")
    
    return tokenizer


if __name__ == "__main__":
    # Example usage
    print("Amharic Tokenizer Module")
    print("Use train_amharic_tokenizer() to train a new tokenizer")
