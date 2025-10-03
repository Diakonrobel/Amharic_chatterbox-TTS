"""
Amharic G2P (Grapheme-to-Phoneme) Converter
Converts Amharic text in Ge'ez/Ethiopic script to phonemes
"""

import re
import json
from typing import List, Dict, Tuple
from pathlib import Path
import unicodedata


class AmharicG2P:
    """
    Advanced Amharic G2P converter
    
    Amharic uses the Ge'ez/Ethiopic script (ግዕዝ)
    - Each character represents a consonant-vowel (CV) combination
    - 7 orders (forms) for each base character
    """
    
    def __init__(self, custom_rules_path: str = None):
        """Initialize Amharic G2P converter"""
        
        # Amharic phoneme inventory
        self.phonemes = {
            'consonants': [
                'p', 'b', 't', 'd', 'k', 'g', 'ʔ',  # Plosives
                'ts', 'dz', 'tʃ', 'dʒ',  # Affricates
                'f', 'v', 's', 'z', 'ʃ', 'ʒ', 'h',  # Fricatives
                'm', 'n', 'ɲ', 'ŋ',  # Nasals
                'l', 'r', 'j', 'w',  # Approximants
                "p'", "t'", "ts'", "tʃ'", "k'",  # Ejectives
            ],
            'vowels': ['ə', 'a', 'i', 'u', 'e', 'o']
        }
        
        # Load custom rules
        self.custom_rules = self._load_custom_rules(custom_rules_path)
        
        # Build character to IPA mapping
        self.char_to_ipa = self._build_char_mapping()
    
    def _build_char_mapping(self) -> Dict[str, str]:
        """
        Build mapping of Amharic characters to IPA
        
        Amharic Fidel structure:
        - 33 base consonants × 7 orders (vowel modifications)
        - Orders: ə, u, i, a, e, ɨ/ə, o
        """
        mapping = {}
        
        # Base consonants and their IPA equivalents
        base_consonants = {
            # ሀ-ህ: h-series
            'ሀ': 'h', 'ለ': 'l', 'ሐ': 'h', 'መ': 'm',
            'ሠ': 's', 'ረ': 'r', 'ሰ': 's', 'ሸ': 'ʃ',
            'ቀ': 'k', 'ቐ': 'k', 'በ': 'b', 'ቨ': 'v',
            'ተ': 't', 'ቸ': 'tʃ', 'ኀ': 'h', 'ነ': 'n',
            'ኘ': 'ɲ', 'አ': 'ʔ', 'ከ': 'k', 'ኸ': 'k',
            'ወ': 'w', 'ዘ': 'z', 'ዠ': 'ʒ', 'የ': 'j',
            'ደ': 'd', 'ጀ': 'dʒ', 'ገ': 'g', 'ጠ': "t'",
            'ጨ': "tʃ'", 'ጰ': "p'", 'ጸ': "ts'", 'ፀ': "ts'",
            'ፈ': 'f', 'ፐ': 'p',
        }
        
        # Vowel orders
        vowel_orders = ['ə', 'u', 'i', 'a', 'e', 'ɨ', 'o']
        
        # Build full mapping for each base consonant
        for base_char, ipa_consonant in base_consonants.items():
            base_code = ord(base_char)
            for i, vowel in enumerate(vowel_orders):
                char = chr(base_code + i)
                # Combine consonant + vowel
                mapping[char] = ipa_consonant + vowel
        
        return mapping
    
    def _load_custom_rules(self, rules_path: str = None) -> List[Tuple]:
        """Load custom pronunciation rules"""
        if rules_path and Path(rules_path).exists():
            with open(rules_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        
        # Default rules for Amharic phonology
        return [
            # Gemination (double consonants)
            (r'([ልምርስንብ])\1', r'\1ː'),
            # Palatalization
            (r'([kg])i', r'c'),
            # Assimilations
            (r'nb', r'mb'),
            (r'nk', r'ŋk'),
        ]
    
    def grapheme_to_phoneme(self, text: str) -> str:
        """
        Convert Amharic text to IPA phonemes
        
        Args:
            text: Amharic text in Ethiopic script
            
        Returns:
            IPA phoneme representation
        """
        # Normalize Unicode
        text = unicodedata.normalize('NFC', text)
        
        # Remove punctuation
        text = re.sub(r'[።፣፤፥፦፧፨\d]+', ' ', text)
        
        phonemes = []
        
        for char in text:
            if char.isspace():
                phonemes.append(' ')
            elif char in self.char_to_ipa:
                phonemes.append(self.char_to_ipa[char])
            else:
                # Keep unknown characters as-is
                phonemes.append(char)
        
        result = ''.join(phonemes)
        
        # Apply custom phonological rules
        for pattern, replacement in self.custom_rules:
            result = re.sub(pattern, replacement, result)
        
        return result.strip()
    
    def text_to_sequence(self, text: str) -> List[str]:
        """
        Convert text to phoneme sequence (list format for TTS)
        
        Args:
            text: Amharic text
            
        Returns:
            List of phonemes
        """
        ipa_text = self.grapheme_to_phoneme(text)
        
        # Split into individual phonemes
        phonemes = []
        i = 0
        while i < len(ipa_text):
            # Check for multi-character phonemes
            if i < len(ipa_text) - 1:
                two_char = ipa_text[i:i+2]
                # Check ejectives and affricates
                if two_char in self.phonemes['consonants']:
                    phonemes.append(two_char)
                    i += 2
                    continue
            
            # Single character
            if ipa_text[i] != ' ':
                phonemes.append(ipa_text[i])
            else:
                phonemes.append('_')  # Space marker
            i += 1
        
        return phonemes
    
    def build_phoneme_dict(self, texts: List[str], output_path: str):
        """
        Build phoneme dictionary from texts
        
        Args:
            texts: List of Amharic texts
            output_path: Path to save dictionary
        """
        phoneme_dict = {}
        
        for text in texts:
            words = text.split()
            for word in words:
                if word not in phoneme_dict:
                    phonemes = self.grapheme_to_phoneme(word)
                    phoneme_dict[word] = phonemes
        
        # Save dictionary
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(phoneme_dict, f, ensure_ascii=False, indent=2)
        
        print(f"Built phoneme dictionary with {len(phoneme_dict)} entries")
        return phoneme_dict
    
    def get_phoneme_set(self) -> List[str]:
        """Get complete phoneme set for Amharic"""
        all_phonemes = (
            self.phonemes['consonants'] + 
            self.phonemes['vowels'] + 
            ['_', ' ']  # Special tokens
        )
        return sorted(set(all_phonemes))


# Example usage
if __name__ == "__main__":
    g2p = AmharicG2P()
    
    # Test conversion
    test_text = "ሰላም ለዓለም"  # "Hello World" in Amharic
    
    print(f"Text: {test_text}")
    print(f"IPA: {g2p.grapheme_to_phoneme(test_text)}")
    print(f"Sequence: {g2p.text_to_sequence(test_text)}")
    print(f"Phoneme set: {g2p.get_phoneme_set()}")
