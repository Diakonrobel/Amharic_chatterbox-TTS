#!/usr/bin/env python3
"""
🩺 Quick UNK Token Diagnostic Test

This script provides immediate feedback on your UNK token issues.
Run this first to understand the problem before applying fixes.
"""

import sys
import json
from pathlib import Path

# Add project paths
sys.path.append('.')

def test_amharic_characters():
    """Test basic Amharic character handling"""
    print("🔍 TESTING AMHARIC CHARACTER SUPPORT")
    print("=" * 40)
    
    test_texts = [
        "ሰላም ለዓለም",  # Hello World
        "አዲስ አበባ",   # Addis Ababa
        "እንኳን ደህና መጡ",  # Welcome
    ]
    
    for text in test_texts:
        print(f"Text: '{text}'")
        
        # Check Unicode code points
        for char in text:
            if not char.isspace():
                code_point = ord(char)
                in_ethiopic = 0x1200 <= code_point <= 0x137F
                print(f"  '{char}' → U+{code_point:04X} ({'Ethiopic' if in_ethiopic else 'Other'})")
        print()

def test_g2p():
    """Test G2P conversion"""
    print("🔤 TESTING G2P CONVERSION")
    print("=" * 40)
    
    try:
        from src.g2p.amharic_g2p import AmharicG2P
        
        g2p = AmharicG2P()
        print("✓ G2P system loaded successfully")
        
        test_texts = ["ሰላም ለዓለም", "አማርኛ"]
        
        for text in test_texts:
            try:
                phonemes = g2p.grapheme_to_phoneme(text)
                sequence = g2p.text_to_sequence(text)
                print(f"'{text}' → '{phonemes}' → {sequence}")
            except Exception as e:
                print(f"❌ G2P failed for '{text}': {e}")
        
    except ImportError as e:
        print(f"❌ Cannot import G2P: {e}")
        return False
    
    return True

def test_tokenizer():
    """Test tokenizer loading and encoding"""
    print("\n🎯 TESTING TOKENIZER")
    print("=" * 40)
    
    try:
        from src.g2p.amharic_g2p import AmharicG2P
        from src.tokenizer.amharic_tokenizer import AmharicTokenizer
        
        g2p = AmharicG2P()
        
        # Try different tokenizer paths
        tokenizer_paths = [
            "models/tokenizer",
            "models/tokenizer/amharic_tokenizer"
        ]
        
        tokenizer = None
        for path in tokenizer_paths:
            if Path(path).exists():
                try:
                    tokenizer = AmharicTokenizer.load(path, g2p=g2p)
                    print(f"✓ Tokenizer loaded from: {path}")
                    break
                except Exception as e:
                    print(f"⚠ Failed to load from {path}: {e}")
        
        if not tokenizer:
            print("❌ No tokenizer found. You need to train one first!")
            print("\nRun this to train a tokenizer:")
            print("python scripts/fix_unk_issues.py --data-path data/srt_datasets/YOUR_DATASET --config config/training_config.yaml")
            return False
        
        # Test tokenization
        test_texts = ["ሰላም ለዓለም", "አዲስ አበባ የኢትዮጵያ ዋና ከተማ ናት"]
        
        for text in test_texts:
            try:
                # Test both modes
                tokens_grapheme = tokenizer.encode(text, use_phonemes=False)
                tokens_phoneme = tokenizer.encode(text, use_phonemes=True)
                
                print(f"Text: '{text}'")
                print(f"  Grapheme tokens: {tokens_grapheme[:10]}...")
                print(f"  Phoneme tokens: {tokens_phoneme[:10]}...")
                
                # Check for UNK tokens
                unk_id = tokenizer.vocab.get('<UNK>', 1)
                unk_ratio_g = tokens_grapheme.count(unk_id) / len(tokens_grapheme) if tokens_grapheme else 1.0
                unk_ratio_p = tokens_phoneme.count(unk_id) / len(tokens_phoneme) if tokens_phoneme else 1.0
                
                print(f"  UNK ratio - Grapheme: {unk_ratio_g:.1%}, Phoneme: {unk_ratio_p:.1%}")
                
                if unk_ratio_p > 0.3:
                    print(f"  🚨 HIGH UNK RATIO! This will cause training issues.")
                else:
                    print(f"  ✓ UNK ratio is acceptable")
                
            except Exception as e:
                print(f"❌ Tokenization failed for '{text}': {e}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Cannot import tokenizer: {e}")
        return False

def test_training_data():
    """Test training data format"""
    print("\n📊 TESTING TRAINING DATA")
    print("=" * 40)
    
    # Common data paths
    data_paths = [
        "data/srt_datasets",
        "data/processed", 
        "data/raw"
    ]
    
    found_datasets = []
    for base_path in data_paths:
        if Path(base_path).exists():
            for item in Path(base_path).iterdir():
                if item.is_dir():
                    metadata_file = item / "metadata.csv"
                    if metadata_file.exists():
                        found_datasets.append(item)
    
    if not found_datasets:
        print("❌ No datasets found with metadata.csv")
        print("Expected structure:")
        print("  data/srt_datasets/YOUR_DATASET/metadata.csv")
        print("  data/srt_datasets/YOUR_DATASET/wavs/")
        return False
    
    print(f"✓ Found {len(found_datasets)} dataset(s):")
    
    for dataset_path in found_datasets:
        print(f"\n📁 {dataset_path}")
        metadata_file = dataset_path / "metadata.csv"
        
        try:
            with open(metadata_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()[:5]  # Check first 5 lines
            
            print(f"  Entries: {len(lines)}")
            
            amharic_found = 0
            for i, line in enumerate(lines):
                parts = line.strip().split('|')
                if len(parts) >= 2:
                    text = parts[1]
                    amharic_chars = sum(1 for c in text if 0x1200 <= ord(c) <= 0x137F)
                    amharic_found += amharic_chars
                    
                    if i == 0:  # Show first line in detail
                        print(f"  Sample: '{text}' ({amharic_chars} Amharic chars)")
            
            if amharic_found == 0:
                print(f"  ❌ No Amharic characters found!")
                print(f"  🔍 Your data might be in English or incorrectly formatted")
            else:
                print(f"  ✓ Found {amharic_found} Amharic characters in samples")
            
        except Exception as e:
            print(f"  ❌ Cannot read metadata: {e}")
    
    return len(found_datasets) > 0

def main():
    print("🩺 AMHARIC TTS UNK TOKEN DIAGNOSTIC")
    print("=" * 50)
    print("This will quickly diagnose your UNK token issues\n")
    
    # Run tests
    tests_passed = 0
    total_tests = 4
    
    # Test 1: Basic Amharic character support
    test_amharic_characters()
    tests_passed += 1
    
    # Test 2: G2P system
    if test_g2p():
        tests_passed += 1
    
    # Test 3: Training data
    if test_training_data():
        tests_passed += 1
    
    # Test 4: Tokenizer
    if test_tokenizer():
        tests_passed += 1
    
    print(f"\n{'='*50}")
    print(f"SUMMARY: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed < total_tests:
        print("\n🔧 TO FIX THE ISSUES:")
        print("1. Run the comprehensive fixer:")
        print("   python scripts/fix_unk_issues.py --data-path data/srt_datasets/YOUR_DATASET --config config/training_config.yaml")
        print("\n2. Or follow these steps manually:")
        print("   - Train Amharic tokenizer on your data")
        print("   - Merge with Chatterbox base tokenizer") 
        print("   - Extend model embeddings")
        print("   - Update config with correct vocab sizes")
    else:
        print("\n🎉 All tests passed! Your setup should work for training.")

if __name__ == "__main__":
    main()