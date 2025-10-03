"""
Merge Amharic tokenizer with Chatterbox base tokenizer
Based on practical experience from Japanese-English multilingual training

This script:
1. Loads Chatterbox base tokenizer (704 tokens)
2. Loads Amharic tokenizer (500 tokens)  
3. Merges them by re-indexing Amharic tokens starting from 704
4. Ensures no duplicate tokens
5. Saves merged tokenizer
"""

import json
import argparse
from pathlib import Path
from typing import Dict, Set


def load_tokenizer_vocab(tokenizer_path: str) -> Dict:
    """Load tokenizer vocabulary from JSON file"""
    with open(tokenizer_path, 'r', encoding='utf-8') as f:
        vocab = json.load(f)
    return vocab


def merge_tokenizers(base_tokenizer_path: str, 
                     amharic_tokenizer_path: str,
                     output_path: str,
                     base_vocab_size: int = 704):
    """
    Merge Amharic tokenizer with Chatterbox base tokenizer
    
    Args:
        base_tokenizer_path: Path to base Chatterbox tokenizer vocab
        amharic_tokenizer_path: Path to Amharic tokenizer vocab
        output_path: Path to save merged tokenizer
        base_vocab_size: Size of base vocabulary (default 704)
    """
    
    print("="*60)
    print("TOKENIZER MERGING")
    print("="*60)
    
    # Load tokenizers
    print(f"\n[1/5] Loading base tokenizer from: {base_tokenizer_path}")
    base_vocab = load_tokenizer_vocab(base_tokenizer_path)
    print(f"      Base vocabulary size: {len(base_vocab)}")
    
    print(f"\n[2/5] Loading Amharic tokenizer from: {amharic_tokenizer_path}")
    amharic_vocab = load_tokenizer_vocab(amharic_tokenizer_path)
    print(f"      Amharic vocabulary size: {len(amharic_vocab)}")
    
    # Get existing tokens to avoid duplicates
    print(f"\n[3/5] Checking for overlapping tokens...")
    base_tokens = set(base_vocab.keys())
    amharic_tokens = set(amharic_vocab.keys())
    
    overlapping = base_tokens.intersection(amharic_tokens)
    if overlapping:
        print(f"      Warning: {len(overlapping)} overlapping tokens found")
        print(f"      These will not be added: {list(overlapping)[:10]}...")
    else:
        print(f"      ✓ No overlapping tokens")
    
    # Merge vocabularies
    print(f"\n[4/5] Merging vocabularies...")
    merged_vocab = base_vocab.copy()
    
    # Start indexing Amharic tokens after base tokens
    next_index = base_vocab_size
    added_count = 0
    
    for token, _ in amharic_vocab.items():
        # Skip if token already exists in base
        if token not in merged_vocab:
            merged_vocab[token] = next_index
            next_index += 1
            added_count += 1
    
    print(f"      ✓ Added {added_count} new Amharic tokens")
    print(f"      ✓ Total merged vocabulary size: {len(merged_vocab)}")
    
    # Save merged tokenizer
    print(f"\n[5/5] Saving merged tokenizer to: {output_path}")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(merged_vocab, f, ensure_ascii=False, indent=2)
    
    print(f"      ✓ Merged tokenizer saved")
    
    # Summary
    print("\n" + "="*60)
    print("MERGE SUMMARY")
    print("="*60)
    print(f"  Base tokens:     {len(base_vocab)}")
    print(f"  Amharic tokens:  {len(amharic_vocab)}")
    print(f"  New tokens added: {added_count}")
    print(f"  Total tokens:    {len(merged_vocab)}")
    print(f"  Output:          {output_path}")
    print("="*60)
    
    return merged_vocab


def validate_merged_tokenizer(merged_vocab_path: str):
    """
    Validate the merged tokenizer
    
    Args:
        merged_vocab_path: Path to merged tokenizer
    """
    print("\n" + "="*60)
    print("VALIDATION")
    print("="*60)
    
    vocab = load_tokenizer_vocab(merged_vocab_path)
    
    # Check for duplicate indices
    indices = list(vocab.values())
    if len(indices) != len(set(indices)):
        print("  ✗ ERROR: Duplicate indices found!")
        return False
    else:
        print("  ✓ No duplicate indices")
    
    # Check index continuity
    sorted_indices = sorted(indices)
    expected = list(range(len(indices)))
    if sorted_indices != expected:
        print(f"  ⚠ Warning: Indices not continuous")
        print(f"    Expected: 0 to {len(indices)-1}")
        print(f"    Found: {sorted_indices[0]} to {sorted_indices[-1]}")
    else:
        print("  ✓ Indices are continuous")
    
    print(f"  ✓ Vocabulary size: {len(vocab)}")
    print("="*60)
    
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Merge Amharic tokenizer with Chatterbox base tokenizer"
    )
    parser.add_argument(
        '--base', 
        type=str,
        required=True,
        help='Path to base Chatterbox tokenizer vocab.json'
    )
    parser.add_argument(
        '--amharic',
        type=str,
        required=True,
        help='Path to Amharic tokenizer vocab.json'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Path to save merged tokenizer vocab.json'
    )
    parser.add_argument(
        '--base-size',
        type=int,
        default=704,
        help='Base tokenizer vocabulary size (default: 704)'
    )
    parser.add_argument(
        '--validate',
        action='store_true',
        help='Validate merged tokenizer'
    )
    
    args = parser.parse_args()
    
    # Merge tokenizers
    merged_vocab = merge_tokenizers(
        args.base,
        args.amharic,
        args.output,
        args.base_size
    )
    
    # Validate if requested
    if args.validate:
        validate_merged_tokenizer(args.output)
