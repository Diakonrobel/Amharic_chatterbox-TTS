#!/usr/bin/env python3
"""
Auto-detect correct vocab size from tokenizer and update all configs
This prevents vocab size mismatches between training and inference
"""

import json
import sys
from pathlib import Path
import re

def find_tokenizer():
    """Find the merged tokenizer"""
    tokenizer_paths = [
        Path("models/tokenizer/Am_tokenizer_merged.json"),
        Path("models/tokenizer/am-merged_merged.json"),
        Path("models/tokenizer/amharic_tokenizer/vocab.json"),
    ]
    
    for path in tokenizer_paths:
        if path.exists():
            return path
    return None

def get_vocab_size_from_tokenizer(tokenizer_path):
    """Get vocab size from tokenizer JSON"""
    try:
        with open(tokenizer_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Try different JSON structures
        if 'vocab' in data:
            vocab = data['vocab']
        elif 'model' in data and 'vocab' in data['model']:
            vocab = data['model']['vocab']
        else:
            print(f"⚠️ Unknown tokenizer format: {tokenizer_path}")
            return None
        
        return len(vocab)
    except Exception as e:
        print(f"❌ Error reading tokenizer: {e}")
        return None

def update_config_vocab_size(config_path, new_vocab_size):
    """Update n_vocab in a YAML config file"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Find current vocab size
        match = re.search(r'n_vocab:\s*(\d+)', content)
        if not match:
            print(f"⚠️ Could not find n_vocab in {config_path}")
            return False
        
        old_vocab_size = int(match.group(1))
        
        if old_vocab_size == new_vocab_size:
            print(f"✓ {config_path.name}: Already correct ({new_vocab_size})")
            return True
        
        # Replace vocab size
        new_content = re.sub(
            r'n_vocab:\s*\d+',
            f'n_vocab: {new_vocab_size}',
            content
        )
        
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        print(f"✓ {config_path.name}: Updated {old_vocab_size} → {new_vocab_size}")
        return True
        
    except Exception as e:
        print(f"❌ Error updating {config_path}: {e}")
        return False

def main():
    print("="*70)
    print("🔍 AUTO-DETECTING VOCAB SIZE")
    print("="*70)
    print()
    
    # Find tokenizer
    print("Step 1: Finding tokenizer...")
    tokenizer_path = find_tokenizer()
    
    if not tokenizer_path:
        print("❌ No tokenizer found!")
        print("   Checked:")
        print("   - models/tokenizer/Am_tokenizer_merged.json")
        print("   - models/tokenizer/am-merged_merged.json")
        print("   - models/tokenizer/amharic_tokenizer/vocab.json")
        sys.exit(1)
    
    print(f"✓ Found tokenizer: {tokenizer_path}")
    print()
    
    # Get vocab size
    print("Step 2: Reading tokenizer vocab size...")
    vocab_size = get_vocab_size_from_tokenizer(tokenizer_path)
    
    if not vocab_size:
        print("❌ Could not determine vocab size!")
        sys.exit(1)
    
    print(f"✓ Tokenizer vocab size: {vocab_size}")
    print()
    
    # Update config files
    print("Step 3: Updating config files...")
    config_files = [
        Path("config/training_config.yaml"),
        Path("config/training_config_finetune_FIXED.yaml"),
        Path("config/training_config_stable.yaml"),
    ]
    
    updated = 0
    for config_path in config_files:
        if config_path.exists():
            if update_config_vocab_size(config_path, vocab_size):
                updated += 1
        else:
            print(f"⚠️ {config_path.name}: Not found")
    
    print()
    print("="*70)
    print("✅ VOCAB SIZE SYNC COMPLETE")
    print("="*70)
    print()
    print(f"Updated {updated} config files to vocab size: {vocab_size}")
    print()
    print("🎯 Next steps:")
    print("   1. Restart Gradio app")
    print("   2. Test inference - vocab sizes should now match!")
    print("="*70)

if __name__ == "__main__":
    main()
