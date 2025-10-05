#!/usr/bin/env python3
"""
Comprehensive analysis of why extended model didn't load during training
"""

import torch
import json
from pathlib import Path

print("="*70)
print("🔍 ANALYZING MODEL LOADING FAILURE")
print("="*70)
print()

# Check extended models
print("Step 1: Checking Extended Models")
print("-"*70)

models_to_check = [
    "models/pretrained/chatterbox_extended.pt",
    "models/pretrained/chatterbox_extended_2559.pt"
]

model_info = {}

for model_path in models_to_check:
    if Path(model_path).exists():
        print(f"\n✓ Found: {model_path}")
        try:
            ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
            
            # Get embedding size
            if 'model' in ckpt and 'text_emb.weight' in ckpt['model']:
                emb = ckpt['model']['text_emb.weight']
                vocab_size, d_model = emb.shape
                
                model_info[model_path] = {
                    'vocab_size': vocab_size,
                    'd_model': d_model,
                    'keys': list(ckpt.keys())
                }
                
                print(f"  Vocab size: {vocab_size}")
                print(f"  Embedding dim: {d_model}")
                print(f"  Top-level keys: {list(ckpt.keys())}")
            else:
                print(f"  ⚠ Unexpected structure")
                print(f"  Keys: {list(ckpt.keys())}")
                
        except Exception as e:
            print(f"  ❌ Error loading: {e}")
    else:
        print(f"\n❌ Not found: {model_path}")

print()

# Check tokenizer
print("Step 2: Checking Tokenizer")
print("-"*70)

tokenizer_paths = [
    "models/tokenizer/Am_tokenizer_merged.json",
    "models/tokenizer/am-merged_merged.json",
    "models/tokenizer/amharic_tokenizer/vocab.json"
]

tokenizer_info = {}

for tok_path in tokenizer_paths:
    if Path(tok_path).exists():
        print(f"\n✓ Found: {tok_path}")
        try:
            with open(tok_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Get vocab
            if 'vocab' in data:
                vocab = data['vocab']
            elif 'model' in data and 'vocab' in data['model']:
                vocab = data['model']['vocab']
            else:
                vocab = {}
            
            tokenizer_info[tok_path] = len(vocab)
            print(f"  Vocab size: {len(vocab)}")
            
        except Exception as e:
            print(f"  ❌ Error reading: {e}")
    else:
        print(f"\n⚠ Not found: {tok_path}")

print()

# Check configs
print("Step 3: Checking Config Files")
print("-"*70)

config_paths = [
    "config/training_config.yaml",
    "config/training_config_finetune_FIXED.yaml",
    "config/temp_training_config.yaml"
]

import re

for config_path in config_paths:
    if Path(config_path).exists():
        print(f"\n✓ Found: {config_path}")
        with open(config_path, 'r') as f:
            content = f.read()
        
        # Extract key settings
        vocab_match = re.search(r'n_vocab:\s*(\d+)', content)
        lr_match = re.search(r'learning_rate:\s*([\d.e-]+)', content)
        pretrained_match = re.search(r'pretrained_model:\s*["\']?([^"\'\n]+)', content)
        
        if vocab_match:
            print(f"  n_vocab: {vocab_match.group(1)}")
        if lr_match:
            print(f"  learning_rate: {lr_match.group(1)}")
        if pretrained_match:
            print(f"  pretrained_model: {pretrained_match.group(1)}")
    else:
        print(f"\n⚠ Not found: {config_path}")

print()

# Analyze the mismatch
print("="*70)
print("📊 MISMATCH ANALYSIS")
print("="*70)
print()

print("Models:")
for path, info in model_info.items():
    print(f"  {Path(path).name}: {info['vocab_size']} tokens")

print()
print("Tokenizers:")
for path, size in tokenizer_info.items():
    print(f"  {Path(path).name}: {size} tokens")

print()

# Identify the problem
print("="*70)
print("🎯 ROOT CAUSE ANALYSIS")
print("="*70)
print()

if model_info and tokenizer_info:
    # Get most common sizes
    model_sizes = [info['vocab_size'] for info in model_info.values()]
    tok_sizes = list(tokenizer_info.values())
    
    print("Issue Detection:")
    print()
    
    # Check if any match
    matching = False
    for model_path, info in model_info.items():
        for tok_path, tok_size in tokenizer_info.items():
            if info['vocab_size'] == tok_size:
                print(f"✅ MATCH FOUND:")
                print(f"   Model: {Path(model_path).name} ({info['vocab_size']})")
                print(f"   Tokenizer: {Path(tok_path).name} ({tok_size})")
                matching = True
    
    if not matching:
        print("❌ NO MATCHES FOUND!")
        print()
        print("This explains why model didn't load:")
        print("  - Training tried to create model with tokenizer size")
        print("  - Extended model has different size")
        print("  - Size mismatch → weights can't load")
        print("  - Falls back to random initialization")
        print()
        print("Solution:")
        if 2559 in tok_sizes:
            print("  ✓ Use: chatterbox_extended_2559.pt")
            print("  ✓ With: am-merged_merged.json tokenizer")
        elif 2535 in tok_sizes:
            print("  ✓ Use: chatterbox_extended.pt")
            print("  ✓ Create matching tokenizer (2535 tokens)")

print()
print("="*70)
print("✅ ANALYSIS COMPLETE")
print("="*70)
