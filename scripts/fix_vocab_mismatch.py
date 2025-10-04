#!/usr/bin/env python3
"""
Fix Vocab Size Mismatch Between Extended Model and Tokenizer
Extends Chatterbox model embeddings to match tokenizer vocab size
"""

import torch
import sys
from pathlib import Path

def extend_embeddings_to_match_tokenizer(
    model_path: str,
    output_path: str,
    original_vocab_size: int,
    target_vocab_size: int
):
    """
    Extend model embeddings from original size to target size
    
    Args:
        model_path: Path to base model (.safetensors or .pt)
        output_path: Path to save extended model
        original_vocab_size: Original vocabulary size (e.g., 2454)
        target_vocab_size: Target vocabulary size (e.g., 2559)
    """
    
    print("="*70)
    print("🔧 FIXING VOCAB SIZE MISMATCH")
    print("="*70)
    print()
    
    # Load base model
    print(f"📥 Loading base model: {model_path}")
    
    if model_path.endswith('.safetensors'):
        try:
            from safetensors.torch import load_file
            state_dict = load_file(model_path)
            print("✓ Loaded .safetensors model")
        except ImportError:
            print("❌ safetensors library not found!")
            print("   Install: pip install safetensors")
            sys.exit(1)
    else:
        state_dict = torch.load(model_path, map_location='cpu', weights_only=False)
        if 'model' in state_dict:
            state_dict = state_dict['model']
        elif 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        print("✓ Loaded .pt model")
    
    print()
    
    # Find text embedding key
    embedding_key = None
    for key in state_dict.keys():
        if 'text' in key.lower() and 'emb' in key.lower():
            embedding_key = key
            break
    
    if not embedding_key:
        print("❌ Could not find text embedding in model!")
        print("   Available keys:")
        for key in list(state_dict.keys())[:10]:
            print(f"     - {key}")
        sys.exit(1)
    
    print(f"✓ Found embedding key: {embedding_key}")
    
    # Get original embeddings
    original_embeddings = state_dict[embedding_key]
    current_vocab_size, embedding_dim = original_embeddings.shape
    
    print()
    print(f"📊 Current State:")
    print(f"   Vocab size in model: {current_vocab_size}")
    print(f"   Target vocab size: {target_vocab_size}")
    print(f"   Embedding dimension: {embedding_dim}")
    print()
    
    if current_vocab_size == target_vocab_size:
        print("✅ Vocab sizes already match! No extension needed.")
        return
    
    if current_vocab_size > target_vocab_size:
        print(f"❌ Error: Current vocab ({current_vocab_size}) > target ({target_vocab_size})")
        print("   Cannot reduce vocab size!")
        sys.exit(1)
    
    # Calculate how many new embeddings to add
    num_new_tokens = target_vocab_size - current_vocab_size
    
    print(f"🔧 Extension Plan:")
    print(f"   Original embeddings (0-{original_vocab_size-1}): PRESERVED")
    print(f"   Extended embeddings ({original_vocab_size}-{current_vocab_size-1}): PRESERVED")
    print(f"   New embeddings ({current_vocab_size}-{target_vocab_size-1}): {num_new_tokens} tokens")
    print()
    
    # Create new embeddings (random initialization)
    print(f"🎲 Initializing {num_new_tokens} new embeddings...")
    new_embeddings = torch.randn(num_new_tokens, embedding_dim) * 0.02  # Small random values
    
    # Concatenate old and new embeddings
    extended_embeddings = torch.cat([original_embeddings, new_embeddings], dim=0)
    
    print(f"✓ Extended embeddings shape: {extended_embeddings.shape}")
    print()
    
    # Update state dict
    state_dict[embedding_key] = extended_embeddings
    
    # Save extended model
    print(f"💾 Saving extended model to: {output_path}")
    torch.save({
        'model': state_dict,
        'vocab_size': target_vocab_size,
        'embedding_dim': embedding_dim,
        'original_vocab_size': original_vocab_size,
        'extended_from': model_path
    }, output_path)
    
    print("✓ Model saved successfully!")
    print()
    
    # Verification
    print("="*70)
    print("✅ EXTENSION COMPLETE")
    print("="*70)
    print()
    print(f"Extended Model Info:")
    print(f"   Location: {output_path}")
    print(f"   Vocab Size: {current_vocab_size} → {target_vocab_size}")
    print(f"   New Tokens Added: {num_new_tokens}")
    print(f"   File Size: {Path(output_path).stat().st_size / (1024**3):.2f} GB")
    print()
    print(f"🎯 Next Steps:")
    print(f"   1. Update training config to use this model:")
    print(f"      finetuning:")
    print(f"        pretrained_model: {output_path}")
    print()
    print(f"   2. Verify vocab size in config matches:")
    print(f"      model:")
    print(f"        n_vocab: {target_vocab_size}")
    print()
    print(f"   3. Start training!")
    print("="*70)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Fix vocab size mismatch")
    parser.add_argument("--model", required=True, help="Path to base model")
    parser.add_argument("--output", required=True, help="Path to save extended model")
    parser.add_argument("--original-size", type=int, default=2454, 
                       help="Original pretrained vocab size (default: 2454)")
    parser.add_argument("--target-size", type=int, required=True,
                       help="Target vocab size (must match tokenizer)")
    
    args = parser.parse_args()
    
    # Validate
    if args.target_size <= args.original_size:
        print(f"❌ Error: target_size ({args.target_size}) must be > original_size ({args.original_size})")
        sys.exit(1)
    
    extend_embeddings_to_match_tokenizer(
        args.model,
        args.output,
        args.original_size,
        args.target_size
    )
