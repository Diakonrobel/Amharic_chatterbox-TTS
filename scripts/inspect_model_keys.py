"""
Inspect Chatterbox Model Keys

This script inspects the structure of a Chatterbox model checkpoint
to identify the correct key names for text embeddings.
"""

import argparse
import torch
from pathlib import Path


def inspect_model_keys(model_path: str):
    """Inspect model checkpoint to find embedding layers"""
    
    print("="*70)
    print("CHATTERBOX MODEL CHECKPOINT INSPECTOR")
    print("="*70)
    print(f"\nModel Path: {model_path}\n")
    
    # Check if it's a safetensors file
    if model_path.endswith('.safetensors'):
        try:
            from safetensors import safe_open
            from safetensors.torch import load_file
            
            print("Loading safetensors file...")
            state_dict = load_file(model_path)
            
            print(f"✓ Loaded {len(state_dict)} tensors\n")
            
            # Wrap in checkpoint format for consistency
            checkpoint = {'model': state_dict}
            
        except ImportError:
            print("⚠ safetensors not installed. Install with: pip install safetensors")
            print("Trying torch.load instead...\n")
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    else:
        print("Loading PyTorch checkpoint...")
        try:
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        except TypeError:
            checkpoint = torch.load(model_path, map_location='cpu')
        print(f"✓ Loaded checkpoint\n")
    
    # Get the model state dict
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
        print("Found 'model' key in checkpoint")
    else:
        state_dict = checkpoint
        print("No 'model' key - checkpoint IS the state dict")
    
    print("\n" + "="*70)
    print("SEARCHING FOR TEXT/TOKEN EMBEDDING LAYERS")
    print("="*70)
    
    # Search for keys containing relevant keywords
    keywords = ['text', 'token', 'embedding', 'vocab', 'embed']
    found_keys = []
    
    for key in state_dict.keys():
        key_lower = key.lower()
        if any(keyword in key_lower for keyword in keywords):
            tensor = state_dict[key]
            found_keys.append((key, tensor.shape))
    
    if found_keys:
        print(f"\n✓ Found {len(found_keys)} relevant keys:\n")
        for key, shape in found_keys:
            print(f"  {key}")
            print(f"    Shape: {shape}")
            print(f"    Dimensions: {len(shape)}")
            if len(shape) == 2:
                print(f"    → Likely embedding: vocab_size={shape[0]}, embed_dim={shape[1]}")
            print()
    else:
        print("\n✗ No keys found containing: text, token, embedding, vocab")
    
    print("="*70)
    print("ALL 2D TENSORS (Potential Embedding Layers)")
    print("="*70)
    
    # Find all 2D tensors (embeddings are typically 2D)
    embedding_candidates = []
    
    for key in state_dict.keys():
        tensor = state_dict[key]
        if len(tensor.shape) == 2:
            embedding_candidates.append((key, tensor.shape))
    
    if embedding_candidates:
        print(f"\n✓ Found {len(embedding_candidates)} 2D tensors:\n")
        for key, shape in embedding_candidates[:20]:  # Show first 20
            print(f"  {key}")
            print(f"    Shape: {shape} (vocab_size={shape[0]}, dim={shape[1]})")
            print()
        
        if len(embedding_candidates) > 20:
            print(f"  ... and {len(embedding_candidates) - 20} more")
    else:
        print("\n✗ No 2D tensors found")
    
    print("\n" + "="*70)
    print("FIRST 30 KEYS IN MODEL")
    print("="*70)
    all_keys = list(state_dict.keys())
    for i, key in enumerate(all_keys[:30], 1):
        tensor = state_dict[key]
        print(f"{i:2d}. {key}")
        print(f"    Shape: {tensor.shape}, dtype: {tensor.dtype}")
    
    if len(all_keys) > 30:
        print(f"\n... and {len(all_keys) - 30} more keys")
    
    print("\n" + "="*70)
    print("INSPECTION COMPLETE")
    print("="*70)
    print(f"\nTotal tensors in model: {len(state_dict)}")
    print(f"Total 2D tensors: {len(embedding_candidates)}")
    print(f"Text/token-related keys: {len(found_keys)}")
    print("\n" + "="*70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Inspect Chatterbox model checkpoint structure"
    )
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to Chatterbox model checkpoint'
    )
    
    args = parser.parse_args()
    
    if not Path(args.model).exists():
        print(f"ERROR: Model file not found: {args.model}")
        exit(1)
    
    inspect_model_keys(args.model)
