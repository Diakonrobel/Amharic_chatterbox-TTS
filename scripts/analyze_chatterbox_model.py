"""
Analyze Chatterbox Model Architecture

This script inspects the extended Chatterbox model to understand:
- Model architecture and layers
- Input/output shapes
- Required parameters for forward pass
"""

import torch
import argparse
from pathlib import Path
from collections import OrderedDict


def analyze_model_structure(model_path: str):
    """Analyze the model checkpoint structure"""
    
    print("="*70)
    print("CHATTERBOX MODEL ARCHITECTURE ANALYZER")
    print("="*70)
    print(f"\nModel Path: {model_path}\n")
    
    # Load checkpoint
    print("Loading checkpoint...")
    try:
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    except TypeError:
        checkpoint = torch.load(model_path, map_location='cpu')
    
    print(f"✓ Loaded checkpoint\n")
    
    # Check checkpoint structure
    print("="*70)
    print("CHECKPOINT KEYS")
    print("="*70)
    for key in checkpoint.keys():
        if isinstance(checkpoint[key], dict):
            print(f"  {key}: dict with {len(checkpoint[key])} items")
        elif isinstance(checkpoint[key], torch.Tensor):
            print(f"  {key}: Tensor {checkpoint[key].shape}")
        else:
            print(f"  {key}: {type(checkpoint[key]).__name__}")
    
    # Get state dict
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
        print("\nUsing 'model' key from checkpoint")
    elif 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        print("\nUsing 'model_state_dict' key from checkpoint")
    else:
        state_dict = checkpoint
        print("\nCheckpoint IS the state dict")
    
    # Analyze layers
    print("\n" + "="*70)
    print("MODEL LAYERS STRUCTURE")
    print("="*70)
    
    # Group layers by module
    layer_groups = {}
    for key in state_dict.keys():
        parts = key.split('.')
        if len(parts) > 1:
            module = parts[0]
            if module not in layer_groups:
                layer_groups[module] = []
            layer_groups[module].append(key)
        else:
            if 'root' not in layer_groups:
                layer_groups['root'] = []
            layer_groups['root'].append(key)
    
    print(f"\nFound {len(layer_groups)} main modules:\n")
    for module, layers in sorted(layer_groups.items()):
        print(f"📦 {module}: {len(layers)} parameters")
        # Show first few layers as example
        for layer in sorted(layers)[:3]:
            shape = state_dict[layer].shape
            print(f"   └─ {layer}")
            print(f"      Shape: {shape}, dtype: {state_dict[layer].dtype}")
        if len(layers) > 3:
            print(f"   └─ ... and {len(layers) - 3} more")
        print()
    
    # Analyze embeddings
    print("="*70)
    print("EMBEDDING LAYERS")
    print("="*70)
    embedding_keys = ['text_emb', 'speech_emb', 'pos_emb', 'embedding']
    found_embeddings = []
    
    for key in state_dict.keys():
        if any(emb in key.lower() for emb in embedding_keys):
            found_embeddings.append((key, state_dict[key].shape))
    
    if found_embeddings:
        print(f"\n✓ Found {len(found_embeddings)} embedding layers:\n")
        for key, shape in found_embeddings:
            print(f"  {key}")
            print(f"    Shape: {shape}")
            if len(shape) == 2:
                print(f"    → vocab_size={shape[0]}, embed_dim={shape[1]}")
            print()
    
    # Analyze attention/transformer layers
    print("="*70)
    print("TRANSFORMER LAYERS")
    print("="*70)
    
    tfmr_layers = [k for k in state_dict.keys() if 'tfmr' in k.lower() or 'transformer' in k.lower()]
    if tfmr_layers:
        # Count layers
        layer_nums = set()
        for key in tfmr_layers:
            if 'layers.' in key:
                try:
                    layer_num = int(key.split('layers.')[1].split('.')[0])
                    layer_nums.add(layer_num)
                except:
                    pass
        
        print(f"\n✓ Found {len(layer_nums)} transformer layers\n")
        if layer_nums:
            print(f"  Layer indices: {sorted(layer_nums)}")
            
            # Show example layer structure
            if layer_nums:
                example_layer = min(layer_nums)
                example_keys = [k for k in tfmr_layers if f'layers.{example_layer}.' in k]
                print(f"\n  Example (Layer {example_layer}):")
                for key in sorted(example_keys)[:5]:
                    print(f"    {key}: {state_dict[key].shape}")
                if len(example_keys) > 5:
                    print(f"    ... and {len(example_keys) - 5} more")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    total_params = sum(p.numel() for p in state_dict.values())
    total_size_mb = sum(p.numel() * p.element_size() for p in state_dict.values()) / (1024**2)
    
    print(f"\n  Total Parameters: {total_params:,}")
    print(f"  Total Size: {total_size_mb:.2f} MB")
    print(f"  Number of Tensors: {len(state_dict)}")
    print(f"  Main Modules: {len(layer_groups)}")
    
    # Detect model type
    print("\n" + "="*70)
    print("MODEL TYPE DETECTION")
    print("="*70)
    
    has_speech = any('speech' in k.lower() for k in state_dict.keys())
    has_text = any('text' in k.lower() for k in state_dict.keys())
    has_tfmr = any('tfmr' in k.lower() for k in state_dict.keys())
    has_cond = any('cond' in k.lower() for k in state_dict.keys())
    
    print(f"\n  Speech components: {'✓' if has_speech else '✗'}")
    print(f"  Text components: {'✓' if has_text else '✗'}")
    print(f"  Transformer: {'✓' if has_tfmr else '✗'}")
    print(f"  Conditioning: {'✓' if has_cond else '✗'}")
    
    if has_speech and has_text and has_tfmr:
        print("\n  → Detected: Chatterbox T3 (Transformer-based TTS)")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze Chatterbox model architecture")
    parser.add_argument('--model', type=str, required=True, help='Path to model checkpoint')
    
    args = parser.parse_args()
    
    if not Path(args.model).exists():
        print(f"ERROR: Model file not found: {args.model}")
        exit(1)
    
    analyze_model_structure(args.model)
