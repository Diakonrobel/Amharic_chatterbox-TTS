"""
Extend Chatterbox T3 Model Embeddings
Based on practical experience from multilingual training

This script extends the text embedding table in Chatterbox's T3 model
to accommodate additional tokens from the Amharic tokenizer.

Key points from video experience:
- Original Chatterbox T3 has 704 token embeddings
- We extend this to support merged vocabulary (e.g., 2000 tokens)
- New embeddings are randomly initialized
- Original embeddings can be frozen during training to preserve English
"""

import torch
import argparse
from pathlib import Path
from typing import Dict


def load_chatterbox_model(model_path: str) -> Dict:
    """Load Chatterbox T3 model checkpoint"""
    print(f"Loading model from: {model_path}")
    checkpoint = torch.load(model_path, map_location='cpu')
    return checkpoint


def extend_text_embeddings(checkpoint: Dict, 
                           original_vocab_size: int,
                           new_vocab_size: int) -> Dict:
    """
    Extend text embedding table in the model
    
    Args:
        checkpoint: Model checkpoint dictionary
        original_vocab_size: Original vocabulary size (e.g., 704)
        new_vocab_size: New vocabulary size (e.g., 2000)
        
    Returns:
        Modified checkpoint with extended embeddings
    """
    
    print("\n" + "="*60)
    print("EXTENDING TEXT EMBEDDINGS")
    print("="*60)
    
    # Find embedding layers in the model
    # Common key names in Chatterbox T3 model
    embedding_keys = [
        'model.text_embedding.weight',
        'text_embedding.weight',
        'encoder.text_embedding.weight',
        'model.encoder.text_embedding.weight'
    ]
    
    found_key = None
    for key in embedding_keys:
        if key in checkpoint.get('model', checkpoint):
            found_key = key
            break
    
    if not found_key:
        # Search for any key containing 'text' and 'embedding'
        model_dict = checkpoint.get('model', checkpoint)
        for key in model_dict.keys():
            if 'text' in key.lower() and 'embedding' in key.lower():
                found_key = key
                break
    
    if not found_key:
        print("  ✗ ERROR: Could not find text embedding layer in model")
        print("  Available keys:", list(checkpoint.get('model', checkpoint).keys())[:10])
        return None
    
    print(f"\n[1/3] Found text embedding: {found_key}")
    
    # Get original embeddings
    if 'model' in checkpoint:
        original_embeddings = checkpoint['model'][found_key]
    else:
        original_embeddings = checkpoint[found_key]
    
    original_shape = original_embeddings.shape
    print(f"      Original shape: {original_shape}")
    print(f"      Original vocab size: {original_shape[0]}")
    print(f"      Embedding dimension: {original_shape[1]}")
    
    if original_shape[0] != original_vocab_size:
        print(f"      ⚠ Warning: Expected {original_vocab_size}, found {original_shape[0]}")
    
    # Create extended embeddings
    print(f"\n[2/3] Creating extended embeddings...")
    print(f"      New vocabulary size: {new_vocab_size}")
    
    embedding_dim = original_shape[1]
    extended_embeddings = torch.nn.Embedding(new_vocab_size, embedding_dim)
    
    # Copy original embeddings
    with torch.no_grad():
        extended_embeddings.weight[:original_vocab_size] = original_embeddings
        # New embeddings are randomly initialized by default
        print(f"      ✓ Copied {original_vocab_size} original embeddings")
        print(f"      ✓ Initialized {new_vocab_size - original_vocab_size} new embeddings")
    
    # Update checkpoint
    print(f"\n[3/3] Updating checkpoint...")
    new_embeddings = extended_embeddings.weight
    
    if 'model' in checkpoint:
        checkpoint['model'][found_key] = new_embeddings
    else:
        checkpoint[found_key] = new_embeddings
    
    print(f"      ✓ Updated embedding shape: {new_embeddings.shape}")
    
    print("\n" + "="*60)
    print("EXTENSION SUMMARY")
    print("="*60)
    print(f"  Original vocab size: {original_vocab_size}")
    print(f"  New vocab size:      {new_vocab_size}")
    print(f"  Embedding dimension: {embedding_dim}")
    print(f"  New embeddings added: {new_vocab_size - original_vocab_size}")
    print("="*60)
    
    return checkpoint


def save_extended_model(checkpoint: Dict, output_path: str):
    """Save extended model checkpoint"""
    print(f"\nSaving extended model to: {output_path}")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    torch.save(checkpoint, output_path)
    print("✓ Model saved successfully")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extend Chatterbox T3 model embeddings for new vocabulary"
    )
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to original Chatterbox T3 model checkpoint'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Path to save extended model'
    )
    parser.add_argument(
        '--original-size',
        type=int,
        default=704,
        help='Original vocabulary size (default: 704)'
    )
    parser.add_argument(
        '--new-size',
        type=int,
        required=True,
        help='New vocabulary size after merging (e.g., 2000)'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.new_size <= args.original_size:
        print(f"ERROR: New size ({args.new_size}) must be larger than original size ({args.original_size})")
        exit(1)
    
    # Load model
    checkpoint = load_chatterbox_model(args.model)
    
    # Extend embeddings
    extended_checkpoint = extend_text_embeddings(
        checkpoint,
        args.original_size,
        args.new_size
    )
    
    if extended_checkpoint:
        # Save extended model
        save_extended_model(extended_checkpoint, args.output)
        
        print("\n" + "="*60)
        print("SUCCESS!")
        print("="*60)
        print("\nNext steps:")
        print("1. Update your training config to use the extended model")
        print("2. Set vocab_size in config to", args.new_size)
        print("3. Freeze original embeddings during training if needed")
        print("="*60)
    else:
        print("\n✗ Failed to extend model embeddings")
        exit(1)
