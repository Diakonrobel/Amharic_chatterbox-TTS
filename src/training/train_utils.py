"""
Training Utilities for Amharic TTS
Includes embedding freezing logic based on video experience
"""

import torch
import torch.nn as nn
from typing import Dict


def freeze_text_embeddings(model: nn.Module, freeze_until_index: int):
    """
    Freeze text embeddings up to a certain index
    This preserves learned English tokens while training new Amharic tokens
    
    Based on practical experience from video:
    - Freeze first 704 tokens (English)
    - Allow training of tokens 704+ (Amharic)
    
    Args:
        model: TTS model
        freeze_until_index: Index up to which to freeze embeddings (e.g., 704)
    """
    print(f"\n{'='*60}")
    print("FREEZING TEXT EMBEDDINGS")
    print(f"{'='*60}")
    print(f"  Freezing embeddings 0 to {freeze_until_index-1}")
    
    # Find text embedding layer
    embedding_layer = None
    embedding_param = None
    
    for name, param in model.named_parameters():
        if 'text' in name.lower() and 'embedding' in name.lower():
            if param.dim() == 2:  # Embedding layer is 2D (vocab_size, embedding_dim)
                embedding_layer = name
                embedding_param = param
                break
    
    if embedding_param is None:
        print("  ⚠ Warning: Could not find text embedding layer")
        return
    
    print(f"  Found embedding: {embedding_layer}")
    print(f"  Shape: {embedding_param.shape}")
    
    # Register hook to zero out gradients for frozen embeddings
    def freeze_hook(grad):
        """Zero out gradients for indices < freeze_until_index"""
        mask = torch.zeros_like(grad)
        mask[freeze_until_index:] = 1.0  # Only allow gradients for new tokens
        return grad * mask
    
    embedding_param.register_hook(freeze_hook)
    
    print(f"  ✓ Registered gradient hook")
    print(f"  ✓ Frozen: 0-{freeze_until_index-1} ({freeze_until_index} tokens)")
    print(f"  ✓ Trainable: {freeze_until_index}-{embedding_param.shape[0]-1} "
          f"({embedding_param.shape[0] - freeze_until_index} tokens)")
    print(f"{'='*60}\n")


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """
    Count model parameters
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'frozen': frozen_params
    }


def print_model_info(model: nn.Module, config: Dict):
    """
    Print model information
    
    Args:
        model: TTS model
        config: Training configuration
    """
    print("\n" + "="*60)
    print("MODEL INFORMATION")
    print("="*60)
    
    params = count_parameters(model)
    print(f"  Total parameters:     {params['total']:,}")
    print(f"  Trainable parameters: {params['trainable']:,}")
    print(f"  Frozen parameters:    {params['frozen']:,}")
    
    if config['model'].get('freeze_original_embeddings'):
        freeze_idx = config['model']['freeze_until_index']
        print(f"\n  Embedding freezing enabled:")
        print(f"    Frozen tokens: 0-{freeze_idx-1}")
        print(f"    Trainable tokens: {freeze_idx}-{config['model']['n_vocab']-1}")
    
    print("="*60 + "\n")


def save_checkpoint(model: nn.Module, optimizer, epoch: int, step: int,
                   loss: float, save_path: str):
    """
    Save training checkpoint
    
    Args:
        model: TTS model
        optimizer: Optimizer
        epoch: Current epoch
        step: Current step
        loss: Current loss
        save_path: Path to save checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    
    torch.save(checkpoint, save_path)
    print(f"✓ Checkpoint saved: {save_path}")


def load_checkpoint(model: nn.Module, checkpoint_path: str, 
                   optimizer=None, device='cpu'):
    """
    Load training checkpoint
    
    Args:
        model: TTS model
        checkpoint_path: Path to checkpoint
        optimizer: Optimizer (optional)
        device: Device to load to
        
    Returns:
        (epoch, step, loss)
    """
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint.get('epoch', 0)
    step = checkpoint.get('step', 0)
    loss = checkpoint.get('loss', 0.0)
    
    print(f"✓ Checkpoint loaded (Epoch: {epoch}, Step: {step}, Loss: {loss:.4f})")
    
    return epoch, step, loss


if __name__ == "__main__":
    print("Training utilities for Amharic TTS")
    print("Use these functions in your training script")
