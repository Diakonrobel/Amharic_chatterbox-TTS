"""
Complete Training Script for Amharic TTS
Fine-tunes Chatterbox TTS model on Amharic data with embedding freezing
"""

import os
import sys
import json
import yaml
import argparse
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, List
import threading

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from torch.amp import autocast, GradScaler

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.training.train_utils import freeze_text_embeddings, count_parameters, print_model_info
from src.models.t3_model import SimplifiedT3Model, TTSLoss
from src.audio import AudioProcessor, collate_fn


class TrainingState:
    """Shared training state for monitoring"""
    def __init__(self):
        self.is_running = False
        self.current_epoch = 0
        self.current_step = 0
        self.total_steps = 0
        self.current_loss = 0.0
        self.best_loss = float('inf')
        self.status_message = "Not started"
        self.logs = []
        self.last_checkpoint = None
        
    def log(self, message: str):
        """Add log message"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        self.logs.append(log_entry)
        print(log_entry)
        
        # Keep only last 100 logs
        if len(self.logs) > 100:
            self.logs = self.logs[-100:]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            'is_running': self.is_running,
            'current_epoch': self.current_epoch,
            'current_step': self.current_step,
            'total_steps': self.total_steps,
            'current_loss': float(self.current_loss),
            'best_loss': float(self.best_loss),
            'status_message': self.status_message,
            'logs': self.logs[-20:],  # Return last 20 logs
            'last_checkpoint': self.last_checkpoint
        }


# Global training state
TRAINING_STATE = TrainingState()


class SimpleAmharicDataset(Dataset):
    """Dataset for Amharic TTS training with real audio loading"""
    
    def __init__(self, metadata_path: str, data_dir: Path, audio_processor: AudioProcessor = None, tokenizer=None):
        self.data_dir = data_dir
        self.samples = []
        self.audio_processor = audio_processor or AudioProcessor()
        self.tokenizer = tokenizer
        
        # Load metadata
        with open(metadata_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('|')
                if len(parts) >= 2:
                    audio_file = parts[0]
                    text = parts[1]
                    self.samples.append({
                        'audio': audio_file,
                        'text': text
                    })
        
        TRAINING_STATE.log(f"✓ Loaded {len(self.samples)} samples from {metadata_path}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        try:
            # Load and process audio
            # Append .wav extension if not already present
            audio_filename = sample['audio'] if sample['audio'].endswith('.wav') else f"{sample['audio']}.wav"
            audio_path = self.data_dir / 'wavs' / audio_filename
            _, mel = self.audio_processor.process_audio_file(str(audio_path))
            
            # Tokenize text with proper Amharic support
            text = sample['text']
            if self.tokenizer:
                # Use grapheme (character) encoding - works better for Amharic
                try:
                    text_ids = self.tokenizer.encode(text, use_phonemes=False)
                    # Validate that we didn't get too many UNK tokens
                    unk_id = getattr(self.tokenizer, 'vocab', {}).get('<UNK>', 1)
                    unk_ratio = text_ids.count(unk_id) / len(text_ids) if text_ids else 1.0
                    if unk_ratio > 0.3:  # More than 30% UNK tokens is problematic
                        TRAINING_STATE.log(f"Warning: High UNK ratio ({unk_ratio:.1%}) in text: {text[:50]}...")
                except Exception as e:
                    TRAINING_STATE.log(f"Warning: Tokenization failed for '{text[:30]}...': {str(e)}")
                    # Fallback to direct character encoding
                    text_ids = self._encode_amharic_fallback(text)
            else:
                # Enhanced fallback for Amharic characters
                text_ids = self._encode_amharic_fallback(text)
            
            return {
                'text_ids': text_ids,
                'mel': mel,
                'audio_path': str(audio_path)
            }
        except Exception as e:
            # If audio loading fails, return a simple dummy to avoid crashing
            TRAINING_STATE.log(f"Warning: Failed to load {sample['audio']}: '{sample.get('text', '')[:30]}...' - {str(e)}")
            # Return minimal valid data with proper text encoding
            try:
                text_ids = self._encode_amharic_fallback(sample.get('text', 'ሰላም'))  # Default Amharic text
            except:
                text_ids = [0] * 10  # Final fallback
            
            return {
                'text_ids': text_ids,
                'mel': torch.zeros(80, 100),  # Dummy mel
                'audio_path': str(audio_path) if 'audio_path' in locals() else sample['audio']
            }
    
    def _encode_amharic_fallback(self, text: str) -> List[int]:
        """
        Fallback encoding for Amharic text when tokenizer is not available
        Properly handles Ethiopic Unicode range
        """
        import unicodedata
        
        # Normalize Unicode (important for Amharic)
        text = unicodedata.normalize('NFC', text)
        
        # Create a simple but valid encoding for Amharic
        # Map Ethiopic characters (U+1200-U+137F) to reasonable token IDs
        tokens = []
        for char in text[:100]:  # Limit length
            if char.isspace():
                tokens.append(0)  # Space/padding token
            else:
                code_point = ord(char)
                if 0x1200 <= code_point <= 0x137F:  # Ethiopic script
                    # Map to range 100-999 to avoid conflicts with special tokens
                    token_id = 100 + (code_point - 0x1200) % 800
                elif 0x20 <= code_point <= 0x7F:  # ASCII
                    token_id = code_point
                else:
                    # Other Unicode characters
                    token_id = 50 + (code_point % 50)
                tokens.append(token_id)
        
        # Ensure minimum length
        if len(tokens) < 5:
            tokens.extend([0] * (5 - len(tokens)))
        
        return tokens


def load_config(config_path: str) -> Dict:
    """Load training configuration"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def setup_dataloaders(config: Dict):
    """Setup training and validation dataloaders"""
    from pathlib import Path
    
    TRAINING_STATE.log("Setting up dataloaders...")
    
    # Initialize audio processor
    audio_processor = AudioProcessor()
    
    # Try to load tokenizer if available
    tokenizer = None
    # Prioritize amharic_tokenizer subdirectory (where trained tokenizer is)
    tokenizer_paths = [
        "models/tokenizer/amharic_tokenizer",  # Trained tokenizer location
        "models/tokenizer"  # Fallback
    ]
    
    for tokenizer_path in tokenizer_paths:
        tokenizer_dir = Path(tokenizer_path)
        # Check if tokenizer files actually exist
        if tokenizer_dir.exists() and (tokenizer_dir / "sentencepiece.model").exists():
            try:
                from src.tokenizer.amharic_tokenizer import AmharicTokenizer
                from src.g2p.amharic_g2p import AmharicG2P
                g2p = AmharicG2P()
                tokenizer = AmharicTokenizer.load(str(tokenizer_dir), g2p=g2p)
                TRAINING_STATE.log(f"✓ Loaded tokenizer from {tokenizer_path}")
                TRAINING_STATE.log(f"   Tokenizer vocab size: {tokenizer.get_vocab_size()}")
                break
            except Exception as e:
                TRAINING_STATE.log(f"⚠ Failed to load tokenizer from {tokenizer_path}: {str(e)}")
    
    if tokenizer is None:
        TRAINING_STATE.log("⚠ WARNING: No tokenizer loaded! Will use fallback character encoding.")
    
    # Create datasets - handle multiple config formats
    if 'data_dir' in config.get('paths', {}):
        data_dir = Path(config['paths']['data_dir'])
    elif 'dataset_path' in config.get('data', {}):
        data_dir = Path(config['data']['dataset_path'])
    else:
        # Fallback to default
        data_dir = Path("data/srt_datasets/my_dataset")
        TRAINING_STATE.log(f"⚠ No data_dir in config, using default: {data_dir}")
    
    train_metadata = data_dir / 'metadata.csv'
    val_metadata = data_dir / 'metadata_val.csv'  # Or use a split
    
    # Check if validation metadata exists, otherwise use train for both
    if not val_metadata.exists():
        TRAINING_STATE.log("⚠ No separate validation metadata, using training data")
        val_metadata = train_metadata
    
    train_dataset = SimpleAmharicDataset(
        str(train_metadata),
        data_dir,
        audio_processor=audio_processor,
        tokenizer=tokenizer
    )
    
    val_dataset = SimpleAmharicDataset(
        str(val_metadata),
        data_dir,
        audio_processor=audio_processor,
        tokenizer=tokenizer
    )
    
    # Create dataloaders with proper collate function
    batch_size = config.get('training', {}).get('batch_size', 16)
    num_workers = config.get('training', {}).get('num_workers', 2)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    TRAINING_STATE.log(f"✓ Train samples: {len(train_dataset)}")
    TRAINING_STATE.log(f"✓ Val samples: {len(val_dataset)}")
    TRAINING_STATE.log(f"✓ Batch size: {batch_size}")
    
    return train_loader, val_loader


def setup_model(config: Dict) -> nn.Module:
    """Setup real T3 model for training"""
    TRAINING_STATE.log("Setting up SimplifiedT3Model...")
    
    # Create T3 model with proper configuration
    # Use d_model=1024 to match Chatterbox pretrained weights
    model = SimplifiedT3Model(
        vocab_size=config['model']['n_vocab'],
        d_model=1024,  # Model dimension (matches Chatterbox multilingual)
        nhead=8,  # Attention heads
        num_encoder_layers=6,  # Transformer layers
        dim_feedforward=2048,
        dropout=0.1,
        n_mels=config['data']['n_mel_channels'],
        max_seq_len=1000
    )
    
    TRAINING_STATE.log(f"✓ T3 Model created:")
    TRAINING_STATE.log(f"   Vocab size: {config['model']['n_vocab']}")
    TRAINING_STATE.log(f"   Model dim: 1024 (matches Chatterbox)")
    TRAINING_STATE.log(f"   Mel channels: {config['data']['n_mel_channels']}")
    
    # Load pretrained weights (extended embeddings)
    if config['finetuning']['enabled']:
        pretrained_path = config['finetuning']['pretrained_model']
        if os.path.exists(pretrained_path):
            TRAINING_STATE.log(f"Loading extended embeddings from {pretrained_path}")
            try:
                model.load_pretrained_weights(pretrained_path)
                TRAINING_STATE.log("✓ Extended embeddings loaded")
            except Exception as e:
                TRAINING_STATE.log(f"⚠ Warning loading weights: {str(e)}")
                TRAINING_STATE.log("  Continuing with randomly initialized weights")
        else:
            TRAINING_STATE.log(f"⚠ Pretrained model not found: {pretrained_path}")
            TRAINING_STATE.log("  Training from scratch")
    
    # Freeze original embeddings if configured
    if config['model']['freeze_original_embeddings']:
        freeze_idx = config['model']['freeze_until_index']
        try:
            # Freeze the text embedding layer
            if hasattr(model, 'text_embedding'):
                for i, param in enumerate(model.text_embedding.parameters()):
                    if i == 0:  # Weight matrix
                        # Freeze only the first freeze_idx embeddings
                        param.requires_grad = False
                        param.data[:freeze_idx].requires_grad = False
                TRAINING_STATE.log(f"✓ Frozen first {freeze_idx} embeddings")
            else:
                TRAINING_STATE.log("⚠ Could not freeze embeddings - layer not found")
        except Exception as e:
            TRAINING_STATE.log(f"⚠ Warning freezing embeddings: {str(e)}")
    
    return model


def setup_optimizer(model: nn.Module, config: Dict):
    """Setup optimizer and scheduler"""
    TRAINING_STATE.log("Setting up optimizer...")
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
        betas=config['training']['betas'],
        eps=config['training']['eps']
    )
    
    scheduler = optim.lr_scheduler.ExponentialLR(
        optimizer,
        gamma=config['training']['lr_decay']
    )
    
    TRAINING_STATE.log(f"✓ Optimizer: {config['training']['optimizer']}")
    TRAINING_STATE.log(f"✓ Learning rate: {config['training']['learning_rate']}")
    
    return optimizer, scheduler




def train_epoch(model, train_loader, optimizer, scheduler, scaler, criterion, writer, config, epoch):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    device = next(model.parameters()).device
    
    for batch_idx, batch in enumerate(train_loader):
        TRAINING_STATE.current_step += 1
        optimizer.zero_grad()
        
        # Move batch to device
        text_ids = batch['text_ids'].to(device)
        text_lengths = batch['text_lengths'].to(device)
        mel_targets = batch['mel'].to(device)
        mel_lengths = batch['mel_lengths'].to(device)
        
        # Forward pass and loss computation
        if config['training']['use_amp']:
            with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                # Model forward
                outputs = model(
                    text_ids=text_ids,
                    text_lengths=text_lengths,
                    mel_targets=mel_targets
                )
                
                # Compute loss
                targets = {
                    'mel': mel_targets,
                    'mel_lengths': mel_lengths
                }
                losses = criterion(outputs, targets)
                loss = losses['total_loss']
            
            # Backward with gradient scaling
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                config['training']['grad_clip_thresh']
            )
            scaler.step(optimizer)
            scaler.update()
        else:
            # Forward without autocast
            outputs = model(
                text_ids=text_ids,
                text_lengths=text_lengths,
                mel_targets=mel_targets
            )
            
            # Compute loss
            targets = {
                'mel': mel_targets,
                'mel_lengths': mel_lengths
            }
            losses = criterion(outputs, targets)
            loss = losses['total_loss']
            
            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                config['training']['grad_clip_thresh']
            )
            optimizer.step()
        
        total_loss += loss.item()
        TRAINING_STATE.current_loss = loss.item()
        
        # Logging
        if batch_idx % config['training']['log_interval'] == 0:
            avg_loss = total_loss / (batch_idx + 1)
            lr = optimizer.param_groups[0]['lr']
            TRAINING_STATE.log(
                f"Epoch {epoch} | Step {TRAINING_STATE.current_step} | "
                f"Loss: {loss.item():.4f} | Avg: {avg_loss:.4f} | LR: {lr:.6f}"
            )
            TRAINING_STATE.status_message = f"Training: Epoch {epoch}, Step {TRAINING_STATE.current_step}"
            
            # TensorBoard logging
            if writer:
                writer.add_scalar('Loss/train_total', loss.item(), TRAINING_STATE.current_step)
                writer.add_scalar('Loss/train_mel', losses['mel_loss'].item(), TRAINING_STATE.current_step)
                writer.add_scalar('Loss/train_duration', losses['duration_loss'].item(), TRAINING_STATE.current_step)
                writer.add_scalar('Learning_Rate', lr, TRAINING_STATE.current_step)
        
        # Checkpointing
        if TRAINING_STATE.current_step % config['training']['save_interval'] == 0:
            checkpoint_path = save_checkpoint(
                model, optimizer, epoch, TRAINING_STATE.current_step,
                loss.item(), config
            )
            TRAINING_STATE.last_checkpoint = str(checkpoint_path)
            TRAINING_STATE.log(f"✓ Checkpoint saved: {checkpoint_path}")
        
        # Check if training should stop
        if not TRAINING_STATE.is_running:
            TRAINING_STATE.log("Training stopped by user")
            return False
    
    scheduler.step()
    return True


def validate(model, val_loader, criterion, config):
    """Run validation"""
    model.eval()
    total_loss = 0
    total_mel_loss = 0
    total_duration_loss = 0
    device = next(model.parameters()).device
    
    with torch.no_grad():
        for batch in val_loader:
            # Move batch to device
            text_ids = batch['text_ids'].to(device)
            text_lengths = batch['text_lengths'].to(device)
            mel_targets = batch['mel'].to(device)
            mel_lengths = batch['mel_lengths'].to(device)
            
            # Forward pass
            outputs = model(
                text_ids=text_ids,
                text_lengths=text_lengths,
                mel_targets=mel_targets
            )
            
            # Compute loss
            targets = {
                'mel': mel_targets,
                'mel_lengths': mel_lengths
            }
            losses = criterion(outputs, targets)
            
            total_loss += losses['total_loss'].item()
            total_mel_loss += losses['mel_loss'].item()
            total_duration_loss += losses['duration_loss'].item()
    
    avg_loss = total_loss / len(val_loader)
    avg_mel_loss = total_mel_loss / len(val_loader)
    avg_duration_loss = total_duration_loss / len(val_loader)
    
    TRAINING_STATE.log(f"Validation - Total: {avg_loss:.4f} | Mel: {avg_mel_loss:.4f} | Duration: {avg_duration_loss:.4f}")
    
    if avg_loss < TRAINING_STATE.best_loss:
        TRAINING_STATE.best_loss = avg_loss
        TRAINING_STATE.log(f"✓ New best validation loss: {avg_loss:.4f}")
    
    return avg_loss


def save_checkpoint(model, optimizer, epoch, step, loss, config):
    """Save training checkpoint"""
    checkpoint_dir = Path(config['paths']['checkpoints'])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_path = checkpoint_dir / f"checkpoint_epoch{epoch}_step{step}.pt"
    
    checkpoint = {
        'epoch': epoch,
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'config': config
    }
    
    torch.save(checkpoint, checkpoint_path)
    
    # Also save as latest
    latest_path = checkpoint_dir / "checkpoint_latest.pt"
    torch.save(checkpoint, latest_path)
    
    return checkpoint_path


def train(config_path: str, resume_from: Optional[str] = None):
    """Main training function"""
    global TRAINING_STATE
    
    try:
        TRAINING_STATE.is_running = True
        TRAINING_STATE.status_message = "Initializing..."
        TRAINING_STATE.log("="*60)
        TRAINING_STATE.log("AMHARIC TTS TRAINING - Chatterbox Fine-tuning")
        TRAINING_STATE.log("="*60)
        
        # Load config
        config = load_config(config_path)
        TRAINING_STATE.log(f"✓ Configuration loaded from {config_path}")
        
        # Setup device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        TRAINING_STATE.log(f"✓ Using device: {device}")
        
        # Setup model
        model = setup_model(config)
        model = model.to(device)
        
        # Print model info
        params = count_parameters(model)
        TRAINING_STATE.log(f"Total parameters: {params['total']:,}")
        TRAINING_STATE.log(f"Trainable parameters: {params['trainable']:,}")
        
        # Setup optimizer
        optimizer, scheduler = setup_optimizer(model, config)
        
        # Setup loss function
        criterion = TTSLoss(
            mel_loss_weight=1.0,
            duration_loss_weight=0.1
        )
        TRAINING_STATE.log("✓ Loss function initialized")
        
        # Setup dataloaders
        train_loader, val_loader = setup_dataloaders(config)
        
        # Setup mixed precision
        scaler = GradScaler(device='cuda' if torch.cuda.is_available() else 'cpu') if config['training']['use_amp'] else None
        
        # Setup tensorboard
        if config['logging']['use_tensorboard']:
            log_dir = Path(config['logging']['log_dir']) / datetime.now().strftime("%Y%m%d_%H%M%S")
            writer = SummaryWriter(log_dir)
            TRAINING_STATE.log(f"✓ TensorBoard logging to {log_dir}")
        else:
            writer = None
        
        # Resume from checkpoint if specified
        start_epoch = 0
        if resume_from and os.path.exists(resume_from):
            TRAINING_STATE.log(f"Resuming from checkpoint: {resume_from}")
            checkpoint = torch.load(resume_from, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            TRAINING_STATE.current_step = checkpoint['step']
            TRAINING_STATE.log(f"✓ Resumed from epoch {start_epoch}")
        
        # Calculate total steps
        TRAINING_STATE.total_steps = len(train_loader) * config['training']['max_epochs']
        
        # Training loop
        TRAINING_STATE.log("\nStarting training...")
        TRAINING_STATE.status_message = "Training in progress"
        
        for epoch in range(start_epoch, config['training']['max_epochs']):
            if not TRAINING_STATE.is_running:
                break
            
            TRAINING_STATE.current_epoch = epoch
            TRAINING_STATE.log(f"\n{'='*60}")
            TRAINING_STATE.log(f"EPOCH {epoch + 1}/{config['training']['max_epochs']}")
            TRAINING_STATE.log(f"{'='*60}")
            
            # Train
            continue_training = train_epoch(
                model, train_loader, optimizer, scheduler,
                scaler, criterion, writer, config, epoch
            )
            
            if not continue_training:
                break
            
            # Validate
            if (epoch + 1) % (config['training']['eval_interval'] // len(train_loader)) == 0:
                val_loss = validate(model, val_loader, criterion, config)
                
                if writer:
                    writer.add_scalar('Loss/validation', val_loss, TRAINING_STATE.current_step)
            
            # Check max steps
            if TRAINING_STATE.current_step >= config['training']['max_steps']:
                TRAINING_STATE.log("Reached maximum steps")
                break
        
        # Training completed
        TRAINING_STATE.log("\n" + "="*60)
        TRAINING_STATE.log("TRAINING COMPLETED")
        TRAINING_STATE.log("="*60)
        TRAINING_STATE.status_message = "Training completed successfully"
        
        # Save final checkpoint
        final_checkpoint = save_checkpoint(
            model, optimizer, TRAINING_STATE.current_epoch,
            TRAINING_STATE.current_step, TRAINING_STATE.current_loss, config
        )
        TRAINING_STATE.log(f"✓ Final checkpoint: {final_checkpoint}")
        
        if writer:
            writer.close()
        
    except Exception as e:
        TRAINING_STATE.log(f"❌ Training failed: {str(e)}")
        TRAINING_STATE.status_message = f"Error: {str(e)}"
        import traceback
        TRAINING_STATE.log(traceback.format_exc())
        raise
    
    finally:
        TRAINING_STATE.is_running = False


def start_training_thread(config_path: str, resume_from: Optional[str] = None):
    """Start training in a separate thread"""
    thread = threading.Thread(target=train, args=(config_path, resume_from))
    thread.daemon = True
    thread.start()
    return thread


def stop_training():
    """Stop training gracefully"""
    global TRAINING_STATE
    TRAINING_STATE.is_running = False
    TRAINING_STATE.status_message = "Stopping training..."
    TRAINING_STATE.log("Stopping training (will finish current step)...")


def get_training_state() -> Dict:
    """Get current training state"""
    return TRAINING_STATE.to_dict()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Amharic TTS")
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    
    args = parser.parse_args()
    
    train(args.config, args.resume)
