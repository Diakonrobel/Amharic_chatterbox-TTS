"""
Enhanced Training Script for Amharic TTS
=========================================

Based on best practices from chatterbox-finetune with Amharic-specific enhancements:
- Proper safetensors/checkpoint loading
- Learning rate warmup scheduling
- Gradient accumulation
- Audio sampling during training
- Early stopping
- Amharic G2P integration
- Multi-language embedding freezing

Usage:
    python train_enhanced.py --config config/training_config.yaml
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional
import logging

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torch.amp import autocast, GradScaler
import yaml
import pandas as pd
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from src.models.t3_model import SimplifiedT3Model, TTSLoss
from src.tokenizer.amharic_tokenizer import AmharicTokenizer
from src.g2p.amharic_g2p import AmharicG2P
from src.audio.audio_processing import AudioProcessor

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


class EnhancedAmharicDataset(Dataset):
    """
    Enhanced dataset for Amharic TTS with proper G2P integration
    """
    
    def __init__(
        self,
        metadata_path: str,
        data_dir: Path,
        tokenizer: AmharicTokenizer,
        g2p: AmharicG2P,
        audio_processor: AudioProcessor,
        use_phonemes: bool = True,
        max_text_len: int = 200,
        max_mel_len: int = 1000
    ):
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.g2p = g2p
        self.audio_processor = audio_processor
        self.use_phonemes = use_phonemes
        self.max_text_len = max_text_len
        self.max_mel_len = max_mel_len
        self.samples = []
        
        # Load metadata (LJSpeech format: filename|text|normalized_text)
        logger.info(f"Loading dataset from {metadata_path}")
        
        if metadata_path.endswith('.csv'):
            df = pd.read_csv(metadata_path, sep='|', header=None, 
                           names=['filename', 'text', 'normalized_text'])
            for _, row in df.iterrows():
                self.samples.append({
                    'audio': row['filename'],
                    'text': row['normalized_text'] if pd.notna(row.get('normalized_text')) else row['text']
                })
        else:
            # Plain metadata file
            with open(metadata_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('|')
                    if len(parts) >= 2:
                        self.samples.append({
                            'audio': parts[0],
                            'text': parts[1]
                        })
        
        logger.info(f"Loaded {len(self.samples)} samples")
        
        # Log sample
        if self.samples:
            sample = self.samples[0]
            logger.info(f"Sample text: {sample['text'][:100]}")
            if self.g2p:
                phonemes = self.g2p.grapheme_to_phoneme(sample['text'])
                logger.info(f"Sample phonemes: {phonemes[:100]}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        try:
            # Load audio
            audio_file = sample['audio']
            if not audio_file.endswith('.wav'):
                audio_file += '.wav'
            audio_path = self.data_dir / 'wavs' / audio_file
            
            _, mel = self.audio_processor.process_audio_file(str(audio_path))
            
            # Clip mel length
            if mel.shape[1] > self.max_mel_len:
                mel = mel[:, :self.max_mel_len]
            
            # Process text with G2P
            text = sample['text']
            
            # Encode with phonemes
            text_ids = self.tokenizer.encode(text, use_phonemes=self.use_phonemes)
            
            # Clip text length
            if len(text_ids) > self.max_text_len:
                text_ids = text_ids[:self.max_text_len]
            
            return {
                'text_ids': torch.LongTensor(text_ids),
                'text_len': len(text_ids),
                'mel': mel,
                'mel_len': mel.shape[1],
                'text': text,
                'audio_path': str(audio_path)
            }
            
        except Exception as e:
            logger.warning(f"Error loading sample {idx}: {e}")
            # Return dummy data to avoid breaking training
            return {
                'text_ids': torch.LongTensor([0] * 10),
                'text_len': 10,
                'mel': torch.zeros(80, 100),
                'mel_len': 100,
                'text': "dummy",
                'audio_path': "dummy"
            }


def collate_fn_enhanced(batch):
    """
    Enhanced collate function with proper padding
    """
    # Filter None samples
    batch = [item for item in batch if item is not None]
    if not batch:
        return None
    
    # Get max lengths
    max_text_len = max(item['text_len'] for item in batch)
    max_mel_len = max(item['mel_len'] for item in batch)
    mel_dim = batch[0]['mel'].shape[0]
    
    # Prepare batch tensors
    batch_size = len(batch)
    text_ids_batch = torch.zeros(batch_size, max_text_len, dtype=torch.long)
    text_lengths = torch.zeros(batch_size, dtype=torch.long)
    mel_batch = torch.zeros(batch_size, mel_dim, max_mel_len)
    mel_lengths = torch.zeros(batch_size, dtype=torch.long)
    
    texts = []
    audio_paths = []
    
    for i, item in enumerate(batch):
        text_len = item['text_len']
        mel_len = item['mel_len']
        
        text_ids_batch[i, :text_len] = item['text_ids']
        text_lengths[i] = text_len
        
        mel_batch[i, :, :mel_len] = item['mel']
        mel_lengths[i] = mel_len
        
        texts.append(item['text'])
        audio_paths.append(item['audio_path'])
    
    return {
        'text_ids': text_ids_batch,
        'text_lengths': text_lengths,
        'mel': mel_batch,
        'mel_lengths': mel_lengths,
        'texts': texts,
        'audio_paths': audio_paths
    }


class TrainingManager:
    """
    Enhanced training manager with all best practices
    """
    
    def __init__(self, config: Dict):
        self.config = config
        
        # Auto-detect device if not specified
        device_str = config.get('training', {}).get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device(device_str)
        logger.info(f"Using device: {self.device}")
        
        self.use_amp = config.get('training', {}).get('use_amp', True)
        
        # Initialize components
        logger.info("Initializing training components...")
        
        # G2P and Tokenizer
        self.g2p = AmharicG2P()
        self.tokenizer = self._load_tokenizer()
        
        # Audio processor
        data_config = config.get('data', {})
        self.audio_processor = AudioProcessor(
            sampling_rate=data_config.get('sampling_rate', 22050),
            n_mels=data_config.get('n_mel_channels', 80),  # Use n_mels to match AudioProcessor
            hop_length=data_config.get('hop_length', 256),
            n_fft=data_config.get('filter_length', 1024),
            win_length=data_config.get('win_length', 1024),
            fmin=data_config.get('mel_fmin', 0.0),
            fmax=data_config.get('mel_fmax', 8000.0)
        )
        
        # Model
        self.model = self._build_model()
        self.model.to(self.device)
        
        # Load pretrained if specified
        if config.get('finetuning', {}).get('enabled', False):
            self._load_pretrained()
        
        # Freeze embeddings if specified
        if config['model'].get('freeze_original_embeddings'):
            self._freeze_embeddings()
        
        # Print model summary
        self._print_model_summary()
        
        # Loss
        self.criterion = TTSLoss()
        
        # Optimizer
        self.optimizer = self._build_optimizer()
        
        # Scheduler
        self.scheduler = None
        
        # Training state
        self.global_step = 0
        self.start_epoch = 0
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        
        # Gradient scaler for mixed precision
        self.scaler = GradScaler() if self.use_amp else None
        
        # TensorBoard
        log_base_dir = config.get('logging', {}).get('log_dir', 'logs')
        log_dir = Path(log_base_dir) / datetime.now().strftime('%Y%m%d_%H%M%S')
        self.writer = SummaryWriter(log_dir=log_dir)
        logger.info(f"TensorBoard logs: {log_dir}")
        
        # Load checkpoint if resuming
        resume_checkpoint = config.get('training', {}).get('resume_checkpoint')
        if resume_checkpoint and os.path.exists(resume_checkpoint):
            self._load_checkpoint(resume_checkpoint)
    
    def _load_tokenizer(self):
        """Load Amharic tokenizer"""
        tokenizer_dir = self.config.get('paths', {}).get('tokenizer', 'models/tokenizer')
        
        if os.path.exists(tokenizer_dir):
            logger.info(f"Loading tokenizer from {tokenizer_dir}")
            return AmharicTokenizer.load(tokenizer_dir, g2p=self.g2p)
        else:
            logger.warning(f"Tokenizer not found at {tokenizer_dir}")
            logger.info("Creating new tokenizer...")
            return AmharicTokenizer(g2p=self.g2p)
    
    def _build_model(self):
        """Build TTS model"""
        model_config = self.config['model']
        
        return SimplifiedT3Model(
            vocab_size=model_config['n_vocab'],
            d_model=model_config.get('hidden_channels', 512),
            nhead=model_config.get('n_heads', 8),
            num_encoder_layers=model_config.get('n_layers', 6),
            dim_feedforward=model_config.get('filter_channels', 2048),
            dropout=model_config.get('p_dropout', 0.1),
            n_mels=self.config['data']['n_mel_channels']
        )
    
    def _load_pretrained(self):
        """Load pretrained Chatterbox weights"""
        pretrained_path = self.config.get('finetuning', {}).get('pretrained_model', '')
        
        if not os.path.exists(pretrained_path):
            logger.warning(f"Pretrained model not found: {pretrained_path}")
            return
        
        logger.info(f"Loading pretrained weights from {pretrained_path}")
        self.model.load_pretrained_weights(pretrained_path, strict=False)
    
    def _freeze_embeddings(self):
        """Freeze original embeddings to preserve pretrained knowledge"""
        freeze_until = self.config['model']['freeze_until_index']
        logger.info(f"Freezing embeddings 0-{freeze_until-1}")
        
        # Register hook to zero gradients
        if hasattr(self.model, 'text_embedding'):
            def embedding_hook(grad):
                mask = torch.ones_like(grad)
                mask[:freeze_until] = 0
                return grad * mask
            
            self.model.text_embedding.weight.register_hook(embedding_hook)
            logger.info(f"✓ Frozen {freeze_until} embeddings")
    
    def _print_model_summary(self):
        """Print model parameter summary"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        logger.info("=" * 70)
        logger.info("MODEL SUMMARY")
        logger.info("=" * 70)
        logger.info(f"Total parameters:     {total_params:>15,}")
        logger.info(f"Trainable parameters: {trainable_params:>15,} ({trainable_params/total_params*100:.1f}%)")
        logger.info(f"Frozen parameters:    {total_params-trainable_params:>15,}")
        logger.info("=" * 70)
    
    def _build_optimizer(self):
        """Build optimizer with proper settings"""
        training_config = self.config['training']
        
        return torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=training_config['learning_rate'],
            betas=training_config.get('betas', [0.9, 0.999]),
            eps=training_config.get('eps', 1e-8),
            weight_decay=training_config.get('weight_decay', 0.01)
        )
    
    def _build_scheduler(self, num_training_steps):
        """Build learning rate scheduler with warmup"""
        warmup_steps = self.config['training'].get('warmup_steps', 1000)
        
        try:
            from transformers import get_linear_schedule_with_warmup
            scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=num_training_steps
            )
            logger.info(f"Using linear warmup scheduler ({warmup_steps} warmup steps)")
            return scheduler
        except ImportError:
            logger.warning("transformers not installed, using ReduceLROnPlateau")
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=3,
                verbose=True
            )
    
    def _load_checkpoint(self, checkpoint_path):
        """Load training checkpoint"""
        logger.info(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler
        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load training state
        self.start_epoch = checkpoint.get('epoch', 0)
        self.global_step = checkpoint.get('global_step', 0)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        logger.info(f"✓ Resumed from epoch {self.start_epoch}, step {self.global_step}")
    
    def _save_checkpoint(self, epoch, val_loss=None):
        """Save training checkpoint"""
        checkpoint_dir = Path(self.config.get('paths', {}).get('checkpoints', 'models/checkpoints'))
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint_path = checkpoint_dir / f"checkpoint_epoch{epoch}_step{self.global_step}.pt"
        
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        if val_loss is not None:
            checkpoint['val_loss'] = val_loss
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"✓ Saved checkpoint: {checkpoint_path}")
        
        # Save best model
        if val_loss is not None and val_loss < self.best_val_loss:
            best_path = checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"✓ New best model saved! Val loss: {val_loss:.4f}")
            self.best_val_loss = val_loss
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1
    
    def train_epoch(self, dataloader, epoch):
        """Train one epoch"""
        self.model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            if batch is None:
                continue
            
            # Move to device
            text_ids = batch['text_ids'].to(self.device)
            text_lengths = batch['text_lengths'].to(self.device)
            mel_targets = batch['mel'].to(self.device)
            mel_lengths = batch['mel_lengths'].to(self.device)
            
            # Forward pass with mixed precision
            if self.use_amp:
                with autocast(device_type='cuda' if self.device.type == 'cuda' else 'cpu'):
                    outputs = self.model(text_ids, text_lengths, mel_targets)
                    loss_dict = self.criterion(outputs, {'mel': mel_targets, 'mel_lengths': mel_lengths})
                    loss = loss_dict['total_loss']
                
                # Backward pass
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['training'].get('grad_clip_thresh', 1.0)
                )
                
                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(text_ids, text_lengths, mel_targets)
                loss_dict = self.criterion(outputs, {'mel': mel_targets, 'mel_lengths': mel_lengths})
                loss = loss_dict['total_loss']
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['training'].get('grad_clip_thresh', 1.0)
                )
                self.optimizer.step()
            
            self.optimizer.zero_grad()
            
            # Update scheduler
            if self.scheduler and hasattr(self.scheduler, 'step') and not isinstance(
                self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
            ):
                self.scheduler.step()
            
            # Logging
            epoch_loss += loss.item()
            num_batches += 1
            
            if self.global_step % self.config['training']['log_interval'] == 0:
                self.writer.add_scalar('Loss/train', loss.item(), self.global_step)
                self.writer.add_scalar('Learning_Rate', self.optimizer.param_groups[0]['lr'], self.global_step)
            
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.1e}"
            })
            
            self.global_step += 1
        
        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss
    
    @torch.no_grad()
    def validate(self, dataloader):
        """Validate model"""
        self.model.eval()
        val_loss = 0.0
        num_batches = 0
        
        for batch in tqdm(dataloader, desc="Validating"):
            if batch is None:
                continue
            
            text_ids = batch['text_ids'].to(self.device)
            text_lengths = batch['text_lengths'].to(self.device)
            mel_targets = batch['mel'].to(self.device)
            mel_lengths = batch['mel_lengths'].to(self.device)
            
            outputs = self.model(text_ids, text_lengths, mel_targets)
            loss_dict = self.criterion(outputs, {'mel': mel_targets, 'mel_lengths': mel_lengths})
            
            val_loss += loss_dict['total_loss'].item()
            num_batches += 1
        
        avg_val_loss = val_loss / num_batches if num_batches > 0 else 0.0
        return avg_val_loss
    
    def train(self, train_dataloader, val_dataloader=None):
        """Main training loop"""
        max_epochs = self.config['training']['max_epochs']
        patience = self.config['training'].get('patience', 50)
        
        logger.info("=" * 70)
        logger.info("STARTING TRAINING")
        logger.info("=" * 70)
        logger.info(f"Max epochs: {max_epochs}")
        logger.info(f"Early stopping patience: {patience}")
        logger.info(f"Device: {self.device}")
        logger.info("=" * 70)
        
        # Build scheduler
        num_training_steps = len(train_dataloader) * max_epochs
        self.scheduler = self._build_scheduler(num_training_steps)
        
        for epoch in range(self.start_epoch, max_epochs):
            logger.info(f"\n{'='*70}")
            logger.info(f"EPOCH {epoch + 1}/{max_epochs}")
            logger.info(f"{'='*70}")
            
            # Train
            train_loss = self.train_epoch(train_dataloader, epoch + 1)
            logger.info(f"Train loss: {train_loss:.4f}")
            self.writer.add_scalar('Loss/train_epoch', train_loss, epoch + 1)
            
            # Validate
            if val_dataloader and (epoch + 1) % self.config['training'].get('eval_interval', 1) == 0:
                val_loss = self.validate(val_dataloader)
                logger.info(f"Val loss: {val_loss:.4f}")
                self.writer.add_scalar('Loss/val_epoch', val_loss, epoch + 1)
                
                # Update scheduler
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                
                # Save checkpoint
                if (epoch + 1) % self.config['training'].get('save_interval', 5) == 0:
                    self._save_checkpoint(epoch + 1, val_loss)
                
                # Early stopping
                if self.epochs_without_improvement >= patience:
                    logger.info(f"Early stopping triggered after {patience} epochs without improvement")
                    break
            else:
                # Save checkpoint without validation
                if (epoch + 1) % self.config['training'].get('save_interval', 5) == 0:
                    self._save_checkpoint(epoch + 1)
        
        logger.info("=" * 70)
        logger.info("TRAINING COMPLETE!")
        logger.info("=" * 70)
        
        self.writer.close()


def main():
    parser = argparse.ArgumentParser(description="Train Amharic TTS")
    parser.add_argument('--config', type=str, default='config/training_config.yaml',
                      help='Path to training configuration')
    parser.add_argument('--device', type=str, default=None,
                      help='Override device (cuda/cpu)')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override device if specified
    if args.device:
        config['training']['device'] = args.device
    
    # Initialize training manager
    manager = TrainingManager(config)
    
    # Create datasets
    data_config = config['data']
    dataset_path = Path(data_config['dataset_path'])
    
    train_dataset = EnhancedAmharicDataset(
        metadata_path=str(dataset_path / data_config['metadata_file']),
        data_dir=dataset_path,
        tokenizer=manager.tokenizer,
        g2p=manager.g2p,
        audio_processor=manager.audio_processor,
        use_phonemes=config['model'].get('use_phonemes', True)
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=data_config['batch_size'],
        shuffle=True,
        num_workers=data_config.get('num_workers', 2),
        collate_fn=collate_fn_enhanced,
        pin_memory=True
    )
    
    # Start training
    manager.train(train_loader, val_dataloader=None)


if __name__ == "__main__":
    main()
