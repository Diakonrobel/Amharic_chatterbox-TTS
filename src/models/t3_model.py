"""
Simplified T3-style TTS Model
Compatible with Chatterbox architecture for Amharic fine-tuning

This is a simplified implementation that:
- Works with extended embeddings
- Uses transformer architecture
- Can load Chatterbox pretrained weights
- Supports fine-tuning on Amharic data
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
from pathlib import Path


class SimplifiedT3Model(nn.Module):
    """
    Simplified T3-style transformer TTS model
    
    Architecture:
    - Text embedding (extended for Amharic)
    - Positional encoding
    - Transformer encoder
    - Mel decoder
    """
    
    def __init__(self,
                 vocab_size: int = 3000,
                 d_model: int = 512,
                 nhead: int = 8,
                 num_encoder_layers: int = 6,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1,
                 n_mels: int = 80,
                 max_seq_len: int = 1000):
        """
        Initialize T3 model
        
        Args:
            vocab_size: Size of text vocabulary
            d_model: Dimension of model
            nhead: Number of attention heads
            num_encoder_layers: Number of transformer layers
            dim_feedforward: Dimension of feedforward network
            dropout: Dropout rate
            n_mels: Number of mel channels
            max_seq_len: Maximum sequence length
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_mels = n_mels
        
        # Text embedding
        self.text_embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, dropout, max_seq_len)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers
        )
        
        # Mel decoder (simple projection for now)
        self.mel_decoder = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, n_mels)
        )
        
        # Duration predictor (optional, for alignment)
        self.duration_predictor = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        nn.init.xavier_uniform_(self.text_embedding.weight)
        for p in self.mel_decoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for p in self.duration_predictor.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, text_ids: torch.Tensor,
                text_lengths: Optional[torch.Tensor] = None,
                mel_targets: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            text_ids: Text token IDs [batch, seq_len]
            text_lengths: Text sequence lengths [batch]
            mel_targets: Target mel-spectrograms [batch, n_mels, time] (for training)
            
        Returns:
            Dictionary containing:
                - mel_outputs: Predicted mel-spectrograms
                - durations: Predicted durations
                - encoder_outputs: Encoder hidden states
        """
        batch_size, seq_len = text_ids.shape
        
        # Embed text
        text_embed = self.text_embedding(text_ids)  # [batch, seq_len, d_model]
        
        # Add positional encoding
        text_embed = self.pos_encoding(text_embed)
        
        # Create attention mask
        if text_lengths is not None:
            mask = self._create_padding_mask(text_lengths, seq_len)
        else:
            mask = None
        
        # Transformer encoding
        encoder_out = self.transformer_encoder(
            text_embed,
            src_key_padding_mask=mask
        )  # [batch, seq_len, d_model]
        
        # Predict durations
        durations = self.duration_predictor(encoder_out).squeeze(-1)  # [batch, seq_len]
        durations = F.softplus(durations)  # Ensure positive
        
        # Decode to mel
        if mel_targets is not None:
            # Teacher forcing: use target mel length
            target_len = mel_targets.shape[2]
            # Expand encoder outputs to match mel length (simple upsampling)
            encoder_expanded = self._length_regulate(encoder_out, durations, target_len)
        else:
            # Inference: use predicted durations
            encoder_expanded = self._length_regulate(encoder_out, durations)
        
        # Decode to mel-spectrogram
        mel_outputs = self.mel_decoder(encoder_expanded)  # [batch, time, n_mels]
        mel_outputs = mel_outputs.transpose(1, 2)  # [batch, n_mels, time]
        
        return {
            'mel_outputs': mel_outputs,
            'durations': durations,
            'encoder_outputs': encoder_out
        }
    
    def _create_padding_mask(self, lengths: torch.Tensor, max_len: int) -> torch.Tensor:
        """Create padding mask for variable length sequences"""
        batch_size = lengths.shape[0]
        mask = torch.arange(max_len, device=lengths.device).unsqueeze(0).expand(batch_size, -1)
        mask = mask >= lengths.unsqueeze(1)
        return mask
    
    def _length_regulate(self, encoder_out: torch.Tensor,
                        durations: torch.Tensor,
                        target_len: Optional[int] = None) -> torch.Tensor:
        """
        Expand encoder outputs according to predicted durations
        
        Args:
            encoder_out: [batch, seq_len, d_model]
            durations: [batch, seq_len]
            target_len: Target length (for teacher forcing)
            
        Returns:
            Expanded outputs [batch, target_len, d_model]
        """
        batch_size, seq_len, d_model = encoder_out.shape
        
        # Round durations to integers
        durations_int = torch.round(durations).long()
        durations_int = torch.clamp(durations_int, min=1)  # At least 1 frame per token
        
        if target_len is None:
            target_len = durations_int.sum(dim=1).max().item()
        
        # Expand each token according to its duration
        expanded = torch.zeros(batch_size, target_len, d_model, device=encoder_out.device)
        
        for b in range(batch_size):
            pos = 0
            for t in range(seq_len):
                dur = durations_int[b, t].item()
                if pos + dur <= target_len:
                    expanded[b, pos:pos+dur] = encoder_out[b, t:t+1].expand(dur, -1)
                    pos += dur
                else:
                    # Fill remaining with last token
                    expanded[b, pos:] = encoder_out[b, t:t+1].expand(target_len - pos, -1)
                    break
        
        return expanded
    
    def load_pretrained_weights(self, checkpoint_path: str, strict: bool = False):
        """
        Load pretrained Chatterbox weights
        
        Args:
            checkpoint_path: Path to checkpoint file
            strict: Whether to strictly enforce key matching
        """
        print(f"Loading pretrained weights from: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        # Get state dict
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # Try to load compatible weights
        model_dict = self.state_dict()
        pretrained_dict = {}
        
        for k, v in state_dict.items():
            # Map Chatterbox keys to our model keys
            if 'text_emb.weight' in k:
                # Load text embeddings (extended)
                pretrained_dict['text_embedding.weight'] = v
                print(f"  ✓ Loaded text_embedding: {v.shape}")
            # Add more mappings as needed
        
        # Load weights
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict, strict=False)
        
        print(f"✓ Loaded {len(pretrained_dict)} pretrained weight tensors")


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TTSLoss(nn.Module):
    """Combined loss for TTS training"""
    
    def __init__(self,
                 mel_loss_weight: float = 1.0,
                 duration_loss_weight: float = 0.1):
        super().__init__()
        self.mel_loss_weight = mel_loss_weight
        self.duration_loss_weight = duration_loss_weight
    
    def forward(self, outputs: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute TTS losses
        
        Args:
            outputs: Model outputs
            targets: Ground truth targets
            
        Returns:
            Dictionary of losses
        """
        # Mel reconstruction loss
        mel_pred = outputs['mel_outputs']
        mel_target = targets['mel']
        
        # Handle variable lengths
        if 'mel_lengths' in targets:
            mel_lengths = targets['mel_lengths']
            max_len = mel_target.shape[2]
            mask = torch.arange(max_len, device=mel_lengths.device).unsqueeze(0).expand(mel_target.shape[0], -1)
            mask = mask < mel_lengths.unsqueeze(1)
            mask = mask.unsqueeze(1).expand_as(mel_target)
            
            mel_loss = F.mse_loss(mel_pred * mask, mel_target * mask)
        else:
            mel_loss = F.mse_loss(mel_pred, mel_target)
        
        # Duration loss (if ground truth durations available)
        if 'durations' in targets:
            dur_pred = outputs['durations']
            dur_target = targets['durations']
            duration_loss = F.mse_loss(dur_pred, dur_target)
        else:
            # Regularization: prefer shorter durations
            duration_loss = outputs['durations'].mean() * 0.01
        
        # Total loss
        total_loss = (
            self.mel_loss_weight * mel_loss +
            self.duration_loss_weight * duration_loss
        )
        
        return {
            'total_loss': total_loss,
            'mel_loss': mel_loss,
            'duration_loss': duration_loss
        }
