"""
Audio Processing for Amharic TTS
Handles mel-spectrogram extraction, audio loading, and preprocessing
"""

import torch
import torch.nn.functional as F
import librosa
import numpy as np
from pathlib import Path
from typing import Tuple, Optional


class AudioProcessor:
    """Audio preprocessing for TTS training"""
    
    def __init__(self,
                 sampling_rate: int = 22050,
                 n_fft: int = 1024,
                 hop_length: int = 256,
                 win_length: int = 1024,
                 n_mels: int = 80,
                 fmin: float = 0.0,
                 fmax: float = 8000.0):
        """
        Initialize audio processor
        
        Args:
            sampling_rate: Target sampling rate
            n_fft: FFT size
            hop_length: Hop length for STFT
            win_length: Window length for STFT
            n_mels: Number of mel filterbanks
            fmin: Minimum frequency
            fmax: Maximum frequency
        """
        self.sampling_rate = sampling_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax
        
        # Create mel filterbank
        self.mel_basis = librosa.filters.mel(
            sr=sampling_rate,
            n_fft=n_fft,
            n_mels=n_mels,
            fmin=fmin,
            fmax=fmax
        )
    
    def load_audio(self, audio_path: str) -> np.ndarray:
        """
        Load audio file and resample to target sr
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Audio waveform as numpy array
        """
        audio, sr = librosa.load(audio_path, sr=self.sampling_rate)
        return audio
    
    def get_mel_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract mel-spectrogram from audio
        
        Args:
            audio: Audio waveform
            
        Returns:
            Mel-spectrogram (n_mels, time)
        """
        # Compute STFT
        stft = librosa.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window='hann',
            center=True,
            pad_mode='reflect'
        )
        
        # Magnitude spectrogram
        magnitude = np.abs(stft)
        
        # Apply mel filterbank
        mel = np.dot(self.mel_basis, magnitude)
        
        # Convert to log scale
        mel = np.log(np.clip(mel, a_min=1e-5, a_max=None))
        
        return mel
    
    def process_audio_file(self, audio_path: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load audio and extract mel-spectrogram
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Tuple of (waveform, mel_spectrogram) as tensors
        """
        # Load audio
        audio = self.load_audio(audio_path)
        
        # Extract mel-spectrogram
        mel = self.get_mel_spectrogram(audio)
        
        # Convert to tensors
        audio_tensor = torch.FloatTensor(audio)
        mel_tensor = torch.FloatTensor(mel)
        
        return audio_tensor, mel_tensor
    
    def normalize_mel(self, mel: torch.Tensor, 
                     mean: Optional[float] = None,
                     std: Optional[float] = None) -> torch.Tensor:
        """
        Normalize mel-spectrogram
        
        Args:
            mel: Mel-spectrogram tensor
            mean: Mean for normalization (computed if None)
            std: Std for normalization (computed if None)
            
        Returns:
            Normalized mel-spectrogram
        """
        if mean is None:
            mean = mel.mean()
        if std is None:
            std = mel.std()
        
        return (mel - mean) / (std + 1e-8)
    
    def trim_silence(self, audio: np.ndarray, 
                     top_db: float = 40.0) -> np.ndarray:
        """
        Trim silence from audio
        
        Args:
            audio: Audio waveform
            top_db: Threshold in dB below reference
            
        Returns:
            Trimmed audio
        """
        audio, _ = librosa.effects.trim(audio, top_db=top_db)
        return audio
    
    def pad_or_trim(self, audio: np.ndarray, 
                   target_length: int) -> np.ndarray:
        """
        Pad or trim audio to target length
        
        Args:
            audio: Audio waveform
            target_length: Target length in samples
            
        Returns:
            Padded or trimmed audio
        """
        if len(audio) > target_length:
            # Trim
            audio = audio[:target_length]
        elif len(audio) < target_length:
            # Pad
            audio = np.pad(audio, (0, target_length - len(audio)), mode='constant')
        
        return audio
    
    def mel_to_audio(self, mel: np.ndarray, n_iter: int = 32) -> np.ndarray:
        """
        Convert mel-spectrogram to audio using Griffin-Lim algorithm
        
        Args:
            mel: Mel-spectrogram (n_mels, time)
            n_iter: Number of Griffin-Lim iterations
            
        Returns:
            Audio waveform as numpy array
        """
        # Convert from log scale
        mel = np.exp(mel)
        
        # Invert mel filterbank to get magnitude spectrogram
        mel_basis_inv = np.linalg.pinv(self.mel_basis)
        magnitude = np.dot(mel_basis_inv, mel)
        
        # Use Griffin-Lim to reconstruct phase and audio
        audio = librosa.griffinlim(
            magnitude,
            n_iter=n_iter,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window='hann',
            center=True,
            length=None
        )
        
        return audio


def collate_fn(batch):
    """
    Collate function for DataLoader
    Handles variable-length sequences
    
    Args:
        batch: List of (text_ids, mel, audio_path) tuples
        
    Returns:
        Batched tensors with padding
    """
    # Separate batch components
    text_ids_list = [item['text_ids'] for item in batch]
    mel_list = [item['mel'] for item in batch]
    audio_paths = [item['audio_path'] for item in batch]
    
    # Get max lengths
    max_text_len = max(len(text) for text in text_ids_list)
    max_mel_len = max(mel.shape[1] for mel in mel_list)
    
    # Pad sequences
    batch_size = len(batch)
    n_mels = mel_list[0].shape[0]
    
    # Initialize padded tensors
    text_ids_padded = torch.zeros(batch_size, max_text_len, dtype=torch.long)
    mel_padded = torch.zeros(batch_size, n_mels, max_mel_len)
    text_lengths = torch.LongTensor([len(text) for text in text_ids_list])
    mel_lengths = torch.LongTensor([mel.shape[1] for mel in mel_list])
    
    # Fill padded tensors
    for i in range(batch_size):
        text = text_ids_list[i]
        mel = mel_list[i]
        
        text_ids_padded[i, :len(text)] = torch.LongTensor(text)
        mel_padded[i, :, :mel.shape[1]] = mel
    
    return {
        'text_ids': text_ids_padded,
        'text_lengths': text_lengths,
        'mel': mel_padded,
        'mel_lengths': mel_lengths,
        'audio_paths': audio_paths
    }


# Default audio processor instance
# Commented out to avoid initialization errors on import
# default_audio_processor = AudioProcessor()
