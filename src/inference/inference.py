"""
Inference Script for Amharic TTS
Loads trained/finetuned Amharic model and generates speech from text
"""

import torch
import torch.nn.functional as F
import numpy as np
import soundfile as sf
from pathlib import Path
from typing import Optional, Tuple, Dict
import yaml
import sys

# Add project root
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.models.t3_model import SimplifiedT3Model
from src.g2p.amharic_g2p import AmharicG2P
from src.audio import AudioProcessor


class AmharicTTSInference:
    """
    Inference engine for Amharic TTS
    
    Loads trained checkpoint and generates speech from Amharic text
    """
    
    def __init__(self, 
                 checkpoint_path: str,
                 config_path: Optional[str] = None,
                 device: str = 'auto'):
        """
        Initialize inference engine
        
        Args:
            checkpoint_path: Path to trained model checkpoint (.pt file)
            config_path: Path to training config (optional, will try to auto-detect)
            device: Device to use ('cuda', 'cpu', or 'auto')
        """
        print(f"🔧 Initializing Amharic TTS Inference...")
        print(f"📂 Checkpoint: {checkpoint_path}")
        
        # Set device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        print(f"🖥️  Device: {self.device}")
        
        # Load config
        self.config = self._load_config(config_path, checkpoint_path)
        print(f"✓ Config loaded")
        
        # Initialize G2P
        self.g2p = AmharicG2P()
        print(f"✓ G2P initialized")
        
        # Initialize audio processor
        self.audio_processor = AudioProcessor(
            sample_rate=self.config['data'].get('sample_rate', 24000),
            n_fft=self.config['data'].get('n_fft', 1024),
            hop_length=self.config['data'].get('hop_length', 256),
            win_length=self.config['data'].get('win_length', 1024),
            n_mels=self.config['data'].get('n_mel_channels', 80)
        )
        print(f"✓ Audio processor initialized")
        
        # Load tokenizer
        self.tokenizer = self._load_tokenizer()
        if self.tokenizer:
            print(f"✓ Tokenizer loaded (vocab size: {self.tokenizer.get_vocab_size()})")
        else:
            print(f"⚠️  Warning: Tokenizer not found, using fallback encoding")
        
        # Load model
        self.model = self._load_model(checkpoint_path)
        self.model.eval()
        print(f"✓ Model loaded and ready for inference")
        print(f"✅ Initialization complete!\n")
    
    def _load_config(self, config_path: Optional[str], checkpoint_path: str) -> Dict:
        """Load or create config"""
        # Try to load config from multiple locations
        config_candidates = []
        
        if config_path:
            config_candidates.append(Path(config_path))
        
        # Check near checkpoint
        checkpoint_dir = Path(checkpoint_path).parent
        config_candidates.extend([
            checkpoint_dir / 'config.yaml',
            checkpoint_dir.parent / 'config.yaml',
            checkpoint_dir.parent.parent / 'config.yaml',
        ])
        
        # Check standard locations
        config_candidates.extend([
            project_root / 'config' / 'training_config.yaml',
            project_root / 'configs' / 'training_config.yaml',
        ])
        
        for config_file in config_candidates:
            if config_file.exists():
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                print(f"  Loaded config from: {config_file}")
                return config
        
        # Default config if not found
        print(f"  Using default config (no config file found)")
        return {
            'model': {
                'n_vocab': 2535,  # Default for merged tokenizer
                'd_model': 1024,
                'nhead': 8,
                'num_encoder_layers': 6,
                'dim_feedforward': 2048,
                'dropout': 0.1,
                'max_seq_len': 1000
            },
            'data': {
                'sample_rate': 24000,
                'n_fft': 1024,
                'hop_length': 256,
                'win_length': 1024,
                'n_mel_channels': 80
            }
        }
    
    def _load_tokenizer(self):
        """Load the tokenizer"""
        try:
            from src.tokenizer.merged_tokenizer import MergedTokenizer
            from src.tokenizer.amharic_tokenizer import AmharicTokenizer
            
            # Priority order for tokenizer loading
            tokenizer_candidates = [
                project_root / 'models' / 'tokenizer' / 'Am_tokenizer_merged.json',
                project_root / 'models' / 'tokenizer' / 'amharic_tokenizer',
                project_root / 'models' / 'tokenizer',
            ]
            
            for tokenizer_path in tokenizer_candidates:
                if tokenizer_path.exists():
                    try:
                        if 'merged' in str(tokenizer_path).lower():
                            tokenizer = MergedTokenizer.load(str(tokenizer_path))
                        else:
                            tokenizer = AmharicTokenizer.load(str(tokenizer_path), g2p=self.g2p)
                        return tokenizer
                    except Exception as e:
                        print(f"  Failed to load tokenizer from {tokenizer_path}: {e}")
                        continue
            
            return None
        except Exception as e:
            print(f"  Tokenizer loading error: {e}")
            return None
    
    def _load_model(self, checkpoint_path: str) -> SimplifiedT3Model:
        """Load trained model from checkpoint"""
        # Create model with config parameters
        model = SimplifiedT3Model(
            vocab_size=self.config['model']['n_vocab'],
            d_model=self.config['model']['d_model'],
            nhead=self.config['model']['nhead'],
            num_encoder_layers=self.config['model']['num_encoder_layers'],
            dim_feedforward=self.config['model']['dim_feedforward'],
            dropout=self.config['model']['dropout'],
            n_mels=self.config['data']['n_mel_channels'],
            max_seq_len=self.config['model']['max_seq_len']
        )
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        # Extract state dict
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            print(f"  Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
            print(f"  Training loss: {checkpoint.get('loss', 'unknown')}")
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        
        # Load weights
        model.load_state_dict(state_dict, strict=True)
        model.to(self.device)
        
        return model
    
    def _encode_text(self, text: str, use_phonemes: bool = False) -> torch.Tensor:
        """
        Encode text to token IDs
        
        Args:
            text: Amharic text to encode
            use_phonemes: Whether to use phoneme encoding (False for grapheme/character encoding)
        
        Returns:
            Token IDs tensor [1, seq_len]
        """
        if self.tokenizer:
            # Use tokenizer
            token_ids = self.tokenizer.encode(text, use_phonemes=use_phonemes)
        else:
            # Fallback: simple character-based encoding for Amharic
            import unicodedata
            text = unicodedata.normalize('NFC', text)
            
            token_ids = []
            for char in text[:100]:  # Limit length
                if char.isspace():
                    token_ids.append(0)
                else:
                    code_point = ord(char)
                    if 0x1200 <= code_point <= 0x137F:  # Ethiopic script
                        token_id = 100 + (code_point - 0x1200) % 800
                    elif 0x20 <= code_point <= 0x7F:  # ASCII
                        token_id = code_point
                    else:
                        token_id = 50 + (code_point % 50)
                    token_ids.append(token_id)
        
        # Convert to tensor
        token_tensor = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0)  # [1, seq_len]
        return token_tensor.to(self.device)
    
    def synthesize(self, 
                   text: str,
                   output_path: Optional[str] = None,
                   use_phonemes: bool = False,
                   speed: float = 1.0,
                   pitch: float = 1.0) -> Tuple[np.ndarray, int, Dict]:
        """
        Synthesize speech from Amharic text
        
        Args:
            text: Amharic text to synthesize
            output_path: Path to save audio file (optional)
            use_phonemes: Whether to use phoneme encoding (False = grapheme/character)
            speed: Speed multiplier (1.0 = normal)
            pitch: Pitch multiplier (1.0 = normal)
        
        Returns:
            (audio_waveform, sample_rate, info_dict)
        """
        print(f"\n🎙️  Synthesizing: {text}")
        
        # Encode text
        text_ids = self._encode_text(text, use_phonemes=use_phonemes)
        print(f"  Token IDs shape: {text_ids.shape}")
        
        # Generate mel-spectrogram
        with torch.no_grad():
            outputs = self.model(text_ids)
            mel_output = outputs['mel_outputs']  # [1, n_mels, time]
        
        print(f"  Generated mel shape: {mel_output.shape}")
        
        # Convert mel to audio using Griffin-Lim
        mel_np = mel_output.squeeze(0).cpu().numpy()  # [n_mels, time]
        audio = self.audio_processor.mel_to_audio(mel_np)
        
        # Apply speed adjustment
        if speed != 1.0:
            audio = self._adjust_speed(audio, speed)
        
        # Apply pitch adjustment
        if pitch != 1.0:
            audio = self._adjust_pitch(audio, pitch)
        
        sample_rate = self.config['data']['sample_rate']
        print(f"  Audio shape: {audio.shape}, duration: {len(audio)/sample_rate:.2f}s")
        
        # Save audio if output path provided
        if output_path:
            sf.write(output_path, audio, sample_rate)
            print(f"  ✓ Saved to: {output_path}")
        
        # Prepare info
        info = {
            'text': text,
            'text_length': len(text),
            'token_count': text_ids.shape[1],
            'mel_frames': mel_output.shape[2],
            'audio_duration': len(audio) / sample_rate,
            'sample_rate': sample_rate,
            'use_phonemes': use_phonemes,
            'speed': speed,
            'pitch': pitch
        }
        
        return audio, sample_rate, info
    
    def _adjust_speed(self, audio: np.ndarray, speed: float) -> np.ndarray:
        """Adjust audio speed"""
        try:
            import librosa
            return librosa.effects.time_stretch(audio, rate=speed)
        except:
            # Fallback: simple resampling
            import scipy.signal
            new_length = int(len(audio) / speed)
            return scipy.signal.resample(audio, new_length)
    
    def _adjust_pitch(self, audio: np.ndarray, pitch: float) -> np.ndarray:
        """Adjust audio pitch"""
        try:
            import librosa
            n_steps = 12 * np.log2(pitch)  # Convert to semitones
            return librosa.effects.pitch_shift(
                audio, 
                sr=self.config['data']['sample_rate'],
                n_steps=n_steps
            )
        except:
            # Fallback: return unchanged
            print("  Warning: pitch adjustment not available (install librosa)")
            return audio
    
    def batch_synthesize(self, 
                        texts: list,
                        output_dir: str,
                        use_phonemes: bool = False) -> list:
        """
        Synthesize multiple texts in batch
        
        Args:
            texts: List of Amharic texts
            output_dir: Directory to save audio files
            use_phonemes: Whether to use phoneme encoding
        
        Returns:
            List of output paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_paths = []
        for i, text in enumerate(texts):
            output_path = output_dir / f"synthesis_{i:04d}.wav"
            try:
                self.synthesize(text, str(output_path), use_phonemes=use_phonemes)
                output_paths.append(str(output_path))
            except Exception as e:
                print(f"  ❌ Failed to synthesize text {i}: {e}")
                output_paths.append(None)
        
        return output_paths


def main():
    """Command-line interface for inference"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Amharic TTS Inference')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--text', type=str, required=True,
                       help='Amharic text to synthesize')
    parser.add_argument('--output', type=str, default='output.wav',
                       help='Output audio file path')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config file (optional)')
    parser.add_argument('--use-phonemes', action='store_true',
                       help='Use phoneme encoding instead of grapheme')
    parser.add_argument('--speed', type=float, default=1.0,
                       help='Speed multiplier (default: 1.0)')
    parser.add_argument('--pitch', type=float, default=1.0,
                       help='Pitch multiplier (default: 1.0)')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cuda', 'cpu'],
                       help='Device to use')
    
    args = parser.parse_args()
    
    # Initialize inference engine
    tts = AmharicTTSInference(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        device=args.device
    )
    
    # Synthesize
    audio, sr, info = tts.synthesize(
        text=args.text,
        output_path=args.output,
        use_phonemes=args.use_phonemes,
        speed=args.speed,
        pitch=args.pitch
    )
    
    # Print info
    print("\n📊 Synthesis Info:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    print(f"\n✅ Done! Audio saved to: {args.output}")


if __name__ == '__main__':
    main()
