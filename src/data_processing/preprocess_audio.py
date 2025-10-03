"""
Audio Preprocessing for Amharic TTS
Handles audio validation, normalization, and conversion to LJSpeech format
"""

import librosa
import soundfile as sf
import numpy as np
from pathlib import Path
from tqdm import tqdm
import pandas as pd
from typing import Dict, Tuple


class AudioPreprocessor:
    """Preprocess audio files for TTS training"""
    
    def __init__(self, target_sr: int = 22050, target_db: float = -20.0):
        """
        Initialize preprocessor
        
        Args:
            target_sr: Target sample rate (default: 22050 Hz)
            target_db: Target dB level for normalization
        """
        self.target_sr = target_sr
        self.target_db = target_db
    
    def validate_audio(self, audio_path: str) -> Tuple[bool, Dict]:
        """
        Validate audio file quality
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            (is_valid, metadata_dict)
        """
        try:
            audio, sr = librosa.load(audio_path, sr=None)
            duration = librosa.get_duration(y=audio, sr=sr)
            
            is_valid = True
            issues = []
            
            # Duration check (2-15 seconds for TTS)
            if duration < 2.0 or duration > 15.0:
                issues.append(f"Duration {duration:.2f}s outside range")
                is_valid = False
            
            # Sample rate check
            if sr not in [16000, 22050, 24000, 44100, 48000]:
                issues.append(f"Unusual sample rate: {sr}")
            
            # Silence check
            non_silent = librosa.effects.split(audio, top_db=30)
            if len(non_silent) == 0:
                issues.append("Audio is silent")
                is_valid = False
            
            # Clipping check
            if np.max(np.abs(audio)) > 0.99:
                issues.append("Audio clipping detected")
            
            metadata = {
                "duration": duration,
                "sample_rate": sr,
                "channels": 1 if audio.ndim == 1 else audio.shape[0],
                "is_valid": is_valid,
                "issues": issues
            }
            
            return is_valid, metadata
            
        except Exception as e:
            return False, {"error": str(e)}
    
    def normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """
        Normalize audio to target dB level
        
        Args:
            audio: Audio array
            
        Returns:
            Normalized audio
        """
        # RMS normalization
        rms = np.sqrt(np.mean(audio**2))
        target_rms = 10**(self.target_db / 20)
        
        if rms > 0:
            audio = audio * (target_rms / rms)
        
        # Peak normalization (prevent clipping)
        peak = np.max(np.abs(audio))
        if peak > 0.99:
            audio = audio * (0.99 / peak)
        
        return audio
    
    def remove_silence(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Remove leading and trailing silence
        
        Args:
            audio: Audio array
            sr: Sample rate
            
        Returns:
            Trimmed audio
        """
        intervals = librosa.effects.split(audio, top_db=30)
        
        if len(intervals) == 0:
            return audio
        
        start = intervals[0][0]
        end = intervals[-1][1]
        
        # Keep small padding
        padding = int(0.1 * sr)  # 100ms
        start = max(0, start - padding)
        end = min(len(audio), end + padding)
        
        return audio[start:end]
    
    def process_file(self, input_path: str, output_path: str) -> Dict:
        """
        Process single audio file
        
        Args:
            input_path: Input audio file path
            output_path: Output audio file path
            
        Returns:
            Processing metadata
        """
        # Load audio
        audio, sr = librosa.load(input_path, sr=None)
        
        # Resample if needed
        if sr != self.target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.target_sr)
        
        # Remove silence
        audio = self.remove_silence(audio, self.target_sr)
        
        # Normalize
        audio = self.normalize_audio(audio)
        
        # Save processed audio
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        sf.write(output_path, audio, self.target_sr)
        
        return {
            "duration": len(audio) / self.target_sr,
            "sample_rate": self.target_sr,
            "samples": len(audio)
        }
    
    def create_ljspeech_dataset(self, audio_dir: str, transcript_file: str, 
                               output_dir: str):
        """
        Convert to LJSpeech format
        
        Format: wavs/ directory with audio files
                metadata.csv with: filename|text|text
        
        Args:
            audio_dir: Directory containing audio files
            transcript_file: File with format: filename|text
            output_dir: Output directory
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        wavs_dir = output_dir / "wavs"
        wavs_dir.mkdir(exist_ok=True)
        
        metadata = []
        
        print("Processing audio files...")
        with open(transcript_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        for line in tqdm(lines, desc="Converting to LJSpeech format"):
            parts = line.strip().split('|')
            if len(parts) < 2:
                continue
            
            filename, text = parts[0], parts[1]
            audio_path = Path(audio_dir) / filename
            
            if not audio_path.exists():
                continue
            
            is_valid, audio_metadata = self.validate_audio(str(audio_path))
            
            if is_valid:
                # Process and save
                new_filename = f"amh_{len(metadata):06d}.wav"
                new_path = wavs_dir / new_filename
                
                try:
                    self.process_file(str(audio_path), str(new_path))
                    metadata.append(f"{new_filename}|{text}|{text}")
                except Exception as e:
                    print(f"Error processing {filename}: {e}")
        
        # Save metadata.csv
        with open(output_dir / "metadata.csv", 'w', encoding='utf-8') as f:
            f.write('\n'.join(metadata))
        
        print(f"\n✓ Created LJSpeech format dataset")
        print(f"  Total samples: {len(metadata)}")
        print(f"  Output: {output_dir}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Preprocess Amharic audio for TTS")
    parser.add_argument('--audio-dir', type=str, required=True,
                       help='Directory containing audio files')
    parser.add_argument('--transcript', type=str, required=True,
                       help='Transcript file (filename|text format)')
    parser.add_argument('--output', type=str, required=True,
                       help='Output directory for processed dataset')
    parser.add_argument('--sample-rate', type=int, default=22050,
                       help='Target sample rate (default: 22050)')
    
    args = parser.parse_args()
    
    preprocessor = AudioPreprocessor(target_sr=args.sample_rate)
    preprocessor.create_ljspeech_dataset(
        args.audio_dir,
        args.transcript,
        args.output
    )
