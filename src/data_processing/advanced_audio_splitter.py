"""
Advanced Audio Splitter for TTS Dataset Creation
=================================================

Precise audio splitting with:
- Voice Activity Detection (VAD)
- Intelligent padding
- Silence trimming without cutting speech
- Language-specific optimizations
- No audio cutoff at start/end
- Breath detection
- Energy-based refinement

Based on best practices from dataset-maker and enhanced for Amharic.
"""

import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from typing import Tuple, List, Optional, Dict
import scipy.signal as signal
from dataclasses import dataclass


@dataclass
class SplitConfig:
    """Configuration for audio splitting"""
    # Voice Activity Detection
    vad_threshold_db: float = -40.0  # dB threshold for voice activity
    vad_frame_length: int = 2048
    vad_hop_length: int = 512
    
    # Padding
    pre_padding_ms: float = 100.0  # Milliseconds before speech
    post_padding_ms: float = 200.0  # Milliseconds after speech (for breath)
    
    # Silence trimming
    trim_top_db: int = 35  # Trim silence above this dB
    trim_frame_length: int = 2048
    trim_hop_length: int = 512
    
    # Safety margins (prevent cutting speech)
    safety_margin_start_ms: float = 50.0  # Extra margin at start
    safety_margin_end_ms: float = 100.0  # Extra margin at end
    
    # Energy refinement
    energy_percentile: float = 5.0  # Percentile for energy threshold
    
    # Smoothing
    smooth_window_size: int = 5  # Frames to smooth energy
    
    # Language-specific (Amharic)
    min_speech_duration_ms: float = 300.0  # Minimum speech length
    max_silence_duration_ms: float = 500.0  # Max silence within speech


class AdvancedAudioSplitter:
    """
    Advanced audio splitter with precise boundary detection
    
    Features:
    - Voice Activity Detection (VAD)
    - Intelligent padding
    - Breath detection
    - No speech cutoff
    - Language-specific optimizations
    """
    
    def __init__(self, config: Optional[SplitConfig] = None):
        self.config = config or SplitConfig()
    
    def detect_voice_activity(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Detect voice activity using energy-based method
        
        Returns:
            Binary array (1 = voice, 0 = silence)
        """
        # Calculate frame energy
        frame_length = self.config.vad_frame_length
        hop_length = self.config.vad_hop_length
        
        # Compute short-time energy
        energy = librosa.feature.rms(
            y=audio,
            frame_length=frame_length,
            hop_length=hop_length
        )[0]
        
        # Convert to dB
        energy_db = librosa.amplitude_to_db(energy, ref=np.max)
        
        # Smooth energy to reduce noise
        if len(energy_db) > self.config.smooth_window_size:
            energy_db = signal.medfilt(energy_db, self.config.smooth_window_size)
        
        # Adaptive threshold based on percentile
        threshold = np.percentile(energy_db, self.config.energy_percentile)
        threshold = max(threshold, self.config.vad_threshold_db)
        
        # Create voice activity mask
        vad = energy_db > threshold
        
        return vad
    
    def find_speech_boundaries(self, audio: np.ndarray, sr: int) -> Tuple[int, int]:
        """
        Find precise speech boundaries using multiple techniques
        
        Returns:
            (start_sample, end_sample)
        """
        # 1. Voice Activity Detection
        vad = self.detect_voice_activity(audio, sr)
        
        # Find voice regions
        hop_length = self.config.vad_hop_length
        
        # Convert VAD to sample indices
        vad_starts = []
        vad_ends = []
        in_speech = False
        
        for i, active in enumerate(vad):
            sample_idx = i * hop_length
            
            if active and not in_speech:
                vad_starts.append(sample_idx)
                in_speech = True
            elif not active and in_speech:
                vad_ends.append(sample_idx)
                in_speech = False
        
        if in_speech:
            vad_ends.append(len(audio))
        
        # 2. Find first and last speech
        if not vad_starts:
            # No speech detected, return safe margins
            return 0, len(audio)
        
        start_sample = vad_starts[0]
        end_sample = vad_ends[-1] if vad_ends else len(audio)
        
        # 3. Apply safety margins
        start_margin = int(self.config.safety_margin_start_ms * sr / 1000)
        end_margin = int(self.config.safety_margin_end_ms * sr / 1000)
        
        start_sample = max(0, start_sample - start_margin)
        end_sample = min(len(audio), end_sample + end_margin)
        
        # 4. Apply padding
        pre_padding = int(self.config.pre_padding_ms * sr / 1000)
        post_padding = int(self.config.post_padding_ms * sr / 1000)
        
        start_sample = max(0, start_sample - pre_padding)
        end_sample = min(len(audio), end_sample + post_padding)
        
        return start_sample, end_sample
    
    def detect_breath_sounds(self, audio: np.ndarray, sr: int, 
                            end_sample: int) -> int:
        """
        Detect breath sounds after speech and include them
        
        Returns:
            Adjusted end_sample including breath
        """
        # Look in window after detected speech end
        window_ms = 500  # Look 500ms after
        window_samples = int(window_ms * sr / 1000)
        
        search_end = min(len(audio), end_sample + window_samples)
        
        if end_sample >= search_end:
            return end_sample
        
        # Analyze audio after speech
        post_speech = audio[end_sample:search_end]
        
        if len(post_speech) == 0:
            return end_sample
        
        # Detect low-energy sounds (like breath)
        energy = librosa.feature.rms(y=post_speech, frame_length=512, hop_length=128)[0]
        energy_db = librosa.amplitude_to_db(energy, ref=np.max)
        
        # Breath is typically -50 to -30 dB
        breath_threshold_low = -55
        breath_threshold_high = -25
        
        breath_frames = (energy_db > breath_threshold_low) & (energy_db < breath_threshold_high)
        
        if np.any(breath_frames):
            # Find last breath frame
            last_breath = np.where(breath_frames)[0][-1]
            breath_end = last_breath * 128  # hop_length
            
            # Add breath to segment
            adjusted_end = end_sample + breath_end + int(0.1 * sr)  # 100ms after breath
            return min(len(audio), adjusted_end)
        
        return end_sample
    
    def refine_with_zero_crossings(self, audio: np.ndarray, sr: int,
                                   start_sample: int, end_sample: int) -> Tuple[int, int]:
        """
        Refine boundaries to nearest zero crossings to avoid clicks
        
        Returns:
            (refined_start, refined_end)
        """
        # Find zero crossings
        zero_crossings = librosa.zero_crossings(audio, pad=False)
        
        # Find nearest zero crossing to start
        search_window = int(0.01 * sr)  # 10ms window
        start_search_begin = max(0, start_sample - search_window)
        start_search_end = min(len(audio), start_sample + search_window)
        
        start_zc = np.where(zero_crossings[start_search_begin:start_search_end])[0]
        if len(start_zc) > 0:
            start_sample = start_search_begin + start_zc[0]
        
        # Find nearest zero crossing to end
        end_search_begin = max(0, end_sample - search_window)
        end_search_end = min(len(audio), end_sample + search_window)
        
        end_zc = np.where(zero_crossings[end_search_begin:end_search_end])[0]
        if len(end_zc) > 0:
            end_sample = end_search_begin + end_zc[-1]
        
        return start_sample, end_sample
    
    def apply_fade_in_out(self, audio: np.ndarray, sr: int,
                         fade_ms: float = 10.0) -> np.ndarray:
        """
        Apply smooth fade in/out to prevent clicks
        
        Args:
            audio: Audio signal
            sr: Sample rate
            fade_ms: Fade duration in milliseconds
        
        Returns:
            Audio with fades applied
        """
        fade_samples = int(fade_ms * sr / 1000)
        
        if len(audio) < fade_samples * 2:
            return audio
        
        # Create fade curves
        fade_in = np.linspace(0, 1, fade_samples)
        fade_out = np.linspace(1, 0, fade_samples)
        
        # Apply fades
        audio_faded = audio.copy()
        audio_faded[:fade_samples] *= fade_in
        audio_faded[-fade_samples:] *= fade_out
        
        return audio_faded
    
    def split_segment_precise(self, audio_full: np.ndarray, sr: int,
                              srt_start_time: float, srt_end_time: float,
                              apply_vad: bool = True,
                              include_breath: bool = True,
                              apply_fades: bool = True) -> np.ndarray:
        """
        Precisely split audio segment with advanced techniques
        
        Args:
            audio_full: Full audio array
            sr: Sample rate
            srt_start_time: Start time from SRT (seconds)
            srt_end_time: End time from SRT (seconds)
            apply_vad: Apply voice activity detection
            include_breath: Include breath sounds at end
            apply_fades: Apply fade in/out
        
        Returns:
            Precisely split audio segment
        """
        # Convert times to samples
        start_sample = int(srt_start_time * sr)
        end_sample = int(srt_end_time * sr)
        
        # Extract region around SRT times (with generous margins)
        margin_samples = int(1.0 * sr)  # 1 second margin on each side
        extract_start = max(0, start_sample - margin_samples)
        extract_end = min(len(audio_full), end_sample + margin_samples)
        
        audio_region = audio_full[extract_start:extract_end]
        
        if len(audio_region) == 0:
            return np.array([])
        
        # Calculate offset for later adjustment
        offset = extract_start
        
        if apply_vad:
            # Find precise speech boundaries within region
            speech_start, speech_end = self.find_speech_boundaries(audio_region, sr)
            
            # Detect and include breath sounds
            if include_breath:
                speech_end = self.detect_breath_sounds(audio_region, sr, speech_end)
            
            # Refine to zero crossings
            speech_start, speech_end = self.refine_with_zero_crossings(
                audio_region, sr, speech_start, speech_end
            )
            
            # Extract final segment
            audio_segment = audio_region[speech_start:speech_end]
        else:
            # Use SRT times directly but trim leading/trailing silence
            audio_segment, _ = librosa.effects.trim(
                audio_region,
                top_db=self.config.trim_top_db,
                frame_length=self.config.trim_frame_length,
                hop_length=self.config.trim_hop_length
            )
        
        # Apply fades to prevent clicks
        if apply_fades and len(audio_segment) > 0:
            audio_segment = self.apply_fade_in_out(audio_segment, sr)
        
        return audio_segment
    
    def analyze_segment_quality(self, audio: np.ndarray, sr: int,
                                text: str) -> Dict[str, any]:
        """
        Analyze segment quality metrics
        
        Returns:
            Dictionary with quality metrics
        """
        if len(audio) == 0:
            return {
                "duration": 0,
                "is_valid": False,
                "issues": ["Empty audio"]
            }
        
        duration = len(audio) / sr
        issues = []
        
        # Duration check
        if duration < 0.5:
            issues.append(f"Too short: {duration:.2f}s")
        elif duration > 20.0:
            issues.append(f"Too long: {duration:.2f}s")
        
        # Silence check
        non_silent = librosa.effects.split(audio, top_db=30)
        if len(non_silent) == 0:
            issues.append("Audio is silent")
        
        # Calculate speech-to-total ratio
        if len(non_silent) > 0:
            speech_samples = sum(end - start for start, end in non_silent)
            speech_ratio = speech_samples / len(audio)
            
            if speech_ratio < 0.3:
                issues.append(f"Low speech ratio: {speech_ratio:.2%}")
        
        # Clipping check
        if np.max(np.abs(audio)) > 0.99:
            issues.append("Audio clipping detected")
        
        # Energy check
        rms = np.sqrt(np.mean(audio**2))
        if rms < 0.01:
            issues.append("Very low audio energy")
        elif rms > 0.5:
            issues.append("Very high audio energy (possible clipping)")
        
        # SNR estimation
        try:
            # Simple SNR estimation
            if len(non_silent) > 0:
                speech_regions = np.concatenate([audio[start:end] for start, end in non_silent])
                speech_power = np.mean(speech_regions**2)
                
                # Estimate noise from silent regions
                silence_mask = np.ones(len(audio), dtype=bool)
                for start, end in non_silent:
                    silence_mask[start:end] = False
                
                if np.any(silence_mask):
                    noise_regions = audio[silence_mask]
                    noise_power = np.mean(noise_regions**2)
                    
                    if noise_power > 0:
                        snr = 10 * np.log10(speech_power / noise_power)
                        
                        if snr < 10:
                            issues.append(f"Low SNR: {snr:.1f} dB")
        except:
            pass
        
        # Text-to-audio ratio
        if text:
            chars_per_second = len(text) / duration
            if chars_per_second > 35:
                issues.append(f"Speech too fast: {chars_per_second:.1f} chars/sec")
            elif chars_per_second < 3:
                issues.append(f"Speech too slow: {chars_per_second:.1f} chars/sec")
        
        return {
            "duration": duration,
            "rms_energy": float(rms),
            "speech_ratio": float(speech_ratio) if len(non_silent) > 0 else 0.0,
            "num_speech_segments": len(non_silent),
            "is_valid": len(issues) == 0,
            "issues": issues
        }


class AmharicOptimizedSplitter(AdvancedAudioSplitter):
    """
    Splitter optimized for Amharic language characteristics
    
    Amharic-specific considerations:
    - Gemination (double consonants)
    - Ejective consonants
    - Typical speech patterns
    - Breath patterns
    """
    
    def __init__(self):
        # Amharic-optimized configuration
        config = SplitConfig(
            vad_threshold_db=-38.0,  # Slightly higher for clarity
            pre_padding_ms=120.0,  # More pre-padding for ejectives
            post_padding_ms=250.0,  # More post-padding for breath
            trim_top_db=33,  # Less aggressive trimming
            safety_margin_start_ms=60.0,
            safety_margin_end_ms=120.0,
            min_speech_duration_ms=400.0,  # Amharic tends to be syllable-timed
            max_silence_duration_ms=600.0
        )
        super().__init__(config)
    
    def detect_gemination(self, audio: np.ndarray, sr: int) -> List[Tuple[int, int]]:
        """
        Detect potential gemination (double consonants) in audio
        These should not be split
        
        Returns:
            List of (start_sample, end_sample) for gemination regions
        """
        # Gemination shows as sustained energy at consonant frequencies
        # Typically 1-4 kHz for Amharic consonants
        
        # High-pass filter to isolate consonants
        sos = signal.butter(4, 1000, 'hp', fs=sr, output='sos')
        filtered = signal.sosfilt(sos, audio)
        
        # Find sustained high-energy regions
        energy = librosa.feature.rms(y=filtered, frame_length=512, hop_length=128)[0]
        
        # Detect sustained regions (potential gemination)
        threshold = np.percentile(energy, 70)
        sustained = energy > threshold
        
        # Find continuous regions
        regions = []
        in_region = False
        start = 0
        
        for i, is_sustained in enumerate(sustained):
            sample_idx = i * 128
            
            if is_sustained and not in_region:
                start = sample_idx
                in_region = True
            elif not is_sustained and in_region:
                # Check duration (gemination is typically 100-200ms)
                duration_ms = (sample_idx - start) / sr * 1000
                if 80 < duration_ms < 250:
                    regions.append((start, sample_idx))
                in_region = False
        
        return regions
    
    def refine_for_amharic(self, audio: np.ndarray, sr: int,
                          start_sample: int, end_sample: int) -> Tuple[int, int]:
        """
        Refine boundaries considering Amharic phonetic features
        
        Returns:
            (refined_start, refined_end)
        """
        # Detect gemination regions
        gemination_regions = self.detect_gemination(audio, sr)
        
        # Ensure we don't split gemination
        for gem_start, gem_end in gemination_regions:
            # If boundary is in middle of gemination, extend it
            if start_sample > gem_start and start_sample < gem_end:
                start_sample = gem_start
            
            if end_sample > gem_start and end_sample < gem_end:
                end_sample = gem_end
        
        return start_sample, end_sample


# Convenience function for easy use
def split_audio_precise(audio_path: str, srt_start: float, srt_end: float,
                       output_path: str, optimize_for_amharic: bool = True,
                       sr: int = 22050) -> Dict[str, any]:
    """
    Split audio file precisely using advanced techniques
    
    Args:
        audio_path: Path to full audio file
        srt_start: Start time from SRT (seconds)
        srt_end: End time from SRT (seconds)
        output_path: Path to save split audio
        optimize_for_amharic: Use Amharic-specific optimizations
        sr: Target sample rate
    
    Returns:
        Quality metrics dictionary
    """
    # Load audio
    audio, original_sr = librosa.load(audio_path, sr=sr)
    
    # Choose splitter
    if optimize_for_amharic:
        splitter = AmharicOptimizedSplitter()
    else:
        splitter = AdvancedAudioSplitter()
    
    # Split segment
    segment = splitter.split_segment_precise(
        audio, sr, srt_start, srt_end,
        apply_vad=True,
        include_breath=True,
        apply_fades=True
    )
    
    # Save
    if len(segment) > 0:
        sf.write(output_path, segment, sr)
    
    # Analyze quality
    quality = splitter.analyze_segment_quality(segment, sr, "")
    
    return quality


if __name__ == "__main__":
    # Example usage
    print("Advanced Audio Splitter for Amharic TTS")
    print("========================================")
    print()
    print("This module provides precise audio splitting with:")
    print("  - Voice Activity Detection (VAD)")
    print("  - Intelligent padding")
    print("  - Breath detection")
    print("  - No speech cutoff")
    print("  - Amharic-specific optimizations")
    print()
    print("Use with SRTDatasetBuilder for best results!")
