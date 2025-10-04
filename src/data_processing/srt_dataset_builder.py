"""
Advanced SRT Dataset Builder for Amharic TTS
==============================================

Imports audio/video with SRT transcriptions and prepares training datasets.
Supports multiple imports, merging, filtering, and full dataset management.

Features:
- Import from SRT + audio/video files
- Extract audio from video files
- Split audio based on SRT timestamps
- Merge multiple imports into one dataset
- Quality filtering and validation
- Dataset statistics and management
- LJSpeech format output

Author: Enhanced for CHATTERBOX_STRUCTURED-AMHARIC
"""

import os
import re
import json
import subprocess
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import timedelta
import pandas as pd
import numpy as np
import soundfile as sf
import librosa
from tqdm import tqdm
import shutil
import hashlib

# Import advanced audio splitter (optional enhancement)
try:
    from advanced_audio_splitter import AmharicOptimizedSplitter, AdvancedAudioSplitter
    ADVANCED_SPLITTER_AVAILABLE = True
except ImportError:
    ADVANCED_SPLITTER_AVAILABLE = False
    # Using basic extraction (works perfectly fine)
    pass


@dataclass
class SRTEntry:
    """Single SRT subtitle entry"""
    index: int
    start_time: float  # seconds
    end_time: float  # seconds
    text: str
    duration: float
    
    def __post_init__(self):
        self.duration = self.end_time - self.start_time


@dataclass
class AudioSegment:
    """Processed audio segment"""
    segment_id: str
    source_file: str
    text: str
    start_time: float
    end_time: float
    duration: float
    audio_path: str
    sample_rate: int
    is_valid: bool
    issues: List[str]
    metadata: Dict


class SRTParser:
    """Parse SRT subtitle files"""
    
    @staticmethod
    def parse_timestamp(timestamp: str) -> float:
        """Convert SRT timestamp to seconds"""
        # Format: HH:MM:SS,mmm
        time_parts = timestamp.replace(',', '.').split(':')
        hours = int(time_parts[0])
        minutes = int(time_parts[1])
        seconds = float(time_parts[2])
        
        return hours * 3600 + minutes * 60 + seconds
    
    @staticmethod
    def parse_srt_file(srt_path: str) -> List[SRTEntry]:
        """Parse SRT file and extract all entries"""
        entries = []
        
        with open(srt_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split by empty lines (subtitle blocks)
        blocks = re.split(r'\n\s*\n', content.strip())
        
        for block in blocks:
            lines = block.strip().split('\n')
            
            if len(lines) < 3:
                continue
            
            try:
                # Line 1: Index
                index = int(lines[0].strip())
                
                # Line 2: Timestamps
                timestamp_line = lines[1].strip()
                start_str, end_str = timestamp_line.split('-->')
                start_time = SRTParser.parse_timestamp(start_str.strip())
                end_time = SRTParser.parse_timestamp(end_str.strip())
                
                # Line 3+: Text
                text = ' '.join(lines[2:]).strip()
                
                # Clean text
                text = SRTParser.clean_text(text)
                
                if text:  # Only add if text is not empty
                    entry = SRTEntry(
                        index=index,
                        start_time=start_time,
                        end_time=end_time,
                        text=text,
                        duration=end_time - start_time
                    )
                    entries.append(entry)
                    
            except Exception as e:
                print(f"Warning: Failed to parse block: {e}")
                continue
        
        return entries
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Clean subtitle text"""
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove formatting markers
        text = re.sub(r'\{[^}]+\}', '', text)
        text = re.sub(r'\[[^\]]+\]', '', text)
        
        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text)
        
        # Remove speaker labels (e.g., "SPEAKER: ")
        text = re.sub(r'^[A-Z\s]+:\s*', '', text)
        
        return text.strip()


class AudioExtractor:
    """Extract and process audio from video/audio files"""
    
    def __init__(self, temp_dir: str = "temp_extraction", use_advanced_splitting: bool = True):
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.use_advanced_splitting = use_advanced_splitting and ADVANCED_SPLITTER_AVAILABLE
        
        # Initialize advanced splitter if available
        if self.use_advanced_splitting:
            self.advanced_splitter = AmharicOptimizedSplitter()
            print("✓ Advanced audio splitter enabled (Amharic-optimized)")
        else:
            self.advanced_splitter = None
    
    def check_ffmpeg(self) -> bool:
        """Check if ffmpeg is available"""
        try:
            subprocess.run(['ffmpeg', '-version'], 
                          capture_output=True, 
                          check=True)
            return True
        except:
            return False
    
    def extract_audio_from_video(self, video_path: str, output_path: str, 
                                 sample_rate: int = 22050) -> bool:
        """Extract audio from video file using ffmpeg"""
        try:
            cmd = [
                'ffmpeg',
                '-i', video_path,
                '-vn',  # No video
                '-acodec', 'pcm_s16le',  # PCM 16-bit
                '-ar', str(sample_rate),  # Sample rate
                '-ac', '1',  # Mono
                '-y',  # Overwrite
                output_path
            ]
            
            subprocess.run(cmd, capture_output=True, check=True)
            return True
            
        except Exception as e:
            print(f"Error extracting audio: {e}")
            return False
    
    def extract_segment(self, audio_path: str, start_time: float, 
                       end_time: float, output_path: str) -> bool:
        """Extract audio segment using ffmpeg (faster than loading full file)"""
        try:
            duration = end_time - start_time
            
            cmd = [
                'ffmpeg',
                '-ss', str(start_time),  # Start time
                '-t', str(duration),  # Duration
                '-i', audio_path,
                '-acodec', 'pcm_s16le',
                '-ar', '22050',
                '-ac', '1',
                '-y',
                output_path
            ]
            
            subprocess.run(cmd, capture_output=True, check=True)
            return True
            
        except Exception as e:
            print(f"Error extracting segment: {e}")
            return False
    
    def extract_segment_librosa(self, audio_path: str, start_time: float,
                                end_time: float, output_path: str,
                                sample_rate: int = 22050) -> bool:
        """Extract segment using librosa (fallback method)"""
        try:
            # Load audio
            audio, sr = librosa.load(audio_path, sr=sample_rate)
            
            # Calculate sample indices
            start_sample = int(start_time * sr)
            end_sample = int(end_time * sr)
            
            # Extract segment
            segment = audio[start_sample:end_sample]
            
            # Save
            sf.write(output_path, segment, sr)
            return True
            
        except Exception as e:
            print(f"Error with librosa extraction: {e}")
            return False
    
    def extract_segment_advanced(self, audio_path: str, start_time: float,
                                end_time: float, output_path: str,
                                sample_rate: int = 22050) -> Tuple[bool, Dict]:
        """
        Extract segment using advanced splitter with VAD and precise boundaries
        
        Returns:
            (success, quality_metrics)
        """
        if not self.use_advanced_splitting or not self.advanced_splitter:
            # Fallback to basic extraction
            success = self.extract_segment_librosa(audio_path, start_time, end_time, output_path, sample_rate)
            return success, {}
        
        try:
            # Load full audio
            audio, sr = librosa.load(audio_path, sr=sample_rate)
            
            # Use advanced splitter
            segment = self.advanced_splitter.split_segment_precise(
                audio_full=audio,
                sr=sr,
                srt_start_time=start_time,
                srt_end_time=end_time,
                apply_vad=True,
                include_breath=True,
                apply_fades=True
            )
            
            if len(segment) == 0:
                return False, {"error": "Empty segment after processing"}
            
            # Save
            sf.write(output_path, segment, sr)
            
            # Analyze quality
            quality = self.advanced_splitter.analyze_segment_quality(segment, sr, "")
            
            return True, quality
            
        except Exception as e:
            print(f"Error with advanced extraction: {e}")
            # Fallback to basic
            success = self.extract_segment_librosa(audio_path, start_time, end_time, output_path, sample_rate)
            return success, {}


class DatasetValidator:
    """Validate audio segments for TTS training"""
    
    def __init__(self, min_duration: float = 1.0, max_duration: float = 15.0,
                 target_sr: int = 22050):
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.target_sr = target_sr
    
    def validate_segment(self, audio_path: str, text: str, 
                        duration: float) -> Tuple[bool, List[str]]:
        """Validate audio segment"""
        issues = []
        
        # Check if file exists
        if not os.path.exists(audio_path):
            issues.append("Audio file not found")
            return False, issues
        
        try:
            # Load audio
            audio, sr = librosa.load(audio_path, sr=None)
            
            # Duration check
            actual_duration = len(audio) / sr
            if actual_duration < self.min_duration:
                issues.append(f"Too short: {actual_duration:.2f}s")
            elif actual_duration > self.max_duration:
                issues.append(f"Too long: {actual_duration:.2f}s")
            
            # Silence check
            non_silent = librosa.effects.split(audio, top_db=30)
            if len(non_silent) == 0:
                issues.append("Audio is silent")
            
            # Clipping check
            if np.max(np.abs(audio)) > 0.99:
                issues.append("Audio clipping detected")
            
            # RMS energy check
            rms = np.sqrt(np.mean(audio**2))
            if rms < 0.01:
                issues.append("Very low audio energy")
            
            # Text length check
            if len(text.strip()) < 3:
                issues.append("Text too short")
            
            # Text-to-audio ratio (rough check for synchronization)
            chars_per_second = len(text) / actual_duration
            if chars_per_second > 30 or chars_per_second < 3:
                issues.append(f"Unusual speech rate: {chars_per_second:.1f} chars/sec")
            
            is_valid = len(issues) == 0
            return is_valid, issues
            
        except Exception as e:
            issues.append(f"Validation error: {str(e)}")
            return False, issues


class SRTDatasetBuilder:
    """
    Main class for building TTS datasets from SRT files
    """
    
    def __init__(self, base_output_dir: str = "data/srt_datasets"):
        self.base_output_dir = Path(base_output_dir)
        self.base_output_dir.mkdir(parents=True, exist_ok=True)
        
        self.srt_parser = SRTParser()
        self.audio_extractor = AudioExtractor()
        self.validator = DatasetValidator()
        
        # Check for ffmpeg
        self.has_ffmpeg = self.audio_extractor.check_ffmpeg()
        if not self.has_ffmpeg:
            print("⚠️  Warning: ffmpeg not found. Some features may be limited.")
            print("   Install ffmpeg: https://ffmpeg.org/download.html")
    
    def import_from_srt(self, srt_path: str, media_path: str, 
                       dataset_name: str, speaker_name: str = "speaker_01",
                       auto_validate: bool = True) -> Dict:
        """
        Import a single SRT + media file pair
        
        Args:
            srt_path: Path to SRT file
            media_path: Path to audio/video file
            dataset_name: Name for this dataset
            speaker_name: Speaker identifier
            auto_validate: Automatically validate segments
        
        Returns:
            Dictionary with import statistics
        """
        print(f"\n{'='*60}")
        print(f"Importing Dataset: {dataset_name}")
        print(f"{'='*60}")
        
        # Create dataset directory
        dataset_dir = self.base_output_dir / dataset_name
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        wavs_dir = dataset_dir / "wavs"
        wavs_dir.mkdir(exist_ok=True)
        
        # Step 1: Parse SRT
        print(f"\n[1/5] Parsing SRT file: {srt_path}")
        entries = self.srt_parser.parse_srt_file(srt_path)
        print(f"   ✓ Found {len(entries)} subtitle entries")
        
        # Step 2: Extract audio from video if needed
        print(f"\n[2/5] Processing media file: {media_path}")
        
        audio_file = media_path
        is_video = Path(media_path).suffix.lower() in ['.mp4', '.mkv', '.avi', '.mov', '.webm']
        
        if is_video and self.has_ffmpeg:
            print("   → Video file detected, extracting audio...")
            temp_audio = dataset_dir / f"temp_audio_{dataset_name}.wav"
            
            if self.audio_extractor.extract_audio_from_video(media_path, str(temp_audio)):
                audio_file = str(temp_audio)
                print("   ✓ Audio extracted successfully")
            else:
                print("   ✗ Failed to extract audio from video")
                return {"error": "Audio extraction failed"}
        
        # Step 3: Extract segments
        use_advanced = self.audio_extractor.use_advanced_splitting
        method_name = "Advanced (VAD + Amharic-optimized)" if use_advanced else "Standard"
        print(f"\n[3/5] Extracting {len(entries)} audio segments... [Method: {method_name}]")
        
        segments = []
        for entry in tqdm(entries, desc="Extracting segments"):
            # Generate unique segment ID
            segment_id = f"{dataset_name}_{speaker_name}_{entry.index:06d}"
            audio_output = wavs_dir / f"{segment_id}.wav"
            
            # Try advanced extraction first (if enabled)
            success = False
            quality_metrics = {}
            
            if use_advanced:
                success, quality_metrics = self.audio_extractor.extract_segment_advanced(
                    audio_file, entry.start_time, entry.end_time, str(audio_output)
                )
            
            # Fallback to FFmpeg or librosa if advanced fails or disabled
            if not success:
                if self.has_ffmpeg:
                    success = self.audio_extractor.extract_segment(
                        audio_file, entry.start_time, entry.end_time, str(audio_output)
                    )
                
                if not success:
                    success = self.audio_extractor.extract_segment_librosa(
                        audio_file, entry.start_time, entry.end_time, str(audio_output)
                    )
            
            if success:
                # Get actual duration from quality metrics or file
                if quality_metrics and 'duration' in quality_metrics:
                    actual_duration = quality_metrics['duration']
                else:
                    actual_duration = entry.duration
                
                segment = AudioSegment(
                    segment_id=segment_id,
                    source_file=os.path.basename(media_path),
                    text=entry.text,
                    start_time=entry.start_time,
                    end_time=entry.end_time,
                    duration=actual_duration,
                    audio_path=str(audio_output.relative_to(dataset_dir)),
                    sample_rate=22050,
                    is_valid=quality_metrics.get('is_valid', True) if quality_metrics else True,
                    issues=quality_metrics.get('issues', []) if quality_metrics else [],
                    metadata={
                        "speaker": speaker_name,
                        "dataset": dataset_name,
                        "srt_index": entry.index,
                        "extraction_method": "advanced" if use_advanced else "standard",
                        "quality_metrics": quality_metrics if quality_metrics else None
                    }
                )
                segments.append(segment)
        
        print(f"   ✓ Extracted {len(segments)}/{len(entries)} segments")
        
        # Step 4: Validate segments
        if auto_validate:
            print(f"\n[4/5] Validating segments...")
            valid_count = 0
            
            for segment in tqdm(segments, desc="Validating"):
                audio_full_path = dataset_dir / segment.audio_path
                is_valid, issues = self.validator.validate_segment(
                    str(audio_full_path), segment.text, segment.duration
                )
                segment.is_valid = is_valid
                segment.issues = issues
                
                if is_valid:
                    valid_count += 1
            
            print(f"   ✓ Valid segments: {valid_count}/{len(segments)}")
            print(f"   ✗ Invalid segments: {len(segments) - valid_count}")
        
        # Step 5: Save metadata
        print(f"\n[5/5] Saving dataset metadata...")
        
        # Create metadata.csv (LJSpeech format)
        metadata_lines = []
        for seg in segments:
            if seg.is_valid or not auto_validate:
                # Format: filename|text|normalized_text
                metadata_lines.append(f"{seg.segment_id}|{seg.text}|{seg.text}")
        
        metadata_file = dataset_dir / "metadata.csv"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(metadata_lines))
        
        # Save detailed JSON metadata
        json_metadata = {
            "dataset_name": dataset_name,
            "source_srt": os.path.basename(srt_path),
            "source_media": os.path.basename(media_path),
            "speaker": speaker_name,
            "total_segments": len(segments),
            "valid_segments": sum(1 for s in segments if s.is_valid),
            "segments": [asdict(seg) for seg in segments]
        }
        
        json_file = dataset_dir / "dataset_info.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(json_metadata, f, ensure_ascii=False, indent=2)
        
        # Save statistics
        stats = self._calculate_statistics(segments)
        stats_file = dataset_dir / "statistics.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2)
        
        print(f"\n   ✓ Saved to: {dataset_dir}")
        print(f"   ✓ Metadata: {metadata_file}")
        print(f"   ✓ Details: {json_file}")
        print(f"   ✓ Statistics: {stats_file}")
        
        # Clean up temp audio if created
        if is_video and self.has_ffmpeg:
            temp_audio_path = dataset_dir / f"temp_audio_{dataset_name}.wav"
            if temp_audio_path.exists():
                os.remove(temp_audio_path)
        
        print(f"\n{'='*60}")
        print(f"Import Complete!")
        print(f"{'='*60}\n")
        
        return stats
    
    def merge_datasets(self, dataset_names: List[str], merged_name: str,
                      filter_invalid: bool = True) -> Dict:
        """
        Merge multiple datasets into one
        
        Args:
            dataset_names: List of dataset names to merge
            merged_name: Name for merged dataset
            filter_invalid: Remove invalid segments
        
        Returns:
            Statistics for merged dataset
        """
        print(f"\n{'='*60}")
        print(f"Merging Datasets into: {merged_name}")
        print(f"{'='*60}")
        
        # Create merged dataset directory
        merged_dir = self.base_output_dir / merged_name
        merged_dir.mkdir(parents=True, exist_ok=True)
        
        wavs_dir = merged_dir / "wavs"
        wavs_dir.mkdir(exist_ok=True)
        
        all_segments = []
        dataset_stats = []
        
        # Load all datasets
        for dataset_name in dataset_names:
            dataset_dir = self.base_output_dir / dataset_name
            json_file = dataset_dir / "dataset_info.json"
            
            if not json_file.exists():
                print(f"⚠️  Warning: Dataset '{dataset_name}' not found, skipping...")
                continue
            
            print(f"\n   Loading: {dataset_name}")
            
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            segments = [AudioSegment(**seg) for seg in data['segments']]
            
            # Filter if needed
            if filter_invalid:
                valid_segments = [s for s in segments if s.is_valid]
                print(f"   → Loaded {len(valid_segments)}/{len(segments)} valid segments")
                segments = valid_segments
            else:
                print(f"   → Loaded {len(segments)} segments")
            
            # Copy audio files and update paths
            for seg in segments:
                old_audio_path = dataset_dir / seg.audio_path
                # Use original filename without adding dataset prefix
                # This prevents: Merged-2nd_Merged-Amharic_...
                original_filename = Path(seg.audio_path).name
                new_audio_path = wavs_dir / original_filename
                
                # Copy audio file
                shutil.copy2(old_audio_path, new_audio_path)
                
                # Update segment info
                seg.audio_path = str(Path("wavs") / original_filename)
                seg.metadata["original_dataset"] = dataset_name
            
            all_segments.extend(segments)
            dataset_stats.append({
                "name": dataset_name,
                "segments": len(segments)
            })
        
        print(f"\n   Total segments: {len(all_segments)}")
        
        # Create merged metadata
        metadata_lines = []
        for seg in all_segments:
            filename = Path(seg.audio_path).stem
            metadata_lines.append(f"{filename}|{seg.text}|{seg.text}")
        
        metadata_file = merged_dir / "metadata.csv"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(metadata_lines))
        
        # Save merged info
        merged_info = {
            "dataset_name": merged_name,
            "merged_from": dataset_names,
            "merge_date": pd.Timestamp.now().isoformat(),
            "total_segments": len(all_segments),
            "source_datasets": dataset_stats,
            "segments": [asdict(seg) for seg in all_segments]
        }
        
        json_file = merged_dir / "dataset_info.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(merged_info, f, ensure_ascii=False, indent=2)
        
        # Calculate statistics
        stats = self._calculate_statistics(all_segments)
        stats_file = merged_dir / "statistics.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2)
        
        print(f"\n   ✓ Merged dataset saved to: {merged_dir}")
        print(f"   ✓ Total duration: {stats['total_duration_hours']:.2f} hours")
        
        print(f"\n{'='*60}")
        print(f"Merge Complete!")
        print(f"{'='*60}\n")
        
        return stats
    
    def list_datasets(self) -> List[Dict]:
        """List all available datasets"""
        datasets = []
        
        for dataset_dir in self.base_output_dir.iterdir():
            if dataset_dir.is_dir():
                info_file = dataset_dir / "dataset_info.json"
                stats_file = dataset_dir / "statistics.json"
                
                if info_file.exists():
                    with open(info_file, 'r') as f:
                        info = json.load(f)
                    
                    stats = {}
                    if stats_file.exists():
                        with open(stats_file, 'r') as f:
                            stats = json.load(f)
                    
                    datasets.append({
                        "name": dataset_dir.name,
                        "segments": info.get("total_segments", 0),
                        "valid_segments": info.get("valid_segments", 0),
                        "duration_hours": stats.get("total_duration_hours", 0),
                        "path": str(dataset_dir)
                    })
        
        return datasets
    
    def _calculate_statistics(self, segments: List[AudioSegment]) -> Dict:
        """Calculate dataset statistics"""
        valid_segments = [s for s in segments if s.is_valid]
        
        if not segments:
            return {}
        
        durations = [s.duration for s in valid_segments]
        text_lengths = [len(s.text) for s in valid_segments]
        
        stats = {
            "total_segments": len(segments),
            "valid_segments": len(valid_segments),
            "invalid_segments": len(segments) - len(valid_segments),
            "total_duration_seconds": sum(durations),
            "total_duration_minutes": sum(durations) / 60,
            "total_duration_hours": sum(durations) / 3600,
            "average_duration": np.mean(durations) if durations else 0,
            "median_duration": np.median(durations) if durations else 0,
            "min_duration": min(durations) if durations else 0,
            "max_duration": max(durations) if durations else 0,
            "average_text_length": np.mean(text_lengths) if text_lengths else 0,
            "total_characters": sum(text_lengths)
        }
        
        return stats


# CLI Interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="SRT Dataset Builder for TTS")
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Import command
    import_parser = subparsers.add_parser('import', help='Import SRT + media file')
    import_parser.add_argument('--srt', required=True, help='Path to SRT file')
    import_parser.add_argument('--media', required=True, help='Path to audio/video file')
    import_parser.add_argument('--name', required=True, help='Dataset name')
    import_parser.add_argument('--speaker', default='speaker_01', help='Speaker name')
    import_parser.add_argument('--no-validate', action='store_true', help='Skip validation')
    
    # Merge command
    merge_parser = subparsers.add_parser('merge', help='Merge multiple datasets')
    merge_parser.add_argument('--datasets', nargs='+', required=True, help='Dataset names to merge')
    merge_parser.add_argument('--output', required=True, help='Output dataset name')
    merge_parser.add_argument('--include-invalid', action='store_true', help='Include invalid segments')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List all datasets')
    
    args = parser.parse_args()
    
    builder = SRTDatasetBuilder()
    
    if args.command == 'import':
        builder.import_from_srt(
            srt_path=args.srt,
            media_path=args.media,
            dataset_name=args.name,
            speaker_name=args.speaker,
            auto_validate=not args.no_validate
        )
    
    elif args.command == 'merge':
        builder.merge_datasets(
            dataset_names=args.datasets,
            merged_name=args.output,
            filter_invalid=not args.include_invalid
        )
    
    elif args.command == 'list':
        datasets = builder.list_datasets()
        
        if not datasets:
            print("No datasets found.")
        else:
            print(f"\n{'='*80}")
            print(f"Available Datasets ({len(datasets)})")
            print(f"{'='*80}\n")
            
            for ds in datasets:
                print(f"📁 {ds['name']}")
                print(f"   Segments: {ds['valid_segments']}/{ds['segments']} valid")
                print(f"   Duration: {ds['duration_hours']:.2f} hours")
                print(f"   Path: {ds['path']}")
                print()
    
    else:
        parser.print_help()
