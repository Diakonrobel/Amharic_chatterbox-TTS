"""
Interactive Dataset Manager - Cross-Platform CLI
=================================================

Interactive command-line interface for managing SRT-based TTS datasets.
Works on Windows, Linux, and macOS.

Features:
- Interactive menu system
- Batch import multiple SRT files
- Dataset management and inspection
- Merging and filtering
- Statistics and reports

Usage:
    python dataset_manager.py
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Optional
import json

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))

try:
    from srt_dataset_builder import SRTDatasetBuilder
except ImportError:
    print("Error: srt_dataset_builder.py not found!")
    print("Make sure srt_dataset_builder.py is in the same directory.")
    sys.exit(1)


class DatasetManager:
    """Interactive dataset management CLI"""
    
    def __init__(self):
        self.builder = SRTDatasetBuilder()
        self.running = True
        
    def clear_screen(self):
        """Clear terminal screen (cross-platform)"""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def print_header(self, title: str):
        """Print formatted header"""
        width = 70
        print("\n" + "=" * width)
        print(f"{title:^{width}}")
        print("=" * width + "\n")
    
    def print_menu(self, title: str, options: List[str]):
        """Print menu options"""
        self.print_header(title)
        for i, option in enumerate(options, 1):
            print(f"  {i}. {option}")
        print(f"\n  0. Back/Exit")
        print("-" * 70)
    
    def get_choice(self, max_option: int) -> int:
        """Get user choice with validation"""
        while True:
            try:
                choice = input("\nEnter your choice: ").strip()
                choice_num = int(choice)
                if 0 <= choice_num <= max_option:
                    return choice_num
                else:
                    print(f"❌ Please enter a number between 0 and {max_option}")
            except ValueError:
                print("❌ Please enter a valid number")
            except KeyboardInterrupt:
                print("\n\n👋 Exiting...")
                sys.exit(0)
    
    def get_input(self, prompt: str, default: Optional[str] = None) -> str:
        """Get user input with optional default"""
        if default:
            full_prompt = f"{prompt} [{default}]: "
        else:
            full_prompt = f"{prompt}: "
        
        value = input(full_prompt).strip()
        return value if value else (default or "")
    
    def get_yes_no(self, prompt: str, default: bool = True) -> bool:
        """Get yes/no input"""
        default_str = "Y/n" if default else "y/N"
        response = input(f"{prompt} [{default_str}]: ").strip().lower()
        
        if not response:
            return default
        return response in ['y', 'yes']
    
    def pause(self):
        """Pause and wait for user"""
        input("\nPress Enter to continue...")
    
    def main_menu(self):
        """Main menu"""
        while self.running:
            self.clear_screen()
            options = [
                "📥 Import Single SRT Dataset",
                "📥 Batch Import Multiple SRT Files",
                "🔗 Merge Datasets",
                "📊 View Dataset Statistics",
                "📋 List All Datasets",
                "🔍 Inspect Dataset Details",
                "🗑️  Delete Dataset",
                "🚀 Prepare for Training",
                "ℹ️  Help & Documentation"
            ]
            
            self.print_menu("🎙️  AMHARIC TTS DATASET MANAGER", options)
            
            choice = self.get_choice(len(options))
            
            if choice == 0:
                self.exit_program()
            elif choice == 1:
                self.import_single()
            elif choice == 2:
                self.batch_import()
            elif choice == 3:
                self.merge_datasets_menu()
            elif choice == 4:
                self.view_statistics()
            elif choice == 5:
                self.list_datasets()
            elif choice == 6:
                self.inspect_dataset()
            elif choice == 7:
                self.delete_dataset()
            elif choice == 8:
                self.prepare_for_training()
            elif choice == 9:
                self.show_help()
    
    def import_single(self):
        """Import single SRT dataset"""
        self.clear_screen()
        self.print_header("📥 Import Single SRT Dataset")
        
        print("This will import one SRT file with its corresponding audio/video file.\n")
        
        # Get SRT file
        srt_path = self.get_input("SRT file path")
        if not srt_path or not os.path.exists(srt_path):
            print(f"❌ SRT file not found: {srt_path}")
            self.pause()
            return
        
        # Get media file
        media_path = self.get_input("Audio/Video file path")
        if not media_path or not os.path.exists(media_path):
            print(f"❌ Media file not found: {media_path}")
            self.pause()
            return
        
        # Get dataset name
        default_name = Path(srt_path).stem
        dataset_name = self.get_input("Dataset name", default_name)
        
        # Get speaker name
        speaker_name = self.get_input("Speaker name", "speaker_01")
        
        # Validation option
        validate = self.get_yes_no("Validate segments (recommended)", True)
        
        print("\n" + "=" * 70)
        print("Starting import...")
        print("=" * 70 + "\n")
        
        try:
            stats = self.builder.import_from_srt(
                srt_path=srt_path,
                media_path=media_path,
                dataset_name=dataset_name,
                speaker_name=speaker_name,
                auto_validate=validate
            )
            
            self.print_import_summary(stats)
            
        except Exception as e:
            print(f"\n❌ Error during import: {e}")
        
        self.pause()
    
    def batch_import(self):
        """Batch import multiple SRT files"""
        self.clear_screen()
        self.print_header("📥 Batch Import Multiple SRT Files")
        
        print("Import multiple SRT files at once.\n")
        print("Structure your files as:")
        print("  - file1.srt + file1.mp3 (or .mp4, .wav, etc.)")
        print("  - file2.srt + file2.mp4")
        print("  - etc.\n")
        
        # Get directory
        directory = self.get_input("Directory containing SRT and media files")
        if not directory or not os.path.isdir(directory):
            print(f"❌ Directory not found: {directory}")
            self.pause()
            return
        
        # Find SRT files
        srt_files = list(Path(directory).glob("*.srt"))
        
        if not srt_files:
            print(f"❌ No SRT files found in: {directory}")
            self.pause()
            return
        
        print(f"\n✓ Found {len(srt_files)} SRT file(s)")
        
        # Match with media files
        pairs = []
        media_extensions = ['.mp3', '.wav', '.mp4', '.mkv', '.avi', '.mov', '.webm', '.m4a']
        
        for srt_file in srt_files:
            base_name = srt_file.stem
            media_file = None
            
            # Try to find matching media file
            for ext in media_extensions:
                potential_media = srt_file.parent / f"{base_name}{ext}"
                if potential_media.exists():
                    media_file = potential_media
                    break
            
            if media_file:
                pairs.append((srt_file, media_file))
                print(f"  ✓ {srt_file.name} → {media_file.name}")
            else:
                print(f"  ⚠️  {srt_file.name} (no matching media file)")
        
        if not pairs:
            print("\n❌ No valid SRT+media pairs found!")
            self.pause()
            return
        
        print(f"\n✓ Found {len(pairs)} valid pair(s)")
        
        # Confirm
        if not self.get_yes_no(f"\nImport {len(pairs)} dataset(s)", True):
            return
        
        # Get base options
        speaker_prefix = self.get_input("Speaker name prefix", "speaker")
        validate = self.get_yes_no("Validate segments", True)
        merge_after = self.get_yes_no("Merge into single dataset after import", True)
        
        # Import all
        print("\n" + "=" * 70)
        print("Starting batch import...")
        print("=" * 70 + "\n")
        
        imported_names = []
        total_segments = 0
        
        for i, (srt_file, media_file) in enumerate(pairs, 1):
            dataset_name = f"dataset_{srt_file.stem}"
            speaker_name = f"{speaker_prefix}_{i:02d}"
            
            print(f"\n[{i}/{len(pairs)}] Importing: {srt_file.name}")
            print("-" * 70)
            
            try:
                stats = self.builder.import_from_srt(
                    srt_path=str(srt_file),
                    media_path=str(media_file),
                    dataset_name=dataset_name,
                    speaker_name=speaker_name,
                    auto_validate=validate
                )
                
                imported_names.append(dataset_name)
                total_segments += stats.get('valid_segments', 0)
                
            except Exception as e:
                print(f"❌ Error importing {srt_file.name}: {e}")
        
        print("\n" + "=" * 70)
        print(f"✓ Batch Import Complete!")
        print(f"  Imported: {len(imported_names)} dataset(s)")
        print(f"  Total segments: {total_segments}")
        print("=" * 70)
        
        # Merge if requested
        if merge_after and len(imported_names) > 1:
            print("\nMerging datasets...")
            merged_name = self.get_input("Merged dataset name", "merged_dataset")
            
            try:
                stats = self.builder.merge_datasets(
                    dataset_names=imported_names,
                    merged_name=merged_name,
                    filter_invalid=True
                )
                print(f"\n✓ Merged into: {merged_name}")
                print(f"  Total duration: {stats['total_duration_hours']:.2f} hours")
            except Exception as e:
                print(f"❌ Error merging: {e}")
        
        self.pause()
    
    def merge_datasets_menu(self):
        """Merge multiple datasets"""
        self.clear_screen()
        self.print_header("🔗 Merge Datasets")
        
        # List available datasets
        datasets = self.builder.list_datasets()
        
        if len(datasets) < 2:
            print("❌ Need at least 2 datasets to merge!")
            self.pause()
            return
        
        print("Available datasets:\n")
        for i, ds in enumerate(datasets, 1):
            print(f"  {i}. {ds['name']}")
            print(f"     └─ {ds['valid_segments']}/{ds['segments']} segments, "
                  f"{ds['duration_hours']:.2f}h\n")
        
        # Get datasets to merge
        print("Enter dataset numbers to merge (comma-separated):")
        print("Example: 1,2,3")
        
        selection = self.get_input("Dataset numbers")
        
        try:
            indices = [int(x.strip()) - 1 for x in selection.split(',')]
            selected = [datasets[i]['name'] for i in indices if 0 <= i < len(datasets)]
            
            if len(selected) < 2:
                print("❌ Need at least 2 datasets to merge!")
                self.pause()
                return
            
            print(f"\nSelected datasets: {', '.join(selected)}")
            
            # Get merge options
            merged_name = self.get_input("Name for merged dataset", "merged_dataset")
            filter_invalid = self.get_yes_no("Filter out invalid segments", True)
            
            # Confirm
            if not self.get_yes_no(f"\nMerge {len(selected)} datasets", True):
                return
            
            print("\n" + "=" * 70)
            print("Merging...")
            print("=" * 70 + "\n")
            
            stats = self.builder.merge_datasets(
                dataset_names=selected,
                merged_name=merged_name,
                filter_invalid=filter_invalid
            )
            
            print(f"\n✓ Merge Complete!")
            self.print_statistics(stats)
            
        except Exception as e:
            print(f"❌ Error: {e}")
        
        self.pause()
    
    def view_statistics(self):
        """View dataset statistics"""
        self.clear_screen()
        self.print_header("📊 Dataset Statistics")
        
        datasets = self.builder.list_datasets()
        
        if not datasets:
            print("No datasets found.")
            self.pause()
            return
        
        # Calculate totals
        total_segments = sum(ds['segments'] for ds in datasets)
        total_valid = sum(ds['valid_segments'] for ds in datasets)
        total_hours = sum(ds['duration_hours'] for ds in datasets)
        
        print(f"Total Datasets: {len(datasets)}")
        print(f"Total Segments: {total_valid:,}/{total_segments:,} valid")
        print(f"Total Duration: {total_hours:.2f} hours ({total_hours/60:.2f} minutes)")
        print("\n" + "-" * 70 + "\n")
        
        for ds in datasets:
            print(f"📁 {ds['name']}")
            print(f"   Segments: {ds['valid_segments']:,}/{ds['segments']:,} valid")
            print(f"   Duration: {ds['duration_hours']:.2f}h")
            print(f"   Path: {ds['path']}\n")
        
        self.pause()
    
    def list_datasets(self):
        """List all datasets"""
        self.clear_screen()
        self.print_header("📋 Available Datasets")
        
        datasets = self.builder.list_datasets()
        
        if not datasets:
            print("No datasets found.")
        else:
            for i, ds in enumerate(datasets, 1):
                print(f"{i}. {ds['name']}")
                print(f"   ├─ Segments: {ds['valid_segments']:,}/{ds['segments']:,}")
                print(f"   ├─ Duration: {ds['duration_hours']:.2f} hours")
                print(f"   └─ Path: {ds['path']}\n")
        
        self.pause()
    
    def inspect_dataset(self):
        """Inspect dataset details"""
        self.clear_screen()
        self.print_header("🔍 Inspect Dataset")
        
        datasets = self.builder.list_datasets()
        
        if not datasets:
            print("No datasets found.")
            self.pause()
            return
        
        # Show list
        print("Available datasets:\n")
        for i, ds in enumerate(datasets, 1):
            print(f"  {i}. {ds['name']}")
        
        # Get selection
        choice = self.get_choice(len(datasets))
        if choice == 0:
            return
        
        dataset = datasets[choice - 1]
        
        # Load details
        info_path = Path(dataset['path']) / "dataset_info.json"
        stats_path = Path(dataset['path']) / "statistics.json"
        
        self.clear_screen()
        self.print_header(f"Dataset: {dataset['name']}")
        
        if info_path.exists():
            with open(info_path, 'r', encoding='utf-8') as f:
                info = json.load(f)
            
            print(f"Source SRT: {info.get('source_srt', 'N/A')}")
            print(f"Source Media: {info.get('source_media', 'N/A')}")
            print(f"Speaker: {info.get('speaker', 'N/A')}")
            print(f"Total Segments: {info.get('total_segments', 0):,}")
            print(f"Valid Segments: {info.get('valid_segments', 0):,}")
            
            if 'merged_from' in info:
                print(f"\nMerged from: {', '.join(info['merged_from'])}")
        
        if stats_path.exists():
            with open(stats_path, 'r') as f:
                stats = json.load(f)
            
            print("\n" + "-" * 70)
            print("Statistics:")
            print("-" * 70)
            self.print_statistics(stats)
        
        # Show sample segments
        if info_path.exists():
            print("\n" + "-" * 70)
            print("Sample Segments (first 5):")
            print("-" * 70 + "\n")
            
            segments = info.get('segments', [])[:5]
            for seg in segments:
                print(f"ID: {seg.get('segment_id', 'N/A')}")
                print(f"Text: {seg.get('text', 'N/A')[:100]}...")
                print(f"Duration: {seg.get('duration', 0):.2f}s")
                print(f"Valid: {'✓' if seg.get('is_valid', False) else '✗'}")
                if seg.get('issues'):
                    print(f"Issues: {', '.join(seg['issues'])}")
                print()
        
        self.pause()
    
    def delete_dataset(self):
        """Delete a dataset"""
        self.clear_screen()
        self.print_header("🗑️  Delete Dataset")
        
        datasets = self.builder.list_datasets()
        
        if not datasets:
            print("No datasets found.")
            self.pause()
            return
        
        print("⚠️  WARNING: This action cannot be undone!\n")
        print("Available datasets:\n")
        for i, ds in enumerate(datasets, 1):
            print(f"  {i}. {ds['name']}")
        
        choice = self.get_choice(len(datasets))
        if choice == 0:
            return
        
        dataset = datasets[choice - 1]
        
        # Confirm deletion
        print(f"\n⚠️  You are about to delete: {dataset['name']}")
        print(f"   Path: {dataset['path']}")
        print(f"   Segments: {dataset['segments']:,}")
        
        if not self.get_yes_no("\nAre you absolutely sure", False):
            print("Cancelled.")
            self.pause()
            return
        
        # Delete
        try:
            import shutil
            shutil.rmtree(dataset['path'])
            print(f"\n✓ Deleted: {dataset['name']}")
        except Exception as e:
            print(f"\n❌ Error deleting dataset: {e}")
        
        self.pause()
    
    def prepare_for_training(self):
        """Prepare dataset for training"""
        self.clear_screen()
        self.print_header("🚀 Prepare Dataset for Training")
        
        datasets = self.builder.list_datasets()
        
        if not datasets:
            print("No datasets found.")
            self.pause()
            return
        
        print("Select dataset to prepare for training:\n")
        for i, ds in enumerate(datasets, 1):
            print(f"  {i}. {ds['name']} ({ds['duration_hours']:.2f}h)")
        
        choice = self.get_choice(len(datasets))
        if choice == 0:
            return
        
        dataset = datasets[choice - 1]
        dataset_path = Path(dataset['path'])
        
        # Check if already in correct format
        metadata_file = dataset_path / "metadata.csv"
        wavs_dir = dataset_path / "wavs"
        
        if metadata_file.exists() and wavs_dir.exists():
            print(f"\n✓ Dataset is already in LJSpeech format!")
            print(f"\nTo use for training:")
            print(f"  1. Copy to: amharic-tts/data/processed/")
            print(f"  2. Update training config:")
            print(f"     data:")
            print(f"       dataset_path: \"data/processed/{dataset['name']}\"")
            print(f"\nOr copy now:")
            
            if self.get_yes_no("Copy to data/processed now", True):
                try:
                    target_dir = Path("data/processed") / dataset['name']
                    target_dir.parent.mkdir(parents=True, exist_ok=True)
                    
                    import shutil
                    if target_dir.exists():
                        print(f"⚠️  Target directory exists. Overwrite?")
                        if not self.get_yes_no("Overwrite", False):
                            self.pause()
                            return
                        shutil.rmtree(target_dir)
                    
                    shutil.copytree(dataset_path, target_dir)
                    print(f"\n✓ Copied to: {target_dir}")
                    print(f"\n✓ Ready for training!")
                    print(f"\nNext steps:")
                    print(f"  1. Train tokenizer on this dataset")
                    print(f"  2. Update config/training_config.yaml")
                    print(f"  3. Start training")
                    
                except Exception as e:
                    print(f"❌ Error copying: {e}")
        
        self.pause()
    
    def show_help(self):
        """Show help and documentation"""
        self.clear_screen()
        self.print_header("ℹ️  Help & Documentation")
        
        help_text = """
WORKFLOW OVERVIEW:
==================

1. IMPORT SRT FILES
   - Use "Import Single" for one SRT+media pair
   - Use "Batch Import" for multiple files at once
   - Supports: .mp3, .wav, .mp4, .mkv, .avi, .mov, .webm

2. VALIDATION
   - Checks audio quality (duration, silence, clipping)
   - Validates text content
   - Filters problematic segments

3. MERGE DATASETS (Optional)
   - Combine multiple imports into one dataset
   - Useful for building larger training sets
   - Can filter invalid segments during merge

4. PREPARE FOR TRAINING
   - Datasets are saved in LJSpeech format
   - Copy to data/processed/ directory
   - Update training configuration

FILE STRUCTURE:
===============

Your SRT files should match media files by name:
  video1.srt → video1.mp4
  audio2.srt → audio2.mp3
  etc.

SRT FORMAT:
===========

1
00:00:01,000 --> 00:00:05,000
ሰላም ለዓለም

2
00:00:06,000 --> 00:00:10,000
አዲስ አበባ የኢትዮጵያ ዋና ከተማ ናት

DATASET REQUIREMENTS:
=====================

For good TTS training:
  - Minimum: 10+ hours of audio
  - Recommended: 20+ hours
  - Audio quality: 22050 Hz, mono
  - Segment duration: 2-15 seconds
  - Clean transcriptions

TIPS:
=====

1. Start with small test imports to verify format
2. Always enable validation during import
3. Review statistics before training
4. Merge datasets for larger training sets
5. Keep original SRT files as backup

TROUBLESHOOTING:
================

Q: "ffmpeg not found"
A: Install ffmpeg from https://ffmpeg.org/download.html

Q: "No matching media file"
A: Ensure SRT and media files have matching names

Q: "Too many invalid segments"
A: Check audio quality and SRT timing accuracy

Q: "Import is slow"
A: Large video files take time. Consider extracting audio first.

COMMAND-LINE USAGE:
===================

You can also use the CLI directly:

# Import single dataset
python srt_dataset_builder.py import \\
    --srt path/to/file.srt \\
    --media path/to/audio.mp3 \\
    --name my_dataset

# Merge datasets
python srt_dataset_builder.py merge \\
    --datasets dataset1 dataset2 dataset3 \\
    --output merged_dataset

# List datasets
python srt_dataset_builder.py list

CONTACT & SUPPORT:
==================

For issues or questions, check:
  - README.md in the project root
  - Documentation in docs/ folder
  - Project repository
"""
        
        print(help_text)
        self.pause()
    
    def print_import_summary(self, stats: Dict):
        """Print import summary"""
        print("\n" + "=" * 70)
        print("IMPORT SUMMARY")
        print("=" * 70)
        self.print_statistics(stats)
    
    def print_statistics(self, stats: Dict):
        """Print statistics in formatted way"""
        if not stats:
            print("No statistics available.")
            return
        
        print(f"\nTotal Segments: {stats.get('total_segments', 0):,}")
        print(f"Valid Segments: {stats.get('valid_segments', 0):,}")
        print(f"Invalid Segments: {stats.get('invalid_segments', 0):,}")
        
        print(f"\nDuration:")
        print(f"  Total: {stats.get('total_duration_hours', 0):.2f} hours")
        print(f"  Average: {stats.get('average_duration', 0):.2f} seconds")
        print(f"  Range: {stats.get('min_duration', 0):.2f}s - {stats.get('max_duration', 0):.2f}s")
        
        print(f"\nText:")
        print(f"  Total Characters: {stats.get('total_characters', 0):,}")
        print(f"  Average Length: {stats.get('average_text_length', 0):.0f} characters")
    
    def exit_program(self):
        """Exit the program"""
        self.clear_screen()
        print("\n" + "=" * 70)
        print("Thank you for using Amharic TTS Dataset Manager!")
        print("=" * 70 + "\n")
        self.running = False
        sys.exit(0)


def main():
    """Main entry point"""
    try:
        manager = DatasetManager()
        manager.main_menu()
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
