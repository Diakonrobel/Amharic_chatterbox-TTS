"""
Example: SRT Dataset Import
============================

This script demonstrates how to programmatically use the SRT dataset builder
for importing and managing TTS training datasets.

Usage:
    python example_srt_import.py
"""

import sys
from pathlib import Path

# Add the src directory to path
sys.path.append(str(Path(__file__).parent.parent / "src" / "data_processing"))

from srt_dataset_builder import SRTDatasetBuilder


def example_single_import():
    """Example 1: Import a single SRT+media pair"""
    
    print("\n" + "="*70)
    print("EXAMPLE 1: Single Import")
    print("="*70 + "\n")
    
    # Initialize builder
    builder = SRTDatasetBuilder(base_output_dir="data/srt_datasets")
    
    # Import single dataset
    stats = builder.import_from_srt(
        srt_path="path/to/your/video.srt",
        media_path="path/to/your/video.mp4",
        dataset_name="example_dataset_1",
        speaker_name="speaker_01",
        auto_validate=True
    )
    
    # Print results
    print(f"\n✓ Import complete!")
    print(f"  Valid segments: {stats['valid_segments']}")
    print(f"  Total duration: {stats['total_duration_hours']:.2f} hours")
    print(f"  Average segment: {stats['average_duration']:.2f} seconds")


def example_batch_import():
    """Example 2: Batch import multiple files"""
    
    print("\n" + "="*70)
    print("EXAMPLE 2: Batch Import")
    print("="*70 + "\n")
    
    builder = SRTDatasetBuilder()
    
    # Define multiple file pairs
    file_pairs = [
        ("data/raw/video1.srt", "data/raw/video1.mp4", "dataset_1"),
        ("data/raw/video2.srt", "data/raw/video2.mp4", "dataset_2"),
        ("data/raw/audio3.srt", "data/raw/audio3.wav", "dataset_3"),
    ]
    
    imported_datasets = []
    
    for srt_path, media_path, dataset_name in file_pairs:
        # Check if files exist
        if not Path(srt_path).exists() or not Path(media_path).exists():
            print(f"⚠️  Skipping {dataset_name} (files not found)")
            continue
        
        print(f"\nImporting: {dataset_name}")
        print("-" * 70)
        
        try:
            stats = builder.import_from_srt(
                srt_path=srt_path,
                media_path=media_path,
                dataset_name=dataset_name,
                speaker_name=f"speaker_{len(imported_datasets)+1:02d}",
                auto_validate=True
            )
            
            imported_datasets.append(dataset_name)
            print(f"✓ {dataset_name}: {stats['valid_segments']} segments")
            
        except Exception as e:
            print(f"✗ Error: {e}")
    
    print(f"\n✓ Imported {len(imported_datasets)} datasets")
    return imported_datasets


def example_merge_datasets():
    """Example 3: Merge multiple datasets"""
    
    print("\n" + "="*70)
    print("EXAMPLE 3: Merge Datasets")
    print("="*70 + "\n")
    
    builder = SRTDatasetBuilder()
    
    # Merge multiple datasets
    stats = builder.merge_datasets(
        dataset_names=["dataset_1", "dataset_2", "dataset_3"],
        merged_name="merged_training_set",
        filter_invalid=True
    )
    
    print(f"\n✓ Merge complete!")
    print(f"  Total segments: {stats['valid_segments']}")
    print(f"  Total duration: {stats['total_duration_hours']:.2f} hours")
    print(f"  Ready for training!")


def example_inspect_datasets():
    """Example 4: List and inspect datasets"""
    
    print("\n" + "="*70)
    print("EXAMPLE 4: Inspect Datasets")
    print("="*70 + "\n")
    
    builder = SRTDatasetBuilder()
    
    # List all datasets
    datasets = builder.list_datasets()
    
    print(f"Found {len(datasets)} dataset(s):\n")
    
    for ds in datasets:
        print(f"📁 {ds['name']}")
        print(f"   Segments: {ds['valid_segments']}/{ds['segments']}")
        print(f"   Duration: {ds['duration_hours']:.2f} hours")
        print(f"   Path: {ds['path']}\n")
    
    # Get detailed statistics
    if datasets:
        total_hours = sum(ds['duration_hours'] for ds in datasets)
        total_segments = sum(ds['valid_segments'] for ds in datasets)
        
        print(f"Total across all datasets:")
        print(f"  Duration: {total_hours:.2f} hours")
        print(f"  Segments: {total_segments:,}")


def example_custom_validation():
    """Example 5: Custom validation settings"""
    
    print("\n" + "="*70)
    print("EXAMPLE 5: Custom Validation")
    print("="*70 + "\n")
    
    from srt_dataset_builder import DatasetValidator
    
    # Create custom validator with different thresholds
    validator = DatasetValidator(
        min_duration=2.0,   # Minimum 2 seconds
        max_duration=12.0,  # Maximum 12 seconds
        target_sr=22050
    )
    
    # Use it during validation
    # (You would integrate this into the import process)
    
    print("Custom validator created:")
    print(f"  Min duration: {validator.min_duration}s")
    print(f"  Max duration: {validator.max_duration}s")
    print(f"  Target sample rate: {validator.target_sr} Hz")


def example_complete_workflow():
    """Example 6: Complete workflow from import to training prep"""
    
    print("\n" + "="*70)
    print("EXAMPLE 6: Complete Workflow")
    print("="*70 + "\n")
    
    builder = SRTDatasetBuilder()
    
    # Step 1: Import datasets
    print("[1/4] Importing datasets...")
    
    datasets_to_import = [
        ("lecture1.srt", "lecture1.mp4", "lecture1"),
        ("lecture2.srt", "lecture2.mp4", "lecture2"),
    ]
    
    imported = []
    for srt, media, name in datasets_to_import:
        # In real usage, check if files exist
        print(f"  → Would import: {name}")
        imported.append(name)
    
    # Step 2: Merge if multiple
    print("\n[2/4] Merging datasets...")
    if len(imported) > 1:
        print(f"  → Would merge: {', '.join(imported)}")
        final_dataset = "amharic_lectures_full"
    else:
        final_dataset = imported[0]
    
    # Step 3: Review statistics
    print("\n[3/4] Reviewing statistics...")
    datasets = builder.list_datasets()
    
    if datasets:
        for ds in datasets[:3]:  # Show first 3
            print(f"  → {ds['name']}: {ds['duration_hours']:.2f}h, "
                  f"{ds['valid_segments']} segments")
    
    # Step 4: Prepare for training
    print("\n[4/4] Preparing for training...")
    print(f"  → Dataset location: data/srt_datasets/{final_dataset}")
    print(f"  → Metadata format: LJSpeech (metadata.csv)")
    print(f"  → Next: Train tokenizer and update config")
    
    print("\n✓ Workflow complete!")
    print("\nNext commands:")
    print("  1. Copy dataset to data/processed/")
    print("  2. python -m src.tokenizer.amharic_tokenizer")
    print("  3. Update config/training_config.yaml")
    print("  4. Start training")


def main():
    """Run all examples"""
    
    print("\n" + "="*70)
    print("SRT DATASET BUILDER - USAGE EXAMPLES")
    print("="*70)
    
    print("\nThese examples demonstrate how to use the SRT dataset system.")
    print("Note: Some examples require actual SRT and media files to work.")
    print("\nAvailable examples:")
    print("  1. Single import")
    print("  2. Batch import")
    print("  3. Merge datasets")
    print("  4. Inspect datasets")
    print("  5. Custom validation")
    print("  6. Complete workflow")
    
    choice = input("\nEnter example number to run (1-6, or 'all'): ").strip()
    
    if choice == "1":
        example_single_import()
    elif choice == "2":
        example_batch_import()
    elif choice == "3":
        example_merge_datasets()
    elif choice == "4":
        example_inspect_datasets()
    elif choice == "5":
        example_custom_validation()
    elif choice == "6":
        example_complete_workflow()
    elif choice.lower() == "all":
        example_single_import()
        example_batch_import()
        example_merge_datasets()
        example_inspect_datasets()
        example_custom_validation()
        example_complete_workflow()
    else:
        print("Invalid choice!")
        return
    
    print("\n" + "="*70)
    print("EXAMPLE COMPLETE")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
