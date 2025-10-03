#!/usr/bin/env python3
"""
Validate Dataset and Clean Metadata
Removes entries from metadata.csv for missing audio files
"""

import os
import sys
from pathlib import Path
import argparse


def validate_and_clean_dataset(dataset_path: str, fix: bool = False):
    """
    Validate dataset and optionally fix metadata
    
    Args:
        dataset_path: Path to dataset directory
        fix: If True, create cleaned metadata file
    """
    dataset_dir = Path(dataset_path)
    metadata_file = dataset_dir / "metadata.csv"
    wavs_dir = dataset_dir / "wavs"
    
    if not metadata_file.exists():
        print(f"❌ metadata.csv not found: {metadata_file}")
        return
    
    if not wavs_dir.exists():
        print(f"❌ wavs/ directory not found: {wavs_dir}")
        return
    
    print("=" * 70)
    print(f"VALIDATING DATASET: {dataset_dir.name}")
    print("=" * 70)
    
    # Read metadata
    valid_entries = []
    invalid_entries = []
    
    with open(metadata_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    total_entries = len(lines)
    print(f"\n📊 Total entries in metadata.csv: {total_entries}")
    print(f"🔍 Checking for missing audio files...\n")
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        parts = line.split('|')
        if len(parts) < 2:
            invalid_entries.append((line, "Invalid format"))
            continue
        
        audio_filename = parts[0]
        text = parts[1]
        
        # Check if audio file exists
        audio_path = wavs_dir / audio_filename
        
        # Try with .wav extension if not present
        if not audio_path.exists() and not audio_filename.endswith('.wav'):
            audio_path = wavs_dir / f"{audio_filename}.wav"
        
        if audio_path.exists():
            valid_entries.append(line)
        else:
            invalid_entries.append((audio_filename, "File not found"))
    
    # Print results
    print("=" * 70)
    print("VALIDATION RESULTS")
    print("=" * 70)
    print(f"✅ Valid entries:   {len(valid_entries)} ({len(valid_entries)/total_entries*100:.1f}%)")
    print(f"❌ Invalid entries: {len(invalid_entries)} ({len(invalid_entries)/total_entries*100:.1f}%)")
    print("=" * 70)
    
    if invalid_entries:
        print(f"\n⚠️  Found {len(invalid_entries)} missing files!")
        print(f"\nFirst 10 missing files:")
        for i, (filename, reason) in enumerate(invalid_entries[:10]):
            print(f"  {i+1}. {filename} - {reason}")
        
        if len(invalid_entries) > 10:
            print(f"  ... and {len(invalid_entries) - 10} more")
    
    # Fix if requested
    if fix and invalid_entries:
        print(f"\n🔧 FIXING METADATA...")
        
        # Backup original
        backup_file = dataset_dir / "metadata_backup.csv"
        with open(backup_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        print(f"✅ Backed up original to: {backup_file}")
        
        # Write cleaned metadata
        with open(metadata_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(valid_entries))
            if valid_entries:  # Add newline at end
                f.write('\n')
        
        print(f"✅ Wrote {len(valid_entries)} valid entries to metadata.csv")
        print(f"✅ Removed {len(invalid_entries)} invalid entries")
        
        # Also create validation metadata
        if valid_entries:
            val_size = max(1, len(valid_entries) // 10)  # 10% for validation
            val_entries = valid_entries[:val_size]
            
            val_file = dataset_dir / "metadata_val.csv"
            with open(val_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(val_entries))
                f.write('\n')
            print(f"✅ Created metadata_val.csv with {len(val_entries)} validation samples")
        
        print("\n" + "=" * 70)
        print("✅ DATASET CLEANED SUCCESSFULLY!")
        print("=" * 70)
        print(f"\nYour dataset now has {len(valid_entries)} valid samples")
        print(f"Ready to train! 🚀")
        
    elif not fix and invalid_entries:
        print(f"\n💡 To fix the metadata, run:")
        print(f"   python scripts/validate_dataset.py --dataset {dataset_path} --fix")
    
    return len(valid_entries), len(invalid_entries)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Validate dataset and clean metadata"
    )
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        help='Path to dataset directory (e.g., data/srt_datasets/Merged-Amharic_shanta-channel)'
    )
    parser.add_argument(
        '--fix',
        action='store_true',
        help='Fix metadata by removing invalid entries'
    )
    
    args = parser.parse_args()
    
    validate_and_clean_dataset(args.dataset, args.fix)
