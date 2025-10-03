#!/usr/bin/env python3
"""
Fix Metadata Filename Prefixes
Removes doubled dataset name prefixes from metadata.csv
"""

import sys
from pathlib import Path
import argparse


def fix_metadata_prefixes(dataset_path: str, dry_run: bool = True):
    """
    Fix metadata by removing doubled dataset name prefixes
    
    Args:
        dataset_path: Path to dataset directory
        dry_run: If True, just show what would be changed
    """
    dataset_dir = Path(dataset_path)
    metadata_file = dataset_dir / "metadata.csv"
    wavs_dir = dataset_dir / "wavs"
    
    if not metadata_file.exists():
        print(f"❌ metadata.csv not found: {metadata_file}")
        return
    
    dataset_name = dataset_dir.name
    prefix_to_remove = f"{dataset_name}_"
    
    print("=" * 70)
    print(f"FIXING METADATA PREFIXES: {dataset_name}")
    print("=" * 70)
    print(f"Dataset: {dataset_dir}")
    print(f"Prefix to remove: {prefix_to_remove}")
    print()
    
    # Read metadata
    with open(metadata_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    fixed_lines = []
    changed_count = 0
    matched_count = 0
    
    print(f"Processing {len(lines)} entries...")
    print()
    
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        
        parts = line.split('|')
        if len(parts) < 2:
            fixed_lines.append(line)
            continue
        
        filename = parts[0]
        text = '|'.join(parts[1:])
        
        # Check if filename starts with the dataset prefix
        if filename.startswith(prefix_to_remove):
            new_filename = filename[len(prefix_to_remove):]
            fixed_line = f"{new_filename}|{text}"
            fixed_lines.append(fixed_line)
            changed_count += 1
            
            # Check if file exists with new name
            if wavs_dir.exists():
                # Try with and without .wav extension
                file_path = wavs_dir / new_filename
                if not file_path.exists() and not new_filename.endswith('.wav'):
                    file_path = wavs_dir / f"{new_filename}.wav"
                
                if file_path.exists():
                    matched_count += 1
            
            if i < 5:  # Show first 5 changes
                print(f"Example {i+1}:")
                print(f"  Before: {filename}")
                print(f"  After:  {new_filename}")
                if wavs_dir.exists() and file_path.exists():
                    print(f"  ✓ File exists!")
                print()
        else:
            fixed_lines.append(line)
    
    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total entries:      {len(lines)}")
    print(f"Changed:            {changed_count}")
    print(f"Unchanged:          {len(lines) - changed_count}")
    if wavs_dir.exists():
        print(f"Files matched:      {matched_count}")
        print(f"Files not found:    {changed_count - matched_count}")
    print("=" * 70)
    
    if not dry_run:
        # Backup original
        backup_file = dataset_dir / "metadata_original.csv"
        with open(backup_file, 'w', encoding='utf-8') as f:
            f.writelines(lines)
        print(f"\n✅ Backed up original to: {backup_file}")
        
        # Write fixed metadata
        with open(metadata_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(fixed_lines))
            if fixed_lines:
                f.write('\n')
        
        print(f"✅ Wrote fixed metadata with {len(fixed_lines)} entries")
        print()
        print("=" * 70)
        print("✅ METADATA FIXED!")
        print("=" * 70)
        print("\nYou can now restart training. The filenames should match!")
    else:
        print()
        print("💡 This was a DRY RUN. To apply the fix, run:")
        print(f"   python scripts/fix_metadata_prefixes.py --dataset {dataset_path} --apply")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fix doubled dataset name prefixes in metadata"
    )
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        help='Path to dataset directory'
    )
    parser.add_argument(
        '--apply',
        action='store_true',
        help='Apply the fix (default is dry-run)'
    )
    
    args = parser.parse_args()
    
    fix_metadata_prefixes(args.dataset, dry_run=not args.apply)
