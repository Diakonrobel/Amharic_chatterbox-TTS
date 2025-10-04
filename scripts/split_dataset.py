"""
Split Dataset into Train/Validation/Test Sets

This script splits a metadata.csv file into three separate files:
- metadata_train.csv (e.g., 80%)
- metadata_val.csv   (e.g., 15%)
- metadata_test.csv  (e.g., 5%)

Usage:
    python scripts/split_dataset.py --dataset data/srt_datasets/my_dataset --train 0.80 --val 0.15 --test 0.05
"""

import argparse
import random
from pathlib import Path
from typing import Tuple, List


def load_metadata(metadata_path: Path) -> List[str]:
    """Load metadata lines from file"""
    with open(metadata_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    return lines


def split_data(lines: List[str], train_ratio: float, val_ratio: float, test_ratio: float) -> Tuple[List[str], List[str], List[str]]:
    """Split data into train/val/test sets"""
    # Verify ratios sum to 1.0
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 0.01:
        raise ValueError(f"Ratios must sum to 1.0, got {total}")
    
    # Shuffle data for random split
    data = lines.copy()
    random.shuffle(data)
    
    # Calculate split points
    n_total = len(data)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    # Split
    train_data = data[:n_train]
    val_data = data[n_train:n_train + n_val]
    test_data = data[n_train + n_val:]
    
    return train_data, val_data, test_data


def save_split(data: List[str], output_path: Path):
    """Save split data to file"""
    with open(output_path, 'w', encoding='utf-8') as f:
        for line in data:
            f.write(line + '\n')


def calculate_duration(dataset_dir: Path, lines: List[str]) -> float:
    """Calculate total duration of audio files (if dataset_info.json exists)"""
    import json
    
    info_file = dataset_dir / "dataset_info.json"
    if not info_file.exists():
        return 0.0
    
    with open(info_file, 'r', encoding='utf-8') as f:
        info = json.load(f)
    
    # Get duration per sample (total / count)
    total_duration = info.get('total_duration', 0)
    total_samples = info.get('num_samples', len(lines))
    
    if total_samples == 0:
        return 0.0
    
    avg_duration_per_sample = total_duration / total_samples
    return avg_duration_per_sample * len(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Split dataset into train/validation/test sets"
    )
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        help='Path to dataset directory containing metadata.csv'
    )
    parser.add_argument(
        '--train',
        type=float,
        default=0.80,
        help='Training set ratio (default: 0.80)'
    )
    parser.add_argument(
        '--val',
        type=float,
        default=0.15,
        help='Validation set ratio (default: 0.15)'
    )
    parser.add_argument(
        '--test',
        type=float,
        default=0.05,
        help='Test set ratio (default: 0.05)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    parser.add_argument(
        '--backup',
        action='store_true',
        help='Backup original metadata.csv before splitting'
    )
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    
    # Setup paths
    dataset_dir = Path(args.dataset)
    metadata_path = dataset_dir / 'metadata.csv'
    
    if not metadata_path.exists():
        print(f"❌ Error: {metadata_path} not found!")
        return 1
    
    print("=" * 60)
    print("DATASET SPLITTING")
    print("=" * 60)
    print(f"Dataset: {dataset_dir}")
    print(f"Split ratios: Train={args.train:.0%}, Val={args.val:.0%}, Test={args.test:.0%}")
    print(f"Random seed: {args.seed}")
    print()
    
    # Backup original if requested
    if args.backup:
        backup_path = dataset_dir / 'metadata_original.csv'
        if not backup_path.exists():
            import shutil
            shutil.copy(metadata_path, backup_path)
            print(f"✓ Backed up original to: {backup_path}")
        else:
            print(f"ℹ Backup already exists: {backup_path}")
    
    # Load data
    print(f"Loading metadata from: {metadata_path}")
    lines = load_metadata(metadata_path)
    print(f"✓ Loaded {len(lines)} samples")
    
    # Split data
    print(f"\nSplitting data...")
    train_data, val_data, test_data = split_data(
        lines, args.train, args.val, args.test
    )
    
    # Calculate durations (if possible)
    train_dur = calculate_duration(dataset_dir, train_data)
    val_dur = calculate_duration(dataset_dir, val_data)
    test_dur = calculate_duration(dataset_dir, test_data)
    
    # Save splits
    train_path = dataset_dir / 'metadata_train.csv'
    val_path = dataset_dir / 'metadata_val.csv'
    test_path = dataset_dir / 'metadata_test.csv'
    
    save_split(train_data, train_path)
    save_split(val_data, val_path)
    save_split(test_data, test_path)
    
    print("\n" + "=" * 60)
    print("SPLITTING COMPLETE")
    print("=" * 60)
    print(f"Train set: {len(train_data)} samples ({len(train_data)/len(lines)*100:.1f}%)")
    if train_dur > 0:
        print(f"           {train_dur/3600:.2f} hours")
    print(f"  → {train_path}")
    print()
    print(f"Val set:   {len(val_data)} samples ({len(val_data)/len(lines)*100:.1f}%)")
    if val_dur > 0:
        print(f"           {val_dur/3600:.2f} hours")
    print(f"  → {val_path}")
    print()
    print(f"Test set:  {len(test_data)} samples ({len(test_data)/len(lines)*100:.1f}%)")
    if test_dur > 0:
        print(f"           {test_dur/3600:.2f} hours")
    print(f"  → {test_path}")
    print()
    print("=" * 60)
    print("NEXT STEPS:")
    print("=" * 60)
    print("1. Update your training config to use metadata_train.csv")
    print("2. Training will automatically use metadata_val.csv if it exists")
    print("3. Use metadata_test.csv for final model evaluation")
    print()
    print("Example config update:")
    print("  paths:")
    print(f"    data_dir: {dataset_dir}")
    print()
    print("The training script will automatically find:")
    print(f"  - {train_path}")
    print(f"  - {val_path}")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    exit(main())
