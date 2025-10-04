#!/usr/bin/env python3
"""
⚡ Lightning AI Tokenizer Setup Script

This script should be run on Lightning AI to properly set up the Amharic tokenizer
after the diagnostic found missing vocab.json and sentencepiece.model files.

Usage on Lightning AI:
    python setup_tokenizer_lightning.py --dataset merged_3
"""

import os
import sys
import argparse
from pathlib import Path

# Add project root
sys.path.append('.')

def setup_tokenizer_on_lightning(dataset_name: str = "merged_3", vocab_size: int = 1000):
    """
    Complete tokenizer setup for Lightning AI environment
    """
    print("⚡ LIGHTNING AI - AMHARIC TOKENIZER SETUP")
    print("=" * 50)
    
    # Step 1: Install dependencies
    print("\n📦 Step 1: Installing dependencies...")
    os.system("pip install sentencepiece -q")
    print("✓ sentencepiece installed")
    
    # Step 2: Train Amharic tokenizer
    print(f"\n🔤 Step 2: Training Amharic tokenizer on '{dataset_name}' dataset...")
    
    from src.tokenizer.amharic_tokenizer import train_amharic_tokenizer
    
    metadata_path = f"data/srt_datasets/{dataset_name}/metadata.csv"
    output_dir = "models/tokenizer/amharic_tokenizer"
    
    if not Path(metadata_path).exists():
        print(f"❌ Error: Dataset not found at {metadata_path}")
        print("\nAvailable datasets:")
        srt_dir = Path("data/srt_datasets")
        if srt_dir.exists():
            for item in srt_dir.iterdir():
                if item.is_dir() and (item / "metadata.csv").exists():
                    print(f"  - {item.name}")
        return False
    
    try:
        print(f"Training on: {metadata_path}")
        print(f"Vocab size: {vocab_size}")
        print(f"Output: {output_dir}")
        print("\nThis may take 2-5 minutes...")
        
        tokenizer = train_amharic_tokenizer(
            data_file=metadata_path,
            output_dir=output_dir,
            vocab_size=vocab_size
        )
        
        print("\n✓ Tokenizer trained successfully!")
        
        # Verify files were created
        required_files = [
            Path(output_dir) / "vocab.json",
            Path(output_dir) / "sentencepiece.model",
            Path(output_dir) / "config.json"
        ]
        
        print("\n📁 Verifying tokenizer files:")
        all_exist = True
        for file_path in required_files:
            exists = file_path.exists()
            status = "✓" if exists else "❌"
            print(f"  {status} {file_path}")
            all_exist = all_exist and exists
        
        if not all_exist:
            print("\n⚠️ Warning: Some tokenizer files are missing!")
            return False
        
        # Step 3: Update config to match actual vocab size
        print("\n⚙️ Step 3: Checking vocabulary size...")
        
        import json
        with open(Path(output_dir) / "vocab.json", 'r', encoding='utf-8') as f:
            vocab = json.load(f)
        
        actual_vocab_size = len(vocab)
        print(f"Actual Amharic vocab size: {actual_vocab_size}")
        
        # Check if merged vocab exists
        merged_vocab_path = Path("models/tokenizer/merged_vocab.json")
        if merged_vocab_path.exists():
            with open(merged_vocab_path, 'r', encoding='utf-8') as f:
                merged_vocab = json.load(f)
            merged_size = len(merged_vocab)
            print(f"Merged vocab size: {merged_size}")
            
            # Update config
            import yaml
            config_path = Path("config/training_config.yaml")
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            config_vocab_size = config.get('model', {}).get('n_vocab', 0)
            print(f"Config n_vocab: {config_vocab_size}")
            
            if config_vocab_size != merged_size:
                print(f"\n⚠️ Config mismatch detected!")
                print(f"  Updating config: {config_vocab_size} → {merged_size}")
                
                if 'model' not in config:
                    config['model'] = {}
                config['model']['n_vocab'] = merged_size
                
                with open(config_path, 'w', encoding='utf-8') as f:
                    yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
                
                print("✓ Config updated!")
        
        # Step 4: Test tokenization
        print("\n🧪 Step 4: Testing tokenization...")
        
        from src.g2p.amharic_g2p import AmharicG2P
        from src.tokenizer.amharic_tokenizer import AmharicTokenizer
        
        g2p = AmharicG2P()
        tokenizer = AmharicTokenizer.load(output_dir, g2p=g2p)
        
        test_texts = [
            "ሰላም ለዓለም",
            "አዲስ አበባ የኢትዮጵያ ዋና ከተማ ናት"
        ]
        
        all_good = True
        for text in test_texts:
            tokens = tokenizer.encode(text, use_phonemes=True)
            unk_id = tokenizer.vocab.get('<UNK>', 1)
            unk_ratio = tokens.count(unk_id) / len(tokens) if tokens else 1.0
            
            status = "✓" if unk_ratio < 0.3 else "❌"
            print(f"  {status} '{text[:20]}...' - UNK ratio: {unk_ratio:.1%}")
            
            if unk_ratio >= 0.3:
                all_good = False
        
        if all_good:
            print("\n🎉 SUCCESS! Tokenizer is working perfectly!")
            print("\nYou can now start training:")
            print("  python src/training/train.py --config config/training_config.yaml")
            return True
        else:
            print("\n⚠️ Warning: High UNK ratios detected. Consider retraining with larger vocab_size.")
            return False
            
    except Exception as e:
        print(f"\n❌ Error during tokenizer training: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Setup Amharic tokenizer on Lightning AI")
    parser.add_argument("--dataset", default="merged_3", help="Dataset name (default: merged_3)")
    parser.add_argument("--vocab-size", type=int, default=1000, help="Vocabulary size (default: 1000)")
    
    args = parser.parse_args()
    
    success = setup_tokenizer_on_lightning(args.dataset, args.vocab_size)
    
    if success:
        print("\n" + "="*50)
        print("✅ SETUP COMPLETE!")
        print("="*50)
        sys.exit(0)
    else:
        print("\n" + "="*50)
        print("❌ SETUP FAILED - Check errors above")
        print("="*50)
        sys.exit(1)


if __name__ == "__main__":
    main()
