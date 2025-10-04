#!/usr/bin/env python3
"""
🔧 Fix UNK Token Issues in Amharic TTS Training

This script diagnoses and fixes the root causes of UNK token warnings
during Amharic TTS training with Chatterbox.

Usage:
    python scripts/fix_unk_issues.py --data-path data/srt_datasets/my_dataset --config config/training_config.yaml
"""

import os
import sys
import json
import yaml
import argparse
from pathlib import Path
from typing import Dict, List, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

try:
    from src.g2p.amharic_g2p import AmharicG2P
    from src.tokenizer.amharic_tokenizer import AmharicTokenizer, train_amharic_tokenizer
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("Make sure you're running from the project root and have installed dependencies.")
    sys.exit(1)


class UNKTokenFixer:
    """Diagnoses and fixes UNK token issues"""
    
    def __init__(self, data_path: str, config_path: str):
        self.data_path = Path(data_path)
        self.config_path = Path(config_path)
        self.issues_found = []
        self.fixes_applied = []
        
        print("🔍 Amharic TTS UNK Token Issue Fixer")
        print("=" * 50)
    
    def load_config(self) -> Dict:
        """Load training configuration"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            print(f"✓ Config loaded from {self.config_path}")
            return config
        except Exception as e:
            print(f"❌ Failed to load config: {e}")
            return {}
    
    def diagnose_data(self) -> Dict:
        """Diagnose data-related issues"""
        print("\n📊 DIAGNOSING DATA ISSUES")
        print("-" * 30)
        
        issues = {}
        
        # Check if metadata exists
        metadata_file = self.data_path / 'metadata.csv'
        if not metadata_file.exists():
            issues['missing_metadata'] = f"Metadata file not found: {metadata_file}"
            print(f"❌ {issues['missing_metadata']}")
        else:
            print(f"✓ Metadata file found: {metadata_file}")
            
            # Analyze metadata content
            try:
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                if not lines:
                    issues['empty_metadata'] = "Metadata file is empty"
                    print(f"❌ {issues['empty_metadata']}")
                else:
                    print(f"✓ Metadata has {len(lines)} entries")
                    
                    # Check first few lines for format
                    sample_lines = lines[:3]
                    amharic_chars_found = 0
                    
                    for i, line in enumerate(sample_lines):
                        parts = line.strip().split('|')
                        if len(parts) >= 2:
                            text = parts[1]
                            # Count Amharic characters (Ethiopic script)
                            amharic_count = sum(1 for c in text if 0x1200 <= ord(c) <= 0x137F)
                            amharic_chars_found += amharic_count
                            print(f"  Line {i+1}: '{text[:50]}...' ({amharic_count} Amharic chars)")
                    
                    if amharic_chars_found == 0:
                        issues['no_amharic_text'] = "No Amharic characters found in sample data"
                        print(f"❌ {issues['no_amharic_text']}")
                    else:
                        print(f"✓ Found {amharic_chars_found} Amharic characters in samples")
                        
            except Exception as e:
                issues['metadata_read_error'] = f"Failed to read metadata: {e}"
                print(f"❌ {issues['metadata_read_error']}")
        
        return issues
    
    def diagnose_tokenizer(self) -> Dict:
        """Diagnose tokenizer-related issues"""
        print("\n🔤 DIAGNOSING TOKENIZER ISSUES")
        print("-" * 30)
        
        issues = {}
        
        # Check if Amharic tokenizer exists
        amharic_tokenizer_paths = [
            "models/tokenizer",
            "models/tokenizer/amharic_tokenizer"
        ]
        
        amharic_tokenizer_found = False
        for path in amharic_tokenizer_paths:
            if Path(path).exists():
                print(f"✓ Found tokenizer at: {path}")
                amharic_tokenizer_found = True
                
                # Check tokenizer files
                vocab_file = Path(path) / "vocab.json"
                sp_model = Path(path) / "sentencepiece.model"
                
                if vocab_file.exists():
                    try:
                        with open(vocab_file, 'r', encoding='utf-8') as f:
                            vocab = json.load(f)
                        print(f"  ✓ Vocabulary size: {len(vocab)}")
                        
                        # Check for Amharic characters in vocab
                        amharic_tokens = [token for token in vocab.keys() 
                                        if any(0x1200 <= ord(c) <= 0x137F for c in token)]
                        print(f"  ✓ Amharic tokens: {len(amharic_tokens)}")
                        
                        if len(amharic_tokens) < 100:
                            issues['insufficient_amharic_vocab'] = f"Only {len(amharic_tokens)} Amharic tokens in vocabulary"
                            
                    except Exception as e:
                        issues['vocab_read_error'] = f"Failed to read vocabulary: {e}"
                        print(f"  ❌ {issues['vocab_read_error']}")
                else:
                    issues['missing_vocab'] = f"vocab.json not found in {path}"
                    print(f"  ❌ {issues['missing_vocab']}")
                
                if sp_model.exists():
                    print(f"  ✓ SentencePiece model found")
                else:
                    issues['missing_sentencepiece'] = f"sentencepiece.model not found in {path}"
                    print(f"  ❌ {issues['missing_sentencepiece']}")
                
                break
        
        if not amharic_tokenizer_found:
            issues['no_amharic_tokenizer'] = "No Amharic tokenizer found"
            print(f"❌ {issues['no_amharic_tokenizer']}")
        
        # Check for merged tokenizer
        merged_vocab_path = Path("models/tokenizer/merged_vocab.json")
        if merged_vocab_path.exists():
            try:
                with open(merged_vocab_path, 'r', encoding='utf-8') as f:
                    merged_vocab = json.load(f)
                print(f"✓ Merged vocabulary found: {len(merged_vocab)} tokens")
            except Exception as e:
                issues['merged_vocab_error'] = f"Failed to read merged vocabulary: {e}"
                print(f"❌ {issues['merged_vocab_error']}")
        else:
            issues['no_merged_vocab'] = "Merged vocabulary not found"
            print(f"❌ {issues['no_merged_vocab']}")
        
        return issues
    
    def diagnose_model_config(self, config: Dict) -> Dict:
        """Diagnose model configuration issues"""
        print("\n⚙️ DIAGNOSING MODEL CONFIG")
        print("-" * 30)
        
        issues = {}
        
        # Check vocabulary size settings
        model_config = config.get('model', {})
        n_vocab = model_config.get('n_vocab', 0)
        freeze_until = model_config.get('freeze_until_index', 0)
        
        print(f"Config n_vocab: {n_vocab}")
        print(f"Config freeze_until_index: {freeze_until}")
        
        if n_vocab == 0:
            issues['missing_n_vocab'] = "n_vocab not set in config"
            print(f"❌ {issues['missing_n_vocab']}")
        
        if freeze_until == 0:
            issues['missing_freeze_index'] = "freeze_until_index not set in config"
            print(f"❌ {issues['missing_freeze_index']}")
        
        # Check if extended model exists
        pretrained_path = config.get('finetuning', {}).get('pretrained_model', '')
        if pretrained_path:
            if Path(pretrained_path).exists():
                print(f"✓ Extended model found: {pretrained_path}")
            else:
                issues['missing_extended_model'] = f"Extended model not found: {pretrained_path}"
                print(f"❌ {issues['missing_extended_model']}")
        else:
            issues['no_pretrained_path'] = "No pretrained model path in config"
            print(f"❌ {issues['no_pretrained_path']}")
        
        return issues
    
    def fix_tokenizer_issues(self, data_issues: Dict, tokenizer_issues: Dict) -> bool:
        """Fix tokenizer-related issues"""
        print("\n🔧 FIXING TOKENIZER ISSUES")
        print("-" * 30)
        
        success = True
        
        # If no Amharic tokenizer, train one
        if 'no_amharic_tokenizer' in tokenizer_issues or 'insufficient_amharic_vocab' in tokenizer_issues:
            print("Training new Amharic tokenizer...")
            
            metadata_file = self.data_path / 'metadata.csv'
            if metadata_file.exists():
                try:
                    # Create output directory
                    output_dir = Path("models/tokenizer/amharic_tokenizer")
                    output_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Train tokenizer
                    tokenizer = train_amharic_tokenizer(
                        data_file=str(metadata_file),
                        output_dir=str(output_dir),
                        vocab_size=1000  # Reasonable size for Amharic
                    )
                    
                    print("✓ Amharic tokenizer trained successfully")
                    self.fixes_applied.append("Trained new Amharic tokenizer")
                    
                except Exception as e:
                    print(f"❌ Failed to train tokenizer: {e}")
                    success = False
            else:
                print("❌ Cannot train tokenizer: no metadata file")
                success = False
        
        return success
    
    def fix_model_config(self, config: Dict, config_issues: Dict) -> Dict:
        """Fix model configuration issues"""
        print("\n🔧 FIXING CONFIG ISSUES")
        print("-" * 30)
        
        fixed_config = config.copy()
        changes_made = False
        
        # Fix vocabulary size if merged tokenizer exists
        merged_vocab_path = Path("models/tokenizer/merged_vocab.json")
        if merged_vocab_path.exists():
            try:
                with open(merged_vocab_path, 'r', encoding='utf-8') as f:
                    merged_vocab = json.load(f)
                
                actual_vocab_size = len(merged_vocab)
                config_vocab_size = fixed_config.get('model', {}).get('n_vocab', 0)
                
                if config_vocab_size != actual_vocab_size:
                    print(f"Updating n_vocab: {config_vocab_size} → {actual_vocab_size}")
                    if 'model' not in fixed_config:
                        fixed_config['model'] = {}
                    fixed_config['model']['n_vocab'] = actual_vocab_size
                    changes_made = True
                    self.fixes_applied.append(f"Updated n_vocab to {actual_vocab_size}")
                    
            except Exception as e:
                print(f"❌ Failed to read merged vocab: {e}")
        
        # Set reasonable defaults if missing
        if 'missing_freeze_index' in config_issues:
            print("Setting freeze_until_index to 2454 (Chatterbox multilingual size)")
            if 'model' not in fixed_config:
                fixed_config['model'] = {}
            fixed_config['model']['freeze_until_index'] = 2454
            changes_made = True
            self.fixes_applied.append("Set freeze_until_index to 2454")
        
        # Save updated config if changes were made
        if changes_made:
            try:
                with open(self.config_path, 'w', encoding='utf-8') as f:
                    yaml.dump(fixed_config, f, default_flow_style=False, allow_unicode=True)
                print(f"✓ Updated configuration saved to {self.config_path}")
                self.fixes_applied.append("Updated training configuration")
            except Exception as e:
                print(f"❌ Failed to save config: {e}")
        
        return fixed_config
    
    def test_tokenization(self) -> bool:
        """Test tokenization with sample Amharic text"""
        print("\n🧪 TESTING TOKENIZATION")
        print("-" * 30)
        
        test_texts = [
            "ሰላም ለዓለም",
            "አዲስ አበባ የኢትዮጵያ ዋና ከተማ ናት",
            "እንኳን ደህና መጡ"
        ]
        
        try:
            # Initialize G2P
            g2p = AmharicG2P()
            print("✓ G2P system initialized")
            
            # Try to load tokenizer
            tokenizer_paths = [
                "models/tokenizer/amharic_tokenizer",
                "models/tokenizer"
            ]
            
            tokenizer = None
            for path in tokenizer_paths:
                if Path(path).exists():
                    try:
                        tokenizer = AmharicTokenizer.load(path, g2p=g2p)
                        print(f"✓ Tokenizer loaded from {path}")
                        break
                    except Exception as e:
                        print(f"⚠ Failed to load from {path}: {e}")
            
            if not tokenizer:
                print("❌ No tokenizer could be loaded")
                return False
            
            # Test each text
            all_successful = True
            for text in test_texts:
                try:
                    # Test G2P
                    phonemes = g2p.grapheme_to_phoneme(text)
                    print(f"  '{text}' → '{phonemes}'")
                    
                    # Test tokenization
                    tokens = tokenizer.encode(text, use_phonemes=True)
                    print(f"    Tokens: {tokens[:10]}{'...' if len(tokens) > 10 else ''}")
                    
                    # Check for excessive UNK tokens
                    unk_id = tokenizer.vocab.get('<UNK>', 1)
                    unk_count = tokens.count(unk_id)
                    unk_ratio = unk_count / len(tokens) if tokens else 1.0
                    
                    if unk_ratio > 0.3:
                        print(f"    ⚠ High UNK ratio: {unk_ratio:.1%}")
                        all_successful = False
                    else:
                        print(f"    ✓ UNK ratio: {unk_ratio:.1%}")
                        
                except Exception as e:
                    print(f"    ❌ Failed: {e}")
                    all_successful = False
            
            return all_successful
            
        except Exception as e:
            print(f"❌ Testing failed: {e}")
            return False
    
    def generate_report(self) -> str:
        """Generate diagnostic report"""
        report = []
        report.append("🔍 AMHARIC TTS UNK TOKEN DIAGNOSTIC REPORT")
        report.append("=" * 50)
        
        if self.issues_found:
            report.append("\n❌ ISSUES FOUND:")
            for issue in self.issues_found:
                report.append(f"  • {issue}")
        else:
            report.append("\n✓ No issues found!")
        
        if self.fixes_applied:
            report.append(f"\n🔧 FIXES APPLIED:")
            for fix in self.fixes_applied:
                report.append(f"  • {fix}")
        
        return "\n".join(report)
    
    def run_diagnosis(self):
        """Run complete diagnosis and fixes"""
        print("Starting comprehensive diagnosis...\n")
        
        # Load configuration
        config = self.load_config()
        
        # Run diagnostics
        data_issues = self.diagnose_data()
        tokenizer_issues = self.diagnose_tokenizer()
        config_issues = self.diagnose_model_config(config)
        
        # Collect all issues
        all_issues = {**data_issues, **tokenizer_issues, **config_issues}
        self.issues_found.extend(all_issues.values())
        
        # Apply fixes
        if tokenizer_issues:
            self.fix_tokenizer_issues(data_issues, tokenizer_issues)
        
        if config_issues:
            self.fix_model_config(config, config_issues)
        
        # Test the fixes
        self.test_tokenization()
        
        # Generate final report
        print("\n" + self.generate_report())
        
        if not self.issues_found:
            print("\n🎉 All checks passed! Your setup should work for training.")
        else:
            print("\n⚠️ Some issues remain. Check the report above.")


def main():
    parser = argparse.ArgumentParser(description="Fix UNK token issues in Amharic TTS")
    parser.add_argument("--data-path", required=True, help="Path to dataset directory")
    parser.add_argument("--config", required=True, help="Path to training config file")
    
    args = parser.parse_args()
    
    fixer = UNKTokenFixer(args.data_path, args.config)
    fixer.run_diagnosis()


if __name__ == "__main__":
    main()