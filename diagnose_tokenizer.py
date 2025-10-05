#!/usr/bin/env python3
"""
Diagnose tokenizer loading issue on Lightning AI
"""

from pathlib import Path
import json

print("=" * 70)
print("🔍 TOKENIZER DIAGNOSTIC")
print("=" * 70)

project_root = Path(__file__).parent

# Check all possible tokenizer paths
tokenizer_candidates = [
    project_root / 'tokenizers' / 'am-merged_merged.json',
    project_root / 'models' / 'tokenizer' / 'Am_tokenizer_merged.json',
    project_root / 'models' / 'tokenizer' / 'amharic_tokenizer',
    project_root / 'models' / 'tokenizer',
]

print("\n📂 Checking tokenizer paths:")
print("-" * 70)

found_tokenizers = []

for i, path in enumerate(tokenizer_candidates, 1):
    exists = path.exists()
    status = "✅ EXISTS" if exists else "❌ NOT FOUND"
    print(f"\n{i}. {path}")
    print(f"   Status: {status}")
    
    if exists:
        if path.is_dir():
            print(f"   Type: Directory")
            # List contents
            files = list(path.glob("*"))
            if files:
                print(f"   Contents ({len(files)} items):")
                for f in files[:5]:  # Show first 5
                    print(f"     - {f.name}")
        else:
            print(f"   Type: File")
            # Try to load and check vocab size
            try:
                if path.suffix == '.json':
                    with open(path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    # Check structure
                    if isinstance(data, dict):
                        if 'model' in data and 'vocab' in data['model']:
                            vocab = data['model']['vocab']
                            vocab_size = len(vocab)
                        else:
                            vocab_size = len(data)
                        
                        print(f"   Vocab size: {vocab_size}")
                        found_tokenizers.append((path, vocab_size))
            except Exception as e:
                print(f"   Error reading: {e}")

print("\n" + "=" * 70)
print("📊 SUMMARY")
print("=" * 70)

if found_tokenizers:
    print(f"\n✅ Found {len(found_tokenizers)} tokenizer(s):")
    for path, size in found_tokenizers:
        status = "✅ CORRECT!" if size == 2559 else f"⚠️  Expected 2559, got {size}"
        print(f"\n  • {path.name}")
        print(f"    Path: {path}")
        print(f"    Vocab size: {size} {status}")
else:
    print("\n❌ NO TOKENIZERS FOUND!")
    print("\nThis explains why inference loads old tokenizer!")
    print("\n🔧 SOLUTION:")
    print("  The tokenizer file needs to be in one of the checked paths.")
    print("  Most likely it's in a different location on Lightning AI.")

# Check what the inference.py is looking for
print("\n" + "=" * 70)
print("📄 CHECKING INFERENCE.PY")
print("=" * 70)

inference_file = project_root / 'src' / 'inference' / 'inference.py'
if inference_file.exists():
    with open(inference_file, 'r') as f:
        content = f.read()
    
    if 'am-merged_merged.json' in content:
        print("\n✅ inference.py HAS the fix (looks for am-merged_merged.json)")
    else:
        print("\n❌ inference.py DOES NOT have the fix yet!")
        print("   Git pull may not have happened or file not updated.")
else:
    print("\n❌ inference.py not found!")

print("\n" + "=" * 70)
