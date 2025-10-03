"""
Quick Analysis of Downloaded Chatterbox Components
Run this to understand what you have and what's next
"""

from pathlib import Path
import json

print("="*70)
print("CHATTERBOX COMPONENTS ANALYSIS")
print("="*70)

base_path = Path("models/pretrained/chatterbox")

components = [
    ("mtl_tokenizer.json", "Multilingual Tokenizer"),
    ("t3_mtl23ls_v2.safetensors", "T3 Model (Main TTS)"),
    ("s3gen.safetensors", "Speech Generator"),
    ("ve.safetensors", "Voice Encoder"),
    ("conds.pt", "Conditioning Embeddings"),
]

print("\n✓ Downloaded Components:\n")
for filename, description in components:
    filepath = base_path / filename
    if filepath.exists():
        size_mb = filepath.stat().st_size / (1024**2)
        print(f"  ✓ {description}")
        print(f"    File: {filename}")
        print(f"    Size: {size_mb:.2f} MB")
        print()
    else:
        print(f"  ✗ {description} - NOT FOUND")
        print(f"    Expected: {filename}")
        print()

# Check extracted tokenizer
tokenizer_path = Path("models/pretrained/chatterbox_tokenizer.json")
print("="*70)
print("EXTRACTED TOKENIZER")
print("="*70)

if tokenizer_path.exists():
    print(f"\n✓ Tokenizer extracted to: {tokenizer_path}")
    
    with open(tokenizer_path, 'r', encoding='utf-8') as f:
        vocab = json.load(f)
    
    print(f"  Vocabulary size: {len(vocab)} tokens")
    print(f"  Sample tokens:")
    for i, (token, idx) in enumerate(list(vocab.items())[:10]):
        print(f"    {idx}: '{token}'")
    if len(vocab) > 10:
        print(f"    ... and {len(vocab) - 10} more")
else:
    print(f"\n✗ Extracted tokenizer not found at: {tokenizer_path}")

print("\n" + "="*70)
print("NEXT STEPS")
print("="*70)

steps_complete = []
steps_todo = []

# Check what's done
if (base_path / "t3_mtl23ls_v2.safetensors").exists():
    steps_complete.append("✓ Chatterbox model downloaded")
else:
    steps_todo.append("Download Chatterbox model")

if tokenizer_path.exists():
    steps_complete.append("✓ Base tokenizer extracted")
else:
    steps_todo.append("Extract base tokenizer")

# Check Amharic tokenizer
amharic_tok_path = Path("models/tokenizer/amharic_tokenizer")
if amharic_tok_path.exists():
    steps_complete.append("✓ Amharic tokenizer trained")
else:
    steps_todo.append("Train Amharic tokenizer (Gradio Tab 4)")

# Check merged tokenizer
merged_tok_path = Path("models/tokenizer/merged")
if merged_tok_path.exists():
    steps_complete.append("✓ Tokenizers merged")
else:
    steps_todo.append("Merge tokenizers (Gradio Tab 5)")

# Check extended model
extended_model = Path("models/pretrained/chatterbox_extended.pt")
if extended_model.exists():
    size_mb = extended_model.stat().st_size / (1024**2)
    steps_complete.append(f"✓ Model embeddings extended ({size_mb:.2f} MB)")
else:
    steps_todo.append("Extend model embeddings (Gradio Tab 5)")

# Check dataset
dataset_path = Path("data/srt_datasets")
if dataset_path.exists() and list(dataset_path.glob("*/")):
    datasets = [d.name for d in dataset_path.glob("*/") if d.is_dir()]
    steps_complete.append(f"✓ Datasets ready: {', '.join(datasets)}")
else:
    steps_todo.append("Import dataset (Gradio Tab 2)")

print("\n✅ Completed:")
for step in steps_complete:
    print(f"  {step}")

if steps_todo:
    print("\n⏳ To Do:")
    for i, step in enumerate(steps_todo, 1):
        print(f"  {i}. {step}")
else:
    print("\n🎉 All prerequisites complete! Ready to train!")

print("\n" + "="*70)
print("RECOMMENDED NEXT ACTION")
print("="*70)

if not (amharic_tok_path).exists():
    print("""
→ Train Amharic Tokenizer
  1. Go to Gradio UI Tab 4
  2. Set dataset path: data/srt_datasets/your_dataset/metadata.csv
  3. Set vocab size: 500-1000
  4. Click 'Train Tokenizer'
""")
elif not merged_tok_path.exists():
    print("""
→ Merge Tokenizers
  1. Go to Gradio UI Tab 5
  2. Base tokenizer: models/pretrained/chatterbox_tokenizer.json (auto-filled)
  3. Amharic tokenizer: models/tokenizer/amharic_tokenizer/vocab.json
  4. Click 'Merge Tokenizers'
""")
elif not extended_model.exists():
    print("""
→ Extend Model Embeddings
  1. Go to Gradio UI Tab 5, Step 2
  2. Model path: models/pretrained/chatterbox/t3_mtl23ls_v2.safetensors
  3. Original size: 2454 (should be pre-filled)
  4. New size: (check merged tokenizer size)
  5. Click 'Extend Embeddings'
""")
else:
    print("""
→ START TRAINING! 🚀
  1. Go to Gradio UI Tab 6
  2. Configure settings
  3. Select dataset
  4. Click 'Start Training'
  
  OR run:
  cd /teamspace/studios/this_studio/amharic-tts
  git pull origin main
  # Then update training script and run
""")

print("\n" + "="*70)
