# 🎯 Chatterbox + Amharic TTS Setup Guide

Complete guide to download official Chatterbox models and merge with your Amharic tokenizer.

---

## 📋 Prerequisites

- ✅ Amharic tokenizer trained (you have this!)
- ✅ Lightning AI or local GPU
- ✅ Internet connection to download models

---

## 🚀 Step 1: Install Chatterbox Package

On Lightning AI terminal:

```bash
cd ~/Amharic_chatterbox-TTS

# Install official Chatterbox package
pip install chatterbox-tts

# Verify installation
python -c "import chatterbox; print(chatterbox.__version__)"
```

**Expected output:** Version number (e.g., `1.0.0`)

---

## 📥 Step 2: Download and Extract Chatterbox Models

### Option A: Download via Python API (Automatic)

```bash
python << 'EOF'
from chatterbox.tts import ChatterboxTTS
import torch
from pathlib import Path

print("Downloading Chatterbox pretrained model...")
print("This may take a few minutes (model is ~2-3 GB)...")

# This will auto-download the model
model = ChatterboxTTS.from_pretrained(device="cpu")  # Use CPU to save GPU memory

print("✓ Model downloaded successfully!")
print(f"Model config: {model.config}")

# The model files are cached in Hugging Face cache
# Typically at: ~/.cache/huggingface/hub/models--ResembleAI--chatterbox/
print("\nModel cached in Hugging Face hub directory")
EOF
```

### Option B: Download Files Manually

```bash
# Create directories
mkdir -p models/pretrained/chatterbox

# Download tokenizer
wget https://huggingface.co/ResembleAI/chatterbox/resolve/main/tokenizer.json \
  -O models/pretrained/chatterbox/tokenizer.json

# Download model weights (choose one)
# English model
wget https://huggingface.co/ResembleAI/chatterbox/resolve/main/t3_cfg.safetensors \
  -O models/pretrained/chatterbox/t3_cfg.safetensors

# OR Multilingual model (larger, 23 languages)
wget https://huggingface.co/ResembleAI/chatterbox/resolve/main/t3_mtl23ls_v2.safetensors \
  -O models/pretrained/chatterbox/t3_mtl23ls_v2.safetensors
```

---

## 🔍 Step 3: Extract Chatterbox Tokenizer Vocabulary

Create a script to extract the tokenizer vocab:

```bash
cat > scripts/extract_chatterbox_vocab.py << 'EOF'
"""
Extract vocabulary from Chatterbox tokenizer
"""
import json
from pathlib import Path
from huggingface_hub import hf_hub_download

def extract_chatterbox_vocab(output_path: str):
    """Download and extract Chatterbox tokenizer vocabulary"""
    
    print("="*60)
    print("EXTRACTING CHATTERBOX TOKENIZER")
    print("="*60)
    
    # Download tokenizer from HuggingFace
    print("\n[1/3] Downloading tokenizer from HuggingFace...")
    tokenizer_path = hf_hub_download(
        repo_id="ResembleAI/chatterbox",
        filename="tokenizer.json"
    )
    print(f"   ✓ Downloaded: {tokenizer_path}")
    
    # Load tokenizer
    print("\n[2/3] Loading tokenizer...")
    with open(tokenizer_path, 'r', encoding='utf-8') as f:
        tokenizer_data = json.load(f)
    
    # Extract vocabulary
    print("\n[3/3] Extracting vocabulary...")
    
    # Chatterbox uses different tokenizer formats
    # Try to extract vocab from different possible structures
    vocab = {}
    
    if 'model' in tokenizer_data and 'vocab' in tokenizer_data['model']:
        # Standard tokenizer format
        vocab = tokenizer_data['model']['vocab']
    elif 'vocab' in tokenizer_data:
        vocab = tokenizer_data['vocab']
    else:
        # Try to build from tokens
        print("   Building vocab from tokens...")
        if 'added_tokens' in tokenizer_data:
            for token_data in tokenizer_data['added_tokens']:
                vocab[token_data['content']] = token_data['id']
    
    print(f"   ✓ Extracted {len(vocab)} tokens")
    
    # Save vocabulary
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    
    print(f"\n   ✓ Saved to: {output_path}")
    
    # Show sample tokens
    print("\n   Sample tokens:")
    for i, (token, idx) in enumerate(list(vocab.items())[:10]):
        print(f"      {idx}: {repr(token)}")
    
    print("="*60)
    return vocab

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--output',
        default='models/pretrained/chatterbox_tokenizer.json',
        help='Output path for extracted vocabulary'
    )
    args = parser.parse_args()
    
    extract_chatterbox_vocab(args.output)
EOF

# Run the extraction
python scripts/extract_chatterbox_vocab.py \
  --output models/pretrained/chatterbox_tokenizer.json
```

---

## 🔗 Step 4: Merge Chatterbox + Amharic Tokenizers

Now merge the two tokenizers:

```bash
python scripts/merge_tokenizers.py \
  --base models/pretrained/chatterbox_tokenizer.json \
  --amharic models/tokenizer/amharic_tokenizer/vocab.json \
  --output models/tokenizer/merged_vocab.json \
  --base-size 704 \
  --validate
```

**Expected output:**
```
==================================================
TOKENIZER MERGING
==================================================

[1/5] Loading base tokenizer from: models/pretrained/chatterbox_tokenizer.json
      Base vocabulary size: 704

[2/5] Loading Amharic tokenizer from: models/tokenizer/amharic_tokenizer/vocab.json
      Amharic vocabulary size: 500

[3/5] Checking for overlapping tokens...
      ✓ No overlapping tokens

[4/5] Merging vocabularies...
      ✓ Added 500 new Amharic tokens
      ✓ Total merged vocabulary size: 1204

[5/5] Saving merged tokenizer to: models/tokenizer/merged_vocab.json
      ✓ Merged tokenizer saved

==================================================
MERGE SUMMARY
==================================================
  Base tokens:     704
  Amharic tokens:  500
  New tokens added: 500
  Total tokens:    1204
  Output:          models/tokenizer/merged_vocab.json
==================================================
```

---

## 🔧 Step 5: Extend Model Embeddings (Optional but Recommended)

Extend the Chatterbox model to support the new vocabulary size:

```bash
# First, locate the cached model
python << 'EOF'
from huggingface_hub import hf_hub_download
import shutil

# Download model to known location
print("Locating Chatterbox model...")
model_path = hf_hub_download(
    repo_id="ResembleAI/chatterbox",
    filename="t3_cfg.safetensors"  # or t3_mtl23ls_v2.safetensors for multilingual
)

# Copy to our pretrained directory
import shutil
shutil.copy(model_path, "models/pretrained/chatterbox_base.pt")
print(f"✓ Model copied to: models/pretrained/chatterbox_base.pt")
EOF

# Extend embeddings
python scripts/extend_model_embeddings.py \
  --model models/pretrained/chatterbox_base.pt \
  --output models/pretrained/chatterbox_extended.pt \
  --original-size 704 \
  --new-size 1204
```

---

## ⚙️ Step 6: Update Training Configuration

Edit your training config to use the merged tokenizer:

```bash
cat > config/training_config_finetuned.yaml << 'EOF'
# Amharic TTS Training Configuration (Fine-tuned from Chatterbox)

model:
  name: "chatterbox_amharic"
  architecture: "chatterbox"
  
  # Model dimensions (match Chatterbox)
  hidden_channels: 192
  filter_channels: 768
  n_heads: 2
  n_layers: 6
  kernel_size: 3
  p_dropout: 0.1
  
  # Vocabulary settings
  original_vocab_size: 704    # Chatterbox base
  n_vocab: 1204               # Merged: 704 + 500
  use_phonemes: true
  
  # Freeze English embeddings (preserve base model)
  freeze_original_embeddings: true
  freeze_until_index: 704

# Dataset configuration
data:
  dataset_path: "data/srt_datasets/YOUR_MERGED_DATASET"
  metadata_file: "metadata.csv"
  
  # Audio settings (match Chatterbox)
  sampling_rate: 22050
  filter_length: 1024
  hop_length: 256
  win_length: 1024
  n_mel_channels: 80
  mel_fmin: 0.0
  mel_fmax: 8000.0
  
  # Data splits
  train_ratio: 0.85
  val_ratio: 0.10
  test_ratio: 0.05
  
  # Data loading
  batch_size: 8
  num_workers: 2
  pin_memory: true

# Training hyperparameters
training:
  optimizer: "AdamW"
  learning_rate: 1.0e-4  # Lower for fine-tuning
  weight_decay: 0.01
  
  # Training duration
  max_epochs: 500
  max_steps: 50000
  
  # Checkpointing
  save_interval: 1000
  eval_interval: 500
  log_interval: 100
  
  # Gradient
  grad_clip_thresh: 1.0
  
  # Mixed precision
  use_amp: true
  
  # Single GPU
  use_ddp: false
  gpu_ids: [0]

# Fine-tuning from pretrained Chatterbox
finetuning:
  enabled: true
  pretrained_model: "models/pretrained/chatterbox_extended.pt"
  
  # Freezing strategy
  freeze_encoder: false
  freeze_decoder: false
  freeze_vocoder: true  # Keep vocoder frozen initially

# Logging
logging:
  use_wandb: false
  use_tensorboard: true
  log_dir: "logs"
  log_audio: true
  log_spectrograms: true

# Validation
validation:
  n_samples: 3
  sample_texts:
    - "ሰላም ለዓለም"
    - "አዲስ አበባ የኢትዮጵያ ዋና ከተማ ናት"
    - "እንኳን ደህና መጡ"

# Paths
paths:
  tokenizer: "models/tokenizer"
  merged_vocab: "models/tokenizer/merged_vocab.json"
  extended_model: "models/pretrained/chatterbox_extended.pt"
  checkpoints: "models/checkpoints"
EOF
```

---

## 🎯 Step 7: Test Before Training

Test that everything is set up correctly:

```bash
python << 'EOF'
import json
from pathlib import Path

print("="*60)
print("PRE-TRAINING CHECKLIST")
print("="*60)

checks = {
    "Amharic tokenizer": "models/tokenizer/amharic_tokenizer/vocab.json",
    "Chatterbox tokenizer": "models/pretrained/chatterbox_tokenizer.json",
    "Merged tokenizer": "models/tokenizer/merged_vocab.json",
    "Training config": "config/training_config_finetuned.yaml",
}

all_good = True
for name, path in checks.items():
    if Path(path).exists():
        print(f"✓ {name}: {path}")
        if path.endswith('.json'):
            with open(path, 'r') as f:
                data = json.load(f)
                if isinstance(data, dict):
                    print(f"  Size: {len(data)} tokens")
    else:
        print(f"✗ {name}: MISSING - {path}")
        all_good = False

print("="*60)
if all_good:
    print("✓ All checks passed! Ready to train.")
else:
    print("✗ Some files are missing. Please complete setup.")
print("="*60)
EOF
```

---

## 🚀 Step 8: Start Training!

### Via Gradio UI (Recommended):

1. Open Gradio: `python gradio_app/full_training_app.py --share`
2. Go to **Tab 6: Training Pipeline**
3. Configuration settings:
   - Config file: `config/training_config_finetuned.yaml`
   - Dataset path: `data/srt_datasets/YOUR_MERGED_DATASET`
   - Resume checkpoint: `None (Start from scratch)` or select if resuming
4. Click **"▶️ Start Training"**

### Via Command Line:

```bash
python src/training/train.py \
  --config config/training_config_finetuned.yaml \
  --resume models/pretrained/chatterbox_extended.pt
```

---

## 📊 Expected Training Progress

```
Epoch 1/500
Step 100: loss=2.34, duration=15h (est)
Step 200: loss=1.87
Step 500: loss=1.23 [Checkpoint saved]
...

Validation:
  Sample 1: ሰላም ለዓለም ✓
  Sample 2: አዲስ አበባ... ✓
  MOS: 3.8/5.0
```

---

## 🎉 Benefits of Using Pretrained Chatterbox

| Feature | From Scratch | With Chatterbox |
|---------|-------------|-----------------|
| **Training time** | 100+ hours | 20-50 hours |
| **Data needed** | 10+ hours | 3+ hours |
| **Voice quality** | Good | Excellent |
| **Multilingual** | No | Yes (English + Amharic) |
| **Zero-shot** | No | Yes (voice cloning) |

---

## 🐛 Troubleshooting

### Issue: "Can't download from HuggingFace"

```bash
# Set up HuggingFace token
huggingface-cli login

# Or download manually
wget https://huggingface.co/ResembleAI/chatterbox/resolve/main/tokenizer.json
```

### Issue: "Out of memory during training"

```yaml
# In config file, reduce batch size
data:
  batch_size: 4  # Or even 2
```

### Issue: "Model not loading"

Check Chatterbox version:
```bash
pip install --upgrade chatterbox-tts
```

---

## 📝 Summary

1. ✅ Install `chatterbox-tts` package
2. ✅ Download Chatterbox model (auto or manual)
3. ✅ Extract tokenizer vocabulary
4. ✅ Merge with your Amharic tokenizer
5. ✅ (Optional) Extend model embeddings
6. ✅ Update training config
7. ✅ Start training via Gradio UI

**You're now ready to train a high-quality Amharic TTS using Chatterbox!** 🚀

---

## 🔗 Resources

- **Chatterbox HuggingFace**: https://huggingface.co/ResembleAI/chatterbox
- **Chatterbox GitHub**: https://github.com/resemble-ai/Chatterbox
- **Your Project**: Current directory

Happy training! 🎉
