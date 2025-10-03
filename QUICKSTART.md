# 🚀 Quick Start Guide

Get started with Amharic TTS in 5 minutes!

## Step 1: Setup (2 minutes)

```powershell
# Clone or navigate to project
cd amharic-tts

# Run automated setup
.\setup.ps1

# Activate environment
.\venv\Scripts\Activate.ps1
```

## Step 2: Test Components (1 minute)

```powershell
# Test G2P
python -c "from src.g2p.amharic_g2p import AmharicG2P; g=AmharicG2P(); print(g.grapheme_to_phoneme('ሰላም'))"

# Expected output: IPA phonemes
```

## Step 3: Launch UI (1 minute)

```powershell
# Start web interface
python gradio_app/app.py
```

Visit: `http://localhost:7860`

## Step 4: Test with Amharic Text

Try these examples in the UI:
- `ሰላም ለዓለም` (Hello World)
- `አዲስ አበባ` (Addis Ababa)
- `እንኳን ደህና መጡ` (Welcome)

---

## Next Steps for Training

### 1. Prepare Your Data

Place audio files and transcripts in:
```
data/raw/
  audio/
    file001.wav
    file002.wav
  transcripts.txt  (format: filename|text)
```

### 2. Preprocess

```powershell
python src/data_processing/preprocess_audio.py `
  --audio-dir "data/raw/audio" `
  --transcript "data/raw/transcripts.txt" `
  --output "data/processed/ljspeech_format"
```

### 3. Train Tokenizer

```powershell
python -c "from src.tokenizer.amharic_tokenizer import train_amharic_tokenizer; train_amharic_tokenizer('data/processed/ljspeech_format/metadata.csv', 'models/tokenizer', 500)"
```

### 4. Get Chatterbox Base Model

Download the base Chatterbox model and place in:
```
models/pretrained/chatterbox_base.pt
models/pretrained/chatterbox_tokenizer.json
```

### 5. Merge & Extend

```powershell
# Merge tokenizers
python scripts/merge_tokenizers.py `
  --base "models/pretrained/chatterbox_tokenizer.json" `
  --amharic "models/tokenizer/vocab.json" `
  --output "models/tokenizer/merged_vocab.json" `
  --validate

# Extend model
python scripts/extend_model_embeddings.py `
  --model "models/pretrained/chatterbox_base.pt" `
  --output "models/pretrained/chatterbox_extended.pt" `
  --original-size 704 `
  --new-size 2000
```

### 6. Train

Follow Chatterbox training documentation using:
- Extended model: `models/pretrained/chatterbox_extended.pt`
- Config: `config/training_config.yaml`
- Use `freeze_text_embeddings()` from `src/training/train_utils.py`

---

## Troubleshooting

**Virtual environment activation fails:**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**Dependencies fail to install:**
```powershell
pip install --upgrade pip
pip install -r requirements.txt --no-cache-dir
```

**CUDA not available:**
- Install CUDA 11.8 from NVIDIA
- Or use CPU-only (slower): `pip install torch torchvision torchaudio`

---

For detailed information, see [README.md](README.md)
