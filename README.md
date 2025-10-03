# የአማርኛ ጽሁፍ ወደ ንግግር | Amharic Text-to-Speech

A comprehensive Amharic TTS system built on Chatterbox, featuring custom G2P, tokenizer extension, and multilingual fine-tuning.

🗣️ **Status:** Development/Training Phase
📚 **Language:** Amharic (አማርኛ) + English
🎯 **Goal:** High-quality Amharic speech synthesis

---

## 📋 Table of Contents

- [Features](#-features)
- [Quick Start](#-quick-start)
- [Installation](#-installation)
- [Data Preparation](#-data-preparation)
- [Training Pipeline](#-training-pipeline)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Technical Details](#-technical-details)
- [Troubleshooting](#-troubleshooting)

---

## ✨ Features

- **Native Amharic G2P**: Converts Ge'ez/Ethiopic script to phonemes
- **Extended Tokenizer**: Merges Amharic tokens with Chatterbox base (704 → 2000 tokens)
- **Embedding Freezing**: Preserves English tokens while training Amharic
- **Clean Gradio UI**: User-friendly interface with Amharic font support
- **LJSpeech Format**: Standard format for easy integration
- **Multilingual**: Supports both Amharic and English

---

## 🚀 Quick Start

```powershell
# 1. Clone and setup
git clone <your-repo>
cd amharic-tts
.\setup.ps1

# 2. Activate environment
.\venv\Scripts\Activate.ps1

# 3. Test G2P and tokenizer
python -m src.g2p.amharic_g2p

# 4. Launch web interface (demo mode)
python gradio_app/app.py
```

---

## 📦 Installation

### Requirements

- Windows 10/11
- Python 3.10 or later
- CUDA-capable GPU (for training)
- 8GB+ RAM
- 10GB+ disk space

### Automated Setup (Recommended)

```powershell
# Run the setup script
.\setup.ps1
```

### Manual Setup

```powershell
# Create virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install PyTorch (CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
pip install -r requirements.txt
```

---

## 📁 Data Preparation

### Step 1: Collect Amharic Audio Data

Prepare your dataset in one of these formats:

**Format A: Simple (filename|text)**
```
audio001.wav|ሰላም ለዓለም
audio002.wav|አዲስ አበባ የኢትዮጵያ ዋና ከተማ ናት
```

**Format B: LJSpeech**
```
wavs/
  amh_000001.wav
  amh_000002.wav
metadata.csv
```

### Step 2: Preprocess Audio

```powershell
python src/data_processing/preprocess_audio.py `
  --audio-dir "path/to/your/audio" `
  --transcript "path/to/transcripts.txt" `
  --output "data/processed/ljspeech_format"
```

**Recommendations:**
- ✅ 10+ hours of audio (minimum)
- ✅ 22050 Hz sample rate
- ✅ 2-15 seconds per clip
- ✅ Clean audio (minimal noise)
- ✅ Single speaker (for best results)

---

## 🎓 Training Pipeline

### Overview

```
1. Train Amharic Tokenizer
2. Merge with Base Tokenizer
3. Extend Model Embeddings
4. Fine-tune with Frozen Embeddings
5. Evaluate and Deploy
```

### Step 1: Train Amharic Tokenizer

```powershell
python -c "
from src.tokenizer.amharic_tokenizer import train_amharic_tokenizer
train_amharic_tokenizer(
    data_file='data/processed/ljspeech_format/metadata.csv',
    output_dir='models/tokenizer',
    vocab_size=500
)
"
```

**Output:**
- `models/tokenizer/sentencepiece.model`
- `models/tokenizer/vocab.json`

### Step 2: Merge Tokenizers

⚠️ **Important:** You need the base Chatterbox tokenizer first!

```powershell
# Download base Chatterbox tokenizer (if not available)
# Place it in models/pretrained/chatterbox_tokenizer.json

python scripts/merge_tokenizers.py `
  --base "models/pretrained/chatterbox_tokenizer.json" `
  --amharic "models/tokenizer/vocab.json" `
  --output "models/tokenizer/merged_vocab.json" `
  --validate
```

**What this does:**
- Loads base English tokens (0-703)
- Adds Amharic tokens (704+)
- Ensures no duplicates
- Creates merged vocabulary

### Step 3: Extend Model Embeddings

```powershell
# Download base Chatterbox model first
# Place it in models/pretrained/chatterbox_base.pt

python scripts/extend_model_embeddings.py `
  --model "models/pretrained/chatterbox_base.pt" `
  --output "models/pretrained/chatterbox_extended.pt" `
  --original-size 704 `
  --new-size 2000
```

**What this does:**
- Extends text embedding table from 704 → 2000
- Preserves original 704 embeddings
- Randomly initializes new embeddings

### Step 4: Configure Training

Edit `config/training_config.yaml`:

```yaml
model:
  n_vocab: 2000  # Match your merged vocab size
  freeze_original_embeddings: true
  freeze_until_index: 704  # Freeze English tokens

data:
  dataset_path: "data/processed/ljspeech_format"

finetuning:
  pretrained_model: "models/pretrained/chatterbox_extended.pt"
```

### Step 5: Train the Model

```powershell
# Training script (integrate with Chatterbox training)
# Follow Chatterbox documentation for training
# Make sure to use the training utilities:

python your_training_script.py --config config/training_config.yaml
```

**Key Points:**
- Use `freeze_text_embeddings()` from `src/training/train_utils.py`
- Monitor both English and Amharic validation samples
- Save checkpoints every 5000 steps

---

## 🎯 Usage

### Web Interface

```powershell
# Launch Gradio app
python gradio_app/app.py --port 7860

# With trained model
python gradio_app/app.py `
  --model "models/checkpoints/best.pt" `
  --config "config/training_config.yaml"

# Create public link
python gradio_app/app.py --share
```

Visit: `http://localhost:7860`

### Python API

```python
from src.g2p.amharic_g2p import AmharicG2P
from src.tokenizer.amharic_tokenizer import AmharicTokenizer

# Initialize
g2p = AmharicG2P()
tokenizer = AmharicTokenizer.load("models/tokenizer", g2p=g2p)

# Convert text to phonemes
text = "ሰላም ለዓለም"
phonemes = g2p.grapheme_to_phoneme(text)
print(f"Phonemes: {phonemes}")

# Tokenize
tokens = tokenizer.encode(text, use_phonemes=True)
print(f"Tokens: {tokens}")
```

---

## 📂 Project Structure

```
amharic-tts/
├── config/
│   └── training_config.yaml       # Training configuration
├── data/
│   ├── raw/                        # Raw audio and transcripts
│   ├── processed/                  # Preprocessed LJSpeech format
│   └── metadata/                   # Dataset metadata
├── src/
│   ├── data_processing/
│   │   └── preprocess_audio.py    # Audio preprocessing
│   ├── g2p/
│   │   └── amharic_g2p.py         # Amharic G2P converter
│   ├── tokenizer/
│   │   └── amharic_tokenizer.py   # Amharic tokenizer
│   ├── training/
│   │   └── train_utils.py         # Training utilities
│   └── inference/
├── scripts/
│   ├── merge_tokenizers.py        # Merge tokenizers
│   └── extend_model_embeddings.py # Extend model
├── gradio_app/
│   └── app.py                      # Web interface
├── models/
│   ├── tokenizer/                  # Trained tokenizers
│   ├── checkpoints/                # Training checkpoints
│   └── pretrained/                 # Pretrained models
├── logs/                           # Training logs
├── requirements.txt                # Dependencies
├── setup.ps1                       # Setup script
└── README.md                       # This file
```

---

## 🔬 Technical Details

### Amharic G2P

The G2P system handles the Ge'ez/Ethiopic script:

- **33 base consonants** × **7 vowel orders** = 231+ characters
- Converts to IPA phonemes
- Handles gemination, palatalization, assimilation

### Tokenizer Architecture

**Base Chatterbox:**
- 704 tokens (English)
- BPE (Byte Pair Encoding)

**Extended:**
- 2000 tokens (English + Amharic)
- Indices 0-703: English (frozen during training)
- Indices 704+: Amharic (trainable)

### Training Strategy

Based on practical multilingual training experience:

1. **Freeze English embeddings** to preserve learned representations
2. **Train only Amharic embeddings** (704+)
3. **Use English data occasionally** to prevent forgetting (optional)
4. **Monitor both languages** during validation

---

## 🐛 Troubleshooting

### Common Issues

**1. Import Errors**

```powershell
# Make sure virtual environment is activated
.\venv\Scripts\Activate.ps1

# Reinstall dependencies
pip install -r requirements.txt
```

**2. CUDA Out of Memory**

- Reduce `batch_size` in config
- Use smaller audio clips
- Enable gradient accumulation

**3. Tokenizer Not Found**

```powershell
# Train tokenizer first
python -m src.tokenizer.amharic_tokenizer
```

**4. Model Loading Fails**

- Check file paths in config
- Ensure model was extended correctly
- Verify vocab sizes match

**5. Poor Quality Output**

- Need more training data (10+ hours minimum)
- Increase training epochs
- Adjust learning rate
- Check audio preprocessing quality

### Getting Help

- Check logs in `logs/` directory
- Review training configuration
- Ensure all paths are correct
- Validate data preprocessing

---

## 📚 Resources

### Chatterbox TTS
- Repository: https://github.com/Diakonrobel/chatterbox-finetune
- Paper: [Link to paper if available]

### Amharic Resources
- Amharic Wikipedia: https://am.wikipedia.org
- Common Voice Amharic: https://commonvoice.mozilla.org

### Tools
- Epitran (G2P): https://github.com/dmort27/epitran
- Gradio: https://gradio.app
- PyTorch: https://pytorch.org

---

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- Additional Amharic datasets
- Improved G2P rules
- Multi-speaker support
- Voice cloning capabilities
- Better preprocessing pipelines

---

## 📄 License

[Specify your license here]

---

## 🙏 Acknowledgments

- **Chatterbox TTS** for the base architecture
- **Epitran** for G2P foundation
- **Gradio** for the UI framework
- Video tutorial on multilingual training

---

## 📞 Contact

[Your contact information]

---

**Made with ❤️ for the Amharic language community**

የአማርኛ ቋንቋ ማህበረሰብ ለመገልገል በፍቅር የተሰራ
