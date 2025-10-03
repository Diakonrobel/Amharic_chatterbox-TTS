# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

---

## Project Overview

**Amharic Text-to-Speech (TTS) System** - A production-ready multilingual TTS system for Amharic (አማርኛ) built on top of Chatterbox TTS. This project extends the base Chatterbox multilingual model (23 languages) with Amharic support through custom G2P, tokenizer extension, and embedding freezing techniques.

**Key Innovation:** Preserves the base model's multilingual capabilities (English, etc.) while training new Amharic tokens using selective embedding freezing - a proven approach from multilingual TTS training.

**Status:** Development/Training phase with complete infrastructure ready for training.

---

## Essential Commands

### Setup and Environment

```powershell
# Initial setup (creates venv, installs dependencies)
.\setup.ps1

# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Install dependencies manually
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

# Verify CUDA availability
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
```

### Testing Core Components

```powershell
# Test Amharic G2P (Grapheme-to-Phoneme)
python -m src.g2p.amharic_g2p

# Test G2P with custom text
python -c "from src.g2p.amharic_g2p import AmharicG2P; g=AmharicG2P(); print(g.grapheme_to_phoneme('ሰላም ለዓለም'))"

# Verify training setup
python verify_training_setup.py
```

### Web Interface

```powershell
# Launch Gradio web UI (demo mode)
python gradio_app/app.py

# Launch with specific port
python gradio_app/app.py --port 7860

# Launch with public sharing enabled
python gradio_app/app.py --share

# Full training interface with all features
python gradio_app/full_training_app.py
```

### Data Processing

```powershell
# Preprocess audio files to LJSpeech format
python src/data_processing/preprocess_audio.py `
  --audio-dir "data/raw/audio" `
  --transcript "data/raw/transcripts.txt" `
  --output "data/processed/ljspeech_format"

# Import SRT dataset (interactive)
python src/data_processing/dataset_manager.py

# Import single SRT with media file
python src/data_processing/srt_dataset_builder.py import `
  --srt "path/to/video.srt" `
  --media "path/to/video.mp4" `
  --name "dataset_name" `
  --speaker "speaker_01"

# Merge multiple datasets
python src/data_processing/srt_dataset_builder.py merge `
  --datasets dataset1 dataset2 dataset3 `
  --output merged_final
```

### Tokenizer Training

```powershell
# Train Amharic tokenizer
python -c "from src.tokenizer.amharic_tokenizer import train_amharic_tokenizer; train_amharic_tokenizer('data/processed/ljspeech_format/metadata.csv', 'models/tokenizer', 500)"

# Merge base Chatterbox tokenizer with Amharic tokenizer
python scripts/merge_tokenizers.py `
  --base "models/pretrained/chatterbox_tokenizer.json" `
  --amharic "models/tokenizer/vocab.json" `
  --output "models/tokenizer/merged_vocab.json" `
  --validate
```

### Model Extension

```powershell
# Extend Chatterbox model embeddings for Amharic tokens
python scripts/extend_model_embeddings.py `
  --model "models/pretrained/chatterbox_base.pt" `
  --output "models/pretrained/chatterbox_extended.pt" `
  --original-size 2454 `
  --new-size 3000

# Analyze Chatterbox model structure
python scripts/analyze_chatterbox_model.py --model "models/pretrained/chatterbox_base.pt"
```

### Training

```powershell
# Start training with config file
python src/training/train.py --config config/training_config.yaml

# Monitor training with TensorBoard
tensorboard --logdir logs
```

### Docker

```powershell
# Build and run with Docker Compose
docker-compose up --build

# Run in detached mode
docker-compose up -d

# View logs
docker-compose logs -f amharic-tts

# Stop containers
docker-compose down
```

### Code Quality

```powershell
# Run tests
pytest tests/

# Format code
black src/ scripts/ tests/

# Lint code
flake8 src/ scripts/
```

---

## Architecture Overview

### High-Level Training Pipeline

The complete workflow follows this sequence:

```
Raw Data → Audio Processing → Tokenizer Training → Vocab Merge → 
Model Extension → Embedding Freezing → Fine-tuning → Evaluation
```

### Critical Multi-stage Process

**1. Tokenizer Extension Strategy**
- Base Chatterbox uses 2454 tokens (multilingual: 23 languages)
- Amharic tokenizer trained separately (500-1000 tokens)
- Vocabularies merged with sequential indexing: base tokens (0-2453), new Amharic tokens (2454+)
- Final merged vocabulary typically ~3000 tokens
- **Key Implementation:** `scripts/merge_tokenizers.py` ensures no duplicate tokens and validates index continuity

**2. Model Embedding Extension**
- Chatterbox T3 model's text embedding layer extended from 2454 → 3000 tokens
- Original embeddings (0-2453) preserved exactly as-is
- New embeddings (2454-2999) randomly initialized
- **Key Implementation:** `scripts/extend_model_embeddings.py` handles safetensors and PyTorch checkpoint formats

**3. Selective Embedding Freezing (Most Critical)**
- During training, base language embeddings (0-2453) are frozen via gradient hooks
- Only new Amharic embeddings (2454+) receive gradient updates
- This preserves the model's existing multilingual capabilities (English, etc.)
- Prevents "catastrophic forgetting" of pre-trained languages
- **Key Implementation:** `src/training/train_utils.py::freeze_text_embeddings()` uses `register_hook()` to zero gradients for frozen indices
- Based on proven approach from Japanese-English multilingual training

### Component Architecture

**Core Modules:**

- **`src/g2p/amharic_g2p.py`** - Amharic Grapheme-to-Phoneme converter
  - Handles Ge'ez/Ethiopic script (33 consonants × 7 vowel orders = 231+ characters)
  - Maps to IPA phonemes with phonological rules (gemination, palatalization, assimilation)
  - Character-to-IPA mapping built dynamically for all fidel combinations
  
- **`src/tokenizer/amharic_tokenizer.py`** - BPE tokenizer for Amharic
  - Uses SentencePiece for subword tokenization
  - Phoneme-based tokenization option
  - Save/load functionality for training and inference

- **`src/training/train_utils.py`** - Training utilities
  - **`freeze_text_embeddings(model, freeze_until_index)`** - Critical function for selective training
  - Uses gradient hooks to mask gradients for indices < freeze_until_index
  - Also includes checkpoint management and parameter counting

- **`src/models/t3_model.py`** - Simplified T3 model wrapper for Chatterbox
  - Interfaces with Chatterbox's architecture
  - Handles audio processing and mel-spectrogram generation

- **`src/audio/audio_processing.py`** - Audio processing utilities
  - Loads, validates, and normalizes audio
  - Generates mel-spectrograms (22050 Hz, 80 mel channels)
  - Custom collate function for batching variable-length sequences

### Data Pipeline

**SRT Dataset Builder** (`src/data_processing/srt_dataset_builder.py`):
- Extracts audio from video using ffmpeg
- Splits audio based on SRT subtitle timestamps
- Validates segment quality (duration, silence, clipping)
- Outputs LJSpeech-compatible format: `metadata.csv` + `wavs/` directory

**Audio Preprocessing** (`src/data_processing/preprocess_audio.py`):
- Normalizes sample rate to 22050 Hz
- Validates audio quality (clipping detection, silence removal)
- Duration filtering (2-15 seconds per clip recommended)
- Batch processing with progress tracking

### Configuration System

**`config/training_config.yaml`** - Central training configuration:

```yaml
model:
  original_vocab_size: 2454  # Base Chatterbox multilingual
  n_vocab: 3000              # Extended with Amharic
  freeze_original_embeddings: true
  freeze_until_index: 2454   # Critical: freeze base tokens

finetuning:
  enabled: true
  pretrained_model: "models/pretrained/chatterbox_extended.pt"
  freeze_vocoder: true       # Keep vocoder frozen initially
```

**Key Parameters to Adjust:**
- `n_vocab` - Must match merged tokenizer size
- `freeze_until_index` - Must match original_vocab_size
- `batch_size` - Adjust based on GPU memory (16 default)
- `learning_rate` - 2e-4 recommended for fine-tuning

---

## Directory Structure

```
amharic-tts/
├── src/                          # Source code
│   ├── g2p/                      # Grapheme-to-Phoneme
│   │   └── amharic_g2p.py        # Amharic G2P converter
│   ├── tokenizer/                # Tokenization
│   │   └── amharic_tokenizer.py  # BPE tokenizer
│   ├── training/                 # Training infrastructure
│   │   ├── train.py              # Main training script
│   │   └── train_utils.py        # Embedding freezing utilities
│   ├── models/                   # Model definitions
│   │   └── t3_model.py           # T3 model wrapper
│   ├── audio/                    # Audio processing
│   │   └── audio_processing.py   # Mel-spectrogram generation
│   └── data_processing/          # Data preparation
│       ├── preprocess_audio.py   # Audio preprocessing
│       ├── srt_dataset_builder.py # SRT import
│       └── dataset_manager.py    # Interactive dataset manager
├── scripts/                      # Utility scripts
│   ├── merge_tokenizers.py       # Merge vocab files
│   ├── extend_model_embeddings.py # Extend embedding layer
│   └── analyze_chatterbox_model.py # Model inspection
├── gradio_app/                   # Web interfaces
│   ├── app.py                    # Simple demo UI
│   └── full_training_app.py      # Full training UI
├── config/                       # Configuration
│   └── training_config.yaml      # Training config
├── data/                         # Data directories
│   ├── raw/                      # Raw audio/transcripts
│   ├── processed/                # Preprocessed LJSpeech format
│   ├── srt_datasets/             # SRT-imported datasets
│   └── metadata/                 # Dataset metadata
├── models/                       # Model storage
│   ├── tokenizer/                # Trained tokenizers
│   ├── pretrained/               # Base Chatterbox models
│   └── checkpoints/              # Training checkpoints
├── logs/                         # Training logs
├── tests/                        # Test files
├── setup.ps1                     # Windows setup script
└── requirements.txt              # Python dependencies
```

**Important Paths:**
- Chatterbox base model: `models/pretrained/chatterbox/t3_mtl23ls_v2.safetensors`
- Extended model: `models/pretrained/chatterbox_extended.pt`
- Merged tokenizer: `models/tokenizer/merged_vocab.json`
- Training checkpoints: `models/checkpoints/checkpoint_step_*.pt`

---

## Development Workflow

### Complete Training Pipeline

1. **Data Preparation**
   - Collect Amharic audio with transcripts (10+ hours recommended)
   - Use SRT import for video sources: `python src/data_processing/dataset_manager.py`
   - Or preprocess audio manually: `python src/data_processing/preprocess_audio.py`
   - Output: LJSpeech format in `data/processed/` or `data/srt_datasets/`

2. **Tokenizer Training**
   - Train Amharic tokenizer: `train_amharic_tokenizer()` function
   - Vocab size: 500-1000 tokens recommended
   - Output: `models/tokenizer/vocab.json`

3. **Vocabulary Merging**
   - Merge base + Amharic tokenizers: `scripts/merge_tokenizers.py`
   - Note the final vocab size (e.g., 2954)
   - Validate: check for duplicate tokens and index continuity

4. **Model Extension**
   - Extend embeddings: `scripts/extend_model_embeddings.py`
   - Use vocab size from step 3
   - Update `config/training_config.yaml` with correct `n_vocab`

5. **Training**
   - Start training: `python src/training/train.py --config config/training_config.yaml`
   - Embedding freezing automatically applied if configured
   - Monitor with TensorBoard: `tensorboard --logdir logs`
   - Checkpoints saved every 5000 steps

6. **Validation**
   - Test on sample Amharic texts (defined in config)
   - Verify both Amharic AND English still work (multilingual preservation)
   - Evaluate audio quality and pronunciation

### Key Development Principles

**Embedding Freezing is Critical:**
- Always verify `freeze_original_embeddings: true` in config
- Check training logs for "FREEZING TEXT EMBEDDINGS" message
- Gradient hooks must be registered before training starts
- Without freezing, the model will forget English/other languages

**Dataset Quality Matters:**
- Audio: 22050 Hz, mono, clean (minimal noise)
- Duration: 2-15 seconds per clip (optimal for TTS)
- Transcripts: Accurate Amharic text in Ethiopic script
- Minimum 30 minutes for testing, 10+ hours for production

**Model Size Matching:**
- `merged_vocab.json` size MUST equal `n_vocab` in config
- Extended model embedding size MUST match `n_vocab`
- Mismatches cause "size mismatch" errors during training

### Common Development Tasks

**Adding New Audio Data:**
1. Place raw audio in `data/raw/`
2. Run preprocessing: `preprocess_audio.py`
3. Or use SRT import for video sources
4. Update dataset path in `config/training_config.yaml`

**Resuming Training:**
1. Find last checkpoint: `models/checkpoints/checkpoint_step_*.pt`
2. Update config with checkpoint path
3. Run training script - automatically resumes

**Testing G2P Changes:**
1. Edit `src/g2p/amharic_g2p.py`
2. Test: `python -m src.g2p.amharic_g2p`
3. Verify phoneme output for test words

**Adding Validation Texts:**
1. Edit `config/training_config.yaml`
2. Add texts to `validation.sample_texts`
3. These are synthesized during training for quality checks

---

## Important Technical Notes

### Chatterbox Integration

This project uses **Chatterbox T3 multilingual model** as the base:
- Pre-trained on 23 languages including English
- 2454-token vocabulary covering multiple scripts
- Uses T3 architecture (text encoder + vocoder)
- Model format: safetensors or PyTorch checkpoint

**Model Loading:**
- Safetensors support: `from safetensors.torch import load_file`
- PyTorch 2.6+ requires `weights_only=False` for trusted checkpoints
- Both formats handled by `extend_model_embeddings.py`

### Audio Processing Specifics

**Mel-Spectrogram Settings (fixed by Chatterbox):**
- Sample rate: 22050 Hz
- FFT size: 1024
- Hop length: 256
- Window length: 1024
- Mel channels: 80
- Frequency range: 0-8000 Hz

**Audio Validation Checks:**
- Duration: 1-20 seconds (configurable)
- Silence: Max 30% (configurable)
- Clipping: Max 0.1% samples (configurable)
- Sample rate: Must be 22050 Hz or resample automatically

### G2P Implementation Details

**Ethiopic Script Structure:**
- 33 base consonants (ሀ, ለ, ሐ, መ, ሠ, ረ, ሰ, ሸ, ቀ, ቐ, በ, ተ, ቸ, ኀ, ነ, ኘ, አ, ከ, ኸ, ወ, ዘ, ዠ, የ, ደ, ጀ, ገ, ጠ, ጨ, ጰ, ጸ, ፀ, ፈ, ፐ)
- 7 vowel orders per consonant: ə, u, i, a, e, ɨ, o
- Special characters: labialized, palatalized, geminated forms
- Total: 231+ distinct characters

**IPA Phoneme Set:**
- Consonants: p, b, t, d, k, g, ʔ, ts, dz, tʃ, dʒ, f, v, s, z, ʃ, ʒ, h, m, n, ɲ, ŋ, l, r, j, w
- Ejectives: p', t', ts', tʃ', k'
- Vowels: ə, a, i, u, e, o

**Phonological Rules Applied:**
- Gemination: double consonants marked with ː
- Palatalization: k/g + i → c
- Assimilation: nb → mb, nk → ŋk

### Training Hyperparameters

**Recommended Settings (from config):**
- Learning rate: 2e-4 (Adam optimizer)
- Batch size: 16 (adjust for GPU memory)
- Weight decay: 0.01
- Gradient clipping: 1.0
- Mixed precision: enabled (AMP)
- LR scheduler: ExponentialLR with decay 0.999875

**Training Duration:**
- Minimum: 50k steps (~5 hours on GPU)
- Recommended: 200k-500k steps
- Max: 500k steps (stop when validation loss plateaus)

**Checkpointing:**
- Save interval: 5000 steps
- Eval interval: 1000 steps
- Keep best 5 checkpoints based on validation loss

---

## Platform-Specific Notes

### Windows (Primary Development Platform)

**PowerShell Setup:**
```powershell
# Enable script execution
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Run setup
.\setup.ps1
```

**Path Handling:**
- All scripts use `pathlib.Path` for cross-platform compatibility
- Windows paths with backslashes work correctly
- Use raw strings for paths: `r"C:\path\to\file"`

**CUDA on Windows:**
- Requires NVIDIA GPU driver + CUDA Toolkit 11.8 or 12.1
- PyTorch CUDA install: `pip install torch --index-url https://download.pytorch.org/whl/cu118`

### Docker

**Environment Variables:**
- `GRADIO_SERVER_NAME=0.0.0.0` - Allow external access
- `GRADIO_SERVER_PORT=7860` - Web UI port
- `CUDA_VISIBLE_DEVICES=0` - GPU selection (GPU version)

**Volume Mounts:**
- `./data:/app/data` - Persist datasets
- `./models:/app/models` - Persist models
- `./logs:/app/logs` - Persist logs

### Cross-Platform Audio

**FFmpeg Required:**
- Windows: Install via Chocolatey or from ffmpeg.org
- Linux: `apt-get install ffmpeg`
- macOS: `brew install ffmpeg`

**Format Support:**
- Input: WAV, MP3, MP4, MKV, FLAC, OGG
- Output: WAV (22050 Hz, mono, 16-bit PCM)

---

## Troubleshooting

### Common Issues

**"Import Error: No module named 'src'"**
- Activate virtual environment: `.\venv\Scripts\Activate.ps1`
- Or install dependencies: `pip install -r requirements.txt`

**"CUDA out of memory"**
- Reduce `batch_size` in config (try 8 or 4)
- Enable gradient accumulation: `grad_accumulation_steps: 2`
- Use smaller audio clips (max 10 seconds)

**"Size mismatch in text embedding"**
- Check `n_vocab` in config matches merged tokenizer size
- Re-run `extend_model_embeddings.py` with correct size
- Verify with: `python scripts/analyze_chatterbox_model.py`

**"Embedding freezing not working"**
- Verify `freeze_original_embeddings: true` in config
- Check `freeze_until_index` matches original vocab size
- Look for "FREEZING TEXT EMBEDDINGS" in training logs

**"Poor Amharic quality but English still works"**
- Need more Amharic training data (aim for 10+ hours)
- Increase training steps (try 200k+)
- Check G2P output for phoneme accuracy

**"Model forgot English"**
- Embedding freezing not enabled - check config
- Re-extend model and retrain with freezing enabled
- Cannot recover - must start from base model again

**"ffmpeg not found"**
- Install ffmpeg for your platform
- Add to PATH environment variable
- Verify: `ffmpeg -version`

---

## Testing

### Running Tests

```powershell
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_g2p.py

# Run with verbose output
pytest -v tests/

# Run with coverage
pytest --cov=src tests/
```

**Note:** Test files not currently present in repository. To add tests:
- Create `tests/test_g2p.py` for G2P testing
- Create `tests/test_tokenizer.py` for tokenizer testing
- Use pytest fixtures for model loading

---

## Related Documentation

- **README.md** - Main project documentation and quick start
- **QUICKSTART.md** - 5-minute setup guide
- **PROJECT_SUMMARY.md** - Detailed project completion status
- **TRAINING_WORKFLOW.md** - Complete training pipeline guide
- **CHATTERBOX_FINETUNING_REFERENCE.md** - Chatterbox integration notes
- **SRT_DATASET_GUIDE.md** - SRT subtitle dataset import guide
- **DEPLOYMENT_CHECKLIST.md** - Production deployment guide

**Chatterbox Resources:**
- Official repo: https://github.com/resemble-ai/chatterbox
- HuggingFace: https://huggingface.co/ResembleAI/chatterbox
- Fine-tuning reference: https://github.com/stlohrey/chatterbox-finetuning

**Amharic Resources:**
- Common Voice Amharic: https://commonvoice.mozilla.org/am
- Amharic Wikipedia: https://am.wikipedia.org
