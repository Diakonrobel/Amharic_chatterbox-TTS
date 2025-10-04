# Amharic TTS Inference Guide

Complete guide for using your trained/finetuned Amharic TTS models to generate speech.

## 📋 Table of Contents
1. [Quick Start](#quick-start)
2. [Command Line Usage](#command-line-usage)
3. [Python API Usage](#python-api-usage)
4. [Lightning AI Inference](#lightning-ai-inference)
5. [Troubleshooting](#troubleshooting)

---

## 🚀 Quick Start

### Prerequisites
- Trained Amharic TTS model checkpoint (`.pt` file)
- Required dependencies installed
- Merged tokenizer in `models/tokenizer/`

### Basic Inference (Command Line)

```bash
# Synthesize Amharic text
python src/inference/inference.py \
    --checkpoint models/checkpoints/checkpoint_best.pt \
    --text "ሰላም ለዓለም" \
    --output output.wav
```

---

## 💻 Command Line Usage

### Full Command Options

```bash
python src/inference/inference.py \
    --checkpoint <path_to_checkpoint> \
    --text "<amharic_text>" \
    --output <output_audio_path> \
    [--config <config_path>] \
    [--use-phonemes] \
    [--speed 1.0] \
    [--pitch 1.0] \
    [--device auto|cuda|cpu]
```

### Arguments

| Argument | Required | Description | Default |
|----------|----------|-------------|---------|
| `--checkpoint` | ✅ Yes | Path to trained model checkpoint | - |
| `--text` | ✅ Yes | Amharic text to synthesize | - |
| `--output` | No | Output audio file path | `output.wav` |
| `--config` | No | Training config file (auto-detected if not provided) | None |
| `--use-phonemes` | No | Use phoneme encoding instead of grapheme/character | False |
| `--speed` | No | Speed multiplier (0.5-2.0) | 1.0 |
| `--pitch` | No | Pitch multiplier (0.5-2.0) | 1.0 |
| `--device` | No | Device to use (auto/cuda/cpu) | auto |

### Examples

**Basic synthesis:**
```bash
python src/inference/inference.py \
    --checkpoint models/checkpoints/checkpoint_epoch_50.pt \
    --text "አማርኛ በጌዕዝ ፊደል ይጻፋል" \
    --output amharic_speech.wav
```

**With speed and pitch adjustment:**
```bash
python src/inference/inference.py \
    --checkpoint models/checkpoints/checkpoint_best.pt \
    --text "እንኳን ደህና መጡ" \
    --output welcome.wav \
    --speed 0.9 \
    --pitch 1.1
```

**Force CPU inference:**
```bash
python src/inference/inference.py \
    --checkpoint models/checkpoints/checkpoint_latest.pt \
    --text "ኢትዮጵያ በምስራቅ አፍሪካ የምትገኝ ሀገር ናት" \
    --output ethiopia.wav \
    --device cpu
```

**Use phoneme encoding (if tokenizer trained on phonemes):**
```bash
python src/inference/inference.py \
    --checkpoint models/checkpoints/checkpoint_best.pt \
    --text "ሰላም ለዓለም" \
    --output hello_world.wav \
    --use-phonemes
```

---

## 🐍 Python API Usage

### Basic Usage

```python
from src.inference import AmharicTTSInference

# Initialize inference engine
tts = AmharicTTSInference(
    checkpoint_path='models/checkpoints/checkpoint_best.pt',
    device='auto'  # or 'cuda' or 'cpu'
)

# Synthesize speech
audio, sample_rate, info = tts.synthesize(
    text="ሰላም ለዓለም",
    output_path="output.wav"
)

print(f"Generated audio: {info['audio_duration']:.2f} seconds")
```

### Advanced Usage

```python
from src.inference import AmharicTTSInference

# Initialize with custom config
tts = AmharicTTSInference(
    checkpoint_path='models/checkpoints/checkpoint_epoch_100.pt',
    config_path='config/training_config.yaml',
    device='cuda'
)

# Synthesize with adjustments
audio, sr, info = tts.synthesize(
    text="አዲስ አበባ የኢትዮጵያ ዋና ከተማ ናት",
    output_path="addis_ababa.wav",
    use_phonemes=False,  # Use grapheme/character encoding
    speed=1.0,
    pitch=1.0
)

# Print synthesis info
print("\n📊 Synthesis Information:")
for key, value in info.items():
    print(f"  {key}: {value}")
```

### Batch Synthesis

```python
from src.inference import AmharicTTSInference

# Initialize
tts = AmharicTTSInference(
    checkpoint_path='models/checkpoints/checkpoint_best.pt'
)

# List of Amharic texts
texts = [
    "ሰላም ለዓለም",
    "እንኳን ደህና መጡ",
    "አማርኛ በጌዕዝ ፊደል ይጻፋል",
    "ኢትዮጵያ በምስራቅ አፍሪካ የምትገኝ ሀገር ናት"
]

# Synthesize all
output_paths = tts.batch_synthesize(
    texts=texts,
    output_dir='outputs',
    use_phonemes=False
)

print(f"Generated {len(output_paths)} audio files")
```

### Get synthesis without saving

```python
# Synthesize without saving to file
audio, sr, info = tts.synthesize(
    text="ሰላም ለዓለም",
    output_path=None  # Don't save
)

# audio is a numpy array
print(f"Audio shape: {audio.shape}")
print(f"Sample rate: {sr}")

# You can now process or play the audio programmatically
```

---

## ☁️ Lightning AI Inference

### On Lightning AI Cloud

When running inference on Lightning AI:

1. **Download your trained checkpoint:**
```python
# In your Lightning AI notebook/terminal
!ls -lh models/checkpoints/
```

2. **Run inference:**
```bash
python src/inference/inference.py \
    --checkpoint models/checkpoints/checkpoint_best.pt \
    --text "ሰላም ለዓለም" \
    --output /teamspace/studios/this_studio/outputs/test.wav
```

3. **Download the generated audio:**
```python
# Use Lightning AI file browser or:
from IPython.display import Audio, display
display(Audio('/teamspace/studios/this_studio/outputs/test.wav'))
```

### Jupyter Notebook on Lightning AI

```python
import sys
sys.path.append('/teamspace/studios/this_studio/amharic-tts')

from src.inference import AmharicTTSInference
from IPython.display import Audio, display

# Initialize
tts = AmharicTTSInference(
    checkpoint_path='/teamspace/studios/this_studio/amharic-tts/models/checkpoints/checkpoint_best.pt',
    device='cuda'
)

# Synthesize
audio, sr, info = tts.synthesize(
    text="ሰላም ለዓለም",
    output_path="output.wav"
)

# Play audio in notebook
display(Audio(audio, rate=sr))

print(f"✅ Generated {info['audio_duration']:.2f}s of speech")
```

---

## 🔧 Troubleshooting

### Common Issues and Solutions

#### 1. **Error: "Checkpoint file not found"**

**Problem:** The checkpoint path is incorrect or file doesn't exist.

**Solutions:**
- Verify checkpoint exists:
  ```bash
  ls -lh models/checkpoints/
  ```
- Use absolute path:
  ```bash
  --checkpoint /full/path/to/checkpoint_best.pt
  ```
- Check Lightning AI paths (usually in `/teamspace/studios/this_studio/`)

---

#### 2. **Error: "Vocab size mismatch"**

**Problem:** Model was trained with different vocab size than tokenizer.

**Solutions:**
- Ensure you're using the same tokenizer used during training
- Check config file matches training config:
  ```bash
  cat config/training_config.yaml | grep n_vocab
  ```
- If using merged tokenizer, ensure it's in `models/tokenizer/Am_tokenizer_merged.json`

---

#### 3. **Error: "Tokenizer not found"**

**Problem:** Tokenizer files missing.

**Solutions:**
- Check tokenizer exists:
  ```bash
  ls -lh models/tokenizer/
  ```
- Copy tokenizer to inference environment:
  ```bash
  cp -r models/tokenizer/ /path/to/inference/models/
  ```
- The system will use fallback encoding if tokenizer is missing (may reduce quality)

---

#### 4. **Error: "CUDA out of memory"**

**Problem:** GPU doesn't have enough memory for inference.

**Solutions:**
- Use CPU instead:
  ```bash
  --device cpu
  ```
- Close other programs using GPU
- Use smaller batch sizes if doing batch synthesis

---

#### 5. **Error: "Config file not found"**

**Problem:** Config file missing or wrong path.

**Solutions:**
- System will use default config (usually works fine)
- Explicitly provide config:
  ```bash
  --config config/training_config.yaml
  ```
- Copy config from training:
  ```bash
  cp /path/to/training/config.yaml config/
  ```

---

#### 6. **Warning: "High UNK token ratio"**

**Problem:** Text contains characters not in tokenizer vocabulary.

**Solutions:**
- Use grapheme encoding (default):
  ```bash
  # Don't use --use-phonemes flag
  ```
- Check your tokenizer was trained on Amharic characters
- Verify text is properly encoded (UTF-8)

---

#### 7. **Poor Audio Quality**

**Problem:** Generated audio sounds robotic or distorted.

**Possible causes:**
- Model not trained enough (try later checkpoint)
- Wrong tokenizer encoding (phonemes vs graphemes)
- Dataset issues during training

**Solutions:**
- Try different checkpoints:
  ```bash
  --checkpoint models/checkpoints/checkpoint_best.pt
  ```
- Ensure using grapheme encoding (don't use `--use-phonemes`)
- Check training logs for convergence
- Verify dataset was properly preprocessed

---

#### 8. **Audio too fast/slow or pitch wrong**

**Problem:** Default synthesis doesn't match desired speech characteristics.

**Solutions:**
- Adjust speed:
  ```bash
  --speed 0.8  # Slower
  --speed 1.2  # Faster
  ```
- Adjust pitch:
  ```bash
  --pitch 0.9  # Lower pitch
  --pitch 1.1  # Higher pitch
  ```

---

### Debug Mode

For detailed debugging information:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

from src.inference import AmharicTTSInference

tts = AmharicTTSInference(
    checkpoint_path='models/checkpoints/checkpoint_best.pt',
    device='auto'
)
```

---

## 📊 Expected Output

When running inference, you should see output like:

```
🔧 Initializing Amharic TTS Inference...
📂 Checkpoint: models/checkpoints/checkpoint_best.pt
🖥️  Device: cuda
  Loaded config from: config/training_config.yaml
✓ Config loaded
✓ G2P initialized
✓ Audio processor initialized
✓ Tokenizer loaded (vocab size: 2535)
  Loaded checkpoint from epoch 50
  Training loss: 1.234
✓ Model loaded and ready for inference
✅ Initialization complete!

🎙️  Synthesizing: ሰላም ለዓለም
  Token IDs shape: torch.Size([1, 12])
  Generated mel shape: torch.Size([1, 80, 95])
  Audio shape: (24320,), duration: 1.01s
  ✓ Saved to: output.wav

📊 Synthesis Info:
  text: ሰላም ለዓለም
  text_length: 12
  token_count: 12
  mel_frames: 95
  audio_duration: 1.01
  sample_rate: 24000
  use_phonemes: False
  speed: 1.0
  pitch: 1.0

✅ Done! Audio saved to: output.wav
```

---

## 🎯 Best Practices

1. **Use grapheme encoding by default** (don't use `--use-phonemes` unless your tokenizer was specifically trained on phonemes)

2. **Use the best checkpoint** from training (usually `checkpoint_best.pt` or the one with lowest validation loss)

3. **Keep text length reasonable** (under 100 words per synthesis for best quality)

4. **Verify checkpoint matches tokenizer** used during training

5. **Test on Lightning AI first** before deploying elsewhere

6. **Use CUDA for faster inference** when available

---

## 📚 Additional Resources

- [Training Guide](TRAINING_GUIDE.md) - How to train/finetune models
- [Dataset Guide](DATASET_GUIDE.md) - Preparing Amharic datasets
- [Tokenizer Guide](TOKENIZER_GUIDE.md) - Understanding tokenization
- [Model Architecture](MODEL_ARCHITECTURE.md) - Understanding the model

---

## 🆘 Getting Help

If you encounter issues not covered here:

1. Check the error message carefully
2. Verify all file paths are correct
3. Ensure dependencies are installed
4. Check training logs for clues
5. Try with a smaller/simpler text first

For Lightning AI specific issues, check their documentation or support.

---

**Happy Synthesizing! 🎙️🇪🇹**
