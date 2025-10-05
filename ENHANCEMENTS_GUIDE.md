# Amharic TTS Enhancements Guide

## Overview

This document details the enhancements made to your Amharic TTS system based on best practices from the [chatterbox-finetune](https://github.com/alisson-anjos/chatterbox-finetune) reference implementation.

---

## 🎯 Key Enhancements

### 1. **Enhanced Training Script** (`train_enhanced.py`)

Complete production-ready training with:

- ✅ **Proper Checkpoint Loading**: Handles safetensors and regular PyTorch checkpoints
- ✅ **Learning Rate Warmup**: Linear warmup with gradual decay (from `transformers`)
- ✅ **Mixed Precision Training**: Automatic mixed precision with gradient scaling
- ✅ **Gradient Clipping**: Prevents exploding gradients
- ✅ **Early Stopping**: Prevents overfitting with configurable patience
- ✅ **TensorBoard Integration**: Real-time training monitoring
- ✅ **Embedding Freezing**: Preserves multilingual Chatterbox embeddings
- ✅ **Amharic G2P Integration**: Proper phoneme conversion for Amharic text

### 2. **Enhanced Dataset Processing**

**EnhancedAmharicDataset** Features:
- Proper G2P (Grapheme-to-Phoneme) integration
- Smart tokenization with phoneme support
- Robust error handling for corrupted audio
- Length clipping for stability
- LJSpeech format compatibility

### 3. **Enhanced Model Loading**

From `t3_model.py`:
```python
def load_pretrained_weights(self, checkpoint_path: str, strict: bool = False):
    """
    Loads FULL Chatterbox model (not just embeddings!)
    - Handles safetensors format
    - Strips 'module.' prefix from DataParallel
    - Extends embeddings if needed
    - Initializes new Amharic embeddings properly
    """
```

Key improvements:
- ✅ Loads ALL compatible weights (encoder, decoder, etc.)
- ✅ Proper embedding extension with smart initialization
- ✅ Handles both `.pt` and `.safetensors` formats
- ✅ Module prefix handling for multi-GPU trained models

---

## 📚 What You Learned from Reference Implementation

### From `chatterbox-finetune/train.py`:

1. **S3Token2Mel Architecture**
   - Uses Chatterbox's internal S3 speech tokenizer
   - Mel-spectrogram generation from speech tokens
   - Speaker encoder (CAMPPlus) for voice conditioning

2. **Data Collation Strategy**
   ```python
   # Reference implementation extracts:
   - S3 speech tokens (16kHz audio → discrete tokens)
   - Mel features (24kHz audio → spectrograms)
   - Speaker embeddings (from reference audio)
   - Proper padding and alignment
   ```

3. **Training Best Practices**
   - Very low learning rate (1e-5 to 5e-6) for finetuning
   - Gradient accumulation for effective larger batches
   - Warmup steps (1000+) for stability
   - Audio sampling every N steps for monitoring
   - Checkpoint saving with full state preservation

4. **Layer Freezing Approach**
   ```python
   # Freeze patterns like:
   --freeze_layers "tokenizer" "speaker_encoder" "flow.encoder"
   
   # Your equivalent:
   freeze_original_embeddings: true
   freeze_until_index: 2454  # Freeze multilingual Chatterbox tokens
   ```

---

## 🚀 Usage Guide

### Step 1: Prepare Your Dataset

Ensure your dataset is in LJSpeech format:
```
data/processed/my_amharic_dataset/
├── wavs/
│   ├── audio001.wav
│   ├── audio002.wav
│   └── ...
└── metadata.csv
```

**metadata.csv format:**
```
audio001|ሰላም ለዓለም|salam leʔalem
audio002|አዲስ አበባ|ʔadis ʔababa
```

### Step 2: Train Amharic Tokenizer

```bash
python -m src.tokenizer.amharic_tokenizer
```

Or programmatically:
```python
from src.tokenizer.amharic_tokenizer import train_amharic_tokenizer

train_amharic_tokenizer(
    data_file='data/processed/my_dataset/metadata.csv',
    output_dir='models/tokenizer',
    vocab_size=500  # Amharic-specific tokens
)
```

### Step 3: Configure Training

Edit `config/training_config.yaml`:

```yaml
model:
  n_vocab: 2535  # 2454 (Chatterbox) + 81 (Amharic)
  freeze_original_embeddings: true
  freeze_until_index: 2454  # Freeze Chatterbox multilingual tokens

data:
  dataset_path: "data/processed/my_amharic_dataset"
  batch_size: 16
  num_workers: 2

training:
  learning_rate: 1.0e-5  # CRITICAL: Very low for finetuning!
  max_epochs: 1000
  warmup_steps: 4000
  grad_clip_thresh: 0.5
  use_amp: true  # Mixed precision
  
  # Early stopping
  early_stopping: true
  patience: 50
  
finetuning:
  enabled: true
  pretrained_model: "models/pretrained/chatterbox/t3_mtl23ls_v2.safetensors"
```

### Step 4: Run Enhanced Training

```bash
# Basic training
python train_enhanced.py --config config/training_config.yaml

# Override device
python train_enhanced.py --config config/training_config.yaml --device cuda

# Resume from checkpoint
python train_enhanced.py --config config/training_config.yaml
```

Monitor training:
```bash
tensorboard --logdir logs/
```

---

## 🔑 Critical Hyperparameters

Based on reference implementation experience:

| Parameter | Value | Why |
|-----------|-------|-----|
| **Learning Rate** | 1e-5 to 5e-6 | Too high destroys pretrained weights |
| **Warmup Steps** | 4000+ | Stabilizes training from scratch |
| **Gradient Clipping** | 0.5 | Prevents spikes in loss |
| **Batch Size** | 16-32 (effective) | Use gradient accumulation if OOM |
| **Freeze Until Index** | 2454 | Preserves Chatterbox multilingual knowledge |
| **Max Audio Length** | 15 seconds | Prevents OOM, maintains quality |
| **Patience** | 50 epochs | Gives enough time for convergence |

---

## 📊 Training Monitoring

### What to Watch:

1. **Training Loss**: Should decrease steadily
   - If spiking: Reduce learning rate
   - If NaN: Check gradient clipping, reduce LR

2. **Validation Loss**: Should track training loss
   - Divergence = overfitting (early stopping will help)

3. **Learning Rate**: Should have smooth warmup + decay

4. **Gradient Norm**: Should be stable
   - Spikes indicate instability

### TensorBoard Metrics:

```
Loss/train          - Per-step training loss
Loss/train_epoch    - Average epoch training loss
Loss/val_epoch      - Validation loss
Learning_Rate       - Current LR (check warmup)
```

---

## 🎨 Architecture Comparison

### Your Simplified T3 Model:
```python
SimplifiedT3Model(
    text_embedding → pos_encoding → transformer_encoder → 
    mel_decoder + duration_predictor
)
```

### Full Chatterbox (Reference):
```python
T3(text_encoder) → S3Gen(speech_tokens → mel) → HiFiGAN(mel → audio)
```

**Key Difference:**
- Reference uses **S3 speech tokenizer** (discrete speech representation)
- Your model uses **direct mel prediction** (simpler, may have lower quality)

**Recommendation**: If quality is insufficient, consider:
1. Integrating S3Tokenizer from reference
2. Using Chatterbox's full pipeline (T3 + S3Gen + HiFiGAN)
3. Your current approach is fine for proof-of-concept and testing

---

## 🛠️ Troubleshooting

### Common Issues:

**1. CUDA Out of Memory**
```yaml
# Reduce batch size
batch_size: 8  # or lower

# Enable gradient accumulation
grad_accumulation_steps: 4  # Effective batch = 8 * 4 = 32
```

**2. Loss NaN/Inf**
```yaml
# More aggressive gradient clipping
grad_clip_thresh: 0.2

# Lower learning rate
learning_rate: 5.0e-6

# Reduce max audio length
max_audio_len_s: 10.0
```

**3. Poor Amharic Pronunciation**
```python
# Check G2P conversion
from src.g2p.amharic_g2p import AmharicG2P
g2p = AmharicG2P()

text = "ሰላም"
phonemes = g2p.grapheme_to_phoneme(text)
print(f"Phonemes: {phonemes}")  # Should be: "səlaːm" or similar
```

**4. Tokenizer Issues**
```bash
# Retrain with more data
python -m src.tokenizer.amharic_tokenizer

# Check vocabulary size
# Should have good coverage of Amharic characters
```

**5. Training Too Slow**
```yaml
# Increase num_workers (data loading parallelism)
num_workers: 4

# Use smaller validation set
max_eval_samples: 100
```

---

## 📈 Expected Training Timeline

For **10+ hours** of Amharic audio:

- **Initial**: Loss ~10-15 (random initialization)
- **After warmup (4k steps)**: Loss ~5-8
- **Convergence (50-100 epochs)**: Loss ~2-4
- **Good quality**: Val loss < 3.0

**Total Time**: 
- Small dataset (10h): 2-5 days on single GPU
- Medium dataset (50h): 1-2 weeks
- Large dataset (100h+): 2-4 weeks

---

## 🎯 Next Steps

1. **Test Current Setup**
   ```bash
   # Quick test with small batch
   python train_enhanced.py --config config/training_config.yaml
   ```

2. **Monitor First Few Epochs**
   - Check loss decreases
   - Verify no NaN/Inf
   - Confirm reasonable training speed

3. **Full Training**
   - Let it run for 50-100 epochs
   - Monitor validation loss
   - Save best checkpoint

4. **Inference Testing**
   ```python
   # Load best model
   from src.inference.inference import AmharicTTSInference
   
   tts = AmharicTTSInference(
       model_path='models/checkpoints/best_model.pt',
       config_path='config/training_config.yaml'
   )
   
   audio = tts.synthesize("ሰላም ለዓለም")
   ```

5. **Iterative Improvement**
   - Collect more diverse data
   - Fine-tune hyperparameters
   - Add data augmentation if needed

---

## 📚 Additional Resources

### Reference Implementations:
- [Chatterbox Finetune](https://github.com/alisson-anjos/chatterbox-finetune) - Portuguese TTS
- [Original Chatterbox](https://github.com/ResembleAI/Chatterbox) - Base model
- [Alternative Finetune Fork](https://github.com/stlohrey/chatterbox-finetuning) - Mentioned in README

### Papers & Documentation:
- Chatterbox paper (if available)
- Transformer TTS architectures
- G2P for low-resource languages

### Community:
- GitHub Issues on reference repos
- TTS forums and communities
- Ethiopic/Amharic NLP groups

---

## 🙏 Acknowledgments

Enhancements based on practical finetuning experience from:
- [alisson-anjos/chatterbox-finetune](https://github.com/alisson-anjos/chatterbox-finetune)
- Chatterbox team at ResembleAI
- Your excellent foundational work on Amharic G2P and dataset management!

---

## 📝 Summary of Files

### New/Enhanced Files:
```
amharic-tts/
├── train_enhanced.py                    # NEW: Production training script
├── ENHANCEMENTS_GUIDE.md               # NEW: This document
├── src/
│   ├── models/
│   │   └── t3_model.py                 # ENHANCED: Better pretrained loading
│   └── training/
│       └── train_utils.py              # ENHANCED: Added more utilities
└── config/
    └── training_config.yaml            # UPDATED: Optimal hyperparameters
```

---

## 🚦 Quick Start Checklist

- [ ] Dataset in LJSpeech format
- [ ] Train Amharic tokenizer
- [ ] Download Chatterbox pretrained model
- [ ] Update `config/training_config.yaml`
- [ ] Run: `python train_enhanced.py`
- [ ] Monitor: `tensorboard --logdir logs/`
- [ ] Wait for convergence (check validation loss)
- [ ] Test inference on Amharic text
- [ ] Iterate and improve!

---

**Good luck with your Amharic TTS training! 🎤🇪🇹**

For questions or issues, refer to the troubleshooting section or check the reference implementations.
