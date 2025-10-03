# Practical Amharic TTS Fine-tuning Guide

Following the approach from: https://github.com/stlohrey/chatterbox-finetuning

## 🎯 Goal

Fine-tune Chatterbox multilingual TTS model for high-quality Amharic speech synthesis.

---

## 📋 Prerequisites

### What You Already Have:
- ✅ Extended Chatterbox model (2454 → 3000 tokens)
- ✅ Chatterbox pretrained weights (`models/pretrained/chatterbox/t3_mtl23ls_v2.safetensors`)
- ✅ Extended model (`models/pretrained/chatterbox_extended.pt`)
- ✅ Training infrastructure (verified with dummy model)
- ✅ Dataset import tools (SRT → training data)

### What You Need:
- 📦 Amharic dataset (minimum 30 minutes, recommended 3-5 hours)
- 💻 GPU with at least 8GB VRAM
- ⏱️ Time: 2-3 days for quality training

---

## 🔄 Complete Workflow

### Phase 1: Data Preparation (Already Done!)

```bash
# You've already done this via Gradio UI!
# 1. Import SRT + Media → data/srt_datasets/your_dataset/
# 2. Format: metadata.csv + wavs/*.wav
```

**Your datasets are ready at:** `data/srt_datasets/`

### Phase 2: Tokenizer Setup (Already Done!)

```bash
# You've already:
# 1. Trained Amharic tokenizer ✅
# 2. Extended Chatterbox embeddings ✅
# 3. Saved extended model ✅
```

### Phase 3: Fine-tuning (Next Step!)

This is where we are now - ready to fine-tune!

---

## 🚀 Fine-tuning Process

### Approach 1: Using Our Simplified Model (Current)

Since we may not have the full Chatterbox package, we'll use a **practical hybrid approach**:

1. **Load extended embeddings** from `chatterbox_extended.pt`
2. **Use simplified T3 architecture** (already implemented)
3. **Train with real audio data** (next step)
4. **Transfer learn** on Amharic dataset

### Approach 2: Using Full Chatterbox Package (If Available)

```python
# If Chatterbox package is installed:
from chatterbox import Chatterbox, ChatterboxConfig

config = ChatterboxConfig.from_pretrained("multilingual")
model = Chatterbox.from_pretrained("multilingual", config=config)
model.train(train_data="data/srt_datasets/your_dataset")
```

---

## 📁 Expected Dataset Structure

```
data/srt_datasets/my_amharic_dataset/
├── metadata.csv          # Format: filename|text|speaker
├── wavs/
│   ├── segment_001.wav   # 22050 Hz, mono
│   ├── segment_002.wav
│   └── ...
└── stats.json            # Auto-generated statistics
```

**Your datasets already follow this format!** ✅

---

## 🎓 Training Strategy for Amharic

### Stage 1: Warm-up (Epochs 1-50)
- **Freeze:** Original embeddings (indices 0-2453)
- **Train:** Only new Amharic embeddings (indices 2454-2999)
- **Learning rate:** 1e-4
- **Goal:** Let Amharic tokens learn representations

### Stage 2: Fine-tuning (Epochs 51-200)
- **Unfreeze:** All embeddings
- **Learning rate:** 5e-5 (lower!)
- **Goal:** Adapt full model to Amharic

### Stage 3: Polish (Epochs 201+)
- **Unfreeze:** Everything
- **Learning rate:** 1e-5 (very low)
- **Goal:** Final quality improvements

---

## 🔧 Configuration for Amharic

### Key Settings:

```yaml
model:
  vocab_size: 3000                    # Extended for Amharic
  original_vocab_size: 2454           # Chatterbox base
  freeze_original_embeddings: true    # Stage 1
  freeze_until_index: 2454

data:
  sampling_rate: 22050
  n_mels: 80
  batch_size: 16                      # Adjust based on GPU

training:
  learning_rate: 1e-4                 # Start conservative
  max_epochs: 200
  save_interval: 1000
  eval_interval: 500
```

---

## 📊 Expected Training Metrics

### Good Training Progress:

```
Epoch 1   | Mel Loss: 8.2341 | Duration Loss: 2.1234
Epoch 10  | Mel Loss: 5.6789 | Duration Loss: 1.4567
Epoch 50  | Mel Loss: 2.3456 | Duration Loss: 0.8901
Epoch 100 | Mel Loss: 1.2345 | Duration Loss: 0.4567
Epoch 200 | Mel Loss: 0.8765 | Duration Loss: 0.2345
```

**Signs of good training:**
- ✅ Loss decreases steadily
- ✅ Mel loss < 2.0 by epoch 50
- ✅ Validation loss tracks training loss
- ✅ Generated samples sound clear

**Warning signs:**
- ⚠️ Loss increases (learning rate too high)
- ⚠️ Loss stuck (learning rate too low)
- ⚠️ Validation >> Training (overfitting)
- ⚠️ NaN values (gradient explosion)

---

## 🎤 Testing Synthesis

After training, test on Amharic text:

```python
# Via Gradio UI (Tab 1)
text = "ሰላም ለዓለም"
# Click "Generate Speech"
# Listen to output

# Via Python
from your_model import synthesize
audio = synthesize("አዲስ አበባ የኢትዮጵያ ዋና ከተማ ናት")
```

---

## 🗂️ Directory Structure After Training

```
amharic-tts/
├── models/
│   ├── pretrained/
│   │   ├── chatterbox/
│   │   │   └── t3_mtl23ls_v2.safetensors   # Original
│   │   └── chatterbox_extended.pt          # Extended
│   ├── checkpoints/
│   │   ├── checkpoint_epoch50_step8500.pt  # Warm-up done
│   │   ├── checkpoint_epoch100_step17000.pt
│   │   ├── checkpoint_epoch200_step34000.pt # Final
│   │   └── best_model.pt                   # Best validation
│   └── tokenizer/
│       ├── amharic_tokenizer/              # Your tokenizer
│       └── merged/                         # Merged vocab
├── data/
│   └── srt_datasets/
│       └── my_amharic_dataset/             # Your data
└── logs/
    └── 20251003_HHMMSS/                    # TensorBoard logs
```

---

## 🐛 Common Issues and Solutions

### Issue 1: Out of Memory (OOM)

**Solution:**
```yaml
# Reduce batch size
batch_size: 8  # or 4

# Enable gradient accumulation
grad_accumulation_steps: 2

# Use mixed precision
use_amp: true
```

### Issue 2: Loss Not Decreasing

**Solutions:**
1. Check dataset quality (clean audio, correct transcriptions)
2. Verify tokenizer works on Amharic text
3. Lower learning rate: `5e-5` or `1e-5`
4. Check for data loading errors in logs

### Issue 3: Poor Audio Quality

**Solutions:**
1. Train longer (more epochs)
2. Use more training data (5+ hours)
3. Check input audio quality (22050 Hz, mono)
4. Verify mel-spectrogram extraction

### Issue 4: Model Diverges (NaN loss)

**Solutions:**
```yaml
# Lower learning rate drastically
learning_rate: 1e-5

# Reduce gradient clipping threshold
grad_clip_thresh: 0.5

# Check for corrupt audio files
```

---

## 📈 Monitoring Training

### Via Gradio UI:
- Tab 6: Real-time progress
- See loss curves
- Check sample outputs

### Via TensorBoard:
```bash
tensorboard --logdir logs
# Open http://localhost:6006
```

### Via Logs:
```bash
tail -f logs/training.log
```

---

## ✅ Quality Checklist

Before considering training complete:

- [ ] Mel loss < 1.0
- [ ] Duration loss < 0.5
- [ ] Validation loss similar to training loss
- [ ] Generated Amharic sounds natural
- [ ] Pronunciation is correct
- [ ] Prosody (rhythm/intonation) sounds good
- [ ] No artifacts or glitches
- [ ] Works on unseen Amharic text

---

## 🎯 Next Steps

### Immediate (Ready to run):
1. ✅ Verify dataset is ready
2. ✅ Verify extended model exists
3. ✅ Update training script to use real model
4. ✅ Start training!

### After Initial Training:
1. Evaluate quality
2. Fine-tune hyperparameters
3. Add more data if needed
4. Train longer for production quality

---

## 📚 References

- **Chatterbox Official:** https://github.com/resemble-ai/chatterbox
- **Fine-tuning Example:** https://github.com/stlohrey/chatterbox-finetuning
- **Our Implementation:** See `scripts/finetune_chatterbox_amharic.py`

---

## 🎊 Ready to Start!

Everything is prepared. The next step is to run the fine-tuning script with real audio data.

**Command:**
```bash
cd /teamspace/studios/this_studio/amharic-tts
bash scripts/install_and_test_chatterbox.sh  # Verify setup
python scripts/finetune_chatterbox_amharic.py  # Start training!
```

Good luck! 🚀🇪🇹
