# Training Configuration Files

This directory contains training configurations for fine-tuning the Chatterbox TTS model on Amharic data.

## 📋 Available Configurations

### 1. `training_config_finetune_FIXED.yaml` ⭐ **RECOMMENDED**
**Use this for production finetuning!**

This is the **FIXED** configuration that properly preserves multilingual capabilities while adding Amharic.

**Key Features:**
- ✅ **Freezes original embeddings** (tokens 0-2453)
- ✅ **Low learning rate** (1e-5) to prevent weight destruction
- ✅ **Proper gradient clipping** (0.5) for stability
- ✅ **Conservative optimizer settings**
- ✅ **Validates against all languages** (Amharic + English + French + German)

**When to use:**
- Starting fresh with extended model
- Fixing broken finetuning that destroyed pretrained languages
- Production deployments
- When you need reliable multilingual TTS

**Expected Results:**
- 🎯 Amharic: Clear speech after training
- ✅ English: Still works (preserved)
- ✅ French: Still works (preserved)
- ✅ German: Still works (preserved)

---

### 2. `training_config.yaml`
**Standard configuration** - Now enhanced with safer finetuning settings.

**Recent improvements:**
- Learning rate reduced to `1e-5` (from `2e-4`)
- Beta values changed to `[0.9, 0.999]` (more conservative)
- Gradient clipping tightened to `0.5` (from `1.0`)
- Early stopping patience increased to `50`

**When to use:**
- General training
- When you've already validated your setup
- Intermediate experiments

---

### 3. `training_config_stable.yaml`
**Optimized for small datasets** (<1000 samples)

**Special Features:**
- Smaller batch size (8)
- More frequent checkpoints (every 500 steps)
- More frequent evaluation (every 100 steps)
- Higher dropout (0.2) to prevent overfitting
- Lower max epochs (200) to prevent overfitting

**When to use:**
- Training on small Amharic datasets (<1 hour of audio)
- Limited compute resources
- Quick experiments
- When you see overfitting signs

---

## 🎯 Which Config Should I Use?

```
┌─────────────────────────────────────────────────────────────┐
│  DECISION TREE                                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Q: Is your English/French audio noisy after training?     │
│     ├─ YES → Use training_config_finetune_FIXED.yaml  ✅   │
│     └─ NO  → Continue...                                   │
│                                                             │
│  Q: Is this your first time training?                      │
│     ├─ YES → Use training_config_finetune_FIXED.yaml  ✅   │
│     └─ NO  → Continue...                                   │
│                                                             │
│  Q: Do you have <1000 audio samples?                       │
│     ├─ YES → Use training_config_stable.yaml           ✅   │
│     └─ NO  → Use training_config.yaml                  ✅   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 🚨 Critical Settings Explained

### **1. Embedding Freezing** 🔒
```yaml
freeze_original_embeddings: true
freeze_until_index: 2454
```

**What it does:**
- Freezes embeddings for indices 0-2453 (English, French, etc.)
- Allows training only embeddings 2454+ (Amharic)

**Why it matters:**
- ✅ Preserves pretrained knowledge
- ✅ Prevents catastrophic forgetting
- ✅ Keeps all 23 languages working

**Warning:** If you set this to `false`, you WILL destroy pretrained languages! 💀

---

### **2. Learning Rate** 📉
```yaml
learning_rate: 1.0e-5  # FIXED config
learning_rate: 5.0e-5  # STABLE config
```

**Comparison:**
```
Pretraining LR:  2e-4  → Train from scratch
Finetuning LR:   1e-5  → Preserve & adapt (FIXED)
Small data LR:   5e-5  → Fast adaptation (STABLE)
```

**Why low LR?**
- High LR = Big weight updates = Destroys pretrained knowledge ❌
- Low LR = Small updates = Preserves knowledge + learns new ✅

---

### **3. Vocabulary Sizes** 📚
```yaml
original_vocab_size: 2454  # Chatterbox (23 languages)
n_vocab: 2535              # Extended (23 langs + Amharic)
```

**Math:**
```
2454 (Chatterbox original)
+  81 (Amharic tokens)
─────
2535 (Total)
```

**Important:** Your merged tokenizer MUST have exactly this size!

---

### **4. Gradient Clipping** ✂️
```yaml
grad_clip_thresh: 0.5
```

**What it prevents:**
- Exploding gradients
- NaN losses
- Weight corruption
- Training instability

**Rule:** Lower = more stable, but slower learning

---

## 📊 Expected Training Behavior

### **Normal Training (with FIXED config):**
```
Epoch 1-10:    Loss ~10 → Basic patterns
Epoch 10-50:   Loss ~5  → Words forming
Epoch 50-100:  Loss ~3  → Clear words
Epoch 100-200: Loss ~2  → Natural speech
Epoch 200+:    Loss ~1  → High quality
```

**Checkpoints to test:**
- Test Amharic at epoch 50
- Test English/French at epoch 50 (should still work!)
- Continue if both work

---

### **Warning Signs:**

❌ **Loss increases:**
- Learning rate too high
- Bad data in batch
- Gradient explosion

❌ **Loss stays flat:**
- Learning rate too low
- Embeddings actually frozen (not just first 2454)
- Bad initialization

❌ **Loss decreases but audio is noise:**
- Wrong audio preprocessing
- Mel-spectrogram issues
- Vocoder problems

---

## 🔧 How to Use These Configs

### **Option 1: Gradio UI (Recommended)**
```python
# In Gradio app:
1. Go to "Training Pipeline" tab
2. Select config: training_config_finetune_FIXED.yaml
3. Verify settings:
   ✅ Freeze Original Embeddings: CHECKED
   ✅ Freeze Until Index: 2454
   ✅ Learning Rate: 0.00001
4. Click "Start Training"
```

---

### **Option 2: Command Line**
```bash
# Production finetuning (FIXED config)
python src/training/train.py \
  --config config/training_config_finetune_FIXED.yaml \
  --data data/srt_datasets/my_dataset

# Small dataset (STABLE config)
python src/training/train.py \
  --config config/training_config_stable.yaml \
  --data data/srt_datasets/my_small_dataset

# Standard training
python src/training/train.py \
  --config config/training_config.yaml \
  --data data/srt_datasets/my_dataset
```

---

## 🎓 Understanding the Settings

### **Batch Size & Gradient Accumulation**
```yaml
batch_size: 16
grad_accumulation_steps: 2
# Effective batch size = 16 × 2 = 32
```

**Why gradient accumulation?**
- Larger effective batch size without OOM
- More stable gradients
- Better training dynamics

---

### **Early Stopping**
```yaml
early_stopping: true
patience: 50
min_epochs: 30
min_delta: 0.001
```

**How it works:**
```
If validation loss doesn't improve by 0.001
for 50 consecutive evaluations (after epoch 30)
→ Stop training
→ Restore best checkpoint
```

---

## 📈 Monitoring Training

### **What to watch:**
1. **Training loss** should decrease steadily
2. **Validation loss** should decrease (but slower)
3. **Learning rate** should decay gradually
4. **Generated audio** should improve over time

### **TensorBoard:**
```bash
tensorboard --logdir logs
# Open: http://localhost:6006
```

---

## 🆘 Troubleshooting

### **Problem: All languages sound like noise**
**Cause:** `freeze_original_embeddings: false` or LR too high

**Fix:**
1. Use `training_config_finetune_FIXED.yaml`
2. Start from fresh extended model (not corrupted checkpoint)
3. Verify `freeze_original_embeddings: true`

---

### **Problem: Loss doesn't decrease**
**Possible causes:**
- Learning rate too low → Try 2e-5
- Dataset issues → Verify audio files
- Model not loaded → Check pretrained path

---

### **Problem: Training too slow**
**Solutions:**
- Increase batch size (if GPU allows)
- Increase learning rate slightly (max 2e-5)
- Use fewer evaluation steps

---

### **Problem: Overfitting (train loss low, val loss high)**
**Solutions:**
- Use `training_config_stable.yaml`
- Increase dropout
- Get more data
- Enable early stopping

---

## 📚 Additional Resources

- **[Fixing Broken Finetuning Guide](../docs/FIXING_BROKEN_FINETUNING.md)** - Comprehensive recovery guide
- **[Training Documentation](../docs/TRAINING.md)** - Detailed training guide
- **[Dataset Preparation](../docs/DATASET_PREPARATION.md)** - How to prepare your data

---

## ✅ Pre-Flight Checklist

Before starting training, verify:

- [ ] Extended model exists at `models/pretrained/chatterbox_extended.pt`
- [ ] Merged tokenizer exists at `models/tokenizer/Am_tokenizer_merged.json`
- [ ] Tokenizer vocab size is 2535
- [ ] Dataset has train/val splits (`metadata_train.csv`, `metadata_val.csv`)
- [ ] Config has `freeze_original_embeddings: true`
- [ ] Config has `freeze_until_index: 2454`
- [ ] Config has `learning_rate: 1e-5` (for production)
- [ ] Audio files are in `data/.../wavs/` directory
- [ ] GPU is available (check with `nvidia-smi`)

---

## 🎉 Success Criteria

Your finetuning is successful when:

✅ **Amharic works:**
```
Input:  "ሰላም ለዓለም"
Output: Clear Amharic speech 🎙️
```

✅ **English still works:**
```
Input:  "Hello world"
Output: Clear English speech 🎙️
```

✅ **French still works:**
```
Input:  "Bonjour le monde"
Output: Clear French speech 🎙️
```

🎊 **Congratulations! You've successfully added Amharic to a multilingual TTS model!**

---

**Last Updated:** 2025-01-04  
**Recommended Config:** `training_config_finetune_FIXED.yaml`
