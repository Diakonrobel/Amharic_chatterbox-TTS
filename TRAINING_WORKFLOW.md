# Complete Amharic TTS Training Workflow ✅

## 🎉 Status: Model Extension Complete!

Your Chatterbox model has been successfully extended with Amharic embeddings!

- ✅ **Model inspected:** Found `text_emb.weight` key
- ✅ **Embeddings extended:** 2454 → 3000 tokens
- ✅ **Configuration updated:** Ready for training
- ✅ **New embeddings:** 546 Amharic token slots initialized

---

## 📋 Complete Training Pipeline

### Overview

```
Dataset → Tokenizer → Merge Vocab → Extend Model → Train → Evaluate
```

### Step-by-Step Workflow

#### ✅ Step 1: Dataset Preparation (DONE if you have data)

**Option A: Import from SRT + Video/Audio** (Recommended)
```bash
# Use Gradio UI - Tab 2: Dataset Import
# Or command line:
python scripts/import_srt_dataset.py \
  --srt path/to/subtitle.srt \
  --media path/to/video.mp4 \
  --output data/srt_datasets/my_dataset \
  --speaker speaker_01
```

**Option B: Use existing dataset**
Your dataset should be in format:
```
data/srt_datasets/my_dataset/
├── metadata.csv  (audio_file|text format)
└── wavs/
    ├── segment_001.wav
    ├── segment_002.wav
    └── ...
```

#### ✅ Step 2: Train Amharic Tokenizer

**Via Gradio UI (Easiest):**
1. Go to Tab 4: "🔤 Tokenizer Training"
2. Set dataset path: `data/srt_datasets/my_dataset/metadata.csv`
3. Set vocab size: `500-1000` (recommended for most datasets)
4. Click "🚀 Train Tokenizer"

**Via Command Line:**
```bash
python scripts/train_tokenizer.py \
  --dataset data/srt_datasets/my_dataset/metadata.csv \
  --output models/tokenizer/amharic_tokenizer \
  --vocab-size 500
```

**Output:** `models/tokenizer/amharic_tokenizer/vocab.json`

#### ✅ Step 3: Download Chatterbox Pretrained Model (DONE if already downloaded)

**Via Gradio UI:**
- Tab 5: "🔧 Model Setup" → Step 0: Download Chatterbox Model
- Select "Multilingual" (recommended)
- Click "📥 Download"

**Via Command Line:**
```bash
python scripts/download_chatterbox.py --model-type multilingual
```

**Output:** 
- `models/pretrained/chatterbox/t3_mtl23ls_v2.safetensors`
- `models/pretrained/chatterbox_tokenizer.json`

#### ✅ Step 4: Merge Tokenizers

**Via Gradio UI:**
- Tab 5: "🔧 Model Setup" → Step 1: Merge Tokenizers
- Base tokenizer: `models/pretrained/chatterbox_tokenizer.json` (auto-filled)
- Amharic tokenizer: `models/tokenizer/amharic_tokenizer/vocab.json`
- Output name: `merged`
- Click "🔗 Merge Tokenizers"

**Via Command Line:**
```bash
python scripts/merge_tokenizers.py \
  --base models/pretrained/chatterbox_tokenizer.json \
  --amharic models/tokenizer/amharic_tokenizer/vocab.json \
  --output models/tokenizer/merged
```

**Important:** Note the merged vocabulary size (e.g., 2954 tokens).

#### ✅ Step 5: Extend Model Embeddings (COMPLETED! ✅)

**You've already done this successfully!**

If you need to adjust the vocab size to match your merged tokenizer:
```bash
python scripts/extend_model_embeddings.py \
  --model models/pretrained/chatterbox/t3_mtl23ls_v2.safetensors \
  --output models/pretrained/chatterbox_extended.pt \
  --original-size 2454 \
  --new-size <YOUR_MERGED_VOCAB_SIZE>
```

**Current setting:** 3000 tokens (2454 original + 546 new)

#### 🚀 Step 6: Start Training

**Via Gradio UI (Recommended):**
1. Go to Tab 6: "🎓 Training Pipeline"
2. Configure:
   - Config file: `config/training_config.yaml` ✅ (already updated)
   - Dataset path: `data/srt_datasets/my_dataset`
   - Batch size: `16` (adjust based on GPU memory)
   - Learning rate: `2e-4`
   - Max epochs: `1000`
   - Freeze embeddings: ✅ (keeps multilingual capabilities)
3. Click "🚀 Start Training"

**Via Command Line:**
```bash
python scripts/train.py \
  --config config/training_config.yaml \
  --dataset data/srt_datasets/my_dataset
```

#### 📊 Step 7: Monitor Training

**Via Gradio UI:**
- Tab 6: Real-time progress tracking
- Loss curves and metrics
- Live log viewing
- Checkpoint status

**Via TensorBoard:**
```bash
tensorboard --logdir logs
```

---

## ⚙️ Current Configuration Summary

| Setting | Value |
|---------|-------|
| **Base Model** | Chatterbox Multilingual (23 languages) |
| **Original Vocab Size** | 2454 tokens |
| **Extended Vocab Size** | 3000 tokens |
| **New Amharic Tokens** | 546 slots |
| **Embedding Dimension** | 1024 |
| **Extended Model Path** | `models/pretrained/chatterbox_extended.pt` |
| **Freeze Original Embeddings** | ✅ Yes (preserves multilingual capabilities) |

---

## 🎯 Training Configuration (`config/training_config.yaml`)

The configuration has been updated with:

### Model Settings
```yaml
model:
  original_vocab_size: 2454  # Chatterbox multilingual base
  n_vocab: 3000              # Extended with Amharic
  freeze_original_embeddings: true
  freeze_until_index: 2454   # Freeze base tokens
```

### Fine-tuning Settings
```yaml
finetuning:
  enabled: true
  pretrained_model: "models/pretrained/chatterbox_extended.pt"  # Using extended model
```

### Training Hyperparameters (Defaults)
```yaml
training:
  learning_rate: 2e-4
  batch_size: 16
  max_epochs: 1000
  save_interval: 5000
  eval_interval: 1000
  use_amp: true  # Mixed precision for faster training
```

---

## 🔧 Adjusting Vocab Size (If Needed)

**If your merged tokenizer has a different vocab size:**

1. **Check merged tokenizer size:**
   ```bash
   python -c "import json; print(len(json.load(open('models/tokenizer/merged/vocab.json'))))"
   ```

2. **Re-extend model with correct size:**
   ```bash
   python scripts/extend_model_embeddings.py \
     --model models/pretrained/chatterbox/t3_mtl23ls_v2.safetensors \
     --output models/pretrained/chatterbox_extended.pt \
     --original-size 2454 \
     --new-size <MERGED_VOCAB_SIZE>
   ```

3. **Update config:**
   ```yaml
   model:
     n_vocab: <MERGED_VOCAB_SIZE>
   ```

---

## 📝 Important Notes

### 1. **Freezing Embeddings**
- **Recommended:** Keep `freeze_original_embeddings: true`
- **Why:** Preserves the pretrained multilingual knowledge
- **What gets trained:** Only the 546 new Amharic token embeddings + rest of model

### 2. **Dataset Size**
- **Minimum:** ~30 minutes of clean audio
- **Recommended:** 1-3 hours for good quality
- **Optimal:** 5+ hours for excellent quality

### 3. **GPU Memory**
Adjust batch size based on your GPU:
- **8GB GPU:** batch_size = 4-8
- **16GB GPU:** batch_size = 16
- **24GB GPU:** batch_size = 32
- **40GB+ GPU:** batch_size = 64

### 4. **Training Duration**
- **Quick test:** 10k-50k steps (~few hours)
- **Good quality:** 100k-200k steps (~1-2 days)
- **Production quality:** 300k-500k steps (~3-5 days)

---

## 🐛 Troubleshooting

### Issue: OOM (Out of Memory) Error
**Solution:** Reduce batch size in config or UI

### Issue: Loss not decreasing
**Solutions:**
1. Check dataset quality (clean audio, correct transcriptions)
2. Verify tokenizer is properly merged
3. Try lower learning rate: `1e-4`

### Issue: Audio quality poor
**Solutions:**
1. Train longer (more steps)
2. Use more training data
3. Check dataset audio quality

### Issue: Model not loading
**Solution:** Verify paths in `config/training_config.yaml`

---

## 🎉 You're Ready to Train!

### Quick Start Checklist

- ✅ Dataset prepared
- ✅ Tokenizer trained
- ✅ Chatterbox model downloaded
- ✅ Tokenizers merged
- ✅ **Model embeddings extended** ← YOU ARE HERE
- ⏳ Start training
- ⏳ Monitor and evaluate

**Next:** Go to the Gradio UI, Tab 6: "🎓 Training Pipeline", configure your settings, and click "🚀 Start Training"!

---

## 📚 Additional Resources

- **CHATTERBOX_SETUP_GUIDE.md** - Detailed Chatterbox integration guide
- **LIGHTNING_AI_SETUP.md** - Lightning AI deployment guide
- **TROUBLESHOOTING_EMBEDDINGS.md** - Embedding extension troubleshooting
- **README.md** - Main project documentation

Good luck with your training! 🚀🇪🇹
