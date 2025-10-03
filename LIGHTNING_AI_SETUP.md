# Lightning AI Setup Guide for Amharic TTS Training

Complete guide to train your Amharic TTS model on Lightning AI's free GPU.

---

## 🚀 Quick Start

### 1. Prepare Your Repository

**Before pushing to GitHub:**

```bash
# Add .gitignore to exclude large files
# Already created - see .gitignore in repo

# Commit your changes
git add .
git commit -m "Prepared for Lightning AI training"
git push origin main
```

### 2. Sign Up for Lightning AI

1. Go to https://lightning.ai
2. Sign up for a free account
3. You get free GPU credits each month
4. Create a new Studio

---

## 📦 What's Included

Your repository is already set up with:

✅ **Training Script:** `src/training/train.py`
✅ **Configuration:** `config/training_config.yaml`
✅ **Requirements:** `requirements.txt`
✅ **Setup Scripts:** `setup_lightning.sh` (for Linux/cloud)
✅ **Gradio UI:** `gradio_app/full_training_app.py`

---

## 🔧 Lightning AI Setup Steps

### Step 1: Create a Studio

1. Click **"New Studio"** in Lightning AI
2. Choose **"Start from Scratch"** or **"Import from GitHub"**
3. Select GPU: **T4** (free tier) or **A10G** (if available)
4. Name: `amharic-tts-training`

### Step 2: Clone Your Repository

In the Lightning AI terminal:

```bash
# Clone your repo
git clone https://github.com/YOUR_USERNAME/amharic-tts.git
cd amharic-tts

# Run setup script
chmod +x setup_lightning.sh
./setup_lightning.sh
```

### Step 3: Upload Your Dataset

**Option A: Via Lightning AI UI**
1. Click **"Files"** → **"Upload"**
2. Upload your prepared dataset to `data/srt_datasets/`

**Option B: Via CLI (if you have dataset on cloud)**
```bash
# Example: Download from Google Drive
pip install gdown
gdown YOUR_DATASET_LINK -O data/dataset.zip
unzip data/dataset.zip -d data/srt_datasets/
```

**Option C: Via S3/Cloud Storage**
```bash
# AWS S3
aws s3 cp s3://your-bucket/dataset.zip data/
unzip data/dataset.zip -d data/srt_datasets/

# Google Cloud Storage
gsutil cp gs://your-bucket/dataset.zip data/
unzip data/dataset.zip -d data/srt_datasets/
```

---

## 🎯 Training on Lightning AI

### Method 1: Using Gradio UI (Recommended)

```bash
# Start the Gradio UI
python gradio_app/full_training_app.py --share

# This will give you a public URL like:
# Running on public URL: https://xxxxx.gradio.live
```

**Access the UI:**
- Open the public URL in your browser
- Configure training parameters
- Start training
- Monitor progress in real-time

### Method 2: Using Command Line

```bash
# 1. Prepare your dataset (if not done)
python examples/example_srt_import.py

# 2. Train tokenizer
python -c "
from src.tokenizer.amharic_tokenizer import train_amharic_tokenizer
train_amharic_tokenizer(
    data_file='data/srt_datasets/my_dataset/metadata.csv',
    output_dir='models/tokenizer',
    vocab_size=500
)
"

# 3. Start training
python src/training/train.py \
    --config config/training_config.yaml
```

---

## ⚙️ Configuration for Lightning AI

### Update `config/training_config.yaml`

Key settings for cloud training:

```yaml
data:
  dataset_path: "data/srt_datasets/your_dataset"
  batch_size: 32  # Increase for GPU
  num_workers: 4  # Lightning AI supports this

training:
  use_amp: true  # Enable mixed precision
  max_epochs: 100  # Reduce for testing
  save_interval: 1000  # Save frequently
  
  # GPU settings
  use_ddp: false  # Single GPU
  gpu_ids: [0]
```

---

## 📊 Monitoring Training

### Option 1: Gradio UI
- Real-time status updates
- Live logs
- Loss tracking
- Checkpoint management

### Option 2: TensorBoard

```bash
# Start TensorBoard
tensorboard --logdir logs --host 0.0.0.0 --port 6006

# Lightning AI will provide a public URL
```

### Option 3: View Logs

```bash
# Real-time logs
tail -f logs/training.log

# Or check in Gradio UI
```

---

## 💾 Saving and Downloading Models

### During Training
Checkpoints auto-save to: `models/checkpoints/`

### After Training

**Download via Lightning AI UI:**
1. Go to **Files** tab
2. Navigate to `models/checkpoints/`
3. Download `checkpoint_latest.pt` or best checkpoint

**Download via CLI:**
```bash
# Package the model
tar -czf amharic_tts_model.tar.gz models/checkpoints/ models/tokenizer/

# Download using Lightning CLI or web interface
```

---

## 🔄 Resume Training

If training stops, resume from checkpoint:

**Via Gradio UI:**
1. Go to "Training Pipeline" tab
2. Select checkpoint from dropdown
3. Click "Start Training"

**Via CLI:**
```bash
python src/training/train.py \
    --config config/training_config.yaml \
    --resume models/checkpoints/checkpoint_latest.pt
```

---

## 📝 Best Practices for Lightning AI

### 1. **Save Frequently**
```yaml
save_interval: 1000  # Every 1000 steps
```

### 2. **Use Mixed Precision**
```yaml
use_amp: true  # Faster training
```

### 3. **Monitor GPU Usage**
```bash
# Check GPU status
nvidia-smi

# Watch GPU usage
watch -n 1 nvidia-smi
```

### 4. **Optimize Batch Size**
Start small and increase:
- T4 GPU: batch_size = 16-32
- A10G GPU: batch_size = 32-64

### 5. **Regular Checkpoints**
Lightning AI sessions can timeout, so save often!

---

## ⚠️ Fixing Warnings in Your Logs

### 1. ✅ FIXED: Gradio Dropdown Warning

**Warning you saw:**
```
UserWarning: The value passed into gr.Dropdown() is not in the list of choices.
```

**Status:** ✅ **FIXED** - Code has been updated with `allow_custom_value=True`

### 2. FFmpeg Not Found

**Warning:**
```
⚠️  Warning: ffmpeg not found. Some features may be limited.
```

**Fix on Lightning AI:**
```bash
# Install via conda (recommended)
conda install -y -c conda-forge ffmpeg

# Verify installation
ffmpeg -version
```

### 3. Tokenizer Not Found

**Warning:**
```
⚠ Tokenizer not found
```

**This is EXPECTED!** You need to train it first:

1. Import dataset (Tab 2: Dataset Import)
2. Train tokenizer (Tab 4: Tokenizer Training)
   - Dataset Path: `data/srt_datasets/your_dataset/metadata.csv`
   - Vocabulary Size: 500-2000
   - Click "🚀 Train Tokenizer"

### 4. TTS Model Not Loaded

**Warning:**
```
⚠ TTS model not loaded (placeholder mode)
```

**This is EXPECTED!** The model needs training:

1. Complete dataset import + tokenizer training
2. Go to Tab 6: "🎓 Training Pipeline"
3. Configure and start training

### 5. Advanced Audio Splitter Not Available

**Warning:**
```
Warning: Advanced audio splitter not available. Using basic extraction.
```

**Fix:**
```bash
pip install --upgrade pydub soundfile librosa
conda install -y -c conda-forge ffmpeg
```

**Note:** Basic extraction still works, but advanced splitter provides better quality.

---

## 🐛 Other Troubleshooting

### Issue: Out of Memory
```yaml
# Reduce batch size
data:
  batch_size: 8  # or 16
  
# Enable gradient accumulation
training:
  grad_accumulation_steps: 2
```

### Issue: Session Timeout
- Lightning AI free tier has time limits
- Save checkpoints frequently
- Resume from last checkpoint

### Issue: Dataset Not Found
```bash
# Verify dataset path
ls -la data/srt_datasets/

# Check config
cat config/training_config.yaml | grep dataset_path
```

### Issue: CUDA Not Available
```python
# Check in Python
import torch
print(torch.cuda.is_available())
print(torch.cuda.device_count())
```

---

## 📚 Training Workflow Summary

```
1. Clone repo on Lightning AI
   ↓
2. Run setup_lightning.sh
   ↓
3. Upload dataset to data/srt_datasets/
   ↓
4. Start Gradio UI (python gradio_app/full_training_app.py --share)
   ↓
5. Configure training parameters in UI
   ↓
6. Start training
   ↓
7. Monitor progress in real-time
   ↓
8. Download trained model
```

---

## 🎓 Training Tips

### For Small Datasets (<5 hours)
```yaml
training:
  max_epochs: 500
  learning_rate: 2.0e-4
  batch_size: 16
```

### For Large Datasets (>10 hours)
```yaml
training:
  max_epochs: 200
  learning_rate: 1.0e-4
  batch_size: 32
```

### Fine-tuning Settings (Preserve English)
```yaml
model:
  freeze_original_embeddings: true
  freeze_until_index: 704  # Chatterbox base vocab
```

---

## 📦 Export Trained Model

After training completes:

```bash
# 1. Package everything
python scripts/export_model.py \
    --checkpoint models/checkpoints/checkpoint_best.pt \
    --output amharic_tts_final.zip

# 2. Download from Lightning AI
# Files → Download → amharic_tts_final.zip
```

---

## 🌐 Public Access (Optional)

Share your training progress:

```bash
# Enable public Gradio link
python gradio_app/full_training_app.py --share

# Share the https://xxxxx.gradio.live URL
```

---

## 📞 Support

**Lightning AI Documentation:**
- https://lightning.ai/docs

**This Project:**
- Check README.md
- Review QUICKSTART.md
- See examples/ directory

---

## ✅ Pre-Flight Checklist

Before starting training on Lightning AI:

- [ ] Dataset prepared and uploaded
- [ ] requirements.txt dependencies installed
- [ ] Config file updated with dataset path
- [ ] GPU is available (nvidia-smi)
- [ ] Enough disk space for checkpoints
- [ ] TensorBoard or Gradio UI running for monitoring

---

## 🚀 Launch Command

```bash
# Quick start command for Lightning AI
./setup_lightning.sh && \
python gradio_app/full_training_app.py --share --port 7860
```

**Then open the Gradio public URL and start training!**

---

Happy Training! 🎉
