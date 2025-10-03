# 🚀 Lightning AI Deployment Checklist

## ✅ What You Need to Do

### Step 1: Prepare Repository (Do This Now on Windows)

```powershell
# Navigate to your project
cd C:\Users\Abrsh-1\Downloads\CHATTERBOX_STRUCTURED-AMHARIC\amharic-tts

# Check git status
git status

# Add all files
git add .

# Commit
git commit -m "Complete training system with Gradio UI for Lightning AI"

# Push to GitHub (create repo if needed)
git remote add origin https://github.com/YOUR_USERNAME/amharic-tts.git
git push -u origin main
```

**Important:** Make sure your dataset is NOT in the repo (it's in .gitignore).

---

### Step 2: Sign Up for Lightning AI

1. Go to: https://lightning.ai
2. Sign up (free)
3. Click "New Studio"
4. Select **GPU: T4** (free tier)
5. Name it: `amharic-tts-training`

---

### Step 3: On Lightning AI Terminal

```bash
# Clone your repo
git clone https://github.com/YOUR_USERNAME/amharic-tts.git
cd amharic-tts

# Run automated setup
chmod +x setup_lightning.sh
./setup_lightning.sh
```

**This will:**
- ✅ Install PyTorch with CUDA
- ✅ Install all dependencies  
- ✅ Create necessary directories
- ✅ Verify GPU availability

---

### Step 4: Upload Your Dataset

**Choose one method:**

**Method A: Direct Upload (if < 1GB)**
1. In Lightning AI: Files → Upload
2. Upload your dataset.zip
3. Extract:
```bash
cd data/srt_datasets
unzip ../../dataset.zip
```

**Method B: Google Drive (Recommended for larger datasets)**
1. Upload dataset to Google Drive
2. Share publicly and get link
3. In Lightning AI:
```bash
pip install gdown
gdown YOUR_FILE_ID -O dataset.zip
unzip dataset.zip -d data/srt_datasets/
```

---

### Step 5: Launch Training UI

```bash
# Quick start (checks everything + launches UI)
python lightning_quickstart.py
```

**OR manually:**
```bash
python gradio_app/full_training_app.py --share
```

This gives you a public URL like: `https://xxxxx.gradio.live`

---

### Step 6: Start Training (in Gradio UI)

1. Open the public URL
2. Go to **"Training Pipeline"** tab
3. Configure:
   - Dataset Path: `data/srt_datasets/YOUR_DATASET`
   - Batch Size: `24` (for T4 GPU)
   - Max Epochs: `100` (or more)
   - Freeze Embeddings: ✅ (preserve English)
   - Use AMP: ✅ (faster training)
4. Click **"🚀 Start Training"**
5. Monitor progress in real-time!

---

## 📊 What's Already Set Up

✅ **Training Script:** `src/training/train.py` - Complete training loop
✅ **Gradio UI:** `gradio_app/full_training_app.py` - Full web interface
✅ **Configuration:** `config/training_config.yaml` - Default settings
✅ **Setup Script:** `setup_lightning.sh` - Automated installation
✅ **Quick Start:** `lightning_quickstart.py` - One-command launch
✅ **Documentation:** Multiple guides (see below)

---

## 📚 Documentation Available

1. **LIGHTNING_AI_SETUP.md** - Comprehensive guide
2. **README_LIGHTNING.md** - Quick reference
3. **DEPLOYMENT_CHECKLIST.md** - This file
4. **README.md** - Main project documentation
5. **QUICKSTART.md** - Quick start guide

---

## 🎯 Training Features

### In the Gradio UI, you can:

✅ **Import SRT Datasets** - Upload video/audio with subtitles
✅ **Manage Datasets** - View, merge, validate
✅ **Train Tokenizer** - Custom Amharic tokenizer
✅ **Merge Tokenizers** - Combine with base Chatterbox
✅ **Extend Embeddings** - Support Amharic vocabulary
✅ **Configure Training** - All parameters via sliders/inputs
✅ **Start/Stop Training** - Full control
✅ **Monitor Progress** - Real-time status and logs
✅ **Resume from Checkpoint** - Dropdown selection
✅ **Download Models** - After training completes

---

## ⚙️ Key Configuration Parameters

### Dataset Settings
- **Dataset Path:** Where your prepared data is
- **Batch Size:** 24 for T4, 16 if OOM

### Training Hyperparameters
- **Learning Rate:** 2e-4 (default, good for fine-tuning)
- **Max Epochs:** 100-1000 (depends on dataset size)
- **Max Steps:** 500,000 (safety limit)
- **Save Interval:** 2000 (save every 2000 steps)
- **Eval Interval:** 1000 (validate every 1000 steps)

### Embedding Freezing (Critical for Amharic!)
- **Freeze Original Embeddings:** ✅ Enable
- **Freeze Until Index:** 704 (Chatterbox base vocab)

### Performance
- **Use Mixed Precision (AMP):** ✅ Enable (2x faster)

---

## 🔄 Typical Workflow

```
1. Clone repo on Lightning AI
   ↓
2. Run setup_lightning.sh
   ↓
3. Upload dataset
   ↓
4. Launch Gradio UI
   ↓
5. Configure in "Training Pipeline" tab
   ↓
6. Start training
   ↓
7. Monitor (can close browser, training continues)
   ↓
8. Check back periodically
   ↓
9. Download trained model
```

---

## 💾 After Training

### Download Your Model

**Option 1: Lightning AI UI**
```
Files → models/checkpoints/ → Download checkpoint_latest.pt
```

**Option 2: Package Everything**
```bash
tar -czf amharic_tts_model.tar.gz \
    models/checkpoints/checkpoint_latest.pt \
    models/tokenizer/ \
    config/training_config.yaml
```

**Option 3: Upload to HuggingFace**
```bash
pip install huggingface_hub
huggingface-cli login
# Then upload via Files UI or:
python -c "
from huggingface_hub import HfApi
HfApi().upload_folder(
    folder_path='models/checkpoints',
    repo_id='YOUR_USERNAME/amharic-tts',
    repo_type='model'
)
"
```

---

## 📈 Expected Training Time

**On Lightning AI T4 GPU:**
- 5 hours audio → 6-12 hours training
- 10 hours audio → 12-24 hours training  
- 20 hours audio → 24-48 hours training

**Monitor:**
- Loss should decrease steadily
- GPU usage should be 70-90%
- Checkpoints save every 2000 steps

---

## 🐛 Troubleshooting

### If Setup Fails
```bash
# Re-run setup
./setup_lightning.sh

# Check GPU
nvidia-smi

# Check Python
python --version
```

### If Training Stops
```bash
# Resume from checkpoint
python gradio_app/full_training_app.py --share
# Then select checkpoint_latest.pt in UI
```

### If Out of Memory
Reduce batch size in Gradio UI:
- Try 16 instead of 24
- Or enable gradient accumulation

### If Gradio Won't Start
```bash
# Try different port
python gradio_app/full_training_app.py --share --port 7861
```

---

## ✨ Key Features of This System

### 1. Comprehensive Gradio UI
- All-in-one interface
- No need to edit config files
- Real-time monitoring
- Easy checkpoint management

### 2. Advanced Audio Processing
- SRT-based dataset import
- Advanced audio splitter with VAD
- Amharic-specific optimizations
- Automatic validation

### 3. Smart Training
- Embedding freezing (preserves English)
- Mixed precision (2x faster)
- Automatic checkpointing
- Resume capability

### 4. Cloud-Ready
- Optimized for Lightning AI
- Public Gradio URLs
- Dataset upload helpers
- Automated setup

---

## 🎓 Pro Tips

1. **Save Often:** Set save_interval to 1000-2000 for Lightning AI
2. **Use Share Mode:** Access Gradio from anywhere
3. **Monitor GPU:** `watch -n 1 nvidia-smi`
4. **Test First:** Run 1 epoch to verify everything works
5. **Batch Size:** Start with 24, reduce if OOM

---

## 📞 Need Help?

**Check:**
1. Logs: `tail -f logs/training.log`
2. GPU: `nvidia-smi`
3. Dataset: `ls -la data/srt_datasets/`
4. Config: `cat config/training_config.yaml`

**Documentation:**
- LIGHTNING_AI_SETUP.md (detailed)
- README_LIGHTNING.md (quick reference)
- README.md (main docs)

---

## ✅ Final Checklist

Before starting training:

- [ ] Repo pushed to GitHub
- [ ] Lightning AI account created
- [ ] Studio created with T4 GPU
- [ ] Repository cloned on Lightning AI
- [ ] Setup script run successfully
- [ ] GPU verified (nvidia-smi shows T4)
- [ ] Dataset uploaded and extracted
- [ ] Gradio UI launched with --share
- [ ] Public URL accessible
- [ ] Training parameters configured
- [ ] Ready to click "Start Training"!

---

## 🚀 Quick Start Command

```bash
# One command to rule them all:
./setup_lightning.sh && python lightning_quickstart.py
```

---

**You're all set! Push to GitHub and start training on Lightning AI! 🎉**
