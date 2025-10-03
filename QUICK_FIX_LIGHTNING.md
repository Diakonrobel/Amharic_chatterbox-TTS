# 🚀 Quick Fix for Lightning AI Warnings

## What You're Seeing

```
⚡ main ~/Amharic_chatterbox-TTS python gradio_app/full_training_app.py --share
Warning: Advanced audio splitter not available. Using basic extraction.
✓ G2P loaded
⚠ Tokenizer not found
⚠ TTS model not loaded (placeholder mode)
⚠️  Warning: ffmpeg not found. Some features may be limited.
✓ SRT Dataset Builder loaded
✓ Initialization complete
```

---

## ✅ What's Already Fixed

### Gradio Dropdown Warning
**Status:** ✅ **FIXED** in latest commit

Just pull the latest changes:
```bash
git pull origin main
```

---

## 🔧 What You Need to Fix (2 minutes)

### Step 1: Update Repository on Lightning AI

```bash
# Pull the fixes
cd ~/Amharic_chatterbox-TTS
git pull origin main
```

### Step 2: Install FFmpeg

```bash
# One command to fix ffmpeg warning
conda install -y -c conda-forge ffmpeg
```

### Step 3: Run Setup Script (Optional)

```bash
# This creates all necessary directories
bash lightning_setup.sh
```

---

## ✅ What's NORMAL (Not Errors!)

These warnings are **EXPECTED** until you train:

### ⚠ Tokenizer not found
**This is normal!** You need to:
1. Import a dataset first (Tab 2)
2. Train tokenizer (Tab 4)

### ⚠ TTS model not loaded
**This is normal!** You need to:
1. Complete tokenizer training
2. Start model training (Tab 6)

### ⚠ Advanced audio splitter not available
**This is optional!** Basic extraction works fine.

---

## 🎯 Complete Fix Commands

Run these on Lightning AI:

```bash
# Navigate to project
cd ~/Amharic_chatterbox-TTS

# Pull latest fixes
git pull origin main

# Install ffmpeg
conda install -y -c conda-forge ffmpeg

# Run setup script
bash lightning_setup.sh

# Restart the app
python gradio_app/full_training_app.py --share
```

---

## 📊 After Fixes - You Should See

```
⚡ main ~/Amharic_chatterbox-TTS python gradio_app/full_training_app.py --share

============================================================
LAUNCHING AMHARIC TTS TRAINING SYSTEM
============================================================

Initializing Amharic TTS Training System...
✓ G2P loaded
⚠ Tokenizer not found                    ← Train it in UI!
⚠ TTS model not loaded (placeholder mode) ← Train it in UI!
✓ SRT Dataset Builder loaded
✓ Initialization complete

Starting server on port 7860...
Share mode: Enabled

* Running on local URL:  http://127.0.0.1:7860
* Running on public URL: https://xxxxx.gradio.live
```

**No more:**
- ❌ Gradio dropdown warning
- ❌ ffmpeg warning
- ❌ Audio splitter warning

---

## 🎓 Training Workflow

Now you can start training:

```
1. Open the Gradio public URL
   ↓
2. Tab 2: Import Dataset
   - Upload SRT + video/audio
   ↓
3. Tab 4: Train Tokenizer
   - Dataset path: data/srt_datasets/your_dataset/metadata.csv
   - Vocab size: 500
   ↓
4. Tab 6: Start Training
   - Configure parameters
   - Click "Start Training"
```

---

## ⚡ TL;DR - Copy & Paste This

```bash
cd ~/Amharic_chatterbox-TTS && \
git pull origin main && \
conda install -y -c conda-forge ffmpeg && \
bash lightning_setup.sh && \
python gradio_app/full_training_app.py --share
```

**Done! All warnings fixed!** 🎉

---

## 📞 Need More Help?

- **Full guide:** See `LIGHTNING_AI_SETUP.md`
- **Training guide:** See `README.md`
- **Dataset guide:** See `SRT_DATASET_GUIDE.md`
