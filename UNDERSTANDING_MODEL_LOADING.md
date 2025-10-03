# 🧠 Understanding TTS Model Loading

## ✅ **This is NORMAL Behavior!**

When you see this message:
```
✓ TTS model not loaded (training mode - will load from checkpoints during training)
ℹ This is NORMAL on first run. Model will be available after training starts.
```

**This is the expected and correct behavior!** Let me explain why:

## 📁 Two Different Files, Two Different Purposes

### 1. **Extended Embeddings** (For Training)
```
models/pretrained/chatterbox_extended.pt
```
**Purpose:** Initialize training with pretrained weights  
**Used by:** Training script (`src/training/train.py`)  
**Used when:** You start training  
**Not used for:** Inference/demo in web interface

### 2. **Training Checkpoints** (For Inference)
```
models/checkpoints/checkpoint_latest.pt
models/checkpoints/checkpoint_epoch1_step5000.pt
```
**Purpose:** Save trained models for inference  
**Used by:** Web interface (`gradio_app/full_training_app.py`)  
**Used when:** You want to generate audio in the demo  
**Created when:** Training runs and saves checkpoints

## 🔄 The Complete Flow

### Stage 1: Fresh Install (What you see now)
```
✓ G2P loaded
✓ Tokenizer loaded  
✓ TTS model not loaded (training mode)
  ℹ This is NORMAL on first run
✓ SRT Dataset Builder loaded
```

**Status:** Ready to import data and train ✅  
**Can generate audio:** No (no trained model yet)  
**Next step:** Import dataset → Train model

### Stage 2: Training Started
```bash
python gradio_app/full_training_app.py --share
# In UI: Training Pipeline → Start Training
```

**What happens:**
1. Training script loads `models/pretrained/chatterbox_extended.pt` (if exists)
2. Uses those weights to initialize the model
3. Trains on your Amharic data
4. Saves checkpoints to `models/checkpoints/`

### Stage 3: After Training (Checkpoints Created)
```
✓ G2P loaded
✓ Tokenizer loaded
✓ TTS model loaded successfully  ← Now shows this!
✓ SRT Dataset Builder loaded
```

**Status:** Full system ready ✅  
**Can generate audio:** Yes!  
**Model loaded from:** `models/checkpoints/checkpoint_latest.pt`

## 🤔 Why Don't We Load Extended Embeddings in the Demo?

The extended embeddings file (`chatterbox_extended.pt`) contains:
- Partially initialized weights (English from Chatterbox + random Amharic)
- Not a fully trained model
- Designed for training initialization, not inference

If we loaded it in the demo:
- ❌ Would generate garbage audio (untrained Amharic embeddings)
- ❌ Confusing - users think it should work but audio is terrible
- ❌ Misleading - it's not actually a working TTS model yet

**Better approach:**
- ✅ Wait for actual trained checkpoints
- ✅ Clear messaging: "Train first, then generate audio"
- ✅ Only load models that can actually generate good audio

## 📊 File Purpose Summary

| File | Purpose | Loaded By | For |
|------|---------|-----------|-----|
| `chatterbox_extended.pt` | Training init | Training script | Weight initialization |
| `checkpoint_*.pt` | Trained model | Web interface | Audio generation |
| `tokenizer/` | Text → tokens | Both | Text processing |
| `metadata.csv` | Training data | Training script | Model learning |

## 🎯 What You Should Do

### Option A: Train Your Model (Recommended)
```bash
# 1. Launch interface
python gradio_app/full_training_app.py --share

# 2. In web UI:
#    - Import your SRT dataset
#    - Go to "Training Pipeline"
#    - Start training

# 3. Wait for checkpoints to be created
# 4. Model auto-loads from checkpoints!
```

### Option B: Just Test Text Processing
The current state lets you:
- ✅ Test G2P (text → phonemes)
- ✅ Test tokenization
- ✅ Import and manage datasets
- ✅ Configure training parameters

Audio generation will work once you train!

## 🔍 How to Check Status

### Check if extended embeddings exist (for training):
```bash
ls -la models/pretrained/chatterbox_extended.pt
```
If exists: Training can use pretrained weights ✅  
If not: Training will start from scratch ⚠️

### Check if trained checkpoints exist (for inference):
```bash
ls -la models/checkpoints/
```
If checkpoints exist: Demo can generate audio ✅  
If empty: Demo is in training mode (expected) ⚠️

## 💡 Key Takeaway

```
Extended Embeddings ≠ Trained Model
```

- **Extended embeddings** = Starting point for training
- **Trained checkpoint** = Actual working TTS model

The message you're seeing is **perfect and correct**. It's telling you:
> "I'm ready to train, but I don't have a trained model yet for generating audio."

Once you train and create checkpoints, the status will automatically change to:
> "✓ TTS model loaded successfully"

## 🚀 Next Steps

1. ✅ Pull latest code: `git pull origin main`
2. ✅ Status shows training mode: **Perfect!**
3. ✅ Import your Amharic dataset
4. ✅ Start training
5. ✅ Checkpoints auto-created
6. ✅ Model auto-loads
7. ✅ Generate Amharic speech!

The system is working exactly as designed! 🎉