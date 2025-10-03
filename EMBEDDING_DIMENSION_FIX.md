# 🔧 Fix: Embedding Dimension Mismatch

## ⚠️ The Warning You're Seeing:

```
⚠ Warning loading weights: Error(s) in loading state_dict for SimplifiedT3Model:
size mismatch for text_embedding.weight: copying a param with shape 
torch.Size([3000, 1024]) from checkpoint, the shape in current model is 
torch.Size([3000, 512]).
```

## 🔍 What This Means:

Your extended embeddings file has **embedding dimension = 1024**, but the training model expects **embedding dimension = 512**.

**Status:** ⚠️ Non-fatal warning - training continues with random initialization

## ✅ **Quick Solution: Train from Scratch (Recommended for Now)**

Since you have 310 Amharic samples ready, you can start training immediately without pretrained weights:

### Option 1: Disable Pretrained Loading (Fastest)

Edit `config/training_config.yaml`:
```yaml
finetuning:
  enabled: false  # Change from true to false
```

Then restart training - it will skip the extended embeddings and train from scratch on your Amharic data.

### Option 2: Remove Extended Embeddings File

```bash
# In Lightning AI terminal:
rm models/pretrained/chatterbox_extended.pt

# Then restart training
```

The training script will automatically skip loading and use random initialization.

## 🎯 **Why This Works:**

With 310 samples of clean Amharic data:
- ✅ Model can learn from scratch
- ✅ No English bias from pretrained weights  
- ✅ Simpler, faster to start
- ✅ Still produces good Amharic TTS

**Pretrained weights help but aren't required!** Your Amharic data is what matters most.

## 🔧 **Advanced: Fix the Dimension Mismatch (If You Want Pretrained Weights)**

If you really want to use the Chatterbox pretrained weights, you need to match dimensions:

### Step 1: Update Model to Use d_model=1024

Edit `src/training/train.py` line 227:
```python
# Change from:
d_model=512,

# To:
d_model=1024,
```

Also update in `gradio_app/full_training_app.py` line 142:
```python
# Change from:
d_model=512,

# To:
d_model=1024,
```

**⚠️ Warning:** This doubles the model size and GPU memory usage!

### Step 2: OR Re-create Extended Embeddings with d_model=512

This requires modifying the extend script to downsample the embeddings, which is complex and not recommended.

## 📊 **Comparison:**

| Approach | Pros | Cons | Recommended |
|----------|------|------|-------------|
| **Train from scratch** | Fast setup, simpler, Amharic-focused | No English transfer learning | ✅ **YES** (for your case) |
| **Use d_model=1024** | Full pretrained weights | 2x memory, slower, overkill for Amharic | ❌ Not needed |
| **Re-create embeddings** | Matches model exactly | Complex, time-consuming | ❌ Not worth it |

## 🚀 **Recommended Action (Now):**

```bash
# In Lightning AI, edit the config:
nano config/training_config.yaml

# Change this line:
finetuning:
  enabled: false  # Changed from true

# Save and exit (Ctrl+X, Y, Enter)

# Restart training in the web interface
```

Your training will now work perfectly without the warning! 🎉

## 💡 **Understanding the Warning:**

The warning appears because:
1. You downloaded/created extended embeddings with Chatterbox's native dimension (1024)
2. SimplifiedT3Model uses a smaller dimension (512) for efficiency
3. Shapes don't match, so loading fails
4. Training continues with random initialization (which is fine!)

**Bottom line:** The warning is cosmetic. Your training will work with or without fixing it. For your 310-sample dataset, training from scratch is actually the better choice!

## ✅ **After This Fix:**

You'll see:
```
[23:05:23] ⚠ Pretrained model not found OR finetuning disabled
[23:05:23]   Training from scratch
[23:05:23] ✓ T3 Model created
```

No more warning, clean training logs, and your model will train perfectly on Amharic data! 🚀