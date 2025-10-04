# Permanent Fixes for Common Issues

This document describes the permanent fixes implemented to prevent recurring issues with vocab size mismatches and configuration problems.

---

## 🔧 **New Tools Added:**

### **1. `scripts/auto_detect_vocab_size.py`**

**What it does:**
- Automatically detects vocab size from your tokenizer
- Updates ALL config files to match
- Prevents "size mismatch" errors

**When to use:**
- After creating/merging a new tokenizer
- Before starting training
- Before inference/testing

**How to run:**
```bash
cd /teamspace/studios/this_studio/Amharic_chatterbox-TTS
python scripts/auto_detect_vocab_size.py
```

**Output:**
```
═══════════════════════════════════════════════════════════════════════
🔍 AUTO-DETECTING VOCAB SIZE
═══════════════════════════════════════════════════════════════════════

Step 1: Finding tokenizer...
✓ Found tokenizer: models/tokenizer/am-merged_merged.json

Step 2: Reading tokenizer vocab size...
✓ Tokenizer vocab size: 2559

Step 3: Updating config files...
✓ training_config.yaml: Updated 2535 → 2559
✓ training_config_finetune_FIXED.yaml: Updated 2535 → 2559
✓ training_config_stable.yaml: Updated 2535 → 2559

═══════════════════════════════════════════════════════════════════════
✅ VOCAB SIZE SYNC COMPLETE
═══════════════════════════════════════════════════════════════════════
```

---

### **2. `start_gradio.sh` - Smart Startup Script**

**What it does:**
- Automatically runs vocab sync before starting Gradio
- Checks for required files (tokenizer, extended model, checkpoints)
- Provides clear status messages
- Supports --share mode

**How to use:**

```bash
# Local mode
./start_gradio.sh

# Public URL mode (--share)
./start_gradio.sh --share
```

**Features:**
- ✅ Auto-syncs vocab sizes
- ✅ Validates required files
- ✅ Clear pre-flight checks
- ✅ Handles errors gracefully

---

## 🚨 **Problems These Fix:**

### **Problem 1: Vocab Size Mismatch**
```
❌ Error: size mismatch for text_embedding.weight:
   copying a param with shape torch.Size([2559, 1024])
   from checkpoint, the shape in current model is
   torch.Size([2535, 1024])
```

**Root Cause:**
- Tokenizer has 2559 tokens
- Config files have 2535 tokens
- Model loads with 2535, checkpoint has 2559
- → MISMATCH!

**Fix:**
`auto_detect_vocab_size.py` automatically updates all configs to match tokenizer.

---

### **Problem 2: Wrong Learning Rate**
```
❌ Training uses LR: 0.000200 (should be 0.000010)
```

**Root Cause:**
- Gradio UI creates temp config with old default values
- Temp config overrides your FIXED config

**Fix:**
- Use configs directly via smart startup script
- Configs are now always in sync

---

### **Problem 3: Model Fails to Load**
```
⚠️ Warning loading weights: Error(s) in loading state_dict
Continuing with randomly initialized weights
```

**Root Cause:**
- Vocab size mismatch (see Problem 1)
- Model creates wrong size embeddings
- Checkpoint can't load

**Fix:**
Same as Problem 1 - vocab sync prevents this.

---

## 📋 **Usage Workflow:**

### **Standard Workflow (With Smart Startup):**

```bash
# 1. Pull latest code
git pull origin main

# 2. Start Gradio with auto-fix
./start_gradio.sh --share

# Done! Vocab sizes auto-synced
```

### **Manual Workflow (If Needed):**

```bash
# 1. Sync vocab sizes manually
python scripts/auto_detect_vocab_size.py

# 2. Start Gradio normally
python gradio_app/full_training_app.py --share
```

### **After Creating New Tokenizer:**

```bash
# Always run this after merging tokenizers
python scripts/auto_detect_vocab_size.py

# Then restart Gradio
./start_gradio.sh --share
```

---

## 🎯 **Checklist for Error-Free Operation:**

Before training/inference:

- [ ] Run `./start_gradio.sh` (auto-fixes everything) OR
- [ ] Run `python scripts/auto_detect_vocab_size.py` manually
- [ ] Verify all configs show same vocab size
- [ ] Check that extended model exists
- [ ] Verify tokenizer exists

After these checks:
- ✅ Training will use correct vocab size
- ✅ Inference will load checkpoints successfully
- ✅ No size mismatch errors

---

## 🔍 **How to Verify Vocab Sizes are Correct:**

### **Check Tokenizer:**
```bash
python -c "import json; print('Tokenizer:', len(json.load(open('models/tokenizer/am-merged_merged.json'))['vocab']))"
```

### **Check Config:**
```bash
grep 'n_vocab:' config/training_config_finetune_FIXED.yaml
```

### **Check Extended Model:**
```bash
python -c "import torch; ckpt=torch.load('models/pretrained/chatterbox_extended_2559.pt', map_location='cpu', weights_only=False); print('Model:', ckpt['model']['text_emb.weight'].shape[0])"
```

**All three should show the same number!**

---

## 🆘 **Troubleshooting:**

### **If auto_detect_vocab_size.py fails:**
```
❌ No tokenizer found!
```

**Solution:**
Create/merge tokenizer first using Gradio UI → Model Setup tab

---

### **If vocab sizes don't match after running script:**
```bash
# Force update all configs to 2559
sed -i 's/n_vocab: [0-9]*/n_vocab: 2559/' config/*.yaml

# Verify
grep 'n_vocab:' config/*.yaml
```

---

### **If Gradio still shows wrong vocab size:**
1. Stop Gradio (Ctrl+C)
2. Run `python scripts/auto_detect_vocab_size.py`
3. Restart Gradio with `./start_gradio.sh --share`
4. Test inference again

---

## 📚 **Related Documentation:**

- **[Lightning AI Fix Guide](LIGHTNING_AI_FIX_GUIDE.md)** - Complete troubleshooting
- **[Fixing Broken Finetuning](FIXING_BROKEN_FINETUNING.md)** - Recovery from corrupted training
- **[Config README](../config/README.md)** - Configuration explained

---

## ✅ **Success Indicators:**

After using these fixes, you should see:

```
✓ Tokenizer vocab size: 2559
✓ Config vocab size: 2559
✓ Model loads successfully
✓ No "size mismatch" errors
✓ Inference generates audio
✓ Training works with correct LR
```

---

## 🎉 **Benefits:**

- ✅ No more manual config editing
- ✅ Automatic error prevention
- ✅ Consistent vocab sizes across system
- ✅ Faster debugging
- ✅ Easier maintenance

---

**Last Updated:** 2025-01-04  
**Tools Version:** 1.0  
**Status:** Production-ready ✅
