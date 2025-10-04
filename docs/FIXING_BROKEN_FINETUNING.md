# CRITICAL: Finetuning Broke Pretrained Model

## 🚨 Problem

You finetuned a multilingual Chatterbox model (23 languages) on Amharic, but now:
- ❌ **English sounds like noise**
- ❌ **French sounds like noise**  
- ❌ **Amharic sounds like noise**
- ❌ **ALL languages broken!**

**Root Cause:** The finetuning process **destroyed the original pretrained weights** instead of preserving them.

---

## 🔍 What Went Wrong

### **Critical Mistake: Embeddings NOT Frozen**

During finetuning, the first 2454 embeddings (English, French, etc.) should have been **FROZEN** (not updated), but they were accidentally **unfrozen** and got overwritten with random Amharic patterns.

**Result:** Original language knowledge lost → noise for all languages.

---

## 🎯 **IMMEDIATE FIX: Start Over Correctly**

### **Step 1: Diagnose the Damage**

Run this on Lightning AI:

```bash
cd /teamspace/studios/this_studio/Amharic_chatterbox-TTS
git pull origin main

python scripts/test_pretrained_english.py
```

This will show you:
- ✓ If pretrained model still exists
- ✓ How much your embeddings changed
- ✓ Whether they can be recovered

---

### **Step 2: Re-Download Pretrained Model**

```bash
# In Gradio UI:
1. Go to "Model Setup" tab
2. Click "Download Chatterbox Model"
3. Select "Multilingual"
4. Wait for download to complete
```

---

### **Step 3: Re-Extend Embeddings Correctly**

```bash
# In Gradio UI → Model Setup tab:

1. Merge Tokenizers section:
   - Base: models/pretrained/chatterbox_tokenizer.json
   - Amharic: models/tokenizer/amharic_tokenizer/vocab.json
   - Output: merged
   - Click "Merge Tokenizers"

2. Extend Embeddings section:
   - Base Model: models/pretrained/chatterbox/t3_mtl23ls_v2.safetensors
   - Original Vocab: 2454
   - New Vocab: 2535 (or your merged size)
   - Click "Extend Embeddings"
```

---

### **Step 4: Start Finetuning with CORRECT Settings**

Edit `config/training_config.yaml`:

```yaml
model:
  n_vocab: 2535  # Your merged tokenizer size
  
  # CRITICAL: FREEZE ORIGINAL EMBEDDINGS
  freeze_original_embeddings: true  # MUST BE TRUE!
  freeze_until_index: 2454  # Freeze first 2454 (English, French, etc.)
  
  # Model architecture
  d_model: 1024
  nhead: 8
  num_encoder_layers: 6
  dim_feedforward: 2048
  dropout: 0.1

training:
  # CRITICAL: USE LOW LEARNING RATE FOR FINETUNING
  learning_rate: 1e-5  # Much lower than 2e-4!
  
  batch_size: 16
  max_epochs: 1000
  
  # Gradient clipping prevents wild updates
  grad_clip_val: 1.0
  
  # Early stopping
  early_stopping: true
  patience: 50

data:
  dataset_path: data/srt_datasets/your_amharic_dataset
```

---

### **Step 5: Start Training with Frozen Embeddings**

```bash
# In Gradio UI → Training Pipeline tab:

1. Configuration File: config/training_config.yaml
2. Resume from Checkpoint: None (Start from scratch)
3. Dataset: your_amharic_dataset
4. Tokenizer: Am_tokenizer_merged.json

5. CRITICAL SETTINGS:
   - Freeze Original Embeddings: ✅ CHECKED
   - Freeze Until Index: 2454
   - Learning Rate: 0.00001 (1e-5)
   - Batch Size: 8-16

6. Click "Start Training"
```

---

## ✅ **What This Does Correctly:**

### **Embedding Freezing:**
```
Tokens 0-2453:   FROZEN (English, French, etc.)
Tokens 2454-2534: TRAINABLE (Amharic only)
```

**Result:**
- ✅ English still works
- ✅ French still works  
- ✅ Amharic learns new patterns
- ✅ Multilingual capability preserved!

### **Low Learning Rate:**
```
High LR (2e-4): Big updates → destroys weights ❌
Low LR (1e-5):  Small updates → preserves knowledge ✅
```

---

## 📊 **Expected Results:**

### **After Correct Finetuning:**

**English (token 1000):**
```
Before finetuning: "Hello world" → clear speech ✅
After finetuning:  "Hello world" → clear speech ✅ (preserved!)
```

**Amharic (token 2500):**
```
Before finetuning: "ሰላም ለዓለም" → noise ❌ (not trained)
After finetuning:  "ሰላም ለዓለም" → clear speech ✅ (learned!)
```

---

## 🎯 **Verification Checklist:**

Before starting training, verify:

1. ✅ **Config has `freeze_original_embeddings: true`**
2. ✅ **Config has `freeze_until_index: 2454`**
3. ✅ **Learning rate is 1e-5 or lower**
4. ✅ **Extended embeddings loaded (vocab 2535)**
5. ✅ **Freeze checkbox is CHECKED in UI**

During training, verify:

1. ✅ **Only ~81 embedding params trainable** (not all 2535)
2. ✅ **Training loss decreases normally**
3. ✅ **Validation loss improves**

After training, verify:

1. ✅ **English inference still works**
2. ✅ **French inference still works**
3. ✅ **Amharic inference improves**

---

## 🚨 **Common Mistakes to Avoid:**

### **1. Not Freezing Embeddings**
```yaml
# WRONG:
freeze_original_embeddings: false  # ❌

# RIGHT:
freeze_original_embeddings: true   # ✅
```

### **2. Learning Rate Too High**
```yaml
# WRONG (for finetuning):
learning_rate: 2e-4  # ❌ Too high, destroys weights

# RIGHT (for finetuning):
learning_rate: 1e-5  # ✅ Safe for finetuning
```

### **3. Wrong Freeze Index**
```yaml
# WRONG:
freeze_until_index: 0     # ❌ Nothing frozen!
freeze_until_index: 704   # ❌ Too small, only English

# RIGHT:
freeze_until_index: 2454  # ✅ All 23 languages
```

### **4. Using Wrong Checkpoint**
```yaml
# WRONG:
# Using your broken checkpoint that already destroyed embeddings

# RIGHT:
# Start fresh from newly extended embeddings
```

---

## 💡 **Why This Happened:**

### **What Should Have Happened:**
```
1. Download pretrained (23 langs) ✅
2. Extend embeddings (+81 Amharic) ✅
3. FREEZE first 2454 embeddings ✅
4. Train only new 81 embeddings ✅
5. Use low LR (1e-5) ✅
```

### **What Actually Happened (Your Case):**
```
1. Download pretrained (23 langs) ✅
2. Extend embeddings (+81 Amharic) ✅
3. Did NOT freeze embeddings ❌
4. Trained ALL 2535 embeddings ❌
5. Used high LR (2e-4?) ❌
→ Original knowledge destroyed! ❌
```

---

## 🔧 **Recovery Steps Summary:**

```bash
# 1. Pull latest code
git pull origin main

# 2. Test current damage
python scripts/test_pretrained_english.py

# 3. Re-download pretrained model (if needed)
# Use Gradio UI → Model Setup → Download

# 4. Re-extend embeddings
# Use Gradio UI → Model Setup → Extend Embeddings

# 5. Fix config
# Edit config/training_config.yaml:
#   - freeze_original_embeddings: true
#   - freeze_until_index: 2454
#   - learning_rate: 1e-5

# 6. Start training with correct settings
# Use Gradio UI → Training Pipeline
#   - Check "Freeze Original Embeddings"
#   - Set freeze index to 2454
#   - Use learning rate 1e-5
```

---

## 📈 **Expected Training Timeline:**

With correct finetuning (frozen embeddings, low LR):

```
Epoch 1-50:   Loss ~5 → Amharic learning basic patterns
Epoch 50-200: Loss ~2 → Amharic improving
Epoch 200-500: Loss ~1 → Amharic clear speech
```

**Throughout training:**
- ✅ English stays working
- ✅ French stays working
- ✅ Amharic gradually improves

---

## ✅ **Final Checklist:**

Before you start finetuning again:

- [ ] Downloaded original pretrained model
- [ ] Extended embeddings with correct sizes
- [ ] Config has `freeze_original_embeddings: true`
- [ ] Config has `freeze_until_index: 2454`
- [ ] Learning rate is 1e-5
- [ ] Deleted old broken checkpoint
- [ ] Ready to start fresh training

---

## 🎯 **Expected Outcome:**

After correct finetuning:
- ✅ English: "Hello world" → clear English speech
- ✅ French: "Bonjour le monde" → clear French speech
- ✅ Amharic: "ሰላም ለዓለም" → clear Amharic speech
- 🎉 **All 24 languages working!**

---

**Pull latest code and start over with correct settings!** This time, your multilingual capability will be preserved while adding Amharic! 🎙️🇪🇹✨
