# Lightning AI Training Fix Guide

## 🚨 Problem Summary

Your training completed with **WRONG configuration**:
- ❌ Learning Rate: **0.000198** (should be 0.000010)
- ❌ Final Loss: **3.0716** (should be ~1.0-1.5)
- ❌ Pretrained languages likely **CORRUPTED**

---

## ✅ Solution: Run Diagnostic & Fix Scripts

I've created automated scripts to diagnose and fix your training setup on Lightning AI.

---

## 📋 **Step-by-Step Instructions**

### **Step 1: SSH into Lightning AI**

Open your terminal (PowerShell on Windows) and run:

```powershell
ssh s_01k6mtt0e2dytnw6d8nvbpbrnd@ssh.lightning.ai
```

You'll be connected to your Lightning AI studio.

---

### **Step 2: Navigate to Project**

```bash
cd /teamspace/studios/this_studio/Amharic_chatterbox-TTS
```

---

### **Step 3: Run Diagnostic Script**

```bash
bash scripts/diagnose_and_fix_training.sh
```

**This will:**
- ✅ Check git status
- ✅ Analyze all config files
- ✅ Test your checkpoint
- ✅ Review training logs
- ✅ Provide detailed recommendations

**Expected output:**
```
═══════════════════════════════════════════════════════════════════════
🔍 AMHARIC TTS TRAINING DIAGNOSTIC & FIX SCRIPT
═══════════════════════════════════════════════════════════════════════

📦 STEP 1: Checking Git Status
...
📋 STEP 2: Analyzing Configuration Files
...
💾 STEP 3: Analyzing Checkpoints
...
🎯 DIAGNOSIS SUMMARY
...
💡 RECOMMENDATIONS
```

**Read the output carefully!** It will tell you exactly what's wrong.

---

### **Step 4: Apply Automated Fix**

If the diagnostic confirms the problem, run:

```bash
bash scripts/apply_training_fix.sh
```

**This will automatically:**
- ✅ Pull latest code from GitHub
- ✅ Backup corrupted checkpoint
- ✅ Verify configuration
- ✅ Check extended model and tokenizer

---

### **Step 5: Test Current Checkpoint (Optional)**

To see if your current checkpoint is usable:

```bash
python scripts/test_checkpoint_multilingual.py
```

**This analyzes:**
- Embedding corruption levels
- Expected audio quality
- Multilingual capability preservation

---

### **Step 6: Restart Training with Correct Config**

After the fix is applied:

1. **Start Gradio App:**
   ```bash
   python gradio_app/full_training_app.py
   ```

2. **In Gradio UI → Training Pipeline tab:**
   - **Config File:** `config/training_config_finetune_FIXED.yaml`
   - **Resume from Checkpoint:** `None` (start from extended embeddings)
   - **Click:** `Start Training`

3. **CRITICAL: Verify these in logs:**
   ```
   ✅ LR: 0.000010 (NOT 0.000198)
   ✅ Shows: 🔒 FREEZING ORIGINAL EMBEDDINGS
   ✅ Shows: Frozen (0-2453): 2454 ❄️
   ✅ Shows: Trainable (2454-2534): 81 🔥
   ```

---

## 🎯 **Expected Results with Correct Config**

### **Training Logs Should Look Like:**
```
Epoch 1  | Step 1   | Loss: 10.2 | Avg: 10.2 | LR: 0.000010  ← Notice LR!
Epoch 5  | Step 200 | Loss: 7.5  | Avg: 7.5  | LR: 0.000010
Epoch 10 | Step 400 | Loss: 5.2  | Avg: 5.2  | LR: 0.000010
Epoch 20 | Step 800 | Loss: 3.1  | Avg: 3.1  | LR: 0.000010
Epoch 50 | Step 2000| Loss: 1.8  | Avg: 1.8  | LR: 0.000010
Epoch 100| Step 4000| Loss: 1.2  | Avg: 1.2  | LR: 0.000010  ← Good!
```

### **Key Differences:**

| Metric | Wrong (Old) | Correct (Fixed) | Status |
|--------|-------------|-----------------|--------|
| Learning Rate | 0.000198 | 0.000010 | ✅ 20x lower |
| Final Loss (100 epochs) | 3.0716 | ~1.0-1.5 | ✅ 2-3x better |
| Loss Trend | Erratic | Smooth | ✅ Stable |
| English Audio | Noise | Clear | ✅ Preserved |
| Amharic Audio | Poor | Good | ✅ Learning |

---

## 🧪 **Testing Your Checkpoint**

After training (or to test current checkpoint):

1. **In Gradio UI → Advanced TTS tab**

2. **Test English:**
   - Input: `Hello world, this is a test of the English language`
   - Expected: Clear English speech
   - If noise: ❌ Model is corrupted

3. **Test Amharic:**
   - Input: `ሰላም ለዓለም`
   - Expected: Clear Amharic speech (improves over training)
   - Quality improves as training progresses

4. **Test French:**
   - Input: `Bonjour le monde`
   - Expected: Clear French speech
   - If noise: ❌ Model is corrupted

---

## 📞 **Quick Commands Reference**

### **Diagnostic:**
```bash
ssh s_01k6mtt0e2dytnw6d8nvbpbrnd@ssh.lightning.ai
cd /teamspace/studios/this_studio/Amharic_chatterbox-TTS
bash scripts/diagnose_and_fix_training.sh
```

### **Apply Fix:**
```bash
bash scripts/apply_training_fix.sh
```

### **Test Checkpoint:**
```bash
python scripts/test_checkpoint_multilingual.py
```

### **Check Config:**
```bash
grep "learning_rate:" config/training_config_finetune_FIXED.yaml
# Should show: learning_rate: 1.0e-5
```

### **View Recent Training Logs:**
```bash
tail -50 logs/training.log
```

---

## ❓ **FAQ**

### **Q: Can I salvage my current checkpoint?**
**A:** Unlikely. The high learning rate (0.000198) likely corrupted the pretrained embeddings. You should start fresh.

### **Q: How long will correct training take?**
**A:** Same as before (~2 hours for 100 epochs), but with proper results.

### **Q: Will I lose my dataset?**
**A:** No! Your dataset in `data/srt_datasets/` is safe.

### **Q: What about my tokenizer?**
**A:** Your merged tokenizer is safe. You'll reuse it.

### **Q: What if the scripts fail?**
**A:** Copy the error message and share it. I'll help debug.

---

## 🎯 **Success Checklist**

Before restarting training, verify:

- [ ] SSH'd into Lightning AI successfully
- [ ] Ran `bash scripts/diagnose_and_fix_training.sh`
- [ ] Read diagnostic output thoroughly
- [ ] Ran `bash scripts/apply_training_fix.sh`
- [ ] Saw `✅ FIX APPLIED SUCCESSFULLY`
- [ ] Verified config has `learning_rate: 1.0e-5`
- [ ] Ready to restart training in Gradio

After starting training, verify:

- [ ] Logs show `LR: 0.000010` (not 0.000198)
- [ ] See `🔒 FREEZING ORIGINAL EMBEDDINGS` message
- [ ] See frozen/trainable embedding counts
- [ ] Loss decreases smoothly (not erratic)
- [ ] After 10 epochs: loss ~5-6 (not stuck at ~3)

---

## 🚀 **You're Ready!**

Run the diagnostic script now:

```bash
ssh s_01k6mtt0e2dytnw6d8nvbpbrnd@ssh.lightning.ai
cd /teamspace/studios/this_studio/Amharic_chatterbox-TTS
bash scripts/diagnose_and_fix_training.sh
```

The script will guide you through the rest! 🎉

---

**Last Updated:** 2025-01-04  
**Your SSH Command:** `ssh s_01k6mtt0e2dytnw6d8nvbpbrnd@ssh.lightning.ai`
