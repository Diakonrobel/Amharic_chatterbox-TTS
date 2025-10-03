# Lightning AI Setup - Quick Start

## 🚀 Pull and Test on Lightning AI

### 1. Pull Latest Changes
```bash
cd ~/amharic-tts  # or your repo location
git pull origin main
```

### 2. Quick Test (2-3 steps)
```bash
# Run training and stop after a few steps (Ctrl+C)
python src/training/train.py --config configs/training_config.yaml
```

**Expected Output:**
```
[HH:MM:SS] Setting up SimplifiedT3Model...
[HH:MM:SS] ✓ Loss function initialized
[HH:MM:SS] ✓ Loaded XXX samples
[HH:MM:SS] Epoch 1 | Step 1 | Loss: 142.3456 | Avg: 142.3456
```

✅ **Success:** Loss is positive (100-150), no errors

### 3. Start Full Training
```bash
nohup python src/training/train.py --config configs/training_config.yaml > training.log 2>&1 &
tail -f training.log
```

## 🔧 What's Ready

✅ Real training pipeline (no dummy code)  
✅ setup_dataloaders() with AudioProcessor  
✅ Real model forward pass and loss  
✅ Validation loop with metrics  
✅ TensorBoard logging  

## 📊 Expected Loss

Initial: ~100-150  
After 1k steps: ~50-80  
After 10k steps: ~20-40  

Loss should **decrease**!

## 🎯 Config

Edit `configs/training_config.yaml` for batch_size, learning_rate, etc.

Ready to train! 🚀
