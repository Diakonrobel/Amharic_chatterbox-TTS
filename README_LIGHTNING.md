# 🚀 Lightning AI Training Guide - Quick Reference

## 🎯 One-Command Setup

```bash
# Clone your repo
git clone https://github.com/YOUR_USERNAME/amharic-tts.git
cd amharic-tts

# Run automated setup
chmod +x setup_lightning.sh && ./setup_lightning.sh

# Quick start with Gradio UI
python lightning_quickstart.py
```

---

## 📊 What You Get on Lightning AI

### Free Tier
- ✅ **GPU:** NVIDIA T4 (16GB VRAM)
- ✅ **RAM:** 32GB  
- ✅ **Storage:** 50GB
- ✅ **Time:** Limited hours per month
- ✅ **Public URLs:** For Gradio/TensorBoard

### Recommended Settings for T4 GPU

```yaml
# config/training_config.yaml
data:
  batch_size: 24  # Optimal for T4
  num_workers: 4

training:
  use_amp: true  # Essential for T4
  max_epochs: 100
  save_interval: 2000  # Save often!
```

---

## 📦 Dataset Upload Options

### Option 1: Direct Upload (Small datasets < 1GB)
```bash
# In Lightning AI Studio:
# Files → Upload → Select your dataset.zip
cd data/srt_datasets
unzip ../../dataset.zip
```

### Option 2: Google Drive (Recommended)
```bash
# 1. Share your dataset folder/file publicly on Google Drive
# 2. Get the file ID from the share link
# 3. Download in Lightning AI:

pip install gdown
gdown FILE_ID -O dataset.zip
unzip dataset.zip -d data/srt_datasets/
```

### Option 3: Hugging Face Hub
```bash
# If you've uploaded to HF
pip install huggingface_hub
python -c "
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id='YOUR_USERNAME/amharic-dataset',
    filename='dataset.zip',
    local_dir='data'
)
"
unzip data/dataset.zip -d data/srt_datasets/
```

---

## 🎮 Three Ways to Train

### 1️⃣ Gradio Web UI (Easiest ⭐)

```bash
python gradio_app/full_training_app.py --share
```

**Then:**
1. Open the public URL (https://xxxxx.gradio.live)
2. Go to "Training Pipeline" tab
3. Configure parameters
4. Click "Start Training"
5. Monitor in real-time!

### 2️⃣ Quick Start Script

```bash
python lightning_quickstart.py
```

Automatically:
- Checks setup
- Verifies GPU
- Launches Gradio UI

### 3️⃣ Command Line

```bash
# Direct training
python src/training/train.py \
    --config config/training_config.yaml
```

---

## 📈 Monitoring Options

### Gradio UI (Built-in)
- Real-time status
- Live logs
- Loss tracking
- Checkpoint management

### TensorBoard
```bash
tensorboard --logdir logs --host 0.0.0.0 --port 6006
# Access via Lightning AI's public URL
```

### Direct Logs
```bash
# Watch logs in real-time
tail -f logs/training.log

# Or check specific checkpoint info
ls -lh models/checkpoints/
```

---

## ⚡ Performance Optimization

### Memory Management
```yaml
# If you get OOM errors:
data:
  batch_size: 12  # Reduce from 24
  
training:
  grad_accumulation_steps: 2  # Effective batch = 12*2 = 24
  use_amp: true  # Must be enabled
```

### Speed Optimization
```yaml
data:
  num_workers: 4  # Parallel data loading
  pin_memory: true  # Faster GPU transfer
  
training:
  use_amp: true  # 2x faster
```

### Checkpointing Strategy
```yaml
training:
  save_interval: 2000  # Every 2000 steps
  eval_interval: 1000  # Validate often
```

**Why frequent saves?**
- Lightning AI sessions can timeout
- Resume easily from last checkpoint
- No progress lost

---

## 💾 Save & Download Model

### During Training
Checkpoints save automatically to `models/checkpoints/`

### After Training

**Method 1: Lightning AI UI**
```
Files → models/checkpoints/ → Download
```

**Method 2: Package for Download**
```bash
# Create archive
tar -czf amharic_tts_model.tar.gz \
    models/checkpoints/checkpoint_latest.pt \
    models/tokenizer/ \
    config/training_config.yaml

# Download via UI or:
# Right-click → Download
```

**Method 3: Upload to HuggingFace**
```bash
pip install huggingface_hub
python -c "
from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(
    folder_path='models/checkpoints',
    repo_id='YOUR_USERNAME/amharic-tts-model',
    repo_type='model'
)
"
```

---

## 🔄 Resume Training

If session disconnects or times out:

**Via Gradio UI:**
1. Restart: `python gradio_app/full_training_app.py --share`
2. Training Pipeline tab
3. Click 🔄 next to checkpoint dropdown
4. Select `checkpoint_latest.pt`
5. Start Training

**Via CLI:**
```bash
python src/training/train.py \
    --config config/training_config.yaml \
    --resume models/checkpoints/checkpoint_latest.pt
```

---

## 🐛 Common Issues & Solutions

### Issue: Session Timeout
**Solution:**
- Save checkpoints every 1000-2000 steps
- Resume from latest checkpoint
- Consider upgrading to paid tier for longer sessions

### Issue: Out of Memory
**Solution:**
```yaml
data:
  batch_size: 8  # Start small
  
training:
  grad_accumulation_steps: 4  # Maintain effective batch size
```

### Issue: Slow Training
**Solution:**
- Verify GPU: `nvidia-smi`
- Enable AMP: `use_amp: true`
- Increase workers: `num_workers: 4`

### Issue: Dataset Not Found
**Solution:**
```bash
# Check dataset location
ls -la data/srt_datasets/

# Update config
nano config/training_config.yaml
# Set: dataset_path: "data/srt_datasets/YOUR_DATASET"
```

### Issue: Gradio Not Accessible
**Solution:**
```bash
# Use --share flag for public URL
python gradio_app/full_training_app.py --share --port 7860

# If port busy, try different port
python gradio_app/full_training_app.py --share --port 7861
```

---

## 📊 Training Progress Checklist

- [ ] Repository cloned
- [ ] Setup script run successfully
- [ ] GPU verified (`nvidia-smi`)
- [ ] Dataset uploaded to `data/srt_datasets/`
- [ ] Config updated with dataset path
- [ ] Gradio UI launched with `--share`
- [ ] Training started
- [ ] TensorBoard running (optional)
- [ ] Checkpoints saving regularly
- [ ] Monitoring training progress

---

## 💡 Pro Tips

### 1. **Use Gradio Share Mode**
```bash
python gradio_app/full_training_app.py --share
```
Access from anywhere, even after closing laptop!

### 2. **Save Frequently**
Set `save_interval: 1000` for Lightning AI's free tier.

### 3. **Monitor GPU Usage**
```bash
watch -n 1 nvidia-smi
```

### 4. **Test First**
Run 1 epoch first to verify everything works:
```yaml
training:
  max_epochs: 1  # Test run
```

### 5. **Batch Size Calculator**
```
Available VRAM: 16GB (T4)
Model Size: ~400MB
Safe Batch Size = (VRAM - Model) / (Memory per sample)

For T4: Start with batch_size: 24
```

---

## 🎓 Typical Training Session

```bash
# 1. Start
git clone YOUR_REPO && cd amharic-tts
./setup_lightning.sh

# 2. Upload dataset
# (Use Files UI or gdown)

# 3. Launch
python lightning_quickstart.py

# 4. In Gradio UI:
# - Configure parameters
# - Start training
# - Monitor progress

# 5. Wait for completion
# (Training runs in background)

# 6. Download model
# Files → models/checkpoints/ → Download
```

**Estimated Time:**
- 5 hours of audio → ~6-12 hours training
- 10 hours of audio → ~12-24 hours training
- 20+ hours of audio → ~24-48 hours training

---

## 📞 Need Help?

1. **Check Logs:** `tail -f logs/training.log`
2. **GPU Status:** `nvidia-smi`
3. **Gradio Status:** Check public URL
4. **Lightning AI Docs:** https://lightning.ai/docs

---

## 🎉 Success Indicators

✅ **Training is working when you see:**
- Gradio public URL active
- GPU usage > 70% (nvidia-smi)
- Loss decreasing steadily
- Checkpoints saving regularly
- No error messages in logs

---

**Happy Training on Lightning AI! ⚡🚀**
