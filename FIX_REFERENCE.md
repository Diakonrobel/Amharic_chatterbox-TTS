# 🔧 Fix Reference - Training Errors Resolved

## 🐛 Errors Fixed (Commit: 02d5686)

### Error 1: KeyError: 'data_dir'
```python
KeyError: 'data_dir'
File "/teamspace/studios/this_studio/Amharic_chatterbox-TTS/src/training/train.py", line 155
    data_dir = Path(config['paths']['data_dir'])
```

**Root Cause:** The training script expected `paths.data_dir` but the config used `data.dataset_path`.

**Fix Applied:**
- ✅ Multi-location config search: checks `paths.data_dir`, `data.dataset_path`, and fallback default
- ✅ Gradio UI now sets both `data.dataset_path` AND `paths.data_dir` for compatibility
- ✅ Safe config retrieval with `.get()` and defaults

### Error 2: Shape Mismatch (Embeddings)
```
size mismatch for text_embedding.weight: 
copying a param with shape torch.Size([3000, 1024]) from checkpoint, 
the shape in current model is torch.Size([3000, 512]).
```

**Root Cause:** The extended embeddings were created with `d_model=1024` but SimplifiedT3Model uses `d_model=512`.

**Status:** ⚠️ Warning (non-fatal) - Model continues with randomly initialized weights.

**Proper Fix Needed:**
1. **Option A** - Re-extend the embeddings with correct dimension:
   ```bash
   python scripts/extend_model_embeddings.py \
     --model models/pretrained/chatterbox/t3_mtl23ls_v2.safetensors \
     --output models/pretrained/chatterbox_extended.pt \
     --original-size 2454 \
     --new-size 3000 \
     --d-model 512  # Add this parameter
   ```

2. **Option B** - Skip pretrained loading (train from scratch):
   - Set `finetuning.enabled: false` in config
   - Or remove the extended model file

3. **Option C** - Match the model dimension to embeddings:
   - Change `d_model: 512` to `d_model: 1024` in training script
   - **Not recommended** - increases memory usage significantly

## ✅ What's Working Now

After pulling the latest changes (`git pull origin main`):

1. **Data Directory Loading** ✅
   - Checks multiple config locations
   - Falls back gracefully to defaults
   - Clear error messages if data not found

2. **Config Robustness** ✅
   - All required sections auto-created if missing
   - Safe retrieval of nested config values
   - Batch size and num_workers have defaults

3. **Lightning AI Compatibility** ✅
   - num_workers reduced to 2 (prevents worker crashes)
   - Config validation before training starts
   - Better error reporting

## 🚀 Quick Start (Lightning AI)

After `git pull`:

```bash
# 1. Check your dataset location
ls -la data/srt_datasets/

# 2. Update config if needed
# Edit config/training_config.yaml:
# paths:
#   data_dir: "data/srt_datasets/YOUR_DATASET_NAME"

# 3. Launch training interface
python gradio_app/full_training_app.py --share

# 4. In web interface:
#    - Go to "Training Pipeline" tab
#    - Set "Dataset Path" to your actual dataset location
#    - Click "Start Training"
```

## 🔍 Troubleshooting

### Still getting KeyError?

Check your dataset location:
```bash
# Your dataset should have this structure:
data/srt_datasets/my_dataset/
├── metadata.csv
├── metadata_val.csv (optional)
└── wavs/
    ├── audio1.wav
    ├── audio2.wav
    └── ...
```

Update the path in UI or config to match your actual dataset name.

### Shape mismatch warning?

This is **non-fatal**. Training will continue, but:
- **If you have extended embeddings**: Re-create them with correct dimension (512)
- **If training from scratch**: Ignore this warning - it's expected
- **Model loads but with random weights**: Expected behavior, training will learn from scratch

### Training starts but crashes immediately?

Check:
1. Dataset path is correct
2. `metadata.csv` exists and has correct format: `filename.wav|transcription text`
3. WAV files exist in `wavs/` subdirectory
4. CUDA is available: `python -c "import torch; print(torch.cuda.is_available())"`

## 📝 Config Changes Summary

**Before:**
```yaml
# Old config structure - would cause KeyError
data:
  dataset_path: "data/processed/ljspeech_format"
# Missing: paths.data_dir
```

**After:**
```yaml
# New config structure - robust
data:
  dataset_path: "data/processed/ljspeech_format"

paths:
  data_dir: "data/srt_datasets/my_dataset"  # Added!
```

**Now supports both!** The training script checks multiple locations automatically.

## 🎯 Next Steps

1. ✅ Pull changes: `git pull origin main`
2. ✅ Verify dataset structure (see above)
3. ✅ Update config with correct data path
4. ✅ Launch training interface
5. ✅ Start training!

If shape mismatch persists and you want to use pretrained weights properly:
- Re-generate extended embeddings with matching dimensions
- Or train from scratch (faster to start, longer to converge)

---

**Commit Hash:** `02d5686`  
**Files Changed:** 3 core files + docs  
**Status:** Ready for training! 🚀