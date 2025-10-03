# Training Completion Checklist

Quick reference for implementing real TTS training. See `TRAINING_COMPLETION_GUIDE.md` for detailed instructions.

---

## ✅ Tasks to Complete

### Step 1: Setup Dataloaders Function
- [ ] Add `setup_dataloaders()` function after line 140 in `train.py`
- [ ] Function should:
  - Initialize `AudioProcessor()`
  - Create `SimpleAmharicDataset` for train and validation
  - Create `DataLoader` instances with `collate_fn`
  - Return `train_loader, val_loader`

### Step 2: Initialize Loss Function  
- [ ] Add after line 370 (after optimizer setup):
```python
criterion = TTSLoss(mel_loss_weight=1.0, duration_loss_weight=0.1)
```

### Step 3: Update `train_epoch()` Signature
- [ ] Change line 223 from:
  ```python
  def train_epoch(model, train_loader, optimizer, scheduler, scaler, config, epoch):
  ```
  To:
  ```python
  def train_epoch(model, train_loader, optimizer, scheduler, scaler, criterion, writer, config, epoch):
  ```

- [ ] Update call at line 414 to pass `criterion` and `writer`

### Step 4: Replace Dummy Forward Pass
- [ ] Replace lines 233-262 with real forward pass:
  - Move batch tensors to device
  - Call `model(text_ids, text_lengths, mel_targets)`
  - Compute loss with `criterion(outputs, targets)`
  - Use actual `loss` instead of dummy

### Step 5: Update Validation Function
- [ ] Change line 295 signature from:
  ```python
  def validate(model, val_loader, config):
  ```
  To:
  ```python
  def validate(model, val_loader, criterion, config):
  ```

- [ ] Replace lines 296-313 with real validation:
  - Move batch to device
  - Forward pass with model
  - Compute losses with criterion
  - Log detailed metrics (total, mel, duration)

- [ ] Update call at line 424 to pass `criterion`

### Step 6: Add TensorBoard Logging (Optional)
- [ ] Add after line 275 in `train_epoch()`:
```python
if writer:
    writer.add_scalar('Loss/train_total', loss.item(), TRAINING_STATE.current_step)
    writer.add_scalar('Loss/train_mel', losses['mel_loss'].item(), TRAINING_STATE.current_step)
    writer.add_scalar('Loss/train_duration', losses['duration_loss'].item(), TRAINING_STATE.current_step)
```

---

## 🧪 Testing After Changes

### Quick Test (2-3 minutes)
```bash
# Create minimal test config
python src/training/train.py --config configs/training_config.yaml
# Stop after 1-2 steps to verify it runs
```

### Verify These Things Work:
- [ ] Data loading (no errors loading audio/metadata)
- [ ] Model forward pass executes
- [ ] Loss values are realistic (not dummy negative values)
- [ ] Backpropagation completes
- [ ] Checkpoint saves successfully
- [ ] TensorBoard logs created (if enabled)

### Expected First Loss Values:
- **Total Loss:** ~100-150 (MSE mel loss dominates)
- **Mel Loss:** ~100-150
- **Duration Loss:** ~0.5-2.0

If you see these ranges, training is working correctly! ✓

---

## 🔧 Common Fixes

### Error: `collate_fn not found`
```python
# Add to imports at top of train.py
from src.audio import AudioProcessor, collate_fn
```

### Error: `setup_dataloaders not defined`
Make sure you added the function definition in Step 1.

### Error: Shape mismatch in forward pass
Check that `collate_fn` properly pads all sequences to same length.

### Warning: Loss is NaN
- Reduce learning rate
- Check mel normalization in audio preprocessing
- Verify no inf/NaN values in dataset

---

## 📊 Monitoring Training

After starting real training:

1. **Watch console logs** for loss progression
2. **Check TensorBoard** (if enabled):
   ```bash
   tensorboard --logdir logs/
   ```
3. **Verify checkpoints** are being saved regularly
4. **Monitor GPU memory** usage

---

## ✨ Success Indicators

You'll know training is working when you see:

- ✓ Loss values decrease over time
- ✓ Mel loss starting around 100-150 and going down
- ✓ Duration loss staying stable around 1-5
- ✓ No crashes or NaN losses
- ✓ Checkpoints saving every N steps
- ✓ GPU utilization high (if using CUDA)

---

## 📝 Notes

- All line numbers are approximate - adjust based on your file
- Test with small dataset first (5-10 samples)
- Keep the dummy error handling in dataset `__getitem__`
- Backup your working code before making changes!

---

**Ready to implement?** Follow the detailed guide in `TRAINING_COMPLETION_GUIDE.md` for full code snippets! 🚀
