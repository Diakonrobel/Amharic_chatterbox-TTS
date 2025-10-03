# Training Completion Guide - Real Model Implementation

This guide provides **step-by-step instructions** to update your training script from dummy placeholder logic to **real TTS model training** with the SimplifiedT3Model.

## Current Status ✓

Your training infrastructure is **fully functional**:
- ✓ Model architecture (SimplifiedT3Model) implemented
- ✓ Audio processing (AudioProcessor, mel spectrograms)
- ✓ Dataset loading with real audio
- ✓ Loss function (TTSLoss) implemented
- ✓ Training loop skeleton with checkpointing
- ✓ Mixed precision (AMP) support
- ✓ TensorBoard logging
- ✓ Extended embeddings loading

## What Needs to Be Done 🎯

You need to replace the **dummy forward pass and loss computation** in `src/training/train.py` with **real model training logic**.

---

## Step-by-Step Changes

### **STEP 1: Add Missing Import for setup_dataloaders**

**Location:** `src/training/train.py`, after line 30

**What to add:**
```python
from src.data.dataset import create_dataloaders
```

**Why:** The function `setup_dataloaders()` is called at line 373 but not defined. You need to either import it or define it in this file.

**Alternative (if the function doesn't exist):** Define it directly in `train.py` (see STEP 2).

---

### **STEP 2: Implement setup_dataloaders Function**

**Location:** `src/training/train.py`, after line 140 (after `load_config` function)

**What to add:**
```python
def setup_dataloaders(config: Dict):
    """Setup training and validation dataloaders"""
    from pathlib import Path
    
    TRAINING_STATE.log("Setting up dataloaders...")
    
    # Initialize audio processor
    audio_processor = AudioProcessor()
    
    # TODO: Add tokenizer loading here if needed
    tokenizer = None  # Replace with actual tokenizer if available
    
    # Create datasets
    data_dir = Path(config['paths']['data_dir'])
    train_metadata = data_dir / 'metadata.csv'
    val_metadata = data_dir / 'metadata_val.csv'  # Or use a split
    
    # Check if validation metadata exists, otherwise use train for both
    if not val_metadata.exists():
        TRAINING_STATE.log("⚠ No separate validation metadata, using training data")
        val_metadata = train_metadata
    
    train_dataset = SimpleAmharicDataset(
        str(train_metadata),
        data_dir,
        audio_processor=audio_processor,
        tokenizer=tokenizer
    )
    
    val_dataset = SimpleAmharicDataset(
        str(val_metadata),
        data_dir,
        audio_processor=audio_processor,
        tokenizer=tokenizer
    )
    
    # Create dataloaders with proper collate function
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers'],
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers'],
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    TRAINING_STATE.log(f"✓ Train samples: {len(train_dataset)}")
    TRAINING_STATE.log(f"✓ Val samples: {len(val_dataset)}")
    TRAINING_STATE.log(f"✓ Batch size: {config['training']['batch_size']}")
    
    return train_loader, val_loader
```

**Why:** This properly initializes the dataloaders with real audio processing and collation.

---

### **STEP 3: Initialize Loss Function**

**Location:** `src/training/train.py`, in the `train()` function after line 370 (after optimizer setup)

**What to add:**
```python
        # Setup loss function
        criterion = TTSLoss(
            mel_loss_weight=1.0,
            duration_loss_weight=0.1
        )
        TRAINING_STATE.log("✓ Loss function initialized")
```

**Why:** You need the loss function instance to compute real losses.

---

### **STEP 4: Replace Dummy Training Loop - Forward Pass**

**Location:** `src/training/train.py`, lines 233-262

**Current code to REPLACE:**
```python
        # TODO: Implement actual forward pass
        # For now, use dummy loss computation
        if config['training']['use_amp']:
            with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                # Dummy computation inside autocast
                dummy_input = torch.randn(1, 10, device=device)
                dummy_output = model.linear(model.embedding(torch.tensor([0], device=device)))
                loss = dummy_output.mean()  # Dummy loss
            
            # Backward with gradient scaling
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                config['training']['grad_clip_thresh']
            )
            scaler.step(optimizer)
            scaler.update()
        else:
            # Dummy computation without autocast
            dummy_input = torch.randn(1, 10, device=device)
            dummy_output = model.linear(model.embedding(torch.tensor([0], device=device)))
            loss = dummy_output.mean()  # Dummy loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                config['training']['grad_clip_thresh']
            )
            optimizer.step()
```

**New code to INSERT:**
```python
        # Move batch to device
        text_ids = batch['text_ids'].to(device)
        text_lengths = batch['text_lengths'].to(device)
        mel_targets = batch['mel'].to(device)
        mel_lengths = batch['mel_lengths'].to(device)
        
        # Forward pass and loss computation
        if config['training']['use_amp']:
            with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                # Model forward
                outputs = model(
                    text_ids=text_ids,
                    text_lengths=text_lengths,
                    mel_targets=mel_targets
                )
                
                # Compute loss
                targets = {
                    'mel': mel_targets,
                    'mel_lengths': mel_lengths
                }
                losses = criterion(outputs, targets)
                loss = losses['total_loss']
            
            # Backward with gradient scaling
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                config['training']['grad_clip_thresh']
            )
            scaler.step(optimizer)
            scaler.update()
        else:
            # Forward without autocast
            outputs = model(
                text_ids=text_ids,
                text_lengths=text_lengths,
                mel_targets=mel_targets
            )
            
            # Compute loss
            targets = {
                'mel': mel_targets,
                'mel_lengths': mel_lengths
            }
            losses = criterion(outputs, targets)
            loss = losses['total_loss']
            
            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                config['training']['grad_clip_thresh']
            )
            optimizer.step()
```

**Why:** This replaces dummy computation with real model forward pass and loss calculation using actual batched data.

---

### **STEP 5: Update train_epoch Function Signature**

**Location:** `src/training/train.py`, line 223

**Current code:**
```python
def train_epoch(model, train_loader, optimizer, scheduler, scaler, config, epoch):
```

**New code:**
```python
def train_epoch(model, train_loader, optimizer, scheduler, scaler, criterion, config, epoch):
```

**Also update the call at line 414:**

**Current code:**
```python
            continue_training = train_epoch(
                model, train_loader, optimizer, scheduler,
                scaler, config, epoch
            )
```

**New code:**
```python
            continue_training = train_epoch(
                model, train_loader, optimizer, scheduler,
                scaler, criterion, config, epoch
            )
```

**Why:** Pass the loss function to the training loop.

---

### **STEP 6: Replace Dummy Validation Loop**

**Location:** `src/training/train.py`, lines 295-313

**Current code to REPLACE:**
```python
def validate(model, val_loader, config):
    """Run validation"""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in val_loader:
            # TODO: Implement validation
            loss = torch.tensor(1.0)
            total_loss += loss.item()
    
    avg_loss = total_loss / len(val_loader)
    TRAINING_STATE.log(f"Validation loss: {avg_loss:.4f}")
    
    if avg_loss < TRAINING_STATE.best_loss:
        TRAINING_STATE.best_loss = avg_loss
        TRAINING_STATE.log(f"✓ New best validation loss: {avg_loss:.4f}")
    
    return avg_loss
```

**New code to INSERT:**
```python
def validate(model, val_loader, criterion, config):
    """Run validation"""
    model.eval()
    total_loss = 0
    total_mel_loss = 0
    total_duration_loss = 0
    device = next(model.parameters()).device
    
    with torch.no_grad():
        for batch in val_loader:
            # Move batch to device
            text_ids = batch['text_ids'].to(device)
            text_lengths = batch['text_lengths'].to(device)
            mel_targets = batch['mel'].to(device)
            mel_lengths = batch['mel_lengths'].to(device)
            
            # Forward pass
            outputs = model(
                text_ids=text_ids,
                text_lengths=text_lengths,
                mel_targets=mel_targets
            )
            
            # Compute loss
            targets = {
                'mel': mel_targets,
                'mel_lengths': mel_lengths
            }
            losses = criterion(outputs, targets)
            
            total_loss += losses['total_loss'].item()
            total_mel_loss += losses['mel_loss'].item()
            total_duration_loss += losses['duration_loss'].item()
    
    avg_loss = total_loss / len(val_loader)
    avg_mel_loss = total_mel_loss / len(val_loader)
    avg_duration_loss = total_duration_loss / len(val_loader)
    
    TRAINING_STATE.log(f"Validation - Total: {avg_loss:.4f} | Mel: {avg_mel_loss:.4f} | Duration: {avg_duration_loss:.4f}")
    
    if avg_loss < TRAINING_STATE.best_loss:
        TRAINING_STATE.best_loss = avg_loss
        TRAINING_STATE.log(f"✓ New best validation loss: {avg_loss:.4f}")
    
    return avg_loss
```

**Also update the call at line 424:**

**Current code:**
```python
                val_loss = validate(model, val_loader, config)
```

**New code:**
```python
                val_loss = validate(model, val_loader, criterion, config)
```

**Why:** Implement real validation with actual model evaluation.

---

### **STEP 7: Add TensorBoard Logging for Detailed Metrics**

**Location:** `src/training/train.py`, after line 275 (in train_epoch, after logging)

**What to add:**
```python
            # TensorBoard logging
            if writer:
                writer.add_scalar('Loss/train_total', loss.item(), TRAINING_STATE.current_step)
                writer.add_scalar('Loss/train_mel', losses['mel_loss'].item(), TRAINING_STATE.current_step)
                writer.add_scalar('Loss/train_duration', losses['duration_loss'].item(), TRAINING_STATE.current_step)
                writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], TRAINING_STATE.current_step)
```

**Note:** You'll need to pass `writer` to the `train_epoch` function.

**Update function signature (line 223):**
```python
def train_epoch(model, train_loader, optimizer, scheduler, scaler, criterion, writer, config, epoch):
```

**Update the call (line 414):**
```python
            continue_training = train_epoch(
                model, train_loader, optimizer, scheduler,
                scaler, criterion, writer, config, epoch
            )
```

**Why:** Log detailed training metrics to TensorBoard for monitoring.

---

### **STEP 8: Handle Edge Cases in Dataset**

**Location:** `src/training/train.py`, lines 102-132 (in `__getitem__`)

**Current code already has exception handling, but ensure it's comprehensive:**

No changes needed here - your dataset already has good error handling that returns dummy data on failure. This is appropriate.

---

## Summary of All Changes

Here's a checklist of all modifications:

- [ ] **STEP 1:** Add import or define `setup_dataloaders`
- [ ] **STEP 2:** Implement `setup_dataloaders()` function
- [ ] **STEP 3:** Initialize `criterion` (loss function) in `train()` function
- [ ] **STEP 4:** Replace dummy forward/backward pass with real model computation
- [ ] **STEP 5:** Update `train_epoch()` signature to accept `criterion`
- [ ] **STEP 6:** Replace dummy validation with real validation loop
- [ ] **STEP 7:** Add TensorBoard logging for detailed metrics (optional but recommended)

---

## Testing Before Full Training

After making these changes, test the setup with a **small test run**:

```bash
# On Lightning AI, test with minimal config
python src/training/train.py --config configs/training_config_test.yaml
```

Create a test config (`configs/training_config_test.yaml`) with:
```yaml
training:
  max_epochs: 1
  max_steps: 10
  batch_size: 2
```

This will verify:
- Data loading works
- Model forward pass executes
- Loss computation succeeds
- Backpropagation runs
- Checkpointing works

---

## Expected Output After Changes

When training starts, you should see:
```
[HH:MM:SS] Setting up SimplifiedT3Model...
[HH:MM:SS] ✓ T3 Model created:
[HH:MM:SS]    Vocab size: 2xxx
[HH:MM:SS]    Model dim: 512
[HH:MM:SS]    Mel channels: 80
[HH:MM:SS] Loading extended embeddings from ...
[HH:MM:SS] ✓ Extended embeddings loaded
[HH:MM:SS] Total parameters: X,XXX,XXX
[HH:MM:SS] Trainable parameters: X,XXX,XXX
[HH:MM:SS] Setting up dataloaders...
[HH:MM:SS] ✓ Loaded XXX samples from metadata.csv
[HH:MM:SS] ✓ Train samples: XXX
[HH:MM:SS] Starting training...
[HH:MM:SS] Epoch 1 | Step 1 | Loss: 125.4321 | Avg: 125.4321 | LR: 0.000100
```

Notice the loss values are now **realistic** (e.g., ~100-150 initially for MSE mel loss) instead of dummy negative values.

---

## Troubleshooting Common Issues

### Issue 1: "collate_fn not found"
**Solution:** Make sure `from src.audio import AudioProcessor, collate_fn` is at the top of the file.

### Issue 2: "text_ids tensor shape mismatch"
**Solution:** The collate function should pad sequences. Verify `collate_fn` is properly implemented in `src/audio/audio_processing.py`.

### Issue 3: "CUDA out of memory"
**Solution:** Reduce `batch_size` in config or enable gradient accumulation.

### Issue 4: "Loss is NaN"
**Solution:** 
- Check data preprocessing (normalize mel spectrograms)
- Reduce learning rate
- Check for inf/NaN in input data

---

## Next Steps After Completion

Once you've made these changes and verified training works:

1. **Train on full dataset** with proper config
2. **Monitor TensorBoard** for loss curves
3. **Evaluate checkpoints** with inference script
4. **Fine-tune hyperparameters** (learning rate, batch size, etc.)
5. **Generate audio samples** from trained model

---

## Questions or Issues?

If you encounter any problems while implementing these changes:
- Check the logs carefully for error messages
- Verify all imports are correct
- Ensure your config file has all required fields
- Test with a minimal dataset first (5-10 samples)

Good luck with your Amharic TTS training! 🚀
