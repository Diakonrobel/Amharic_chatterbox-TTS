# Code Changes Reference - Side-by-Side Comparison

This document shows the exact code changes needed in `src/training/train.py` to implement real training.

---

## Change 1: Add setup_dataloaders Function

**Insert after line 140 (after `load_config` function):**

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

---

## Change 2: Initialize Loss Function in train()

**Insert after line 370 (after optimizer setup, before dataloaders):**

```python
        # Setup loss function
        criterion = TTSLoss(
            mel_loss_weight=1.0,
            duration_loss_weight=0.1
        )
        TRAINING_STATE.log("✓ Loss function initialized")
```

---

## Change 3: Update train_epoch Signature

**Line 223 - BEFORE:**
```python
def train_epoch(model, train_loader, optimizer, scheduler, scaler, config, epoch):
```

**Line 223 - AFTER:**
```python
def train_epoch(model, train_loader, optimizer, scheduler, scaler, criterion, writer, config, epoch):
```

**Line 414 - BEFORE:**
```python
            continue_training = train_epoch(
                model, train_loader, optimizer, scheduler,
                scaler, config, epoch
            )
```

**Line 414 - AFTER:**
```python
            continue_training = train_epoch(
                model, train_loader, optimizer, scheduler,
                scaler, criterion, writer, config, epoch
            )
```

---

## Change 4: Replace Dummy Forward Pass

**Lines 233-262 - BEFORE:**
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

**Lines 233-262 - AFTER:**
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

---

## Change 5: Add TensorBoard Logging in train_epoch

**Insert after line 275 (after the log message):**

```python
            # TensorBoard logging
            if writer:
                writer.add_scalar('Loss/train_total', loss.item(), TRAINING_STATE.current_step)
                writer.add_scalar('Loss/train_mel', losses['mel_loss'].item(), TRAINING_STATE.current_step)
                writer.add_scalar('Loss/train_duration', losses['duration_loss'].item(), TRAINING_STATE.current_step)
                writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], TRAINING_STATE.current_step)
```

---

## Change 6: Replace validate Function

**Lines 295-313 - BEFORE:**
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

**Lines 295-313 - AFTER:**
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

**Line 424 - BEFORE:**
```python
                val_loss = validate(model, val_loader, config)
```

**Line 424 - AFTER:**
```python
                val_loss = validate(model, val_loader, criterion, config)
```

---

## Summary of Files Modified

- **File:** `src/training/train.py`
- **Total Changes:** 6 major modifications
- **Lines Added:** ~150 lines
- **Lines Removed:** ~30 lines (dummy code)

---

## Quick Implementation Order

1. ✅ Add `setup_dataloaders()` function (new function)
2. ✅ Add `criterion = TTSLoss(...)` in `train()` function
3. ✅ Update `train_epoch()` signature
4. ✅ Replace dummy forward pass with real training
5. ✅ Add TensorBoard logging (optional)
6. ✅ Replace `validate()` function with real validation
7. ✅ Update all function calls to match new signatures

---

## Verification Checklist

After making all changes:

- [ ] File saves without syntax errors
- [ ] All function signatures match their calls
- [ ] `criterion` is passed to `train_epoch` and `validate`
- [ ] `writer` is passed to `train_epoch`
- [ ] Dummy code is completely removed
- [ ] Real model forward pass is used
- [ ] Loss computation uses `criterion`

---

## Testing Command

```bash
# Quick syntax check
python -m py_compile src/training/train.py

# Test training (stop after 1-2 steps)
python src/training/train.py --config configs/training_config.yaml
```

---

**Implementation Tips:**

1. **Work in order** - Start with Change 1 and proceed sequentially
2. **Test after each change** - Run syntax check after each modification
3. **Keep a backup** - Copy the original file before editing
4. **Use search** - Use Ctrl+F to find exact line numbers in your file
5. **Check indentation** - Python is sensitive to indentation!

---

Good luck with your implementation! 🎯
