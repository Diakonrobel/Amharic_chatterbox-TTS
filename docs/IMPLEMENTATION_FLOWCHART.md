# Implementation Flowchart and Architecture

Visual guide to the training implementation and data flow.

---

## 🔄 Implementation Flow

```
START
  │
  ├─► [1] Read Implementation Summary
  │         └─► Understand current status & required changes
  │
  ├─► [2] Backup train.py
  │         └─► Create safety copy of working code
  │
  ├─► [3] Open Documentation
  │         ├─► TRAINING_COMPLETION_GUIDE.md (main reference)
  │         ├─► TRAINING_COMPLETION_CHECKLIST.md (progress tracker)
  │         └─► CODE_CHANGES_REFERENCE.md (code snippets)
  │
  ├─► [4] Implement Changes (30-45 min)
  │         ├─► Step 1: Add setup_dataloaders()
  │         ├─► Step 2: Initialize criterion (TTSLoss)
  │         ├─► Step 3: Update train_epoch signature
  │         ├─► Step 4: Replace dummy forward pass
  │         ├─► Step 5: Update validation function
  │         ├─► Step 6: Add TensorBoard logging (optional)
  │         └─► Step 7: Update all function calls
  │
  ├─► [5] Syntax Check
  │         └─► python -m py_compile src/training/train.py
  │
  ├─► [6] Quick Test (1-2 steps)
  │         └─► Verify loss values are realistic
  │
  └─► [7] Full Training
            └─► Monitor and evaluate
  
SUCCESS! 🎉
```

---

## 🏗️ Training Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      TRAINING PIPELINE                       │
└─────────────────────────────────────────────────────────────┘

┌──────────────────────┐
│   Training Config    │
│  (training_config.   │
│      yaml)           │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│   Main Training      │
│   Function           │
│   train()            │
└──────────┬───────────┘
           │
           ├──────────────────────────┐
           │                          │
           ▼                          ▼
┌──────────────────────┐   ┌──────────────────────┐
│   Setup Model        │   │  Setup Dataloaders   │
│  setup_model()       │   │ setup_dataloaders()  │
│                      │   │                      │
│ • SimplifiedT3Model  │   │ • AudioProcessor     │
│ • Load pretrained    │   │ • SimpleAmharicDS    │
│ • Freeze embeddings  │   │ • DataLoader         │
└──────────┬───────────┘   └──────────┬───────────┘
           │                          │
           │                          │
           └──────────┬───────────────┘
                      │
                      ▼
           ┌──────────────────────┐
           │ Setup Training Tools │
           │                      │
           │ • Optimizer (AdamW)  │
           │ • Scheduler          │
           │ • Criterion (TTS     │
           │   Loss)              │
           │ • GradScaler (AMP)   │
           │ • TensorBoard Writer │
           └──────────┬───────────┘
                      │
                      ▼
           ┌──────────────────────┐
           │   Training Loop      │
           │   (Epoch Loop)       │
           └──────────┬───────────┘
                      │
                      ▼
           ┌──────────────────────┐
           │   train_epoch()      │
           │                      │
           │  For each batch:     │
           │  1. Load batch       │
           │  2. Forward pass     │
           │  3. Compute loss     │
           │  4. Backward         │
           │  5. Update weights   │
           │  6. Log metrics      │
           └──────────┬───────────┘
                      │
                      ├──► Checkpointing
                      │    (every N steps)
                      │
                      └──► Validation
                           validate()
```

---

## 📊 Data Flow Through Model

```
┌─────────────────────────────────────────────────────────────┐
│                       DATA PIPELINE                          │
└─────────────────────────────────────────────────────────────┘

INPUT:
  metadata.csv (audio_file|text)
       │
       ▼
┌──────────────────────┐
│ SimpleAmharicDataset │
│  __getitem__         │
└──────────┬───────────┘
           │
           ├─► Load WAV file
           │     │
           │     ▼
           │   AudioProcessor
           │     │
           │     └─► Mel Spectrogram [80 x Time]
           │
           └─► Tokenize Text
                 │
                 └─► Token IDs [seq_len]
                 
           │
           ▼
┌──────────────────────┐
│   collate_fn()       │
│   (Batching)         │
└──────────┬───────────┘
           │
           │  Pad sequences to same length
           │  Stack into tensors
           │
           ▼
    BATCH DICTIONARY:
    {
      'text_ids': [B, seq_len],
      'text_lengths': [B],
      'mel': [B, 80, T],
      'mel_lengths': [B]
    }
           │
           ▼
┌──────────────────────┐
│  SimplifiedT3Model   │
│     forward()        │
└──────────┬───────────┘
           │
           ├─► Text Embedding [B, seq_len, d_model]
           │     │
           │     ▼
           │   Positional Encoding
           │     │
           │     ▼
           │   Transformer Encoder [B, seq_len, d_model]
           │     │
           │     ├─► Duration Predictor → durations [B, seq_len]
           │     │
           │     └─► Length Regulation → expanded [B, T, d_model]
           │           │
           │           ▼
           │         Mel Decoder
           │           │
           │           └─► mel_outputs [B, 80, T]
           │
           ▼
    MODEL OUTPUTS:
    {
      'mel_outputs': [B, 80, T],
      'durations': [B, seq_len],
      'encoder_outputs': [B, seq_len, d_model]
    }
           │
           ▼
┌──────────────────────┐
│     TTSLoss          │
│     forward()        │
└──────────┬───────────┘
           │
           ├─► Mel MSE Loss
           │     MSE(mel_pred, mel_target)
           │
           └─► Duration Loss
                 MSE(dur_pred, dur_target)
           │
           ▼
    LOSS DICTIONARY:
    {
      'total_loss': scalar,
      'mel_loss': scalar,
      'duration_loss': scalar
    }
           │
           ▼
    BACKPROPAGATION
           │
           ▼
    WEIGHT UPDATE
```

---

## 🔍 Current vs. Target Implementation

```
┌─────────────────────────────────────────────────────────────┐
│                    CURRENT STATE (Dummy)                     │
└─────────────────────────────────────────────────────────────┘

train_epoch():
  for batch in loader:
    ╔═══════════════════════════════════════╗
    ║  # TODO: Real forward pass            ║
    ║  dummy_input = torch.randn(...)       ║  ← PLACEHOLDER
    ║  loss = dummy_input.mean()            ║  ← DUMMY LOSS
    ╚═══════════════════════════════════════╝
    loss.backward()
    optimizer.step()

validate():
  for batch in loader:
    ╔═══════════════════════════════════════╗
    ║  loss = torch.tensor(1.0)             ║  ← PLACEHOLDER
    ╚═══════════════════════════════════════╝


┌─────────────────────────────────────────────────────────────┐
│                  TARGET STATE (Real Training)                │
└─────────────────────────────────────────────────────────────┘

train_epoch(model, loader, optimizer, ..., criterion, writer):
  for batch in loader:
    ╔═══════════════════════════════════════╗
    ║  text_ids = batch['text_ids']         ║
    ║  mel_targets = batch['mel']           ║
    ║                                       ║
    ║  outputs = model(                     ║  ← REAL FORWARD
    ║      text_ids=text_ids,               ║
    ║      mel_targets=mel_targets          ║
    ║  )                                    ║
    ║                                       ║
    ║  losses = criterion(outputs, targets) ║  ← REAL LOSS
    ║  loss = losses['total_loss']          ║
    ╚═══════════════════════════════════════╝
    loss.backward()
    optimizer.step()
    
    # Log detailed metrics
    writer.add_scalar('Loss/train_total', loss)
    writer.add_scalar('Loss/train_mel', losses['mel_loss'])

validate(model, loader, criterion):
  for batch in loader:
    ╔═══════════════════════════════════════╗
    ║  outputs = model(text_ids, ...)       ║  ← REAL FORWARD
    ║  losses = criterion(outputs, targets) ║  ← REAL LOSS
    ║  total_loss += losses['total_loss']   ║
    ╚═══════════════════════════════════════╝
```

---

## 📝 Code Change Map

```
src/training/train.py

Line 30:   [ADD] from src.audio import AudioProcessor, collate_fn

Line 140:  [ADD] setup_dataloaders() function definition
             (60 lines of code)

Line 370:  [ADD] criterion = TTSLoss(...)
             (3 lines)

Line 223:  [MODIFY] train_epoch signature
             Add: criterion, writer parameters

Line 233:  [REPLACE] Dummy forward pass → Real forward pass
             (30 lines replaced)

Line 275:  [ADD] TensorBoard logging
             (4 lines)

Line 295:  [REPLACE] validate() → Real validation
             (entire function, ~45 lines)

Line 414:  [MODIFY] train_epoch() call
             Add: criterion, writer arguments

Line 424:  [MODIFY] validate() call
             Add: criterion argument
```

---

## ✅ Verification Flow

```
After Implementation:

1. Syntax Check
   └─► python -m py_compile src/training/train.py
         │
         ├─► ✅ Success → Continue
         └─► ❌ Error → Fix syntax and retry

2. Quick Test Run (1-2 steps)
   └─► python src/training/train.py --config ...
         │
         ├─► Check: Loss values realistic? (100-150)
         ├─► Check: No errors in forward pass?
         ├─► Check: Checkpoint saved?
         │
         ├─► ✅ All pass → Ready for full training
         └─► ❌ Any fail → Debug using guides

3. Full Training
   └─► Monitor:
         ├─► Console logs (decreasing loss)
         ├─► TensorBoard (loss curves)
         ├─► GPU utilization
         └─► Checkpoint quality
```

---

## 🎯 Key Implementation Points

```
Critical Changes Summary:

┌─────────────────────────────────────────────────────────────┐
│ 1. Setup Dataloaders                                         │
│    → Initialize AudioProcessor, create DataLoader            │
│    → Use collate_fn for proper batching                      │
│    → Handle train/val split                                  │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ 2. Initialize Loss Function                                  │
│    → criterion = TTSLoss(mel_weight=1.0, dur_weight=0.1)     │
│    → Pass to train_epoch and validate                        │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ 3. Real Forward Pass                                         │
│    → Load batch: text_ids, mel_targets, lengths             │
│    → Call: outputs = model(text_ids, mel_targets)           │
│    → Compute: losses = criterion(outputs, targets)          │
│    → Use: loss = losses['total_loss']                       │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ 4. Real Validation                                           │
│    → Same as training but with torch.no_grad()              │
│    → Log detailed metrics (total, mel, duration)            │
│    → Track best validation loss                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start Command Sequence

```bash
# 1. Backup
cp src/training/train.py src/training/train.py.backup

# 2. Implement changes (follow guides)
# ... edit train.py ...

# 3. Verify syntax
python -m py_compile src/training/train.py

# 4. Quick test (stop after 2 steps)
python src/training/train.py --config configs/training_config.yaml

# 5. Check output
# Expected: Loss: 142.3456 (not negative dummy values)

# 6. Start full training
python src/training/train.py --config configs/training_config.yaml

# 7. Monitor
tensorboard --logdir logs/
```

---

## 📚 Documentation Navigation

```
docs/
├─► IMPLEMENTATION_SUMMARY.md          ← START HERE (Overview)
│     │
│     └─► Read first to understand status
│
├─► TRAINING_COMPLETION_GUIDE.md       ← MAIN GUIDE
│     │
│     └─► Detailed step-by-step instructions
│
├─► TRAINING_COMPLETION_CHECKLIST.md   ← PROGRESS TRACKER
│     │
│     └─► Use while implementing
│
├─► CODE_CHANGES_REFERENCE.md          ← CODE SNIPPETS
│     │
│     └─► Copy exact code from here
│
└─► IMPLEMENTATION_FLOWCHART.md        ← THIS FILE
      │
      └─► Visual reference
```

---

## 🎓 Remember

**The implementation is straightforward:**
1. You have all the pieces ready
2. Just replace dummy code with real model calls
3. Follow the guides step-by-step
4. Test incrementally
5. Start training!

**Estimated completion time: ~45 minutes**

Good luck! 🚀
