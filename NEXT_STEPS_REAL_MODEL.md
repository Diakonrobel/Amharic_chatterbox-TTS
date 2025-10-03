# Next Steps: Implementing Real Model Training

## 🎯 Current Status

### ✅ What's Complete:
1. **Training Infrastructure** - Verified with 1000+ epochs of dummy training
2. **Model Extension** - Embeddings extended from 2454 → 3000 tokens
3. **Pretrained Model** - Chatterbox multilingual downloaded
4. **Dataset Tools** - SRT import working perfectly
5. **Tokenizer System** - Amharic tokenizer trained
6. **Audio Processing** - Module created (`src/audio/audio_processing.py`)
7. **T3 Model Architecture** - Simplified implementation (`src/models/t3_model.py`)
8. **Documentation** - Complete guides created

### ⏳ What's Next:
1. **Update training script** to use real model instead of dummy
2. **Update dataset class** to load real audio with mel-spectrograms
3. **Test with small dataset** (100 steps)
4. **Full training** on Amharic data

---

## 📋 Implementation Plan (Step by Step)

### Step 1: Update Training Script ✅ (Next)

**File:** `src/training/train.py`

**Changes needed:**
1. Replace `DummyModel` with `SimplifiedT3Model`
2. Add audio processor initialization
3. Update dataset to load real audio
4. Use real loss functions (mel + duration)

### Step 2: Update Dataset Class

**File:** `src/training/train.py` (SimpleAmharicDataset)

**Changes:**
```python
# Current (dummy):
def __getitem__(self, idx):
    return {'text': sample['text'], 'audio_path': sample['audio']}

# New (real):
def __getitem__(self, idx):
    # Load audio
    audio_path = self.data_dir / 'wavs' / sample['audio']
    _, mel = self.audio_processor.process_audio_file(str(audio_path))
    
    # Tokenize text
    text_ids = self.tokenizer.encode(sample['text'])
    
    return {
        'text_ids': text_ids,
        'mel': mel,
        'audio_path': str(audio_path)
    }
```

### Step 3: Update Training Loop

**Changes:**
```python
# Use real model
from src.models.t3_model import SimplifiedT3Model, TTSLoss

model = SimplifiedT3Model(
    vocab_size=config['model']['n_vocab'],
    d_model=512,
    n_mels=80
)

# Load extended embeddings
if config['finetuning']['enabled']:
    model.load_pretrained_weights(pretrained_path)

# Use real loss
criterion = TTSLoss()

# Forward pass
outputs = model(
    text_ids=batch['text_ids'],
    text_lengths=batch['text_lengths'],
    mel_targets=batch['mel']
)
losses = criterion(outputs, batch)
loss = losses['total_loss']
```

---

## 🚀 Quick Start Commands (Lightning AI)

### 1. Pull Latest Code
```bash
cd /teamspace/studios/this_studio/amharic-tts
git pull origin main
```

### 2. Verify Setup
```bash
bash scripts/install_and_test_chatterbox.sh
```

### 3. Check What You Have
```bash
# Check datasets
ls -la data/srt_datasets/

# Check models
ls -la models/pretrained/

# Check extended model
ls -lh models/pretrained/chatterbox_extended.pt
```

### 4. Test Audio Processing
```bash
python << 'EOF'
from src.audio import AudioProcessor
import os

processor = AudioProcessor()

# Find first audio file
for root, dirs, files in os.walk('data/srt_datasets'):
    for file in files:
        if file.endswith('.wav'):
            audio_path = os.path.join(root, file)
            print(f"Testing audio: {audio_path}")
            
            audio, mel = processor.process_audio_file(audio_path)
            print(f"✓ Audio shape: {audio.shape}")
            print(f"✓ Mel shape: {mel.shape}")
            break
    break
EOF
```

### 5. Test Model Creation
```bash
python << 'EOF'
from src.models.t3_model import SimplifiedT3Model
import torch

model = SimplifiedT3Model(vocab_size=3000, d_model=512, n_mels=80)
print(f"✓ Model created")
print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

# Test forward pass
batch_size = 2
seq_len = 10
mel_len = 100

text_ids = torch.randint(0, 3000, (batch_size, seq_len))
mel_targets = torch.randn(batch_size, 80, mel_len)

outputs = model(text_ids, mel_targets=mel_targets)
print(f"✓ Forward pass successful")
print(f"  Mel output shape: {outputs['mel_outputs'].shape}")
EOF
```

---

## 📊 Expected Outcomes

### After Step 1-3 (Real Model Integration):
- ✅ Model loads extended embeddings
- ✅ Audio data loads correctly
- ✅ Mel-spectrograms extracted
- ✅ Forward pass works
- ✅ Loss computed correctly

### After Test Training (100 steps):
```
Epoch 1 | Step 1   | Mel Loss: 12.456 | Duration Loss: 3.123
Epoch 1 | Step 10  | Mel Loss: 10.234 | Duration Loss: 2.789
Epoch 1 | Step 50  | Mel Loss: 7.891  | Duration Loss: 2.134
Epoch 1 | Step 100 | Mel Loss: 6.234  | Duration Loss: 1.891
```

**Good signs:**
- ✅ Losses are positive (not negative like dummy!)
- ✅ Losses decrease
- ✅ No crashes or OOM errors
- ✅ Reasonable values (not NaN)

---

## 🔧 Files That Need Updates

### Priority 1 (Critical):
1. `src/training/train.py`
   - Replace DummyModel → SimplifiedT3Model
   - Update SimpleAmharicDataset
   - Update train_epoch function
   - Add real loss computation

### Priority 2 (Important):
2. `src/training/train.py`
   - Add tokenizer loading
   - Add audio processor
   - Update collate_fn
   - Add proper validation

### Priority 3 (Nice to have):
3. Create `scripts/test_real_model.py`
   - Quick test script
   - Verify all components work
   - Before full training

---

## ⚠️ Potential Issues and Solutions

### Issue 1: Tokenizer Not Found
**Solution:** Make sure you've trained an Amharic tokenizer via Gradio UI (Tab 4)

### Issue 2: Audio Files Not Found
**Solution:** Import dataset via Gradio UI (Tab 2)

### Issue 3: Extended Model Not Found
**Solution:** Extend embeddings via Gradio UI (Tab 5, Step 2)

### Issue 4: OOM During Training
**Solution:**
```yaml
# In config/training_config.yaml
data:
  batch_size: 4  # Reduce from 16
training:
  use_amp: true  # Enable mixed precision
```

---

## 📝 Progress Checklist

- [x] Training infrastructure built
- [x] Dummy model tested (1000 epochs successful)
- [x] Model embeddings extended
- [x] Audio processing module created
- [x] T3 model architecture created
- [x] Documentation completed
- [ ] Training script updated for real model ← **YOU ARE HERE**
- [ ] Dataset class updated for audio loading
- [ ] Test training (100 steps)
- [ ] Full Amharic training
- [ ] Quality evaluation
- [ ] Production deployment

---

## 🎯 Immediate Next Task

**Update** `src/training/train.py` to use the real model.

I can help you with this now! Should I:
1. **Update the training script step by step** (safer, more controlled)
2. **Create a new test script first** (verify components work)
3. **Update everything at once** (faster, but riskier)

Which approach do you prefer? 🤔
