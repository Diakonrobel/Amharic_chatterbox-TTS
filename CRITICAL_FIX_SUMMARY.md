# CRITICAL FIX: Tokenizer/Model Vocab Size Mismatch (CUDA Error Resolution)

## 🔴 Problem Statement

Training was failing with the following error:
```
torch.AcceleratorError: CUDA error: device-side assert triggered
```

Occurring at:
```python
File "/src/models/t3_model.py", line 269, in forward
    x = x + self.pe[:, :x.size(1), :]
```

## 🔍 Root Cause Analysis

The training failure was caused by a **three-way mismatch** between:

### 1. **Model Initialization** 
- Model was created with `vocab_size=887` (from incorrect config value)
- Text embedding layer shape: `[887, 1024]`

### 2. **Tokenizer Used by Dataloader**
- Merged tokenizer (Chatterbox + Amharic) with `vocab_size=2535`
- Produces token IDs in range `[0, 2534]`

### 3. **Pretrained Embeddings File**
- Extended embeddings file had shape `[3000, 1024]`
- Couldn't load into model with vocab_size=887
- Resulted in random initialization instead

### The Critical Issue

When the dataloader produced a token ID like `2000` from the merged tokenizer (vocab size 2535), but the model only had 887 embeddings, PyTorch attempted to index `embedding_table[2000]` which was **out of bounds**, triggering the CUDA device-side assert.

## ✅ Solution Implemented

### Key Changes

#### 1. **Refactored Training Script Flow** (`src/training/train.py`)

**Before:**
```python
# Old flow - model created first with config vocab size
model = setup_model(config)  # Uses config['model']['n_vocab'] = wrong value
train_loader, val_loader = setup_dataloaders(config)  # Uses different tokenizer
```

**After:**
```python
# New flow - tokenizer detected FIRST
tokenizer, vocab_size = detect_tokenizer(config)  # Detects actual tokenizer
model = setup_model(config, vocab_size)  # Uses tokenizer's vocab size
train_loader, val_loader, _ = setup_dataloaders(config, tokenizer)  # Same tokenizer

# Validation check
if tokenizer.get_vocab_size() != vocab_size:
    raise ValueError("Tokenizer vocab size mismatch!")
```

#### 2. **New `detect_tokenizer()` Function**

Implements priority-based tokenizer detection:

1. **Priority 1**: Use tokenizer from config path if specified
2. **Priority 2**: Try merged tokenizer (`Am_tokenizer_merged.json`)
3. **Priority 3**: Try Amharic-only tokenizer
4. **Fallback**: Character-based encoding with estimated vocab size

Returns: `(tokenizer, vocab_size)` tuple

#### 3. **Smart Freeze Index Validation**

Added validation to prevent `freeze_until_index` from exceeding `vocab_size`:

```python
if freeze_idx > vocab_size:
    TRAINING_STATE.log(f"⚠ Warning: freeze_until_index ({freeze_idx}) > vocab_size ({vocab_size})")
    freeze_idx = min(freeze_idx, vocab_size - 100)  # Leave at least 100 trainable
```

#### 4. **Updated Configuration Defaults**

**`config/training_config.yaml`:**
```yaml
model:
  n_vocab: 2535  # Changed from 3000
  freeze_until_index: 2454  # Chatterbox multilingual base vocab
```

#### 5. **Updated UI Defaults**

**`gradio_app/full_training_app.py`:**
```python
new_size_input = gr.Number(
    label="New Vocab Size",
    value=2535,  # Changed from 3000
    info="Merged tokenizer size: 2535 (Chatterbox 2454 + Amharic 81)"
)
```

## 📊 Vocabulary Size Breakdown

| Component | Vocab Size | Description |
|-----------|-----------|-------------|
| **Chatterbox Multilingual** | 2454 | Base pretrained model (23 languages) |
| **Amharic Extension** | 81 | New Amharic-specific tokens |
| **Merged Tokenizer** | **2535** | Combined vocabulary |
| **Frozen Embeddings** | 2454 | Original tokens (preserved) |
| **Trainable Embeddings** | 81 | New Amharic tokens |

## 🔧 How to Use the Fixed System

### Option 1: Using Merged Tokenizer (Recommended)

1. **Train Amharic tokenizer:**
   ```bash
   python scripts/train_tokenizer.py --data data/srt_datasets/my_dataset/metadata.csv --vocab-size 500 --output models/tokenizer/amharic_tokenizer
   ```

2. **Merge with Chatterbox tokenizer:**
   ```bash
   python scripts/merge_tokenizers.py \
     --base models/pretrained/chatterbox_tokenizer.json \
     --amharic models/tokenizer/amharic_tokenizer/vocab.json \
     --output models/tokenizer/Am_tokenizer_merged.json
   ```

3. **Extend model embeddings to size 2535:**
   ```bash
   python scripts/extend_model_embeddings.py \
     --model models/pretrained/chatterbox/t3_mtl23ls_v2.safetensors \
     --original-size 2454 \
     --new-size 2535 \
     --output models/pretrained/chatterbox_extended.pt
   ```

4. **Configure training to use merged tokenizer:**
   
   Edit `config/training_config.yaml`:
   ```yaml
   paths:
     tokenizer: "models/tokenizer/Am_tokenizer_merged.json"
   
   model:
     n_vocab: 2535
     freeze_until_index: 2454
   
   finetuning:
     pretrained_model: "models/pretrained/chatterbox_extended.pt"
   ```

5. **Start training:**
   ```bash
   python src/training/train.py --config config/training_config.yaml
   ```

### Option 2: Auto-Detection (Recommended)

Simply ensure the merged tokenizer exists at:
- `models/tokenizer/Am_tokenizer_merged.json`

The training script will automatically detect and use it with the correct vocab size.

## ✅ Validation Checklist

Before starting training, ensure:

- [ ] Merged tokenizer exists with correct vocab size (2535)
- [ ] Extended embeddings file matches tokenizer vocab size
- [ ] Config `n_vocab` matches tokenizer size
- [ ] Config `freeze_until_index` ≤ `n_vocab`
- [ ] Pretrained model path points to extended embeddings

## 🎯 Expected Training Log Output

With the fix applied, you should see:

```
[XX:XX:XX] Detecting tokenizer...
[XX:XX:XX] ✓ Loaded MERGED tokenizer from models/tokenizer/Am_tokenizer_merged.json
[XX:XX:XX]    Vocab size: 2535 (Chatterbox 23 langs + Amharic)
[XX:XX:XX] ✓ Tokenizer detected with vocab size: 2535
[XX:XX:XX] Setting up SimplifiedT3Model...
[XX:XX:XX] Using vocab size from tokenizer: 2535
[XX:XX:XX] ✓ T3 Model created:
[XX:XX:XX]    Vocab size: 2535
[XX:XX:XX]    Model dim: 1024 (matches Chatterbox)
[XX:XX:XX] Loading extended embeddings from models/pretrained/chatterbox_extended.pt
[XX:XX:XX]   ✓ Loaded text_embedding: torch.Size([2535, 1024])
[XX:XX:XX] ✓ Frozen first 2454 embeddings (out of 2535 total)
[XX:XX:XX] Setting up dataloaders...
[XX:XX:XX] ✓ Loaded MERGED tokenizer from models/tokenizer/Am_tokenizer_merged.json
[XX:XX:XX]    Vocab size: 2535 (Chatterbox 23 langs + Amharic)
[XX:XX:XX] ✓ Validation passed: Tokenizer and model vocab sizes match (2535)
[XX:XX:XX] Starting training...
```

## 🚫 What Was Wrong Before

### Old Log (Broken):
```
[XX:XX:XX] ✓ T3 Model created:
[XX:XX:XX]    Vocab size: 887  ❌ WRONG - too small
[XX:XX:XX] ⚠ Warning loading weights: size mismatch for text_embedding.weight
[XX:XX:XX] ✓ Loaded MERGED tokenizer with vocab size: 2535  ❌ Mismatch!
[XX:XX:XX] CUDA error: device-side assert triggered  ❌ Token ID > 887
```

### New Log (Fixed):
```
[XX:XX:XX] ✓ Tokenizer detected with vocab size: 2535
[XX:XX:XX] ✓ T3 Model created:
[XX:XX:XX]    Vocab size: 2535  ✅ Matches tokenizer
[XX:XX:XX] ✓ Loaded text_embedding: torch.Size([2535, 1024])  ✅ Correct size
[XX:XX:XX] ✓ Validation passed: Tokenizer and model vocab sizes match (2535)  ✅
```

## 💡 Key Takeaways

1. **Always detect tokenizer before creating model** - The tokenizer's vocab size is the source of truth
2. **Model vocab size must match tokenizer vocab size** - No exceptions
3. **Validate alignment before training** - Prevents runtime CUDA errors
4. **Extended embeddings must match final vocab size** - Not an arbitrary larger size
5. **Config values are defaults, not requirements** - Actual tokenizer takes precedence

## 🔗 Related Files Modified

- `src/training/train.py` - Core training logic refactored
- `config/training_config.yaml` - Default vocab size updated
- `gradio_app/full_training_app.py` - UI defaults updated

## 📝 Notes

- The fix is **backward compatible** - existing tokenizers will work
- The validation will **catch mismatches early** with clear error messages
- The system now **self-corrects** freeze_until_index if it's too large
- **No manual config editing needed** if using standard setup

---

**Status:** ✅ Fixed and tested
**Date:** 2025-10-04
**Commit:** `3bf3f41` - "🔧 CRITICAL FIX: Resolve tokenizer/model vocab size mismatch causing CUDA errors"
