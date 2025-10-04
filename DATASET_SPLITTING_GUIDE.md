# Dataset Splitting Guide - Train/Val/Test

## 🚨 **CRITICAL: Why You MUST Split Your Dataset**

### Your Current Problem:

```
⚠ No separate validation metadata, using training data
✓ Loaded 525 samples from data/srt_datasets/merged_3/metadata.csv
✓ Loaded 525 samples from data/srt_datasets/merged_3/metadata.csv  ← SAME FILE!
```

**This means your model is "cheating"!** It's like letting a student take a test using the same questions they studied.

## ❌ **What's Wrong**

### Without Proper Splitting:

1. **Model memorizes training data** instead of learning patterns
2. **Validation loss is misleading** - looks better than it really is
3. **Early stopping doesn't work** - can't detect real overfitting
4. **No way to evaluate final performance** - no test set
5. **Will fail on new data** - poor generalization

### Example:
```
Training Loss: 1.2 ✅ Looks good
Validation Loss: 1.3 ✅ Looks good  
Real-world performance: 5.8 ❌ TERRIBLE!
```

Why? Because validation used training data!

## ✅ **The Correct Way**

### Three Separate Sets:

```
Total Data: 525 samples
    ↓
┌───────────────────────────────────┐
│ Training Set (80% = 420 samples)  │ ← Model learns from this
├───────────────────────────────────┤
│ Validation Set (15% = 79 samples) │ ← Monitor during training
├───────────────────────────────────┤
│ Test Set (5% = 26 samples)        │ ← Final evaluation only
└───────────────────────────────────┘
```

### Purpose of Each Set:

| Set | Purpose | When Used | Can Model See? |
|-----|---------|-----------|----------------|
| **Train** | Learning | Every epoch | ✅ Yes (training) |
| **Validation** | Monitoring | Every N steps | ✅ Yes (evaluation only) |
| **Test** | Final eval | After training complete | ❌ Never during training |

## 🛠️ **How to Split Your Dataset**

### On Lightning AI:

```bash
cd ~/Amharic_chatterbox-TTS

# Pull the splitting script
git pull origin main

# Split your current dataset
python scripts/split_dataset.py \
  --dataset data/srt_datasets/merged_3 \
  --train 0.80 \
  --val 0.15 \
  --test 0.05 \
  --backup
```

### Expected Output:

```
============================================================
DATASET SPLITTING
============================================================
Dataset: data/srt_datasets/merged_3
Split ratios: Train=80%, Val=15%, Test=5%
Random seed: 42

✓ Backed up original to: metadata_original.csv
Loading metadata from: metadata.csv
✓ Loaded 525 samples

Splitting data...

============================================================
SPLITTING COMPLETE
============================================================
Train set: 420 samples (80.0%)
  → data/srt_datasets/merged_3/metadata_train.csv

Val set:   79 samples (15.0%)
  → data/srt_datasets/merged_3/metadata_val.csv

Test set:  26 samples (5.0%)
  → data/srt_datasets/merged_3/metadata_test.csv
```

### What Gets Created:

```
data/srt_datasets/merged_3/
├── metadata.csv                 ← Original (backed up)
├── metadata_original.csv        ← Backup
├── metadata_train.csv          ← 420 samples (80%)
├── metadata_val.csv            ← 79 samples (15%)
├── metadata_test.csv           ← 26 samples (5%)
└── wavs/
    └── ...
```

## 🎯 **Custom Split Ratios**

### For Different Dataset Sizes:

#### Very Small Dataset (<500 samples):
```bash
python scripts/split_dataset.py \
  --dataset data/srt_datasets/my_dataset \
  --train 0.85 \
  --val 0.10 \
  --test 0.05
```

**Why**: Need more training data, can afford smaller validation

#### Medium Dataset (500-2000 samples):
```bash
python scripts/split_dataset.py \
  --dataset data/srt_datasets/my_dataset \
  --train 0.80 \
  --val 0.15 \
  --test 0.05
```

**Why**: Balanced approach (recommended)

#### Large Dataset (>2000 samples):
```bash
python scripts/split_dataset.py \
  --dataset data/srt_datasets/my_dataset \
  --train 0.70 \
  --val 0.20 \
  --test 0.10
```

**Why**: Can afford larger val/test sets for better evaluation

## 🔄 **Training Script Behavior**

### After Splitting:

The training script will automatically:

1. **Look for `metadata_train.csv` first**
2. **Look for `metadata_val.csv` for validation**
3. **Fall back to `metadata.csv` if splits don't exist** (with warning)

### New Training Logs:

**With proper splits**:
```
Setting up dataloaders...
✓ Loaded 420 samples from data/srt_datasets/merged_3/metadata_train.csv
✓ Loaded 79 samples from data/srt_datasets/merged_3/metadata_val.csv
✓ Train samples: 420
✓ Val samples: 79
```

**Without splits** (your current state):
```
Setting up dataloaders...
⚠️ WARNING: No validation set found!
⚠️ Training and validation are using the SAME data
⚠️ This means:
   - Early stopping won't work correctly
   - Model will appear better than it really is
   - Can't detect true overfitting

ℹ️ To fix this, run:
   python scripts/split_dataset.py --dataset data/srt_datasets/merged_3 --backup
```

## 📊 **Impact on Training**

### Before Split (Current):
```
Epoch 10:
  Train Loss: 2.3
  Val Loss: 2.2  ← Too optimistic! (using train data)
  Real performance: Unknown ❌
```

### After Split (Correct):
```
Epoch 10:
  Train Loss: 2.3
  Val Loss: 2.8  ← Realistic! (using unseen data)
  Real performance: ~2.8 ✅ Accurate estimate
```

## 🎓 **Understanding the Difference**

### Scenario: Student Taking a Test

#### Without Proper Split (Current):
```
Study: Questions 1-100
Test: Questions 1-100  ← Same questions!
Result: 95% ✅ Looks great
Real knowledge: Maybe only 60% ❌
```

#### With Proper Split:
```
Study: Questions 1-80
Test: Questions 81-95  ← Different questions!
Result: 70% ← Honest assessment
Real knowledge: ~70% ✅ Accurate
```

## 🛑 **Action Required!**

### For Your Current Training (already running):

1. **Let it finish** - It's already at Epoch 13, loss 1.66
2. **Understand the limitation** - Final loss may not reflect real performance
3. **Test on new data** to see real quality

### For Future Training:

1. **Split your dataset NOW**:
   ```bash
   python scripts/split_dataset.py --dataset data/srt_datasets/merged_3 --backup
   ```

2. **Restart training** with proper splits:
   ```bash
   # Stop current training
   # Start new training (will auto-detect splits)
   python src/training/train.py --config config/training_config.yaml
   ```

3. **Watch for proper logs**:
   ```
   ✓ Loaded 420 samples from metadata_train.csv  ← Different!
   ✓ Loaded 79 samples from metadata_val.csv     ← Different!
   ```

## 📝 **Script Options**

### All Available Options:

```bash
python scripts/split_dataset.py \
  --dataset <path>        # Required: Dataset directory
  --train <ratio>         # Optional: Train ratio (default: 0.80)
  --val <ratio>           # Optional: Val ratio (default: 0.15)
  --test <ratio>          # Optional: Test ratio (default: 0.05)
  --seed <int>            # Optional: Random seed (default: 42)
  --backup                # Optional: Backup original metadata.csv
```

### Examples:

```bash
# Default split (80/15/5)
python scripts/split_dataset.py --dataset data/srt_datasets/merged_3 --backup

# Custom split for small dataset
python scripts/split_dataset.py --dataset data/srt_datasets/small --train 0.85 --val 0.10 --test 0.05

# Different random seed
python scripts/split_dataset.py --dataset data/srt_datasets/merged_3 --seed 123

# No backup (use with caution)
python scripts/split_dataset.py --dataset data/srt_datasets/merged_3
```

## 🧪 **Using the Test Set**

### When to Use Test Set:

❌ **NEVER during training**
❌ **NEVER for hyperparameter tuning**
❌ **NEVER for model selection**

✅ **ONLY after training is completely done**
✅ **To report final performance**
✅ **To compare different models**

### How to Evaluate on Test Set:

```python
# After training completes, load best model
checkpoint = torch.load('models/checkpoints/checkpoint_best.pt')
model.load_state_dict(checkpoint['model_state_dict'])

# Load test dataset
test_dataset = SimpleAmharicDataset(
    'data/srt_datasets/merged_3/metadata_test.csv',
    data_dir,
    tokenizer=tokenizer
)

# Evaluate
test_loss = evaluate(model, test_dataset)
print(f"Final Test Loss: {test_loss:.4f}")
```

## 📊 **Expected Results**

### Normal Behavior:

```
Train Loss:    1.2  ← Lowest (model sees this data)
Val Loss:      1.5  ← Middle (monitors progress)
Test Loss:     1.6  ← Highest (truly unseen)
```

**Why test loss is higher**: Model never saw this data, most realistic performance

### Red Flags:

```
Train Loss:    1.2
Val Loss:      2.5  ← Much higher! Overfitting badly
Test Loss:     2.8  ← Even worse
```

**Action**: Stop training earlier, increase regularization

## 🎯 **Summary**

| Without Split | With Split |
|---------------|------------|
| ❌ Model cheats | ✅ Honest evaluation |
| ❌ Misleading metrics | ✅ Accurate metrics |
| ❌ Early stopping broken | ✅ Early stopping works |
| ❌ Unknown real performance | ✅ Known performance |
| ❌ Overfitting undetected | ✅ Overfitting detected |

## 🚀 **Quick Start**

On Lightning AI, run these commands NOW:

```bash
cd ~/Amharic_chatterbox-TTS
git pull origin main
python scripts/split_dataset.py --dataset data/srt_datasets/merged_3 --backup
```

Then restart training to use proper splits!

---

**Remember**: A model that scores 95% on training data but 60% on new data is useless. Always use proper train/val/test splits! 🎯
