# ⚡ Lightning AI - Quick Fix for UNK Token Issues

## 🔍 Problem Summary

Your diagnostic output shows:
- ✅ Amharic data is perfect (525 entries with proper Ethiopic characters)
- ✅ G2P system working excellently 
- ✅ Tokenization logic is sound (0% UNK ratio when properly loaded)
- ❌ Missing `vocab.json` and `sentencepiece.model` in `models/tokenizer/`
- ⚠️ Vocab size mismatch: Config expects 3000, merged vocab has 887

## 🚀 Quick Fix (2 Commands)

### Step 1: Pull Latest Fixes
```bash
cd ~/Amharic_chatterbox-TTS
git pull origin main
```

### Step 2: Run Automated Setup
```bash
python setup_tokenizer_lightning.py --dataset merged_3 --vocab-size 1000
```

This will:
1. ✅ Install sentencepiece
2. ✅ Train Amharic tokenizer on your 525-entry dataset
3. ✅ Create vocab.json and sentencepiece.model
4. ✅ Update config with correct vocab sizes
5. ✅ Test tokenization with 0% UNK ratio

**Expected output:**
```
⚡ LIGHTNING AI - AMHARIC TOKENIZER SETUP
==================================================

📦 Step 1: Installing dependencies...
✓ sentencepiece installed

🔤 Step 2: Training Amharic tokenizer on 'merged_3' dataset...
Training on: data/srt_datasets/merged_3/metadata.csv
Vocab size: 1000
Output: models/tokenizer/amharic_tokenizer

This may take 2-5 minutes...
[Training progress...]

✓ Tokenizer trained successfully!

📁 Verifying tokenizer files:
  ✓ models/tokenizer/amharic_tokenizer/vocab.json
  ✓ models/tokenizer/amharic_tokenizer/sentencepiece.model
  ✓ models/tokenizer/amharic_tokenizer/config.json

⚙️ Step 3: Checking vocabulary size...
Actual Amharic vocab size: 1000
Merged vocab size: 887
Config n_vocab: 3000

⚠️ Config mismatch detected!
  Updating config: 3000 → 887
✓ Config updated!

🧪 Step 4: Testing tokenization...
  ✓ 'ሰላም ለዓለም...' - UNK ratio: 0.0%
  ✓ 'አዲስ አበባ የኢትዮጵያ ዋና ከተማ ናት...' - UNK ratio: 0.0%

🎉 SUCCESS! Tokenizer is working perfectly!

You can now start training:
  python src/training/train.py --config config/training_config.yaml

==================================================
✅ SETUP COMPLETE!
==================================================
```

## 🎯 Alternative: Manual Fix

If the automated script has issues, follow these steps:

### 1. Install Dependencies
```bash
pip install sentencepiece
```

### 2. Train Tokenizer Manually
```bash
python -c "
from src.tokenizer.amharic_tokenizer import train_amharic_tokenizer

tokenizer = train_amharic_tokenizer(
    data_file='data/srt_datasets/merged_3/metadata.csv',
    output_dir='models/tokenizer/amharic_tokenizer',
    vocab_size=1000
)

print('✓ Tokenizer training complete!')
"
```

### 3. Verify Files Created
```bash
ls -la models/tokenizer/amharic_tokenizer/
```

Should show:
- `vocab.json` - Character vocabulary
- `sentencepiece.model` - Trained BPE model
- `config.json` - Tokenizer configuration

### 4. Update Config
Edit `config/training_config.yaml`:
```yaml
model:
  n_vocab: 887  # Update to match merged_vocab.json size
  freeze_until_index: 2454
```

### 5. Test Before Training
```bash
python test_unk_fix.py
```

Should show:
```
SUMMARY: 4/4 tests passed
🎉 All tests passed! Your setup should work for training.
```

## 📊 Understanding the Diagnostics

### What Your Results Mean:

**Good News:**
- ✅ **Amharic Characters**: All Ethiopic (U+1200-U+137F) detected correctly
- ✅ **G2P Conversion**: Amharic → IPA phonemes working perfectly
  - Example: `'ሰላም ለዓለም' → 'səlamɨ ləዓləmɨ'`
- ✅ **Training Data**: 525 quality entries in merged_3 dataset
- ✅ **Tokenization**: 0% UNK ratio when tokenizer properly loaded

**Issues Fixed:**
- ❌ **Missing Files**: vocab.json and sentencepiece.model needed creation
- ⚠️ **Config Mismatch**: n_vocab updated from 3000 → 887

## 🎓 Why This Happened

The diagnostic found that while your tokenizer **logic** works perfectly (0% UNK ratio), the actual tokenizer **files** were missing from the `models/tokenizer/` directory. This is likely because:

1. The tokenizer was never trained on your merged_3 dataset
2. Or files were in a different location (amharic_tokenizer subdirectory)
3. The config expected 3000 tokens but merged vocab only had 887

The fix trains a fresh tokenizer on your **actual Amharic data** (525 entries from merged_3), ensuring perfect compatibility.

## 🚀 After Setup: Start Training

Once setup is complete with 0% UNK ratio:

```bash
# Start training with proper tokenizer
python src/training/train.py --config config/training_config.yaml
```

You should now see:
- ✅ No more `<UNK>` warnings
- ✅ All Amharic text properly tokenized
- ✅ Training progressing normally

## 🔧 Troubleshooting

### If tokenizer training fails:

**Check dataset path:**
```bash
ls -la data/srt_datasets/merged_3/metadata.csv
head -5 data/srt_datasets/merged_3/metadata.csv
```

**Check Amharic content:**
```bash
python -c "
with open('data/srt_datasets/merged_3/metadata.csv', 'r', encoding='utf-8') as f:
    line = f.readline()
    parts = line.split('|')
    if len(parts) >= 2:
        text = parts[1]
        amharic_chars = sum(1 for c in text if 0x1200 <= ord(c) <= 0x137F)
        print(f'Sample: {text[:50]}...')
        print(f'Amharic chars: {amharic_chars}')
"
```

### If still getting UNK tokens during training:

1. Increase vocab size:
   ```bash
   python setup_tokenizer_lightning.py --dataset merged_3 --vocab-size 2000
   ```

2. Check if using the right tokenizer path in training code

3. Verify phoneme conversion is enabled:
   ```python
   # In training code, should use:
   tokens = tokenizer.encode(text, use_phonemes=True)
   ```

## 📞 Support

If issues persist:
1. Run: `python test_unk_fix.py` and share output
2. Check: `models/tokenizer/amharic_tokenizer/` for files
3. Verify: Your training data has Amharic content

## ✅ Success Indicators

You'll know everything is working when:
- ✅ `test_unk_fix.py` shows 4/4 tests passed
- ✅ Tokenizer files exist in `models/tokenizer/amharic_tokenizer/`
- ✅ Config n_vocab matches merged_vocab.json size
- ✅ Test tokenization shows 0% UNK ratio
- ✅ Training starts without UNK warnings

---

**Ready to train!** 🚀🇪🇹
