# 🔧 Tokenizer Training Issue - FIXED

## ❌ The Problem

**Error:** `KeyError: 'text'`

**Root Cause:** 
The SRT Dataset Builder creates `metadata.csv` files in **LJSpeech format** (pipe-delimited):
```
filename|text|normalized_text
audio_001|ሰላም ለዓለም|ሰላም ለዓለም
audio_002|አማርኛ|አማርኛ
```

But the tokenizer training code was trying to read it as a **standard CSV** (comma-delimited) and expected a column named `text`.

---

## ✅ The Fix

Updated `src/tokenizer/amharic_tokenizer.py` to:

1. **Try pipe-delimited format first** (LJSpeech standard)
2. **Fallback to comma-delimited** if that fails
3. **Auto-detect text columns** with better error messages
4. **Validate data** before training

---

## 🚀 How to Apply the Fix on Lightning AI

### Step 1: Pull the Latest Code

```bash
cd ~/Amharic_chatterbox-TTS
git pull origin main
```

### Step 2: Restart Your App

```bash
# Stop the current app (Ctrl+C)
python gradio_app/full_training_app.py --share
```

### Step 3: Try Tokenizer Training Again

1. Open the Gradio URL
2. Go to **Tab 4: Tokenizer Training**
3. Enter your dataset path: `data/srt_datasets/YOUR_MERGED_DATASET/metadata.csv`
4. Set vocabulary size: 500-2000
5. Click "🚀 Train Tokenizer"

---

## 📊 Expected Output (Success)

```
Loading data...
✓ Loaded LJSpeech format (pipe-delimited)
Training tokenizer on 1234 samples...
Sample texts:
  1. ሰላም ለዓለም...
  2. አዲስ አበባ የኢትዮጵያ ዋና ከተማ ናት...
  3. እንኳን ደህና መጡ...
Building vocabulary...
✓ Vocabulary size: 850
Training SentencePiece model...
✓ Trained SentencePiece model
  Vocabulary size: 500
✓ Tokenizer saved to: models/tokenizer/amharic_tokenizer
```

---

## 🔍 Understanding Your Dataset Format

### LJSpeech Format (What SRT Builder Creates):

```
filename|text|normalized_text
audio_001|ሰላም ለዓለም|ሰላም ለዓለም
audio_002|አማርኛ|አማርኛ
```

**Features:**
- Pipe-delimited (`|`)
- No header row
- Three columns: filename, text, normalized_text

### Standard CSV Format (Also Supported):

```
filename,text,normalized_text
audio_001,ሰላም ለዓለም,ሰላም ለዓለም
audio_002,አማርኛ,አማርኛ
```

**The fix now handles BOTH formats automatically!**

---

## 🛠️ What Changed in the Code

### Before (Line 241-244):
```python
# Load data
print("Loading data...")
df = pd.read_csv(data_file)
texts = df['text'].tolist()  # ❌ KeyError if pipe-delimited!
```

### After (Line 241-267):
```python
# Load data
print("Loading data...")

# Try to load with different formats
try:
    # First try pipe-delimited (LJSpeech format)
    df = pd.read_csv(data_file, sep='|', header=None, 
                     names=['filename', 'text', 'normalized_text'])
    texts = df['text'].tolist()
    print(f"✓ Loaded LJSpeech format (pipe-delimited)")
except:
    try:
        # Try comma-delimited with 'text' column
        df = pd.read_csv(data_file)
        if 'text' in df.columns:
            texts = df['text'].tolist()
        # ... more fallbacks ...
    except Exception as e:
        raise ValueError(f"Could not load data file. Error: {e}")

# Validate data
if not texts:
    raise ValueError("No text data found in the file!")

# Remove empty texts
texts = [t for t in texts if t and str(t).strip()]
```

---

## 🎯 Dataset Path Examples

### For Individual Dataset:
```
data/srt_datasets/my_dataset/metadata.csv
```

### For Merged Dataset:
```
data/srt_datasets/merged_amharic/metadata.csv
```

### Verify Your Path:
```bash
# On Lightning AI, check if file exists
ls -la data/srt_datasets/YOUR_DATASET/metadata.csv

# Preview first few lines
head -5 data/srt_datasets/YOUR_DATASET/metadata.csv
```

Should show:
```
audio_001|ሰላም ለዓለም|ሰላም ለዓለም
audio_002|አማርኛ|አማርኛ
```

---

## 📝 Training Tips

### Vocabulary Size Guidelines:

- **Small dataset** (<1 hour): 300-500
- **Medium dataset** (1-5 hours): 500-1000  
- **Large dataset** (>5 hours): 1000-2000

### Common Issues:

1. **"File not found"**
   - Check path: Must be `metadata.csv` not `dataset_info.json`
   - Full path: `data/srt_datasets/DATASET_NAME/metadata.csv`

2. **"No text data found"**
   - Check if metadata.csv has content
   - Run: `wc -l data/srt_datasets/YOUR_DATASET/metadata.csv`

3. **"All text entries are empty"**
   - Dataset might be corrupted
   - Re-import or re-merge your datasets

---

## ✅ Verification

After pulling the fix, you should be able to train on:
- ✅ Individual SRT datasets
- ✅ Merged datasets
- ✅ Pipe-delimited format (LJSpeech)
- ✅ Comma-delimited format (CSV)

---

## 🎉 Next Steps After Tokenizer Training

Once tokenizer training succeeds:

1. **Check output:**
   ```bash
   ls -la models/tokenizer/amharic_tokenizer/
   ```
   Should show:
   - `sentencepiece.model`
   - `vocab.json`
   - `config.json`

2. **Start Model Training:**
   - Go to Tab 6: "🎓 Training Pipeline"
   - Configure parameters
   - Click "▶️ Start Training"

---

## 📞 Still Having Issues?

If you still see the error after pulling:

1. **Check you pulled correctly:**
   ```bash
   git log --oneline -1
   # Should show: "914aaa5 Fix tokenizer training to handle pipe-delimited..."
   ```

2. **Verify your dataset:**
   ```bash
   head -5 data/srt_datasets/YOUR_DATASET/metadata.csv
   ```

3. **Check file permissions:**
   ```bash
   ls -la data/srt_datasets/YOUR_DATASET/metadata.csv
   ```

Happy training! 🚀
