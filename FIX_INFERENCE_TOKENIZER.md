# 🚨 CRITICAL FIX: Inference Tokenizer Mismatch

## The Problem

Your inference is loading the **WRONG tokenizer**:
- ❌ Loading: Tokenizer with **1000 tokens** (old Amharic-only)
- ✅ Should load: Tokenizer with **2559 tokens** (`am-merged_merged.json`)

This is why ALL languages produce noise - the model and tokenizer don't match!

## The Root Cause

File: `src/inference/inference.py` (lines 171-175)

The tokenizer loading paths are in the wrong order. It's loading an old tokenizer instead of the merged one.

## The Fix

### Option 1: Manual Edit (Fastest)

1. **SSH into Lightning AI:**
   ```bash
   ssh ssh.lightning.ai
   cd /teamspace/studios/this_studio/Amharic_chatterbox-TTS
   ```

2. **Edit the inference file:**
   ```bash
   nano src/inference/inference.py
   ```

3. **Find line 171** (search for "tokenizer_candidates"):
   ```python
   # OLD (WRONG):
   tokenizer_candidates = [
       project_root / 'models' / 'tokenizer' / 'Am_tokenizer_merged.json',
       project_root / 'models' / 'tokenizer' / 'amharic_tokenizer',
       project_root / 'models' / 'tokenizer',
   ]
   ```

4. **Replace with**:
   ```python
   # NEW (CORRECT):
   tokenizer_candidates = [
       # CRITICAL: Load the merged tokenizer with 2559 tokens!
       project_root / 'tokenizers' / 'am-merged_merged.json',  # CORRECT tokenizer
       project_root / 'models' / 'tokenizer' / 'Am_tokenizer_merged.json',
       project_root / 'models' / 'tokenizer' / 'amharic_tokenizer',
       project_root / 'models' / 'tokenizer',
   ]
   ```

5. **Save and exit:**
   - Press `Ctrl+O` to save
   - Press `Enter` to confirm
   - Press `Ctrl+X` to exit

6. **Restart Gradio:**
   - Stop the current Gradio process (Ctrl+C in the terminal running it)
   - Restart: `python gradio_app/full_training_app.py --share`
   - OR just refresh your browser - Gradio might auto-reload

### Option 2: Copy Fixed File

I've already fixed the file locally. Copy it to Lightning AI:

```bash
# From your Windows machine:
scp src/inference/inference.py ssh.lightning.ai:/teamspace/studios/this_studio/Amharic_chatterbox-TTS/src/inference/
```

Then restart Gradio.

## After Applying the Fix

1. **Refresh the Gradio page** in your browser
2. **Test synthesis** with:
   - Amharic: `ሰላም፣ እንደምን ነህ?`
   - English: `Hello, how are you?`

3. **You should see**:
   ```
   ✓ Tokenizer loaded (vocab size: 2559)  ← CORRECT!
   ```

4. **Audio should be clear**, not noise!

## Why This Happened

The Gradio app uses the inference module, which had hardcoded paths that didn't include the correct tokenizer location (`tokenizers/am-merged_merged.json`).

Training worked fine because it uses a different tokenizer loading path.

## Verification

After the fix, the log should show:
```
✓ Tokenizer loaded (vocab size: 2559)  ← Must be 2559, NOT 1000!
```

If you still see `vocab size: 1000`, the fix didn't apply. Double-check the file path.
