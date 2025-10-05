# 🚀 Apply Fix on Lightning AI via Git Pull

## What Was Fixed

✅ Pushed to GitHub:
- `src/inference/inference.py` - Corrected tokenizer path (2559 tokens)
- `test_synthesis_cli.py` - CLI synthesis testing tool
- `FIX_INFERENCE_TOKENIZER.md` - Documentation

## Steps to Apply on Lightning AI

### 1. Open Lightning AI Terminal

Go to your Lightning AI studio and open the terminal.

### 2. Navigate to Project Directory

```bash
cd /teamspace/studios/this_studio/Amharic_chatterbox-TTS
```

### 3. Pull the Latest Changes

```bash
git pull origin main
```

You should see:
```
Updating c48d514..08d41d3
Fast-forward
 FIX_INFERENCE_TOKENIZER.md | 102 +++++++++++++++++++++++++++++
 src/inference/inference.py |   3 +
 test_synthesis_cli.py      | 277 ++++++++++++++++++++++++++++++++++++++++++
 3 files changed, 381 insertions(+)
```

### 4. Verify the Fix

Check that the tokenizer path is correct:

```bash
grep -A 3 "tokenizer_candidates = \[" src/inference/inference.py
```

You should see:
```python
tokenizer_candidates = [
    # CRITICAL: Load the merged tokenizer with 2559 tokens!
    project_root / 'tokenizers' / 'am-merged_merged.json',  # CORRECT tokenizer
```

### 5. Restart Gradio

**Option A: If Gradio is running in a terminal:**
- Press `Ctrl+C` to stop
- Restart: `python gradio_app/full_training_app.py --share`

**Option B: If you're not sure:**
- Find the process: `ps aux | grep gradio`
- Kill it: `kill <PID>`
- Restart: `python gradio_app/full_training_app.py --share`

**Option C: Easiest - Just refresh your browser**
- Sometimes Gradio auto-reloads changed files
- If not, use Options A or B

### 6. Test the Fix

1. **Refresh the Gradio page** in your browser

2. **Check the logs** when you try synthesis - you should see:
   ```
   ✓ Tokenizer loaded (vocab size: 2559)  ← MUST BE 2559!
   ```

3. **Test Amharic:**
   - Enter: `ሰላም፣ እንደምን ነህ?`
   - Click Synthesize
   - Listen → Should be CLEAR speech, not noise!

4. **Test English:**
   - Enter: `Hello, how are you?`
   - Click Synthesize
   - Listen → Should be CLEAR English!

## If Git Pull Fails

If you get merge conflicts or errors:

```bash
# Save any local changes
git stash

# Pull again
git pull origin main

# Reapply your changes if needed
git stash pop
```

Or force pull (ONLY if you don't have uncommitted changes you want to keep):

```bash
git fetch origin
git reset --hard origin/main
```

## Verification Checklist

After applying the fix, verify:

- ✅ Git pull successful
- ✅ File `src/inference/inference.py` updated
- ✅ Gradio restarted
- ✅ Log shows: `✓ Tokenizer loaded (vocab size: 2559)`
- ✅ Audio is clear, not noise
- ✅ Both Amharic and English work

## Expected Results

**Before fix:**
```
✓ Tokenizer loaded (vocab size: 1000)  ← WRONG!
Audio: Noise for all languages
```

**After fix:**
```
✓ Tokenizer loaded (vocab size: 2559)  ← CORRECT!
Audio: Clear speech for all languages
```

## Training Not Affected

Your training will continue uninterrupted! This fix only affects the **inference/synthesis** side. The training loop uses a different tokenizer loading mechanism that was already correct.

**Current training status: Epoch 75+, Loss ~3-10, Validation ~9.95 - Excellent!** 🎉
