# Troubleshooting Model Embedding Extension

## Problem: ❌ Extension Failed

If you're seeing "Extension Failed" when trying to extend the Chatterbox model embeddings, this guide will help you fix it.

## Root Cause

The embedding extension script is looking for specific key names in the model checkpoint (like `text_embedding.weight`), but your Chatterbox model checkpoint may use different key names.

## Solution Steps

### Step 1: Pull Latest Changes on Lightning AI

```bash
cd /teamspace/studios/this_studio/amharic-tts
git pull origin main
```

### Step 2: Install Required Package

```bash
pip install safetensors
```

### Step 3: Inspect Your Model Checkpoint

Run the inspection script to identify the actual key names in your model:

```bash
python scripts/inspect_model_keys.py --model models/pretrained/chatterbox/t3_mtl23ls_v2.safetensors
```

**This will print:**
- All keys containing "text", "token", "embedding", or "vocab"
- All 2D tensors (potential embedding layers) with their shapes
- The first 30 keys in the model

### Step 4: Share the Output

**Copy the entire output** from Step 3 and share it. Look for:

✓ Keys that might contain text embeddings (look for patterns like):
  - `*.text_embedding.weight`
  - `*.token_embedding.weight`
  - `*.embedding.weight`
  - Or any 2D tensor with shape like `[704, 512]` (vocab_size, embedding_dim)

### Step 5: Update the Extension Script

Once we know the correct key names, we'll update `scripts/extend_model_embeddings.py` to search for those specific keys.

## Example Output to Look For

The inspection script will show something like:

```
✓ Found 2 relevant keys:

  model.encoder.text_emb.weight
    Shape: torch.Size([704, 512])
    Dimensions: 2
    → Likely embedding: vocab_size=704, embed_dim=512

  model.decoder.output.weight
    Shape: torch.Size([512, 704])
    Dimensions: 2
```

In this example, `model.encoder.text_emb.weight` would be the text embedding layer we need to extend.

## Common Issues

### Issue: Model file not found

**Error:** `ERROR: Model file not found`

**Solution:** 
1. Make sure you've downloaded the Chatterbox model first (Tab 5, Step 0 in UI)
2. Verify the model path is correct: `models/pretrained/chatterbox/t3_mtl23ls_v2.safetensors`

### Issue: safetensors not installed

**Error:** `⚠ safetensors not installed`

**Solution:**
```bash
pip install safetensors
```

### Issue: No embedding keys found

**Error:** `✗ No keys found containing: text, token, embedding, vocab`

**Solution:**
This means the model checkpoint uses completely different key names. The inspection script will still show all 2D tensors - look through those for shapes that match embedding dimensions (typically `[vocab_size, embedding_dim]`).

## Quick Commands Summary

```bash
# 1. Pull latest code
cd /teamspace/studios/this_studio/amharic-tts
git pull origin main

# 2. Install dependencies
pip install safetensors

# 3. Inspect model
python scripts/inspect_model_keys.py --model models/pretrained/chatterbox/t3_mtl23ls_v2.safetensors

# 4. Copy and share the output!
```

## What Happens Next?

Once you share the inspection output:
1. I'll identify the correct embedding key names
2. Update `extend_model_embeddings.py` to use those keys
3. Push the fix to your repository
4. You can pull and try the extension again - it will work! ✅

---

**Note:** This is a one-time setup issue. Once we fix the key names, the extension will work smoothly for all future uses.
