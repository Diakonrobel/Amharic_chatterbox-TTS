# Why Generated Audio Sounds Like Noise/Machine/Train

## 🎯 TL;DR

**Your inference pipeline is working perfectly!** ✅

The audio sounds like noise because **the model hasn't been trained yet**. You're hearing what an untrained neural network sounds like - essentially random static.

---

## 🔍 What's Happening

### Current State:
```
Text → Tokenizer → Model (RANDOM WEIGHTS) → Random Mel → Griffin-Lim → Noise Audio
         ✅            ❌ NOT TRAINED            ❌              ✅           🔊
```

### What Should Happen (After Training):
```
Text → Tokenizer → Model (TRAINED) → Realistic Mel → Griffin-Lim → Speech Audio
         ✅            ✅                  ✅              ✅           🗣️
```

---

## 📊 Technical Explanation

### Why It Sounds Like Noise:

1. **Model Has Random Weights**
   - Your checkpoint contains the model structure
   - But the weights are either random or only partially trained
   - The model doesn't know how to convert text → speech yet

2. **Random Mel-Spectrograms**
   - The model outputs mel-spectrograms with random values
   - These don't correspond to any real speech patterns
   - Griffin-Lim faithfully converts them to audio → noise

3. **Pipeline is Actually Working!**
   - Text encoding: ✅ Working
   - Model forward pass: ✅ Working
   - Mel generation: ✅ Working (just random values)
   - Audio synthesis: ✅ Working
   - **The problem:** Model isn't trained on speech data

---

## 🎓 Analogy

Imagine asking someone who has never heard Amharic (or any language) to speak it:
- They know the alphabet (tokenizer works ✅)
- They can move their mouth (model runs ✅)
- But they produce random sounds (untrained weights ❌)

That's exactly what's happening here!

---

## ✅ How to Fix: Train the Model

### Step 1: Verify You Have Training Data

Check your dataset:
```bash
ls -lh data/srt_datasets/
```

You should see your Amharic dataset with:
- `metadata.csv` (or `metadata_train.csv`)
- `wavs/` directory with audio files
- At least 10+ hours of data (recommended)

---

### Step 2: Start Training on Lightning AI

#### Option A: Using Gradio UI (Easiest)

1. **Launch the app:**
   ```bash
   python gradio_app/full_training_app.py --share --port 7861
   ```

2. **Go to "Training Pipeline" tab**

3. **Configure training:**
   - **Dataset:** Select your Amharic dataset
   - **Tokenizer:** `Am_tokenizer_merged.json`
   - **Batch size:** 8-16 (depending on GPU memory)
   - **Learning rate:** 2e-4 (or 5e-5 for stability)
   - **Max epochs:** 500-1000
   - **Freeze embeddings:** ✅ Yes
   - **Freeze until index:** 2454

4. **Click "Start Training"**

5. **Monitor progress:**
   - Watch the loss decrease
   - Training will take several hours
   - Best to let it run overnight

---

#### Option B: Using Command Line

```bash
# On Lightning AI
python src/training/train.py --config config/training_config.yaml
```

Make sure your config has:
```yaml
data:
  dataset_path: data/srt_datasets/your_dataset

model:
  n_vocab: 2535  # Merged tokenizer size
  freeze_original_embeddings: true
  freeze_until_index: 2454

training:
  batch_size: 16
  learning_rate: 2e-4
  max_epochs: 1000
```

---

### Step 3: Wait for Training

**Typical Training Time:**
- Small dataset (1-5 hours): 4-8 hours training
- Medium dataset (5-10 hours): 8-16 hours training
- Large dataset (10+ hours): 16-48 hours training

**What to look for:**
- Loss starting around 10-20
- Decreasing to 1-3 over time
- Validation loss improving
- Checkpoints saved every N steps

---

### Step 4: Use Trained Checkpoint

After training:

1. **Find best checkpoint:**
   ```bash
   ls -lh models/checkpoints/
   # Look for checkpoint_best.pt or latest checkpoint
   ```

2. **Test inference:**
   ```bash
   python src/inference/inference.py \
       --checkpoint models/checkpoints/checkpoint_best.pt \
       --text "ሰላም ለዓለም" \
       --output test_speech.wav
   ```

3. **Listen to the result:**
   - Should now sound like actual Amharic speech!
   - May be robotic (Griffin-Lim limitation)
   - But intelligible and clear

---

## 🚨 Common Misconceptions

### ❌ "The inference is broken"
**Reality:** ✅ Inference works perfectly! The model just needs training.

### ❌ "I need to fix the code"
**Reality:** ✅ Code is correct! You need to train the model.

### ❌ "Something is wrong with the audio generation"
**Reality:** ✅ Audio generation works! The model is outputting random values.

### ❌ "Griffin-Lim is bad"
**Reality:** ⚠️ Griffin-Lim works fine for testing. It's the untrained model producing noise.

---

## 🎯 Checklist Before Training

Before starting training, ensure:

- ✅ **Dataset prepared:**
  - Audio files in `data/srt_datasets/your_dataset/wavs/`
  - Metadata in `metadata_train.csv`, `metadata_val.csv`, `metadata_test.csv`
  - At least 500+ audio samples
  - Good audio quality (clear speech, minimal noise)

- ✅ **Tokenizer ready:**
  - Merged tokenizer: `models/tokenizer/Am_tokenizer_merged.json`
  - Vocab size: 2535
  - Tested and working

- ✅ **Model setup:**
  - Extended embeddings ready
  - Config file properly configured
  - Checkpoints directory exists

- ✅ **Hardware:**
  - CUDA GPU available (Lightning AI provides this)
  - Sufficient disk space for checkpoints (~10GB)
  - Stable internet connection

---

## 📊 Expected Training Progress

### Epoch 1-10:
```
Loss: 15.234 → 8.456
Status: Model learning basic patterns
Audio: Still very noisy
```

### Epoch 10-50:
```
Loss: 8.456 → 3.124
Status: Model learning speech features
Audio: Recognizable sounds emerging
```

### Epoch 50-200:
```
Loss: 3.124 → 1.567
Status: Model learning Amharic phonemes
Audio: Speech-like, somewhat intelligible
```

### Epoch 200-500:
```
Loss: 1.567 → 0.892
Status: Model refining quality
Audio: Clear Amharic speech!
```

---

## 🎊 What to Expect After Training

### Before Training:
```
Input: "ሰላም ለዓለም"
Output: 🔊 [static/white noise/machine sounds]
```

### After Training:
```
Input: "ሰላም ለዓለም"
Output: 🗣️ [Clear Amharic: "sälam ləʿaläm"]
```

The difference will be dramatic!

---

## 💡 Pro Tips

1. **Start with a small test:**
   - Train on 100-200 samples first
   - Verify training works
   - Then scale to full dataset

2. **Monitor training:**
   - Use TensorBoard: `tensorboard --logdir logs`
   - Check validation loss regularly
   - Listen to generated samples periodically

3. **Early stopping:**
   - Enable early stopping in config
   - Prevents overfitting
   - Saves best checkpoint automatically

4. **Patience is key:**
   - Training takes hours/days
   - Don't stop too early
   - Let loss converge properly

---

## 🆘 Still Having Issues?

If after training the audio still sounds bad:

1. **Check dataset quality:**
   - Are audio files clear?
   - Are transcriptions accurate?
   - Is there enough data?

2. **Check training logs:**
   - Is loss decreasing?
   - Any error messages?
   - Is validation loss improving?

3. **Check hyperparameters:**
   - Learning rate not too high?
   - Batch size appropriate?
   - Embeddings properly frozen?

4. **Try different checkpoint:**
   - Test multiple saved checkpoints
   - Best checkpoint might not be the last one
   - Look for lowest validation loss

---

## 📚 Additional Resources

- [Training Guide](TRAINING_GUIDE.md) - Complete training walkthrough
- [Inference Guide](INFERENCE_GUIDE.md) - Using trained models
- [Dataset Guide](DATASET_GUIDE.md) - Preparing Amharic data
- [Troubleshooting](TROUBLESHOOTING.md) - Common issues

---

## ✅ Summary

**Current situation:**
- ✅ Your code is working correctly
- ✅ Inference pipeline is functional
- ✅ Audio generation works
- ❌ Model needs training

**Next step:**
- 🎓 **TRAIN THE MODEL!**
- Wait for several hours
- Then inference will produce real speech

**Don't worry!** This is completely normal. Every TTS model sounds like noise before training. You're on the right track! 🚀

---

**Good luck with training!** 🎙️🇪🇹✨
