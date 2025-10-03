# Chatterbox Fine-tuning Reference

Based on: https://github.com/stlohrey/chatterbox-finetuning

## Repository Structure (Expected)

```
chatterbox-finetuning/
├── README.md
├── requirements.txt
├── finetune.py              # Main training script
├── prepare_data.py          # Data preparation
├── config.yaml              # Training configuration
└── data/
    └── [your_dataset]/
        ├── metadata.csv     # Format: filename|text|speaker
        └── wavs/
            └── *.wav
```

## Key Steps for Chatterbox Fine-tuning

### 1. Install Chatterbox Package

```bash
# Official Chatterbox from ResembleAI
pip install chatterbox-tts
# OR from source
git clone https://github.com/resemble-ai/chatterbox.git
cd chatterbox && pip install -e .
```

### 2. Prepare Dataset Format

Chatterbox expects:
```
metadata.csv format:
filename|text|speaker_id
audio001.wav|Hello world|speaker_01
audio002.wav|This is a test|speaker_01
```

### 3. Download Pretrained Model

```python
from chatterbox import download_pretrained
model_path = download_pretrained("multilingual")  # or "english"
```

### 4. Fine-tune on Your Data

```python
from chatterbox import Chatterbox, ChatterboxConfig

# Load config
config = ChatterboxConfig.from_pretrained("multilingual")
config.train_data = "data/your_dataset"
config.vocab_size = 3000  # Extended for Amharic

# Load model
model = Chatterbox.from_pretrained("multilingual", config=config)

# Fine-tune
model.train(
    train_data="data/your_dataset",
    num_epochs=100,
    batch_size=16,
    learning_rate=2e-4
)
```

## Critical Implementation Notes

### For Amharic Fine-tuning:

1. **Tokenizer Extension**
   - Extend Chatterbox tokenizer with Amharic characters
   - Merge vocabularies (base + Amharic)
   - Update model embeddings accordingly

2. **Audio Format**
   - 22050 Hz sample rate
   - Mono channel
   - WAV format recommended

3. **Dataset Size**
   - Minimum: 30 minutes (for testing)
   - Recommended: 3-5 hours (for good quality)
   - Optimal: 10+ hours (for production)

4. **Training Strategy**
   - Start with frozen embeddings for base language
   - Unfreeze gradually
   - Low learning rate (1e-4 to 2e-4)
   - Save checkpoints frequently

## Next Steps for Our Implementation

1. ✅ Install official Chatterbox package
2. ✅ Convert our SRT datasets to Chatterbox format
3. ✅ Load pretrained Chatterbox model
4. ✅ Extend embeddings for Amharic
5. ✅ Fine-tune with our prepared data
6. ✅ Test synthesis on Amharic text

## Commands to Run (Lightning AI)

```bash
# 1. Clone reference repo for study
git clone https://github.com/stlohrey/chatterbox-finetuning.git
cd chatterbox-finetuning
cat README.md  # Read the actual approach

# 2. Install Chatterbox
pip install chatterbox-tts

# 3. Return to our project
cd /teamspace/studios/this_studio/amharic-tts

# 4. Test Chatterbox installation
python -c "import chatterbox; print(chatterbox.__version__)"

# 5. Run our adapted fine-tuning script
python scripts/finetune_chatterbox_amharic.py
```

## Reference Links

- Official Chatterbox: https://github.com/resemble-ai/chatterbox
- Fine-tuning Example: https://github.com/stlohrey/chatterbox-finetuning
- ResembleAI HuggingFace: https://huggingface.co/ResembleAI/chatterbox
- Documentation: (Check official repo for docs)

---

**Status:** Ready to implement practical fine-tuning approach
**Next:** Install Chatterbox package and adapt to Amharic
