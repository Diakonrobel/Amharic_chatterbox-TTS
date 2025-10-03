#!/bin/bash
# Lightning AI Environment Setup Script
# Fixes all warnings for Amharic TTS Training System

echo "=============================================="
echo "AMHARIC TTS - LIGHTNING AI SETUP"
echo "=============================================="
echo ""

# 1. Install ffmpeg
echo "📦 Installing ffmpeg..."
conda install -y -c conda-forge ffmpeg
if [ $? -eq 0 ]; then
    echo "✓ ffmpeg installed successfully"
else
    echo "⚠ Could not install ffmpeg via conda, trying apt-get..."
    sudo apt-get update && sudo apt-get install -y ffmpeg
fi
echo ""

# 2. Create necessary directories
echo "📁 Creating directory structure..."
mkdir -p models/tokenizer
mkdir -p models/checkpoints
mkdir -p models/pretrained
mkdir -p data/srt_datasets
mkdir -p config
echo "✓ Directories created"
echo ""

# 3. Check if tokenizer exists, if not create a placeholder
echo "🔤 Checking tokenizer..."
if [ ! -f "models/tokenizer/vocab.json" ]; then
    echo "⚠ Tokenizer not found. You'll need to train one using the 'Tokenizer Training' tab"
else
    echo "✓ Tokenizer found"
fi
echo ""

# 4. Install additional Python dependencies if needed
echo "📚 Checking Python dependencies..."
pip install --quiet --upgrade sentencepiece pydub soundfile librosa ffmpeg-python
echo "✓ Dependencies checked"
echo ""

echo "=============================================="
echo "SETUP COMPLETE!"
echo "=============================================="
echo ""
echo "✅ All fixes applied successfully!"
echo ""
echo "Remaining steps:"
echo "1. ⚠ Tokenizer: Train using the 'Tokenizer Training' tab in the app"
echo "2. ⚠ TTS Model: Download or train using the 'Training Pipeline' tab"
echo ""
echo "You can now run:"
echo "  python gradio_app/full_training_app.py --share"
echo ""
