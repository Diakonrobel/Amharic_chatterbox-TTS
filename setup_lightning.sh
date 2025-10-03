#!/bin/bash
# Lightning AI Setup Script for Amharic TTS
# Run this script after cloning the repo on Lightning AI

echo "============================================================"
echo "AMHARIC TTS - Lightning AI Setup"
echo "============================================================"
echo ""

# Check Python version
echo "📌 Checking Python version..."
python --version

# Check CUDA availability
echo ""
echo "🔍 Checking CUDA availability..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
else
    echo "⚠️  CUDA not available. Training will use CPU (slower)."
fi

# Create necessary directories
echo ""
echo "📁 Creating directories..."
mkdir -p data/srt_datasets
mkdir -p data/processed
mkdir -p models/checkpoints
mkdir -p models/tokenizer
mkdir -p models/pretrained
mkdir -p logs
echo "✓ Directories created"

# Install dependencies
echo ""
echo "📦 Installing dependencies..."
pip install --upgrade pip

# Install PyTorch with CUDA support
echo ""
echo "🔥 Installing PyTorch with CUDA..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other requirements
echo ""
echo "📚 Installing other requirements..."
pip install -r requirements.txt

echo ""
echo "✓ All dependencies installed"

# Verify installation
echo ""
echo "🔍 Verifying installation..."
python -c "
import torch
import gradio
print(f'✓ PyTorch version: {torch.__version__}')
print(f'✓ CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'✓ CUDA version: {torch.version.cuda}')
    print(f'✓ GPU: {torch.cuda.get_device_name(0)}')
print(f'✓ Gradio version: {gradio.__version__}')
"

# Test imports
echo ""
echo "🧪 Testing imports..."
python -c "
try:
    from src.g2p.amharic_g2p import AmharicG2P
    from src.training.train import train
    print('✓ All imports successful')
except Exception as e:
    print(f'⚠️  Import error: {e}')
"

# Display setup summary
echo ""
echo "============================================================"
echo "✅ SETUP COMPLETE!"
echo "============================================================"
echo ""
echo "📋 Next Steps:"
echo ""
echo "1. Upload your dataset:"
echo "   - Place in: data/srt_datasets/YOUR_DATASET/"
echo "   - Should include: metadata.csv and wavs/ folder"
echo ""
echo "2. Update configuration:"
echo "   - Edit: config/training_config.yaml"
echo "   - Set dataset_path to your dataset location"
echo ""
echo "3. Start training:"
echo ""
echo "   Option A - Gradio UI (Recommended):"
echo "   $ python gradio_app/full_training_app.py --share"
echo ""
echo "   Option B - Command Line:"
echo "   $ python src/training/train.py --config config/training_config.yaml"
echo ""
echo "4. Monitor training:"
echo "   - Use Gradio UI for real-time updates"
echo "   - Or check logs: tail -f logs/training.log"
echo ""
echo "============================================================"
echo ""
echo "🎉 Ready to train Amharic TTS!"
echo ""
