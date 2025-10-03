#!/bin/bash
# Install and Test Chatterbox Package for Amharic Fine-tuning

echo "============================================================"
echo "CHATTERBOX INSTALLATION AND TESTING"
echo "============================================================"

# Step 1: Clone reference repository for study
echo ""
echo "[1/6] Cloning chatterbox-finetuning reference repository..."
cd /tmp
if [ -d "chatterbox-finetuning" ]; then
    rm -rf chatterbox-finetuning
fi
git clone https://github.com/stlohrey/chatterbox-finetuning.git
cd chatterbox-finetuning

echo ""
echo "Repository structure:"
ls -la

echo ""
echo "README contents:"
cat README.md || echo "No README.md found"

echo ""
echo "Looking for key files..."
find . -name "*.py" -o -name "*.yaml" -o -name "*.md"

# Step 2: Try to install Chatterbox package
echo ""
echo "[2/6] Attempting to install Chatterbox package..."
echo "Trying: pip install chatterbox-tts"
pip install chatterbox-tts 2>&1 | tee /tmp/chatterbox_install.log

if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo "⚠ chatterbox-tts not found on PyPI"
    echo ""
    echo "Trying alternative: Installing from ResembleAI GitHub..."
    pip install git+https://github.com/resemble-ai/chatterbox.git 2>&1 | tee -a /tmp/chatterbox_install.log
    
    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        echo "⚠ Could not install from GitHub either"
        echo ""
        echo "This is expected - Chatterbox may not be publicly pip-installable"
        echo "We'll use the downloaded model weights directly"
    fi
fi

# Step 3: Test if Chatterbox is importable
echo ""
echo "[3/6] Testing Chatterbox import..."
python3 << 'PYEOF'
import sys
print("Python version:", sys.version)
print("\nTrying to import chatterbox...")

try:
    import chatterbox
    print("✓ Chatterbox package found!")
    print(f"  Version: {chatterbox.__version__ if hasattr(chatterbox, '__version__') else 'unknown'}")
    print(f"  Location: {chatterbox.__file__}")
    
    # List available components
    print("\nAvailable components:")
    for attr in dir(chatterbox):
        if not attr.startswith('_'):
            print(f"  - {attr}")
            
except ImportError as e:
    print("✗ Chatterbox package not installed")
    print(f"  Error: {e}")
    print("\n→ This is OK - we'll use the model weights directly")
    print("→ Chatterbox models can work without the full package")
PYEOF

# Step 4: Check our downloaded Chatterbox model
echo ""
echo "[4/6] Checking our downloaded Chatterbox model..."
cd ~/Amharic_chatterbox-TTS 2>/dev/null || cd /teamspace/studios/this_studio/Amharic_chatterbox-TTS 2>/dev/null || cd "$(dirname "$0")/.." 2>/dev/null || echo "Warning: Could not determine project directory"

if [ -f "models/pretrained/chatterbox/t3_mtl23ls_v2.safetensors" ]; then
    echo "✓ Chatterbox multilingual model found"
    ls -lh models/pretrained/chatterbox/t3_mtl23ls_v2.safetensors
    
    echo ""
    echo "Model size:"
    du -h models/pretrained/chatterbox/t3_mtl23ls_v2.safetensors
else
    echo "✗ Chatterbox model not found"
    echo "Run: python gradio_app/full_training_app.py"
    echo "Then go to Tab 5 and download the model"
fi

# Step 5: Check extended model
echo ""
echo "[5/6] Checking extended model..."
if [ -f "models/pretrained/chatterbox_extended.pt" ]; then
    echo "✓ Extended model found"
    ls -lh models/pretrained/chatterbox_extended.pt
else
    echo "✗ Extended model not found"
    echo "This will be created during embedding extension step"
fi

# Step 6: Check dataset format
echo ""
echo "[6/6] Checking dataset format..."
if [ -d "data/srt_datasets" ]; then
    echo "✓ SRT datasets directory exists"
    echo ""
    echo "Available datasets:"
    ls -d data/srt_datasets/*/ 2>/dev/null || echo "No datasets yet"
    
    echo ""
    echo "Checking metadata format..."
    for dataset in data/srt_datasets/*/; do
        if [ -f "$dataset/metadata.csv" ]; then
            echo ""
            echo "Dataset: $(basename $dataset)"
            echo "First 3 lines of metadata:"
            head -n 3 "$dataset/metadata.csv"
            echo "Format: Should be 'filename|text' or 'filename|text|speaker'"
        fi
    done
else
    echo "✗ No SRT datasets found"
    echo "Import datasets using the Gradio UI (Tab 2)"
fi

# Summary
echo ""
echo "============================================================"
echo "INSTALLATION SUMMARY"
echo "============================================================"

echo ""
echo "Chatterbox Package: $(python3 -c 'import chatterbox' 2>&1 && echo '✓ Installed' || echo '✗ Not installed (OK - using model weights directly)')"
echo "Pretrained Model: $([ -f 'models/pretrained/chatterbox/t3_mtl23ls_v2.safetensors' ] && echo '✓ Downloaded' || echo '✗ Not found')"
echo "Extended Model: $([ -f 'models/pretrained/chatterbox_extended.pt' ] && echo '✓ Created' || echo '✗ Not created yet')"
echo "Datasets: $([ -d 'data/srt_datasets' ] && echo '✓ Directory exists' || echo '✗ Not found')"

echo ""
echo "Next Steps:"
echo "1. If Chatterbox model not found: Download via Gradio UI (Tab 5)"
echo "2. If datasets not found: Import via Gradio UI (Tab 2)"
echo "3. If extended model not found: Extend embeddings via Gradio UI (Tab 5)"
echo "4. Then: Ready for fine-tuning!"

echo ""
echo "============================================================"
