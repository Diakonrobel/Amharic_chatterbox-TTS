#!/bin/bash
# Amharic TTS Setup Script for Linux/Ubuntu
# Run this script to set up the development environment

set -e  # Exit on error

echo "==============================================="
echo "Amharic TTS Development Environment Setup"
echo "==============================================="
echo ""

# Check Python version
echo "[1/6] Checking Python installation..."
if command -v python3.10 &> /dev/null; then
    PYTHON_CMD=python3.10
elif command -v python3 &> /dev/null; then
    PYTHON_CMD=python3
else
    echo "✗ Python 3.10+ not found. Please install Python 3.10 or later."
    echo "  Ubuntu/Debian: sudo apt install python3.10 python3.10-venv"
    exit 1
fi

PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | awk '{print $2}')
echo "✓ Python found: $PYTHON_VERSION"

# Create virtual environment
echo ""
echo "[2/6] Creating virtual environment..."
$PYTHON_CMD -m venv venv
if [ $? -eq 0 ]; then
    echo "✓ Virtual environment created"
else
    echo "✗ Failed to create virtual environment"
    exit 1
fi

# Activate virtual environment
echo ""
echo "[3/6] Activating virtual environment..."
source venv/bin/activate
echo "✓ Virtual environment activated"

# Upgrade pip
echo ""
echo "[4/6] Upgrading pip..."
pip install --upgrade pip setuptools wheel
echo "✓ Pip upgraded"

# Install PyTorch
echo ""
echo "[5/6] Installing PyTorch..."
echo "   This may take a while..."

# Detect CUDA availability
if command -v nvidia-smi &> /dev/null; then
    echo "   CUDA detected, installing PyTorch with CUDA support..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
else
    echo "   CUDA not detected, installing CPU-only PyTorch..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

if [ $? -eq 0 ]; then
    echo "✓ PyTorch installed"
else
    echo "⚠ PyTorch installation had issues, continuing..."
fi

# Install requirements
echo ""
echo "[6/6] Installing dependencies..."
echo "   This may take several minutes..."
pip install -r requirements.txt

if [ $? -eq 0 ]; then
    echo "✓ Dependencies installed"
else
    echo "⚠ Some dependencies may have failed"
fi

echo ""
echo "==============================================="
echo "Setup Complete!"
echo "==============================================="
echo ""
echo "Next steps:"
echo "1. Activate virtual environment: source venv/bin/activate"
echo "2. Prepare your Amharic dataset"
echo "3. Follow the README for training instructions"
echo ""
