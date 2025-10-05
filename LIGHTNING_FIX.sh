#!/bin/bash
# Quick fix script for Lightning AI
# Run this on Lightning AI: bash LIGHTNING_FIX.sh

echo "🔧 Fixing Amharic TTS on Lightning AI..."
echo ""

# Navigate to repo
cd ~/Amharic_chatterbox-TTS || cd Amharic_chatterbox-TTS || { echo "❌ Cannot find repo"; exit 1; }

echo "📍 Current directory: $(pwd)"
echo ""

# Check current git status
echo "📊 Git status:"
git status --short
echo ""

# Stash any local changes
echo "💾 Stashing local changes (if any)..."
git stash
echo ""

# Pull latest fixes
echo "⬇️  Pulling latest fixes from GitHub..."
git pull origin main
echo ""

# Verify the fix is applied
echo "✅ Checking if fix is applied..."
if grep -q "n_mels=data_config.get" train_enhanced.py; then
    echo "✅ Fix applied successfully!"
else
    echo "⚠️  Fix not found, manually patching..."
    
    # Manual patch if needed
    sed -i 's/n_mel_channels=data_config/n_mels=data_config/g' train_enhanced.py
    
    # Verify again
    if grep -q "n_mels=data_config.get" train_enhanced.py; then
        echo "✅ Manual patch successful!"
    else
        echo "❌ Manual patch failed - check file manually"
    fi
fi

echo ""
echo "🎉 Done! Now run:"
echo "   python train_enhanced.py --config config/training_config.yaml --device cpu"
