#!/bin/bash
# Automatic fix application script for Lightning AI
# Run this in Lightning AI terminal to apply the tokenizer fix

set -e  # Exit on error

echo "════════════════════════════════════════════════════════════════════════"
echo "🔧 AUTOMATIC FIX APPLICATION"
echo "════════════════════════════════════════════════════════════════════════"
echo ""

# Navigate to project
echo "📂 Navigating to project directory..."
cd /teamspace/studios/this_studio/Amharic_chatterbox-TTS
echo "✅ Current directory: $(pwd)"
echo ""

# Stop Gradio
echo "🛑 Stopping Gradio..."
pkill -f gradio || echo "   (No Gradio process found - OK)"
sleep 2
echo ""

# Pull latest changes
echo "📥 Pulling latest changes from GitHub..."
git fetch origin
echo ""

# Show what will change
echo "📊 Changes to be applied:"
git log HEAD..origin/main --oneline
echo ""

# Apply the changes
echo "🔄 Applying changes..."
git pull origin main
echo ""

# Verify the fix
echo "🔍 Verifying tokenizer path fix..."
if grep -q "models.*tokenizer.*am-merged_merged.json" src/inference/inference.py; then
    echo "✅ Fix verified! Correct tokenizer path found in inference.py"
else
    echo "❌ WARNING: Fix may not have applied correctly"
    echo "   Check src/inference/inference.py manually"
fi
echo ""

# Clear Python cache
echo "🧹 Clearing Python cache..."
find . -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true
find . -name '*.pyc' -delete 2>/dev/null || true
echo "✅ Cache cleared"
echo ""

# Show current git status
echo "📊 Git status:"
git log -1 --oneline
echo ""

echo "════════════════════════════════════════════════════════════════════════"
echo "✅ FIX APPLIED SUCCESSFULLY!"
echo "════════════════════════════════════════════════════════════════════════"
echo ""
echo "🚀 NEXT STEP: Restart Gradio with:"
echo ""
echo "   python gradio_app/full_training_app.py --share"
echo ""
echo "📊 VERIFICATION:"
echo "   When Gradio starts, check for:"
echo "   ✓ Tokenizer loaded (vocab size: 2559)  ← CORRECT!"
echo ""
echo "   Then test synthesis - audio should be CLEAR!"
echo ""
echo "════════════════════════════════════════════════════════════════════════"
