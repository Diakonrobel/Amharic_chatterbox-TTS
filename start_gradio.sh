#!/bin/bash
#
# Smart Gradio Startup Script
# Automatically fixes common issues before starting
#

set -e

echo ""
echo "═══════════════════════════════════════════════════════════════════════"
echo "🚀 AMHARIC TTS - SMART STARTUP"
echo "═══════════════════════════════════════════════════════════════════════"
echo ""

# Navigate to project root
cd /teamspace/studios/this_studio/Amharic_chatterbox-TTS 2>/dev/null || cd "$(dirname "$0")"

echo "📂 Project directory: $(pwd)"
echo ""

# Step 1: Auto-detect and sync vocab sizes
echo "─────────────────────────────────────────────────────────────────────"
echo "🔧 Step 1: Syncing Vocab Sizes"
echo "─────────────────────────────────────────────────────────────────────"
echo ""

if [ -f "scripts/auto_detect_vocab_size.py" ]; then
    python scripts/auto_detect_vocab_size.py || {
        echo "⚠️ Vocab sync failed, but continuing..."
    }
else
    echo "⚠️ auto_detect_vocab_size.py not found, skipping..."
fi

echo ""

# Step 2: Check for required files
echo "─────────────────────────────────────────────────────────────────────"
echo "✅ Step 2: Checking Required Files"
echo "─────────────────────────────────────────────────────────────────────"
echo ""

# Check tokenizer
if ls models/tokenizer/*.json >/dev/null 2>&1; then
    echo "✓ Tokenizer found"
else
    echo "⚠️ No tokenizer found in models/tokenizer/"
fi

# Check extended model
if ls models/pretrained/chatterbox_extended*.pt >/dev/null 2>&1; then
    echo "✓ Extended model found"
    ls -lh models/pretrained/chatterbox_extended*.pt | awk '{print "  ", $9, "-", $5}'
else
    echo "⚠️ No extended model found"
fi

# Check checkpoints
if ls models/checkpoints/*.pt >/dev/null 2>&1 2>&1; then
    checkpoint_count=$(ls models/checkpoints/*.pt 2>/dev/null | wc -l)
    echo "✓ Found $checkpoint_count checkpoints"
else
    echo "ℹ️ No checkpoints yet (normal for first run)"
fi

echo ""

# Step 3: Start Gradio
echo "─────────────────────────────────────────────────────────────────────"
echo "🎨 Step 3: Starting Gradio Interface"
echo "─────────────────────────────────────────────────────────────────────"
echo ""

# Check if --share flag should be used
SHARE_FLAG=""
if [ "$1" == "--share" ] || [ "$1" == "-s" ]; then
    SHARE_FLAG="--share"
    echo "🌐 Starting with public URL (--share mode)"
else
    echo "🏠 Starting in local mode"
    echo "   Tip: Use './start_gradio.sh --share' for public URL"
fi

echo ""
echo "═══════════════════════════════════════════════════════════════════════"
echo "✅ PRE-FLIGHT CHECKS COMPLETE"
echo "═══════════════════════════════════════════════════════════════════════"
echo ""
echo "Starting Gradio..."
echo ""

# Start Gradio
python gradio_app/full_training_app.py $SHARE_FLAG

# Cleanup on exit
echo ""
echo "═══════════════════════════════════════════════════════════════════════"
echo "👋 Gradio Stopped"
echo "═══════════════════════════════════════════════════════════════════════"
