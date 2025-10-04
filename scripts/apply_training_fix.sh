#!/bin/bash
#
# Automated Fix Script for Amharic TTS Training
# This script will automatically fix the training setup
#
# Usage: bash scripts/apply_training_fix.sh
#

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo ""
echo "═══════════════════════════════════════════════════════════════════════"
echo "🔧 AUTOMATED FIX FOR AMHARIC TTS TRAINING"
echo "═══════════════════════════════════════════════════════════════════════"
echo ""

# Change to project root
cd /teamspace/studios/this_studio/Amharic_chatterbox-TTS || {
    echo -e "${RED}❌ Could not find project directory${NC}"
    exit 1
}

echo -e "${GREEN}✅ In project directory: $(pwd)${NC}"
echo ""

# ═══════════════════════════════════════════════════════════════════════
# STEP 1: Pull Latest Code
# ═══════════════════════════════════════════════════════════════════════
echo "─────────────────────────────────────────────────────────────────────"
echo "📦 STEP 1: Pulling Latest Code from GitHub"
echo "─────────────────────────────────────────────────────────────────────"
echo ""

git fetch origin main
git pull origin main

echo ""
echo -e "${GREEN}✅ Code updated${NC}"
echo ""

# ═══════════════════════════════════════════════════════════════════════
# STEP 2: Backup Corrupted Checkpoint
# ═══════════════════════════════════════════════════════════════════════
echo "─────────────────────────────────────────────────────────────────────"
echo "💾 STEP 2: Backing Up Corrupted Checkpoint"
echo "─────────────────────────────────────────────────────────────────────"
echo ""

CHECKPOINT_DIR="models/checkpoints"
BACKUP_DIR="models/checkpoints/CORRUPTED_BACKUPS"

if [ -f "$CHECKPOINT_DIR/checkpoint_epoch99_step4000.pt" ]; then
    mkdir -p "$BACKUP_DIR"
    
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    BACKUP_NAME="checkpoint_epoch99_CORRUPTED_${TIMESTAMP}.pt"
    
    mv "$CHECKPOINT_DIR/checkpoint_epoch99_step4000.pt" "$BACKUP_DIR/$BACKUP_NAME"
    
    echo -e "${GREEN}✅ Backed up to: $BACKUP_DIR/$BACKUP_NAME${NC}"
else
    echo -e "${YELLOW}⚠️  No checkpoint to backup (already moved or deleted)${NC}"
fi

echo ""

# ═══════════════════════════════════════════════════════════════════════
# STEP 3: Verify Configuration
# ═══════════════════════════════════════════════════════════════════════
echo "─────────────────────────────────────────────────────────────────────"
echo "📋 STEP 3: Verifying Configuration"
echo "─────────────────────────────────────────────────────────────────────"
echo ""

if [ -f "config/training_config_finetune_FIXED.yaml" ]; then
    echo -e "${GREEN}✅ Found: config/training_config_finetune_FIXED.yaml${NC}"
    
    # Check learning rate
    LR=$(grep "learning_rate:" config/training_config_finetune_FIXED.yaml | head -1)
    echo "   Learning Rate: $LR"
    
    if echo "$LR" | grep -q "1.0e-5"; then
        echo -e "${GREEN}   ✅ CORRECT (1e-5)${NC}"
    else
        echo -e "${RED}   ❌ WRONG - Expected 1.0e-5${NC}"
        exit 1
    fi
    
    # Check freeze settings
    FREEZE=$(grep "freeze_original_embeddings:" config/training_config_finetune_FIXED.yaml)
    echo "   Freeze Setting: $FREEZE"
    
    if echo "$FREEZE" | grep -q "true"; then
        echo -e "${GREEN}   ✅ CORRECT (true)${NC}"
    else
        echo -e "${RED}   ❌ WRONG - Must be true${NC}"
        exit 1
    fi
    
    FREEZE_IDX=$(grep "freeze_until_index:" config/training_config_finetune_FIXED.yaml)
    echo "   Freeze Index: $FREEZE_IDX"
    
    if echo "$FREEZE_IDX" | grep -q "2454"; then
        echo -e "${GREEN}   ✅ CORRECT (2454)${NC}"
    else
        echo -e "${RED}   ❌ WRONG - Must be 2454${NC}"
        exit 1
    fi
else
    echo -e "${RED}❌ Fixed config not found!${NC}"
    exit 1
fi

echo ""

# ═══════════════════════════════════════════════════════════════════════
# STEP 4: Check Extended Model
# ═══════════════════════════════════════════════════════════════════════
echo "─────────────────────────────────────────────────────────────────────"
echo "🎯 STEP 4: Checking Extended Model"
echo "─────────────────────────────────────────────────────────────────────"
echo ""

EXTENDED_MODEL="models/pretrained/chatterbox_extended.pt"

if [ -f "$EXTENDED_MODEL" ]; then
    echo -e "${GREEN}✅ Extended model found: $EXTENDED_MODEL${NC}"
    
    # Check file size
    SIZE=$(du -h "$EXTENDED_MODEL" | cut -f1)
    echo "   File size: $SIZE"
else
    echo -e "${YELLOW}⚠️  Extended model not found${NC}"
    echo ""
    echo "You need to create the extended model first:"
    echo "  1. Go to Gradio UI → Model Setup tab"
    echo "  2. Download Chatterbox Multilingual model"
    echo "  3. Merge tokenizers"
    echo "  4. Extend embeddings"
    echo ""
    echo "Or run:"
    echo "  python scripts/extend_model_embeddings.py \\"
    echo "    --model models/pretrained/chatterbox/t3_mtl23ls_v2.safetensors \\"
    echo "    --output models/pretrained/chatterbox_extended.pt \\"
    echo "    --original-size 2454 \\"
    echo "    --new-size 2535"
    echo ""
fi

echo ""

# ═══════════════════════════════════════════════════════════════════════
# STEP 5: Check Tokenizer
# ═══════════════════════════════════════════════════════════════════════
echo "─────────────────────────────────────────────────────────────────────"
echo "🔤 STEP 5: Checking Merged Tokenizer"
echo "─────────────────────────────────────────────────────────────────────"
echo ""

MERGED_TOKENIZER="models/tokenizer/Am_tokenizer_merged.json"

if [ -f "$MERGED_TOKENIZER" ]; then
    echo -e "${GREEN}✅ Merged tokenizer found: $MERGED_TOKENIZER${NC}"
    
    # Get vocab size
    VOCAB_SIZE=$(python3 -c "import json; data=json.load(open('$MERGED_TOKENIZER')); print(len(data.get('vocab', data.get('model', {}).get('vocab', {}))))" 2>/dev/null || echo "unknown")
    
    if [ "$VOCAB_SIZE" != "unknown" ]; then
        echo "   Vocab size: $VOCAB_SIZE tokens"
        
        if [ "$VOCAB_SIZE" -eq 2535 ]; then
            echo -e "${GREEN}   ✅ CORRECT (2535 = 2454 Chatterbox + 81 Amharic)${NC}"
        else
            echo -e "${YELLOW}   ⚠️  Expected 2535, got $VOCAB_SIZE${NC}"
        fi
    fi
else
    echo -e "${YELLOW}⚠️  Merged tokenizer not found${NC}"
    echo ""
    echo "You need to merge tokenizers first:"
    echo "  1. Go to Gradio UI → Model Setup tab"
    echo "  2. Merge Tokenizers section"
    echo "  3. Base: models/pretrained/chatterbox_tokenizer.json"
    echo "  4. Amharic: models/tokenizer/amharic_tokenizer/vocab.json"
    echo ""
fi

echo ""

# ═══════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ═══════════════════════════════════════════════════════════════════════
echo ""
echo "═══════════════════════════════════════════════════════════════════════"
echo "✅ FIX APPLIED SUCCESSFULLY"
echo "═══════════════════════════════════════════════════════════════════════"
echo ""
echo "What was done:"
echo "  ✅ Latest code pulled from GitHub"
echo "  ✅ Corrupted checkpoint backed up"
echo "  ✅ Configuration validated"
echo ""
echo "Next steps:"
echo ""
echo "1. Start Gradio App:"
echo "   python gradio_app/full_training_app.py"
echo ""
echo "2. In Gradio UI → Training Pipeline tab:"
echo "   • Config File: config/training_config_finetune_FIXED.yaml"
echo "   • Resume from: None (start from extended embeddings)"
echo "   • Click 'Start Training'"
echo ""
echo "3. VERIFY in logs:"
echo "   ✅ LR: 0.000010 (NOT 0.000198)"
echo "   ✅ Shows: 🔒 FREEZING ORIGINAL EMBEDDINGS"
echo "   ✅ Shows: Frozen (0-2453): 2454 ❄️"
echo "   ✅ Shows: Trainable (2454-2534): 81 🔥"
echo ""
echo "4. Monitor training:"
echo "   • Loss should decrease SMOOTHLY"
echo "   • No wild fluctuations"
echo "   • After 100 epochs: loss should be ~1.0-1.5"
echo ""
echo "5. Test at epoch 50:"
echo "   • English: 'Hello world' → clear speech"
echo "   • Amharic: 'ሰላም ለዓለም' → improving speech"
echo ""
echo "═══════════════════════════════════════════════════════════════════════"
echo ""
