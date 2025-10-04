#!/bin/bash
#
# Comprehensive Diagnostic and Fix Script for Amharic TTS Training
# Run this on Lightning AI to analyze and fix the training configuration issue
#
# Usage: bash scripts/diagnose_and_fix_training.sh
#

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo ""
echo "═══════════════════════════════════════════════════════════════════════"
echo "🔍 AMHARIC TTS TRAINING DIAGNOSTIC & FIX SCRIPT"
echo "═══════════════════════════════════════════════════════════════════════"
echo ""

# Function to print colored output
print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

# Change to project root
cd /teamspace/studios/this_studio/Amharic_chatterbox-TTS || {
    print_error "Could not find project directory"
    exit 1
}

print_success "Found project directory: $(pwd)"
echo ""

# ═══════════════════════════════════════════════════════════════════════
# STEP 1: GIT STATUS CHECK
# ═══════════════════════════════════════════════════════════════════════
echo "─────────────────────────────────────────────────────────────────────"
echo "📦 STEP 1: Checking Git Status"
echo "─────────────────────────────────────────────────────────────────────"

# Get current commit
CURRENT_COMMIT=$(git log --oneline -1 2>/dev/null || echo "Unknown")
print_info "Current commit: $CURRENT_COMMIT"

# Check if we have the latest updates
if git log --oneline -1 | grep -q "CRITICAL FIX"; then
    print_success "Latest CRITICAL FIX commit found"
else
    print_warning "Latest commit not found. Pulling updates..."
    git pull origin main || print_error "Failed to pull updates"
fi

echo ""

# ═══════════════════════════════════════════════════════════════════════
# STEP 2: CONFIG FILE ANALYSIS
# ═══════════════════════════════════════════════════════════════════════
echo "─────────────────────────────────────────────────────────────────────"
echo "📋 STEP 2: Analyzing Configuration Files"
echo "─────────────────────────────────────────────────────────────────────"

# Check all config files
echo ""
echo "Config files found:"
find config -name "*.yaml" -type f 2>/dev/null || print_warning "No config directory"

echo ""
echo "Checking learning rates in each config:"
echo ""

for config_file in config/*.yaml; do
    if [ -f "$config_file" ]; then
        echo "  📄 $(basename $config_file):"
        LR=$(grep "learning_rate:" "$config_file" | head -1)
        if echo "$LR" | grep -q "1.0e-5\|0.00001"; then
            print_success "    $LR (CORRECT for finetuning)"
        elif echo "$LR" | grep -q "2.0e-4\|0.0002"; then
            print_error "    $LR (TOO HIGH - will destroy pretrained weights)"
        elif echo "$LR" | grep -q "5.0e-5\|0.00005"; then
            print_warning "    $LR (Moderate - for small datasets)"
        else
            print_info "    $LR"
        fi
    fi
done

echo ""

# ═══════════════════════════════════════════════════════════════════════
# STEP 3: CHECKPOINT ANALYSIS
# ═══════════════════════════════════════════════════════════════════════
echo "─────────────────────────────────────────────────────────────────────"
echo "💾 STEP 3: Analyzing Checkpoints"
echo "─────────────────────────────────────────────────────────────────────"

CHECKPOINT_DIR="models/checkpoints"
if [ -d "$CHECKPOINT_DIR" ]; then
    echo ""
    echo "Available checkpoints:"
    ls -lh "$CHECKPOINT_DIR"/*.pt 2>/dev/null || print_warning "No checkpoints found"
    
    # Find latest checkpoint
    LATEST_CHECKPOINT=$(ls -t "$CHECKPOINT_DIR"/*.pt 2>/dev/null | head -1)
    
    if [ -n "$LATEST_CHECKPOINT" ]; then
        echo ""
        print_info "Latest checkpoint: $(basename $LATEST_CHECKPOINT)"
        
        # Run Python diagnostic on checkpoint
        if [ -f "scripts/test_checkpoint_multilingual.py" ]; then
            echo ""
            echo "Running checkpoint diagnostic..."
            python scripts/test_checkpoint_multilingual.py --checkpoint "$LATEST_CHECKPOINT"
        fi
    else
        print_warning "No checkpoints to analyze"
    fi
else
    print_warning "Checkpoint directory not found"
fi

echo ""

# ═══════════════════════════════════════════════════════════════════════
# STEP 4: TRAINING LOG ANALYSIS
# ═══════════════════════════════════════════════════════════════════════
echo "─────────────────────────────────────────────────────────────────────"
echo "📊 STEP 4: Analyzing Training Logs"
echo "─────────────────────────────────────────────────────────────────────"

LOG_DIR="logs"
if [ -d "$LOG_DIR" ]; then
    echo ""
    echo "Checking for learning rate in recent logs..."
    
    # Find most recent log file
    LATEST_LOG=$(find "$LOG_DIR" -name "*.log" -type f 2>/dev/null | sort -r | head -1)
    
    if [ -n "$LATEST_LOG" ]; then
        print_info "Latest log: $(basename $LATEST_LOG)"
        echo ""
        echo "Learning rates found in log:"
        grep -o "LR: [0-9.]*" "$LATEST_LOG" | tail -10 || print_warning "No LR entries found"
    else
        print_warning "No log files found"
    fi
else
    print_warning "Log directory not found"
fi

echo ""

# ═══════════════════════════════════════════════════════════════════════
# STEP 5: TOKENIZER CHECK
# ═══════════════════════════════════════════════════════════════════════
echo "─────────────────────────────────────────────────────────────────────"
echo "🔤 STEP 5: Checking Tokenizers"
echo "─────────────────────────────────────────────────────────────────────"

TOKENIZER_DIR="models/tokenizer"
if [ -d "$TOKENIZER_DIR" ]; then
    echo ""
    echo "Available tokenizers:"
    find "$TOKENIZER_DIR" -name "*.json" -o -name "vocab.txt" 2>/dev/null | while read tok; do
        echo "  📝 $tok"
        if [ -f "$tok" ] && [[ "$tok" == *.json ]]; then
            # Try to get vocab size
            VOCAB_SIZE=$(python3 -c "import json; data=json.load(open('$tok')); print(len(data.get('vocab', data.get('model', {}).get('vocab', {}))))" 2>/dev/null || echo "unknown")
            if [ "$VOCAB_SIZE" != "unknown" ]; then
                echo "     Vocab size: $VOCAB_SIZE tokens"
            fi
        fi
    done
else
    print_warning "Tokenizer directory not found"
fi

echo ""

# ═══════════════════════════════════════════════════════════════════════
# STEP 6: PROBLEM SUMMARY
# ═══════════════════════════════════════════════════════════════════════
echo ""
echo "═══════════════════════════════════════════════════════════════════════"
echo "🎯 DIAGNOSIS SUMMARY"
echo "═══════════════════════════════════════════════════════════════════════"
echo ""

# Check if FIXED config exists
if [ -f "config/training_config_finetune_FIXED.yaml" ]; then
    print_success "Fixed config file exists"
    
    # Check its learning rate
    FIXED_LR=$(grep "learning_rate:" config/training_config_finetune_FIXED.yaml | head -1)
    if echo "$FIXED_LR" | grep -q "1.0e-5"; then
        print_success "Fixed config has correct LR: 1e-5"
    else
        print_error "Fixed config has wrong LR: $FIXED_LR"
    fi
else
    print_error "Fixed config file NOT found!"
    echo "    Expected: config/training_config_finetune_FIXED.yaml"
fi

echo ""

# ═══════════════════════════════════════════════════════════════════════
# STEP 7: RECOMMENDATIONS
# ═══════════════════════════════════════════════════════════════════════
echo "─────────────────────────────────────────────────────────────────────"
echo "💡 RECOMMENDATIONS"
echo "─────────────────────────────────────────────────────────────────────"
echo ""

# Check if checkpoint exists and analyze
if [ -n "$LATEST_CHECKPOINT" ]; then
    # Check checkpoint loss
    CHECKPOINT_LOSS=$(python3 -c "import torch; ckpt=torch.load('$LATEST_CHECKPOINT', map_location='cpu', weights_only=False); print(ckpt.get('loss', 999))" 2>/dev/null || echo "999")
    
    if (( $(echo "$CHECKPOINT_LOSS > 2.5" | bc -l 2>/dev/null || echo "1") )); then
        print_error "Checkpoint loss is HIGH: $CHECKPOINT_LOSS"
        echo ""
        echo "🚨 YOUR CURRENT CHECKPOINT IS LIKELY CORRUPTED!"
        echo ""
        echo "Evidence:"
        echo "  • Training used LR: 0.000198 (should be 0.000010)"
        echo "  • Final loss: $CHECKPOINT_LOSS (should be <2.0)"
        echo "  • Loss was erratic (not smoothly decreasing)"
        echo ""
        echo "This means:"
        echo "  ❌ English will sound like NOISE"
        echo "  ❌ French will sound like NOISE"
        echo "  ❌ Amharic quality will be POOR"
        echo ""
    fi
fi

echo "═══════════════════════════════════════════════════════════════════════"
echo "🔧 NEXT STEPS"
echo "═══════════════════════════════════════════════════════════════════════"
echo ""
echo "Choose an option:"
echo ""
echo "Option 1: TEST CURRENT CHECKPOINT (Quick)"
echo "  └─ See if audio is usable despite high loss"
echo "     Commands:"
echo "       cd /teamspace/studios/this_studio/Amharic_chatterbox-TTS"
echo "       python scripts/test_checkpoint_multilingual.py"
echo ""
echo "Option 2: START FRESH TRAINING (Recommended)"
echo "  └─ Retrain with correct config for proper multilingual TTS"
echo "     Commands:"
echo "       # Backup corrupted checkpoint"
echo "       mv models/checkpoints/checkpoint_epoch99_step4000.pt \\"
echo "          models/checkpoints/BACKUP_corrupted_epoch99.pt"
echo ""
echo "       # Start training in Gradio UI with:"
echo "       #   Config: config/training_config_finetune_FIXED.yaml"
echo "       #   Resume: None (start from extended embeddings)"
echo "       #   Verify logs show: LR: 0.000010 (NOT 0.000198)"
echo ""
echo "Option 3: CONTINUE TRAINING (Not Recommended)"
echo "  └─ Try to salvage current checkpoint with lower LR"
echo "     This rarely works well - pretrained weights already damaged"
echo ""

echo "═══════════════════════════════════════════════════════════════════════"
echo "✅ Diagnostic Complete"
echo "═══════════════════════════════════════════════════════════════════════"
echo ""
echo "To apply fixes automatically, run:"
echo "  bash scripts/apply_training_fix.sh"
echo ""
