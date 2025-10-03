#!/usr/bin/env python3
"""
Lightning AI Setup Script for Amharic TTS
Configures the environment after git pull in Lightning AI cloud
"""

import os
import sys
from pathlib import Path
import subprocess

def setup_lightning_environment():
    """Setup environment for Lightning AI cloud"""
    print("🌩️ Setting up Lightning AI environment for Amharic TTS...")
    
    # Create necessary directories
    directories = [
        "models/checkpoints",
        "models/pretrained", 
        "models/tokenizer",
        "data/srt_datasets",
        "logs",
        "config"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory: {directory}")
    
    # Install additional dependencies if needed
    try:
        print("📦 Installing additional dependencies...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "safetensors", "--quiet"])
        print("✓ safetensors installed")
    except subprocess.CalledProcessError:
        print("⚠ Failed to install safetensors (optional)")
    
    # Check for CUDA availability
    try:
        import torch
        if torch.cuda.is_available():
            print(f"🔥 CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        else:
            print("💻 CUDA not available, using CPU")
    except ImportError:
        print("⚠ PyTorch not available")
    
    # Verify key components
    try:
        from src.models.t3_model import SimplifiedT3Model
        print("✓ TTS model classes available")
    except ImportError as e:
        print(f"❌ TTS model import failed: {e}")
    
    try:
        from src.g2p.amharic_g2p import AmharicG2P
        g2p = AmharicG2P()
        print("✓ Amharic G2P system ready")
    except Exception as e:
        print(f"❌ G2P system failed: {e}")
    
    print("\n" + "="*60)
    print("🚀 LIGHTNING AI SETUP COMPLETE!")
    print("="*60)
    print("\nNext steps:")
    print("1. 📝 Import your dataset using the web interface")
    print("2. 🔤 Train tokenizer (if needed)")  
    print("3. 📥 Download Chatterbox model (optional)")
    print("4. 🎓 Start training!")
    print("\nLaunch the web interface:")
    print("python gradio_app/full_training_app.py --share")
    print()

if __name__ == "__main__":
    setup_lightning_environment()