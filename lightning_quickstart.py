"""
Lightning AI Quick Start Script for Amharic TTS
Run this after setup to quickly start training with Gradio UI
"""

import subprocess
import sys
from pathlib import Path

def check_setup():
    """Check if setup is complete"""
    print("🔍 Checking setup...")
    
    required_dirs = [
        "data/srt_datasets",
        "models/checkpoints",
        "models/tokenizer",
        "logs"
    ]
    
    for dir_path in required_dirs:
        if not Path(dir_path).exists():
            print(f"❌ Missing directory: {dir_path}")
            print("   Run: ./setup_lightning.sh")
            return False
    
    print("✓ All directories exist")
    
    # Check if dataset exists
    dataset_dir = Path("data/srt_datasets")
    datasets = list(dataset_dir.glob("*/"))
    
    if not datasets:
        print("\n⚠️  No datasets found in data/srt_datasets/")
        print("   Upload your dataset before training")
        return False
    
    print(f"✓ Found {len(datasets)} dataset(s)")
    for ds in datasets:
        print(f"  - {ds.name}")
    
    return True


def check_gpu():
    """Check GPU availability"""
    print("\n🔍 Checking GPU...")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA Version: {torch.version.cuda}")
            print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            return True
        else:
            print("⚠️  No GPU detected. Training will be slow on CPU.")
            return False
    except ImportError:
        print("❌ PyTorch not installed. Run: ./setup_lightning.sh")
        return False


def start_gradio_ui(share=True, port=7860):
    """Start Gradio UI"""
    print("\n🚀 Starting Gradio UI...")
    print(f"   Share: {share}")
    print(f"   Port: {port}")
    print("\n" + "="*60)
    
    cmd = [
        sys.executable,
        "gradio_app/full_training_app.py",
        "--port", str(port)
    ]
    
    if share:
        cmd.append("--share")
    
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n\n👋 Shutting down...")


def main():
    """Main function"""
    print("="*60)
    print("AMHARIC TTS - Lightning AI Quick Start")
    print("="*60)
    print()
    
    # Check setup
    if not check_setup():
        print("\n❌ Setup incomplete. Please run:")
        print("   chmod +x setup_lightning.sh")
        print("   ./setup_lightning.sh")
        sys.exit(1)
    
    # Check GPU
    has_gpu = check_gpu()
    
    if not has_gpu:
        response = input("\n⚠️  No GPU detected. Continue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Exiting...")
            sys.exit(0)
    
    print("\n" + "="*60)
    print("✅ All checks passed!")
    print("="*60)
    
    # Start Gradio UI
    start_gradio_ui(share=True, port=7860)


if __name__ == "__main__":
    main()
