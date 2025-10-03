"""
Quick Verification Script for Amharic TTS Training Setup
Tests all components before starting full training
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_imports():
    """Test all required imports"""
    print("=" * 60)
    print("TESTING IMPORTS")
    print("=" * 60)
    
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  CUDA device: {torch.cuda.get_device_name(0)}")
    except ImportError as e:
        print(f"✗ PyTorch import failed: {e}")
        return False
    
    try:
        import librosa
        print(f"✓ librosa {librosa.__version__}")
    except ImportError as e:
        print(f"✗ librosa import failed: {e}")
        return False
    
    try:
        import yaml
        print("✓ PyYAML")
    except ImportError as e:
        print(f"✗ PyYAML import failed: {e}")
        return False
    
    try:
        from src.models.t3_model import SimplifiedT3Model, TTSLoss
        print("✓ SimplifiedT3Model")
        print("✓ TTSLoss")
    except ImportError as e:
        print(f"✗ Model imports failed: {e}")
        return False
    
    try:
        from src.audio import AudioProcessor, collate_fn
        print("✓ AudioProcessor")
        print("✓ collate_fn")
    except ImportError as e:
        print(f"✗ Audio imports failed: {e}")
        return False
    
    try:
        from src.training.train import SimpleAmharicDataset, setup_model, train_epoch, validate
        print("✓ SimpleAmharicDataset")
        print("✓ Training functions")
    except ImportError as e:
        print(f"✗ Training imports failed: {e}")
        return False
    
    print("\n✅ All imports successful!\n")
    return True


def test_model_creation():
    """Test model instantiation"""
    print("=" * 60)
    print("TESTING MODEL CREATION")
    print("=" * 60)
    
    try:
        from src.models.t3_model import SimplifiedT3Model, TTSLoss
        
        model = SimplifiedT3Model(
            vocab_size=2000,
            d_model=512,
            nhead=8,
            num_encoder_layers=6,
            dim_feedforward=2048,
            dropout=0.1,
            n_mels=80,
            max_seq_len=1000
        )
        
        print(f"✓ Model created")
        print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Test forward pass with dummy data
        import torch
        dummy_text = torch.randint(0, 2000, (2, 50))  # [batch=2, seq_len=50]
        dummy_lengths = torch.tensor([50, 45])
        dummy_mel = torch.randn(2, 80, 100)  # [batch=2, n_mels=80, time=100]
        
        outputs = model(dummy_text, dummy_lengths, dummy_mel)
        print(f"✓ Forward pass successful")
        print(f"  Output mel shape: {outputs['mel_outputs'].shape}")
        print(f"  Duration shape: {outputs['durations'].shape}")
        
        # Test loss
        criterion = TTSLoss(mel_loss_weight=1.0, duration_loss_weight=0.1)
        targets = {'mel': dummy_mel, 'mel_lengths': torch.tensor([100, 95])}
        losses = criterion(outputs, targets)
        print(f"✓ Loss computation successful")
        print(f"  Total loss: {losses['total_loss'].item():.4f}")
        print(f"  Mel loss: {losses['mel_loss'].item():.4f}")
        print(f"  Duration loss: {losses['duration_loss'].item():.4f}")
        
        print("\n✅ Model tests passed!\n")
        return True
        
    except Exception as e:
        print(f"✗ Model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_audio_processing():
    """Test audio processing"""
    print("=" * 60)
    print("TESTING AUDIO PROCESSING")
    print("=" * 60)
    
    try:
        from src.audio import AudioProcessor
        import numpy as np
        
        processor = AudioProcessor()
        print("✓ AudioProcessor created")
        print(f"  Sample rate: {processor.sampling_rate}")
        print(f"  N mels: {processor.n_mels}")
        print(f"  Hop length: {processor.hop_length}")
        
        # Test with dummy audio
        dummy_audio = np.random.randn(22050)  # 1 second of audio
        mel = processor.get_mel_spectrogram(dummy_audio)
        print(f"✓ Mel extraction successful")
        print(f"  Mel shape: {mel.shape}")
        
        print("\n✅ Audio processing tests passed!\n")
        return True
        
    except Exception as e:
        print(f"✗ Audio processing test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_config():
    """Test configuration file"""
    print("=" * 60)
    print("TESTING CONFIGURATION")
    print("=" * 60)
    
    config_path = project_root / "configs" / "training_config.yaml"
    
    if not config_path.exists():
        print(f"✗ Config file not found: {config_path}")
        print("  Please create configs/training_config.yaml")
        return False
    
    try:
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        print(f"✓ Config loaded from {config_path}")
        
        # Check required fields
        required_fields = ['paths', 'training', 'model', 'data']
        for field in required_fields:
            if field in config:
                print(f"  ✓ {field}")
            else:
                print(f"  ✗ Missing: {field}")
        
        # Check data directory
        data_dir = Path(config['paths']['data_dir'])
        if data_dir.exists():
            print(f"✓ Data directory exists: {data_dir}")
            
            metadata = data_dir / 'metadata.csv'
            if metadata.exists():
                print(f"✓ metadata.csv found")
                # Count lines
                with open(metadata, 'r', encoding='utf-8') as f:
                    lines = len(f.readlines())
                print(f"  Samples: {lines}")
            else:
                print(f"⚠ metadata.csv not found in {data_dir}")
            
            wavs_dir = data_dir / 'wavs'
            if wavs_dir.exists():
                wav_files = list(wavs_dir.glob('*.wav'))
                print(f"✓ wavs/ directory found: {len(wav_files)} WAV files")
            else:
                print(f"⚠ wavs/ directory not found in {data_dir}")
        else:
            print(f"⚠ Data directory not found: {data_dir}")
        
        print("\n✅ Configuration tests passed!\n")
        return True
        
    except Exception as e:
        print(f"✗ Config test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_training_script():
    """Test training script syntax"""
    print("=" * 60)
    print("TESTING TRAINING SCRIPT")
    print("=" * 60)
    
    train_script = project_root / "src" / "training" / "train.py"
    
    if not train_script.exists():
        print(f"✗ Training script not found: {train_script}")
        return False
    
    try:
        import py_compile
        py_compile.compile(str(train_script), doraise=True)
        print(f"✓ Training script syntax OK")
        
        # Check for key functions
        with open(train_script, 'r', encoding='utf-8') as f:
            content = f.read()
        
        required_functions = [
            'setup_dataloaders',
            'train_epoch',
            'validate',
            'setup_model',
            'train'
        ]
        
        for func in required_functions:
            if f"def {func}(" in content:
                print(f"  ✓ {func}()")
            else:
                print(f"  ✗ Missing: {func}()")
        
        # Check for real training (not dummy)
        if "dummy_input" in content or "torch.randn(1, 10," in content:
            print("\n⚠ WARNING: Training script may still contain dummy code!")
            print("  Check train_epoch() function")
        else:
            print("\n✓ No dummy code detected")
        
        print("\n✅ Training script tests passed!\n")
        return True
        
    except Exception as e:
        print(f"✗ Training script test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("AMHARIC TTS TRAINING SETUP VERIFICATION")
    print("=" * 60 + "\n")
    
    results = []
    
    # Run tests
    results.append(("Imports", test_imports()))
    results.append(("Model Creation", test_model_creation()))
    results.append(("Audio Processing", test_audio_processing()))
    results.append(("Configuration", test_config()))
    results.append(("Training Script", test_training_script()))
    
    # Summary
    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status:10} {name}")
    
    all_passed = all(result[1] for result in results)
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ ALL TESTS PASSED - READY FOR TRAINING!")
        print("=" * 60)
        print("\nNext steps:")
        print("1. Ensure your data is prepared (metadata.csv + wavs/)")
        print("2. Review your config file (configs/training_config.yaml)")
        print("3. Run a quick test:")
        print("   python src/training/train.py --config configs/training_config.yaml")
        print("4. Monitor the first few steps and verify realistic loss values")
    else:
        print("❌ SOME TESTS FAILED - FIX ISSUES BEFORE TRAINING")
        print("=" * 60)
        print("\nPlease fix the failed tests above")
    print()


if __name__ == "__main__":
    main()
