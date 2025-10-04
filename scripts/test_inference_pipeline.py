"""
Quick test script to verify inference pipeline is working
This helps diagnose if the issue is the model or the pipeline
"""

import torch
import numpy as np
import soundfile as sf
from pathlib import Path
import sys

# Add project root
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.models.t3_model import SimplifiedT3Model
from src.audio import AudioProcessor

def test_inference_pipeline():
    """Test that the inference pipeline can generate audio"""
    
    print("=" * 60)
    print("TESTING INFERENCE PIPELINE")
    print("=" * 60)
    
    # Create a simple model
    print("\n1. Creating model...")
    model = SimplifiedT3Model(
        vocab_size=2535,
        d_model=1024,
        nhead=8,
        num_encoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        n_mels=80,
        max_seq_len=1000
    )
    model.eval()
    print("✓ Model created")
    
    # Create dummy input
    print("\n2. Creating test input...")
    text_ids = torch.randint(0, 2535, (1, 20))  # 20 tokens
    print(f"✓ Text IDs shape: {text_ids.shape}")
    
    # Forward pass
    print("\n3. Running forward pass...")
    with torch.no_grad():
        outputs = model(text_ids)
        mel_output = outputs['mel_outputs']
        durations = outputs['durations']
    
    print(f"✓ Mel output shape: {mel_output.shape}")
    print(f"✓ Durations shape: {durations.shape}")
    print(f"✓ Duration values (first 10): {durations[0, :10].tolist()}")
    print(f"✓ Total mel frames: {mel_output.shape[2]}")
    
    # Test mel-to-audio conversion
    print("\n4. Converting mel to audio...")
    audio_processor = AudioProcessor(sampling_rate=24000)
    mel_np = mel_output.squeeze(0).cpu().numpy()
    
    try:
        audio = audio_processor.mel_to_audio(mel_np)
        print(f"✓ Audio generated: {audio.shape}")
        print(f"✓ Audio duration: {len(audio) / 24000:.2f}s")
        
        # Save test audio
        output_path = project_root / "test_output.wav"
        sf.write(output_path, audio, 24000)
        print(f"✓ Saved to: {output_path}")
        
    except Exception as e:
        print(f"✗ Audio conversion failed: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("PIPELINE TEST COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print("\nNOTE: The audio will sound like noise/static because")
    print("the model has RANDOM WEIGHTS (not trained yet).")
    print("\nTo get real Amharic speech:")
    print("1. Train the model on your Amharic dataset")
    print("2. Use the trained checkpoint for inference")
    print("3. Then the audio will be intelligible speech!")
    print("=" * 60)
    
    return True

if __name__ == "__main__":
    success = test_inference_pipeline()
    sys.exit(0 if success else 1)
