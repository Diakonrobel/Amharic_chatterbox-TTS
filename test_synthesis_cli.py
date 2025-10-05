#!/usr/bin/env python3
"""
Quick CLI synthesis test for trained Chatterbox model
Tests both Amharic and English to verify multilingual capability
"""

import sys
from pathlib import Path
import torch
import json
import soundfile as sf
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from src.models.t3_model import SimplifiedT3Model
from src.audio import AudioProcessor

def load_tokenizer(tokenizer_path: str):
    """Load tokenizer vocabulary"""
    with open(tokenizer_path, 'r', encoding='utf-8') as f:
        tokenizer_data = json.load(f)
    return tokenizer_data

def text_to_ids(text: str, tokenizer_data: dict):
    """Convert text to token IDs using the tokenizer vocab"""
    vocab = tokenizer_data.get('model', {}).get('vocab', {})
    
    # Simple character-level tokenization
    token_ids = []
    for char in text:
        if char in vocab:
            token_ids.append(vocab[char])
        else:
            # Use <unk> token if available, otherwise skip
            if '<unk>' in vocab:
                token_ids.append(vocab['<unk>'])
    
    return token_ids

def synthesize(
    checkpoint_path: str,
    tokenizer_path: str,
    test_texts: dict,
    output_dir: str = "test_outputs"
):
    """Synthesize speech from trained checkpoint"""
    
    print("=" * 70)
    print("🎵 CHATTERBOX TTS - CLI SYNTHESIS TEST")
    print("=" * 70)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    print(f"\n📂 Output directory: {output_path.absolute()}")
    
    # 1. Load checkpoint
    print(f"\n1️⃣ Loading checkpoint: {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        # Extract state dict
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        
        # Get model info
        vocab_size = state_dict['text_embedding.weight'].shape[0]
        d_model = state_dict['text_embedding.weight'].shape[1]
        
        epoch = checkpoint.get('epoch', 'N/A')
        step = checkpoint.get('step', 'N/A')
        loss = checkpoint.get('loss', 'N/A')
        
        print(f"   ✓ Checkpoint loaded")
        print(f"   📊 Epoch: {epoch}, Step: {step}, Loss: {loss}")
        print(f"   📐 Vocab size: {vocab_size}, D-model: {d_model}")
        
    except Exception as e:
        print(f"   ❌ Error loading checkpoint: {e}")
        return False
    
    # 2. Load tokenizer
    print(f"\n2️⃣ Loading tokenizer: {tokenizer_path}")
    try:
        tokenizer_data = load_tokenizer(tokenizer_path)
        tokenizer_vocab_size = len(tokenizer_data.get('model', {}).get('vocab', {}))
        print(f"   ✓ Tokenizer loaded")
        print(f"   📐 Tokenizer vocab size: {tokenizer_vocab_size}")
        
        if vocab_size != tokenizer_vocab_size:
            print(f"   ⚠️  WARNING: Vocab size mismatch!")
            print(f"      Model: {vocab_size}, Tokenizer: {tokenizer_vocab_size}")
    except Exception as e:
        print(f"   ❌ Error loading tokenizer: {e}")
        return False
    
    # 3. Create model
    print(f"\n3️⃣ Creating model...")
    try:
        model = SimplifiedT3Model(
            vocab_size=vocab_size,
            d_model=d_model,
            nhead=8,
            num_encoder_layers=6,
            dim_feedforward=2048,
            dropout=0.1,
            n_mels=80,
            max_seq_len=1000
        )
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        print(f"   ✓ Model created and weights loaded")
    except Exception as e:
        print(f"   ❌ Error creating model: {e}")
        return False
    
    # 4. Create audio processor
    print(f"\n4️⃣ Creating audio processor...")
    audio_processor = AudioProcessor(sampling_rate=24000)
    print(f"   ✓ Audio processor ready")
    
    # 5. Synthesize test texts
    print(f"\n5️⃣ Synthesizing {len(test_texts)} test samples...\n")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = []
    
    for i, (lang, text) in enumerate(test_texts.items(), 1):
        print(f"   [{i}/{len(test_texts)}] {lang}: '{text}'")
        
        try:
            # Convert text to IDs
            token_ids = text_to_ids(text, tokenizer_data)
            if not token_ids:
                print(f"      ⚠️  No valid tokens found, skipping...")
                continue
            
            text_tensor = torch.tensor([token_ids], dtype=torch.long)
            print(f"      → Token IDs: {token_ids[:20]}{'...' if len(token_ids) > 20 else ''}")
            print(f"      → Shape: {text_tensor.shape}")
            
            # Generate mel spectrogram
            with torch.no_grad():
                outputs = model(text_tensor)
                mel_output = outputs['mel_outputs']
                durations = outputs['durations']
            
            print(f"      → Mel shape: {mel_output.shape}")
            print(f"      → Total frames: {mel_output.shape[2]}")
            
            # Convert mel to audio
            mel_np = mel_output.squeeze(0).cpu().numpy()
            audio = audio_processor.mel_to_audio(mel_np)
            
            duration_sec = len(audio) / 24000
            print(f"      → Audio duration: {duration_sec:.2f}s")
            
            # Save audio
            filename = f"{timestamp}_{lang.lower()}.wav"
            filepath = output_path / filename
            sf.write(filepath, audio, 24000)
            print(f"      ✓ Saved: {filepath.name}\n")
            
            results.append({
                'language': lang,
                'text': text,
                'file': str(filepath),
                'duration': duration_sec
            })
            
        except Exception as e:
            print(f"      ❌ Error: {e}\n")
            continue
    
    # Summary
    print("=" * 70)
    print("✅ SYNTHESIS COMPLETE")
    print("=" * 70)
    print(f"\n📁 Generated {len(results)} audio files in: {output_path.absolute()}\n")
    
    for result in results:
        print(f"   • {result['language']}: {Path(result['file']).name}")
        print(f"     Text: '{result['text']}'")
        print(f"     Duration: {result['duration']:.2f}s\n")
    
    print("🎧 NEXT STEPS:")
    print("   1. Listen to the generated audio files")
    print("   2. Check if Amharic is clear and intelligible")
    print("   3. Check if English still works (multilingual preserved)")
    print("   4. If quality is good, continue training to epoch 100+")
    print("   5. If quality is poor, may need more training epochs")
    print("=" * 70)
    
    return True

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="CLI synthesis test for Chatterbox TTS")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="models/checkpoints/checkpoint_best.pt",
        help="Path to trained checkpoint"
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="tokenizers/am-merged_merged.json",
        help="Path to tokenizer JSON file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="test_outputs",
        help="Output directory for generated audio"
    )
    parser.add_argument(
        "--text-amharic",
        type=str,
        default="ሰላም፣ እንደምን ነህ?",
        help="Amharic text to synthesize"
    )
    parser.add_argument(
        "--text-english",
        type=str,
        default="Hello, how are you?",
        help="English text to synthesize"
    )
    
    args = parser.parse_args()
    
    # Check if checkpoint exists
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        print(f"\nAvailable checkpoints:")
        checkpoint_dir = Path("models/checkpoints")
        if checkpoint_dir.exists():
            for ckpt in sorted(checkpoint_dir.glob("*.pt")):
                print(f"   - {ckpt}")
        else:
            print(f"   No checkpoints found in {checkpoint_dir}")
        sys.exit(1)
    
    # Check if tokenizer exists
    tokenizer_path = Path(args.tokenizer)
    if not tokenizer_path.exists():
        print(f"❌ Tokenizer not found: {tokenizer_path}")
        sys.exit(1)
    
    # Test texts
    test_texts = {
        'Amharic': args.text_amharic,
        'English': args.text_english,
    }
    
    # Run synthesis
    success = synthesize(
        str(checkpoint_path),
        str(tokenizer_path),
        test_texts,
        args.output_dir
    )
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
