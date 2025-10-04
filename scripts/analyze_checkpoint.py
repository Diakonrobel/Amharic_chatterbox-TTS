"""
Analyze checkpoint to see training progress and suggest next steps
"""

import torch
import sys
from pathlib import Path

# Add project root
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def analyze_checkpoint(checkpoint_path: str):
    """Analyze training checkpoint"""
    
    print("=" * 70)
    print("CHECKPOINT ANALYSIS")
    print("=" * 70)
    
    try:
        # Load checkpoint
        print(f"\n📂 Loading: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        # Extract information
        print("\n" + "=" * 70)
        print("TRAINING PROGRESS")
        print("=" * 70)
        
        if 'epoch' in checkpoint:
            print(f"✓ Epoch: {checkpoint['epoch']}")
        
        if 'step' in checkpoint:
            print(f"✓ Training step: {checkpoint['step']}")
        
        if 'loss' in checkpoint:
            loss = checkpoint['loss']
            print(f"✓ Training loss: {loss:.4f}")
            
            # Assess loss quality
            if loss > 10:
                print("  ⚠️  Loss is very high - needs much more training!")
            elif loss > 5:
                print("  ⚠️  Loss is high - needs more training")
            elif loss > 2:
                print("  ⏳ Loss is moderate - making progress")
            elif loss > 1:
                print("  ✓ Loss is good - model learning well")
            else:
                print("  ✅ Loss is excellent!")
        
        # Check model state
        print("\n" + "=" * 70)
        print("MODEL INFORMATION")
        print("=" * 70)
        
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            print(f"✓ Model weights present: {len(state_dict)} tensors")
            
            # Check if embeddings are there
            if 'text_embedding.weight' in state_dict:
                vocab_size = state_dict['text_embedding.weight'].shape[0]
                d_model = state_dict['text_embedding.weight'].shape[1]
                print(f"✓ Text embedding: vocab={vocab_size}, d_model={d_model}")
        
        # Provide recommendations
        print("\n" + "=" * 70)
        print("RECOMMENDATIONS")
        print("=" * 70)
        
        if 'loss' in checkpoint:
            loss = checkpoint['loss']
            
            if loss > 5:
                print("\n🎓 YOUR MODEL NEEDS MORE TRAINING!")
                print("\n📊 Current situation:")
                print(f"   - Loss: {loss:.4f} (should be < 2.0 for decent quality)")
                print("   - Dataset: 0.6 hours (recommended: 5-10+ hours)")
                print("\n✅ Solutions:")
                print("   1. COLLECT MORE DATA (most important!):")
                print("      - Add 5-10+ hours of Amharic speech")
                print("      - More data = better quality")
                print("\n   2. TRAIN LONGER:")
                print("      - Resume training from this checkpoint")
                print("      - Train for 500-1000 more epochs")
                print("      - Monitor loss until it drops below 2.0")
                print("\n   3. OPTIMIZE TRAINING:")
                print("      - Lower learning rate: 5e-5 or 1e-5")
                print("      - Increase batch size if memory allows")
                print("      - Enable early stopping")
                
            elif loss > 2:
                print("\n⏳ MODEL IS LEARNING BUT NEEDS MORE TIME")
                print(f"   - Loss: {loss:.4f} (getting better!)")
                print("   - Continue training for 200-500 more epochs")
                
            else:
                print("\n✅ MODEL IS WELL TRAINED!")
                print(f"   - Loss: {loss:.4f} (good quality expected)")
                print("   - Audio should be intelligible")
                print("   - If audio is still bad, check:")
                print("     * Dataset quality")
                print("     * Transcription accuracy")
                print("     * Audio preprocessing")
        
        print("\n" + "=" * 70)
        print("HOW TO CONTINUE TRAINING")
        print("=" * 70)
        print("\n1. Using Gradio UI:")
        print("   - Go to 'Training Pipeline' tab")
        print("   - Select 'Resume from Checkpoint'")
        print("   - Choose: checkpoint_latest.pt")
        print("   - Click 'Start Training'")
        print("\n2. Using command line:")
        print(f"   python src/training/train.py \\")
        print(f"       --config config/training_config.yaml \\")
        print(f"       --resume {checkpoint_path}")
        
        print("\n" + "=" * 70)
        
    except Exception as e:
        print(f"\n❌ Error analyzing checkpoint: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    checkpoint_path = "models/checkpoints/checkpoint_latest.pt"
    
    if len(sys.argv) > 1:
        checkpoint_path = sys.argv[1]
    
    analyze_checkpoint(checkpoint_path)
