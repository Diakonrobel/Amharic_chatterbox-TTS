#!/usr/bin/env python3
"""
Quick test to verify if finetuned checkpoint preserved multilingual capabilities
Tests English and Amharic inference
"""

import sys
from pathlib import Path
import torch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def test_checkpoint(checkpoint_path: str):
    """Test if checkpoint can generate for English and Amharic"""
    
    print("="*70)
    print("🧪 TESTING CHECKPOINT FOR MULTILINGUAL CAPABILITY")
    print("="*70)
    print(f"\nCheckpoint: {checkpoint_path}\n")
    
    # Load checkpoint
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        print("✓ Checkpoint loaded successfully")
        
        # Check what's in the checkpoint
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        
        print(f"\n📊 Checkpoint Info:")
        print(f"   Epoch: {checkpoint.get('epoch', 'N/A')}")
        print(f"   Step: {checkpoint.get('step', 'N/A')}")
        print(f"   Loss: {checkpoint.get('loss', 'N/A'):.4f}")
        
        # Check embedding size
        if 'text_embedding.weight' in state_dict:
            emb_weight = state_dict['text_embedding.weight']
            vocab_size, d_model = emb_weight.shape
            print(f"\n📐 Model Architecture:")
            print(f"   Vocabulary Size: {vocab_size}")
            print(f"   Embedding Dimension: {d_model}")
            
            # Check if it looks like extended embeddings
            if vocab_size > 2500:
                print(f"   ✓ Extended vocabulary detected (likely includes Amharic)")
            elif vocab_size > 2400:
                print(f"   ✓ Chatterbox multilingual size detected")
            else:
                print(f"   ⚠ Unexpected vocab size")
            
            # Check embedding statistics to detect corruption
            print(f"\n🔬 Embedding Analysis:")
            
            # Original embeddings (0-2453 should be Chatterbox)
            if vocab_size >= 2454:
                original_emb = emb_weight[:2454]
                new_emb = emb_weight[2454:]
                
                orig_mean = original_emb.mean().item()
                orig_std = original_emb.std().item()
                new_mean = new_emb.mean().item()
                new_std = new_emb.std().item()
                
                print(f"   Original embeddings (0-2453):")
                print(f"     Mean: {orig_mean:.4f}, Std: {orig_std:.4f}")
                print(f"   New embeddings (2454+):")
                print(f"     Mean: {new_mean:.4f}, Std: {new_std:.4f}")
                
                # Check if original embeddings were corrupted
                # Healthy embeddings typically have mean near 0, std around 0.01-0.05
                if abs(orig_mean) > 1.0 or orig_std > 10.0:
                    print(f"\n   ❌ CRITICAL: Original embeddings appear CORRUPTED!")
                    print(f"      High mean/std indicates training destroyed pretrained weights")
                    print(f"      English, French, etc. will likely produce NOISE!")
                elif abs(orig_mean - new_mean) < 0.1 and abs(orig_std - new_std) < 0.1:
                    print(f"\n   ⚠ WARNING: Original and new embeddings are too similar!")
                    print(f"      This suggests embeddings were NOT frozen during training")
                    print(f"      Pretrained languages may be corrupted")
                else:
                    print(f"\n   ✓ Embeddings look reasonable (but need audio test to confirm)")
            
        else:
            print(f"\n   ⚠ Could not find text_embedding.weight in checkpoint")
        
        # Summary
        print(f"\n" + "="*70)
        print(f"📋 VERDICT:")
        print(f"="*70)
        
        loss = checkpoint.get('loss', 999)
        if loss > 2.5:
            print(f"❌ Training loss too high: {loss:.4f}")
            print(f"   Expected: <2.0 for good quality")
            print(f"   Got: {loss:.4f}")
            print(f"\n🚨 This checkpoint likely produces poor quality audio!")
        elif loss > 1.5:
            print(f"⚠️  Training loss is moderate: {loss:.4f}")
            print(f"   Expected: <1.5 for production quality")
            print(f"   Audio quality may be acceptable but not optimal")
        else:
            print(f"✅ Training loss is good: {loss:.4f}")
            print(f"   Expected quality level achieved")
        
        print(f"\n🧪 NEXT STEPS:")
        print(f"1. Test inference on English text: 'Hello world'")
        print(f"2. Test inference on Amharic text: 'ሰላም ለዓለም'")
        print(f"3. Listen to both outputs:")
        print(f"   - English should sound like clear English")
        print(f"   - Amharic should sound like clear Amharic")
        print(f"   - If EITHER sounds like noise, the model is corrupted")
        print(f"\n4. If corrupted, you MUST retrain with:")
        print(f"   - config/training_config_finetune_FIXED.yaml")
        print(f"   - Verify LR shows 0.000010 in logs (not 0.000198)")
        print(f"   - Start from fresh extended embeddings")
        
        print("="*70)
        
    except Exception as e:
        print(f"❌ Error loading checkpoint: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test checkpoint for multilingual capability")
    parser.add_argument("--checkpoint", type=str, 
                       default="models/checkpoints/checkpoint_epoch99_step4000.pt",
                       help="Path to checkpoint file")
    
    args = parser.parse_args()
    
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        print(f"\nAvailable checkpoints:")
        checkpoint_dir = Path("models/checkpoints")
        if checkpoint_dir.exists():
            for ckpt in sorted(checkpoint_dir.glob("*.pt")):
                print(f"   - {ckpt}")
        sys.exit(1)
    
    test_checkpoint(str(checkpoint_path))
