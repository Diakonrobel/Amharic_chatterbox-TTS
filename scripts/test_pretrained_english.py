"""
Test if the pretrained multilingual model still works for English/French
This helps diagnose if finetuning broke the original model
"""

import torch
import sys
from pathlib import Path

# Add project root
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def test_pretrained_model():
    """Test pretrained model on English"""
    
    print("=" * 70)
    print("TESTING PRETRAINED MODEL ON ENGLISH")
    print("=" * 70)
    
    # Check if we have the original pretrained model
    pretrained_paths = [
        "models/pretrained/chatterbox/t3_mtl23ls_v2.safetensors",
        "models/pretrained/t3_mtl23ls_v2.safetensors",
        "models/pretrained/chatterbox_multilingual.pt",
    ]
    
    pretrained_model = None
    for path in pretrained_paths:
        p = Path(path)
        if p.exists():
            pretrained_model = p
            break
    
    if not pretrained_model:
        print("\n❌ ERROR: No pretrained model found!")
        print("\nSearched in:")
        for path in pretrained_paths:
            print(f"  - {path}")
        print("\nYou need to download the Chatterbox pretrained model first.")
        print("Go to 'Model Setup' tab → 'Download Chatterbox Model' → Multilingual")
        return False
    
    print(f"\n✓ Found pretrained model: {pretrained_model}")
    
    # Check your current checkpoint
    checkpoint_path = Path("models/checkpoints/checkpoint_latest.pt")
    
    if not checkpoint_path.exists():
        print("\n❌ No checkpoint found at models/checkpoints/checkpoint_latest.pt")
        return False
    
    print(f"✓ Found your checkpoint: {checkpoint_path}")
    
    # Load and compare
    print("\n" + "=" * 70)
    print("ANALYZING CHECKPOINTS")
    print("=" * 70)
    
    try:
        # Load pretrained
        print("\nLoading ORIGINAL pretrained model...")
        if str(pretrained_model).endswith('.safetensors'):
            from safetensors.torch import load_file
            pretrained_weights = load_file(pretrained_model)
        else:
            pretrained_checkpoint = torch.load(pretrained_model, map_location='cpu', weights_only=False)
            if 'model_state_dict' in pretrained_checkpoint:
                pretrained_weights = pretrained_checkpoint['model_state_dict']
            else:
                pretrained_weights = pretrained_checkpoint
        
        print(f"✓ Pretrained model loaded: {len(pretrained_weights)} tensors")
        
        # Load your checkpoint
        print("\nLoading YOUR finetuned checkpoint...")
        your_checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        if 'model_state_dict' in your_checkpoint:
            your_weights = your_checkpoint['model_state_dict']
        else:
            your_weights = your_checkpoint
        
        print(f"✓ Your checkpoint loaded: {len(your_weights)} tensors")
        
        # Compare embeddings
        print("\n" + "=" * 70)
        print("EMBEDDING COMPARISON")
        print("=" * 70)
        
        # Find embedding keys
        pretrained_emb_key = None
        your_emb_key = None
        
        for key in pretrained_weights.keys():
            if 'emb' in key.lower() and 'weight' in key:
                pretrained_emb_key = key
                break
        
        for key in your_weights.keys():
            if 'text_embedding.weight' in key:
                your_emb_key = key
                break
        
        if pretrained_emb_key and your_emb_key:
            pretrained_emb = pretrained_weights[pretrained_emb_key]
            your_emb = your_weights[your_emb_key]
            
            print(f"\nPretrained embeddings: {pretrained_emb.shape}")
            print(f"Your embeddings: {your_emb.shape}")
            
            # Check if first 2454 embeddings are preserved
            if your_emb.shape[0] >= 2454:
                # Compare first 2454 embeddings
                original_part = your_emb[:2454]
                pretrained_part = pretrained_emb[:2454] if pretrained_emb.shape[0] >= 2454 else pretrained_emb
                
                diff = torch.abs(original_part - pretrained_part).mean().item()
                
                print(f"\n📊 Embedding difference (first 2454 tokens):")
                print(f"   Mean absolute difference: {diff:.6f}")
                
                if diff < 0.001:
                    print("   ✅ GOOD: Original embeddings preserved!")
                    print("   → English/French should still work")
                elif diff < 0.1:
                    print("   ⚠️ WARNING: Small changes detected")
                    print("   → Some degradation expected")
                else:
                    print("   ❌ BAD: Original embeddings DESTROYED!")
                    print("   → This is why English/French sound like noise")
                    print("\n🔧 SOLUTION:")
                    print("   1. Re-extend embeddings from scratch")
                    print("   2. Start finetuning again with:")
                    print("      - freeze_original_embeddings: true")
                    print("      - freeze_until_index: 2454")
                    print("      - learning_rate: 1e-5 (much lower!)")
        
        # Check training info
        if 'epoch' in your_checkpoint and 'loss' in your_checkpoint:
            print("\n" + "=" * 70)
            print("TRAINING INFO")
            print("=" * 70)
            print(f"\nEpoch: {your_checkpoint['epoch']}")
            print(f"Loss: {your_checkpoint['loss']:.4f}")
            print(f"Step: {your_checkpoint.get('step', 'N/A')}")
        
        print("\n" + "=" * 70)
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_pretrained_model()
