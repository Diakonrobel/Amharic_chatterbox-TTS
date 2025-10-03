"""
Extract vocabulary from Chatterbox tokenizer
Downloads Chatterbox tokenizer and extracts vocabulary for merging
"""
import json
from pathlib import Path
from huggingface_hub import hf_hub_download


def extract_chatterbox_vocab(output_path: str):
    """Download and extract Chatterbox tokenizer vocabulary"""
    
    print("="*60)
    print("EXTRACTING CHATTERBOX TOKENIZER")
    print("="*60)
    
    # Download tokenizer from HuggingFace
    print("\n[1/3] Downloading tokenizer from HuggingFace...")
    try:
        tokenizer_path = hf_hub_download(
            repo_id="ResembleAI/chatterbox",
            filename="tokenizer.json"
        )
        print(f"   ✓ Downloaded: {tokenizer_path}")
    except Exception as e:
        print(f"   ✗ Error downloading: {e}")
        print(f"   Try manual download:")
        print(f"   wget https://huggingface.co/ResembleAI/chatterbox/resolve/main/tokenizer.json")
        return None
    
    # Load tokenizer
    print("\n[2/3] Loading tokenizer...")
    with open(tokenizer_path, 'r', encoding='utf-8') as f:
        tokenizer_data = json.load(f)
    
    # Extract vocabulary
    print("\n[3/3] Extracting vocabulary...")
    
    # Chatterbox uses different tokenizer formats
    # Try to extract vocab from different possible structures
    vocab = {}
    
    if 'model' in tokenizer_data and 'vocab' in tokenizer_data['model']:
        # Standard tokenizer format
        vocab = tokenizer_data['model']['vocab']
        print(f"   ✓ Found vocab in model.vocab")
    elif 'vocab' in tokenizer_data:
        vocab = tokenizer_data['vocab']
        print(f"   ✓ Found vocab at top level")
    else:
        # Try to build from tokens
        print("   Building vocab from tokens...")
        if 'added_tokens' in tokenizer_data:
            for token_data in tokenizer_data['added_tokens']:
                vocab[token_data['content']] = token_data['id']
        
        # Also check for regular tokens
        if 'model' in tokenizer_data and 'vocab' in tokenizer_data['model']:
            for token, idx in tokenizer_data['model']['vocab'].items():
                vocab[token] = idx
    
    if not vocab:
        print("   ✗ Could not extract vocabulary!")
        print("   Tokenizer structure:")
        print(f"   Keys: {list(tokenizer_data.keys())}")
        if 'model' in tokenizer_data:
            print(f"   Model keys: {list(tokenizer_data['model'].keys())}")
        return None
    
    print(f"   ✓ Extracted {len(vocab)} tokens")
    
    # Save vocabulary
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    
    print(f"\n   ✓ Saved to: {output_path}")
    
    # Show sample tokens
    print("\n   Sample tokens:")
    for i, (token, idx) in enumerate(list(vocab.items())[:10]):
        print(f"      {idx}: {repr(token)}")
    
    # Show statistics
    print(f"\n   Token statistics:")
    print(f"      Total tokens: {len(vocab)}")
    print(f"      Index range: {min(vocab.values())} - {max(vocab.values())}")
    
    print("="*60)
    return vocab


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Extract Chatterbox tokenizer vocabulary"
    )
    parser.add_argument(
        '--output',
        default='models/pretrained/chatterbox_tokenizer.json',
        help='Output path for extracted vocabulary'
    )
    args = parser.parse_args()
    
    vocab = extract_chatterbox_vocab(args.output)
    
    if vocab:
        print("\n✅ Success! Vocabulary extracted.")
        print(f"Next step: Merge with Amharic tokenizer")
        print(f"Run: python scripts/merge_tokenizers.py \\")
        print(f"       --base {args.output} \\")
        print(f"       --amharic models/tokenizer/amharic_tokenizer/vocab.json \\")
        print(f"       --output models/tokenizer/merged_vocab.json \\")
        print(f"       --validate")
    else:
        print("\n❌ Failed to extract vocabulary. See errors above.")
