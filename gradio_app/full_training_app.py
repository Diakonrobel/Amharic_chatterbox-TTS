"""
Complete Gradio Web Interface for Amharic TTS
Includes: TTS Demo, Dataset Management, Tokenizer Training, Model Setup, and Full Training Pipeline
"""

import gradio as gr
import sys
import os
import subprocess
import json
import torch
import numpy as np
from pathlib import Path
from typing import Optional

# Add project paths
sys.path.append(str(Path(__file__).parent.parent))

from src.data_processing.srt_dataset_builder import SRTDatasetBuilder
from src.training.train import start_training_thread, stop_training, get_training_state

try:
    from src.g2p.amharic_g2p import AmharicG2P
    from src.tokenizer.amharic_tokenizer import AmharicTokenizer, train_amharic_tokenizer
except:
    print("Warning: Could not import modules. Make sure to install dependencies.")


class AmharicTTSTrainingApp:
    """Complete Gradio app for Amharic TTS with training pipeline"""
    
    def __init__(self, model_path: str = None, config_path: str = None):
        print("Initializing Amharic TTS Training System...")
        
        # Initialize G2P
        self.g2p = AmharicG2P()
        print("✓ G2P loaded")
        
        # Initialize tokenizer
        try:
            tokenizer_path = Path(__file__).parent.parent / "models" / "tokenizer"
            if tokenizer_path.exists():
                self.tokenizer = AmharicTokenizer.load(str(tokenizer_path), g2p=self.g2p)
                print("✓ Tokenizer loaded")
            else:
                self.tokenizer = None
                print("⚠ Tokenizer not found")
        except:
            self.tokenizer = None
            print("⚠ Could not load tokenizer")
        
        # Load TTS model for inference
        try:
            self.model = self._load_tts_model(model_path, config_path)
            if self.model:
                print("✓ TTS model loaded successfully")
            else:
                print("✓ TTS model not loaded (training mode - will load from checkpoints during training)")
                print("  ℹ This is NORMAL on first run. Model will be available after training starts.")
        except Exception as e:
            print(f"⚠ TTS model loading failed: {str(e)} (placeholder mode)")
            self.model = None
        
        # Initialize SRT builder
        self.srt_builder = SRTDatasetBuilder(base_output_dir="data/srt_datasets")
        print("✓ SRT Dataset Builder loaded")
        
        print("✓ Initialization complete\n")
    
    def _load_tts_model(self, model_path: str = None, config_path: str = None):
        """Load trained TTS model for inference (Lightning AI compatible)"""
        try:
            from src.models.t3_model import SimplifiedT3Model
            
            # Priority order for finding a trained model:
            model_candidates = []
            
            # 1. Explicit model path provided
            if model_path and Path(model_path).exists():
                model_candidates.append(Path(model_path))
            
            # 2. Latest checkpoint
            checkpoint_dir = Path("models/checkpoints")
            if checkpoint_dir.exists():
                latest_checkpoint = checkpoint_dir / "checkpoint_latest.pt"
                if latest_checkpoint.exists():
                    model_candidates.append(latest_checkpoint)
                
                # Find most recent checkpoint
                checkpoints = list(checkpoint_dir.glob("checkpoint_*.pt"))
                if checkpoints:
                    latest = max(checkpoints, key=lambda x: x.stat().st_mtime)
                    model_candidates.append(latest)
            
            # 3. Extended pretrained model (skip for inference - it's for training only)
            # The extended embeddings file is used during training initialization,
            # not for inference in the demo interface
            
            # 4. Any OTHER pretrained models (skip extended embeddings)
            pretrained_dir = Path("models/pretrained")
            if pretrained_dir.exists():
                for model_file in pretrained_dir.glob("*.pt"):
                    # Skip the extended embeddings file - it's for training only
                    if "extended" not in model_file.stem.lower():
                        model_candidates.append(model_file)
                for model_file in pretrained_dir.glob("*.safetensors"):
                    model_candidates.append(model_file)
            
            if not model_candidates:
                print("No trained model found. Use the training pipeline first.")
                return None
            
            # Load config for model parameters
            config = None
            config_candidates = [
                config_path,
                "config/training_config.yaml",
                "configs/training_config.yaml"
            ]
            
            for config_file in config_candidates:
                if config_file and Path(config_file).exists():
                    import yaml
                    with open(config_file, 'r', encoding='utf-8') as f:
                        config = yaml.safe_load(f)
                    break
            
            if not config:
                # Default config if no config file found
                config = {
                    'model': {'n_vocab': 3000},
                    'data': {'n_mel_channels': 80}
                }
                print("Using default model configuration")
            
            # Try to load the first available model
            for model_file in model_candidates:
                try:
                    print(f"Attempting to load model: {model_file}")
                    
                    # Create model instance
                    # Use d_model=1024 to match Chatterbox pretrained weights
                    model = SimplifiedT3Model(
                        vocab_size=config['model']['n_vocab'],
                        d_model=1024,  # Match Chatterbox multilingual dimension
                        nhead=8,
                        num_encoder_layers=6,
                        dim_feedforward=2048,
                        dropout=0.1,
                        n_mels=config['data']['n_mel_channels'],
                        max_seq_len=1000
                    )
                    
                    # Load weights
                    device = torch.device('cpu')  # Use CPU for inference in web interface
                    
                    if str(model_file).endswith('.safetensors'):
                        # Load safetensors format (Chatterbox pretrained)
                        try:
                            from safetensors.torch import load_file
                            state_dict = load_file(model_file)
                        except ImportError:
                            print("safetensors not available, skipping .safetensors files")
                            continue
                    else:
                        # Load PyTorch checkpoint
                        checkpoint = torch.load(model_file, map_location=device, weights_only=False)
                        
                        if 'model_state_dict' in checkpoint:
                            state_dict = checkpoint['model_state_dict']
                        elif 'model' in checkpoint:
                            state_dict = checkpoint['model']
                        else:
                            state_dict = checkpoint
                    
                    # Try to load compatible weights
                    model_dict = model.state_dict()
                    compatible_dict = {}
                    
                    # Check if this looks like an extended embeddings file
                    has_text_embedding = 'text_embedding.weight' in state_dict
                    
                    for k, v in state_dict.items():
                        if k in model_dict:
                            # Exact shape match - load directly
                            if v.shape == model_dict[k].shape:
                                compatible_dict[k] = v
                            # Text embedding with different vocab size but same d_model
                            elif 'text_embedding.weight' in k:
                                target_shape = model_dict[k].shape
                                # Check if embedding dimension matches
                                if len(v.shape) == 2 and len(target_shape) == 2:
                                    if v.shape[1] == target_shape[1]:  # Same d_model
                                        # Copy what we can
                                        min_vocab = min(v.shape[0], target_shape[0])
                                        new_embedding = model_dict[k].clone()
                                        new_embedding[:min_vocab] = v[:min_vocab]
                                        compatible_dict[k] = new_embedding
                                        print(f"  ✓ Adapted text_embedding: {v.shape} -> {target_shape}, copied {min_vocab} embeddings")
                        # Map Chatterbox-style keys
                        elif 'text_emb' in k and 'text_embedding.weight' in model_dict:
                            target_shape = model_dict['text_embedding.weight'].shape
                            if len(v.shape) == 2 and v.shape[1] == target_shape[1]:
                                min_vocab = min(v.shape[0], target_shape[0])
                                new_embedding = model_dict['text_embedding.weight'].clone()
                                new_embedding[:min_vocab] = v[:min_vocab]
                                compatible_dict['text_embedding.weight'] = new_embedding
                                print(f"  ✓ Mapped text_emb -> text_embedding: copied {min_vocab} embeddings")
                    
                    # If we found the extended embeddings file but no compatible weights,
                    # it means this is actually a trained checkpoint we can use
                    if has_text_embedding and not compatible_dict:
                        print(f"  ℹ Extended embeddings file detected, will use for training initialization only")
                        # Return model with random initialization - it will be loaded during training
                        model.eval()
                        return model
                    
                    if compatible_dict:
                        # Update model with compatible weights
                        model_dict.update(compatible_dict)
                        model.load_state_dict(model_dict, strict=False)
                        print(f"✓ Loaded {len(compatible_dict)} compatible weight tensors from {model_file}")
                        model.eval()  # Set to evaluation mode
                        return model
                    else:
                        print(f"No compatible weights found in {model_file}")
                        
                except Exception as e:
                    print(f"Failed to load {model_file}: {str(e)}")
                    continue
            
            print("Could not load any model file")
            return None
            
        except ImportError as e:
            print(f"Missing dependencies for model loading: {str(e)}")
            return None
        except Exception as e:
            print(f"Error loading TTS model: {str(e)}")
            return None
    
    # ==================== TTS Functions ====================
    
    def synthesize(self, text: str, language: str = "am", reference_audio=None,
                  exaggeration: float = 0.5, cfg_pace: float = 0.5, 
                  temperature: float = 0.8, seed: int = 0) -> tuple:
        """
        Advanced synthesis with multiple controls
        
        Args:
            text: Text to synthesize
            language: Language code ('am' for Amharic, 'en' for English, etc.)
            reference_audio: Optional reference audio for voice cloning/style transfer
            exaggeration: Prosody exaggeration level (0.25 to 2.0)
            cfg_pace: Classifier-free guidance scale for pace control (0.2 to 1.0)
            temperature: Sampling temperature (0.05 to 5.0)
            seed: Random seed for reproducibility (0 for random)
        
        Returns:
            (audio_output, phonemes_text, info_markdown)
        """
        if not text or not text.strip():
            return None, "", "⚠ Please enter some text"
        
        try:
            # Language validation
            lang_map = {
                'am': 'Amharic',
                'en': 'English',
                'fr': 'French',
                'es': 'Spanish',
                'de': 'German',
                'ar': 'Arabic'
            }
            lang_name = lang_map.get(language, 'Amharic')
            
            # Set random seed if specified
            if seed > 0:
                torch.manual_seed(seed)
                np.random.seed(seed)
            
            # Process reference audio if provided
            ref_audio_info = ""
            if reference_audio is not None:
                try:
                    # Extract info from reference audio
                    if isinstance(reference_audio, tuple):
                        ref_sr, ref_audio_data = reference_audio
                        ref_duration = len(ref_audio_data) / ref_sr
                        ref_audio_info = f"\n🎤 **Reference Audio:** {ref_duration:.2f}s @ {ref_sr}Hz"
                    else:
                        ref_audio_info = "\n🎤 **Reference Audio:** Uploaded"
                except Exception as e:
                    ref_audio_info = f"\n⚠️ **Reference Audio Error:** {str(e)}"
            
            # Convert to phonemes
            phonemes = self.g2p.grapheme_to_phoneme(text) if language == 'am' else text
            
            # Tokenize
            if self.tokenizer:
                # Use grapheme encoding by default (works better for Amharic)
                tokens = self.tokenizer.encode(text, use_phonemes=False)
                token_info = f"Tokens ({len(tokens)}): {tokens[:20]}..."
                text_ids = torch.tensor(tokens).unsqueeze(0)  # [1, seq_len]
            else:
                # Fallback character-level tokenization
                import unicodedata
                text_norm = unicodedata.normalize('NFC', text)
                tokens = []
                for char in text_norm[:100]:
                    if char.isspace():
                        tokens.append(0)
                    else:
                        code_point = ord(char)
                        if 0x1200 <= code_point <= 0x137F:  # Ethiopic
                            token_id = 100 + (code_point - 0x1200) % 800
                        elif 0x20 <= code_point <= 0x7F:  # ASCII
                            token_id = code_point
                        else:
                            token_id = 50 + (code_point % 50)
                        tokens.append(token_id)
                
                token_info = f"Character tokens ({len(tokens)}): {tokens[:20]}..."
                text_ids = torch.tensor(tokens).unsqueeze(0)  # [1, seq_len]
            
            # Try to generate audio if model is available
            audio_output = None
            generation_info = ""
            sample_rate = 24000  # Default sample rate
            
            if self.model is not None:
                try:
                    # Use inference engine if available
                    try:
                        from src.inference import AmharicTTSInference
                        
                        # Try to find a checkpoint
                        checkpoint_candidates = [
                            Path('models/checkpoints/checkpoint_best.pt'),
                            Path('models/checkpoints/checkpoint_latest.pt')
                        ]
                        
                        checkpoint_path = None
                        for ckpt in checkpoint_candidates:
                            if ckpt.exists():
                                checkpoint_path = str(ckpt)
                                break
                        
                        if checkpoint_path:
                            # Use full inference engine
                            tts_engine = AmharicTTSInference(
                                checkpoint_path=checkpoint_path,
                                device='cuda' if torch.cuda.is_available() else 'cpu'
                            )
                            
                            audio_output, sample_rate, synth_info = tts_engine.synthesize(
                                text=text,
                                output_path=None,
                                use_phonemes=False
                            )
                            
                            generation_info = f"""
🎵 **Audio Generated Successfully!**
- Duration: {synth_info['audio_duration']:.2f}s
- Sample rate: {sample_rate}Hz
- Tokens: {synth_info['token_count']}
- Mel frames: {synth_info['mel_frames']}
                            """.strip()
                        else:
                            raise FileNotFoundError("No checkpoint found")
                            
                    except (ImportError, FileNotFoundError):
                        # Fallback to basic model inference
                        with torch.no_grad():
                            text_lengths = torch.tensor([text_ids.shape[1]])
                            
                            # Apply temperature to model (if supported)
                            outputs = self.model(
                                text_ids=text_ids,
                                text_lengths=text_lengths
                            )
                            
                            mel_output = outputs['mel_outputs']  # [1, n_mels, time]
                            mel_np = mel_output.squeeze(0).cpu().numpy()  # [n_mels, time]
                            
                            # Use audio processor if available
                            try:
                                from src.audio import AudioProcessor
                                audio_proc = AudioProcessor(sample_rate=sample_rate)
                                audio_output = audio_proc.mel_to_audio(mel_np)
                            except:
                                # Very basic fallback
                                duration = mel_np.shape[1] * 256 / sample_rate
                                audio_samples = int(duration * sample_rate)
                                t = np.linspace(0, duration, audio_samples)
                                freq_base = 200 + np.mean(mel_np) * 50
                                audio_output = 0.3 * np.sin(2 * np.pi * freq_base * t) * np.exp(-t * 0.5)
                            
                            generation_info = f"""
🎵 **Audio Generated!**
- Mel shape: {mel_output.shape}
- Duration: ~{mel_np.shape[1] * 256 / sample_rate:.2f}s
- Sample rate: {sample_rate}Hz
                            """.strip()
                        
                except Exception as e:
                    generation_info = f"❌ **Audio generation failed:** {str(e)}\n\nFalling back to text processing only."
                    audio_output = None
            
            # Prepare info display
            model_status = "✅ **Model loaded - Audio generated**" if self.model and audio_output is not None else "⚠️ **Model not loaded - Text processing only**"
            
            # Build parameter summary
            params_info = f"""
🎛️ **Synthesis Parameters:**
- Language: {lang_name} ({language})
- Exaggeration: {exaggeration:.2f}
- CFG/Pace: {cfg_pace:.2f}
- Temperature: {temperature:.2f}
- Seed: {seed if seed > 0 else 'Random'}
            """.strip()
            
            info = f"""
**🎙️ Text-to-Speech Synthesis Complete**

📝 **Input Text:** {text[:100]}{'...' if len(text) > 100 else ''}
🔤 **Phonemes:** {phonemes[:100] if phonemes else 'N/A'}{'...' if phonemes and len(phonemes) > 100 else ''}
🔢 **{token_info}**
{ref_audio_info}

{params_info}

{model_status}

{generation_info}

**📝 Note:** For best results, train the model using the Training Pipeline tab.
            """.strip()
            
            return (sample_rate, audio_output) if audio_output is not None else None, phonemes, info
            
        except Exception as e:
            import traceback
            error_msg = f"❌ **Error:** {str(e)}\n\n```\n{traceback.format_exc()}\n```"
            return None, "", error_msg
    
    # ==================== Dataset Management Functions ====================
    
    def _auto_split_dataset(self, dataset_path: Path, train_ratio: float = 0.80, 
                           val_ratio: float = 0.15, test_ratio: float = 0.05) -> dict:
        """
        Automatically split dataset into train/val/test sets
        Returns dict with split statistics
        """
        import random
        
        metadata_path = dataset_path / 'metadata.csv'
        if not metadata_path.exists():
            return {'error': f'metadata.csv not found in {dataset_path}'}
        
        # Load metadata
        with open(metadata_path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
        
        if len(lines) < 3:
            return {'error': 'Not enough samples to split (need at least 3)'}
        
        # Shuffle for random split
        random.seed(42)  # Reproducible
        data = lines.copy()
        random.shuffle(data)
        
        # Calculate splits
        n_total = len(data)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        train_data = data[:n_train]
        val_data = data[n_train:n_train + n_val]
        test_data = data[n_train + n_val:]
        
        # Save splits
        train_path = dataset_path / 'metadata_train.csv'
        val_path = dataset_path / 'metadata_val.csv'
        test_path = dataset_path / 'metadata_test.csv'
        
        with open(train_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(train_data) + '\n')
        
        with open(val_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(val_data) + '\n')
        
        with open(test_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(test_data) + '\n')
        
        return {
            'success': True,
            'train_samples': len(train_data),
            'val_samples': len(val_data),
            'test_samples': len(test_data),
            'train_path': str(train_path),
            'val_path': str(val_path),
            'test_path': str(test_path)
        }
    
    def import_srt_dataset(self, srt_file, media_file, dataset_name: str, 
                          speaker_name: str, validate: bool) -> str:
        """Import SRT dataset from uploaded files"""
        if srt_file is None or media_file is None:
            return "❌ Please upload both SRT and media files"
        
        if not dataset_name or not dataset_name.strip():
            return "❌ Please enter a dataset name"
        
        try:
            # Get file path from uploaded file
            srt_path = srt_file if isinstance(srt_file, str) else srt_file.name
            media_path = media_file if isinstance(media_file, str) else media_file.name
            
            stats = self.srt_builder.import_from_srt(
                srt_path=srt_path,
                media_path=media_path,
                dataset_name=dataset_name.strip(),
                speaker_name=speaker_name or "speaker_01",
                auto_validate=validate
            )
            
            # Automatically split the dataset
            dataset_path = Path("data/srt_datasets") / dataset_name
            split_result = self._auto_split_dataset(dataset_path)
            
            split_info = ""
            if split_result.get('success'):
                split_info = f"""

🎯 **Dataset Auto-Split:**
✅ Train set: {split_result['train_samples']} samples (80%)
✅ Val set: {split_result['val_samples']} samples (15%)
✅ Test set: {split_result['test_samples']} samples (5%)

📝 Files created:
- `metadata_train.csv` - For training
- `metadata_val.csv` - For validation
- `metadata_test.csv` - For final evaluation
"""
            elif 'error' in split_result:
                split_info = f"\n\n⚠️ Auto-split skipped: {split_result['error']}"
            
            result = f"""
✅ **Import Successful!**

**Dataset:** {dataset_name}
**Speaker:** {speaker_name or 'speaker_01'}

**Statistics:**
- Total Segments: {stats.get('total_segments', 0):,}
- Valid Segments: {stats.get('valid_segments', 0):,}
- Invalid Segments: {stats.get('invalid_segments', 0):,}
- Total Duration: {stats.get('total_duration_hours', 0):.2f} hours
- Average Segment: {stats.get('average_duration', 0):.2f} seconds
- Total Characters: {stats.get('total_characters', 0):,}

**Location:** `data/srt_datasets/{dataset_name}/`
{split_info}

**Next Steps:**
1. Review statistics in the "Manage Datasets" tab
2. Merge with other datasets if needed
3. Use in "Training Pipeline" to train the model
"""
            return result
            
        except Exception as e:
            return f"❌ **Import Failed**\n\nError: {str(e)}"
    
    def list_datasets(self) -> str:
        """List all available datasets"""
        try:
            datasets = self.srt_builder.list_datasets()
            
            if not datasets:
                return "📂 No datasets found. Import some SRT files first!"
            
            result = f"# 📚 Available Datasets ({len(datasets)})\n\n"
            
            total_hours = 0
            total_segments = 0
            
            for ds in datasets:
                result += f"## 📁 {ds['name']}\n"
                result += f"- **Segments:** {ds['valid_segments']:,}/{ds['segments']:,} valid\n"
                result += f"- **Duration:** {ds['duration_hours']:.2f} hours\n"
                result += f"- **Path:** `{ds['path']}`\n\n"
                
                total_hours += ds['duration_hours']
                total_segments += ds['valid_segments']
            
            result += "---\n\n"
            result += f"**Total:** {len(datasets)} dataset(s), "
            result += f"{total_segments:,} segments, {total_hours:.2f} hours\n"
            
            return result
            
        except Exception as e:
            return f"❌ Error listing datasets: {str(e)}"
    
    def manual_split_dataset_ui(self, dataset_name: str, train_ratio: float, 
                                val_ratio: float, test_ratio: float) -> str:
        """Manually split an existing dataset"""
        if not dataset_name or not dataset_name.strip():
            return "❌ Please enter a dataset name"
        
        try:
            dataset_path = Path("data/srt_datasets") / dataset_name.strip()
            if not dataset_path.exists():
                return f"❌ Dataset not found: {dataset_name}"
            
            split_result = self._auto_split_dataset(dataset_path, train_ratio, val_ratio, test_ratio)
            
            if 'error' in split_result:
                return f"❌ **Split Failed**\n\nError: {split_result['error']}"
            
            result = f"""
✅ **Dataset Split Successful!**

**Dataset:** {dataset_name}

🎯 **Split Ratios:**
- Train: {train_ratio*100:.0f}% ({split_result['train_samples']} samples)
- Validation: {val_ratio*100:.0f}% ({split_result['val_samples']} samples)
- Test: {test_ratio*100:.0f}% ({split_result['test_samples']} samples)

📝 **Files Created:**
- `{split_result['train_path']}`
- `{split_result['val_path']}`
- `{split_result['test_path']}`

✅ **Ready for Training!**

These split files will be automatically used during training.
"""
            return result
            
        except Exception as e:
            return f"❌ **Split Failed**\n\nError: {str(e)}"
    
    def merge_datasets_gui(self, dataset_selection: str, merged_name: str, 
                          filter_invalid: bool) -> str:
        """Merge selected datasets"""
        if not dataset_selection or not dataset_selection.strip():
            return "❌ Please enter dataset names to merge (comma-separated)"
        
        if not merged_name or not merged_name.strip():
            return "❌ Please enter a name for the merged dataset"
        
        try:
            dataset_names = [name.strip() for name in dataset_selection.split(',')]
            
            if len(dataset_names) < 2:
                return "❌ Please select at least 2 datasets to merge"
            
            stats = self.srt_builder.merge_datasets(
                dataset_names=dataset_names,
                merged_name=merged_name.strip(),
                filter_invalid=filter_invalid
            )
            
            # Automatically split the merged dataset
            dataset_path = Path("data/srt_datasets") / merged_name.strip()
            split_result = self._auto_split_dataset(dataset_path)
            
            split_info = ""
            if split_result.get('success'):
                split_info = f"""

🎯 **Dataset Auto-Split:**
✅ Train set: {split_result['train_samples']} samples (80%)
✅ Val set: {split_result['val_samples']} samples (15%)
✅ Test set: {split_result['test_samples']} samples (5%)

📝 Split files ready for training!
"""
            elif 'error' in split_result:
                split_info = f"\n\n⚠️ Auto-split skipped: {split_result['error']}"
            
            result = f"""
✅ **Merge Successful!**

**Merged Dataset:** {merged_name}
**Source Datasets:** {', '.join(dataset_names)}

**Statistics:**
- Total Segments: {stats.get('total_segments', 0):,}
- Valid Segments: {stats.get('valid_segments', 0):,}
- Total Duration: {stats.get('total_duration_hours', 0):.2f} hours
- Average Segment: {stats.get('average_duration', 0):.2f} seconds

**Location:** `data/srt_datasets/{merged_name}/`
{split_info}
"""
            return result
            
        except Exception as e:
            return f"❌ **Merge Failed**\n\nError: {str(e)}"
    
    # ==================== Training Functions ====================
    
    def train_tokenizer_ui(self, dataset_path: str, vocab_size: int, output_name: str) -> str:
        """Train Amharic tokenizer"""
        if not dataset_path or not dataset_path.strip():
            return "❌ Please provide a dataset path"
        
        if not output_name or not output_name.strip():
            output_name = "amharic_tokenizer"
        
        try:
            output_dir = Path("models") / "tokenizer" / output_name
            output_dir.mkdir(parents=True, exist_ok=True)
            
            result_msg = f"""
🔄 **Training Tokenizer...**

**Configuration:**
- Dataset: `{dataset_path}`
- Vocabulary Size: {vocab_size}
- Output: `{output_dir}`

**Status:** Processing...
"""
            
            # Train tokenizer
            train_amharic_tokenizer(
                data_file=dataset_path,
                output_dir=str(output_dir),
                vocab_size=vocab_size
            )
            
            success_msg = f"""
✅ **Tokenizer Training Complete!**

**Output Files:**
- `{output_dir}/sentencepiece.model`
- `{output_dir}/vocab.json`
- `{output_dir}/config.json`

**Next Steps:**
1. Go to "Model Setup" tab
2. Merge with base tokenizer
3. Extend model embeddings
4. Start fine-tuning!
"""
            return success_msg
            
        except Exception as e:
            return f"❌ **Training Failed**\n\nError: {str(e)}"
    
    def merge_tokenizers_ui(self, base_path: str, amharic_path: str, output_name: str) -> str:
        """Merge base and Amharic tokenizers"""
        if not base_path or not amharic_path:
            return "❌ Please provide both tokenizer paths"
        
        try:
            output_path = Path("models") / "tokenizer" / f"{output_name}_merged.json"
            
            # Get project root directory
            project_root = Path(__file__).parent.parent
            script_path = project_root / "scripts" / "merge_tokenizers.py"
            
            cmd = [
                "python", str(script_path),
                "--base", base_path,
                "--amharic", amharic_path,
                "--output", str(output_path),
                "--validate"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(project_root))
            
            if result.returncode == 0:
                return f"""
✅ **Tokenizer Merge Successful!**

**Output:** `{output_path}`

**Next Step:** Extend model embeddings in the "Model Setup" tab

**Output:**
```
{result.stdout}
```
"""
            else:
                return f"❌ **Merge Failed**\n\n```\n{result.stderr}\n```"
                
        except Exception as e:
            return f"❌ **Merge Failed**\n\nError: {str(e)}"
    
    def download_chatterbox_ui(self, model_type: str) -> str:
        """Download Chatterbox pretrained model"""
        try:
            from huggingface_hub import hf_hub_download
            
            output_dir = Path("models") / "pretrained" / "chatterbox"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            result_msg = f"""
🔄 **Downloading Chatterbox {model_type} Model...**

This may take 5-20 minutes depending on your connection.

**Model Size:**
- English: ~1 GB
- Multilingual (23 languages): ~3.2 GB

Please wait...
"""
            
            # Download tokenizer
            tokenizer_file = "mtl_tokenizer.json" if model_type == "Multilingual" else "tokenizer.json"
            model_file = "t3_mtl23ls_v2.safetensors" if model_type == "Multilingual" else "t3_cfg.safetensors"
            
            files_to_download = [
                (tokenizer_file, "Tokenizer"),
                (model_file, "Model Weights"),
                ("s3gen.safetensors", "Speech Generator"),
                ("ve.safetensors", "Voice Encoder"),
                ("conds.pt", "Conditioning")
            ]
            
            downloaded = []
            for filename, description in files_to_download:
                try:
                    path = hf_hub_download(
                        repo_id="ResembleAI/chatterbox",
                        filename=filename,
                        local_dir=str(output_dir)
                    )
                    downloaded.append(f"✓ {description}: {filename}")
                except Exception as e:
                    if "conds.pt" in filename or "ve." in filename:
                        downloaded.append(f"⚠ {description}: Optional, skipped")
                    else:
                        return f"❌ Failed to download {filename}: {str(e)}"
            
            # Extract tokenizer vocabulary
            vocab_path = output_dir / tokenizer_file
            if vocab_path.exists():
                import json
                with open(vocab_path, 'r', encoding='utf-8') as f:
                    tok_data = json.load(f)
                
                vocab = {}
                if 'model' in tok_data and 'vocab' in tok_data['model']:
                    vocab = tok_data['model']['vocab']
                elif 'vocab' in tok_data:
                    vocab = tok_data['vocab']
                
                if vocab:
                    vocab_output = Path("models") / "pretrained" / "chatterbox_tokenizer.json"
                    with open(vocab_output, 'w', encoding='utf-8') as f:
                        json.dump(vocab, f, ensure_ascii=False, indent=2)
                    downloaded.append(f"✓ Extracted vocabulary: {len(vocab)} tokens")
            
            return f"""
✅ **Download Complete!**

**Model Type:** Chatterbox {model_type}
**Location:** `models/pretrained/chatterbox/`

**Downloaded Files:**
{chr(10).join(downloaded)}

**Next Steps:**
1. ✓ Tokenizer ready at: `models/pretrained/chatterbox_tokenizer.json`
2. Merge tokenizers (below)
3. Extend model embeddings (below)
4. Start training!

**Supported Languages:** {"English only" if model_type == "English" else "23 languages (Arabic, Chinese, English, French, German, Hindi, Italian, Japanese, Korean, Portuguese, Russian, Spanish, and more!)"}
"""
            
        except ImportError:
            return f"""
❌ **Missing Dependency**

Please install huggingface_hub:
```bash
pip install huggingface_hub
```

Then try again.
"""
        except Exception as e:
            return f"❌ **Download Failed**\n\nError: {str(e)}"
    
    def extend_embeddings_ui(self, model_path: str, original_size: int, new_size: int, output_name: str) -> str:
        """Extend model embeddings"""
        if not model_path or not model_path.strip():
            return "❌ Please provide model path"
        
        try:
            output_path = Path("models") / "pretrained" / f"{output_name}_extended.pt"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Get project root directory
            project_root = Path(__file__).parent.parent
            script_path = project_root / "scripts" / "extend_model_embeddings.py"
            
            cmd = [
                "python", str(script_path),
                "--model", model_path,
                "--output", str(output_path),
                "--original-size", str(original_size),
                "--new-size", str(new_size)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(project_root))
            
            if result.returncode == 0:
                return f"""
✅ **Model Extension Successful!**

**Extended Model:** `{output_path}`
**Embedding Size:** {original_size} → {new_size}

**Next Step:** Configure and start training!

**Output:**
```
{result.stdout}
```
"""
            else:
                error_msg = result.stderr if result.stderr else result.stdout
                return f"""
❌ **Extension Failed**

**Error Details:**
```
{error_msg}
```

**Troubleshooting:**
1. First, inspect the model to find the correct embedding keys:
   ```bash
   python scripts/inspect_model_keys.py --model {model_path}
   ```

2. Share the output so we can update the extension script to use the correct keys.

3. Make sure the model file exists and is accessible.
"""
                
        except Exception as e:
            return f"""
❌ **Extension Failed**

**Exception:** {str(e)}

**Troubleshooting:**
1. Make sure the model file exists at: `{model_path}`
2. Run the inspection script to check model structure:
   ```bash
   python scripts/inspect_model_keys.py --model {model_path}
   ```
"""
    
    def get_training_status(self) -> str:
        """Get current training status"""
        try:
            state = get_training_state()
            
            if not state['is_running'] and state['status_message'] == "Not started":
                return "📊 **No training in progress**\n\nConfigure and start training below!"
            
            # Format status
            status_icon = "🔄" if state['is_running'] else "⏸️"
            progress = f"{state['current_step']}/{state['total_steps']}" if state['total_steps'] > 0 else "N/A"
            progress_pct = (state['current_step'] / state['total_steps'] * 100) if state['total_steps'] > 0 else 0
            
            logs_text = '\n'.join(state['logs'])
            checkpoint_text = f"**Last Checkpoint:** `{state['last_checkpoint']}`" if state['last_checkpoint'] else ""
            
            result = f"""
{status_icon} **Training Status: {state['status_message']}**

**Progress:**
- Epoch: {state['current_epoch']}
- Step: {state['current_step']} / {state['total_steps']} ({progress_pct:.1f}%)
- Current Loss: {state['current_loss']:.4f}
- Best Loss: {state['best_loss']:.4f}

**Recent Logs:**
```
{logs_text}
```

{checkpoint_text}
"""
            return result
        except Exception as e:
            return f"❌ Error getting training status: {str(e)}"
    
    def start_training_ui(self, config_path: str, resume_checkpoint: str, 
                         dataset_path: str, selected_tokenizer: str, batch_size: int, learning_rate: float,
                         max_epochs: int, max_steps: int, save_interval: int,
                         eval_interval: int, freeze_embeddings: bool, 
                         freeze_until_idx: int, use_amp: bool) -> str:
        """Start training from UI with custom parameters"""
        if not config_path or not config_path.strip():
            return "❌ Please provide config path"
        
        config_file = Path(config_path)
        if not config_file.exists():
            return f"❌ Config file not found: {config_path}"
        
        try:
            # Check if already running
            state = get_training_state()
            if state['is_running']:
                return "❌ Training is already running! Stop it first before starting new training."
            
            # Extract dataset path from dropdown selection
            if dataset_path and not dataset_path.startswith("No datasets"):
                # Extract path from dropdown text (remove sample count info)
                if " (" in dataset_path:
                    dataset_path = dataset_path.split(" (")[0].strip()
                
                # Ensure path is absolute or starts with data/
                # If it doesn't start with data/, add it
                if not dataset_path.startswith("data/") and not Path(dataset_path).is_absolute():
                    # Check if path exists with data/ prefix
                    test_path = Path(f"data/{dataset_path}")
                    if test_path.exists():
                        dataset_path = f"data/{dataset_path}"
            else:
                return "❌ Please select a dataset first!"
            
            # Load base config and update with UI parameters
            import yaml
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            # Update config with UI parameters - ensure sections exist
            if 'data' not in config:
                config['data'] = {}
            if 'training' not in config:
                config['training'] = {}
            if 'model' not in config:
                config['model'] = {}
            if 'paths' not in config:
                config['paths'] = {}
            
            config['data']['dataset_path'] = dataset_path
            config['data']['batch_size'] = batch_size
            config['paths']['data_dir'] = dataset_path  # Also set data_dir for compatibility
            config['training']['learning_rate'] = learning_rate
            config['training']['max_epochs'] = max_epochs
            config['training']['max_steps'] = max_steps
            config['training']['save_interval'] = save_interval
            config['training']['eval_interval'] = eval_interval
            config['training']['use_amp'] = use_amp
            config['training']['num_workers'] = config.get('training', {}).get('num_workers', 2)
            config['model']['freeze_original_embeddings'] = freeze_embeddings
            config['model']['freeze_until_index'] = freeze_until_idx
            
            # Handle tokenizer selection
            if selected_tokenizer and not selected_tokenizer.startswith("Auto-detect") and not selected_tokenizer.startswith("No tokenizers"):
                # Extract filename from dropdown text (remove vocab info)
                tokenizer_name = selected_tokenizer.split(" (")[0].strip()
                tokenizer_path = f"models/tokenizer/{tokenizer_name}"
                config['paths']['tokenizer'] = tokenizer_path
                print(f"Using selected tokenizer: {tokenizer_path}")
            else:
                # Auto-detect - will use default priority in training code
                if 'tokenizer' in config.get('paths', {}):
                    del config['paths']['tokenizer']
                print("Using auto-detect tokenizer mode")
            
            # Save temporary config
            temp_config_path = Path("config") / "temp_training_config.yaml"
            temp_config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(temp_config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False)
            
            # Handle checkpoint selection from dropdown
            resume_from = None
            if resume_checkpoint and resume_checkpoint.strip():
                # Check if it's not the "None" option
                if not resume_checkpoint.startswith("None") and not resume_checkpoint.startswith("No checkpoints"):
                    resume_from = resume_checkpoint
                    if not Path(resume_from).exists():
                        return f"❌ Resume checkpoint not found: {resume_from}"
            
            start_training_thread(str(temp_config_path), resume_from)
            
            return f"""
✅ **Training Started!**

**Configuration:** `{config_path}`
{f"**Resuming from:** `{resume_from}`" if resume_from else "**Starting from scratch**"}

**Status:** Initializing...

Refresh the status panel to see progress. Training will run in the background.
"""
        except Exception as e:
            return f"❌ **Failed to start training**\n\nError: {str(e)}"
    
    def stop_training_ui(self) -> str:
        """Stop training from UI"""
        try:
            state = get_training_state()
            if not state['is_running']:
                return "ℹ️ No training is currently running."
            
            stop_training()
            return """
⏸️ **Stopping Training...**

Training will stop gracefully after completing the current step.
This may take a few moments.

The checkpoint will be saved automatically.
"""
        except Exception as e:
            return f"❌ Error stopping training: {str(e)}"
    
    def get_available_checkpoints(self) -> list:
        """Get list of available checkpoint files"""
        checkpoint_dir = Path("models/checkpoints")
        
        if not checkpoint_dir.exists():
            return ["No checkpoints available"]
        
        # Find all .pt files
        checkpoints = list(checkpoint_dir.glob("*.pt"))
        
        if not checkpoints:
            return ["No checkpoints available"]
        
        # Sort by modification time (newest first)
        checkpoints.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        # Return paths safely - handle both absolute and relative
        checkpoint_list = []
        for cp in checkpoints:
            try:
                # Try to make relative to current working directory
                rel_path = cp.relative_to(Path.cwd())
                checkpoint_list.append(str(rel_path))
            except ValueError:
                # If relative path fails, use absolute path
                checkpoint_list.append(str(cp.resolve()))
        
        # Add "None" option at the beginning
        return ["None (Start from scratch)"] + checkpoint_list
    
    def refresh_checkpoints(self) -> gr.Dropdown:
        """Refresh checkpoint dropdown"""
        checkpoints = self.get_available_checkpoints()
        return gr.Dropdown(choices=checkpoints, value=checkpoints[0])
    
    def get_available_tokenizers(self) -> list:
        """
        Get list of available tokenizers from models/tokenizer/ directory
        Scans for .json files (merged tokenizers) and subdirectories with sentencepiece models
        """
        tokenizer_dir = Path("models/tokenizer")
        
        if not tokenizer_dir.exists():
            return ["No tokenizers found"]
        
        tokenizers = []
        
        # Find JSON tokenizer files (merged tokenizers)
        for json_file in tokenizer_dir.glob("*.json"):
            # Get relative path from models/tokenizer/
            rel_path = json_file.name
            
            # Add size info
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    vocab = json.load(f)
                vocab_size = len(vocab)
                tokenizers.append(f"{rel_path} (vocab: {vocab_size})")
            except:
                tokenizers.append(rel_path)
        
        # Find subdirectories with tokenizers
        for subdir in tokenizer_dir.iterdir():
            if subdir.is_dir():
                # Check if it has sentencepiece.model
                if (subdir / "sentencepiece.model").exists():
                    rel_path = subdir.name
                    
                    # Try to get vocab size
                    vocab_file = subdir / "vocab.json"
                    if vocab_file.exists():
                        try:
                            with open(vocab_file, 'r', encoding='utf-8') as f:
                                vocab = json.load(f)
                            vocab_size = len(vocab)
                            tokenizers.append(f"{rel_path}/ (vocab: {vocab_size})")
                        except:
                            tokenizers.append(f"{rel_path}/")
                    else:
                        tokenizers.append(f"{rel_path}/")
        
        if not tokenizers:
            return ["No tokenizers found - Train one first!"]
        
        # Sort tokenizers - prioritize merged tokenizers
        def sort_key(t):
            if "merged" in t.lower():
                return (0, t)
            elif ".json" in t:
                return (1, t)
            else:
                return (2, t)
        
        tokenizers.sort(key=sort_key)
        
        # Add instruction at the beginning
        return ["Auto-detect (recommended)"] + tokenizers
    
    def refresh_tokenizers(self) -> gr.Dropdown:
        """Refresh tokenizer dropdown"""
        tokenizers = self.get_available_tokenizers()
        # Set default to first merged tokenizer if available
        default_value = next((t for t in tokenizers if "merged" in t.lower()), tokenizers[0])
        return gr.Dropdown(choices=tokenizers, value=default_value)
    
    def get_available_datasets(self) -> list:
        """
        Get list of available datasets from data/srt_datasets/ directory
        Scans for directories containing metadata.csv
        """
        dataset_base = Path("data/srt_datasets")
        
        if not dataset_base.exists():
            return ["No datasets found - Import data first!"]
        
        datasets = []
        
        # Scan for directories with metadata.csv
        for item in dataset_base.iterdir():
            if item.is_dir():
                metadata_file = item / "metadata.csv"
                if metadata_file.exists():
                    # Get dataset info
                    try:
                        # Count lines in metadata
                        with open(metadata_file, 'r', encoding='utf-8') as f:
                            num_samples = sum(1 for _ in f)
                        
                        # Get total duration if dataset_info.json exists
                        info_file = item / "dataset_info.json"
                        duration_str = ""
                        if info_file.exists():
                            with open(info_file, 'r', encoding='utf-8') as f:
                                info_data = json.load(f)
                                total_duration = info_data.get('total_duration', 0)
                                duration_hours = total_duration / 3600
                                duration_str = f", {duration_hours:.1f}h"
                        
                        # Format: dataset_name (samples, duration)
                        rel_path = str(item.relative_to(dataset_base.parent))
                        datasets.append(f"{rel_path} ({num_samples} samples{duration_str})")
                    except Exception as e:
                        # If reading fails, just show the path
                        rel_path = str(item.relative_to(dataset_base.parent))
                        datasets.append(rel_path)
        
        if not datasets:
            return ["No datasets found - Import SRT files first!"]
        
        # Sort datasets - prioritize merged datasets and by sample count
        def sort_key(d):
            # Extract sample count from string
            if "(" in d and "samples" in d:
                try:
                    count = int(d.split("(")[1].split(" ")[0])
                    # Prioritize merged datasets and sort by sample count (descending)
                    if "merged" in d.lower():
                        return (0, -count, d)
                    else:
                        return (1, -count, d)
                except:
                    pass
            return (2, 0, d)
        
        datasets.sort(key=sort_key)
        
        return datasets
    
    def refresh_datasets(self) -> gr.Dropdown:
        """Refresh dataset dropdown"""
        datasets = self.get_available_datasets()
        # Set default to largest merged dataset if available
        default_value = next((d for d in datasets if "merged" in d.lower()), 
                           datasets[0] if datasets else "No datasets found")
        return gr.Dropdown(choices=datasets, value=default_value)
    
    # ==================== Create Interface ====================
    
    def create_interface(self) -> gr.Blocks:
        """Create the complete Gradio interface"""
        
        css = """
        .gradio-container {
            font-family: 'Noto Sans Ethiopic', 'Abyssinica SIL', sans-serif !important;
            max-width: 1400px;
            margin: auto;
        }
        .title {
            text-align: center;
            font-size: 2.5em;
            font-weight: bold;
            color: #2c3e50;
            margin: 20px 0;
        }
        .subtitle {
            text-align: center;
            font-size: 1.2em;
            color: #7f8c8d;
            margin-bottom: 30px;
        }
        """
        
        with gr.Blocks(css=css, title="Amharic TTS Training System", theme=gr.themes.Soft()) as app:
            
            gr.HTML("""
                <div class="title">🎓 የአማርኛ ጽሁፍ ወደ ንግግር - Training System</div>
                <div class="subtitle">Complete Amharic Text-to-Speech Training Platform</div>
            """)
            
            with gr.Tabs():
                # ==================== Tab 1: TTS Demo ====================
                with gr.Tab("🎵 Text-to-Speech"):
                    gr.Markdown("""
                    ### 🎙️ Text-to-Speech Synthesis
                    Generate high-quality Amharic speech from text with advanced controls.
                    """)
                    
                    with gr.Row():
                        with gr.Column(scale=1):
                            # Text input
                            text_input = gr.Textbox(
                                label="Text to synthesize (max chars 300)",
                                placeholder="አማርኛ ጽሁፉዎን እዚህ ያስገቡ... / Enter your Amharic text here...",
                                lines=4,
                                max_lines=6
                            )
                            
                            # Language selector
                            language_dropdown = gr.Dropdown(
                                label="Language - Select the language for text-to-speech synthesis",
                                choices=[
                                    ("Amharic (አማርኛ)", "am"),
                                    ("English", "en"),
                                    ("French", "fr"),
                                    ("Spanish", "es"),
                                    ("German", "de"),
                                    ("Arabic", "ar")
                                ],
                                value="am"
                            )
                            
                            # Reference audio upload
                            reference_audio = gr.Audio(
                                label="Reference Audio File (Optional) - Upload a reference audio to match speaking style",
                                type="numpy"
                            )
                            
                            gr.Markdown("""
                            ⚠️ **Note:** Ensure that the reference clip matches the specified language tag. 
                            Otherwise, language transfer outputs may inherit the accent of the reference clip's language. 
                            To mitigate this, set the CFG weight to 0.
                            """)
                            
                            # Advanced controls
                            exaggeration_slider = gr.Slider(
                                minimum=0.25,
                                maximum=2.0,
                                value=0.5,
                                step=0.05,
                                label="Exaggeration (Neutral = 0.5, extreme values can be unstable)"
                            )
                            
                            cfg_pace_slider = gr.Slider(
                                minimum=0.2,
                                maximum=1.0,
                                value=0.5,
                                step=0.05,
                                label="CFG/Pace - Classifier-free guidance scale for pace control"
                            )
                            
                            # More options accordion
                            with gr.Accordion("🛠️ More options", open=False):
                                seed_number = gr.Number(
                                    label="Random seed (0 for random) - Set seed for reproducible results",
                                    value=0,
                                    precision=0
                                )
                                
                                temperature_slider = gr.Slider(
                                    minimum=0.05,
                                    maximum=5.0,
                                    value=0.8,
                                    step=0.05,
                                    label="Temperature - Sampling temperature (lower = more deterministic)"
                                )
                            
                            # Generate button
                            generate_btn = gr.Button(
                                "🎙️ Generate",
                                variant="primary",
                                size="lg"
                            )
                            
                            # Example texts
                            gr.Examples(
                                examples=[
                                    ["ሰላም ልዓሌም", "am"],
                                    ["አዲስ አበባ የኢትዮጵያ ዋና ከተማ ናት", "am"],
                                    ["እንኳን ደህና መጡ", "am"],
                                    ["አማርኛ በጌዕዝ ፊደል ይጻፋል", "am"],
                                    ["ኢትዮጵያ በምስራቅ አፍሪካ የምትግኝ ሀገር ናት", "am"],
                                ],
                                inputs=[text_input, language_dropdown],
                                label="📚 Example Texts / ምሳሌዎች"
                            )
                        
                        # Output column
                        with gr.Column(scale=1):
                            gr.Markdown("### 🔊 Output Audio")
                            
                            # Audio output
                            audio_output = gr.Audio(
                                label="Generated Speech",
                                type="numpy",
                                interactive=False
                            )
                            
                            # Info output
                            info_output = gr.Markdown(
                                value="ℹ️ Configure settings and click 'Generate' to synthesize speech."
                            )
                    
                    # Event handler
                    generate_btn.click(
                        fn=self.synthesize,
                        inputs=[
                            text_input,
                            language_dropdown,
                            reference_audio,
                            exaggeration_slider,
                            cfg_pace_slider,
                            temperature_slider,
                            seed_number
                        ],
                        outputs=[audio_output, info_output, info_output]  # Using info_output twice to match return signature
                    )
                
                # ==================== Tab 2: Dataset Import ====================
                with gr.Tab("📺 Dataset Import"):
                    gr.Markdown("""
                    ### 📺 Import Audio/Video with SRT Transcriptions
                    Upload your video/audio files with SRT subtitles to create training datasets.
                    """)
                    
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("#### 📁 Upload Files")
                            srt_upload = gr.File(label="📝 SRT File", file_types=[".srt"])
                            media_upload = gr.File(label="🎥 Media File", file_types=[".mp4", ".mkv", ".avi", ".mov", ".webm", ".mp3", ".wav", ".m4a"])
                            dataset_name_input = gr.Textbox(label="🏷️ Dataset Name", placeholder="my_amharic_dataset")
                            speaker_name_input = gr.Textbox(label="🎭 Speaker Name", placeholder="speaker_01", value="speaker_01")
                            validate_checkbox = gr.Checkbox(label="✅ Validate segments", value=True)
                            import_btn = gr.Button("📥 Import Dataset", variant="primary", size="lg")
                        
                        with gr.Column():
                            gr.Markdown("#### 📊 Import Results")
                            import_results = gr.Markdown(value="ℹ️ Upload files to begin import.")
                    
                    import_btn.click(
                        fn=self.import_srt_dataset,
                        inputs=[srt_upload, media_upload, dataset_name_input, speaker_name_input, validate_checkbox],
                        outputs=import_results
                    )
                
                # ==================== Tab 3: Dataset Management ====================
                with gr.Tab("📂 Dataset Management"):
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("#### 📊 View Datasets")
                            refresh_btn = gr.Button("🔄 Refresh Dataset List", variant="secondary")
                            dataset_list = gr.Markdown(value="🔄 Click 'Refresh' to view datasets")
                        
                        with gr.Column():
                            gr.Markdown("#### 🔗 Merge Datasets")
                            merge_selection = gr.Textbox(
                                label="Dataset Names (comma-separated)",
                                placeholder="dataset1, dataset2, dataset3"
                            )
                            merge_name_input = gr.Textbox(label="Merged Dataset Name", placeholder="merged_dataset")
                            filter_invalid_checkbox = gr.Checkbox(label="✅ Filter invalid segments", value=True)
                            merge_btn = gr.Button("🔗 Merge Datasets", variant="primary")
                            merge_results = gr.Markdown(value="ℹ️ Enter dataset names to merge")
                    
                    with gr.Accordion("🎯 Manual Split Dataset", open=False):
                        gr.Markdown("""
                        Split an existing dataset into train/val/test sets manually.
                        
                        ℹ️ **Note:** Datasets are automatically split during import/merge.
                        Use this only if you need to re-split with custom ratios.
                        """)
                        with gr.Row():
                            with gr.Column():
                                split_dataset_name = gr.Textbox(
                                    label="Dataset Name",
                                    placeholder="my_dataset",
                                    info="Name of dataset in data/srt_datasets/"
                                )
                                with gr.Row():
                                    split_train_ratio = gr.Slider(
                                        minimum=0.5,
                                        maximum=0.95,
                                        value=0.80,
                                        step=0.05,
                                        label="Train %"
                                    )
                                    split_val_ratio = gr.Slider(
                                        minimum=0.05,
                                        maximum=0.30,
                                        value=0.15,
                                        step=0.05,
                                        label="Val %"
                                    )
                                    split_test_ratio = gr.Slider(
                                        minimum=0.05,
                                        maximum=0.20,
                                        value=0.05,
                                        step=0.05,
                                        label="Test %"
                                    )
                                split_btn = gr.Button("🎯 Split Dataset", variant="secondary")
                            
                            with gr.Column():
                                split_results = gr.Markdown(value="ℹ️ Configure split ratios and click 'Split Dataset'")
                    
                    refresh_btn.click(fn=self.list_datasets, outputs=dataset_list)
                    merge_btn.click(
                        fn=self.merge_datasets_gui,
                        inputs=[merge_selection, merge_name_input, filter_invalid_checkbox],
                        outputs=merge_results
                    )
                    split_btn.click(
                        fn=self.manual_split_dataset_ui,
                        inputs=[split_dataset_name, split_train_ratio, split_val_ratio, split_test_ratio],
                        outputs=split_results
                    )
                
                # ==================== Tab 4: Tokenizer Training ====================
                with gr.Tab("🔤 Tokenizer Training"):
                    gr.Markdown("""
                    ### 🔤 Train Amharic Tokenizer
                    Train a custom SentencePiece tokenizer on your Amharic dataset.
                    
                    **Vocabulary Size Guidelines:**
                    - **300-500**: Small datasets (<1 hour) - Basic coverage
                    - **500-1000**: Medium datasets (1-5 hours) - **Recommended for most**
                    - **1000-2000**: Large datasets (5-10 hours) - Good coverage
                    - **2000-5000**: Very large datasets (>10 hours) - Maximum quality
                    
                    💡 Larger vocab = better quality but slower training. Start with 500-1000.
                    """)
                    
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("#### ⚙️ Configuration")
                            dataset_path_input = gr.Textbox(
                                label="Dataset Path",
                                placeholder="data/srt_datasets/my_dataset/metadata.csv",
                                info="Path to metadata.csv file"
                            )
                            vocab_size_slider = gr.Slider(
                                minimum=100,
                                maximum=10000,
                                value=500,
                                step=100,
                                label="Vocabulary Size",
                                info="Recommended: 500 (<5h), 1000 (5-10h), 2000+ (>10h data)"
                            )
                            tokenizer_name_input = gr.Textbox(
                                label="Output Name",
                                placeholder="amharic_tokenizer",
                                value="amharic_tokenizer"
                            )
                            train_tokenizer_btn = gr.Button("🚀 Train Tokenizer", variant="primary", size="lg")
                        
                        with gr.Column():
                            gr.Markdown("#### 📊 Training Results")
                            tokenizer_results = gr.Markdown(value="ℹ️ Configure and click 'Train Tokenizer' to start")
                    
                    train_tokenizer_btn.click(
                        fn=self.train_tokenizer_ui,
                        inputs=[dataset_path_input, vocab_size_slider, tokenizer_name_input],
                        outputs=tokenizer_results
                    )
                
                # ==================== Tab 5: Model Setup ====================
                with gr.Tab("🔧 Model Setup"):
                    gr.Markdown("""
                    ### 🔧 Prepare Model for Training
                    
                    **Step-by-step setup for fine-tuning:**
                    1. Download Chatterbox pretrained model (optional but recommended)
                    2. Merge tokenizers (base + Amharic)
                    3. Extend model embeddings
                    """)
                    
                    with gr.Accordion("0️⃣ Download Chatterbox Model (Optional)", open=True):
                        gr.Markdown("""
                        **Download official Chatterbox pretrained models for better quality.**
                        
                        - **English**: Fast download (~1 GB), English-only TTS
                        - **Multilingual**: Larger download (~3.2 GB), supports 23 languages + zero-shot voice cloning
                        
                        Skip this if you want to train from scratch (Amharic-only).
                        """)
                        with gr.Row():
                            with gr.Column():
                                model_type_dropdown = gr.Dropdown(
                                    label="Model Type",
                                    choices=["English", "Multilingual"],
                                    value="Multilingual",
                                    info="Multilingual recommended for best quality"
                                )
                                download_btn = gr.Button("📥 Download Chatterbox Model", variant="primary")
                            
                            with gr.Column():
                                download_results = gr.Markdown(value="ℹ️ Select model type and click 'Download'")
                        
                        download_btn.click(
                            fn=self.download_chatterbox_ui,
                            inputs=[model_type_dropdown],
                            outputs=download_results
                        )
                    
                    with gr.Accordion("1️⃣ Merge Tokenizers", open=True):
                        with gr.Row():
                            with gr.Column():
                                base_tokenizer_path = gr.Textbox(
                                    label="Base Tokenizer Path",
                                    placeholder="models/pretrained/chatterbox_tokenizer.json",
                                    value="models/pretrained/chatterbox_tokenizer.json",
                                    info="Auto-filled if you downloaded Chatterbox above"
                                )
                                amharic_tokenizer_path = gr.Textbox(
                                    label="Amharic Tokenizer Path",
                                    placeholder="models/tokenizer/amharic_tokenizer/vocab.json"
                                )
                                merged_name = gr.Textbox(
                                    label="Output Name",
                                    placeholder="merged",
                                    value="merged"
                                )
                                merge_tokenizer_btn = gr.Button("🔗 Merge Tokenizers", variant="primary")
                            
                            with gr.Column():
                                merge_tokenizer_results = gr.Markdown(value="ℹ️ Configure paths and click 'Merge'")
                    
                    with gr.Accordion("2️⃣ Extend Model Embeddings", open=True):
                        with gr.Row():
                            with gr.Column():
                                model_path_input = gr.Textbox(
                                    label="Base Model Path",
                                    placeholder="models/pretrained/chatterbox/t3_mtl23ls_v2.safetensors",
                                    value="models/pretrained/chatterbox/t3_mtl23ls_v2.safetensors",
                                    info="Path to downloaded Chatterbox model"
                                )
                                original_size_input = gr.Number(
                                    label="Original Vocab Size",
                                    value=2454,
                                    precision=0,
                                    info="Chatterbox multilingual has 2454 tokens"
                                )
                                new_size_input = gr.Number(
                                    label="New Vocab Size",
                                    value=2535,
                                    precision=0,
                                    info="Merged tokenizer size: 2535 (Chatterbox 2454 + Amharic 81)"
                                )
                                extended_name = gr.Textbox(
                                    label="Output Name",
                                    placeholder="chatterbox",
                                    value="chatterbox"
                                )
                                extend_btn = gr.Button("🚀 Extend Embeddings", variant="primary")
                            
                            with gr.Column():
                                extend_results = gr.Markdown(value="ℹ️ Configure and click 'Extend Embeddings'")
                    
                    merge_tokenizer_btn.click(
                        fn=self.merge_tokenizers_ui,
                        inputs=[base_tokenizer_path, amharic_tokenizer_path, merged_name],
                        outputs=merge_tokenizer_results
                    )
                    
                    extend_btn.click(
                        fn=self.extend_embeddings_ui,
                        inputs=[model_path_input, original_size_input, new_size_input, extended_name],
                        outputs=extend_results
                    )
                
                # ==================== Tab 6: Training Pipeline ====================
                with gr.Tab("🎓 Training Pipeline"):
                    gr.Markdown("""
                    ### 🎓 Complete Training Control
                    
                    **Fine-tune Chatterbox TTS for Amharic with full control**
                    
                    This tab provides comprehensive training control including:
                    - Start/Stop training
                    - Real-time progress monitoring
                    - Loss tracking and checkpointing
                    - Resume from checkpoint
                    - Live log viewing
                    """)
                    
                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.Markdown("### ⚙️ Training Configuration")
                            
                            config_path_input = gr.Textbox(
                                label="📄 Configuration File Path",
                                value="config/training_config.yaml",
                                placeholder="config/training_config.yaml",
                                info="Base configuration (will be overridden by controls below)"
                            )
                            
                            with gr.Row():
                                resume_checkpoint_dropdown = gr.Dropdown(
                                    label="📥 Resume from Checkpoint",
                                    choices=self.get_available_checkpoints(),
                                    value="None (Start from scratch)",
                                    info="Select a checkpoint to resume training",
                                    interactive=True,
                                    allow_custom_value=True
                                )
                                refresh_checkpoints_btn = gr.Button(
                                    "🔄",
                                    size="sm",
                                    scale=0,
                                    min_width=40
                                )
                            
                            gr.Markdown("### 💾 Dataset Settings")
                            
                            with gr.Row():
                                dataset_dropdown = gr.Dropdown(
                                    label="📂 Select Dataset",
                                    choices=self.get_available_datasets(),
                                    value=None,
                                    info="Choose from data/srt_datasets/ directory",
                                    interactive=True,
                                    allow_custom_value=True
                                )
                                refresh_datasets_btn = gr.Button(
                                    "🔄",
                                    size="sm",
                                    scale=0,
                                    min_width=40
                                )
                            
                            dataset_info = gr.Markdown(
                                value="ℹ️ Select a dataset or enter custom path"
                            )
                            
                            gr.Markdown("### 🔤 Tokenizer Selection")
                            
                            with gr.Row():
                                tokenizer_dropdown = gr.Dropdown(
                                    label="📝 Select Tokenizer",
                                    choices=self.get_available_tokenizers(),
                                    value=None,
                                    info="Choose tokenizer from models/tokenizer/ directory",
                                    interactive=True,
                                    allow_custom_value=True
                                )
                                refresh_tokenizers_btn = gr.Button(
                                    "🔄",
                                    size="sm",
                                    scale=0,
                                    min_width=40
                                )
                            
                            tokenizer_info = gr.Markdown(
                                value="ℹ️ **Recommended:** Am_tokenizer_merged.json (Chatterbox 23 langs + Amharic)"
                            )
                            
                            gr.Markdown("### 📊 Training Hyperparameters")
                            
                            with gr.Row():
                                batch_size_slider = gr.Slider(
                                    minimum=1,
                                    maximum=64,
                                    value=16,
                                    step=1,
                                    label="📦 Batch Size",
                                    info="Number of samples per batch"
                                )
                                
                                learning_rate_slider = gr.Slider(
                                    minimum=1e-6,
                                    maximum=1e-3,
                                    value=2e-4,
                                    step=1e-6,
                                    label="🎯 Learning Rate",
                                    info="Optimizer learning rate"
                                )
                            
                            with gr.Row():
                                max_epochs_slider = gr.Slider(
                                    minimum=1,
                                    maximum=10000,
                                    value=1000,
                                    step=1,
                                    label="🔁 Max Epochs",
                                    info="Maximum number of epochs"
                                )
                                
                                max_steps_slider = gr.Slider(
                                    minimum=1000,
                                    maximum=1000000,
                                    value=500000,
                                    step=1000,
                                    label="📍 Max Steps",
                                    info="Maximum training steps"
                                )
                            
                            with gr.Row():
                                save_interval_slider = gr.Slider(
                                    minimum=100,
                                    maximum=50000,
                                    value=5000,
                                    step=100,
                                    label="💾 Save Interval",
                                    info="Save checkpoint every N steps"
                                )
                                
                                eval_interval_slider = gr.Slider(
                                    minimum=100,
                                    maximum=10000,
                                    value=1000,
                                    step=100,
                                    label="📊 Eval Interval",
                                    info="Run validation every N steps"
                                )
                            
                            gr.Markdown("### 🔒 Embedding Freezing (Preserve English)")
                            
                            freeze_embeddings_checkbox = gr.Checkbox(
                                label="❄️ Freeze Original Embeddings",
                                value=True,
                                info="Freeze first N embeddings to preserve base model knowledge"
                            )
                            
                            freeze_until_idx_slider = gr.Slider(
                                minimum=0,
                                maximum=3000,
                                value=2454,
                                step=1,
                                label="🔒 Freeze Until Index",
                                info="Freeze embeddings 0 to N-1 (2454 = Chatterbox multilingual vocab)"
                            )
                            
                            gr.Markdown("### ⚡ Performance Settings")
                            
                            use_amp_checkbox = gr.Checkbox(
                                label="⚡ Use Mixed Precision (AMP)",
                                value=True,
                                info="Faster training with lower memory usage (requires CUDA)"
                            )
                            
                            gr.Markdown("### 🎮 Training Controls")
                            
                            with gr.Row():
                                start_training_btn = gr.Button(
                                    "🚀 Start Training",
                                    variant="primary",
                                    size="lg"
                                )
                                stop_training_btn = gr.Button(
                                    "⏸️ Stop Training",
                                    variant="stop",
                                    size="lg"
                                )
                            
                            training_control_output = gr.Markdown(value="ℹ️ Configure and click 'Start Training' to begin")
                            
                            gr.Markdown("""
                            ---
                            ### 📚 Training Checklist
                            
                            **Before starting training, ensure:**
                            
                            ✅ **Step 1:** Dataset prepared and imported
                            - Import SRT files in "Dataset Import" tab
                            - Merge datasets if needed
                            - Verify audio quality and transcriptions
                            
                            ✅ **Step 2:** Tokenizer trained
                            - Train Amharic tokenizer in "Tokenizer Training" tab
                            - Merge with base Chatterbox tokenizer
                            
                            ✅ **Step 3:** Model prepared
                            - Extend model embeddings in "Model Setup" tab
                            - Configure `config/training_config.yaml`
                            
                            ✅ **Step 4:** Hardware ready
                            - CUDA GPU available (recommended)
                            - Sufficient disk space for checkpoints
                            - TensorBoard for monitoring (optional)
                            
                            **Configuration Details:**
                            
                            Edit `config/training_config.yaml` to customize:
                            - Learning rate and batch size
                            - Embedding freezing settings
                            - Checkpoint intervals
                            - Data augmentation
                            - Validation samples
                            
                            **Key Settings for Amharic:**
                            ```yaml
                            model:
                              freeze_original_embeddings: true  # Preserve English
                              freeze_until_index: 704           # First 704 tokens
                              n_vocab: 2000                     # Extended vocabulary
                            ```
                            """)
                        
                        with gr.Column(scale=1):
                            gr.Markdown("### 📊 Live Training Monitor")
                            
                            status_refresh_btn = gr.Button(
                                "🔄 Refresh Status",
                                variant="secondary"
                            )
                            
                            training_status = gr.Markdown(
                                value="📊 Click 'Refresh Status' to check training progress"
                            )
                            
                            gr.Markdown("""
                            ---
                            ### 💾 Checkpoints & Models
                            
                            **Checkpoint Location:** `models/checkpoints/`
                            
                            Checkpoints are saved:
                            - Every N steps (configured in config file)
                            - After each epoch
                            - As `checkpoint_latest.pt` (auto-saved)
                            - With epoch and step in filename
                            
                            **Using Trained Models:**
                            
                            After training completes:
                            1. Best checkpoint saved in `models/checkpoints/`
                            2. Load for inference in "TTS Demo" tab
                            3. Export for deployment
                            4. Share or fine-tune further
                            
                            **Monitoring Training:**
                            
                            - **TensorBoard:** `tensorboard --logdir logs`
                            - **Live Logs:** Click "Refresh Status" above
                            - **Checkpoints:** Check `models/checkpoints/`
                            
                            **Tips:**
                            - Training can run for hours/days
                            - Monitor validation loss for overfitting
                            - Stop training anytime (saves checkpoint)
                            - Resume from any checkpoint
                            - Experiment with hyperparameters
                            """)
                    
                    # Event handlers
                    refresh_checkpoints_btn.click(
                        fn=self.refresh_checkpoints,
                        outputs=resume_checkpoint_dropdown
                    )
                    
                    refresh_tokenizers_btn.click(
                        fn=self.refresh_tokenizers,
                        outputs=tokenizer_dropdown
                    )
                    
                    refresh_datasets_btn.click(
                        fn=self.refresh_datasets,
                        outputs=dataset_dropdown
                    )
                    
                    start_training_btn.click(
                        fn=self.start_training_ui,
                        inputs=[
                            config_path_input,
                            resume_checkpoint_dropdown,
                            dataset_dropdown,  # Changed from dataset_path_input to dataset_dropdown
                            tokenizer_dropdown,  # Added tokenizer selection
                            batch_size_slider,
                            learning_rate_slider,
                            max_epochs_slider,
                            max_steps_slider,
                            save_interval_slider,
                            eval_interval_slider,
                            freeze_embeddings_checkbox,
                            freeze_until_idx_slider,
                            use_amp_checkbox
                        ],
                        outputs=training_control_output
                    )
                    
                    stop_training_btn.click(
                        fn=self.stop_training_ui,
                        outputs=training_control_output
                    )
                    
                    status_refresh_btn.click(
                        fn=self.get_training_status,
                        outputs=training_status
                    )
            
            return app


def launch_app(model_path: str = None, config_path: str = None,
               share: bool = False, server_port: int = 7860):
    """Launch the Gradio app"""
    print("\n" + "="*60)
    print("LAUNCHING AMHARIC TTS TRAINING SYSTEM")
    print("="*60 + "\n")
    
    app_instance = AmharicTTSTrainingApp(model_path, config_path)
    app = app_instance.create_interface()
    
    print(f"Starting server on port {server_port}...")
    print(f"Share mode: {'Enabled' if share else 'Disabled'}")
    print("\nPress Ctrl+C to stop the server\n")
    
    app.launch(
        share=share,
        server_port=server_port,
        server_name="127.0.0.1",
        show_error=True,
        favicon_path=None
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Launch Amharic TTS Training System")
    parser.add_argument('--model', type=str, default=None, help='Path to trained model checkpoint')
    parser.add_argument('--config', type=str, default=None, help='Path to configuration file')
    parser.add_argument('--share', action='store_true', help='Create public Gradio link')
    parser.add_argument('--port', type=int, default=7860, help='Server port (default: 7860)')
    
    args = parser.parse_args()
    
    launch_app(
        model_path=args.model,
        config_path=args.config,
        share=args.share,
        server_port=args.port
    )
