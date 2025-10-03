"""
Complete Gradio Web Interface for Amharic TTS
Includes: TTS Demo, Dataset Management, Tokenizer Training, Model Setup, and Full Training Pipeline
"""

import gradio as gr
import sys
import os
import subprocess
import json
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
        
        # TODO: Load actual TTS model when available
        self.model = None
        print("⚠ TTS model not loaded (placeholder mode)")
        
        # Initialize SRT builder
        self.srt_builder = SRTDatasetBuilder(base_output_dir="data/srt_datasets")
        print("✓ SRT Dataset Builder loaded")
        
        print("✓ Initialization complete\n")
    
    # ==================== TTS Functions ====================
    
    def synthesize(self, text: str, speed: float = 1.0, pitch: float = 1.0) -> tuple:
        """Synthesize speech from Amharic text"""
        if not text or not text.strip():
            return None, "", "⚠ Please enter some text"
        
        try:
            phonemes = self.g2p.grapheme_to_phoneme(text)
            
            if self.tokenizer:
                tokens = self.tokenizer.encode(text, use_phonemes=True)
                token_info = f"Tokens ({len(tokens)}): {tokens[:20]}..."
            else:
                token_info = "Tokenizer not available"
            
            info = f"""
**Text Processing Complete**

📝 **Input Text:** {text}
🔤 **Phonemes (IPA):** {phonemes}
🔢 **{token_info}**
⚙️ **Settings:** Speed={speed}, Pitch={pitch}

⚠️ **Note:** Model not loaded. This is a demo of text processing only.
To generate actual audio, train the model first.
            """.strip()
            
            return None, phonemes, info
            
        except Exception as e:
            error_msg = f"❌ Error: {str(e)}"
            return None, "", error_msg
    
    # ==================== Dataset Management Functions ====================
    
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
            
            cmd = [
                "python", "scripts/merge_tokenizers.py",
                "--base", base_path,
                "--amharic", amharic_path,
                "--output", str(output_path),
                "--validate"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
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
    
    def extend_embeddings_ui(self, model_path: str, original_size: int, new_size: int, output_name: str) -> str:
        """Extend model embeddings"""
        if not model_path or not model_path.strip():
            return "❌ Please provide model path"
        
        try:
            output_path = Path("models") / "pretrained" / f"{output_name}_extended.pt"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            cmd = [
                "python", "scripts/extend_model_embeddings.py",
                "--model", model_path,
                "--output", str(output_path),
                "--original-size", str(original_size),
                "--new-size", str(new_size)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
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
                return f"❌ **Extension Failed**\n\n```\n{result.stderr}\n```"
                
        except Exception as e:
            return f"❌ **Extension Failed**\n\nError: {str(e)}"
    
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
                         dataset_path: str, batch_size: int, learning_rate: float,
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
            
            # Load base config and update with UI parameters
            import yaml
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            # Update config with UI parameters
            config['data']['dataset_path'] = dataset_path
            config['data']['batch_size'] = batch_size
            config['training']['learning_rate'] = learning_rate
            config['training']['max_epochs'] = max_epochs
            config['training']['max_steps'] = max_steps
            config['training']['save_interval'] = save_interval
            config['training']['eval_interval'] = eval_interval
            config['training']['use_amp'] = use_amp
            config['model']['freeze_original_embeddings'] = freeze_embeddings
            config['model']['freeze_until_index'] = freeze_until_idx
            
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
        
        # Return relative paths
        checkpoint_list = [str(cp.relative_to(Path.cwd())) for cp in checkpoints]
        
        # Add "None" option at the beginning
        return ["None (Start from scratch)"] + checkpoint_list
    
    def refresh_checkpoints(self) -> gr.Dropdown:
        """Refresh checkpoint dropdown"""
        checkpoints = self.get_available_checkpoints()
        return gr.Dropdown(choices=checkpoints, value=checkpoints[0])
    
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
                with gr.Tab("🎵 TTS Demo"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.Markdown("### 📝 Input")
                            
                            text_input = gr.Textbox(
                                label="የአማርኛ ጽሁፍ | Amharic Text",
                                placeholder="አማርኛ ጽሁፍዎን እዚህ ያስገቡ...",
                                lines=5
                            )
                            
                            gr.Examples(
                                examples=[
                                    ["ሰላም ለዓለም"],
                                    ["አዲስ አበባ የኢትዮጵያ ዋና ከተማ ናት"],
                                    ["እንኳን ደህና መጡ"],
                                ],
                                inputs=text_input
                            )
                            
                            with gr.Accordion("⚙️ Settings", open=False):
                                speed_slider = gr.Slider(0.5, 2.0, 1.0, 0.1, label="Speed")
                                pitch_slider = gr.Slider(0.5, 2.0, 1.0, 0.1, label="Pitch")
                            
                            generate_btn = gr.Button("🎙️ Generate Speech", variant="primary", size="lg")
                        
                        with gr.Column(scale=1):
                            gr.Markdown("### 🔊 Output")
                            audio_output = gr.Audio(label="Generated Audio")
                            phoneme_output = gr.Textbox(label="Phonemes (IPA)", lines=3, interactive=False)
                            info_output = gr.Markdown()
                    
                    generate_btn.click(
                        fn=self.synthesize,
                        inputs=[text_input, speed_slider, pitch_slider],
                        outputs=[audio_output, phoneme_output, info_output]
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
                    
                    refresh_btn.click(fn=self.list_datasets, outputs=dataset_list)
                    merge_btn.click(
                        fn=self.merge_datasets_gui,
                        inputs=[merge_selection, merge_name_input, filter_invalid_checkbox],
                        outputs=merge_results
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
                    Merge tokenizers and extend model embeddings.
                    """)
                    
                    with gr.Accordion("1️⃣ Merge Tokenizers", open=True):
                        with gr.Row():
                            with gr.Column():
                                base_tokenizer_path = gr.Textbox(
                                    label="Base Tokenizer Path",
                                    placeholder="models/pretrained/chatterbox_tokenizer.json"
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
                                    placeholder="models/pretrained/chatterbox_base.pt"
                                )
                                original_size_input = gr.Number(
                                    label="Original Vocab Size",
                                    value=704,
                                    precision=0
                                )
                                new_size_input = gr.Number(
                                    label="New Vocab Size",
                                    value=2000,
                                    precision=0
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
                            
                            dataset_path_input = gr.Textbox(
                                label="📂 Dataset Path",
                                value="data/srt_datasets/my_dataset",
                                placeholder="data/srt_datasets/my_dataset",
                                info="Path to your prepared dataset"
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
                                maximum=2000,
                                value=704,
                                step=1,
                                label="🔒 Freeze Until Index",
                                info="Freeze embeddings 0 to N-1 (704 = Chatterbox base vocab)"
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
                    
                    start_training_btn.click(
                        fn=self.start_training_ui,
                        inputs=[
                            config_path_input,
                            resume_checkpoint_dropdown,
                            dataset_path_input,
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
