"""
Gradio Web Interface for Amharic TTS
Clean, user-friendly interface with Amharic support
"""

import gradio as gr
import sys
from pathlib import Path

# Add project paths
sys.path.append(str(Path(__file__).parent.parent))

from src.data_processing.srt_dataset_builder import SRTDatasetBuilder

try:
    from src.g2p.amharic_g2p import AmharicG2P
    from src.tokenizer.amharic_tokenizer import AmharicTokenizer
except:
    print("Warning: Could not import modules. Make sure to install dependencies.")


class AmharicTTSApp:
    """Gradio app for Amharic TTS"""
    
    def __init__(self, model_path: str = None, config_path: str = None):
        """
        Initialize app
        
        Args:
            model_path: Path to trained model
            config_path: Path to configuration file
        """
        print("Initializing Amharic TTS...")
        
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
    
    def import_srt_dataset(self, srt_file, media_file, dataset_name: str, 
                          speaker_name: str, validate: bool) -> str:
        """
        Import SRT dataset from uploaded files
        """
        if srt_file is None or media_file is None:
            return "❌ Please upload both SRT and media files"
        
        if not dataset_name or not dataset_name.strip():
            return "❌ Please enter a dataset name"
        
        try:
            # Import the dataset
            stats = self.srt_builder.import_from_srt(
                srt_path=srt_file.name,
                media_path=media_file.name,
                dataset_name=dataset_name.strip(),
                speaker_name=speaker_name or "speaker_01",
                auto_validate=validate
            )
            
            # Format results
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
3. Copy to `data/processed/` for training
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
            # Parse dataset names
            dataset_names = [name.strip() for name in dataset_selection.split(',')]
            
            if len(dataset_names) < 2:
                return "❌ Please select at least 2 datasets to merge"
            
            # Merge datasets
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
    
    def synthesize(self, text: str, speed: float = 1.0, 
                  pitch: float = 1.0) -> tuple:
        """
        Synthesize speech from Amharic text
        
        Args:
            text: Amharic text
            speed: Speech speed multiplier
            pitch: Pitch multiplier
            
        Returns:
            (audio_file, phonemes_text, info_text)
        """
        if not text or not text.strip():
            return None, "", "⚠ Please enter some text"
        
        try:
            # Convert to phonemes
            phonemes = self.g2p.grapheme_to_phoneme(text)
            
            # Tokenize if tokenizer available
            if self.tokenizer:
                tokens = self.tokenizer.encode(text, use_phonemes=True)
                token_info = f"Tokens ({len(tokens)}): {tokens[:20]}..."
            else:
                token_info = "Tokenizer not available"
            
            # TODO: Generate actual audio when model is loaded
            # For now, return info
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
    
    def create_interface(self) -> gr.Blocks:
        """Create the Gradio interface"""
        
        # Custom CSS for Amharic font support
        css = """
        .gradio-container {
            font-family: 'Noto Sans Ethiopic', 'Abyssinica SIL', sans-serif !important;
            max-width: 1200px;
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
        .amharic-text {
            font-size: 1.2em;
            line-height: 1.8;
        }
        .info-box {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #3498db;
        }
        """
        
        with gr.Blocks(css=css, title="Amharic TTS", theme=gr.themes.Soft()) as app:
            
            # Header
            gr.HTML("""
                <div class="title">🗣️ የአማርኛ ጽሁፍ ወደ ንግግ</div>
                <div class="subtitle">Amharic Text-to-Speech System</div>
            """)
            
            # Create tabs
            with gr.Tabs():
                # Tab 1: TTS Synthesis
                with gr.Tab("🎵 Text-to-Speech"):
                    with gr.Row():
                        # Left Column: Input
                        with gr.Column(scale=1):
                            gr.Markdown("### 📝 Input / ግቤት")
                            
                            text_input = gr.Textbox(
                                label="የአማርኛ ጽሁፍ | Amharic Text",
                                placeholder="አማርኛ ጽሁፍዎን እዚህ ያስገቡ...\\nEnter your Amharic text here...",
                                lines=5,
                                elem_classes=["amharic-text"]
                            )
                            
                            # Example texts
                            gr.Examples(
                                examples=[
                                    ["ሰላም ለዓለም"],
                                    ["አዲስ አበባ የኢትዮጵያ ዋና ከተማ ናት"],
                                    ["እንኳን ደህና መጡ"],
                                    ["አማርኛ በጌዕዝ ፊደል ይጻፋል"],
                                    ["ኢትዮጵያ በምስራቅ አፍሪካ የምትገኝ ሀገር ናት"],
                                ],
                                inputs=text_input,
                                label="📚 Example Texts / ምሳሌዎች"
                            )
                            
                            # Controls
                            with gr.Accordion("⚙️ Settings / ቅንብሮች", open=False):
                                speed_slider = gr.Slider(
                                    minimum=0.5,
                                    maximum=2.0,
                                    value=1.0,
                                    step=0.1,
                                    label="🏃 Speed / ፍጥነት"
                                )
                                
                                pitch_slider = gr.Slider(
                                    minimum=0.5,
                                    maximum=2.0,
                                    value=1.0,
                                    step=0.1,
                                    label="🎵 Pitch / ድምጽ ከፍታ"
                                )
                            
                            # Generate button
                            generate_btn = gr.Button(
                                "🎙️ Generate Speech / ንግግር ፍጠር",
                                variant="primary",
                                size="lg"
                            )
                
                        # Right Column: Output
                        with gr.Column(scale=1):
                            gr.Markdown("### 🔊 Output / ውጭት")
                            
                            audio_output = gr.Audio(
                                label="Generated Audio / የተፍጠረ ድምጽ"
                            )
                            
                            phoneme_output = gr.Textbox(
                                label="📖 Phonemes (IPA) / ፎነሞች",
                                lines=3,
                                interactive=False
                            )
                            
                            info_output = gr.Markdown(
                                label="ℹ️ Information",
                                elem_classes=["info-box"]
                            )
            
                    # Information Section (still in TTS tab)
                    with gr.Accordion("ℹ️ About This System / ስል ሲስተም", open=False):
                        gr.Markdown("""
                        ### የአማርኛ ጽሁፍ ወደ ንግግር ስርዓት | Amharic Text-to-Speech System
                        
                        **Features / ባህሪያት:**
                        - ✅ Native Amharic G2P (Grapheme-to-Phoneme) conversion
                        - ✅ Custom Amharic tokenizer based on Ethiopic script
                        - ✅ Fine-tuned on Amharic speech data
                        - ✅ Preserves multilingual capability (Amharic + English)
                        - ✅ Adjustable speed and pitch controls
                        
                        **How to Use / እንዴት እንደሚጠቀሙ:**
                        1. Enter Amharic text in the input box / ጽሁፍ ያስገቡ
                        2. Adjust settings if desired (optional) / ቅንብሮችን ያስተካክሉ
                        3. Click "Generate Speech" / "ንግግር ፍጠር" ይጫኑ
                        4. Listen to or download the audio / ድምጹን ያዳምጡ ወይም ያውርዱ
                        
                        **Tips / ጠቃሚ ምክሮች:**
                        - Use proper Amharic punctuation for better results
                        - Keep sentences reasonably short (under 100 words)
                        - Experiment with speed and pitch for different effects
                        
                        **System Requirements:**
                        - Python 3.10+
                        - CUDA-capable GPU (for training)
                        - 8GB+ RAM recommended
                        
                        ---
                        
                        **Based on Chatterbox TTS**
                        
                        This system extends Chatterbox with Amharic language support through:
                        - Custom G2P for Ethiopic script
                        - Extended tokenizer vocabulary  
                        - Multilingual fine-tuning with embedding freezing
                        
                        For more information, see the README.md file.
                        """)
                    
                    # Event handler for TTS tab
                    generate_btn.click(
                        fn=self.synthesize,
                        inputs=[text_input, speed_slider, pitch_slider],
                        outputs=[audio_output, phoneme_output, info_output]
                    )
                
                # Tab 2: SRT Import
                with gr.Tab("📺 Import SRT Dataset"):
                    gr.Markdown("""
                    ### 📺 Import Audio/Video with SRT Transcriptions
                    
                    Upload your video/audio files with SRT subtitles to create training datasets.
                    """)
                    
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("#### 📁 Upload Files")
                            
                            srt_upload = gr.File(
                                label="📝 SRT File",
                                file_types=[".srt"]
                            )
                            
                            media_upload = gr.File(
                                label="🎥 Media File (Video/Audio)",
                                file_types=[".mp4", ".mkv", ".avi", ".mov", ".webm", ".mp3", ".wav", ".m4a"]
                            )
                            
                            dataset_name_input = gr.Textbox(
                                label="🏷️ Dataset Name",
                                placeholder="my_amharic_dataset",
                                value=""
                            )
                            
                            speaker_name_input = gr.Textbox(
                                label="🎭 Speaker Name",
                                placeholder="speaker_01",
                                value="speaker_01"
                            )
                            
                            validate_checkbox = gr.Checkbox(
                                label="✅ Validate segments (recommended)",
                                value=True
                            )
                            
                            import_btn = gr.Button(
                                "📥 Import Dataset",
                                variant="primary",
                                size="lg"
                            )
                        
                        with gr.Column():
                            gr.Markdown("#### 📊 Import Results")
                            
                            import_results = gr.Markdown(
                                value="ℹ️ Upload SRT and media files to begin import."
                            )
                    
                    # Import Guide
                    with gr.Accordion("📚 Import Guide", open=False):
                        gr.Markdown("""
                        ### How to Import SRT Datasets
                        
                        **Requirements:**
                        - SRT file with Amharic transcriptions
                        - Matching audio/video file
                        - FFmpeg installed (for video processing)
                        
                        **Steps:**
                        1. Upload your SRT file
                        2. Upload the corresponding media file  
                        3. Enter a unique dataset name
                        4. Click "Import Dataset"
                        
                        **Features:**
                        - Automatic audio extraction from video
                        - Timestamp-based audio segmentation
                        - Quality validation (duration, silence, clipping)
                        - LJSpeech format output
                        
                        **Output:**
                        - Segments saved in `data/srt_datasets/[dataset_name]/wavs/`
                        - Metadata in LJSpeech format
                        - Statistics and validation reports
                        
                        **Tips:**
                        - Ensure SRT timestamps are accurate (±0.5s)
                        - Keep segments between 2-15 seconds
                        - Use clean audio with minimal noise
                        - One sentence per segment is ideal
                        """)
                    
                    # Event handler for import
                    import_btn.click(
                        fn=self.import_srt_dataset,
                        inputs=[
                            srt_upload,
                            media_upload,
                            dataset_name_input,
                            speaker_name_input,
                            validate_checkbox
                        ],
                        outputs=import_results
                    )
                
                # Tab 3: Manage Datasets
                with gr.Tab("📂 Manage Datasets"):
                    gr.Markdown("""
                    ### 📂 Dataset Management
                    
                    View, merge, and manage your imported datasets.
                    """)
                    
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("#### 📊 View Datasets")
                            
                            refresh_btn = gr.Button(
                                "🔄 Refresh Dataset List",
                                variant="secondary"
                            )
                            
                            dataset_list = gr.Markdown(
                                value="🔄 Click 'Refresh' to view datasets"
                            )
                        
                        with gr.Column():
                            gr.Markdown("#### 🔗 Merge Datasets")
                            
                            merge_selection = gr.Textbox(
                                label="Dataset Names (comma-separated)",
                                placeholder="dataset1, dataset2, dataset3",
                                info="Enter names of datasets to merge"
                            )
                            
                            merge_name_input = gr.Textbox(
                                label="Merged Dataset Name",
                                placeholder="merged_dataset",
                                value=""
                            )
                            
                            filter_invalid_checkbox = gr.Checkbox(
                                label="✅ Filter out invalid segments",
                                value=True
                            )
                            
                            merge_btn = gr.Button(
                                "🔗 Merge Datasets",
                                variant="primary"
                            )
                            
                            merge_results = gr.Markdown(
                                value="ℹ️ Enter dataset names to merge"
                            )
                    
                    # Management Guide
                    with gr.Accordion("📚 Management Guide", open=False):
                        gr.Markdown("""
                        ### Dataset Management
                        
                        **View Datasets:**
                        - Click "Refresh" to see all imported datasets
                        - View statistics: segments, duration, validation status
                        - Datasets are stored in `data/srt_datasets/`
                        
                        **Merge Datasets:**
                        - Combine multiple datasets into one
                        - Useful for creating larger training sets
                        - Filter invalid segments during merge
                        - Example: `dataset1, dataset2, dataset3`
                        
                        **Next Steps:**
                        1. Review dataset statistics
                        2. Merge datasets if needed
                        3. Copy final dataset to `data/processed/`
                        4. Train tokenizer on the dataset
                        5. Update `config/training_config.yaml`
                        6. Start training!
                        
                        **Tips:**
                        - Aim for 10+ hours of audio minimum
                        - 20+ hours recommended for better quality
                        - Review validation statistics before training
                        - Keep backups of your original files
                        """)
                    
                    # Event handlers for management tab
                    refresh_btn.click(
                        fn=self.list_datasets,
                        outputs=dataset_list
                    )
                    
                    merge_btn.click(
                        fn=self.merge_datasets_gui,
                        inputs=[
                            merge_selection,
                            merge_name_input,
                            filter_invalid_checkbox
                        ],
                        outputs=merge_results
                    )
        
        return app


def launch_app(model_path: str = None, config_path: str = None,
               share: bool = False, server_port: int = 7860):
    """
    Launch the Gradio app
    
    Args:
        model_path: Path to trained model
        config_path: Path to config file
        share: Whether to create public link
        server_port: Port to run on
    """
    print("\n" + "="*60)
    print("LAUNCHING AMHARIC TTS WEB INTERFACE")
    print("="*60 + "\n")
    
    # Create app
    tts_app = AmharicTTSApp(model_path, config_path)
    app = tts_app.create_interface()
    
    # Launch
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
    
    parser = argparse.ArgumentParser(description="Launch Amharic TTS Web Interface")
    parser.add_argument('--model', type=str, default=None,
                       help='Path to trained model checkpoint')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to configuration file')
    parser.add_argument('--share', action='store_true',
                       help='Create public Gradio link')
    parser.add_argument('--port', type=int, default=7860,
                       help='Server port (default: 7860)')
    
    args = parser.parse_args()
    
    launch_app(
        model_path=args.model,
        config_path=args.config,
        share=args.share,
        server_port=args.port
    )
