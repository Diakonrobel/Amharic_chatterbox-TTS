# 📺 SRT Dataset Import System - Complete Guide

## Overview

A comprehensive, cross-platform system for importing audio/video files with SRT transcriptions to create training datasets for Amharic TTS.

---

## 🎯 What Was Added

### Core Files

1. **`src/data_processing/srt_dataset_builder.py`** (709 lines)
   - Complete SRT parsing and import engine
   - Audio extraction from video files
   - Segment extraction based on timestamps
   - Quality validation system
   - Dataset merging capabilities
   - LJSpeech format output
   - Command-line interface

2. **`src/data_processing/dataset_manager.py`** (763 lines)
   - Interactive cross-platform CLI menu
   - User-friendly dataset management
   - Batch import functionality
   - Dataset inspection and statistics
   - Training preparation helpers

3. **`import_srt_datasets.ps1`** (195 lines)
   - Windows PowerShell quick-start script
   - Check dependencies (Python, FFmpeg)
   - Multiple import methods
   - Easy access to all features

4. **`examples/example_srt_import.py`** (272 lines)
   - 6 comprehensive usage examples
   - Python API demonstrations
   - Workflow tutorials

5. **`SRT_IMPORT_README.md`** (This file)
   - Complete documentation
   - Usage guide
   - Troubleshooting

---

## ✨ Features

### 📥 Import Capabilities
- ✅ Single SRT+media file import
- ✅ Batch import multiple files
- ✅ Support for videos (.mp4, .mkv, .avi, .mov, .webm)
- ✅ Support for audio (.mp3, .wav, .m4a)
- ✅ Automatic audio extraction from video
- ✅ Segment splitting based on SRT timestamps

### 🔍 Quality Validation
- ✅ Audio duration checks (1-15 seconds)
- ✅ Silence detection
- ✅ Clipping detection
- ✅ Energy level validation
- ✅ Text length validation
- ✅ Speech rate analysis
- ✅ Automatic filtering of invalid segments

### 🔗 Dataset Management
- ✅ Merge multiple datasets
- ✅ Filter invalid segments
- ✅ Calculate statistics
- ✅ Inspect dataset details
- ✅ LJSpeech format output
- ✅ JSON metadata export

### 🖥️ User Interface
- ✅ Interactive CLI menu
- ✅ Command-line interface
- ✅ PowerShell quick-start
- ✅ Python API
- ✅ Cross-platform (Windows, Linux, macOS)

---

## 🚀 Quick Start

### Option 1: PowerShell Script (Easiest)

```powershell
cd amharic-tts
.\import_srt_datasets.ps1
```

### Option 2: Interactive Menu

```powershell
cd amharic-tts\src\data_processing
python dataset_manager.py
```

### Option 3: Command Line

```powershell
python srt_dataset_builder.py import `
    --srt "video.srt" `
    --media "video.mp4" `
    --name "my_dataset"
```

### Option 4: Python API

```python
from srt_dataset_builder import SRTDatasetBuilder

builder = SRTDatasetBuilder()
stats = builder.import_from_srt(
    srt_path="video.srt",
    media_path="video.mp4",
    dataset_name="my_dataset"
)
```

---

## 📋 Requirements

### System Requirements
- Python 3.10+
- FFmpeg (for video processing)
- 8GB+ RAM
- Storage for datasets

### Python Dependencies (already in requirements.txt)
```
numpy
pandas
librosa
soundfile
tqdm
```

---

## 📁 File Structure

### Input Files
```
your_videos/
├── lecture1.srt
├── lecture1.mp4
├── news.srt
└── news.wav
```

### Output Structure
```
data/srt_datasets/
├── lecture1/
│   ├── wavs/                    # Audio segments
│   │   ├── lecture1_speaker_01_000001.wav
│   │   ├── lecture1_speaker_01_000002.wav
│   │   └── ...
│   ├── metadata.csv             # LJSpeech format
│   ├── dataset_info.json        # Detailed metadata
│   └── statistics.json          # Statistics
└── news/
    └── ...
```

---

## 🎓 Usage Examples

### Example 1: Import Single Video

```powershell
cd amharic-tts\src\data_processing

python srt_dataset_builder.py import `
    --srt "C:\Videos\lecture.srt" `
    --media "C:\Videos\lecture.mp4" `
    --name "lecture_dataset" `
    --speaker "professor_01"
```

### Example 2: Batch Import

```powershell
# Place all SRT+media pairs in one folder
# Run interactive menu:
python dataset_manager.py
# Select: 2. Batch Import Multiple SRT Files
# Enter folder path
# Follow prompts
```

### Example 3: Merge Datasets

```powershell
python srt_dataset_builder.py merge `
    --datasets dataset1 dataset2 dataset3 `
    --output final_training_set
```

### Example 4: List All Datasets

```powershell
python srt_dataset_builder.py list
```

### Example 5: Python Script

```python
from srt_dataset_builder import SRTDatasetBuilder

builder = SRTDatasetBuilder()

# Import multiple files
files = [
    ("ep1.srt", "ep1.mp4", "episode1"),
    ("ep2.srt", "ep2.mp4", "episode2"),
    ("ep3.srt", "ep3.mp4", "episode3"),
]

for srt, media, name in files:
    stats = builder.import_from_srt(
        srt_path=srt,
        media_path=media,
        dataset_name=name,
        speaker_name="narrator_01"
    )
    print(f"{name}: {stats['valid_segments']} segments")

# Merge all
merged_stats = builder.merge_datasets(
    dataset_names=["episode1", "episode2", "episode3"],
    merged_name="full_series",
    filter_invalid=True
)

print(f"Total: {merged_stats['total_duration_hours']:.2f} hours")
```

---

## 🔧 Advanced Features

### Custom Validation Thresholds

Edit `srt_dataset_builder.py`:

```python
validator = DatasetValidator(
    min_duration=2.0,    # Adjust minimum
    max_duration=12.0,   # Adjust maximum
    target_sr=22050      # Target sample rate
)
```

### Custom Output Directory

```python
builder = SRTDatasetBuilder(
    base_output_dir="D:\\MyDatasets"
)
```

### Skip Validation

```powershell
python srt_dataset_builder.py import `
    --srt file.srt `
    --media file.mp4 `
    --name dataset `
    --no-validate
```

### Include Invalid Segments

```powershell
python srt_dataset_builder.py merge `
    --datasets d1 d2 `
    --output merged `
    --include-invalid
```

---

## 📊 Output Format

### metadata.csv (LJSpeech Format)
```
filename|text|normalized_text
dataset_speaker_01_000001|ሰላም ለዓለም|ሰላም ለዓለም
dataset_speaker_01_000002|አዲስ አበባ|አዲስ አበባ
```

### dataset_info.json
```json
{
  "dataset_name": "my_dataset",
  "source_srt": "video.srt",
  "source_media": "video.mp4",
  "speaker": "speaker_01",
  "total_segments": 342,
  "valid_segments": 298,
  "segments": [...]
}
```

### statistics.json
```json
{
  "total_segments": 342,
  "valid_segments": 298,
  "total_duration_hours": 12.45,
  "average_duration": 2.85,
  "average_text_length": 153
}
```

---

## 🔍 Validation Checks

The system automatically validates:

| Check | Criteria | Action |
|-------|----------|--------|
| Duration | 1-15 seconds | Flag if outside range |
| Silence | Detects silence | Flag if silent |
| Clipping | Audio > 0.99 | Flag if clipping |
| Energy | RMS level | Flag if too low |
| Text | Min 3 characters | Flag if too short |
| Speech rate | 3-30 chars/sec | Flag if unusual |

---

## 🎯 Complete Workflow

### Step 1: Prepare Files
```
✓ Collect videos/audio with Amharic speech
✓ Create or download SRT subtitle files
✓ Ensure matching filenames
✓ Organize in a folder
```

### Step 2: Import
```powershell
cd amharic-tts
.\import_srt_datasets.ps1
# Select option 1 or 2
# Follow prompts
```

### Step 3: Review
```powershell
cd src\data_processing
python dataset_manager.py
# Select: 4. View Dataset Statistics
```

### Step 4: Merge (if needed)
```powershell
python dataset_manager.py
# Select: 3. Merge Datasets
```

### Step 5: Prepare for Training
```powershell
python dataset_manager.py
# Select: 8. Prepare for Training
# Copy to data/processed/
```

### Step 6: Train Tokenizer
```powershell
cd ..\..  # Back to amharic-tts root

python -c "from src.tokenizer.amharic_tokenizer import train_amharic_tokenizer; train_amharic_tokenizer('data/processed/my_dataset/metadata.csv', 'models/tokenizer', 500)"
```

### Step 7: Update Config
Edit `config/training_config.yaml`:
```yaml
data:
  dataset_path: "data/processed/my_dataset"
  metadata_file: "metadata.csv"
```

### Step 8: Start Training
Follow main README for training instructions.

---

## 🐛 Troubleshooting

### FFmpeg not found

**Windows:**
```powershell
# Install with Chocolatey
choco install ffmpeg

# Or download from https://ffmpeg.org/download.html
# Add to PATH
```

**Linux:**
```bash
sudo apt install ffmpeg
```

### No matching media file

Ensure filenames match:
```
✅ video.srt + video.mp4
❌ video.srt + video_HD.mp4
```

### Import is slow

- Large videos take time
- Extract audio first manually:
```powershell
ffmpeg -i video.mp4 -vn -ar 22050 -ac 1 audio.wav
```

### Too many invalid segments

Check:
- SRT timing accuracy
- Audio quality
- Text transcription quality

---

## 💡 Tips & Best Practices

### SRT Quality
```
✅ Accurate timestamps (±0.5s)
✅ Clean text (no HTML/formatting)
✅ One sentence per segment
✅ Proper Amharic script
```

### Dataset Size
```
Minimum: 10 hours (basic)
Good: 20+ hours (good quality)
Excellent: 50+ hours (excellent)
```

### File Organization
```
✅ Descriptive names: lecture_physics_01.srt
✅ No spaces: use underscores
✅ Include numbers: ep01, ep02, etc.
```

---

## 📚 File Reference

| File | Purpose | Location |
|------|---------|----------|
| `srt_dataset_builder.py` | Core import engine | `src/data_processing/` |
| `dataset_manager.py` | Interactive CLI | `src/data_processing/` |
| `import_srt_datasets.ps1` | Quick start script | `amharic-tts/` |
| `example_srt_import.py` | Usage examples | `examples/` |
| `SRT_IMPORT_README.md` | This guide | `amharic-tts/` |

---

## 🎯 Command Reference

### Import Commands
```powershell
# Single import
python srt_dataset_builder.py import --srt FILE.srt --media FILE.mp4 --name NAME

# With options
python srt_dataset_builder.py import --srt FILE.srt --media FILE.mp4 --name NAME --speaker SPEAKER --no-validate

# Batch (use interactive menu)
python dataset_manager.py
```

### Management Commands
```powershell
# List datasets
python srt_dataset_builder.py list

# Merge datasets
python srt_dataset_builder.py merge --datasets D1 D2 D3 --output MERGED

# Merge with invalid segments
python srt_dataset_builder.py merge --datasets D1 D2 --output MERGED --include-invalid

# Interactive menu
python dataset_manager.py
```

### Quick Start
```powershell
# PowerShell launcher
.\import_srt_datasets.ps1

# Direct interactive
python dataset_manager.py
```

---

## ✅ Integration Checklist

Before training, ensure:

- [ ] FFmpeg is installed
- [ ] SRT files match media files
- [ ] Imported at least one dataset
- [ ] Reviewed statistics (10+ hours recommended)
- [ ] Merged datasets if needed
- [ ] Copied to `data/processed/`
- [ ] Trained tokenizer
- [ ] Updated `config/training_config.yaml`

---

## 🎉 Summary

You now have a complete system for:

1. ✅ Importing audio/video with SRT transcriptions
2. ✅ Automatic quality validation
3. ✅ Batch processing multiple files
4. ✅ Merging datasets for training
5. ✅ LJSpeech format output
6. ✅ Cross-platform CLI interface
7. ✅ Python API for automation
8. ✅ Comprehensive documentation

**Total lines of code added: 2,000+**

---

## 📞 Need Help?

1. Check this README
2. Run interactive menu: `python dataset_manager.py`
3. View examples: `python examples/example_srt_import.py`
4. Check main project README

---

**🎊 You're now ready to import SRT-based datasets for Amharic TTS training!**

**የአማርኛ ድምጽ ማሰልጠኛ ዳታሴት ዝግጁ ነው!**
