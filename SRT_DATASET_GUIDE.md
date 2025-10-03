# 📺 SRT Dataset Import System

Complete guide for importing audio/video with SRT transcriptions for Amharic TTS training.

---

## 🎯 What This Does

This system allows you to:
- ✅ Import video/audio files with SRT subtitles
- ✅ Extract audio from video automatically
- ✅ Split audio based on SRT timestamps
- ✅ Validate quality automatically
- ✅ Merge multiple imports into one dataset
- ✅ Prepare datasets for TTS training

---

## 📦 Installation

### 1. Install FFmpeg (Required for video processing)

**Windows:**
```powershell
# Using Chocolatey
choco install ffmpeg

# Or download from: https://ffmpeg.org/download.html
# Add to PATH after installation
```

**Linux:**
```bash
sudo apt install ffmpeg  # Ubuntu/Debian
sudo yum install ffmpeg  # CentOS/RHEL
```

**macOS:**
```bash
brew install ffmpeg
```

### 2. Verify Installation

```powershell
ffmpeg -version
```

---

## 🚀 Quick Start

### Method 1: Interactive Menu (Recommended)

```powershell
cd amharic-tts\src\data_processing
python dataset_manager.py
```

Then follow the on-screen menus!

### Method 2: Command Line

```powershell
# Import single file
python srt_dataset_builder.py import `
    --srt "path\to\video.srt" `
    --media "path\to\video.mp4" `
    --name "my_dataset" `
    --speaker "speaker_01"

# List datasets
python srt_dataset_builder.py list

# Merge datasets
python srt_dataset_builder.py merge `
    --datasets dataset1 dataset2 dataset3 `
    --output merged_final
```

---

## 📁 File Structure

### Your Source Files

```
your_data/
├── video1.srt      # Subtitle file
├── video1.mp4      # Matching video file
├── audio2.srt
├── audio2.wav
├── lecture3.srt
└── lecture3.mp4
```

**Important:** SRT and media files must have **matching names**!

### Output Structure

```
data/srt_datasets/
└── my_dataset/
    ├── wavs/                    # Audio segments
    │   ├── my_dataset_speaker_01_000001.wav
    │   ├── my_dataset_speaker_01_000002.wav
    │   └── ...
    ├── metadata.csv             # LJSpeech format
    ├── dataset_info.json        # Detailed info
    └── statistics.json          # Dataset stats
```

---

## 📝 SRT Format

### Standard Format

```srt
1
00:00:01,000 --> 00:00:05,000
ሰላም ለዓለም

2
00:00:06,500 --> 00:00:10,200
እንኳን ደህና መጡ

3
00:00:11,000 --> 00:00:15,500
አዲስ አበባ የኢትዮጵያ ዋና ከተማ ናት
```

### Format Details

- **Line 1:** Sequence number
- **Line 2:** Timestamps (HH:MM:SS,mmm --> HH:MM:SS,mmm)
- **Line 3+:** Text (can be multiple lines)
- **Blank line:** Separates entries

---

## 🔧 Usage Examples

### Example 1: Import Single Video with SRT

```powershell
python dataset_manager.py
# Select: 1. Import Single SRT Dataset
# Follow prompts:
#   SRT file: C:\data\lecture1.srt
#   Media file: C:\data\lecture1.mp4
#   Dataset name: lecture1_dataset
#   Speaker name: teacher_01
#   Validate: Yes
```

### Example 2: Batch Import Multiple Files

```powershell
# Organize files:
# C:\amharic_videos\
#   ├── ep01.srt + ep01.mp4
#   ├── ep02.srt + ep02.mp4
#   └── ep03.srt + ep03.mp4

python dataset_manager.py
# Select: 2. Batch Import Multiple SRT Files
# Directory: C:\amharic_videos
# Speaker prefix: episode
# Merge after import: Yes
# Merged name: amharic_episodes_full
```

### Example 3: Command-Line Single Import

```powershell
python srt_dataset_builder.py import `
    --srt "C:\data\news_broadcast.srt" `
    --media "C:\data\news_broadcast.mp4" `
    --name "news_dataset" `
    --speaker "newsreader_01"
```

### Example 4: Command-Line Merge

```powershell
python srt_dataset_builder.py merge `
    --datasets lecture1_dataset lecture2_dataset news_dataset `
    --output amharic_training_final
```

---

## ⚙️ Advanced Features

### 1. Custom Validation Thresholds

Edit `srt_dataset_builder.py`:

```python
validator = DatasetValidator(
    min_duration=2.0,    # Minimum segment length (seconds)
    max_duration=12.0,   # Maximum segment length (seconds)
    target_sr=22050      # Target sample rate
)
```

### 2. Skip Validation (Not Recommended)

```powershell
python srt_dataset_builder.py import `
    --srt file.srt `
    --media file.mp4 `
    --name my_dataset `
    --no-validate
```

### 3. Include Invalid Segments in Merge

```powershell
python srt_dataset_builder.py merge `
    --datasets dataset1 dataset2 `
    --output merged `
    --include-invalid
```

### 4. Python API Usage

```python
from srt_dataset_builder import SRTDatasetBuilder

# Initialize
builder = SRTDatasetBuilder(base_output_dir="data/my_datasets")

# Import
stats = builder.import_from_srt(
    srt_path="video.srt",
    media_path="video.mp4",
    dataset_name="my_data",
    speaker_name="speaker_01",
    auto_validate=True
)

print(f"Imported {stats['valid_segments']} segments")
print(f"Total duration: {stats['total_duration_hours']:.2f} hours")

# List datasets
datasets = builder.list_datasets()
for ds in datasets:
    print(f"{ds['name']}: {ds['duration_hours']:.2f}h")

# Merge
merged_stats = builder.merge_datasets(
    dataset_names=["my_data", "other_data"],
    merged_name="combined",
    filter_invalid=True
)
```

---

## 📊 Dataset Statistics

After import, you'll see:

```
IMPORT SUMMARY
====================================================================

Total Segments: 342
Valid Segments: 298
Invalid Segments: 44

Duration:
  Total: 12.45 hours
  Average: 2.85 seconds
  Range: 1.2s - 14.8s

Text:
  Total Characters: 45,682
  Average Length: 153 characters
```

---

## 🔍 Quality Validation

The system automatically checks:

### Audio Quality
- ✅ Duration (1-15 seconds)
- ✅ Silence detection
- ✅ Clipping detection
- ✅ Energy levels (RMS)
- ✅ Sample rate consistency

### Text Quality
- ✅ Text length (minimum 3 characters)
- ✅ Speech rate (characters per second)
- ✅ HTML/formatting removed

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| "Too short" | Segment < 1s | Adjust SRT timing |
| "Too long" | Segment > 15s | Split into smaller segments |
| "Audio clipping" | Volume too high | Re-encode audio with lower volume |
| "Silent" | No audio content | Check SRT timing matches audio |
| "Unusual speech rate" | Text/audio mismatch | Verify SRT synchronization |

---

## 🎓 Complete Workflow

### Step 1: Prepare Your Files

```
1. Collect video/audio files with Amharic speech
2. Get or create SRT subtitle files
3. Ensure matching filenames (video1.srt + video1.mp4)
4. Organize in a directory
```

### Step 2: Import

```powershell
cd C:\Users\Abrsh-1\Downloads\CHATTERBOX_STRUCTURED-AMHARIC\amharic-tts\src\data_processing

# Interactive mode
python dataset_manager.py

# Or command-line
python srt_dataset_builder.py import --srt file.srt --media file.mp4 --name dataset1
```

### Step 3: Review Statistics

```powershell
python dataset_manager.py
# Select: 4. View Dataset Statistics
```

### Step 4: Merge (Optional)

```powershell
python dataset_manager.py
# Select: 3. Merge Datasets
# Choose datasets: 1,2,3
# Name: final_training_set
```

### Step 5: Prepare for Training

```powershell
python dataset_manager.py
# Select: 8. Prepare for Training
# Choose your dataset
# Copy to data/processed/
```

### Step 6: Train Tokenizer

```powershell
cd ../../  # Back to amharic-tts root

python -c "
from src.tokenizer.amharic_tokenizer import train_amharic_tokenizer
train_amharic_tokenizer(
    data_file='data/processed/final_training_set/metadata.csv',
    output_dir='models/tokenizer',
    vocab_size=500
)
"
```

### Step 7: Update Config

Edit `config/training_config.yaml`:

```yaml
data:
  dataset_path: "data/processed/final_training_set"
  metadata_file: "metadata.csv"
```

### Step 8: Start Training

Follow the main README for training instructions.

---

## 💡 Tips & Best Practices

### 1. Start Small
```
✅ Import 1-2 files first
✅ Verify output quality
✅ Then batch import all files
```

### 2. SRT Quality Matters
```
✅ Accurate timestamps (±0.5s)
✅ Clean text (no formatting codes)
✅ Proper Amharic script
✅ One sentence per segment (preferred)
```

### 3. Audio Quality Matters
```
✅ Clear speech
✅ Minimal background noise
✅ Consistent volume
✅ Mono audio (stereo works but mono is better)
```

### 4. Naming Conventions
```
✅ Use descriptive names: lecture_physics_01.srt
✅ Avoid spaces: use underscores or hyphens
✅ Include episode/part numbers: ep01, ep02, etc.
```

### 5. Dataset Size
```
✅ Minimum: 10 hours (basic quality)
✅ Good: 20+ hours (good quality)
✅ Excellent: 50+ hours (excellent quality)
```

---

## 🐛 Troubleshooting

### Problem: "ffmpeg not found"

**Solution:**
```powershell
# Check if installed
ffmpeg -version

# If not installed, install it:
# Windows: https://ffmpeg.org/download.html
# Add to PATH
```

### Problem: "No matching media file"

**Solution:**
```
Ensure filenames match exactly:
  ✅ video.srt + video.mp4
  ❌ video.srt + video_HD.mp4
```

### Problem: "Audio extraction failed"

**Solutions:**
```powershell
# 1. Try extracting audio manually first:
ffmpeg -i video.mp4 -vn -acodec pcm_s16le -ar 22050 -ac 1 audio.wav

# 2. Then import the audio:
python srt_dataset_builder.py import --srt video.srt --media audio.wav --name my_dataset
```

### Problem: "Too many invalid segments"

**Check:**
```
1. SRT timing accuracy (use subtitle editor)
2. Audio quality (re-encode if needed)
3. Text length (very short segments get flagged)
4. Speech rate (too fast/slow indicates timing issues)
```

### Problem: "Import is very slow"

**Solutions:**
```
1. Large video files take time (extract audio first)
2. Use SSD storage (faster I/O)
3. Close other programs (free up resources)
4. Use smaller batch sizes
```

---

## 🔧 Configuration

### Change Output Directory

```python
# In your script
builder = SRTDatasetBuilder(base_output_dir="D:\\my_datasets")
```

Or set environment variable:

```powershell
$env:SRT_DATASET_DIR = "D:\my_datasets"
```

### Change Validation Rules

Edit `srt_dataset_builder.py`:

```python
class DatasetValidator:
    def __init__(self, 
                 min_duration: float = 1.0,   # Change this
                 max_duration: float = 15.0,  # Change this
                 target_sr: int = 22050):     # Change this
        ...
```

---

## 📚 Additional Resources

### Creating SRT Files

**Free Tools:**
- **Subtitle Edit** (Windows): https://www.nikse.dk/subtitleedit
- **Aegisub** (Cross-platform): https://aegisub.org
- **YouTube Auto-Captions**: Download and edit
- **Whisper AI**: Auto-generate from audio

### SRT Editors

- **Subtitle Edit** - Best for Windows
- **Subtitle Workshop** - Classic tool
- **Visual Studio Code** - With SRT extension

### Video Editing

If you need to clean up your source videos:
- **FFmpeg** (command-line)
- **Audacity** (audio editing)
- **Shotcut** (video editing)

---

## 📞 Support

### Common Commands Quick Reference

```powershell
# Interactive menu
python dataset_manager.py

# Import single
python srt_dataset_builder.py import --srt file.srt --media file.mp4 --name dataset1

# List all
python srt_dataset_builder.py list

# Merge
python srt_dataset_builder.py merge --datasets d1 d2 d3 --output merged

# Help
python srt_dataset_builder.py --help
python dataset_manager.py --help
```

### Files Reference

- `srt_dataset_builder.py` - Core import engine
- `dataset_manager.py` - Interactive CLI menu
- `SRT_DATASET_GUIDE.md` - This guide

---

## ✅ Checklist

Before training, ensure:

- [ ] FFmpeg is installed and working
- [ ] SRT files match media files by name
- [ ] Imported at least one dataset successfully
- [ ] Reviewed statistics (min 10+ hours recommended)
- [ ] Validated segments (check invalid count)
- [ ] Merged datasets if needed
- [ ] Copied to `data/processed/`
- [ ] Updated `config/training_config.yaml`
- [ ] Trained tokenizer on the dataset

---

**🎉 You're now ready to import SRT-based datasets for Amharic TTS training!**

**ለስኬት መልካም ምኞት! (Good luck!)**
