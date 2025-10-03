# 🎉 Gradio WebUI - SRT Import Integration Complete!

## What Was Added

The SRT Dataset Import system is now **fully integrated** into the Gradio web interface!

---

## ✨ New Features in WebUI

### 🎵 Tab 1: Text-to-Speech (Existing)
- Amharic text synthesis
- Phoneme preview
- Speed and pitch controls

### 📺 Tab 2: Import SRT Dataset (NEW!)
- **Upload SRT File** - Drag & drop subtitle files
- **Upload Media File** - Support for video (.mp4, .mkv, .avi, .mov, .webm) and audio (.mp3, .wav, .m4a)
- **Dataset Name** - Name your training dataset
- **Speaker Name** - Identify the speaker
- **Validation** - Automatic quality checks
- **Import Results** - Real-time statistics and feedback

### 📂 Tab 3: Manage Datasets (NEW!)
- **View Datasets** - See all imported datasets with statistics
- **Refresh List** - Update dataset information
- **Merge Datasets** - Combine multiple datasets into one
- **Statistics** - Duration, segments, validation status

---

## 🚀 How to Use

### Start the Web Interface

```powershell
cd amharic-tts
python gradio_app/app.py
```

Or with options:
```powershell
python gradio_app/app.py --port 7860 --share
```

Then open: **http://localhost:7860**

---

## 📖 Usage Guide

### Import SRT Dataset

1. **Go to "Import SRT Dataset" tab**
2. **Upload SRT file** - Click or drag your `.srt` file
3. **Upload media file** - Click or drag your video/audio file
4. **Enter dataset name** - e.g., "my_lecture_dataset"
5. **Enter speaker name** - e.g., "professor_01"
6. **Click "Import Dataset"**
7. **View results** - Statistics will appear on the right

### Manage Datasets

1. **Go to "Manage Datasets" tab**
2. **Click "Refresh Dataset List"** - See all your datasets
3. **To merge datasets:**
   - Enter dataset names: `dataset1, dataset2, dataset3`
   - Enter merged name: `final_training_set`
   - Check "Filter invalid segments"
   - Click "Merge Datasets"

---

## 🎯 Workflow

### Complete Dataset Preparation in WebUI

1. **Import** - Upload SRT+media files in "Import SRT Dataset" tab
2. **Review** - Check statistics in "Manage Datasets" tab
3. **Merge** - Combine datasets if you have multiple
4. **Train** - Use the prepared dataset for TTS training

---

## 💡 Features

### File Upload
- ✅ Drag & drop support
- ✅ Multiple file format support
- ✅ Real-time file validation

### Automatic Processing
- ✅ Audio extraction from video
- ✅ Timestamp-based segmentation
- ✅ Quality validation
- ✅ LJSpeech format output

### Quality Checks
- ✅ Duration validation (1-15 seconds)
- ✅ Silence detection
- ✅ Clipping detection
- ✅ Speech rate analysis
- ✅ Text quality checks

### Dataset Management
- ✅ List all datasets
- ✅ View statistics
- ✅ Merge multiple datasets
- ✅ Filter invalid segments

---

## 📊 Output

After import, datasets are saved in:
```
data/srt_datasets/
└── [dataset_name]/
    ├── wavs/              # Audio segments
    ├── metadata.csv       # LJSpeech format
    ├── dataset_info.json  # Detailed info
    └── statistics.json    # Statistics
```

---

## 🔧 Technical Details

### Integration Points

1. **`gradio_app/app.py`** - Modified to include:
   - Import for `SRTDatasetBuilder`
   - Three new methods:
     - `import_srt_dataset()` - Handle file uploads
     - `list_datasets()` - Display dataset info
     - `merge_datasets_gui()` - Merge functionality
   - Two new tabs with full UI

2. **Backend** - Uses existing:
   - `src/data_processing/srt_dataset_builder.py`
   - All validation and processing logic
   - Dataset management functions

---

## ✅ Benefits

### For Users
- **No CLI needed** - Everything in web interface
- **Visual feedback** - See results immediately
- **Easy file upload** - Drag & drop
- **Interactive** - Click buttons, see results

### For Developers
- **Clean integration** - Reuses existing code
- **Maintainable** - Separate tabs for clarity
- **Extensible** - Easy to add more features

---

## 🎓 Examples

### Example 1: Import Single Video

1. Open web interface: `python gradio_app/app.py`
2. Go to "Import SRT Dataset" tab
3. Upload `lecture.srt` and `lecture.mp4`
4. Enter name: `lecture_physics`
5. Click "Import Dataset"
6. See statistics: "298 valid segments, 12.5 hours"

### Example 2: Merge Datasets

1. Go to "Manage Datasets" tab
2. Click "Refresh Dataset List"
3. See: `lecture_physics`, `news_amharic`, `stories`
4. Enter: `lecture_physics, news_amharic, stories`
5. Merged name: `amharic_final_training`
6. Click "Merge Datasets"
7. Result: "Combined 35 hours of audio"

---

## 🆚 CLI vs WebUI

### CLI Advantages
- Automation scripts
- Batch processing
- Server/headless use

### WebUI Advantages
- Visual interface
- Easier for beginners
- Real-time feedback
- No command typing

**Both work! Use what you prefer.**

---

## 📞 Help

### WebUI Not Loading?

```powershell
# Check dependencies
pip install gradio

# Try different port
python gradio_app/app.py --port 8000
```

### Import Not Working?

1. Check FFmpeg is installed: `ffmpeg -version`
2. Check file formats (.srt, .mp4, .mp3, etc.)
3. Check file names match (video.srt + video.mp4)
4. See error messages in import results

### Need CLI Instead?

CLI tools still work:
```powershell
cd src\data_processing
python dataset_manager.py
```

---

## 🎉 Summary

**You now have:**

1. ✅ **Web-based SRT import** - Upload files in browser
2. ✅ **Dataset management GUI** - View and merge datasets
3. ✅ **Real-time feedback** - See results immediately
4. ✅ **No CLI needed** - Everything in one interface
5. ✅ **CLI still available** - Use both!

**Start using it:**
```powershell
cd amharic-tts
python gradio_app/app.py
```

Open: **http://localhost:7860** 🎊

---

**የአማርኛ ድምጽ ማሰልጠኛ ዳታሴት በዌብ በይነገጽ ዝግጁ ነው!**

**Amharic TTS Dataset Import now available in Web Interface!**
