# Cross-Platform Quick Reference

Quick commands for different operating systems.

---

## 🚀 Setup Commands

### Windows (PowerShell)
```powershell
.\setup.ps1
.\venv\Scripts\Activate.ps1
```

### Linux/macOS (Bash)
```bash
chmod +x setup.sh
./setup.sh
source venv/bin/activate
```

### Docker (All Platforms)
```bash
docker-compose up -d
```

---

## 📂 Path Separators

### Windows
- Use backslash: `data\raw\audio`
- In PowerShell: `.\venv\Scripts\Activate.ps1`

### Linux/macOS
- Use forward slash: `data/raw/audio`
- In terminal: `source venv/bin/activate`

### Python (Works on all platforms)
```python
from pathlib import Path
path = Path("data") / "raw" / "audio"  # Works everywhere!
```

---

## 🔧 Common Commands

| Task | Windows (PowerShell) | Linux/macOS (Bash) |
|------|---------------------|-------------------|
| **Activate venv** | `.\venv\Scripts\Activate.ps1` | `source venv/bin/activate` |
| **Deactivate venv** | `deactivate` | `deactivate` |
| **Run script** | `python script.py` | `python3 script.py` or `python script.py` |
| **Make executable** | N/A | `chmod +x script.sh` |
| **List files** | `Get-ChildItem` or `dir` | `ls -la` |
| **Create directory** | `New-Item -ItemType Directory -Path "dir"` | `mkdir -p dir` |
| **Copy files** | `Copy-Item src dst` | `cp src dst` |
| **Remove directory** | `Remove-Item -Recurse dir` | `rm -rf dir` |

---

## 📝 Running Scripts

### Preprocessing Audio

**Windows:**
```powershell
python src/data_processing/preprocess_audio.py `
  --audio-dir "data\raw\audio" `
  --transcript "data\raw\transcripts.txt" `
  --output "data\processed\ljspeech_format"
```

**Linux/macOS:**
```bash
python src/data_processing/preprocess_audio.py \
  --audio-dir "data/raw/audio" \
  --transcript "data/raw/transcripts.txt" \
  --output "data/processed/ljspeech_format"
```

### Training Tokenizer

**Windows:**
```powershell
python -c "from src.tokenizer.amharic_tokenizer import train_amharic_tokenizer; train_amharic_tokenizer('data\processed\ljspeech_format\metadata.csv', 'models\tokenizer', 500)"
```

**Linux/macOS:**
```bash
python -c "from src.tokenizer.amharic_tokenizer import train_amharic_tokenizer; train_amharic_tokenizer('data/processed/ljspeech_format/metadata.csv', 'models/tokenizer', 500)"
```

### Merging Tokenizers

**Windows:**
```powershell
python scripts\merge_tokenizers.py `
  --base "models\pretrained\chatterbox_tokenizer.json" `
  --amharic "models\tokenizer\vocab.json" `
  --output "models\tokenizer\merged_vocab.json" `
  --validate
```

**Linux/macOS:**
```bash
python scripts/merge_tokenizers.py \
  --base "models/pretrained/chatterbox_tokenizer.json" \
  --amharic "models/tokenizer/vocab.json" \
  --output "models/tokenizer/merged_vocab.json" \
  --validate
```

### Extending Model

**Windows:**
```powershell
python scripts\extend_model_embeddings.py `
  --model "models\pretrained\chatterbox_base.pt" `
  --output "models\pretrained\chatterbox_extended.pt" `
  --original-size 704 `
  --new-size 2000
```

**Linux/macOS:**
```bash
python scripts/extend_model_embeddings.py \
  --model "models/pretrained/chatterbox_base.pt" \
  --output "models/pretrained/chatterbox_extended.pt" \
  --original-size 704 \
  --new-size 2000
```

### Launch Gradio UI

**All platforms:**
```bash
python gradio_app/app.py
python gradio_app/app.py --port 7860
python gradio_app/app.py --share  # Public link
```

---

## 🐳 Docker Commands

### Build and Run

**All platforms:**
```bash
# Build image
docker build -t amharic-tts .

# Run container
docker run -p 7860:7860 amharic-tts

# Run with volume mounts
docker run -p 7860:7860 -v $(pwd)/data:/app/data amharic-tts

# Using docker-compose
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

**Windows PowerShell:**
```powershell
# Run with volume mounts (Windows style)
docker run -p 7860:7860 -v ${PWD}/data:/app/data amharic-tts
```

---

## 🔐 Permissions

### Making Scripts Executable

**Linux/macOS:**
```bash
chmod +x setup.sh
chmod +x scripts/*.py
```

**Windows:**
No need for chmod, but may need to adjust execution policy:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

---

## 🌐 Environment Variables

### Setting Environment Variables

**Windows (PowerShell):**
```powershell
$env:CUDA_VISIBLE_DEVICES = "0"
$env:GRADIO_SERVER_PORT = "7860"
```

**Linux/macOS (Bash):**
```bash
export CUDA_VISIBLE_DEVICES=0
export GRADIO_SERVER_PORT=7860
```

**Permanent (add to profile):**
- Windows: Add to `$PROFILE`
- Linux: Add to `~/.bashrc`
- macOS: Add to `~/.zshrc`

---

## 📊 GPU/CUDA Detection

### Check CUDA Availability

**All platforms:**
```bash
# Check NVIDIA GPU
nvidia-smi

# Check PyTorch CUDA
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA device: {torch.cuda.get_device_name(0)}')"
```

---

## 🔍 Testing Installation

### Quick Tests

**All platforms:**
```bash
# Test Python
python --version

# Test imports
python -c "import torch; print(torch.__version__)"
python -c "import gradio; print(gradio.__version__)"

# Test G2P
python -c "from src.g2p.amharic_g2p import AmharicG2P; print('G2P OK')"

# Test tokenizer
python -c "from src.tokenizer.amharic_tokenizer import AmharicTokenizer; print('Tokenizer OK')"
```

---

## 🐛 Platform-Specific Issues

### Windows

**Issue: Script won't run**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**Issue: Path too long**
```powershell
# Run as Administrator
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
```

### Linux

**Issue: Permission denied**
```bash
chmod +x script.sh
```

**Issue: libsndfile not found**
```bash
sudo apt install libsndfile1
```

### macOS

**Issue: Command not found**
```bash
# Add Python to PATH
echo 'export PATH="/usr/local/opt/python@3.10/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
```

**Issue: M1/M2 compatibility**
```bash
# Use Rosetta if needed
arch -x86_64 /bin/bash setup.sh
```

---

## 📚 Platform Documentation

- **Windows:** See [INSTALL.md](INSTALL.md#windows-installation)
- **Linux:** See [INSTALL.md](INSTALL.md#linux-installation-ubuntudebian)
- **macOS:** See [INSTALL.md](INSTALL.md#macos-installation)
- **Docker:** See [INSTALL.md](INSTALL.md#docker-installation-all-platforms)

---

## 💡 Tips

### Use Platform-Independent Python Code

```python
from pathlib import Path

# ✅ Good - works on all platforms
data_path = Path("data") / "raw" / "audio"

# ❌ Bad - Windows only
data_path = "data\\raw\\audio"

# ❌ Bad - Linux/Mac only
data_path = "data/raw/audio"
```

### Use os.path or pathlib

```python
import os
from pathlib import Path

# Both work cross-platform
path1 = os.path.join("data", "raw", "audio")
path2 = Path("data") / "raw" / "audio"
```

---

**Platform Support:**
- ✅ Windows 10/11
- ✅ Ubuntu 20.04/22.04
- ✅ Debian 11/12
- ✅ macOS 12+ (Intel & Apple Silicon)
- ✅ Docker (all platforms)
- ✅ WSL2 (Windows Subsystem for Linux)
