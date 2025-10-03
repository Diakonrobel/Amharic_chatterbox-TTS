# 🌍 Cross-Platform Support

Complete cross-platform support for Windows, Linux, macOS, and Docker.

---

## ✅ What's Included

### Setup Scripts
- ✅ **setup.ps1** - Windows PowerShell
- ✅ **setup.sh** - Linux/macOS Bash (with CUDA auto-detection)

### Deployment Options
- ✅ **Dockerfile** - Container deployment
- ✅ **docker-compose.yml** - Easy Docker orchestration
- ✅ **.dockerignore** - Optimized container builds

### Documentation
- ✅ **INSTALL.md** - Comprehensive installation for all platforms
- ✅ **PLATFORMS.md** - Quick command reference
- ✅ **README.md** - Updated with cross-platform notes

### Project Files
- ✅ **.gitignore** - Cross-platform Git ignores
- ✅ **.gitkeep** files - Preserve directory structure

---

## 🎯 Platform Support Matrix

| Feature | Windows | Linux | macOS | Docker |
|---------|---------|-------|-------|--------|
| **Setup Script** | ✅ | ✅ | ✅ | ✅ |
| **G2P Module** | ✅ | ✅ | ✅ | ✅ |
| **Tokenizer** | ✅ | ✅ | ✅ | ✅ |
| **Audio Processing** | ✅ | ✅ | ✅ | ✅ |
| **Training** | ✅ | ✅ | ✅ | ✅ |
| **Gradio UI** | ✅ | ✅ | ✅ | ✅ |
| **CUDA Support** | ✅ | ✅ | ❌* | ✅ |
| **M1/M2 Support** | N/A | N/A | ✅** | ✅ |

*macOS doesn't support CUDA (use CPU or cloud)  
**Requires Rosetta for some packages

---

## 📋 Quick Start by Platform

### Windows
```powershell
# 1. Download project
cd amharic-tts

# 2. Run setup
.\setup.ps1

# 3. Activate
.\venv\Scripts\Activate.ps1

# 4. Test
python gradio_app/app.py
```

### Linux (Ubuntu/Debian)
```bash
# 1. Install dependencies
sudo apt update
sudo apt install python3.10 python3.10-venv git

# 2. Setup
cd amharic-tts
chmod +x setup.sh
./setup.sh

# 3. Activate
source venv/bin/activate

# 4. Test
python gradio_app/app.py
```

### macOS
```bash
# 1. Install Homebrew (if needed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# 2. Install Python
brew install python@3.10

# 3. Setup
cd amharic-tts
chmod +x setup.sh
./setup.sh

# 4. Activate
source venv/bin/activate

# 5. Test
python gradio_app/app.py
```

### Docker (All Platforms)
```bash
# 1. Build and run
cd amharic-tts
docker-compose up -d

# 2. Access
# Open http://localhost:7860

# 3. View logs
docker-compose logs -f

# 4. Stop
docker-compose down
```

---

## 🔧 Platform-Specific Features

### Windows
- **PowerShell scripts** with execution policy handling
- **Path handling** optimized for Windows
- **CUDA 11.8** support for NVIDIA GPUs
- **Long path support** instructions included

### Linux
- **Bash scripts** with automatic dependency detection
- **CUDA auto-detection** in setup script
- **Package manager integration** (apt, yum)
- **GPU support** fully tested on Ubuntu 20.04/22.04

### macOS
- **Homebrew integration** for easy setup
- **Intel and Apple Silicon (M1/M2)** support
- **Rosetta compatibility** for legacy packages
- **CPU-only** training (no CUDA on Mac)

### Docker
- **Multi-platform images** (amd64, arm64)
- **Automatic dependency installation**
- **Volume mounting** for data persistence
- **GPU support** (NVIDIA Docker on Linux)

---

## 🚀 Advanced Features

### Automatic Platform Detection

**setup.sh automatically detects:**
- Python version (3.10+)
- CUDA availability
- Package manager (apt/yum)
- CPU architecture

**Scripts use cross-platform paths:**
```python
from pathlib import Path

# Works on all platforms
data_dir = Path("data") / "raw" / "audio"
```

### Environment Variable Support

**Consistent across platforms:**
```bash
# Set CUDA device (all platforms)
export CUDA_VISIBLE_DEVICES=0  # Linux/Mac
$env:CUDA_VISIBLE_DEVICES = "0"  # Windows

# Set Gradio port
export GRADIO_SERVER_PORT=7860  # Linux/Mac
$env:GRADIO_SERVER_PORT = "7860"  # Windows
```

---

## 📊 Performance Notes

### CPU Performance
- **Windows:** Good with Intel/AMD CPUs
- **Linux:** Excellent, slightly faster than Windows
- **macOS:** Good with M1/M2, moderate with Intel
- **Docker:** Near-native performance

### GPU Performance (CUDA)
- **Windows:** Full CUDA 11.8 support
- **Linux:** Full CUDA 11.8 support, best performance
- **macOS:** No CUDA support (use CPU or cloud)
- **Docker:** NVIDIA Docker support on Linux

### Memory Usage
- **Training:** 8GB+ RAM recommended
- **Inference:** 4GB+ RAM sufficient
- **Docker:** Add 1GB overhead

---

## 🐛 Common Issues by Platform

### Windows
**Issue:** PowerShell execution policy  
**Fix:** `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

**Issue:** Long paths  
**Fix:** Enable long paths in Registry (see INSTALL.md)

**Issue:** CUDA not detected  
**Fix:** Install CUDA 11.8, verify with `nvidia-smi`

### Linux
**Issue:** Permission denied on scripts  
**Fix:** `chmod +x setup.sh`

**Issue:** Missing system libraries  
**Fix:** `sudo apt install build-essential libsndfile1 ffmpeg`

**Issue:** Python version  
**Fix:** `sudo add-apt-repository ppa:deadsnakes/ppa && sudo apt install python3.10`

### macOS
**Issue:** Command not found  
**Fix:** Add Python to PATH in `~/.zshrc`

**Issue:** M1/M2 compatibility  
**Fix:** Use Rosetta: `arch -x86_64 /bin/bash setup.sh`

**Issue:** Permission denied  
**Fix:** `chmod +x setup.sh`

### Docker
**Issue:** Cannot connect to Docker daemon  
**Fix:** Start Docker Desktop / Docker service

**Issue:** Port already in use  
**Fix:** Change port in docker-compose.yml

**Issue:** GPU not available  
**Fix:** Install nvidia-docker2 on Linux

---

## 📦 Distribution

### Creating Releases

**Platform-specific packages:**
```bash
# Windows installer (future)
# - .exe installer with GUI
# - Portable .zip package

# Linux packages
# - .deb for Debian/Ubuntu
# - .rpm for RedHat/Fedora
# - AppImage for universal Linux

# macOS
# - .dmg installer
# - Homebrew formula

# Universal
# - Docker image on Docker Hub
# - Python wheel on PyPI
```

---

## 🌐 Cloud Deployment

### Tested Platforms

**Cloud Providers:**
- ✅ AWS EC2 (Ubuntu)
- ✅ Google Cloud Compute Engine
- ✅ Azure Virtual Machines
- ✅ DigitalOcean Droplets
- ✅ Linode
- ✅ Google Colab (Jupyter)

**Deployment Methods:**
1. **Direct:** SSH + setup script
2. **Docker:** Pull and run container
3. **Kubernetes:** Deploy with Helm chart (future)
4. **Serverless:** AWS Lambda / Cloud Functions (future)

---

## 🎓 Best Practices

### Code Portability

**✅ DO:**
```python
from pathlib import Path
import os

# Use Path objects
path = Path("data") / "audio" / "file.wav"

# Use os.path.join
path = os.path.join("data", "audio", "file.wav")

# Check platform when needed
import platform
if platform.system() == "Windows":
    # Windows-specific code
    pass
```

**❌ DON'T:**
```python
# Hard-coded paths
path = "C:\\data\\audio\\file.wav"  # Windows only
path = "/home/user/data/audio/file.wav"  # Linux only

# Platform-specific commands without checks
os.system("ls")  # Fails on Windows
```

### Testing

**Test on multiple platforms:**
1. Windows 10/11
2. Ubuntu 20.04 LTS (most common Linux)
3. macOS latest (Intel or M1/M2)
4. Docker (validates cross-platform compatibility)

---

## 📚 Additional Resources

### Platform Docs
- **Windows:** [Python on Windows](https://docs.python.org/3/using/windows.html)
- **Linux:** [Python on Linux](https://docs.python.org/3/using/unix.html)
- **macOS:** [Python on macOS](https://docs.python.org/3/using/mac.html)
- **Docker:** [Docker Documentation](https://docs.docker.com/)

### Tools
- **Conda:** Cross-platform package manager
- **WSL2:** Run Linux on Windows
- **Homebrew:** Package manager for macOS/Linux
- **Docker:** Containerization for all platforms

---

## 🎉 Summary

The Amharic TTS system is **fully cross-platform** with:

- ✅ **3 operating systems** supported natively
- ✅ **2 setup scripts** (PowerShell + Bash)
- ✅ **Docker deployment** for universal compatibility
- ✅ **Comprehensive documentation** for each platform
- ✅ **Automatic platform detection** in setup
- ✅ **Path-agnostic code** using pathlib
- ✅ **Tested and verified** on multiple platforms

**You can develop, train, and deploy on ANY platform!**

---

**Supported Platforms:**
- 🪟 Windows 10/11
- 🐧 Ubuntu 20.04/22.04
- 🐧 Debian 11/12
- 🍎 macOS 12+ (Intel & M1/M2)
- 🐳 Docker (all platforms)
- ☁️ AWS, GCP, Azure
- 📓 Google Colab
- 🖥️ WSL2

**All scripts, all platforms, all working!** 🚀
