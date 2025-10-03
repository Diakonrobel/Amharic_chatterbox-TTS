# Installation Guide - Cross-Platform

Complete installation instructions for Windows, Linux, and macOS.

---

## 📦 System Requirements

### All Platforms
- Python 3.10 or later
- 8GB+ RAM
- 10GB+ disk space
- (Optional) CUDA-capable GPU for training

---

## 🪟 Windows Installation

### Prerequisites

1. **Install Python 3.10+**
   - Download from: https://www.python.org/downloads/
   - ✅ Check "Add Python to PATH" during installation

2. **Install Git** (optional)
   - Download from: https://git-scm.com/download/win

3. **Install CUDA Toolkit** (for GPU support)
   - Download CUDA 11.8: https://developer.nvidia.com/cuda-11-8-0-download-archive
   - Verify: `nvidia-smi` in PowerShell

### Installation

```powershell
# Navigate to project
cd amharic-tts

# Run setup script
.\setup.ps1

# If execution policy error:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\setup.ps1

# Activate environment
.\venv\Scripts\Activate.ps1
```

### Troubleshooting Windows

**PowerShell Execution Policy:**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**Long Path Issues:**
```powershell
# Run as Administrator
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
```

---

## 🐧 Linux Installation (Ubuntu/Debian)

### Prerequisites

```bash
# Update package list
sudo apt update

# Install Python 3.10+
sudo apt install python3.10 python3.10-venv python3-pip

# Install system dependencies
sudo apt install -y \
    build-essential \
    git \
    ffmpeg \
    libsndfile1 \
    portaudio19-dev \
    python3-dev

# For CUDA support (optional)
# Follow NVIDIA's instructions for your Ubuntu version
# https://developer.nvidia.com/cuda-downloads
```

### Installation

```bash
# Navigate to project
cd amharic-tts

# Make setup script executable
chmod +x setup.sh

# Run setup script
./setup.sh

# Activate environment
source venv/bin/activate
```

### Troubleshooting Linux

**Permission Issues:**
```bash
chmod +x setup.sh
```

**Python Version:**
```bash
# If python3.10 not available
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.10 python3.10-venv
```

**Audio Libraries:**
```bash
sudo apt install -y libsndfile1 portaudio19-dev python3-pyaudio
```

---

## 🍎 macOS Installation

### Prerequisites

1. **Install Homebrew** (if not installed)
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

2. **Install Python and dependencies**
```bash
# Install Python 3.10+
brew install python@3.10

# Install system dependencies
brew install ffmpeg portaudio
```

### Installation

```bash
# Navigate to project
cd amharic-tts

# Make setup script executable
chmod +x setup.sh

# Run setup script
./setup.sh

# Activate environment
source venv/bin/activate
```

### Troubleshooting macOS

**M1/M2 Mac (Apple Silicon):**
```bash
# Use Rosetta for compatibility if needed
arch -x86_64 /bin/bash setup.sh
```

**Command not found:**
```bash
# Add to PATH
echo 'export PATH="/usr/local/opt/python@3.10/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
```

---

## 🐳 Docker Installation (All Platforms)

### Using Docker

```bash
# Build Docker image
docker build -t amharic-tts .

# Run container
docker run -p 7860:7860 -v $(pwd)/data:/app/data amharic-tts

# With GPU support (Linux)
docker run --gpus all -p 7860:7860 -v $(pwd)/data:/app/data amharic-tts
```

### Dockerfile

Save as `Dockerfile` in project root:

```dockerfile
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy project files
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose Gradio port
EXPOSE 7860

# Run application
CMD ["python", "gradio_app/app.py"]
```

---

## ☁️ Cloud Platform Installation

### Google Colab

```python
# Install in Colab notebook
!git clone <your-repo-url> amharic-tts
%cd amharic-tts
!pip install -r requirements.txt

# Run Gradio with public link
!python gradio_app/app.py --share
```

### AWS EC2 / Azure / GCP

```bash
# SSH into instance
ssh user@your-instance

# Install dependencies (Ubuntu)
sudo apt update
sudo apt install -y python3.10 python3.10-venv git

# Clone and setup
git clone <your-repo-url> amharic-tts
cd amharic-tts
chmod +x setup.sh
./setup.sh

# Run with nohup
nohup python gradio_app/app.py > logs/app.log 2>&1 &
```

---

## ✅ Verification

After installation, verify everything works:

### 1. Test Python Environment

```bash
# Activate environment (if not already)
# Windows: .\venv\Scripts\Activate.ps1
# Linux/Mac: source venv/bin/activate

# Check Python version
python --version  # Should be 3.10+

# Check PyTorch
python -c "import torch; print(f'PyTorch: {torch.__version__}')"

# Check CUDA (if GPU)
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

### 2. Test G2P Module

```bash
python -c "from src.g2p.amharic_g2p import AmharicG2P; g=AmharicG2P(); print(g.grapheme_to_phoneme('ሰላም'))"
```

Expected output: IPA phonemes (e.g., `səlam`)

### 3. Launch UI

```bash
python gradio_app/app.py
```

Open browser to: `http://localhost:7860`

---

## 🔧 Platform-Specific Notes

### Windows

- Use **PowerShell** (not CMD)
- Paths use backslashes: `.\venv\Scripts\Activate.ps1`
- May need to run as Administrator for some operations

### Linux

- Use **Bash**
- Paths use forward slashes: `source venv/bin/activate`
- May need `sudo` for system packages

### macOS

- Similar to Linux
- Use **zsh** (default on modern macOS) or **bash**
- M1/M2 Macs may need Rosetta for some packages

---

## 📚 Additional Resources

### System-Specific Documentation

- **Windows:** [Python on Windows](https://docs.python.org/3/using/windows.html)
- **Linux:** [Python on Linux](https://docs.python.org/3/using/unix.html)
- **macOS:** [Python on macOS](https://docs.python.org/3/using/mac.html)

### CUDA Installation

- [CUDA Toolkit Download](https://developer.nvidia.com/cuda-downloads)
- [PyTorch CUDA Compatibility](https://pytorch.org/get-started/locally/)

---

## 🆘 Common Issues

### All Platforms

**Import Errors:**
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

**Virtual Environment Issues:**
```bash
# Delete and recreate
rm -rf venv  # Linux/Mac
Remove-Item -Recurse -Force venv  # Windows

# Run setup again
```

### Platform-Specific

**Windows: DLL Load Failed**
- Install Visual C++ Redistributable
- Download: https://aka.ms/vs/17/release/vc_redist.x64.exe

**Linux: libsndfile.so not found**
```bash
sudo apt install libsndfile1
```

**macOS: Architecture Issues**
```bash
# For M1/M2 Macs
arch -arm64 pip install <package>
```

---

## 🎯 Next Steps

After successful installation:

1. **Read the README:** Complete documentation
2. **Try Quick Start:** 5-minute setup guide
3. **Prepare Data:** Follow data preparation guide
4. **Train Model:** Follow training pipeline

---

**Platform Tested:**
- ✅ Windows 10/11
- ✅ Ubuntu 20.04/22.04
- ✅ macOS 12+ (Intel & Apple Silicon)
- ✅ Docker
- ✅ Cloud platforms (AWS, GCP, Azure)
