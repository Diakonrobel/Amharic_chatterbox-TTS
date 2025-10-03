# Amharic TTS Setup Script for Windows
# Run this script to set up the development environment

Write-Host "===============================================" -ForegroundColor Cyan
Write-Host "Amharic TTS Development Environment Setup" -ForegroundColor Cyan
Write-Host "===============================================" -ForegroundColor Cyan
Write-Host ""

# Check Python version
Write-Host "[1/6] Checking Python installation..." -ForegroundColor Yellow
$pythonVersion = python --version 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ Python found: $pythonVersion" -ForegroundColor Green
} else {
    Write-Host "✗ Python not found. Please install Python 3.10 or later." -ForegroundColor Red
    exit 1
}

# Create virtual environment
Write-Host ""
Write-Host "[2/6] Creating virtual environment..." -ForegroundColor Yellow
python -m venv venv
if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ Virtual environment created" -ForegroundColor Green
} else {
    Write-Host "✗ Failed to create virtual environment" -ForegroundColor Red
    exit 1
}

# Activate virtual environment
Write-Host ""
Write-Host "[3/6] Activating virtual environment..." -ForegroundColor Yellow
& ".\venv\Scripts\Activate.ps1"
Write-Host "✓ Virtual environment activated" -ForegroundColor Green

# Upgrade pip
Write-Host ""
Write-Host "[4/6] Upgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip setuptools wheel
Write-Host "✓ Pip upgraded" -ForegroundColor Green

# Install PyTorch (CUDA 11.8)
Write-Host ""
Write-Host "[5/6] Installing PyTorch..." -ForegroundColor Yellow
Write-Host "   This may take a while..." -ForegroundColor Gray
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ PyTorch installed" -ForegroundColor Green
} else {
    Write-Host "⚠ PyTorch installation had issues, continuing..." -ForegroundColor Yellow
}

# Install requirements
Write-Host ""
Write-Host "[6/6] Installing dependencies..." -ForegroundColor Yellow
Write-Host "   This may take several minutes..." -ForegroundColor Gray
pip install -r requirements.txt
if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ Dependencies installed" -ForegroundColor Green
} else {
    Write-Host "⚠ Some dependencies may have failed" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "===============================================" -ForegroundColor Cyan
Write-Host "Setup Complete!" -ForegroundColor Green
Write-Host "===============================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "1. Activate virtual environment: .\venv\Scripts\Activate.ps1"
Write-Host "2. Prepare your Amharic dataset"
Write-Host "3. Follow the README for training instructions"
Write-Host ""
