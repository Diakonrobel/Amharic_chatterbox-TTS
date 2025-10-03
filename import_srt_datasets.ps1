# SRT Dataset Import - Quick Start Script
# ========================================
# 
# This script helps you quickly import SRT-based datasets for Amharic TTS training.
#
# Usage:
#   .\import_srt_datasets.ps1
#

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  AMHARIC TTS - SRT DATASET IMPORTER" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if Python is available
try {
    $pythonVersion = python --version 2>&1
    Write-Host "✓ Python found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "✗ Python not found! Please install Python first." -ForegroundColor Red
    exit 1
}

# Check if ffmpeg is available
try {
    $ffmpegVersion = ffmpeg -version 2>&1 | Select-Object -First 1
    Write-Host "✓ FFmpeg found: $($ffmpegVersion -replace 'ffmpeg version ', '')" -ForegroundColor Green
} catch {
    Write-Host "⚠ FFmpeg not found! Video processing will be limited." -ForegroundColor Yellow
    Write-Host "  Install from: https://ffmpeg.org/download.html" -ForegroundColor Yellow
}

Write-Host ""

# Navigate to the data processing directory
$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
$dataProcessingPath = Join-Path $scriptPath "src\data_processing"

if (Test-Path $dataProcessingPath) {
    Set-Location $dataProcessingPath
    Write-Host "✓ Changed to: $dataProcessingPath" -ForegroundColor Green
} else {
    Write-Host "✗ Directory not found: $dataProcessingPath" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  CHOOSE IMPORT METHOD" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "1. Interactive Menu (Recommended)" -ForegroundColor White
Write-Host "2. Quick Single Import (Command Line)" -ForegroundColor White
Write-Host "3. Batch Import from Directory" -ForegroundColor White
Write-Host "4. View Existing Datasets" -ForegroundColor White
Write-Host "5. Open Documentation" -ForegroundColor White
Write-Host "0. Exit" -ForegroundColor White
Write-Host ""

$choice = Read-Host "Enter your choice (0-5)"

switch ($choice) {
    "1" {
        # Interactive menu
        Write-Host ""
        Write-Host "Launching interactive menu..." -ForegroundColor Green
        Write-Host ""
        python dataset_manager.py
    }
    
    "2" {
        # Quick single import
        Write-Host ""
        Write-Host "========================================" -ForegroundColor Cyan
        Write-Host "  QUICK SINGLE IMPORT" -ForegroundColor Cyan
        Write-Host "========================================" -ForegroundColor Cyan
        Write-Host ""
        
        $srtFile = Read-Host "SRT file path"
        if (-not (Test-Path $srtFile)) {
            Write-Host "✗ SRT file not found: $srtFile" -ForegroundColor Red
            exit 1
        }
        
        $mediaFile = Read-Host "Audio/Video file path"
        if (-not (Test-Path $mediaFile)) {
            Write-Host "✗ Media file not found: $mediaFile" -ForegroundColor Red
            exit 1
        }
        
        $defaultName = [System.IO.Path]::GetFileNameWithoutExtension($srtFile)
        $datasetName = Read-Host "Dataset name [$defaultName]"
        if ([string]::IsNullOrWhiteSpace($datasetName)) {
            $datasetName = $defaultName
        }
        
        $speakerName = Read-Host "Speaker name [speaker_01]"
        if ([string]::IsNullOrWhiteSpace($speakerName)) {
            $speakerName = "speaker_01"
        }
        
        Write-Host ""
        Write-Host "Importing..." -ForegroundColor Green
        Write-Host ""
        
        python srt_dataset_builder.py import `
            --srt "$srtFile" `
            --media "$mediaFile" `
            --name "$datasetName" `
            --speaker "$speakerName"
    }
    
    "3" {
        # Batch import
        Write-Host ""
        Write-Host "========================================" -ForegroundColor Cyan
        Write-Host "  BATCH IMPORT" -ForegroundColor Cyan
        Write-Host "========================================" -ForegroundColor Cyan
        Write-Host ""
        Write-Host "This will import all SRT+media pairs from a directory." -ForegroundColor Yellow
        Write-Host ""
        
        $directory = Read-Host "Directory containing SRT and media files"
        if (-not (Test-Path $directory)) {
            Write-Host "✗ Directory not found: $directory" -ForegroundColor Red
            exit 1
        }
        
        # Find SRT files
        $srtFiles = Get-ChildItem -Path $directory -Filter "*.srt"
        
        if ($srtFiles.Count -eq 0) {
            Write-Host "✗ No SRT files found in: $directory" -ForegroundColor Red
            exit 1
        }
        
        Write-Host ""
        Write-Host "✓ Found $($srtFiles.Count) SRT file(s)" -ForegroundColor Green
        Write-Host ""
        
        # Launch interactive menu for batch import
        python dataset_manager.py
        # User will select option 2 (Batch Import) from the menu
    }
    
    "4" {
        # View datasets
        Write-Host ""
        Write-Host "Listing existing datasets..." -ForegroundColor Green
        Write-Host ""
        python srt_dataset_builder.py list
        Write-Host ""
        Read-Host "Press Enter to continue"
    }
    
    "5" {
        # Open documentation
        $docPath = Join-Path $scriptPath "SRT_DATASET_GUIDE.md"
        if (Test-Path $docPath) {
            Write-Host ""
            Write-Host "Opening documentation..." -ForegroundColor Green
            Start-Process $docPath
        } else {
            Write-Host "✗ Documentation not found: $docPath" -ForegroundColor Red
        }
    }
    
    "0" {
        Write-Host ""
        Write-Host "Goodbye!" -ForegroundColor Green
        exit 0
    }
    
    default {
        Write-Host ""
        Write-Host "✗ Invalid choice. Please run the script again." -ForegroundColor Red
        exit 1
    }
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  DONE!" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "  1. Review dataset statistics" -ForegroundColor White
Write-Host "  2. Merge datasets if needed" -ForegroundColor White
Write-Host "  3. Copy to data/processed/ for training" -ForegroundColor White
Write-Host "  4. Train tokenizer" -ForegroundColor White
Write-Host "  5. Update training config" -ForegroundColor White
Write-Host ""
Write-Host "For more help, see: SRT_DATASET_GUIDE.md" -ForegroundColor Cyan
Write-Host ""
