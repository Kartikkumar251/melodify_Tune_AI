@echo off
REM BeatFlow AI (Melodyfy) - Setup Script for Windows
REM Installs and verifies core runtime dependencies

echo ==========================================
echo   BeatFlow AI - Environment Setup
echo ==========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found on PATH. Please install Python 3.10+ first.
    pause
    exit /b 1
)

echo [1/4] Checking Python version...
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo Python version: %PYTHON_VERSION%
echo.

echo [2/4] Upgrading pip...
python -m pip install --upgrade pip
if errorlevel 1 (
    echo ERROR: Failed to upgrade pip
    pause
    exit /b 1
)
echo.

echo [3/4] Installing BeatFlow AI dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install dependencies from requirements.txt
    echo Please verify network connectivity and try again.
    pause
    exit /b 1
)
echo.

echo [4/4] Verifying installations...
python -c "import torch; print(f'PyTorch: {torch.__version__} (CUDA: {torch.cuda.is_available()})')" || goto :error
python -c "import transformers; print(f'Transformers: {transformers.__version__}')" || goto :error
python -c "import librosa; print(f'Librosa: {librosa.__version__}')" || goto :error
python -c "import fastapi; print(f'FastAPI: {fastapi.__version__}')" || goto :error
echo.

echo ==========================================
echo   BeatFlow AI Setup Completed Successfully!
echo ==========================================
echo.
echo Quickstart:
echo 1. Start API Server: python api_server.py
echo 2. Open in Browser:  http://localhost:8000
echo.
pause
exit /b 0

:error
echo.
echo ERROR: Dependency verification check failed.
echo Run: pip install -r requirements.txt
pause
exit /b 1
