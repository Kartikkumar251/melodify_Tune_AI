# Phase 1: Project Setup & Dependencies

## Overview
Initialize the project structure, set up the Python environment, and install all required dependencies for the BeatFlow AI backend.

## Objectives
- ✅ Create project directory structure
- ✅ Set up Python virtual environment
- ✅ Install system dependencies
- ✅ Install Python packages
- ✅ Configure environment variables

## 1. Project Structure

```
beatflow-ai-backend/
├── .env
├── .gitignore
├── requirements.txt
├── main.py              # FastAPI server
├── worker.py            # Celery worker
├── config.py            # Configuration
├── models/
│   ├── __init__.py
│   ├── database.py
│   ├── schemas.py
│   └── models.py
├── routes/
│   ├── __init__.py
│   ├── generation.py
│   ├── version_control.py
│   └── audio.py
├── services/
│   ├── __init__.py
│   ├── ai_service.py
│   ├── audio_service.py
│   └── stem_service.py
├── static/
│   ├── audio/
│   └── stems/
└── logs/
```

## 2. System Dependencies

Install the following system-level dependencies:

### Windows
```bash
# Install FFmpeg
# Option 1: Using Chocolatey
choco install ffmpeg

# Option 2: Manual download from https://ffmpeg.org/download.html
# Add FFmpeg to PATH environment variable
```

### Linux/WSL
```bash
sudo apt update
sudo apt install ffmpeg
```

### Verify Installation
```bash
ffmpeg -version
```

## 3. Python Environment Setup

```bash
# Create virtual environment (if not using conda)
python -m venv venv

# Activate virtual environment
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

# Or use conda (recommended for ML projects)
conda create -n beatflow python=3.10
conda activate beatflow
```

## 4. Python Dependencies

### requirements.txt
```
fastapi==0.104.1
uvicorn[standard]==0.24.0
celery==5.3.4
redis==5.0.1
sqlalchemy==2.0.23
psycopg2-binary==2.9.9
python-multipart==0.0.6
pydantic==2.5.0
pydantic-settings==2.1.0

# AI Models
torch==2.1.1
torchaudio==2.1.1
audiocraft==1.0.0
demucs==4.0.1
librosa==0.10.0
madmom==0.16.1

# FFmpeg interaction
ffmpeg-python==0.2.1

# Database
alembic==1.13.0
python-dotenv==1.0.0
```

### Installation Command
```bash
pip install -r requirements.txt
```

## 5. Environment Variables (.env)

Create a `.env` file in the root directory:

```
# Database
DATABASE_URL=postgresql://user:password@localhost:5432/beatflow
SQLALCHEMY_ECHO=False

# Redis & Celery
REDIS_URL=redis://localhost:6379/0
CELERY_BROKER_URL=redis://localhost:6379/0
CELERY_RESULT_BACKEND=redis://localhost:6379/1

# API
API_HOST=localhost
API_PORT=8000
API_RELOAD=True

# AI Models
DEVICE=cuda  # or cpu
MUSICGEN_MODEL=facebook/musicgen-small
MAX_GENERATION_DURATION=30

# Storage
STATIC_PATH=./static
AUDIO_SAMPLE_RATE=44100
OUTPUT_FORMAT=wav

# Logging
LOG_LEVEL=INFO
```

## 6. Directory Creation

Create the necessary directories:

```bash
mkdir -p static/audio static/stems logs
```

## 7. Verification

Verify all installations:

```bash
# Check Python
python --version

# Check FFmpeg
ffmpeg -version

# Check PyTorch CUDA
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"

# Check AudioCraft
python -c "from audiocraft.models import MusicGen; print('✅ AudioCraft installed')"

# Check Demucs
python -c "from demucs.pretrained import get_model; print('✅ Demucs installed')"

# Check FastAPI
python -c "import fastapi; print(f'FastAPI v{fastapi.__version__}')"

# Check Celery
python -c "from celery import Celery; print('✅ Celery installed')"

# Check SQLAlchemy
python -c "import sqlalchemy; print(f'SQLAlchemy v{sqlalchemy.__version__}')"
```

## Checklist for Phase 1 Completion
- [ ] Project directory structure created
- [ ] Python environment activated
- [ ] System dependencies installed (FFmpeg)
- [ ] All Python packages installed
- [ ] .env file configured
- [ ] Static directories created
- [ ] All verifications passed

## Next Steps
→ Proceed to **Phase 2: Database Schema & ORM Configuration**

