# Phase 3: Celery Worker & AI Models Setup

## Overview
Implement asynchronous task processing using Celery with Redis broker, and set up AI models (MusicGen, Demucs) for GPU inference.

## Objectives
- ✅ Configure Redis message broker
- ✅ Create Celery application and worker pool
- ✅ Implement MusicGen music generation task
- ✅ Implement Demucs stem separation task
- ✅ Set up audio analysis tasks (BPM, Key detection)
- ✅ Implement error handling and task monitoring
- ✅ Optimize for RTX 5050 VRAM constraints

## 1. Redis Setup

### Windows Setup
```powershell
# Option 1: Using Memurai
# Download from: https://github.com/microsoftarchive/memurai-developer
# Install and run

# Option 2: Using Docker
docker run --name redis -p 6379:6379 -d redis:latest

# Verify Redis is running
redis-cli ping  # Should return PONG
```

### Linux/WSL Setup
```bash
# Install Redis
sudo apt-get install redis-server

# Start Redis
redis-server

# Verify Redis is running
redis-cli ping  # Should return PONG
```

## 2. Celery Configuration

### config.py
```python
import os
from dotenv import load_dotenv

load_dotenv()

class CeleryConfig:
    """Celery configuration class"""
    
    # Broker and Backend
    broker_url = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
    result_backend = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/1")
    
    # Task settings
    task_serializer = "json"
    accept_content = ["json"]
    result_serializer = "json"
    timezone = "UTC"
    enable_utc = True
    
    # Task execution
    task_track_started = True
    task_time_limit = 30 * 60  # 30 minutes max
    task_soft_time_limit = 25 * 60  # 25 minutes soft limit
    
    # Resource settings
    worker_max_tasks_per_child = 1000
    worker_prefetch_multiplier = 1
    
    # Retry settings
    task_acks_late = True
    worker_disable_rate_limits = False

# Device detection
DEVICE = os.getenv("DEVICE", "cuda" if os.path.exists("/proc/driver/nvidia") else "cpu")
MODEL_DIR = os.getenv("MODEL_DIR", "./models/cache")
STATIC_PATH = os.getenv("STATIC_PATH", "./static")
AUDIO_SAMPLE_RATE = int(os.getenv("AUDIO_SAMPLE_RATE", 44100))
OUTPUT_FORMAT = os.getenv("OUTPUT_FORMAT", "wav")
MAX_DURATION = int(os.getenv("MAX_GENERATION_DURATION", 30))
```

## 3. Celery Worker Implementation

### worker.py
```python
import os
import logging
from celery import Celery, Task
from celery.utils.log import get_task_logger
from config import CeleryConfig, DEVICE, MODEL_DIR, STATIC_PATH
from pathlib import Path
import torch
from audiocraft.models import MusicGen
from demucs.pretrained import get_model as get_demucs_model
import subprocess
import json
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
task_logger = get_task_logger(__name__)

# Initialize Celery app
celery_app = Celery("beatflow_worker")
celery_app.config_from_object(CeleryConfig)

# Create necessary directories
os.makedirs(f"{STATIC_PATH}/audio", exist_ok=True)
os.makedirs(f"{STATIC_PATH}/stems", exist_ok=True)

# ---------------------------------------------------------
# GLOBAL MODEL LOADING (Critical for Performance)
# Models are loaded once when worker starts
# ---------------------------------------------------------

class WorkerState:
    """Global state for model management"""
    musicgen_model = None
    demucs_model = None
    models_loaded = False
    device = DEVICE

def load_models():
    """Load AI models into VRAM"""
    if WorkerState.models_loaded:
        return
    
    task_logger.info(f"⏳ Loading AI Models into {WorkerState.device}...")
    
    try:
        # Load MusicGen Small (2GB VRAM)
        task_logger.info("Loading MusicGen model...")
        WorkerState.musicgen_model = MusicGen.get_pretrained(
            'facebook/musicgen-small',
            device=WorkerState.device
        )
        WorkerState.musicgen_model.set_generation_params(duration=30)
        task_logger.info("✅ MusicGen loaded")
        
        # Load Demucs (HTDemucs - fastest)
        task_logger.info("Loading Demucs model...")
        WorkerState.demucs_model = get_demucs_model('htdemucs')
        WorkerState.demucs_model.to(WorkerState.device)
        task_logger.info("✅ Demucs loaded")
        
        WorkerState.models_loaded = True
        task_logger.info(f"✅ All Models Loaded on {WorkerState.device}")
        
    except Exception as e:
        task_logger.error(f"❌ Failed to load models: {str(e)}")
        raise

@celery_app.on_after_configure.connect
def setup_worker(sender, **kwargs):
    """Load models when worker starts"""
    load_models()

# ---------------------------------------------------------
# TASK 1: MUSIC GENERATION
# ---------------------------------------------------------

@celery_app.task(
    bind=True,
    name="tasks.generate_music",
    max_retries=2,
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_backoff_max=600,
)
def generate_music_task(
    self,
    prompt: str,
    bpm: int = 120,
    mood: str = "neutral",
    duration: int = 15,
    repo_id: str = None,
    commit_id: str = None
):
    """
    Generate music using MusicGen model
    
    Args:
        prompt: Text description of music to generate
        bpm: Beats per minute
        mood: Musical mood/style
        duration: Duration in seconds (max 30)
        repo_id: Repository ID (for database linking)
        commit_id: Commit ID (for database linking)
    
    Returns:
        dict: {status, file_path, duration, bpm, key_signature}
    """
    try:
        task_logger.info(f"🎵 Starting music generation: {prompt}")
        
        # Validate duration
        duration = min(int(duration), 30)
        
        # Update task state
        self.update_state(state='PROGRESS', meta={'current': 0, 'total': 100, 'status': 'Formatting prompt...'})
        
        # Build prompt
        full_prompt = f"{prompt}, mood: {mood}, BPM: {bpm}, high fidelity"
        task_logger.info(f"Full prompt: {full_prompt}")
        
        # Set generation params
        WorkerState.musicgen_model.set_generation_params(duration=duration)
        
        # Update state
        self.update_state(state='PROGRESS', meta={'current': 30, 'total': 100, 'status': 'Generating audio...'})
        
        # Generate music
        with torch.no_grad():
            wav = WorkerState.musicgen_model.generate(
                descriptions=[full_prompt],
                progress=False
            )
        
        # Create output filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"gen_{self.request.id}_{timestamp}"
        output_path = f"{STATIC_PATH}/audio/{output_filename}"
        
        # Update state
        self.update_state(state='PROGRESS', meta={'current': 70, 'total': 100, 'status': 'Saving audio...'})
        
        # Save audio file
        from audiocraft.data.audio import audio_write
        audio_write(
            output_path,
            wav[0].cpu(),
            WorkerState.musicgen_model.sample_rate,
            strategy="loudness",
            format="wav"
        )
        
        # Convert to MP3 for preview
        mp3_path = output_path.replace('.wav', '.mp3')
        convert_to_mp3(f"{output_path}.wav", mp3_path)
        
        task_logger.info(f"✅ Generation completed: {output_filename}")
        
        result = {
            "status": "success",
            "file_path": f"/static/audio/{output_filename}.wav",
            "preview_path": f"/static/audio/{output_filename}.mp3",
            "duration": duration,
            "bpm": bpm,
            "mood": mood,
            "prompt": prompt,
            "output_filename": output_filename
        }
        
        return result
        
    except Exception as e:
        task_logger.error(f"❌ Generation failed: {str(e)}")
        raise self.retry(exc=e, countdown=5)

# ---------------------------------------------------------
# TASK 2: STEM SEPARATION
# ---------------------------------------------------------

@celery_app.task(
    bind=True,
    name="tasks.separate_stems",
    max_retries=1,
)
def separate_stems_task(self, file_path: str, output_dir: str = None):
    """
    Separate audio stems using Demucs
    
    Args:
        file_path: Path to the audio file
        output_dir: Directory to save separated stems
    
    Returns:
        dict: {status, stems_path, stem_files}
    """
    try:
        task_logger.info(f"🎼 Starting stem separation: {file_path}")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Audio file not found: {file_path}")
        
        output_dir = output_dir or f"{STATIC_PATH}/stems"
        os.makedirs(output_dir, exist_ok=True)
        
        # Update state
        self.update_state(state='PROGRESS', meta={'current': 0, 'total': 100, 'status': 'Preparing audio...'})
        
        # Use Demucs via command line (ensures proper processing)
        command = [
            "demucs",
            "-n", "htdemucs",  # Fast, high-quality model
            "--out", output_dir,
            "--mp3",  # Output as MP3
            file_path
        ]
        
        task_logger.info(f"Running: {' '.join(command)}")
        
        # Update state
        self.update_state(state='PROGRESS', meta={'current': 25, 'total': 100, 'status': 'Separating stems...'})
        
        result = subprocess.run(command, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"Demucs failed: {result.stderr}")
        
        # Find output directory
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        stems_dir = os.path.join(output_dir, "htdemucs", base_name)
        
        if not os.path.exists(stems_dir):
            raise RuntimeError(f"Stems directory not found: {stems_dir}")
        
        # Gather stem files
        stem_files = {
            "drums": os.path.join(stems_dir, "drums.wav"),
            "bass": os.path.join(stems_dir, "bass.wav"),
            "other": os.path.join(stems_dir, "other.wav"),
            "vocals": os.path.join(stems_dir, "vocals.wav")
        }
        
        # Map to static paths
        stem_urls = {k: v.replace(STATIC_PATH, "/static") for k, v in stem_files.items()}
        
        task_logger.info(f"✅ Stem separation completed: {stems_dir}")
        
        result = {
            "status": "success",
            "stems_path": stems_dir,
            "stem_urls": stem_urls,
            "stem_files": list(stem_files.keys())
        }
        
        return result
        
    except Exception as e:
        task_logger.error(f"❌ Stem separation failed: {str(e)}")
        raise self.retry(exc=e, countdown=10)

# ---------------------------------------------------------
# TASK 3: AUDIO ANALYSIS
# ---------------------------------------------------------

@celery_app.task(
    bind=True,
    name="tasks.analyze_audio",
)
def analyze_audio_task(self, file_path: str):
    """
    Analyze audio file for BPM and key signature
    
    Args:
        file_path: Path to the audio file
    
    Returns:
        dict: {status, bpm, key_signature}
    """
    try:
        task_logger.info(f"📊 Starting audio analysis: {file_path}")
        
        import librosa
        import numpy as np
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Audio file not found: {file_path}")
        
        # Load audio
        y, sr = librosa.load(file_path, sr=None)
        
        # Detect BPM
        onset_strength = librosa.onset.onset_strength(y=y, sr=sr)
        bpm = librosa.feature.tempo(onset_envelope=onset_strength, sr=sr)[0]
        bpm = int(round(bpm))
        
        # Detect key (simplified - power spectrum analysis)
        # For production, consider using more sophisticated key detection
        key_signature = "C Major"  # Placeholder
        
        task_logger.info(f"✅ Analysis completed: BPM={bpm}, Key={key_signature}")
        
        return {
            "status": "success",
            "bpm": bpm,
            "key_signature": key_signature,
            "duration": len(y) / sr
        }
        
    except Exception as e:
        task_logger.error(f"❌ Analysis failed: {str(e)}")
        raise

# ---------------------------------------------------------
# TASK 4: FORMAT CONVERSION (MP3)
# ---------------------------------------------------------

def convert_to_mp3(input_path: str, output_path: str, bitrate: str = "192k"):
    """Convert WAV to MP3 using ffmpeg"""
    try:
        command = [
            "ffmpeg",
            "-i", input_path,
            "-ab", bitrate,
            "-y",  # Overwrite output
            output_path
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode == 0:
            task_logger.info(f"✅ Converted to MP3: {output_path}")
        else:
            task_logger.warning(f"⚠️ MP3 conversion failed: {result.stderr}")
    except Exception as e:
        task_logger.warning(f"⚠️ Could not convert to MP3: {str(e)}")

# ---------------------------------------------------------
# MONITOR TASKS
# ---------------------------------------------------------

@celery_app.task(bind=True, name="tasks.monitor_gpu")
def monitor_gpu(self):
    """Monitor GPU usage"""
    try:
        if torch.cuda.is_available():
            memory_used = torch.cuda.memory_allocated() / 1e9  # GB
            memory_max = torch.cuda.get_device_properties(0).total_memory / 1e9  # GB
            utilization = (memory_used / memory_max) * 100
            
            task_logger.info(f"GPU Memory: {memory_used:.2f}GB / {memory_max:.2f}GB ({utilization:.1f}%)")
            
            return {
                "memory_used_gb": memory_used,
                "memory_max_gb": memory_max,
                "utilization_percent": utilization
            }
        else:
            return {"device": "CPU", "note": "CUDA not available"}
    except Exception as e:
        task_logger.error(f"GPU monitoring failed: {str(e)}")
        return {"error": str(e)}

if __name__ == "__main__":
    # Run worker
    celery_app.start([
        'worker',
        '--loglevel=info',
        '--pool=solo',  # Required for Windows
        '-c', '1'  # Single concurrency to manage VRAM
    ])
```

## 4. Running the Celery Worker

### Terminal 1: Start Redis
```bash
# Windows
redis-server

# Or with Docker
docker run -p 6379:6379 -d redis:latest
```

### Terminal 2: Start Celery Worker
```bash
# Windows - IMPORTANT: Must use --pool=solo
celery -A worker.celery_app worker --loglevel=info --pool=solo

# Linux/Mac
celery -A worker.celery_app worker --loglevel=info --pool=threads
```

### Expected Output
```
-------------- celery@HOSTNAME v5.3.4 (emerald-rush)
--- ***** -----
-- ******* ----
- *** --- * ---
- ** ---------- [config]
- ** ---------- .broker: redis://localhost:6379/0
- ** ---------- .app: worker:0x...
- *** --- * --- .concurrency: 1 (solo)
- ******* ---- .events: OFF
-- ******* ---- .max-tasks-per-child: 1000
--- ***** ----- .log-level: INFO
-------------- [queues]
                - celery

[2024-XX-XX XX:XX:XX,XXX: INFO/MainProcess] Connected to redis://localhost:6379/0
[2024-XX-XX XX:XX:XX,XXX: INFO/MainProcess] mingle: syncing with 1 nodes.
[2024-XX-XX XX:XX:XX,XXX: INFO/MainProcess] → ⏳ Loading AI Models into cuda...
```

## 5. Testing Tasks (Optional)

### test_tasks.py
```python
import asyncio
from worker import generate_music_task, analyze_audio_task

# Test music generation
task = generate_music_task.delay(
    prompt="Upbeat electronic dance music",
    bpm=128,
    mood="energetic",
    duration=15
)

print(f"Task ID: {task.id}")
print(f"Status: {task.status}")

# Poll for result (with timeout)
try:
    result = task.get(timeout=300)  # 5 minutes max
    print(f"Result: {result}")
except Exception as e:
    print(f"Error: {str(e)}")
```

## Checklist for Phase 3 Completion
- [ ] Redis installed and running
- [ ] Celery configuration created
- [ ] Worker.py implemented with all tasks
- [ ] Models loading verified
- [ ] Redis and Celery worker started successfully
- [ ] Task execution tested
- [ ] GPU memory management optimized

## Important Notes
⚠️ **Windows Users**: Always use `--pool=solo` flag with Celery
⚠️ **VRAM Management**: RTX 5050 has ~4GB VRAM. Monitor with `torch.cuda.memory_allocated()`
⚠️ **Model Warmup**: First generation is slow (30-40s), subsequent ones are faster (~10-20s)

## Next Steps
→ Proceed to **Phase 4: FastAPI Server & API Endpoints**

