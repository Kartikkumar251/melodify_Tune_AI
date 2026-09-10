# Redis + Celery Setup Guide

## Overview

BeatFlow AI uses Celery for async AI task processing (beat generation, stem separation, audio analysis).
Celery requires a Redis message broker. Three options for Windows:

---

## Option A: Docker (Recommended)

```powershell
# Start Redis container
docker run --name beatflow-redis -p 6379:6379 -d redis:7-alpine

# Verify
docker exec beatflow-redis redis-cli ping
# → PONG
```

---

## Option B: Memurai (Windows Redis Port)

1. Download from https://www.memurai.com/
2. Install → runs as Windows service on port 6379
3. Free for dev use

---

## Option C: WSL2 Redis

```bash
# In WSL2 terminal
sudo apt install redis-server
sudo service redis-server start
redis-cli ping  # → PONG
```

---

## Start Celery Worker

```powershell
cd "D:\Ronit Sharma\vs code\ML Models\hack"
& "D:\Ronit Sharma\vs code\ML Models\.conda\python.exe" -m celery -A celery_worker worker --loglevel=info --pool=solo
```

> `--pool=solo` is **required** on Windows (avoids multiprocessing fork issues)

---

## Celery Tasks Defined

| Task | Description |
|------|-------------|
| `generate_beat_task(prompt, label, commit_id)` | Run MusicGen, update DB commit with audio_url |
| `separate_stems_task(audio_path, commit_id)` | Run Demucs, save 4 stems to DB |
| `analyze_audio_task(audio_path, commit_id)` | Run Librosa, update commit BPM/key |

---

## Without Redis (Current State)

All endpoints work **synchronously** without Redis/Celery running.
- `/generate` — runs MusicGen inline (blocks ~13s on GPU)
- `/projects/{id}/commit` — runs Librosa analysis inline
- `/separate` — runs Demucs inline (may be slow)

Celery tasks are defined and ready; they just won't be dispatched until Redis is running.

---

## Flower (Celery Monitoring Dashboard)

```powershell
pip install flower
python -m celery -A celery_worker flower --port=5555
# → Open http://localhost:5555
```
