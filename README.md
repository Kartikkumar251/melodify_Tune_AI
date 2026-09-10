# BeatFlow AI (Melodyfy) 🎵⚡

> **AI-Powered Generative Music Production Suite with "Git for Audio" Version Control & Web DAW**

[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat&logo=FastAPI&logoColor=white)](https://fastapi.tiangolo.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org)
[![MusicGen](https://img.shields.io/badge/MusicGen-Meta-blue)](https://github.com/facebookresearch/audiocraft)
[![Demucs](https://img.shields.io/badge/Demucs-Stem%20Separation-purple)](https://github.com/facebookresearch/demucs)
[![WebAudio](https://img.shields.io/badge/Web%20Audio%20API-Tone.js-orange)](https://tonejs.github.io/)

---

## 🌟 Executive Summary

**BeatFlow AI** bridges generative AI and music engineering. It combines state-of-the-art text/melody conditioning models (Meta MusicGen) with a full multi-track in-browser DAW and a novel **"Git for Audio" version-control engine**. Creators can synthesize tracks, extract stems (drums, bass, vocals, melody), compare harmonic diffs, branch, and collaborate seamlessly.

---

## 🚀 Key Features

- 🎹 **Text-to-Beat & Vibe Conditioning** — Generate production-ready instrumentals across 25+ curated genres with customized BPM and mood prompts.
- 🎤 **Hum-to-Beat (Melody Conditioning)** — Record or upload a vocal melody/humming to condition MusicGen-Melody for structured song synthesis.
- 🎛️ **4-Stem Audio Separation (Demucs HTDemucs)** — Deep learning stem extraction isolating Drums, Bass, Vocals, and Synths into individual WAV tracks.
- 🌿 **"Git for Audio" Genealogy Tree** — Full version control for music projects:
  - Commit audio snapshots with metadata (BPM, key signature, LUFS loudness).
  - Branch, fork, and merge community remix trees.
  - Acoustic and harmonic diff comparison between revisions.
- 🎚️ **In-Browser Web DAW Mixer** — Per-channel volume faders, stereo panning, reverb impulse response, 8D spatial audio, mute/solo, and audio-reactive 60fps canvas visualizers.
- 🪄 **AI Mastering & Audio Intelligence** — Automatic EBU R128 loudness normalization (-14 LUFS standard) and true-peak brickwall limiting.
- 🎼 **Audio-to-MIDI Transcription** — Extract note onsets and melodic pitches directly into downloadable `.mid` format.

---

## 🏗️ Architecture & Data Flow

```mermaid
graph TD
    Client["Frontend Client (Web DAW / Tone.js)"] -->|REST / Multipart| API["FastAPI Application Server"]
    API -->|Async Tasks| Celery["Celery Task Queue (Redis / Memory)"]
    API -->|Session / ORM| DB[("SQL Database (SQLite / Postgres)")]
    
    Celery -->|Inference| MusicGen["MusicGen (Small / Melody)"]
    Celery -->|Stem Extraction| Demucs["Demucs (HTDemucs)"]
    Celery -->|Acoustic Analysis| Librosa["Librosa Audio Engine"]
    
    MusicGen -->|Render WAV| Static["Static Storage (/beat_outputs)"]
    Demucs -->|Stems| StaticStems["Stem Storage (/stems_outputs)"]
    Static --> Client
    StaticStems --> Client
```

---

## 📂 Repository Layout

```
├── api_server.py           # Core FastAPI REST & WebSocket Server
├── audio_processing.py     # Demucs stem splitting, Librosa analysis, AI mastering
├── beat_generator.py       # Standalone MusicGen synthesis engine
├── celery_worker.py        # Asynchronous job queue for compute-heavy ML tasks
├── run_demucs.py           # Demucs runtime wrapper & format patcher
├── models.py               # SQLAlchemy ORM schemas (Users, Repos, Commits, Stems)
├── database.py             # Database engine & session dependency
├── auth.py                 # Argon2 password security & JWT auth pipeline
├── nav.js                  # Shared modern navigation component
├── visualizer.js           # Real-time audio-reactive 60fps Canvas 2D engine
├── index.html              # Landing page & interactive hero
├── studio.html             # Multi-track Web DAW & Stem mixer
├── dashboard.html          # Creator dashboard & quick generator
├── explore.html            # Community discovery & top remix feeds
├── repo.html               # Repository view & commit timeline
├── project_tree.html       # Visual Git audio tree
├── library.html            # Audio stem asset library
├── settings.html           # Audio configuration & API preferences
├── requirements.txt        # Full Python runtime dependencies
├── docs/                   # Engineering blueprints & technical deep dives
└── tests/                  # Model verification & benchmark test suite
```

---

## ⚡ Quickstart Guide

### 1. Clone & Setup Environment

```bash
# Clone the repository
git clone <repository_url>
cd <repository_name>

# Create and activate virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Launch BeatFlow API & Web DAW

```bash
python api_server.py
```

Open your browser and navigate to:
👉 **`http://localhost:8000`**

---

## 📡 REST API Reference

| Method | Endpoint | Description | Auth |
|---|---|---|---|
| `GET` | `/health` | Service health & GPU status | Public |
| `POST` | `/generate` | Generate beat from prompt (synchronous) | Optional |
| `POST` | `/generate/async` | Dispatch Celery generation job | Optional |
| `POST` | `/hum` | Hum-to-beat audio conditioning | Optional |
| `POST` | `/separate` | 4-stem Demucs separation | Optional |
| `POST` | `/analyze` | Extract BPM, key, energy, loudness | Public |
| `POST` | `/master` | -14 LUFS AI Mastering | Public |
| `POST` | `/audio-to-midi` | Melodic Audio to `.mid` transcription | Public |
| `POST` | `/auth/register` | Create user account | Public |
| `POST` | `/auth/login` | Obtain JWT access token | Public |
| `GET` | `/projects` | List public audio repositories | Public |
| `POST` | `/projects` | Create new audio repository | JWT |
| `POST` | `/projects/{id}/commit` | Commit mix snapshot to project tree | JWT |
| `POST` | `/projects/{id}/fork` | Fork audio tree into new project | JWT |

---

## 🛠️ Tech Stack

- **Backend**: FastAPI, Uvicorn, SQLAlchemy, Pydantic, Python-JOSE, Argon2
- **Audio & ML**: Meta MusicGen, HTDemucs, PyTorch, Librosa, PyLoudNorm, SoundFile
- **Async Queue**: Celery, Redis (with automatic in-memory fakeredis fallback)
- **Frontend**: Vanilla HTML5, CSS3 Glassmorphism, Web Audio API, Canvas 2D, Tone.js

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.
