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

- 🎹 **Text-to-Beat & Dynamic Multi-Genre Synthesis** — High-energy instrumentals across Lo-Fi, Trap, Synthwave, Drill, Afrobeats, Rock, Piano, Phonk, and EDM.
- 🎤 **Real-Time Voice Recording & Hum-to-Beat** — Record clean 16-bit PCM WAV in browser and convert vocal melodies into genre-matched beats.
- 🎛️ **4-Stem Audio Separation (Demucs HTDemucs)** — Deep learning stem extraction isolating Drums, Bass, Vocals, and Other tracks.
- ⚡ **FCA (Frequency Coded Audio) Engine** — Research-grade audio compression with **FCA-L (Lossless Reversible Transform with SHA-256 verification)** and **FCA-P (Perceptual with Q1–Q10 control)**.
- 🌿 **"Git for Audio" Genealogy Tree** — Full version control for music projects (commits, branches, forks, harmonic diffs, and Stem Pack ZIP exporter).
- 🎚️ **In-Browser Web DAW Mixer** — Per-channel volume faders, stereo panning, mute/solo, and audio-reactive 60fps canvas visualizers.
- 🪄 **AI Mastering & Audio Intelligence** — Automatic EBU R128 loudness normalization (-14 LUFS standard), dynamic EQ, and true-peak limiting.
- 🎼 **Audio-to-MIDI Transcription** — Extract note onsets and melodic pitches directly into downloadable `.mid` format.
- ✍️ **AI Lyrics & Songwriting Generator** — Structure verses, choruses, and bridges by mood and genre.

---

## ⚡ Quickstart Guide

### 1. Clone & Install Dependencies
```bash
git clone https://github.com/Kartikkumar251/melodify_Tune_AI.git
cd melodify_Tune_AI

pip install -r requirements.txt
```

### 2. Launch BeatFlow Platform
```bash
python run.py
```
Open your browser to:
- **Studio DAW**: 👉 **[http://localhost:8000/ui/studio.html](http://localhost:8000/ui/studio.html)**
- **Community Hub**: 👉 **[http://localhost:8000/ui/index.html](http://localhost:8000/ui/index.html)**

### 3. Run Automated Audit & Verification
To test all 11 endpoints and tools:
```bash
python tests/test_audit.py
```

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
| `POST` | `/tools/fca-optimize` | Encode audio to `.fca` container (Lossless / Perceptual) | Public |
| `POST` | `/tools/audio-to-midi` | Melodic Audio to `.mid` transcription | Public |
| `POST` | `/tools/generate-lyrics` | Structured lyrics songwriting engine | Public |
| `GET` | `/tools/prompt-autocomplete` | Real-time search & prompt autocomplete | Public |
| `POST` | `/auth/register` | Create user account | Public |
| `POST` | `/auth/login` | Obtain JWT access token | Public |
| `GET` | `/projects` | List public audio repositories | Public |
| `POST` | `/projects` | Create new audio repository | JWT |
| `POST` | `/projects/{id}/commit` | Commit mix snapshot to project tree | JWT |
| `POST` | `/projects/{id}/fork` | Fork audio tree into new project | JWT |
| `GET` | `/projects/{repo_id}/commits/{commit_id}/export-stems` | Download Stem Pack ZIP archive | Public |

---

## 🛠️ Tech Stack

- **Backend**: FastAPI, Uvicorn, SQLAlchemy, Pydantic, Python-JOSE, Argon2
- **Audio & ML**: Meta MusicGen, HTDemucs, PyTorch, Librosa, PyLoudNorm, SoundFile
- **Async Queue**: Celery, Redis (with automatic in-memory fakeredis fallback)
- **Frontend**: Vanilla HTML5, CSS3 Glassmorphism, Web Audio API, Canvas 2D, Tone.js

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.
