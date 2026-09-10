Project Overview: BeatFlow AI
1. Core Concept
BeatFlow AI is an AI-powered music production and collaboration platform. It combines a generative AI engine (MusicGen) with a "GitHub-style" version control system for audio. Users can generate beats via text or humming, split them into stems, edit them in a web-based DAW, and "Fork" or "Commit" changes to a visual audio-genealogy tree.
2. Tech Stack
Frontend: Next.js (React), Tone.js (Audio Engine), React Flow (Git Tree Visualization).
Backend: FastAPI (Python), PostgreSQL (Database), SQLAlchemy (ORM).
Asynchronous Processing: Celery + Redis (Task queue for AI generation).
AI Models (Local Inference on RTX 5050):
MusicGen (Small/Melody): Text-to-Audio and Hum-to-Beat generation.
Demucs (HTDemucs): Stem separation (Drums, Bass, Melody, Other).
Librosa: BPM and Key detection.
Storage: Local static storage (development) / S3-compatible storage (production).
3. Key Features & Backend Logic
A. The "Git for Audio" System
Repositories: Every project is a repository.
Commits: Every "Save" creates a commit. A commit stores the full mix URL and links to specific stems.
Parent-Child Logic: Each commit (except the first) has a parent_commit_id. This allows for branching and history tracking.
Forking: Users can fork another user's project, creating a new repository that starts from a specific commit hash.
B. AI Generation Pipeline
Async Workflow: API receives a prompt -> Pushes task to Redis -> Celery worker runs MusicGen on GPU -> Result saved to disk -> DB updated.
Hum-to-Beat: Uses musicgen-melody. Takes an uploaded .wav file (humming) and uses it as a structural guide for the generated music.
Stem Separation: Takes a generated track and splits it into 4 separate .wav files to allow the user to edit specific instruments in the frontend.
C. The "Infinite DJ" (UX Hack)
Real-time "tweaking" is handled via Tone.js on the frontend (crossfading and EQ filters) to avoid server latency. The backend provides the pre-computed stems or tracks.
4. Database Schema Relationships
User 1:N Repository
Repository 1:N Commit
Commit 1:1 Commit (Self-referential parent_id)
Commit 1:N Stem (Drums, Bass, etc.)
5. Developer Context (Internal Use)
GPU: NVIDIA RTX 5050 (Handle VRAM carefully, use musicgen-small).
OS: Windows/Linux (Use pool=solo for Celery on Windows).
Audio Format: Standardize on .wav (44.1kHz) for processing and .mp3 for the library preview.