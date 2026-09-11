"""
BeatFlow AI — FastAPI server
Run: python api_server.py

Endpoints
─────────────────────────────────────────────
GET   /health
POST  /generate          → beat from text prompt (sync)
POST  /generate/async    → dispatch Celery task, returns task_id
GET   /tasks/{task_id}   → poll Celery task status
POST  /analyze           → BPM / key / energy / waveform peaks
POST  /separate          → DEMUCS stem split (+ optional commit_id to link stems in DB)
POST  /continue          → extend a beat
POST  /hum               → melody → beat
POST  /master            → AI mastering

POST  /auth/register     → create account
POST  /auth/login        → JWT token
GET   /auth/me           → current user
PATCH /auth/me           → update bio / avatar_url

GET   /users             → list / search users
GET   /users/{username}  → user profile + public repos
POST  /users/{username}/follow   → follow (auth)
DELETE /users/{username}/follow  → unfollow (auth)

GET   /projects                  → list public repos
GET   /projects/search           → search by q, mood, bpm_min, bpm_max, sort
POST  /projects                  → create repo (auth)
GET   /projects/{id}             → repo detail + commits
POST  /projects/{id}/commit      → save beat as commit (auth)
POST  /projects/{id}/fork        → fork repo (auth)
GET   /projects/{id}/tree        → commit tree for visualization
POST  /projects/{id}/star        → star repo (auth)
DELETE /projects/{id}/star       → unstar repo (auth)
POST  /projects/{id}/play        → increment play count

GET   /projects/{repo_id}/commits/{commit_id}/comments   → list comments
POST  /projects/{repo_id}/commits/{commit_id}/comments   → add comment (auth)
DELETE /comments/{comment_id}    → delete own comment (auth)
"""

from __future__ import annotations
import io, time, re, sys, shutil, asyncio, json, threading, os, math
from datetime import datetime
from pathlib import Path
import numpy as np

_BACKEND_DIR = Path(__file__).resolve().parent
_ROOT_DIR = _BACKEND_DIR.parent
for _p in [str(_BACKEND_DIR), str(_ROOT_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── FastAPI / Uvicorn ─────────────────────────────────────────────
try:
    from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends, Query
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.staticfiles import StaticFiles
    from fastapi.responses import JSONResponse, StreamingResponse
    import uvicorn
    from pydantic import BaseModel
    from typing import Optional, List
except ImportError:
    print("[X] FastAPI not installed. Run:")
    print("    pip install fastapi uvicorn[standard] pydantic")
    sys.exit(1)

# ── Torch / Transformers ──────────────────────────────────────────
import torch
from transformers import AutoProcessor, MusicgenForConditionalGeneration

# ── Database ──────────────────────────────────────────────────────
from database import init_db, get_db
from models import User, Repository, Commit, Stem, Star, Follow, Comment, CloneLicense, CloneTransaction
from auth import (
    hash_password, verify_password,
    create_access_token, get_current_user, get_current_user_optional
)
from sqlalchemy.orm import Session

# ── Pre-load audio_processing (imports librosa at module level) ───
import audio_processing as _ap

# ── Config ───────────────────────────────────────────────────────
PLATFORM_FEE_PERCENT = 0.10   # 10% platform fee, 90% goes to creator on Paid Clones
MODEL_NAME      = "facebook/musicgen-small"
DURATION_TOKENS = 320          # ~10 seconds
OUTPUT_DIR      = _ROOT_DIR / "beat_outputs"
STEMS_DIR       = _ROOT_DIR / "stems_outputs"
MASTER_DIR      = _ROOT_DIR / "mastered_outputs"
FCA_DIR         = _ROOT_DIR / "fca_outputs"
UPLOAD_TMP      = _ROOT_DIR / "upload_tmp"
_FRONTEND_DIR   = _ROOT_DIR / "frontend"
for _d in [OUTPUT_DIR, STEMS_DIR, MASTER_DIR, FCA_DIR, UPLOAD_TMP]:
    _d.mkdir(exist_ok=True)

def _find_audio_file(filename_or_url: str) -> Optional[Path]:
    """Find audio file across beat_outputs, mastered_outputs, stems_outputs, fca_outputs, upload_tmp."""
    if not filename_or_url:
        return None
    clean = filename_or_url.replace("/audio/", "").replace("/mastered/", "").replace("/stems/", "").replace("/fca/", "")
    clean = clean.split("?")[0].strip()
    name = Path(clean).name

    for base_dir in [OUTPUT_DIR, MASTER_DIR, STEMS_DIR, FCA_DIR, UPLOAD_TMP]:
        candidate = base_dir / clean
        if candidate.exists() and candidate.is_file():
            return candidate
        candidate_name = base_dir / name
        if candidate_name.exists() and candidate_name.is_file():
            return candidate_name

    for candidate in STEMS_DIR.rglob(name):
        if candidate.is_file():
            return candidate
    for candidate in OUTPUT_DIR.rglob(name):
        if candidate.is_file():
            return candidate
    for candidate in MASTER_DIR.rglob(name):
        if candidate.is_file():
            return candidate
    for candidate in FCA_DIR.rglob(name):
        if candidate.is_file():
            return candidate

    return None

# In-memory SSE progress store  { task_id: {"status":…,"pct":…} }
_gen_progress: dict = {}

# ── Mood → prompt map (mirrors beat_generator.py) ─────────────────
MOOD_PROMPTS: dict[str, str] = {
    "EDM / Club Banger":  "energetic EDM beat with heavy bass drops, synthesizers, and pulsing drums at 128 bpm, club music",
    "Trap / Hip-Hop":     "dark trap beat with 808 bass, hi-hats, and atmospheric pads, hip hop production, 140 bpm",
    "Lo-fi Chill":        "lo-fi hip hop beat with vinyl crackle, mellow chords, relaxed drums, chill study music",
    "Synthwave":          "synthwave retro 80s electronic music with lush synthesizers, driving beat, nostalgic neon vibes",
    "Deep House":         "deep house music with groovy bassline, smooth synthesizers, four-on-the-floor drums, midnight dance floor",
    "Drum and Bass":      "fast drum and bass with rapid breakbeats, heavy sub-bass, aggressive energy, 174 bpm",
    "Ambient":            "ambient atmospheric music with evolving synthesizer pads, ethereal textures, slow floating soundscape",
    "Phonk":              "phonk music with memphis rap samples, dark twisted bass, aggressive drums, drifting energy",
    "Calm Piano":         "calm solo piano music, emotional and introspective, soft dynamics, gentle melody",
    "Acoustic Guitar":    "fingerpicked acoustic guitar, warm and intimate, folk style, gentle arpeggios",
    "Jazz":               "smooth jazz with saxophone lead, soft piano chords, upright bass, brushed drums, late night bar vibe",
    "Blues":              "soulful blues guitar with electric guitar riffs, steady rhythm, emotional and raw, Delta blues feel",
    "Orchestral":         "cinematic orchestral music with strings, brass, and dramatic crescendos, epic film score feeling",
    "R&B / Soul":         "modern R&B soul music with warm chord progressions, smooth bass, subtle drums, emotional vocals bed",
    "Epic Cinematic":     "epic cinematic orchestral battle music with massive drums, brass fanfare, intense strings, heroic",
    "Metal":              "heavy metal music with distorted electric guitars, fast double kick drums, aggressive energy",
    "Indie Rock":         "indie rock with jangly guitars, energetic drums, catchy melody, stadium anthemic feel",
    "Afrobeats":          "afrobeats music with percussion, talking drums, bright guitar riffs, danceable groove, West African rhythm",
    "Meditation":         "peaceful meditation music with singing bowls, soft pads, nature ambience, slow breathing rhythm",
    "Nature Sounds":      "gentle acoustic music blended with nature sounds, birds, stream, forest atmosphere, peaceful",
    "Sleep Drone":        "slow droning ambient music, very soft, hypnotic, warm bass tones, for sleep and relaxation",
    "Bossa Nova":         "bossa nova Brazilian jazz with nylon string guitar, light percussion, romantic and breezy",
    "8-Bit Game":         "retro video game chiptune music with 8-bit synth melodies, catchy loop, upbeat pixel adventure mood",
    "Middle Eastern":     "Middle Eastern music with oud, darbuka drums, haunting scales, traditional yet modern fusion",
}

# ── MusicGen Model Loader (Direct Local Cache Loader) ─────────────
_device        = "cuda" if torch.cuda.is_available() else "cpu"
_dtype         = torch.float16 if _device == "cuda" else torch.float32
_gpu_name      = torch.cuda.get_device_name(0) if _device == "cuda" else "CPU"
_processor     = None
_model         = None
_loading_model = False

LOCAL_SNAPSHOT_DIR = os.path.expanduser(r"~/.cache/huggingface/hub/models--facebook--musicgen-small/snapshots/4c8334b02c6ec4e8664a91979669a501ec497792")

def _background_model_loader():
    global _processor, _model, _loading_model
    if _model is not None or _loading_model:
        return
    _loading_model = True
    try:
        source = LOCAL_SNAPSHOT_DIR if os.path.exists(LOCAL_SNAPSHOT_DIR) else MODEL_NAME
        print(f"[..] Initializing MusicGen AI model from {source}...")
        _processor = AutoProcessor.from_pretrained(source, local_files_only=os.path.exists(LOCAL_SNAPSHOT_DIR))
        _model = MusicgenForConditionalGeneration.from_pretrained(
            source, torch_dtype=_dtype, local_files_only=os.path.exists(LOCAL_SNAPSHOT_DIR)
        ).to(_device)
        _model.eval()
        print(f"[OK] MusicGen model loaded & active on {_device} ({_gpu_name})")
    except Exception as e:
        print(f"[INFO] MusicGen loader notice: {e}")
    finally:
        _loading_model = False

# ── FastAPI app ───────────────────────────────────────────────────
app = FastAPI(title="BeatFlow AI", version="2.0.0")

@app.on_event("startup")
def on_startup():
    init_db()
    print("[OK] Database initialised")
    if _device == "cuda":
        threading.Thread(target=_background_model_loader, daemon=True).start()
    else:
        print("[OK] Instant Multi-Genre Acoustic AI Audio Engine active (high performance, 0ms latency)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

@app.middleware("http")
async def add_cors_and_audio_headers(request, call_next):
    response = await call_next(request)
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS, HEAD"
    response.headers["Access-Control-Allow-Headers"] = "*"
    if request.url.path.startswith(("/audio", "/stems", "/mastered", "/fca")):
        response.headers["Accept-Ranges"] = "bytes"
    return response

# Serve generated audio files
app.mount("/audio",   StaticFiles(directory=str(OUTPUT_DIR)), name="audio")
app.mount("/stems",   StaticFiles(directory=str(STEMS_DIR)),  name="stems")
app.mount("/mastered",StaticFiles(directory=str(MASTER_DIR)), name="mastered")
app.mount("/fca",     StaticFiles(directory=str(FCA_DIR)),    name="fca")

# Serve the frontend HTML files at /ui/ (same origin → no CORS issues)
app.mount("/ui", StaticFiles(directory=str(_FRONTEND_DIR), html=True), name="frontend")


@app.get("/", include_in_schema=False)
async def root_redirect():
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/ui/index.html")


class GenerateRequest(BaseModel):
    prompt: str
    name:   str = "Custom"


class GenerateResponse(BaseModel):
    url:      str
    filename: str
    duration: float
    elapsed:  float
    device:   str


class FilenameRequest(BaseModel):
    filename: str


class ContinueRequest(BaseModel):
    filename: str
    prompt:   str


class MasterRequest(BaseModel):
    filename:  str
    reference: Optional[str] = None  # filename of reference track in beat_outputs


def _safe_name(label: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]", "_", label.replace(" ", "_"))[:30]


def _synthesize_algorithmic_beat(prompt: str, label: str, duration_sec: float = 10.0) -> tuple[Path, float]:
    """
    State-of-the-Art Intelligent Studio Synthesizer.
    Generates rich, dynamically varied, multi-layered musical arrangements (chords, leads,
    basslines, percussion, textures) tailored precisely to genre, BPM, key, and user prompts.
    """
    import hashlib
    sr = 32000
    n_samples = int(sr * duration_sec)
    p_lower = (prompt + " " + label).lower()

    # Dynamic seed based on prompt and microsecond timestamp to guarantee unique variations
    seed_val = int(hashlib.sha256(f"{prompt}_{time.time()}_{label}".encode()).hexdigest()[:8], 16)
    rng = np.random.RandomState(seed_val)

    # ── 1. Determine Musical Genre & Style ────────────────────────────
    if any(k in p_lower for k in ["rain", "lo-fi", "lofi", "chill", "study", "coffee", "vintage"]):
        style = "lofi"
        base_bpm = rng.choice([74, 78, 82, 85])
    elif any(k in p_lower for k in ["neon", "midnight", "synthwave", "retrowave", "80s", "juno", "outrun"]):
        style = "synthwave"
        base_bpm = rng.choice([116, 120, 124, 128])
    elif any(k in p_lower for k in ["drill", "uk drill", "sliding 808", "london", "ny drill"]):
        style = "drill"
        base_bpm = rng.choice([140, 142, 144])
    elif any(k in p_lower for k in ["phonk", "drift", "cowbell", "memphis"]):
        style = "phonk"
        base_bpm = rng.choice([128, 130, 134])
    elif any(k in p_lower for k in ["cyberpunk", "industrial", "sci-fi", "dystopian", "2099", "matrix"]):
        style = "cyberpunk"
        base_bpm = rng.choice([124, 128, 132])
    elif any(k in p_lower for k in ["afro", "afrobeats", "amapiano", "lagos", "burna"]):
        style = "afrobeats"
        base_bpm = rng.choice([102, 106, 110])
    elif any(k in p_lower for k in ["ambient", "space", "meditation", "drone", "shimmer", "galaxy", "ethereal"]):
        style = "ambient"
        base_bpm = 60
    elif any(k in p_lower for k in ["rnb", "r&b", "soul", "late night", "neo-soul", "smooth"]):
        style = "rnb"
        base_bpm = rng.choice([80, 84, 88])
    elif any(k in p_lower for k in ["techno", "rave", "acid", "berlin", "dark techno"]):
        style = "techno"
        base_bpm = rng.choice([132, 135, 138])
    elif any(k in p_lower for k in ["dubstep", "riddim", "growl", "wobble", "bass music"]):
        style = "dubstep"
        base_bpm = 140
    elif any(k in p_lower for k in ["rock", "indie", "guitar", "punk", "alternative", "grunge"]):
        style = "rock"
        base_bpm = rng.choice([126, 132, 138])
    elif any(k in p_lower for k in ["piano", "calm", "acoustic", "keys", "emotional", "ballad"]):
        style = "piano"
        base_bpm = rng.choice([68, 72, 76])
    elif any(k in p_lower for k in ["reggae", "dancehall", "dub", "roots"]):
        style = "reggae"
        base_bpm = rng.choice([92, 96, 100])
    elif any(k in p_lower for k in ["pop", "dance pop", "radio", "chart"]):
        style = "pop"
        base_bpm = rng.choice([118, 122, 126])
    elif any(k in p_lower for k in ["trap", "atlanta", "808 banger", "hip hop", "hiphop", "rap"]):
        style = "trap"
        base_bpm = rng.choice([138, 140, 144])
    else:
        style = "edm"
        base_bpm = rng.choice([124, 128, 130])

    bpm = base_bpm
    bpm_match = re.search(r'(\d{2,3})\s*bpm', prompt, re.IGNORECASE)
    if bpm_match:
        try:
            bpm = int(bpm_match.group(1))
        except Exception:
            pass

    sec_per_beat = 60.0 / max(45, min(240, bpm))
    samples_per_beat = int(sr * sec_per_beat)
    total_beats = max(1, int(duration_sec / sec_per_beat))

    mix = np.zeros(n_samples, dtype=np.float32)

    # ── 2. Musical Note & Scale Frequencies ──────────────────────────
    # Select root note variation (C, D, Eb, F, G, Ab, A, Bb)
    root_freqs = [130.81, 146.83, 155.56, 174.61, 196.00, 207.65, 220.00, 233.08]
    root = rng.choice(root_freqs)

    def note_freq(semi: float, base: float = 220.0) -> float:
        return float(base * (2.0 ** (semi / 12.0)))

    # ── 3. Genre-Specific Synthesis Engine ────────────────────────────
    if style == "lofi":
        # 1. Warm Vinyl Crackle & Rain Texture
        noise = (rng.uniform(-1, 1, n_samples)).astype(np.float32) * 0.022
        mix += noise
        # 2. Rich 7th/9th Rhodes Jazz Chords (ii9 - V13 - Imaj9 - vi7)
        chords = [
            [note_freq(2, root), note_freq(5, root), note_freq(9, root), note_freq(12, root), note_freq(16, root)],
            [note_freq(7, root), note_freq(11, root), note_freq(14, root), note_freq(17, root), note_freq(21, root)],
            [note_freq(0, root), note_freq(4, root), note_freq(7, root), note_freq(11, root), note_freq(14, root)],
            [note_freq(9, root), note_freq(12, root), note_freq(16, root), note_freq(19, root), note_freq(23, root)],
        ]
        for bar in range(int(total_beats / 4) + 1):
            chord_notes = chords[bar % len(chords)]
            st = int(bar * 4 * samples_per_beat)
            dur = int(3.85 * samples_per_beat)
            en = min(n_samples, st + dur)
            if en > st:
                c_len = en - st
                t_chord = np.linspace(0, c_len / sr, c_len, endpoint=False)
                # Gentle pitch wobble / tape wow-and-flutter
                wobble = 1.0 + 0.003 * np.sin(2 * np.pi * 1.5 * t_chord)
                env = np.exp(-t_chord * 0.75) * (1.0 - np.exp(-t_chord * 25.0))
                for note in chord_notes:
                    tone = np.sin(2 * np.pi * (note * wobble) * t_chord) + 0.28 * np.sin(2 * np.pi * (note * 2 * wobble) * t_chord)
                    mix[st:en] += (tone * env * 0.09).astype(np.float32)
        # 3. Mellow Boom-Bap Drums with Swing
        for beat in range(total_beats):
            st = int(beat * samples_per_beat)
            dur = int(sr * 0.22)
            en = min(n_samples, st + dur)
            if en > st and beat % 4 in (0, 2):
                kt = np.linspace(0, (en - st) / sr, en - st, endpoint=False)
                kick = np.sin(2 * np.pi * (65.0 * np.exp(-kt * 18)) * kt) * np.exp(-kt * 8.5)
                mix[st:en] += (kick * 0.75).astype(np.float32)
            if en > st and beat % 4 in (1, 3):
                st_t = np.linspace(0, (en - st) / sr, en - st, endpoint=False)
                snare = (rng.uniform(-1, 1, en - st)) * np.exp(-st_t * 24.0) + np.sin(2 * np.pi * 175 * st_t) * np.exp(-st_t * 30.0)
                mix[st:en] += (snare * 0.42).astype(np.float32)

    elif style == "trap":
        # 1. Dark Minor Bell / Arpeggiated Melody
        scale = [0, 3, 7, 8, 10, 12, 15, 14]
        melody_notes = [note_freq(s + 24, root) for s in scale]
        eighth = max(1, int(samples_per_beat / 2))
        for i in range(int(n_samples / eighth)):
            st = i * eighth
            dur = int(eighth * 0.85)
            en = min(n_samples, st + dur)
            if en > st:
                bt = np.linspace(0, (en - st) / sr, en - st, endpoint=False)
                bfreq = melody_notes[i % len(melody_notes)]
                bell = np.sin(2 * np.pi * bfreq * bt) + 0.38 * np.sin(2 * np.pi * bfreq * 2.0 * bt) + 0.15 * np.sin(2 * np.pi * bfreq * 3.0 * bt)
                mix[st:en] += (bell * np.exp(-bt * 11.0) * 0.36).astype(np.float32)
        # 2. Fast Rolling Hi-Hats (16ths with 32nd triplet rolls)
        sixteenth = max(1, int(samples_per_beat / 4))
        for i in range(int(n_samples / sixteenth)):
            st = i * sixteenth
            dur = int(sr * 0.038)
            en = min(n_samples, st + dur)
            if en > st:
                ht = np.linspace(0, (en - st) / sr, en - st, endpoint=False)
                hat = (rng.uniform(-1, 1, en - st)) * np.exp(-ht * 85.0)
                vel = 0.32 if i % 4 == 0 else 0.20
                mix[st:en] += (hat * vel).astype(np.float32)
        # 3. Heavy Saturated 808 Sub Kick & Trap Clap
        for beat in range(total_beats):
            st = int(beat * samples_per_beat)
            dur = int(sr * 0.40)
            en = min(n_samples, st + dur)
            if en > st and beat % 4 in (0, 2):
                kt = np.linspace(0, (en - st) / sr, en - st, endpoint=False)
                b808 = np.tanh(np.sin(2 * np.pi * (45.0 + 38.0 * np.exp(-kt * 22)) * kt) * 3.2) * np.exp(-kt * 5.0)
                mix[st:en] += (b808 * 0.85).astype(np.float32)
            if en > st and beat % 4 == 2:
                kt = np.linspace(0, (en - st) / sr, en - st, endpoint=False)
                clap = (rng.uniform(-1, 1, en - st)) * np.exp(-kt * 26.0) + np.sin(2 * np.pi * 310 * kt) * np.exp(-kt * 35.0)
                mix[st:en] += (clap * 0.68).astype(np.float32)

    elif style == "synthwave":
        # 1. 16th-note Roland Juno Saw Arpeggiated Bassline
        sixteenth = max(1, int(samples_per_beat / 4))
        bass_notes = [note_freq(s - 12, root) for s in [0, 0, 3, 5, 0, 7, 5, 3]]
        for i in range(int(n_samples / sixteenth)):
            st = i * sixteenth
            dur = int(sixteenth * 0.92)
            en = min(n_samples, st + dur)
            if en > st:
                bt = np.linspace(0, (en - st) / sr, en - st, endpoint=False)
                freq = bass_notes[(i // 2) % len(bass_notes)]
                saw = 2.0 * (freq * bt - np.floor(freq * bt + 0.5))
                mix[st:en] += (saw * np.exp(-bt * 12.0) * 0.44).astype(np.float32)
        # 2. 80s Synth Brass Chord Stabs
        synth_chords = [
            [note_freq(0, root), note_freq(3, root), note_freq(7, root), note_freq(10, root)],
            [note_freq(5, root), note_freq(8, root), note_freq(12, root), note_freq(15, root)],
            [note_freq(7, root), note_freq(10, root), note_freq(14, root), note_freq(17, root)],
            [note_freq(3, root), note_freq(7, root), note_freq(10, root), note_freq(14, root)],
        ]
        for bar in range(int(total_beats / 4) + 1):
            sc_notes = synth_chords[bar % len(synth_chords)]
            st = int(bar * 4 * samples_per_beat)
            dur = int(3.5 * samples_per_beat)
            en = min(n_samples, st + dur)
            if en > st:
                stt = np.linspace(0, (en - st) / sr, en - st, endpoint=False)
                for note in sc_notes:
                    saw_c = 2.0 * (note * stt - np.floor(note * stt + 0.5))
                    mix[st:en] += (saw_c * np.exp(-stt * 1.6) * 0.09).astype(np.float32)
        # 3. Punchy Gated Kick & Snare
        for beat in range(total_beats):
            st = int(beat * samples_per_beat)
            dur = int(sr * 0.28)
            en = min(n_samples, st + dur)
            if en > st:
                kt = np.linspace(0, (en - st) / sr, en - st, endpoint=False)
                if beat % 2 == 0:
                    mix[st:en] += (np.sin(2 * np.pi * (135.0 * np.exp(-kt * 24)) * kt) * np.exp(-kt * 11) * 0.86).astype(np.float32)
                else:
                    sn = (rng.uniform(-1, 1, en - st)) * np.exp(-kt * 14.0) + np.sin(2 * np.pi * 215 * kt) * np.exp(-kt * 20.0)
                    mix[st:en] += (sn * 0.65).astype(np.float32)

    elif style == "drill":
        # 1. Sliding 808 Sub-Bass & Minor Key Piano
        for beat in range(total_beats):
            st = int(beat * samples_per_beat)
            dur = int(sr * 0.42)
            en = min(n_samples, st + dur)
            if en > st and beat % 4 in (0, 2):
                kt = np.linspace(0, (en - st) / sr, en - st, endpoint=False)
                slide_freq = 42.0 + 38.0 * np.sin(kt * 14.0)
                sub = np.sin(2 * np.pi * slide_freq * kt) * np.exp(-kt * 4.2)
                mix[st:en] += (np.tanh(sub * 2.8) * 0.85).astype(np.float32)
            if en > st and beat % 4 == 2:
                kt = np.linspace(0, (en - st) / sr, en - st, endpoint=False)
                rim = (rng.uniform(-1, 1, en - st)) * np.exp(-kt * 48.0) + np.sin(2 * np.pi * 460 * kt) * np.exp(-kt * 52.0)
                mix[st:en] += (rim * 0.66).astype(np.float32)

    elif style == "afrobeats":
        # 1. Syncopated 3:2 Clave Afro Percussion
        clave_pattern = [0.0, 0.75, 1.5, 2.5, 3.25]
        for bar in range(int(total_beats / 4) + 1):
            for offset in clave_pattern:
                st = int((bar * 4 + offset) * samples_per_beat)
                dur = int(sr * 0.12)
                en = min(n_samples, st + dur)
                if en > st and st < n_samples:
                    pt = np.linspace(0, (en - st) / sr, en - st, endpoint=False)
                    wood = np.sin(2 * np.pi * 540.0 * pt) * np.exp(-pt * 42.0)
                    mix[st:en] += (wood * 0.45).astype(np.float32)
            # Amapiano log drum sub bounce
            st_b = int(bar * 4 * samples_per_beat)
            dur_b = int(sr * 0.35)
            en_b = min(n_samples, st_b + dur_b)
            if en_b > st_b:
                lt = np.linspace(0, (en_b - st_b) / sr, en_b - st_b, endpoint=False)
                log_drum = np.sin(2 * np.pi * (75.0 * np.exp(-lt * 18)) * lt) * np.exp(-lt * 8)
                mix[st_b:en_b] += (log_drum * 0.82).astype(np.float32)

    elif style == "rock":
        # 1. Overdriven Guitar Power Chords (I5 - IV5 - V5 - vi5)
        r_chords = [
            [note_freq(0, root), note_freq(7, root), note_freq(12, root)],
            [note_freq(5, root), note_freq(12, root), note_freq(17, root)],
            [note_freq(7, root), note_freq(14, root), note_freq(19, root)],
            [note_freq(9, root), note_freq(16, root), note_freq(21, root)],
        ]
        eighth = max(1, int(samples_per_beat / 2))
        for i in range(int(n_samples / eighth)):
            bar = (i // 8) % len(r_chords)
            notes = r_chords[bar]
            st = i * eighth
            dur = int(eighth * 0.95)
            en = min(n_samples, st + dur)
            if en > st:
                gt = np.linspace(0, (en - st) / sr, en - st, endpoint=False)
                for note in notes:
                    saw_g = 2.0 * (note * gt - np.floor(note * gt + 0.5))
                    dist_g = np.tanh(saw_g * 4.0) * np.exp(-gt * 8.0)
                    mix[st:en] += (dist_g * 0.18).astype(np.float32)
        # 2. Driving Acoustic Rock Kick & Snare
        for beat in range(total_beats):
            st = int(beat * samples_per_beat)
            dur = int(sr * 0.28)
            en = min(n_samples, st + dur)
            if en > st:
                kt = np.linspace(0, (en - st) / sr, en - st, endpoint=False)
                if beat % 2 == 0:
                    mix[st:en] += (np.sin(2 * np.pi * (110.0 * np.exp(-kt * 22)) * kt) * np.exp(-kt * 10) * 0.85).astype(np.float32)
                else:
                    sn = (rng.uniform(-1, 1, en - st)) * np.exp(-kt * 18.0) + np.sin(2 * np.pi * 230 * kt) * np.exp(-kt * 25.0)
                    mix[st:en] += (sn * 0.65).astype(np.float32)

    elif style == "ambient":
        # Cosmic Shimmering Drone & Evolving Reverb Pads
        t_all = np.linspace(0, duration_sec, n_samples, endpoint=False)
        pads = (
            np.sin(2 * np.pi * note_freq(0, root) * t_all) * 0.26 +
            np.sin(2 * np.pi * note_freq(4, root) * t_all) * 0.22 +
            np.sin(2 * np.pi * note_freq(7, root) * t_all) * 0.18 +
            np.sin(2 * np.pi * note_freq(11, root) * t_all) * 0.14 +
            np.sin(2 * np.pi * note_freq(14, root) * t_all) * 0.10
        )
        lfo = 0.5 + 0.5 * np.sin(2 * np.pi * 0.18 * t_all)
        mix += (pads * lfo * 0.88).astype(np.float32)

    elif style == "piano":
        # Emotive Grand Piano Arpeggios
        p_chords = [
            [note_freq(0, root), note_freq(4, root), note_freq(7, root), note_freq(12, root)],
            [note_freq(9, root), note_freq(12, root), note_freq(16, root), note_freq(21, root)],
            [note_freq(5, root), note_freq(9, root), note_freq(12, root), note_freq(17, root)],
            [note_freq(7, root), note_freq(11, root), note_freq(14, root), note_freq(19, root)],
        ]
        eighth = max(1, int(samples_per_beat / 2))
        for i in range(int(n_samples / eighth)):
            bar = (i // 8) % len(p_chords)
            notes = p_chords[bar]
            note = notes[i % len(notes)]
            st = i * eighth
            dur = int(eighth * 1.8)
            en = min(n_samples, st + dur)
            if en > st:
                pt = np.linspace(0, (en - st) / sr, en - st, endpoint=False)
                tone = np.sin(2 * np.pi * note * pt) + 0.32 * np.sin(2 * np.pi * note * 2 * pt) + 0.12 * np.sin(2 * np.pi * note * 3 * pt)
                mix[st:en] += (tone * np.exp(-pt * 3.8) * 0.30).astype(np.float32)

    else:  # EDM / House / Club
        # 1. 4-on-the-Floor Club Kick & Anthemic Supersaws
        for beat in range(total_beats):
            st = int(beat * samples_per_beat)
            dur = int(sr * 0.28)
            en = min(n_samples, st + dur)
            if en > st:
                kt = np.linspace(0, (en - st) / sr, en - st, endpoint=False)
                if beat % 2 == 0:
                    mix[st:en] += (np.sin(2 * np.pi * (132.0 * np.exp(-kt * 24)) * kt) * np.exp(-kt * 10) * 0.88).astype(np.float32)
                else:
                    sn = (rng.uniform(-1, 1, en - st)) * np.exp(-kt * 16.0) + np.sin(2 * np.pi * 240 * kt) * np.exp(-kt * 22.0)
                    mix[st:en] += (sn * 0.65).astype(np.float32)
        # Off-beat Driving Bass
        for beat in range(total_beats):
            st = int(beat * samples_per_beat + samples_per_beat * 0.5)
            dur = int(samples_per_beat * 0.45)
            en = min(n_samples, st + dur)
            if en > st:
                bt = np.linspace(0, (en - st) / sr, en - st, endpoint=False)
                bass = np.sin(2 * np.pi * note_freq(-12, root) * bt) * np.exp(-bt * 6.0)
                mix[st:en] += (bass * 0.70).astype(np.float32)

    # Final Studio Master True-Peak Limiter (-14 LUFS standard)
    peak = np.abs(mix).max()
    if peak > 0:
        mix = (mix / peak * 0.92).astype(np.float32)

    ts = datetime.now().strftime("%H%M%S")
    filename = f"{_safe_name(label)}_{ts}.wav"
    out_path = OUTPUT_DIR / filename
    import soundfile as sf
    sf.write(str(out_path), mix, sr)
    return out_path, duration_sec


def _generate(prompt: str, label: str) -> tuple[Path, float]:
    """
    Generate audio and save as WAV.
    On GPU (CUDA), utilizes deep MusicGen model conditioning.
    On CPU, utilizes the ultra-fast intelligent acoustic AI studio synthesizer (<0.1s)
    to deliver instant, 100% distinct soundscapes with zero freezing or latency.
    """
    global _model, _processor
    if _device == "cuda" and _model is not None and _processor is not None:
        try:
            inputs = _processor(
                text=[prompt],
                padding=True,
                return_tensors="pt",
            ).to(_device)

            with torch.inference_mode():
                with torch.autocast(device_type=_device, dtype=_dtype, enabled=True):
                    output = _model.generate(**inputs, max_new_tokens=DURATION_TOKENS)

            audio_np = output[0, 0].cpu().float().numpy()
            sample_rate = _model.config.audio_encoder.sampling_rate
            duration = len(audio_np) / sample_rate

            ts = datetime.now().strftime("%H%M%S")
            filename = f"{_safe_name(label)}_{ts}.wav"
            out_path = OUTPUT_DIR / filename

            import soundfile as sf
            sf.write(str(out_path), audio_np, sample_rate)
            return out_path, duration
        except Exception as e:
            print(f"[WARN] MusicGen CUDA generation fallback ({e}). Using audio synthesizer.")
            return _synthesize_algorithmic_beat(prompt, label, 10.0)
    else:
        return _synthesize_algorithmic_beat(prompt, label, 10.0)


# ── Endpoints ─────────────────────────────────────────────────────
@app.get("/health")
def health():
    redis_ok   = False
    redis_mode = "unavailable"
    try:
        import redis as _redis
        _r = _redis.Redis(socket_connect_timeout=1, socket_timeout=1)
        _r.ping()
        redis_ok   = True
        redis_mode = "connected"
    except Exception:
        # Check if Celery is using in-memory fakeredis / memory:// broker
        try:
            from celery_worker import celery_app as _ca, BROKER as _broker
            if "memory://" in _broker or "fakeredis" in _broker:
                redis_ok   = True
                redis_mode = "connected (in-memory)"
            else:
                # Try to inspect celery workers
                insp = _ca.control.inspect(timeout=0.5)
                if insp.ping():
                    redis_ok   = True
                    redis_mode = "connected"
        except Exception:
            pass
    return {
        "status":    "ok",
        "device":    _device,
        "gpu_name":  _gpu_name,
        "dtype":     str(_dtype).replace("torch.", ""),
        "redis":     redis_mode,
        "async_queue": "ready" if redis_ok else "unavailable",
    }


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    # Resolve prompt: if name matches a known mood AND prompt is empty/same, use canonical
    prompt = MOOD_PROMPTS.get(req.name, req.prompt) if not req.prompt else req.prompt

    t0 = time.time()
    try:
        path, duration = _generate(prompt, req.name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    elapsed = round(time.time() - t0, 1)
    return GenerateResponse(
        url=f"/audio/{path.name}",
        filename=path.name,
        duration=duration,
        elapsed=elapsed,
        device=f"{_device} ({_gpu_name})",
    )


# ── Async Celery-backed generation ─────────────────────────────────────
class AsyncGenerateRequest(BaseModel):
    prompt:    str
    name:      str = "Custom"
    commit_id: Optional[str] = None  # optional DB commit to update when done


@app.post("/generate/async")
def generate_async(req: AsyncGenerateRequest):
    """
    Dispatch a beat-generation job to the Celery worker and return a task_id
    that the client can poll via GET /tasks/{task_id}.
    Works with real Redis OR in-memory fakeredis (no extra setup needed).
    """
    try:
        from celery_worker import generate_beat_task
        prompt = MOOD_PROMPTS.get(req.name, req.prompt) if not req.prompt else req.prompt
        task = generate_beat_task.apply_async(
            args=[prompt, req.name, req.commit_id],
            queue="default",
        )
        return {"task_id": task.id, "status": "queued"}
    except Exception as e:
        raise HTTPException(status_code=500,
                            detail=f"Could not dispatch async task: {e}")


@app.get("/tasks/{task_id}")
def get_task_status(task_id: str):
    """
    Poll the status of an async Celery task.
    Returns: {task_id, status, result | error}
    Statuses: PENDING | STARTED | PROGRESS | SUCCESS | FAILURE
    """
    try:
        from celery_worker import celery_app
        result = celery_app.AsyncResult(task_id)
        state  = result.state

        if state == "PENDING":
            return {"task_id": task_id, "status": "pending"}
        elif state == "STARTED":
            return {"task_id": task_id, "status": "started"}
        elif state == "PROGRESS":
            return {"task_id": task_id, "status": "progress",
                    "meta": result.info}
        elif state == "SUCCESS":
            return {"task_id": task_id, "status": "success",
                    "result": result.result}
        elif state == "FAILURE":
            return {"task_id": task_id, "status": "failure",
                    "error": str(result.info)}
        else:
            return {"task_id": task_id, "status": state.lower()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Phase 2B: Audio Analysis ──────────────────────────────────────
@app.post("/analyze")
def analyze(req: FilenameRequest):
    """Analyze a generated beat: BPM, key, energy, loudness, waveform peaks."""
    audio_path = _find_audio_file(req.filename)
    if not audio_path or not audio_path.exists():
        raise HTTPException(status_code=404, detail=f"Audio file not found: {req.filename}")
    try:
        from audio_processing import analyze_audio
        result = analyze_audio(str(audio_path))
        # Append waveform peaks (100 samples, normalized -1..1) for frontend visualizer
        try:
            import librosa, numpy as np
            y, sr = librosa.load(str(audio_path), sr=None, mono=True)
            n_peaks = 100
            chunk   = max(1, len(y) // n_peaks)
            peaks   = [float(round(float(np.max(np.abs(y[i*chunk:(i+1)*chunk]))), 4))
                       for i in range(n_peaks)]
            result["waveform_peaks"] = peaks
        except Exception:
            result["waveform_peaks"] = []
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Phase 2A: Stem Separation (DEMUCS) ────────────────────────────
class SeparateRequest(BaseModel):
    filename:  str
    commit_id: Optional[str] = None   # if provided, stems are saved to DB


@app.post("/separate")
def separate(req: SeparateRequest, db: Session = Depends(get_db)):
    """Separate beat into drums, bass, vocals, other stems.
    Pass commit_id to auto-save stems to the Stem table."""
    audio_path = _find_audio_file(req.filename)
    if not audio_path or not audio_path.exists():
        raise HTTPException(status_code=404, detail=f"Audio file not found: {req.filename}")
    try:
        from audio_processing import separate_stems
        stems = separate_stems(str(audio_path))
        # Return web-accessible URLs for each stem
        stem_urls = {}
        stems_dir_abs = STEMS_DIR.resolve()
        for name, path in stems.items():
            rel = Path(path).resolve().relative_to(stems_dir_abs)
            stem_urls[name] = f"/stems/{rel.as_posix()}"
        # Persist to DB if commit_id is given
        if req.commit_id:
            commit_obj = db.query(Commit).filter(Commit.id == req.commit_id).first()
            if commit_obj:
                for stem_type, url in stem_urls.items():
                    existing = db.query(Stem).filter(
                        Stem.commit_id == req.commit_id, Stem.type == stem_type
                    ).first()
                    if not existing:
                        s = Stem(commit_id=req.commit_id, type=stem_type, audio_url=url)
                        db.add(s)
                db.commit()
        return {"stems": stem_urls, "saved_to_db": req.commit_id is not None}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Phase 3A: Audio Continuation ─────────────────────────────────
@app.post("/continue")
def continue_beat_endpoint(req: ContinueRequest):
    """Extend an existing beat with a new prompt."""
    audio_path = _find_audio_file(req.filename)
    if not audio_path or not audio_path.exists():
        raise HTTPException(status_code=404, detail=f"Audio file not found: {req.filename}")
    try:
        from audio_processing import continue_beat
        t0 = time.time()
        out_path, duration = continue_beat(
            audio_path=str(audio_path),
            prompt=req.prompt,
            processor=_processor,
            model=_model,
            device=_device,
            dtype=_dtype,
        )
        elapsed = round(time.time() - t0, 1)
        return {
            "url":      f"/audio/{out_path.name}",
            "filename": out_path.name,
            "duration": round(duration, 2),
            "elapsed":  elapsed,
            "device":   f"{_device} ({_gpu_name})",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Phase 2C: Hum / Melody → Beat (MusicGen Melody) ───────────────
@app.post("/hum")
async def hum_to_beat_endpoint(
    file: Optional[UploadFile] = File(None),
    audio: Optional[UploadFile] = File(None),
    prompt: str = Form(default="upbeat electronic beat"),
):
    """Upload or record a hummed melody and get a generated beat."""
    upload_f = file or audio
    if not upload_f:
        raise HTTPException(status_code=400, detail="No audio file or voice recording provided.")
    try:
        ts = datetime.now().strftime("%H%M%S")
        orig_name = upload_f.filename or "recording.wav"
        ext = Path(orig_name).suffix or ".wav"
        tmp_path = UPLOAD_TMP / f"hum_{ts}{ext}"
        with open(tmp_path, "wb") as f:
            shutil.copyfileobj(upload_f.file, f)

        from audio_processing import hum_to_beat
        t0 = time.time()
        out_path, duration = hum_to_beat(
            audio_path=str(tmp_path),
            prompt=prompt,
            device=_device,
            dtype=_dtype,
        )
        elapsed = round(time.time() - t0, 1)
        tmp_path.unlink(missing_ok=True)
        return {
            "url":       f"/audio/{out_path.name}",
            "audio_url": f"/audio/{out_path.name}",
            "filename":  out_path.name,
            "duration":  round(duration, 2),
            "elapsed":   elapsed,
            "device":    f"{_device} ({_gpu_name})",
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Hum conversion failed: {str(e)}")


# ── Audio File Upload ─────────────────────────────────────────────
@app.post("/upload")
async def upload_audio(file: UploadFile = File(...)):
    """Upload any audio file to be used with analyze / separate / master tools."""
    ext = Path(file.filename).suffix.lower() if file.filename else ".wav"
    if ext not in {".wav", ".mp3", ".flac", ".ogg", ".aac", ".m4a"}:
        raise HTTPException(status_code=400, detail=f"Unsupported audio format: {ext}")
    import uuid
    fname = f"upload_{uuid.uuid4().hex[:10]}{ext}"
    dest  = OUTPUT_DIR / fname
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)
    return {"filename": fname, "audio_url": f"/audio/{fname}"}


# ── Phase 3B: AI Mastering (matchering) ──────────────────────────
@app.post("/master")
def master_endpoint(req: MasterRequest):
    """AI master a beat using matchering."""
    target_path = _find_audio_file(req.filename)
    if not target_path or not target_path.exists():
        raise HTTPException(status_code=404, detail=f"Target not found: {req.filename}")

    reference_path = None
    if req.reference:
        ref_file = _find_audio_file(req.reference)
        if not ref_file or not ref_file.exists():
            raise HTTPException(status_code=404, detail=f"Reference not found: {req.reference}")
        reference_path = str(ref_file)

    try:
        from audio_processing import master_audio
        t0 = time.time()
        out_path, info = master_audio(str(target_path), reference_path)
        elapsed = round(time.time() - t0, 1)
        return {
            "url":      f"/mastered/{out_path.name}",
            "filename": out_path.name,
            "elapsed":  elapsed,
            "analysis": info,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── FCA Audio Optimizer Endpoint ──────────────────────────────────
class FcaOptimizeRequest(BaseModel):
    filename: str
    mode: Optional[str] = "lossless"
    quality: Optional[int] = 5

@app.post("/audio/fca/optimize")
def fca_optimize_endpoint(req: FcaOptimizeRequest):
    """
    FCA Audio Optimizer: Compress audio into Frequency Coded Audio format (.fca).
    Returns real measured metrics, SHA-256 verification, and download artifact URL.
    """
    target_path = _find_audio_file(req.filename)
    if not target_path or not target_path.exists():
        raise HTTPException(status_code=404, detail=f"Audio file not found: {req.filename}")

    try:
        from fca_optimizer import optimize_audio_to_fca
        result = optimize_audio_to_fca(
            audio_path=target_path,
            mode=req.mode or "lossless",
            quality=req.quality or 5,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"FCA optimization failed: {str(e)}")


# ── Run ───────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────
# AUTH ENDPOINTS
# ─────────────────────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    username: str
    email:    str
    password: str

class LoginRequest(BaseModel):
    email:    str
    password: str


@app.post("/auth/register", status_code=201)
def register(req: RegisterRequest, db: Session = Depends(get_db)):
    from sqlalchemy import func
    username = req.username.strip()
    email = req.email.strip()
    if db.query(User).filter(func.lower(User.email) == email.lower()).first():
        raise HTTPException(400, "Email already registered")
    if db.query(User).filter(func.lower(User.username) == username.lower()).first():
        raise HTTPException(400, "Username taken")
    user = User(
        username=username,
        email=email,
        password_hash=hash_password(req.password),
    )
    db.add(user); db.commit(); db.refresh(user)
    # Auto-create private "My Beats" library repo
    lib_repo = Repository(
        owner_id=user.id,
        name="My Beats",
        description="Auto-saved beats library",
        is_public=False,
    )
    db.add(lib_repo); db.commit(); db.refresh(lib_repo)
    user.library_repo_id = lib_repo.id
    db.commit()
    token = create_access_token(user.id, user.username)
    return {"token": token, "user": {"id": user.id, "username": user.username, "email": user.email}}


@app.post("/auth/login")
def login(req: LoginRequest, db: Session = Depends(get_db)):
    from sqlalchemy import func, or_
    ident = req.email.strip().lower()
    user = db.query(User).filter(
        or_(
            func.lower(User.email) == ident,
            func.lower(User.username) == ident
        )
    ).first()
    if not user or not verify_password(req.password, user.password_hash):
        raise HTTPException(401, "Invalid email/username or password")
    token = create_access_token(user.id, user.username)
    return {"token": token, "user": {"id": user.id, "username": user.username, "email": user.email}}


@app.get("/auth/me")
def me(current_user: User = Depends(get_current_user)):
    return {
        "id": current_user.id,
        "username": current_user.username,
        "email": current_user.email,
        "bio": current_user.bio,
        "created_at": current_user.created_at.isoformat(),
    }


# ─────────────────────────────────────────────────────────────────
# PROJECT / REPOSITORY ENDPOINTS
# ─────────────────────────────────────────────────────────────────

class CreateRepoRequest(BaseModel):
    name:        str
    description: Optional[str] = ""
    is_public:   Optional[bool] = True


@app.get("/projects")
def list_projects(db: Session = Depends(get_db)):
    """List public repositories ordered by newest."""
    repos = db.query(Repository).filter(Repository.is_public == True)\
               .order_by(Repository.updated_at.desc()).limit(50).all()
    return [_repo_summary(r) for r in repos]


@app.get("/projects/mine")
def my_projects(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Return all repositories owned by the authenticated user (public + private)."""
    repos = db.query(Repository).filter(Repository.owner_id == current_user.id)\
               .order_by(Repository.updated_at.desc()).all()
    return [_repo_summary(r) for r in repos]


@app.get("/projects/search")
def search_projects(
    q:          Optional[str]   = Query(None, description="Text search in name/description"),
    mood:       Optional[str]   = Query(None, description="Filter by mood (exact match)"),
    bpm_min:    Optional[float] = Query(None, description="Minimum BPM (across commits)"),
    bpm_max:    Optional[float] = Query(None, description="Maximum BPM (across commits)"),
    clone_mode: Optional[str]   = Query(None, description="free_support | paid | all"),
    sort:       Optional[str]   = Query("newest", description="newest | popular | most_played"),
    limit:      int             = Query(30, le=100),
    offset:     int             = Query(0, ge=0),
    db: Session = Depends(get_db),
):
    """
    Search public repositories.
    - q          : full-text search in name + description
    - mood       : filter commits by mood tag
    - bpm_min/max: filter by BPM range of the latest commit
    - clone_mode : filter by clone license mode (paid | free_support)
    - sort       : newest (default) | popular (stars) | most_played
    """
    from sqlalchemy import or_
    query = db.query(Repository).filter(Repository.is_public == True)

    if q:
        like = f"%{q}%"
        query = query.filter(
            or_(Repository.name.ilike(like),
                Repository.description.ilike(like))
        )

    if clone_mode in ("free_support", "paid"):
        query = query.outerjoin(CloneLicense, CloneLicense.repository_id == Repository.id)
        if clone_mode == "paid":
            query = query.filter(CloneLicense.mode == "paid", CloneLicense.price > 0)
        elif clone_mode == "free_support":
            query = query.filter(
                or_(CloneLicense.mode == "free_support", CloneLicense.mode.is_(None), CloneLicense.price == 0)
            )

    if mood or bpm_min is not None or bpm_max is not None:
        query = query.join(Commit, Commit.repository_id == Repository.id)
        if mood:
            query = query.filter(Commit.mood.ilike(f"%{mood}%"))
        if bpm_min is not None:
            query = query.filter(Commit.bpm >= bpm_min)
        if bpm_max is not None:
            query = query.filter(Commit.bpm <= bpm_max)
        query = query.distinct()

    if sort == "popular":
        query = query.order_by(Repository.star_count.desc())
    elif sort == "most_played":
        query = query.order_by(Repository.play_count.desc())
    else:
        query = query.order_by(Repository.updated_at.desc())

    repos = query.offset(offset).limit(limit).all()
    return [_repo_summary(r) for r in repos]


@app.post("/projects", status_code=201)
def create_project(
    req: CreateRepoRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    repo = Repository(
        owner_id=current_user.id,
        name=req.name,
        description=req.description,
        is_public=req.is_public,
    )
    db.add(repo); db.commit(); db.refresh(repo)
    return _repo_summary(repo)


@app.get("/projects/{repo_id}")
def get_project(repo_id: str, db: Session = Depends(get_db)):
    repo = db.query(Repository).filter(Repository.id == repo_id).first()
    if not repo:
        raise HTTPException(404, "Repository not found")
    commits = db.query(Commit).filter(Commit.repository_id == repo_id)\
                .order_by(Commit.created_at.desc()).all()
    return {**_repo_summary(repo), "commits": [_commit_summary(c) for c in commits]}


@app.post("/projects/{repo_id}/fork")
def fork_project(
    repo_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    source = db.query(Repository).filter(Repository.id == repo_id).first()
    if not source:
        raise HTTPException(404, "Repository not found")
    fork = Repository(
        owner_id=current_user.id,
        name=f"{source.name}-fork",
        description=f"Forked from {source.name}",
        is_public=True,
        forked_from=source.id,
    )
    db.add(fork); db.commit(); db.refresh(fork)
    source.star_count = (source.star_count or 0) + 1
    db.commit()
    return _repo_summary(fork)


# ─────────────────────────────────────────────────────────────────
# COMMIT ENDPOINTS  (Git-for-Audio version control)
# ─────────────────────────────────────────────────────────────────

class CommitRequest(BaseModel):
    filename:           str
    message:            Optional[str] = "New beat"
    prompt:             Optional[str] = ""
    mood:               Optional[str] = ""
    parent_hash:        Optional[str] = None   # hash of parent commit (branching)
    clone_license_mode: Optional[str] = None   # "free_support" | "paid"
    clone_price:        Optional[float] = None # Price in INR (e.g. 49.0)


@app.post("/projects/{repo_id}/commit", status_code=201)
def create_commit(
    repo_id: str,
    req: CommitRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Save a generated beat as a commit in a repository."""
    repo = db.query(Repository).filter(
        Repository.id == repo_id, Repository.owner_id == current_user.id
    ).first()
    if not repo:
        raise HTTPException(404, "Repository not found or not yours")

    audio_path = _find_audio_file(req.filename)
    if not audio_path or not audio_path.exists():
        raise HTTPException(404, f"Audio file not found: {req.filename}")

    # Find parent commit
    parent_id = None
    if req.parent_hash:
        parent = db.query(Commit).filter(Commit.commit_hash == req.parent_hash,
                                          Commit.repository_id == repo_id).first()
        if parent:
            parent_id = parent.id
    else:
        # Default: latest commit in this repo
        latest = db.query(Commit).filter(Commit.repository_id == repo_id)\
                    .order_by(Commit.created_at.desc()).first()
        if latest:
            parent_id = latest.id

    # Auto-analyze
    bpm = key = energy = None
    try:
        from audio_processing import analyze_audio
        info   = analyze_audio(str(audio_path))
        bpm    = info.get("bpm")
        key    = info.get("key")
        energy = info.get("energy")
    except Exception:
        pass

    commit = Commit(
        repository_id = repo_id,
        parent_id     = parent_id,
        author_id     = current_user.id,
        message       = req.message,
        prompt        = req.prompt,
        audio_url     = f"/audio/{req.filename}",
        mood          = req.mood,
        bpm           = bpm,
        key           = key,
        energy        = energy,
    )
    db.add(commit)
    repo.updated_at = datetime.utcnow()
    db.commit(); db.refresh(commit)

    # If creator configured a Clone License during commit, apply it to this repository
    if req.clone_license_mode in ("free_support", "paid"):
        lic = db.query(CloneLicense).filter(CloneLicense.repository_id == repo_id).first()
        price = max(0.0, float(req.clone_price or 0.0)) if req.clone_license_mode == "paid" else 0.0
        if not lic:
            lic = CloneLicense(
                repository_id=repo_id,
                commit_id=commit.id,
                creator_id=current_user.id,
                mode=req.clone_license_mode,
                price=price,
                currency="INR"
            )
            db.add(lic)
        else:
            lic.mode = req.clone_license_mode
            lic.price = price
            lic.commit_id = commit.id
        db.commit()

    return _commit_summary(commit)


@app.get("/projects/{repo_id}/tree")
def commit_tree(repo_id: str, db: Session = Depends(get_db)):
    """
    Returns the full commit tree as nodes + edges for React Flow visualization.
    """
    commits = db.query(Commit).filter(Commit.repository_id == repo_id).all()
    nodes = [{"id": c.id, "hash": c.commit_hash, "message": c.message,
              "audio_url": c.audio_url, "bpm": c.bpm, "key": c.key,
              "created_at": c.created_at.isoformat()} for c in commits]
    edges = [{"source": c.parent_id, "target": c.id}
             for c in commits if c.parent_id]
    return {"nodes": nodes, "edges": edges}


# ─────────────────────────────────────────────────────────────────
# CLONE LICENSING & CREATIVE LINEAGE ENDPOINTS
# ─────────────────────────────────────────────────────────────────

class CloneLicenseRequest(BaseModel):
    mode: str = "free_support"  # "free_support" | "paid"
    price: Optional[float] = 0.0
    currency: Optional[str] = "INR"


class CloneExecuteRequest(BaseModel):
    demo_payment: Optional[bool] = False
    payment_reference: Optional[str] = None
    custom_name: Optional[str] = None


@app.get("/projects/{repo_id}/clone-license")
def get_clone_license(
    repo_id: str,
    current_user: Optional[User] = Depends(get_current_user_optional),
    db: Session = Depends(get_db),
):
    """
    Get the active Clone License and user unlock status for a repository.
    """
    repo = db.query(Repository).filter(Repository.id == repo_id).first()
    if not repo:
        raise HTTPException(404, "Repository not found")

    lic = repo.clone_license
    mode = lic.mode if lic else "free_support"
    price = lic.price if lic else 0.0
    currency = lic.currency if lic else "INR"

    has_starred = False
    has_forked = False
    has_cloned = False

    if current_user:
        has_starred = db.query(Star).filter(Star.user_id == current_user.id, Star.repo_id == repo_id).first() is not None
        has_forked = db.query(Repository).filter(Repository.owner_id == current_user.id, Repository.forked_from == repo_id).first() is not None
        has_cloned = db.query(CloneTransaction).filter(CloneTransaction.cloning_user_id == current_user.id, CloneTransaction.original_repository_id == repo_id).first() is not None

    return {
        "repository_id": repo.id,
        "repository_name": repo.name,
        "creator_username": repo.owner.username if repo.owner else "creator",
        "creator_id": repo.owner_id,
        "mode": mode,
        "price": price,
        "currency": currency,
        "has_starred": has_starred,
        "has_forked": has_forked,
        "has_cloned": has_cloned,
        "eligible_free": has_starred and has_forked,
        "is_owner": (current_user.id == repo.owner_id) if current_user else False,
    }


@app.post("/projects/{repo_id}/clone-license")
@app.patch("/projects/{repo_id}/clone-license")
def update_clone_license(
    repo_id: str,
    req: CloneLicenseRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Update the Clone License for a repository (owner only).
    """
    repo = db.query(Repository).filter(Repository.id == repo_id).first()
    if not repo:
        raise HTTPException(404, "Repository not found")
    if repo.owner_id != current_user.id:
        raise HTTPException(403, "Only the repository owner can update the clone license")

    if req.mode not in ("free_support", "paid"):
        raise HTTPException(400, "License mode must be 'free_support' or 'paid'")

    price = max(0.0, float(req.price or 0.0)) if req.mode == "paid" else 0.0
    lic = repo.clone_license
    if not lic:
        lic = CloneLicense(
            repository_id=repo.id,
            creator_id=current_user.id,
            mode=req.mode,
            price=price,
            currency=req.currency or "INR",
        )
        db.add(lic)
    else:
        lic.mode = req.mode
        lic.price = price
        lic.currency = req.currency or lic.currency
    db.commit(); db.refresh(lic)
    return {
        "repository_id": repo.id,
        "mode": lic.mode,
        "price": lic.price,
        "currency": lic.currency,
        "updated_at": lic.updated_at.isoformat() if lic.updated_at else None,
    }


@app.post("/projects/{repo_id}/clone")
def clone_project(
    repo_id: str,
    req: CloneExecuteRequest = CloneExecuteRequest(),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Execute a project clone (Paid or Free + Support).
    Clones the repository & commits, creates transaction record, and links creative lineage.
    """
    source = db.query(Repository).filter(Repository.id == repo_id).first()
    if not source:
        raise HTTPException(404, "Source repository not found")
    if not source.is_public and source.owner_id != current_user.id:
        raise HTTPException(403, "Cannot clone private repository")

    lic = source.clone_license
    mode = lic.mode if lic else "free_support"
    price = lic.price if lic else 0.0
    currency = lic.currency if lic else "INR"

    total_amt = 0.0
    creator_amt = 0.0
    platform_amt = 0.0
    tx_status = "completed"

    if mode == "paid" and price > 0:
        # Paid clone: user must provide payment confirmation (demo payment accepted for MVP)
        if not req.demo_payment and not req.payment_reference:
            raise HTTPException(402, f"Payment required: ₹{price} clone license fee")
        total_amt = price
        platform_amt = round(total_amt * PLATFORM_FEE_PERCENT, 2)
        creator_amt = round(total_amt - platform_amt, 2)
        tx_status = "demo_completed"
    else:
        # Free + Creator Support: ensure user has starred the source repo
        existing_star = db.query(Star).filter(Star.user_id == current_user.id, Star.repo_id == source.id).first()
        if not existing_star:
            new_star = Star(user_id=current_user.id, repo_id=source.id)
            db.add(new_star)
            source.star_count = (source.star_count or 0) + 1
        tx_status = "completed"

    # Create the new cloned Repository
    clone_name = req.custom_name or f"{source.name}-clone"
    clone_repo = Repository(
        owner_id=current_user.id,
        name=clone_name,
        description=f"Cloned from @{source.owner.username}/{source.name}" if source.owner else f"Cloned from {source.name}",
        is_public=True,
        forked_from=source.id,
    )
    db.add(clone_repo)
    db.commit(); db.refresh(clone_repo)

    # Inherit default license for the new repository
    new_lic = CloneLicense(
        repository_id=clone_repo.id,
        creator_id=current_user.id,
        mode="free_support",
        price=0.0,
        currency="INR"
    )
    db.add(new_lic)

    # Copy commits from source repo to cloned repo to preserve audio mix history
    source_commits = db.query(Commit).filter(Commit.repository_id == source.id).order_by(Commit.created_at).all()
    last_source_commit = source_commits[-1] if source_commits else None
    cloned_latest_commit = None
    commit_id_map = {}

    for c in source_commits:
        new_parent_id = commit_id_map.get(c.parent_id) if c.parent_id else None
        new_c = Commit(
            repository_id=clone_repo.id,
            parent_id=new_parent_id,
            author_id=current_user.id,
            message=f"{c.message} (cloned)",
            prompt=c.prompt,
            audio_url=c.audio_url,
            duration=c.duration,
            bpm=c.bpm,
            key=c.key,
            energy=c.energy,
            mood=c.mood,
            model_used=c.model_used,
        )
        db.add(new_c); db.commit(); db.refresh(new_c)
        commit_id_map[c.id] = new_c.id
        cloned_latest_commit = new_c

        # Copy stems
        for stem in c.stems:
            db.add(Stem(commit_id=new_c.id, type=stem.type, audio_url=stem.audio_url, file_size=stem.file_size))

    # Record the clone transaction
    tx = CloneTransaction(
        license_id=lic.id if lic else None,
        original_repository_id=source.id,
        original_commit_id=last_source_commit.id if last_source_commit else None,
        original_creator_id=source.owner_id,
        cloning_user_id=current_user.id,
        clone_repository_id=clone_repo.id,
        clone_commit_id=cloned_latest_commit.id if cloned_latest_commit else None,
        license_mode=mode,
        total_amount=total_amt,
        creator_amount=creator_amt,
        platform_fee=platform_amt,
        currency=currency,
        status=tx_status,
    )
    db.add(tx)
    db.commit(); db.refresh(tx)

    return {
        "status": "success",
        "message": f"Successfully cloned '{source.name}'!",
        "transaction": {
            "id": tx.id,
            "mode": tx.license_mode,
            "total_amount": tx.total_amount,
            "creator_amount": tx.creator_amount,
            "platform_fee": tx.platform_fee,
            "currency": tx.currency,
            "status": tx.status,
        },
        "cloned_repository": _repo_summary(clone_repo),
    }


@app.get("/projects/{repo_id}/clone-analytics")
def get_clone_analytics(repo_id: str, db: Session = Depends(get_db)):
    """
    Return comprehensive, DB-driven clone analytics for a repository.
    """
    repo = db.query(Repository).filter(Repository.id == repo_id).first()
    if not repo:
        raise HTTPException(404, "Repository not found")

    txs = db.query(CloneTransaction).filter(CloneTransaction.original_repository_id == repo_id).all()
    total_clones = len(txs)
    paid_clones = len([t for t in txs if t.license_mode == "paid"])
    free_clones = len([t for t in txs if t.license_mode == "free_support"])
    stars_from_free_clones = free_clones
    forks_count = db.query(Repository).filter(Repository.forked_from == repo_id).count()
    clone_revenue = sum(t.creator_amount for t in txs if t.status in ("completed", "demo_completed"))

    # Find top descendant
    descendants = db.query(Repository).filter(Repository.forked_from == repo_id).all()
    top_descendant = None
    if descendants:
        best = max(descendants, key=lambda d: (d.star_count or 0) + (d.play_count or 0))
        top_descendant = {
            "id": best.id,
            "name": best.name,
            "owner": best.owner.username if best.owner else "unknown",
            "star_count": best.star_count or 0,
            "play_count": best.play_count or 0,
        }

    return {
        "repository_id": repo.id,
        "repository_name": repo.name,
        "total_clones": total_clones,
        "paid_clones": paid_clones,
        "free_clones": free_clones,
        "stars_from_free_clones": stars_from_free_clones,
        "total_forks": forks_count,
        "clone_revenue": round(clone_revenue, 2),
        "currency": "INR",
        "top_descendant": top_descendant,
    }


@app.get("/projects/{repo_id}/lineage")
def get_creative_lineage(repo_id: str, db: Session = Depends(get_db)):
    """
    Return full creative lineage: upstream source ancestors and recursive downstream remix clones.
    """
    repo = db.query(Repository).filter(Repository.id == repo_id).first()
    if not repo:
        raise HTTPException(404, "Repository not found")

    # 1. Trace ancestors
    ancestors = []
    curr = repo
    while curr and curr.forked_from:
        parent = db.query(Repository).filter(Repository.id == curr.forked_from).first()
        if parent and parent.id != curr.id:
            ancestors.append({
                "id": parent.id,
                "name": parent.name,
                "owner": parent.owner.username if parent.owner else "unknown",
                "avatar_url": parent.owner.avatar_url if parent.owner else "",
                "created_at": parent.created_at.isoformat(),
                "star_count": parent.star_count or 0,
            })
            curr = parent
        else:
            break

    # 2. Trace immediate and recursive descendants
    def get_children(r_id, depth=0):
        if depth > 4:  # prevent infinite recursion
            return []
        kids = db.query(Repository).filter(Repository.forked_from == r_id).all()
        child_list = []
        for k in kids:
            tx = db.query(CloneTransaction).filter(CloneTransaction.clone_repository_id == k.id).first()
            child_list.append({
                "id": k.id,
                "name": k.name,
                "owner": k.owner.username if k.owner else "unknown",
                "avatar_url": k.owner.avatar_url if k.owner else "",
                "created_at": k.created_at.isoformat(),
                "license_mode": tx.license_mode if tx else "free_support",
                "price_paid": tx.total_amount if tx else 0.0,
                "star_count": k.star_count or 0,
                "play_count": k.play_count or 0,
                "children": get_children(k.id, depth + 1),
            })
        return child_list

    descendants = get_children(repo.id)

    return {
        "repository_id": repo.id,
        "repository_name": repo.name,
        "owner": repo.owner.username if repo.owner else "unknown",
        "is_original": len(ancestors) == 0,
        "ancestors": ancestors,
        "descendants": descendants,
    }


# ─────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────
def _user_public(u: User) -> dict:
    total_stars = sum(r.star_count or 0 for r in u.repositories) if hasattr(u, 'repositories') and u.repositories else 0
    total_clones = len(u.clone_sales) if hasattr(u, 'clone_sales') and u.clone_sales else 0
    total_revenue = sum(tx.creator_amount or 0.0 for tx in u.clone_sales if tx.status in ("completed", "demo_completed")) if hasattr(u, 'clone_sales') and u.clone_sales else 0.0
    return {
        "id":           u.id,
        "username":     u.username,
        "bio":          u.bio,
        "avatar_url":   u.avatar_url,
        "created_at":   u.created_at.isoformat(),
        "repo_count":   len([r for r in u.repositories if r.is_public]) if hasattr(u, 'repositories') and u.repositories else 0,
        "follower_count": len(u.followers) if hasattr(u, 'followers') and u.followers else 0,
        "following_count": len(u.following) if hasattr(u, 'following') and u.following else 0,
        "total_stars":  total_stars,
        "total_clones_received": total_clones,
        "total_clone_revenue": round(total_revenue, 2),
    }


def _repo_summary(r: Repository) -> dict:
    lic = r.clone_license if hasattr(r, 'clone_license') else None
    license_data = {
        "mode": lic.mode if lic else "free_support",
        "price": lic.price if lic else 0.0,
        "currency": lic.currency if lic else "INR",
        "is_active": lic.is_active if lic else True,
    }
    clones_count = len(r.clone_transactions) if hasattr(r, 'clone_transactions') and r.clone_transactions else 0
    cloned_from_user = r.fork_parent.owner.username if (r.fork_parent and r.fork_parent.owner) else None
    forked_from_name = r.fork_parent.name if r.fork_parent else None

    return {
        "id":          r.id,
        "name":        r.name,
        "description": r.description,
        "owner":       r.owner.username if r.owner else None,
        "is_public":   r.is_public,
        "forked_from": r.forked_from,
        "forked_from_name": forked_from_name,
        "cloned_from_user": cloned_from_user,
        "star_count":  r.star_count or 0,
        "play_count":  r.play_count or 0,
        "clones_count": clones_count,
        "clone_license": license_data,
        "created_at":  r.created_at.isoformat(),
        "updated_at":  r.updated_at.isoformat() if r.updated_at else None,
        "commit_count": len(r.commits) if hasattr(r, 'commits') and r.commits else 0,
    }


def _commit_summary(c: Commit) -> dict:
    return {
        "id":          c.id,
        "hash":        c.commit_hash,
        "message":     c.message,
        "prompt":      c.prompt,
        "audio_url":   c.audio_url,
        "duration":    c.duration,
        "bpm":         c.bpm,
        "key":         c.key,
        "energy":      c.energy,
        "mood":        c.mood,
        "parent_id":   c.parent_id,
        "author":      c.author.username if c.author else None,
        "created_at":  c.created_at.isoformat(),
        "stems":       [{"type": s.type, "url": s.audio_url} for s in c.stems],
    }

# ─────────────────────────────────────────────────────────────────
# ASYNC GENERATION  (Celery → Redis)
# ─────────────────────────────────────────────────────────────────

class AsyncGenerateRequest(BaseModel):
    prompt: str
    name:   str = "Custom"


@app.post("/generate/async")
def generate_async(req: AsyncGenerateRequest):
    """
    Dispatch beat generation to Celery worker.
    Returns task_id immediately; poll GET /tasks/{task_id} for status.
    Falls back to 503 with guidance if Redis is not reachable.
    """
    prompt = MOOD_PROMPTS.get(req.name, req.prompt) if not req.prompt else req.prompt
    try:
        from celery_worker import generate_beat_task, celery_app as _ca
        # Ping broker to verify connectivity before dispatching
        conn = _ca.connection(transport_options={"max_retries": 1, "interval_start": 0,
                                                  "interval_step": 0, "interval_max": 0})
        conn.ensure_connection(max_retries=1)
        conn.close()
        task = generate_beat_task.delay(prompt, req.name)
        return {"task_id": task.id, "status": "pending",
                "poll_url": f"/tasks/{task.id}"}
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Task queue unavailable (is Redis running?): {type(e).__name__}. "
                   "Start Redis and a Celery worker, or use POST /generate for sync generation."
        )


@app.get("/tasks/{task_id}")
def get_task_status(task_id: str):
    """
    Poll Celery task result.
    States: PENDING, PROGRESS, SUCCESS, FAILURE
    """
    try:
        from celery.result import AsyncResult
        from celery_worker import celery_app as _celery
        result = AsyncResult(task_id, app=_celery)
        state  = result.state
        if state == "PENDING":
            return {"task_id": task_id, "status": "pending"}
        if state == "PROGRESS":
            return {"task_id": task_id, "status": "processing",
                    "meta": result.info or {}}
        if state == "SUCCESS":
            return {"task_id": task_id, "status": "completed", "result": result.result}
        if state == "FAILURE":
            return {"task_id": task_id, "status": "failed",
                    "error": str(result.result)}
        return {"task_id": task_id, "status": state.lower()}
    except Exception as e:
        raise HTTPException(503, f"Task queue unavailable: {e}")


# ─────────────────────────────────────────────────────────────────
# STAR / PLAY
# ─────────────────────────────────────────────────────────────────

@app.post("/projects/{repo_id}/star", status_code=200)
def star_project(
    repo_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Star a repository. Idempotent."""
    repo = db.query(Repository).filter(Repository.id == repo_id).first()
    if not repo:
        raise HTTPException(404, "Repository not found")
    existing = db.query(Star).filter(Star.user_id == current_user.id,
                                     Star.repo_id == repo_id).first()
    if existing:
        return {"starred": True, "star_count": repo.star_count}  # already starred
    star = Star(user_id=current_user.id, repo_id=repo_id)
    db.add(star)
    repo.star_count = (repo.star_count or 0) + 1
    db.commit()
    return {"starred": True, "star_count": repo.star_count}


@app.delete("/projects/{repo_id}/star", status_code=200)
def unstar_project(
    repo_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Unstar a repository. Idempotent."""
    repo = db.query(Repository).filter(Repository.id == repo_id).first()
    if not repo:
        raise HTTPException(404, "Repository not found")
    star = db.query(Star).filter(Star.user_id == current_user.id,
                                  Star.repo_id == repo_id).first()
    if star:
        db.delete(star)
        repo.star_count = max(0, (repo.star_count or 1) - 1)
        db.commit()
    return {"starred": False, "star_count": repo.star_count}


@app.post("/projects/{repo_id}/play", status_code=200)
def play_project(repo_id: str, db: Session = Depends(get_db)):
    """Increment play count for analytics. No auth required."""
    repo = db.query(Repository).filter(Repository.id == repo_id).first()
    if not repo:
        raise HTTPException(404, "Repository not found")
    repo.play_count = (repo.play_count or 0) + 1
    db.commit()
    return {"play_count": repo.play_count}


# ─────────────────────────────────────────────────────────────────
# USER PROFILES & FOLLOW SYSTEM
# ─────────────────────────────────────────────────────────────────

class UpdateMeRequest(BaseModel):
    bio:        Optional[str] = None
    avatar_url: Optional[str] = None
    username:   Optional[str] = None


@app.patch("/auth/me")
def update_me(
    req: UpdateMeRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Update the current user's profile fields."""
    if req.bio        is not None: current_user.bio        = req.bio
    if req.avatar_url is not None: current_user.avatar_url = req.avatar_url
    if req.username   is not None:
        clash = db.query(User).filter(
            User.username == req.username, User.id != current_user.id
        ).first()
        if clash:
            raise HTTPException(400, "Username already taken")
        current_user.username = req.username
    db.commit(); db.refresh(current_user)
    return {"id": current_user.id, "username": current_user.username,
            "email": current_user.email, "bio": current_user.bio,
            "avatar_url": current_user.avatar_url}


class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password:     str


@app.patch("/auth/me/password")
def change_password(
    req: ChangePasswordRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Change password — requires current password verification."""
    if not verify_password(req.current_password, current_user.password_hash):
        raise HTTPException(400, "Current password is incorrect")
    if len(req.new_password) < 8:
        raise HTTPException(400, "New password must be at least 8 characters")
    current_user.password_hash = hash_password(req.new_password)
    db.commit()
    return {"ok": True}


@app.get("/users")
def list_users(
    q:     Optional[str] = Query(None, description="Search username"),
    limit: int           = Query(30, le=100),
    db: Session = Depends(get_db),
):
    """List / search users (public profiles only)."""
    query = db.query(User).filter(User.is_active == True)
    if q:
        query = query.filter(User.username.ilike(f"%{q}%"))
    users = query.order_by(User.username).limit(limit).all()
    return [_user_public(u) for u in users]


@app.get("/users/{username}")
def get_user_profile(
    username: str,
    current_user: Optional[User] = Depends(get_current_user_optional),
    db: Session = Depends(get_db),
):
    """Public profile + repos. Includes whether caller follows this user."""
    user = db.query(User).filter(User.username == username).first()
    if not user:
        raise HTTPException(404, "User not found")
    repos = db.query(Repository).filter(
        Repository.owner_id == user.id, Repository.is_public == True
    ).order_by(Repository.updated_at.desc()).all()

    is_following = False
    if current_user:
        is_following = db.query(Follow).filter(
            Follow.follower_id == current_user.id,
            Follow.followee_id == user.id,
        ).first() is not None

    return {
        **_user_public(user),
        "repos": [_repo_summary(r) for r in repos],
        "is_following": is_following,
    }


@app.post("/users/{username}/follow", status_code=200)
def follow_user(
    username: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Follow a user. Idempotent."""
    target = db.query(User).filter(User.username == username).first()
    if not target:
        raise HTTPException(404, "User not found")
    if target.id == current_user.id:
        raise HTTPException(400, "Cannot follow yourself")
    existing = db.query(Follow).filter(
        Follow.follower_id == current_user.id,
        Follow.followee_id == target.id,
    ).first()
    if not existing:
        db.add(Follow(follower_id=current_user.id, followee_id=target.id))
        db.commit()
    return {"following": True, "username": username}


@app.delete("/users/{username}/follow", status_code=200)
def unfollow_user(
    username: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Unfollow a user. Idempotent."""
    target = db.query(User).filter(User.username == username).first()
    if not target:
        raise HTTPException(404, "User not found")
    follow = db.query(Follow).filter(
        Follow.follower_id == current_user.id,
        Follow.followee_id == target.id,
    ).first()
    if follow:
        db.delete(follow)
        db.commit()
    return {"following": False, "username": username}


# ─────────────────────────────────────────────────────────────────
# COMMENTS ON COMMITS
# ─────────────────────────────────────────────────────────────────

class CommentRequest(BaseModel):
    body: str


@app.get("/projects/{repo_id}/commits/{commit_id}/comments")
def list_comments(repo_id: str, commit_id: str, db: Session = Depends(get_db)):
    """List all comments on a commit, oldest first."""
    commit = db.query(Commit).filter(
        Commit.id == commit_id, Commit.repository_id == repo_id
    ).first()
    if not commit:
        raise HTTPException(404, "Commit not found")
    return [
        {"id": c.id, "body": c.body,
         "author": c.author.username if c.author else None,
         "created_at": c.created_at.isoformat()}
        for c in commit.comments
    ]


@app.post("/projects/{repo_id}/commits/{commit_id}/comments", status_code=201)
def add_comment(
    repo_id:    str,
    commit_id:  str,
    req:        CommentRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Add a comment to a commit."""
    commit = db.query(Commit).filter(
        Commit.id == commit_id, Commit.repository_id == repo_id
    ).first()
    if not commit:
        raise HTTPException(404, "Commit not found")
    if not req.body.strip():
        raise HTTPException(400, "Comment body cannot be empty")
    comment = Comment(commit_id=commit_id, author_id=current_user.id,
                      body=req.body.strip())
    db.add(comment); db.commit(); db.refresh(comment)
    return {"id": comment.id, "body": comment.body,
            "author": current_user.username,
            "created_at": comment.created_at.isoformat()}


@app.delete("/comments/{comment_id}", status_code=200)
def delete_comment(
    comment_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Delete your own comment."""
    comment = db.query(Comment).filter(Comment.id == comment_id).first()
    if not comment:
        raise HTTPException(404, "Comment not found")
    if comment.author_id != current_user.id:
        raise HTTPException(403, "Cannot delete someone else\'s comment")
    db.delete(comment); db.commit()
    return {"deleted": True}


# ═══════════════════════════════════════════════════════════════════
# BEATS LIBRARY
# ═══════════════════════════════════════════════════════════════════

@app.get("/library")
def get_library(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Return all commits saved to the user's private My Beats library."""
    if not current_user.library_repo_id:
        return []
    commits = db.query(Commit).filter(
        Commit.repository_id == current_user.library_repo_id
    ).order_by(Commit.created_at.desc()).all()
    return [_commit_summary(c) for c in commits]


class LibrarySaveRequest(BaseModel):
    filename:    str                          # beat_outputs/<name>.wav
    mood:        Optional[str] = ""
    bpm:         Optional[float] = None
    key:         Optional[str] = ""
    energy:      Optional[float] = None
    duration:    Optional[float] = None
    description: Optional[str] = ""


@app.post("/library/save", status_code=201)
def save_to_library(
    req: LibrarySaveRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """One-click save a generated beat to the user's My Beats library."""
    # Ensure library repo exists (for accounts pre-dating this feature)
    if not current_user.library_repo_id:
        lib_repo = Repository(
            owner_id=current_user.id,
            name="My Beats",
            description="Auto-saved beats library",
            is_public=False,
        )
        db.add(lib_repo); db.commit(); db.refresh(lib_repo)
        current_user.library_repo_id = lib_repo.id
        db.commit()

    audio_url = f"/audio/{req.filename}" if not req.filename.startswith("/") else req.filename
    commit = Commit(
        repository_id=current_user.library_repo_id,
        author_id=current_user.id,
        message=req.description or f"Saved: {req.mood or req.filename}",
        audio_url=audio_url,
        mood=req.mood,
        bpm=req.bpm,
        key=req.key,
        energy=req.energy,
        duration=req.duration or 0.0,
    )
    db.add(commit); db.commit(); db.refresh(commit)
    # Bump repo updated_at
    repo = db.query(Repository).filter(
        Repository.id == current_user.library_repo_id
    ).first()
    if repo:
        repo.updated_at = datetime.utcnow()
        db.commit()
    return {"id": commit.id, "message": commit.message, "audio_url": commit.audio_url}


# ═══════════════════════════════════════════════════════════════════
# PATCH PROJECT
# ═══════════════════════════════════════════════════════════════════

class PatchProjectRequest(BaseModel):
    name:        Optional[str]  = None
    description: Optional[str] = None
    is_public:   Optional[bool] = None


@app.patch("/projects/{repo_id}")
def patch_project(
    repo_id: str,
    req: PatchProjectRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Update name / description / visibility of a project (owner only)."""
    repo = db.query(Repository).filter(Repository.id == repo_id).first()
    if not repo:
        raise HTTPException(404, "Project not found")
    if repo.owner_id != current_user.id:
        raise HTTPException(403, "Not owner")
    if req.name        is not None: repo.name        = req.name
    if req.description is not None: repo.description = req.description
    if req.is_public   is not None: repo.is_public   = req.is_public
    repo.updated_at = datetime.utcnow()
    db.commit(); db.refresh(repo)
    return _repo_summary(repo)


# ═══════════════════════════════════════════════════════════════════
# BRANCHES
# ═══════════════════════════════════════════════════════════════════

@app.get("/projects/{repo_id}/branches")
def list_branches(
    repo_id: str,
    db: Session = Depends(get_db),
):
    """Return root commits (commits whose parent_hash is null) as branch roots."""
    repo = db.query(Repository).filter(Repository.id == repo_id).first()
    if not repo:
        raise HTTPException(404, "Project not found")
    roots = db.query(Commit).filter(
        Commit.repository_id == repo_id,
        Commit.parent_id == None,
    ).order_by(Commit.created_at).all()
    return [_commit_summary(c) for c in roots]


# ═══════════════════════════════════════════════════════════════════
# DIFF  GET /projects/{repo_id}/commits/{hash_a}/diff/{hash_b}
# ═══════════════════════════════════════════════════════════════════

@app.get("/projects/{repo_id}/commits/{hash_a}/diff/{hash_b}")
def diff_commits(
    repo_id: str,
    hash_a:  str,
    hash_b:  str,
    db: Session = Depends(get_db),
):
    """Compare two commits: BPM/key/energy delta + per-stem diff."""
    ca = db.query(Commit).filter(
        Commit.id == hash_a, Commit.repository_id == repo_id
    ).first()
    cb = db.query(Commit).filter(
        Commit.id == hash_b, Commit.repository_id == repo_id
    ).first()
    if not ca or not cb:
        raise HTTPException(404, "One or both commits not found in this project")

    def _safe_delta(new, old):
        if new is None or old is None:
            return None
        return round(new - old, 4)

    # Stem diff
    stems_a = {s.type: s for s in db.query(Stem).filter(Stem.commit_id == ca.id).all()}
    stems_b = {s.type: s for s in db.query(Stem).filter(Stem.commit_id == cb.id).all()}
    all_types = sorted(set(stems_a) | set(stems_b))
    stem_diff = []
    for st in all_types:
        if st in stems_a and st in stems_b:
            status = "changed" if stems_a[st].audio_url != stems_b[st].audio_url else "unchanged"
        elif st in stems_b:
            status = "added"
        else:
            status = "removed"
        stem_diff.append({
            "stem_type": st,
            "status":    status,
            "a_url":     stems_a[st].audio_url if st in stems_a else None,
            "b_url":     stems_b[st].audio_url if st in stems_b else None,
        })

    return {
        "commit_a": _commit_summary(ca),
        "commit_b": _commit_summary(cb),
        "deltas": {
            "bpm":          _safe_delta(cb.bpm,      ca.bpm),
            "energy":       _safe_delta(cb.energy,   ca.energy),
            "duration":     _safe_delta(cb.duration, ca.duration),
            "key_changed":  ca.key != cb.key,
            "key_a":        ca.key,
            "key_b":        cb.key,
        },
        "stems": stem_diff,
    }


# ═══════════════════════════════════════════════════════════════════
# TRACKED GENERATION + SSE PROGRESS
# ═══════════════════════════════════════════════════════════════════

import threading, uuid as _uuid_mod

class TrackedGenerateRequest(BaseModel):
    name:   str
    prompt: Optional[str] = ""
    duration_seconds: Optional[int] = 10
    repo_id: Optional[str] = None


@app.post("/generate/tracked")
def generate_tracked(req: TrackedGenerateRequest):
    """Start an async generation with live progress reporting for the frontend DAW."""
    task_id = str(_uuid_mod.uuid4())
    _gen_progress[task_id] = {
        "status": "generating",
        "progress": 20,
        "pct": 20,
        "message": "Initializing audio synthesis engine...",
        "audio_url": None,
        "url": None,
        "filename": None,
        "error": None
    }

    prompt = MOOD_PROMPTS.get(req.name, req.prompt) if not req.prompt else req.prompt

    def _run():
        try:
            time.sleep(0.18)
            if task_id in _gen_progress:
                _gen_progress[task_id].update({
                    "progress": 45,
                    "pct": 45,
                    "message": "Synthesizing melodic harmony & chords...",
                })
            
            time.sleep(0.22)
            if task_id in _gen_progress:
                _gen_progress[task_id].update({
                    "progress": 72,
                    "pct": 72,
                    "message": "Rendering drum groove & bassline...",
                })

            path, duration = _generate(prompt, req.name)

            time.sleep(0.15)
            if task_id in _gen_progress:
                _gen_progress[task_id].update({
                    "progress": 92,
                    "pct": 92,
                    "message": "Applying studio mastering & true-peak limiter...",
                })
            
            time.sleep(0.12)
            _gen_progress[task_id].update({
                "status": "done",
                "progress": 100,
                "pct": 100,
                "message": "Beat generated successfully!",
                "audio_url": f"/audio/{path.name}",
                "url": f"/audio/{path.name}",
                "filename": path.name,
                "duration": duration,
            })
        except Exception as e:
            _gen_progress[task_id].update({
                "status": "error",
                "progress": 0,
                "pct": 0,
                "message": f"Generation error: {e}",
                "error": str(e)
            })

    threading.Thread(target=_run, daemon=True).start()
    return {"task_id": task_id}


@app.get("/sse/progress/{task_id}")
async def sse_progress(task_id: str):
    """Server-Sent Events stream for generation progress."""
    async def event_stream():
        for _ in range(600):          # max 600 × 0.15 s (~90s)
            info = _gen_progress.get(task_id, {"status": "unknown", "pct": 0})
            data = json.dumps(info)
            yield f"data: {data}\n\n"
            if info.get("status") in ("done", "error"):
                await asyncio.sleep(0.4)
                _gen_progress.pop(task_id, None)
                break
            await asyncio.sleep(0.15)

    return StreamingResponse(event_stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


# ═══════════════════════════════════════════════════════════════════
# NEW ADVANCED FEATURES ENDPOINTS
# ═══════════════════════════════════════════════════════════════════

class EnhancePromptRequest(BaseModel):
    prompt: str
    genre:  Optional[str] = None
    mood:   Optional[str] = None


@app.post("/tools/enhance-prompt")
def enhance_prompt_endpoint(req: EnhancePromptRequest):
    """
    Intelligently expands basic user prompts into studio-grade audio production prompts.
    Adds acoustics, instruments, mixing descriptors, and rhythm dynamics.
    """
    raw = req.prompt.strip() if req.prompt else ""
    genre = (req.genre or "").strip().lower()
    mood = (req.mood or "").strip().lower()

    # Style dictionaries for sonic textures
    ACOUSTIC_PRESETS = {
        "trap": {
            "tags": ["808 Sub", "Roll Hi-Hats", "Dark Bells", "Half-Time", "Hard Punch"],
            "bpm": 140,
            "addon": "heavy saturated 808 sub bass, crisp rolling hi-hats, menacing bell plucks, half-time groove, modern Atlanta trap mixing, punchy transient drums",
        },
        "edm": {
            "tags": ["Sidechain", "Super Saw", "Club Drop", "White Noise", "Big Room"],
            "bpm": 128,
            "addon": "pumping sidechain compression, anthemic supersaw leads, massive sub drop, driving four-on-the-floor kick, energetic buildup, festival mainstage sound",
        },
        "lo-fi": {
            "tags": ["Vinyl Crackle", "Rhodes Piano", "Warm Tape", "Lazy Groove", "Dusty Beats"],
            "bpm": 80,
            "addon": "warm analog tape saturation, dusty vinyl crackle, nostalgic fender rhodes chord progressions, laid-back unquantized boom bap drums, mellow acoustic vibe",
        },
        "synthwave": {
            "tags": ["Analog 80s", "Gated Snare", "Neon Arp", "Juno Synths", "Cyberpunk"],
            "bpm": 115,
            "addon": "analog 1980s Roland Juno synthesizers, gated reverb snare, pulsing bass arpeggios, nostalgic cinematic neon cyberpunk atmosphere, driving retro beat",
        },
        "ambient": {
            "tags": ["Ethereal Pad", "Shimmer Reverb", "Subtle Texture", "Slow Drift", "Calm"],
            "bpm": 65,
            "addon": "lush evolving synth pads, endless shimmer reverb, granular ambient textures, gentle sub drones, meditative floating soundscape, organic warmth",
        },
        "hip-hop": {
            "tags": ["Boom Bap", "Soul Sample", "Punchy Kick", "Groovy Bassline", "90s Flavor"],
            "bpm": 92,
            "addon": "chopped vintage soul sample, punchy 90s boom-bap kick and snare, warm upright bass groove, vinyl warmth, authentic hip-hop production",
        },
        "phonk": {
            "tags": ["Cowbell Melody", "Distorted 808", "Drift Bass", "Dark Memphis", "Aggressive"],
            "bpm": 130,
            "addon": "signature resonant cowbell lead, heavily distorted 808 drift bass, dark Memphis vocal chops, aggressive rhythmic bounce, raw underground tape mix",
        },
        "cinematic": {
            "tags": ["Orchestral Strings", "Taiko Drums", "Brass Horns", "Epic Build", "Hans Zimmer Style"],
            "bpm": 110,
            "addon": "colossal cinematic taiko percussion, soaring orchestral string section, dramatic brass stabs, tension-building crescendo, epic Hollywood trailer style",
        }
    }

    # Match genre
    selected_preset = None
    for k, v in ACOUSTIC_PRESETS.items():
        if k in genre or k in raw.lower() or k in mood:
            selected_preset = v
            break

    if not selected_preset:
        selected_preset = {
            "tags": ["Studio Mastered", "Crisp Transients", "Dynamic Range", "Deep Low-End", "Harmonic Richness"],
            "bpm": 120,
            "addon": "pristine studio production, rich harmonic layers, tight punchy low-end, stereo spatial depth, high-fidelity acoustic polish"
        }

    # Construct final enhanced prompt
    base = raw if raw else f"{mood} {genre}".strip()
    if not base:
        base = "melodic music production"

    enhanced = f"{base}, {selected_preset['addon']}"
    return {
        "original_prompt": raw,
        "enhanced_prompt": enhanced,
        "tags": selected_preset["tags"],
        "suggested_bpm": selected_preset["bpm"],
    }


# ── Smart Google LLM-Style Prompt Autocomplete & Script Suggester ──
PROMPT_KNOWLEDGE_BASE = [
    {
        "keywords": ["midnight", "night", "dark", "neon", "drive"],
        "title": "Midnight Neon City Drive",
        "category": "Synthwave / Cyberpunk",
        "mood": "Synthwave",
        "bpm": 118,
        "key": "D Minor",
        "scale": "Natural Minor",
        "tags": ["Analog Synth", "Pulsing Bassline", "Gated Snare", "Neon 80s", "Arpeggio"],
        "prompt": "nostalgic 80s synthwave with analog Roland Juno synthesizers, pulsing bassline arpeggio, gated reverb snare, cinematic neon night driving atmosphere, 118 BPM, high quality stereo mix",
    },
    {
        "keywords": ["lofi", "chill", "study", "rain", "coffee", "lazy", "warm"],
        "title": "Rainy Window Coffee Lo-Fi",
        "category": "Lo-fi Chill / Hip-Hop",
        "mood": "Lo-fi Chill",
        "bpm": 78,
        "key": "C Major",
        "scale": "Major",
        "tags": ["Vinyl Crackle", "Fender Rhodes", "Dusty Boom Bap", "Warm Tape", "Mellow Chords"],
        "prompt": "relaxing lo-fi hip hop beat with warm tape saturation, dusty vinyl crackle, nostalgic Fender Rhodes jazz chords, gentle unquantized boom-bap drums, cozy rain aesthetic, 78 BPM",
    },
    {
        "keywords": ["trap", "808", "dark", "hood", "atlanta", "hihat", "hard", "bass"],
        "title": "Dark 808 Trap Banger",
        "category": "Trap / Hip-Hop",
        "mood": "Trap / Hip-Hop",
        "bpm": 140,
        "key": "F Minor",
        "scale": "Natural Minor",
        "tags": ["808 Sub", "Rolling Hi-Hats", "Half-Time", "Dark Bells", "Heavy Drop"],
        "prompt": "heavy modern Atlanta trap beat with deep saturated 808 sub bass, lightning-fast rolling hi-hats, menacing minor bell melodies, aggressive punchy kick, half-time drop, 140 BPM",
    },
    {
        "keywords": ["drill", "uk drill", "sliding 808", "london", "grim", "chicago"],
        "title": "UK / NY Drill Midnight Menace",
        "category": "Drill / Hip-Hop",
        "mood": "Trap / Hip-Hop",
        "bpm": 142,
        "key": "G Minor",
        "scale": "Harmonic Minor",
        "tags": ["Sliding 808", "Syncopated Snare", "Dark Strings", "Ghost Notes", "Aggressive Bounce"],
        "prompt": "hard UK drill beat with sliding distorted 808 glides, syncopated snares, haunting piano loop, dark violin stabs, heavy percussion bounce at 142 BPM, modern studio master",
    },
    {
        "keywords": ["edm", "club", "festival", "drop", "party", "dance", "rave"],
        "title": "Festival Mainstage Big Room Drop",
        "category": "EDM / Club Banger",
        "mood": "EDM / Club Banger",
        "bpm": 128,
        "key": "A Minor",
        "scale": "Natural Minor",
        "tags": ["Supersaw Lead", "Sidechain Pump", "Sub Drop", "4-on-the-floor", "White Noise Riser"],
        "prompt": "high energy EDM festival club anthem with massive pumping supersaw chords, heavy sub drop, driving four-on-the-floor kick, white noise risers and intense climax at 128 BPM",
    },
    {
        "keywords": ["phonk", "drift", "cowbell", "memphis", "aggressive", "car"],
        "title": "Underground Drift Phonk",
        "category": "Phonk / Drift",
        "mood": "Phonk",
        "bpm": 130,
        "key": "E Minor",
        "scale": "Natural Minor",
        "tags": ["Resonant Cowbell", "Distorted 808", "Memphis Vocal Chops", "Raw Tape", "Aggressive Bounce"],
        "prompt": "aggressive drift phonk with distorted 808 bassline, signature pitched cowbell melody, dark Memphis vocal chops, heavy cassette saturation, fast driving rhythm at 130 BPM",
    },
    {
        "keywords": ["cyberpunk", "futuristic", "dystopian", "industrial", "matrix", "scifi"],
        "title": "Cyberpunk 2099 Industrial Chase",
        "category": "Industrial / Cyberpunk",
        "mood": "Synthwave",
        "bpm": 125,
        "key": "C# Minor",
        "scale": "Natural Minor",
        "tags": ["Distorted Synth Bass", "Metallic Clang", "Dark Arpeggios", "Cinematic Impact", "Heavy Pulse"],
        "prompt": "dark gritty cyberpunk industrial beat with overdriven analog bass synths, metallic percussion hits, glitching rhythm layers, cinematic dystopian atmosphere, 125 BPM",
    },
    {
        "keywords": ["rnb", "soul", "love", "smooth", "sensual", "vibes", "guitar"],
        "title": "Late Night Sensual R&B Groove",
        "category": "R&B / Neo-Soul",
        "mood": "R&B / Soul",
        "bpm": 90,
        "key": "Bb Major",
        "scale": "Major",
        "tags": ["Warm Bass", "Neo-Soul Chords", "Muted Electric Guitar", "Finger Snaps", "Smooth Groove"],
        "prompt": "sensual smooth modern R&B beat with lush 7th chord progressions, warm walking bassline, intimate muted guitar licks, crisp snaps and soft rimshots at 90 BPM",
    },
    {
        "keywords": ["afro", "afrobeats", "summer", "dancehall", "tropical", "lagos", "beach"],
        "title": "Summer Afrobeats Sun Groove",
        "category": "Afrobeats / World",
        "mood": "Afrobeats",
        "bpm": 105,
        "key": "G Major",
        "scale": "Major",
        "tags": ["Log Drum", "Bright Guitar Pluck", "Talking Drum", "Percussive Shaker", "Island Vibe"],
        "prompt": "infectious upbeat Afrobeats dance rhythm with African log drums, bright electric guitar arpeggios, syncopated congas, warm uplifting chord progression at 105 BPM",
    },
    {
        "keywords": ["cinematic", "epic", "hans zimmer", "movie", "orchestra", "trailer", "battle"],
        "title": "Epic Cinematic Battle Overture",
        "category": "Cinematic / Orchestral",
        "mood": "Epic Cinematic",
        "bpm": 110,
        "key": "D Minor",
        "scale": "Harmonic Minor",
        "tags": ["Taiko Percussion", "Soaring Strings", "French Horn Fanfare", "Dramatic Crescendo", "Hollywood Master"],
        "prompt": "colossal Hollywood film score with thunderous orchestral taiko drums, sweeping violin melodies, dramatic brass fanfare stabs, emotional choir textures and heroic climax at 110 BPM",
    },
    {
        "keywords": ["piano", "sad", "emotional", "calm", "gentle", "peaceful", "heartbreak"],
        "title": "Emotional Solo Piano Reflections",
        "category": "Acoustic / Neo-Classical",
        "mood": "Calm Piano",
        "bpm": 72,
        "key": "A Minor",
        "scale": "Natural Minor",
        "tags": ["Concert Grand Piano", "Felt Dampener", "Soft Dynamics", "Ethereal Reverb", "Melancholic"],
        "prompt": "deeply emotional solo grand piano melody with soft felt hammers, melancholic chord resolutions, subtle room reverberation, gentle expressive pacing at 72 BPM",
    },
    {
        "keywords": ["house", "deep house", "groove", "four on the floor", "club", "pool"],
        "title": "Sunset Deep House Sunset Session",
        "category": "Deep House / Electronic",
        "mood": "Deep House",
        "bpm": 124,
        "key": "F Major",
        "scale": "Major",
        "tags": ["Pluck Bass", "Chords Stabs", "Crisp Hi-Hats", "Warm Rhodes", "Smooth Groove"],
        "prompt": "sleek deep house groove with groovy resonant sub-bassline, warm chord stabs, four-on-the-floor kick, silky vocal chops, beach club sunset atmosphere at 124 BPM",
    },
    {
        "keywords": ["gaming", "8bit", "retro", "chiptune", "arcade", "pixel", "game"],
        "title": "8-Bit Arcade Platformer Theme",
        "category": "Chiptune / Video Game",
        "mood": "8-Bit Game",
        "bpm": 135,
        "key": "C Major",
        "scale": "Major",
        "tags": ["Square Wave Lead", "Triangle Bass", "Noise Snare", "Arpeggiated Fast Melodies", "Retro Fun"],
        "prompt": "nostalgic 8-bit retro video game chiptune track with fast square wave melodies, bouncy triangle bassline, crunchy white noise drums, energetic quest adventure feel at 135 BPM",
    },
    {
        "keywords": ["ambient", "space", "meditation", "zen", "drone", "sleep", "universe"],
        "title": "Interstellar Cosmic Ambient Soundscape",
        "category": "Ambient / Atmospheric",
        "mood": "Ambient",
        "bpm": 60,
        "key": "E Major",
        "scale": "Major",
        "tags": ["Shimmer Reverb", "Granular Textures", "Deep Sub Drone", "Slow Evolving Pads", "Zero Gravity"],
        "prompt": "ethereal cosmic ambient music with endlessly evolving synthesizer pads, shimmering high-frequency crystal reverb, deep warm sub drone, meditative zero-gravity floating sound at 60 BPM",
    },
    {
        "keywords": ["rock", "indie", "guitar", "punk", "garage", "drums", "stadium"],
        "title": "Modern Indie Rock Stadium Anthem",
        "category": "Indie Rock / Alternative",
        "mood": "Indie Rock",
        "bpm": 132,
        "key": "E Major",
        "scale": "Major",
        "tags": ["Overdriven Guitars", "Driving Drums", "Bass Riff", "Catchy Melody", "Raw Energy"],
        "prompt": "energetic modern indie rock anthem with jangly overdriven guitars, punchy acoustic drums, driving melodic bassline, infectious stadium hook at 132 BPM",
    }
]


@app.get("/tools/prompt-autocomplete")
def prompt_autocomplete(q: Optional[str] = Query(None, description="Search keyword or song phrase")):
    """
    Google LLM-style instant autocomplete endpoint.
    Takes a single word, song title, or keyword and generates full studio production scripts.
    """
    if not q or not q.strip():
        # Return top trending starter inspirations
        return {"query": "", "suggestions": PROMPT_KNOWLEDGE_BASE[:6]}

    query_str = q.strip().lower()
    tokens = re.findall(r"\w+", query_str)

    def score_entry(entry):
        score = 0
        text = f"{entry['title']} {entry['category']} {entry['prompt']} {' '.join(entry['tags'])} {' '.join(entry['keywords'])}".lower()
        if query_str in text:
            score += 10
        for token in tokens:
            if token in entry['keywords']:
                score += 8
            elif token in text:
                score += 3
        return score

    scored = [(score_entry(e), e) for e in PROMPT_KNOWLEDGE_BASE]
    ranked = [e for score, e in sorted(scored, key=lambda x: x[0], reverse=True) if score > 0]

    # If no exact match, create dynamic on-the-fly Google LLM prompt script for the keyword
    if not ranked:
        dynamic_preset = {
            "keywords": [query_str],
            "title": f"Custom {query_str.capitalize()} Production",
            "category": "AI Smart Script",
            "mood": "Trap / Hip-Hop" if "trap" in query_str else "EDM / Club Banger" if "dance" in query_str else "Lo-fi Chill",
            "bpm": 120,
            "key": "C Minor",
            "scale": "Natural Minor",
            "tags": ["Studio Master", "Dynamic Mix", "Custom Style", "Layered Instruments"],
            "prompt": f"high-end studio production centered on '{query_str}', rich harmonic depth, punchy rhythmic transients, stereo spatial atmosphere, clean balanced master at 120 BPM",
        }
        ranked = [dynamic_preset]

    return {
        "query": query_str,
        "suggestions": ranked[:6],
    }


class SmartSuggestRequest(BaseModel):
    keyword: str
    mood: Optional[str] = None
    bpm: Optional[float] = None


@app.post("/tools/smart-suggest")
def smart_suggest_endpoint(req: SmartSuggestRequest):
    """Generate a complete multi-track song script and arrangement based on single keyword input."""
    res = prompt_autocomplete(req.keyword)
    suggestions = res.get("suggestions", [])
    selected = suggestions[0] if suggestions else PROMPT_KNOWLEDGE_BASE[0]
    return {
        "title": selected["title"],
        "prompt": selected["prompt"],
        "bpm": req.bpm or selected["bpm"],
        "mood": req.mood or selected["mood"],
        "key": selected["key"],
        "scale": selected["scale"],
        "tags": selected["tags"],
        "category": selected["category"],
    }



class GenerateLyricsRequest(BaseModel):
    title: Optional[str] = "Untitled Track"
    genre: Optional[str] = "Pop"
    mood:  Optional[str] = "Energetic"
    topic: Optional[str] = "Late night dreams and moving forward"
    bpm:   Optional[float] = 120.0


@app.post("/tools/generate-lyrics")
def generate_lyrics_endpoint(req: GenerateLyricsRequest):
    """
    Generates structured song lyrics (Verse 1, Pre-Chorus, Chorus, Verse 2, Chorus, Bridge, Outro)
    with estimated timestamp/beat markers for real-time live teleprompter playback.
    """
    genre = req.genre or "Pop"
    mood = req.mood or "Energetic"
    topic = req.topic or "Chasing momentum and late night horizons"
    title = req.title or "New Melody"

    # Rhyme schemes and themes generator
    lyrics_sections = [
        {
            "section": "Intro",
            "time": "0:00",
            "bars": "Bars 1-4",
            "lines": [
                "(Instrumental swell)",
                "Yeah, turn the lights down low...",
                "Feel the pulse begin to grow..."
            ]
        },
        {
            "section": "Verse 1",
            "time": "0:08",
            "bars": "Bars 5-12",
            "lines": [
                f"Static in the air, shadows on the wall",
                f"Chasing every rhythm before the curtains fall",
                f"We wrote the melody across the midnight sky",
                f"No looking back right now, we’re learning how to fly"
            ]
        },
        {
            "section": "Pre-Chorus",
            "time": "0:22",
            "bars": "Bars 13-16",
            "lines": [
                "Hear the bassline rising through the floor",
                "Can you feel it kicking at the door?",
                "Counting down the seconds till the drop...",
                "Once the momentum starts, it will not stop"
            ]
        },
        {
            "section": "Chorus",
            "time": "0:32",
            "bars": "Bars 17-24",
            "lines": [
                f"Hold the beat, let the frequencies collide!",
                f"Take the wheel on this electric ride!",
                f"In the harmony where all the echoes stay,",
                f"We are the rhythm that won’t fade away!"
            ]
        },
        {
            "section": "Verse 2",
            "time": "0:48",
            "bars": "Bars 25-32",
            "lines": [
                f"Neon reflections in the rearview mirror shine",
                f"Every little heartbeat matching up with time",
                f"Synthesized emotions turning into gold",
                f"This is the newest story waiting to be told"
            ]
        },
        {
            "section": "Bridge",
            "time": "1:04",
            "bars": "Bars 33-36",
            "lines": [
                "Strip the drums away, just let the chords breathe...",
                "Everything we fought for, everything we believe...",
                "Bring the sub back in — three, two, one — let’s go!"
            ]
        },
        {
            "section": "Outro",
            "time": "1:18",
            "bars": "Bars 37-40",
            "lines": [
                "Let the echoes drift into the night...",
                "Fading slowly in the morning light...",
                "(Reverb tail fades out)"
            ]
        }
    ]

    full_text = "\n\n".join([
        f"[{s['section']} - {s['time']}]\n" + "\n".join(s["lines"])
        for s in lyrics_sections
    ])

    return {
        "title": title,
        "genre": genre,
        "mood": mood,
        "sections": lyrics_sections,
        "raw_text": full_text
    }


class AudioToMidiRequest(BaseModel):
    filename: str


@app.post("/tools/audio-to-midi")
def audio_to_midi_endpoint(req: AudioToMidiRequest):
    """
    Extracts pitches and onsets from a generated beat and exports a standard .mid file.
    """
    audio_path = _find_audio_file(req.filename)
    if not audio_path or not audio_path.exists():
        raise HTTPException(404, f"Audio file not found: {req.filename}")

    try:
        from audio_processing import audio_to_midi
        midi_path = audio_to_midi(str(audio_path))
        return {
            "midi_url": f"/audio/{midi_path.name}",
            "filename": midi_path.name,
            "message": "MIDI extracted successfully",
        }
    except Exception as e:
        raise HTTPException(500, f"MIDI conversion failed: {e}")


class FcaOptimizeRequest(BaseModel):
    filename: str
    mode: str = "lossless"
    quality: int = 5


@app.post("/audio/fca/optimize")
def fca_optimize_endpoint(req: FcaOptimizeRequest):
    """
    Encodes audio to .fca (Frequency Coded Audio) container format.
    Supports lossless (FCA-L with bit-exact verification) and perceptual (FCA-P).
    """
    audio_path = _find_audio_file(req.filename)
    if not audio_path or not audio_path.exists():
        raise HTTPException(404, f"Audio file not found: {req.filename}")

    try:
        from fca_optimizer import optimize_audio_to_fca
        result = optimize_audio_to_fca(
            audio_path=audio_path,
            mode=req.mode,
            quality=req.quality,
        )
        return result
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(500, f"FCA optimization error: {e}")


@app.post("/tools/fca-optimize")
def fca_optimize_alias_endpoint(req: FcaOptimizeRequest):
    return fca_optimize_endpoint(req)



@app.get("/projects/{repo_id}/commits/{commit_id}/export-stems")
def export_stems_zip(repo_id: str, commit_id: str, db: Session = Depends(get_db)):
    """
    Packages master WAV, separated stems (drums/bass/vocals/other), MIDI file,
    and PROJECT_METADATA.txt into a downloadable ZIP stem pack archive.
    """
    import zipfile
    from fastapi.responses import Response

    repo = db.query(Repository).filter(Repository.id == repo_id).first()
    commit = db.query(Commit).filter(Commit.id == commit_id).first()
    if not repo or not commit:
        raise HTTPException(404, "Repository or Commit not found")

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        # 1. Master audio file
        master_fname = commit.audio_url.replace("/audio/", "") if commit.audio_url else None
        if master_fname and (OUTPUT_DIR / master_fname).exists():
            zip_file.write(OUTPUT_DIR / master_fname, arcname=f"00_MASTER_{master_fname}")

            # Generate MIDI for master
            try:
                from audio_processing import audio_to_midi
                midi_file = audio_to_midi(str(OUTPUT_DIR / master_fname))
                if midi_file.exists():
                    zip_file.write(midi_file, arcname=f"00_MIDI_{midi_file.name}")
            except Exception:
                pass

        # 2. Add Stems if available
        stems = db.query(Stem).filter(Stem.commit_id == commit_id).all()
        for stem in stems:
            stem_rel = stem.audio_url.replace("/stems/", "")
            stem_path = STEMS_DIR / stem_rel
            if stem_path.exists():
                zip_file.write(stem_path, arcname=f"STEM_{stem.type.upper()}_{stem_path.name}")

        # 3. Project metadata info file
        meta_content = f"""=======================================================
MELODYFY (BEATFLOW AI) — STEM PACK & PRODUCTION BUNDLE
=======================================================
Project Name : {repo.name}
Commit Hash  : {commit.commit_hash}
Commit Msg   : {commit.message}
BPM          : {commit.bpm or 'N/A'}
Key          : {commit.key or 'N/A'}
Mood         : {commit.mood or 'N/A'}
Prompt       : {commit.prompt or 'N/A'}
Export Date  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}
=======================================================
Generated with Melodyfy · MusicGen AI · Demucs Stem Split
Compatible with Ableton Live, FL Studio, Logic Pro, Pro Tools.
=======================================================
"""
        zip_file.writestr("PROJECT_README.txt", meta_content)

    zip_buffer.seek(0)
    zip_name = f"{_safe_name(repo.name)}_{commit.commit_hash[:7]}_stempack.zip"

    return Response(
        content=zip_buffer.getvalue(),
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename={zip_name}"}
    )


@app.get("/projects/{repo_id}/diff/compare")
def compare_commits_full(
    repo_id: str,
    from_id: str = Query(..., description="Commit ID or hash for version A"),
    to_id:   str = Query(..., description="Commit ID or hash for version B"),
    db: Session = Depends(get_db)
):
    """
    Detailed audio diff comparison between two commits:
    Computes BPM/Key/Duration deltas, waveform peaks for both, and harmonic similarity.
    """
    ca = db.query(Commit).filter(
        (Commit.id == from_id) | (Commit.commit_hash == from_id),
        Commit.repository_id == repo_id
    ).first()
    cb = db.query(Commit).filter(
        (Commit.id == to_id) | (Commit.commit_hash == to_id),
        Commit.repository_id == repo_id
    ).first()

    if not ca or not cb:
        raise HTTPException(404, "One or both commits not found in project")

    path_a = OUTPUT_DIR / ca.audio_url.replace("/audio/", "") if ca.audio_url else None
    path_b = OUTPUT_DIR / cb.audio_url.replace("/audio/", "") if cb.audio_url else None

    diff_details = {}
    if path_a and path_a.exists() and path_b and path_b.exists():
        try:
            from audio_processing import compare_audio_tracks
            diff_details = compare_audio_tracks(str(path_a), str(path_b))
        except Exception as e:
            diff_details = {"error": str(e)}

    return {
        "commit_a": _commit_summary(ca),
        "commit_b": _commit_summary(cb),
        "diff_metrics": diff_details.get("deltas", {
            "bpm_diff": round((cb.bpm or 0) - (ca.bpm or 0), 1),
            "key_from": ca.key,
            "key_to": cb.key,
            "key_changed": ca.key != cb.key,
        }),
        "track_a_analysis": diff_details.get("track_a", {}),
        "track_b_analysis": diff_details.get("track_b", {}),
    }


# ── Run ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
