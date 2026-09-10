# Melodyfy (BeatFlow AI) — Existing Project Feature & Architecture Audit

This document is a comprehensive audit of all existing features in the **Melodyfy (BeatFlow AI)** codebase. It details the exact connections between the Frontend, API endpoints, Backend handlers, and Database/AI services, using the actual source code.

---

## Architecture Summary

- **Frontend Tech Stack**: Vanilla HTML5, CSS3, JavaScript (ES6+), Web Audio API, Canvas 2D (`visualizer.js`, `nav.js`).
- **Backend Tech Stack**: FastAPI (Python 3.10+), Uvicorn, SQLAlchemy ORM, SQLite (`beatflow.db`), Celery + Redis / Fakeredis.
- **AI & Audio Processing Engines**:
  - **MusicGen** (`facebook/musicgen-small` via HuggingFace Transformers & PyTorch) — Text-to-audio beat generation.
  - **MusicGen Melody** (`facebook/musicgen-melody`) — Hum / audio conditioning to beat generation.
  - **Demucs** (`htdemucs` via PyTorch & SoundFile wrapper) — 4-stem audio separation (drums, bass, vocals, other).
  - **Librosa** — Audio feature extraction (BPM detection, Chroma key estimation, RMS energy, loudness dBFS, spectral centroid, waveform peaks).
  - **PyLoudNorm & Peak Limiter** — Studio AI mastering (EBU R128 loudness normalization to -14 LUFS, -1 dBFS peak limiting).
  - **Custom MIDI Generator** — Audio pitch tracking (`piptrack`) to Standard MIDI Type 0 (`.mid`) file converter.

---

## 1. Authentication & Session Management

### Feature: User Registration
- **Frontend Page**: `/ui/index.html` (Modal)
- **Button/UI**: "Create account" button inside `#register-modal` (also triggered via "Get started" in `nav.js`)
- **Frontend File**: `index.html` (`doRegister()`), `nav.js`
- **API**: `POST /auth/register`
- **Backend**: `api_server.py` → `register()`
- **Database/Service**: `models.py` (`User`, `Repository` models via SQLite `beatflow.db`), `auth.py` (`hash_password` via Argon2)
- **Flow**: User inputs username, email, password → frontend sends JSON to `POST /auth/register` → backend checks uniqueness, hashes password with Argon2, creates `User` record, auto-creates private "My Beats" `Repository`, generates JWT token → returns JWT and user payload → frontend stores in `localStorage ('bf_token', 'bf_user')` and redirects to `dashboard.html`.

### Feature: User Sign In
- **Frontend Page**: `/ui/index.html` (Modal)
- **Button/UI**: "Sign in" button inside `#signin-modal` (also triggered via "Sign in" in `nav.js`)
- **Frontend File**: `index.html` (`doSignIn()`), `nav.js`
- **API**: `POST /auth/login`
- **Backend**: `api_server.py` → `login()`
- **Database/Service**: `models.py` (`User` model via SQLite `beatflow.db`), `auth.py` (`verify_password` via Argon2, `create_access_token`)
- **Flow**: User inputs email/username and password → frontend sends JSON to `POST /auth/login` → backend verifies user exists and password hash matches → backend generates JWT token → returns JWT and user payload → frontend saves token to `localStorage` and redirects to `dashboard.html`.

### Feature: Get Authenticated User Profile
- **Frontend Page**: `/ui/library.html`, `/ui/community.html`, `/ui/repo.html`
- **Button/UI**: Triggered automatically on page load
- **Frontend File**: `library.html` (`loadLibrary()`), `community.html` (`loadSuggested()`), `repo.html` (`loadMe()`)
- **API**: `GET /auth/me`
- **Backend**: `api_server.py` → `me()` (Dependency: `auth.py` → `get_current_user`)
- **Database/Service**: `models.py` (`User` model via SQLite `beatflow.db`)
- **Flow**: Frontend sends Bearer JWT in `Authorization` header → backend validates token and retrieves current `User` → returns user id, username, email, bio, created_at.

### Feature: User Sign Out
- **Frontend Page**: All pages (`nav.js`, `dashboard.html`, `studio.html`, `library.html`, `projects.html`, `settings.html`, `repo.html`, `community.html`, `project_tree.html`)
- **Button/UI**: "Sign out" button in top navigation and user menus
- **Frontend File**: `nav.js`, `dashboard.html` (`signOut()`), `studio.html` (`signOut()`), etc.
- **API**: None (Frontend-only)
- **Backend**: None
- **Database/Service**: None (`localStorage.removeItem('bf_token')`, `localStorage.removeItem('bf_user')`)
- **Flow**: User clicks "Sign out" → frontend clears `localStorage` auth keys → browser redirects to `index.html`.

---

## 2. Studio & AI Music Generation

### Feature: AI Beat Generation (Tracked with SSE Progress)
- **Frontend Page**: `/ui/studio.html`
- **Button/UI**: "Generate Beat" button (`#gen-btn`)
- **Frontend File**: `studio.html` (`generateBeat()`)
- **API**: `POST /generate/tracked` and `GET /sse/progress/{task_id}`
- **Backend**: `api_server.py` → `generate_tracked()`, `api_server.py` → `sse_progress()`, `api_server.py` → `_generate()`
- **Database/Service**: HuggingFace Transformers `MusicgenForConditionalGeneration` (`facebook/musicgen-small`), PyTorch, SoundFile / Torchaudio
- **Flow**: User enters text prompt, BPM, key, scale, mood, duration → clicks "Generate Beat" → frontend calls `POST /generate/tracked` → backend spawns background thread to run MusicGen and initializes SSE progress dict → frontend connects to `GET /sse/progress/{task_id}` using `EventSource` → backend streams progress updates (10%... 100%) → on completion, backend writes WAV to `beat_outputs/` → frontend closes SSE, hides progress bar, and adds interactive beat card with audio player and waveform to session list.

### Feature: AI Prompt Enhancer
- **Frontend Page**: `/ui/studio.html`
- **Button/UI**: "✨ Enhance" button (`#enhance-prompt-btn`)
- **Frontend File**: `studio.html` (`enhancePrompt()`, `appendTag()`)
- **API**: `POST /tools/enhance-prompt`
- **Backend**: `api_server.py` → `enhance_prompt_endpoint()`
- **Database/Service**: Acoustic preset dictionary (Trap, EDM, Lo-fi, Synthwave, Ambient, Hip-Hop, Phonk, Cinematic)
- **Flow**: User inputs basic prompt or selects genre → clicks "✨ Enhance" → frontend sends prompt and genre to `POST /tools/enhance-prompt` → backend matches genre presets and appends studio-grade acoustic descriptors (e.g. "heavy saturated 808 sub bass, crisp rolling hi-hats...") + suggested BPM + style tags → frontend updates prompt textarea with typewriter effect and displays clickable sound tags.

### Feature: Hum / Melody-to-Beat Conditioning
- **Frontend Page**: `/ui/studio.html`
- **Button/UI**: "Convert Hum to Beat" button (`#hum-submit-btn`) inside `#hum-section` (toggled by "🎤 Hum to Beat")
- **Frontend File**: `studio.html` (`triggerHum()`, `submitHum()`)
- **API**: `POST /hum`
- **Backend**: `api_server.py` → `hum_to_beat_endpoint()`, `audio_processing.py` → `hum_to_beat()`
- **Database/Service**: `facebook/musicgen-melody` (Transformers & PyTorch), Librosa, SoundFile
- **Flow**: User records or selects an audio file of them humming/whistling → frontend sends multipart FormData (`file`, `prompt`) to `POST /hum` → backend resamples audio to 32kHz float32 numpy array → passes audio array + text prompt into `MusicgenMelodyForConditionalGeneration` with guidance scale 5.0 → generates matching full beat → saves to `beat_outputs/hum_to_beat_*.wav` → frontend receives generated audio URL and renders beat player card.

### Feature: Beat Audio Continuation (Extend Track)
- **Frontend Page**: `/ui/studio.html`
- **Button/UI**: "⏭ Continue" button (`#cont-btn`)
- **Frontend File**: `studio.html` (`continueBeat()`)
- **API**: `POST /continue`
- **Backend**: `api_server.py` → `continue_beat_endpoint()`, `audio_processing.py` → `continue_beat()`
- **Database/Service**: `facebook/musicgen-small` (Transformers & PyTorch), Librosa, SoundFile
- **Flow**: User selects an existing beat → clicks "⏭ Continue" → frontend sends `filename` and `prompt` to `POST /continue` → backend loads audio file, feeds audio array + text prompt into MusicGen model continuation → outputs extended continuation beat WAV to `beat_outputs/continued_*.wav` → frontend adds extended track to beat list.

### Feature: Audio File Upload
- **Frontend Page**: `/ui/studio.html`
- **Button/UI**: "Upload File" button (`#audio-upload-input`)
- **Frontend File**: `studio.html` (`uploadAudioFile()`)
- **API**: `POST /upload`
- **Backend**: `api_server.py` → `upload_audio()`
- **Database/Service**: Local filesystem (`beat_outputs/upload_*.wav`)
- **Flow**: User picks any `.wav`, `.mp3`, `.flac`, `.ogg`, `.aac`, `.m4a` file → frontend uploads file via multipart FormData → backend validates format and saves to `beat_outputs/` → returns filename and `/audio/*` URL → frontend loads file as the active track for analysis, stem separation, and mastering.

### Feature: Audio Acoustic Analysis
- **Frontend Page**: `/ui/studio.html`
- **Button/UI**: "Analyze Audio" button inside Production Tools
- **Frontend File**: `studio.html` (`analyzeAudio()`)
- **API**: `POST /analyze`
- **Backend**: `api_server.py` → `analyze()`, `audio_processing.py` → `analyze_audio()`
- **Database/Service**: Librosa (BPM `beat_track`, Chroma CQT key detection, RMS energy, spectral centroid, loudness dBFS, waveform peaks)
- **Flow**: User selects beat → clicks "Analyze Audio" → frontend sends `filename` to `POST /analyze` → backend uses Librosa to compute tempo, musical key (major/minor profile correlation), normalized RMS energy, loudness in dBFS, brightness in Hz, duration, and 100 waveform peak points → returns JSON → frontend displays analysis badge card with exact metrics.

### Feature: AI Stem Separation (DEMUCS)
- **Frontend Page**: `/ui/studio.html`
- **Button/UI**: "Separate Stems (DEMUCS)" button
- **Frontend File**: `studio.html` (`separateStems()`, `initStemEditor()`)
- **API**: `POST /separate`
- **Backend**: `api_server.py` → `separate()`, `audio_processing.py` → `separate_stems()`, `run_demucs.py` (Torchaudio soundfile patch + Demucs `htdemucs`)
- **Database/Service**: Demucs `htdemucs` AI model, `models.py` (`Stem` model in SQLite if `commit_id` provided)
- **Flow**: User selects beat → clicks "Separate Stems" → frontend calls `POST /separate` → backend executes `run_demucs.py` subprocess with `htdemucs` model → outputs 4 isolated WAV files (`drums.wav`, `bass.wav`, `vocals.wav`, `other.wav`) into `stems_outputs/` → returns web URLs (`/stems/...`) → frontend opens interactive DAW Multi-Track Stem Mixer with individual track waveforms.

### Feature: DAW Multi-Track Stem Mixer & Audio Effects
- **Frontend Page**: `/ui/studio.html`
- **Button/UI**: Multi-track panel: Mute (`M`), Solo (`S`), Volume slider, Pan slider, Reverb (`Rev`), 8D Spatial Audio (`8D`), Transport controls (Play/Pause, Stop, Loop, Seek bar), Master Volume slider
- **Frontend File**: `studio.html` (`stemMute()`, `stemSolo()`, `stemSetVol()`, `stemSetPan()`, `stemToggleReverb()`, `stemToggle8D()`, `stemReset()`, `setMasterVol()`, `stemPlayPause()`, `stemStop()`, `stemToggleLoop()`, `stemSeek()`, `_updateBeatLed()`, `drawStemWaveform()`, `drawRuler()`)
- **API**: None (Frontend-only Web Audio API engine)
- **Backend**: None (stems served statically from `/stems/*`)
- **Database/Service**: Web Audio API (`AudioContext`, `GainNode`, `StereoPannerNode`, `ConvolverNode` for algorithmic impulse reverb, `OscillatorNode` LFO for 8D binaural pan sweep, `OfflineAudioContext`, `AnalyserNode`)
- **Flow**: Frontend fetches stem WAV files, decodes to `AudioBuffer`s, builds visual peak waveform caches for HTML5 canvas → creates Web Audio node graph per stem track → allows realtime mixing (gain/pan), reverb impulse convolution, sinusoidal 8D spatial panning, synchronised multi-track playback, looping, and beat LED metronome synchronization.

### Feature: Client-Side Stem Mix Offline Render & WAV Export
- **Frontend Page**: `/ui/studio.html`
- **Button/UI**: "Export Mix (WAV)" button (`#export-mix-btn`)
- **Frontend File**: `studio.html` (`exportStemMix()`, `audioBufferToWav()`)
- **API**: None (Frontend-only client-side render)
- **Backend**: None
- **Database/Service**: Web Audio API `OfflineAudioContext`, client-side WAV PCM-16 binary encoder
- **Flow**: User sets volumes, pans, mutes, solos, and reverb in DAW mixer → clicks "Export Mix (WAV)" → frontend instantiates `OfflineAudioContext` matching stem length → renders audio graph with gain and dry/wet convolver nodes faster than realtime → converts resulting `AudioBuffer` to standard 16-bit stereo WAV Blob (`audioBufferToWav`) → creates download link `stem_mix_<timestamp>.wav`.

### Feature: AI Audio Mastering (Matchering / LUFS Normalization)
- **Frontend Page**: `/ui/studio.html`
- **Button/UI**: "Master Audio" button
- **Frontend File**: `studio.html` (`masterAudio()`)
- **API**: `POST /master`
- **Backend**: `api_server.py` → `master_endpoint()`, `audio_processing.py` → `master_audio()`
- **Database/Service**: PyLoudNorm (EBU R128 integrated loudness meter), SoundFile, Librosa
- **Flow**: User clicks "Master Audio" → frontend calls `POST /master` with filename → backend reads audio, measures integrated loudness via PyLoudNorm meter, normalizes loudness to -14.0 LUFS (streaming standard), applies hard peak limiting at -1 dBFS ceiling → writes mastered WAV to `mastered_outputs/mastered_*.wav` → returns URL, original LUFS, mastered LUFS, peak dB, and sample rate → frontend adds mastered track player and displays mastering analysis card.

### Feature: MIDI Note & Pitch Extraction
- **Frontend Page**: `/ui/studio.html`
- **Button/UI**: "Export MIDI (.mid)" button (`#export-midi-btn`)
- **Frontend File**: `studio.html` (`exportMidi()`)
- **API**: `POST /tools/audio-to-midi`
- **Backend**: `api_server.py` → `audio_to_midi_endpoint()`, `audio_processing.py` → `audio_to_midi()`
- **Database/Service**: Librosa `piptrack` & onset detection, Custom Standard MIDI File Type 0 binary encoder
- **Flow**: User selects beat → clicks "Export MIDI" → frontend calls `POST /tools/audio-to-midi` → backend estimates tempo, extracts melodic pitches and note onsets with `piptrack`, quantizes to 88-key piano MIDI note numbers (21–108) and velocities (40–127) → constructs binary MIDI header (`MThd`) and track (`MTrk`) with tempo and note-on/note-off byte events → saves `.mid` to `beat_outputs/` → frontend triggers automatic browser download of the `.mid` file.

### Feature: Full Stem Pack ZIP Downloader
- **Frontend Page**: `/ui/studio.html`, `/ui/repo.html`
- **Button/UI**: "Download Stem Pack (.zip)" button
- **Frontend File**: `studio.html` (`downloadStemPack()`), `repo.html`
- **API**: `GET /projects/{repo_id}/commits/{commit_id}/export-stems`
- **Backend**: `api_server.py` → `export_stems_zip()`
- **Database/Service**: Python `zipfile`, `models.py` (`Repository`, `Commit`, `Stem` in SQLite), `audio_processing.py` → `audio_to_midi()`
- **Flow**: User selects project/commit → clicks download button → backend queries DB for commit master track and associated stems → bundles Master WAV, auto-generated MIDI file (`00_MIDI_*.mid`), individual stem WAVs (`STEM_DRUMS_*`, `STEM_BASS_*`, etc.), and a formatted `PROJECT_README.txt` into a zip archive in-memory → streams zip file to client as `attachment; filename=<project>_<hash>_stempack.zip`.

### Feature: AI Lyrics Generator & Live Teleprompter
- **Frontend Page**: `/ui/studio.html`
- **Button/UI**: "Generate Lyrics" button (`#gen-lyrics-btn`) and "▶ Start Live Teleprompter" button (`#teleprompter-btn`)
- **Frontend File**: `studio.html` (`generateLyrics()`, `toggleTeleprompter()`)
- **API**: `POST /tools/generate-lyrics`
- **Backend**: `api_server.py` → `generate_lyrics_endpoint()`
- **Database/Service**: Algorithmic lyrics structure engine (Verse 1, Pre-Chorus, Chorus, Verse 2, Bridge, Outro with timestamp and bar markers)
- **Flow**: User specifies genre, mood, prompt topic, BPM → clicks "Generate Lyrics" → frontend calls `POST /tools/generate-lyrics` → backend generates structured rhyme-schemed song lyrics with estimated time markers (`0:00`, `0:08`, `0:22`, `0:32`, `0:48`, `1:04`, `1:18`) → frontend displays formatted lyrics card → clicking "▶ Start Live Teleprompter" starts an auto-scrolling timer interval synchronized for live vocal performance.

### Feature: Commit Beat to Project (Git-for-Audio Version Control)
- **Frontend Page**: `/ui/studio.html`
- **Button/UI**: "Save as Commit" button (`#save-commit-btn`)
- **Frontend File**: `studio.html` (`saveToProject()`, `loadRepos()`)
- **API**: `POST /projects/{repo_id}/commit`
- **Backend**: `api_server.py` → `create_commit()`
- **Database/Service**: `models.py` (`Commit`, `Repository` models via SQLite `beatflow.db`), Librosa auto-analysis
- **Flow**: User selects owned project from dropdown, types commit message (e.g. "Added 808 sub bass") → clicks "Save as Commit" → frontend sends payload (`filename`, `message`, `prompt`, `mood`, `parent_hash`) → backend analyzes audio with Librosa to compute BPM, key, energy → finds parent commit ID (or latest commit) → inserts new `Commit` record with unique 8-character hex hash → updates repository `updated_at` → returns commit summary.

### Feature: Quick Save to "My Beats" Library
- **Frontend Page**: `/ui/studio.html`
- **Button/UI**: "Quick Save" button (`#quick-save-btn`)
- **Frontend File**: `studio.html` (`saveToLibrary()`)
- **API**: `POST /library/save`
- **Backend**: `api_server.py` → `save_to_library()`
- **Database/Service**: `models.py` (`Commit`, `Repository`, `User` models via SQLite `beatflow.db`)
- **Flow**: User clicks "Quick Save" on a generated beat → frontend sends filename, mood, BPM, key, energy, duration to `POST /library/save` → backend retrieves user's private `library_repo_id` (auto-creates if missing) → inserts a `Commit` record into the user's private library repository → returns saved commit ID and confirmation toast.

### Feature: Interactive Real-Time Visualizer
- **Frontend Page**: `/ui/studio.html` (and reusable across pages)
- **Button/UI**: Triggered automatically on audio play/pause
- **Frontend File**: `visualizer.js` (`MusicVisualizer`), `studio.html`
- **API**: None (Frontend-only Web Audio API + HTML5 Canvas)
- **Backend**: None
- **Database/Service**: Web Audio API `AnalyserNode` (1024 FFT size), Canvas 2D linear gradients, glow shadow blurs, frequency band energy split (low/mid/high/kick detection)
- **Flow**: When user plays an `<audio>` element, `visualizer.js` attaches `MediaElementAudioSourceNode` to `AnalyserNode` → calculates realtime bass, mid, high frequencies and kick transients → renders animated multi-layered glowing sine waves with film-grain texture.

---

## 3. Project Management & Version Control (Git-for-Music)

### Feature: List User's Projects
- **Frontend Page**: `/ui/projects.html`, `/ui/studio.html`, `/ui/repo.html`
- **Button/UI**: Triggered on page load or project select dropdown
- **Frontend File**: `projects.html` (`loadRepos()`), `studio.html` (`loadRepos()`), `repo.html` (`loadMyRepos()`)
- **API**: `GET /projects/mine`
- **Backend**: `api_server.py` → `my_projects()` (Dependency: `auth.py` → `get_current_user`)
- **Database/Service**: `models.py` (`Repository` model via SQLite `beatflow.db`)
- **Flow**: Frontend sends Bearer token → backend fetches all repositories where `owner_id == current_user.id` ordered by `updated_at` desc → returns list of repo summaries with commit counts, star counts, play counts.

### Feature: List Public Projects & Search / Filter
- **Frontend Page**: `/ui/explore.html`, `/ui/projects.html`, `/ui/index.html`
- **Button/UI**: Search input (`#search-input`), Mood filter chips, BPM range sliders, Sort dropdown (`#sort-select`)
- **Frontend File**: `explore.html` (`loadExplore()`, `debouncedSearch()`), `index.html` (`loadTrending()`, `loadChart()`)
- **API**: `GET /projects/search?q=...&mood=...&bpm_min=...&bpm_max=...&sort=...&limit=...&offset=...`
- **Backend**: `api_server.py` → `search_projects()`
- **Database/Service**: `models.py` (`Repository`, `Commit` models via SQLite `beatflow.db`)
- **Flow**: User types query or toggles mood chips/BPM filters → frontend queries `GET /projects/search` → backend performs text matching on name/description, joins commits to filter by mood tag or BPM range, sorts by `newest`, `popular` (stars), or `most_played` → returns paginated repo list → frontend renders cards with play previews.

### Feature: Create New Project Repository
- **Frontend Page**: `/ui/projects.html`, `/ui/repo.html`
- **Button/UI**: "+ New Project" button → modal submit
- **Frontend File**: `projects.html` (`createProject()`), `repo.html` (`submitCreateRepo()`)
- **API**: `POST /projects`
- **Backend**: `api_server.py` → `create_project()`
- **Database/Service**: `models.py` (`Repository` model via SQLite `beatflow.db`)
- **Flow**: User inputs project name, description, public/private toggle → clicks create → frontend sends JSON to `POST /projects` → backend creates new `Repository` row linked to current user → returns repo summary → frontend reloads project list or redirects to `repo.html?id=<id>`.

### Feature: Get Single Project Detail & Commit History
- **Frontend Page**: `/ui/repo.html`, `/ui/project_tree.html`
- **Button/UI**: Triggered on page load with URL param `?id={repo_id}` or `?project={repo_id}`
- **Frontend File**: `repo.html` (`loadRepo()`), `project_tree.html` (`loadTree()`)
- **API**: `GET /projects/{repo_id}`
- **Backend**: `api_server.py` → `get_project()`
- **Database/Service**: `models.py` (`Repository`, `Commit`, `Stem` models via SQLite `beatflow.db`)
- **Flow**: Frontend fetches `GET /projects/{repo_id}` → backend retrieves repository details along with all commits ordered by `created_at` desc and associated stems → returns full object → frontend populates header, commit timeline, stem lists, and metadata.

### Feature: Edit / Update Project Metadata
- **Frontend Page**: `/ui/projects.html`
- **Button/UI**: "Edit" button on project card → "Save Changes" in modal
- **Frontend File**: `projects.html` (`openEdit()`, `saveEdit()`)
- **API**: `PATCH /projects/{repo_id}`
- **Backend**: `api_server.py` → `patch_project()`
- **Database/Service**: `models.py` (`Repository` model via SQLite `beatflow.db`)
- **Flow**: User modifies project name, description, or visibility → clicks Save → frontend sends JSON to `PATCH /projects/{repo_id}` → backend verifies caller is repository owner, updates fields, bumps `updated_at` → returns updated repo summary → frontend updates UI.

### Feature: Fork Project Repository
- **Frontend Page**: `/ui/explore.html`, `/ui/repo.html`, `/ui/project_tree.html`
- **Button/UI**: "Fork" button (`#fork-btn`)
- **Frontend File**: `explore.html` (`forkRepo()`), `repo.html` (`forkRepo()`), `project_tree.html` (`forkProject()`)
- **API**: `POST /projects/{repo_id}/fork`
- **Backend**: `api_server.py` → `fork_project()`
- **Database/Service**: `models.py` (`Repository` model via SQLite `beatflow.db`)
- **Flow**: User clicks "Fork" on any public project → frontend calls `POST /projects/{repo_id}/fork` with Bearer auth → backend creates a new `Repository` owned by current user with name `<source>-fork`, sets `forked_from = source.id`, increments source `star_count` → returns forked repository summary → frontend redirects to forked project in `repo.html`.

### Feature: Star / Unstar Project Repository
- **Frontend Page**: `/ui/explore.html`, `/ui/repo.html`
- **Button/UI**: "Star" / "Starred" button (`#star-btn`)
- **Frontend File**: `explore.html` (`toggleStar()`), `repo.html` (`toggleStar()`)
- **API**: `POST /projects/{repo_id}/star` and `DELETE /projects/{repo_id}/star`
- **Backend**: `api_server.py` → `star_project()`, `api_server.py` → `unstar_project()`
- **Database/Service**: `models.py` (`Star`, `Repository` models via SQLite `beatflow.db`)
- **Flow**: User clicks star button → frontend sends `POST` or `DELETE` to `/projects/{repo_id}/star` → backend adds/removes unique `Star(user_id, repo_id)` record, updates denormalized `star_count` on `Repository` → returns updated star state and count → frontend updates star counter and button state.

### Feature: Track Audio Play Count
- **Frontend Page**: `/ui/repo.html`
- **Button/UI**: Triggered when audio begins playing on any commit track
- **Frontend File**: `repo.html` (`bumpPlay()`)
- **API**: `POST /projects/{repo_id}/play`
- **Backend**: `api_server.py` → `play_project()`
- **Database/Service**: `models.py` (`Repository` model via SQLite `beatflow.db`)
- **Flow**: User clicks play on a beat in the repository → frontend makes unauthenticated `POST /projects/{repo_id}/play` call → backend increments `Repository.play_count` → returns updated count.

### Feature: Visual Commit Tree (DAG Node Graph)
- **Frontend Page**: `/ui/project_tree.html`
- **Button/UI**: Page load with `?project={repo_id}` or "Visual Tree" button in `repo.html`
- **Frontend File**: `project_tree.html` (`loadTree()`, `renderTree()`, `selectNode()`, `branchFromHere()`)
- **API**: `GET /projects/{repo_id}/tree`
- **Backend**: `api_server.py` → `commit_tree()`
- **Database/Service**: `models.py` (`Commit` model via SQLite `beatflow.db`)
- **Flow**: Frontend calls `GET /projects/{repo_id}/tree` → backend queries all commits for repo, constructs list of nodes (`id`, `hash`, `message`, `audio_url`, `bpm`, `key`, `created_at`) and directed edges (`source: parent_id, target: id`) → frontend renders interactive SVG DAG graph where clicking a node previews audio and displays commit metadata, and "Branch from here" redirects to `studio.html?repo_id={id}&parent_hash={hash}`.

### Feature: List Project Branch Roots
- **Frontend Page**: `/ui/project_tree.html`
- **Button/UI**: "Branches" button
- **Frontend File**: `project_tree.html` (`loadBranches()`)
- **API**: `GET /projects/{repo_id}/branches`
- **Backend**: `api_server.py` → `list_branches()`
- **Database/Service**: `models.py` (`Commit` model via SQLite `beatflow.db`)
- **Flow**: Frontend calls `GET /projects/{repo_id}/branches` → backend queries commits where `parent_id == None` (branch root commits) → returns commit list → frontend renders branch nodes.

### Feature: Audio Version Diff Comparison
- **Frontend Page**: `/ui/repo.html` ("Audio Diff" tab), `/ui/project_tree.html` ("Compare" mode)
- **Button/UI**: "Compare Commits" / "Run Audio Diff" button (`#run-diff-btn`)
- **Frontend File**: `repo.html` (`runAudioDiff()`, `playDiffTrack()`), `project_tree.html` (`runDiff()`)
- **API**: `GET /projects/{repo_id}/diff/compare?from_id={id}&to_id={id}` and `GET /projects/{repo_id}/commits/{hash_a}/diff/{hash_b}`
- **Backend**: `api_server.py` → `compare_commits_full()`, `api_server.py` → `diff_commits()`, `audio_processing.py` → `compare_audio_tracks()`
- **Database/Service**: `models.py` (`Commit`, `Stem` in SQLite), Librosa Chroma CENS cosine similarity calculation
- **Flow**: User selects Commit A (baseline) and Commit B (target) → clicks Compare → backend loads both audio files, analyzes acoustic differences, computes BPM delta (`+4 BPM`), key changes (`Am → Dm`), duration delta, loudness delta, energy delta, harmonic similarity percentage (0–100%), and per-stem changes (`added`, `removed`, `changed`, `unchanged`) → frontend displays diff card, side-by-side A/B players, and stem status breakdown.

### Feature: Commit Comments (List, Add, Delete)
- **Frontend Page**: `/ui/repo.html` ("Comments" tab)
- **Button/UI**: Comment selector dropdown, Comment text input + "Post Comment" button, "Delete" button (`✕`)
- **Frontend File**: `repo.html` (`loadComments()`, `submitComment()`, `deleteComment()`, `populateCommentPicker()`)
- **API**:
  - `GET /projects/{repo_id}/commits/{commit_id}/comments`
  - `POST /projects/{repo_id}/commits/{commit_id}/comments`
  - `DELETE /comments/{comment_id}`
- **Backend**: `api_server.py` → `list_comments()`, `add_comment()`, `delete_comment()`
- **Database/Service**: `models.py` (`Comment`, `Commit`, `User` models via SQLite `beatflow.db`)
- **Flow**: User picks a commit, types feedback, clicks "Post Comment" → frontend calls `POST /projects/{repo_id}/commits/{commit_id}/comments` → backend adds `Comment(commit_id, author_id, body)` → returns created comment → frontend appends to comment stream. Author can click Delete (`✕`) → calls `DELETE /comments/{comment_id}` → backend verifies authorship and deletes comment.

---

## 4. Library & Community Features

### Feature: User Private Beats Library
- **Frontend Page**: `/ui/library.html`
- **Button/UI**: Page load, Search filter input (`#search-input`), Mood chips, Sort dropdown (`#sort-select`), Play/Pause preview
- **Frontend File**: `library.html` (`loadLibrary()`, `renderBeats()`, `filterBeats()`, `sortBeats()`, `togglePlay()`)
- **API**: `GET /library`
- **Backend**: `api_server.py` → `get_library()` (Dependency: `auth.py` → `get_current_user`)
- **Database/Service**: `models.py` (`Commit`, `User` models via SQLite `beatflow.db`)
- **Flow**: Frontend sends Bearer token to `GET /library` → backend fetches all commits belonging to the user's private `library_repo_id` → returns commits with BPM, key, mood, audio URL → frontend renders library grid with waveform canvases and audio players.

### Feature: Add Beat from Library to Project
- **Frontend Page**: `/ui/library.html`
- **Button/UI**: "+ Add to Project" button on beat card → "Add to Project" in modal `#add-modal`
- **Frontend File**: `library.html` (`openAddModal()`, `doAddToProject()`)
- **API**: `GET /projects` and `POST /projects/{repo_id}/commit`
- **Backend**: `api_server.py` → `list_projects()`, `api_server.py` → `create_commit()`
- **Database/Service**: `models.py` (`Repository`, `Commit` models via SQLite `beatflow.db`)
- **Flow**: User clicks "+ Add to Project" on a library beat → modal opens and fetches user's repositories → user selects destination repo and optional message → frontend calls `POST /projects/{repo_id}/commit` with the audio filename → backend commits beat into the target repository.

### Feature: Discover & Search Community Creators
- **Frontend Page**: `/ui/community.html`
- **Button/UI**: Search input (`#search-input`) and page load
- **Frontend File**: `community.html` (`loadSuggested()`, `runSearch()`, `debouncedSearch()`, `buildCard()`)
- **API**: `GET /users?limit=6` and `GET /users?q={query}&limit=12`
- **Backend**: `api_server.py` → `list_users()`
- **Database/Service**: `models.py` (`User`, `Repository`, `Follow` models via SQLite `beatflow.db`)
- **Flow**: Frontend calls `GET /users` → backend queries active users matching search query (if provided), returns public profile objects (username, bio, avatar_url, public repo count, follower count, following count) → frontend renders creator cards.

### Feature: Follow / Unfollow Creator
- **Frontend Page**: `/ui/community.html`
- **Button/UI**: "Follow" / "Following" button on user card
- **Frontend File**: `community.html` (`toggleFollow()`)
- **API**: `POST /users/{username}/follow` and `DELETE /users/{username}/follow`
- **Backend**: `api_server.py` → `follow_user()`, `api_server.py` → `unfollow_user()`
- **Database/Service**: `models.py` (`Follow`, `User` models via SQLite `beatflow.db`)
- **Flow**: User clicks "Follow" on creator card → frontend calls `POST /users/{username}/follow` with Bearer auth → backend inserts unique `Follow(follower_id, followee_id)` record → button updates to "✓ Following". Clicking again sends `DELETE` to unfollow.

---

## 5. Frontend-Only / Client-Side State Features

### Feature: Producer Dashboard Overview
- **Frontend Page**: `/ui/dashboard.html`
- **Button/UI**: Page load, Quick navigation cards to Studio (`studio.html`) and Explore (`explore.html`)
- **Frontend File**: `dashboard.html` (`setGreeting()`, `loadStats()`, `loadProjects()`, `loadActivity()`, `loadTrending()`)
- **API**: None (Client-side rendering)
- **Backend**: None
- **Database/Service**: `localStorage` (`bf_token`, `mfy_projects`, `mfy_beats`, `mfy_plays`, `mfy_collabs`) with mock demo data fallbacks
- **Flow**: Displays dynamic time-of-day greeting based on JWT username, loads producer stats, recent projects, collaboration activity feed, and trending beats from `localStorage` / fallback constants.

### Feature: User Settings & Studio Preferences
- **Frontend Page**: `/ui/settings.html`
- **Button/UI**:
  - Genre chips + "Save Genres" button (`saveGenres()`)
  - Profile inputs + "Save Profile" button (`saveProfile()`)
  - Audio preference selectors + "Save Preferences" button (`saveAudio()`)
  - Password inputs + "Update Password" button (`changePassword()`)
  - "Delete Account" button + Confirmation modal (`confirmDelete()`)
- **Frontend File**: `settings.html`
- **API**: None (Client-side `localStorage` only)
- **Backend**: None
- **Database/Service**: `localStorage` (`mfy_genres`, `mfy_audio_prefs`, `bf_token`)
- **Flow**: Saves user UI preferences, favorite genres, default BPM/Key/Quality studio defaults, and metronome/waveform toggles into browser `localStorage`. Password and Delete account actions run client-side validations and clear `localStorage`.

---

## 6. Backend-Only / Disconnected API Endpoints & Workers

These endpoints exist and are fully implemented in the backend code, but are currently disconnected or unused by the active frontend HTML pages:

### 1. `GET /health`
- **Backend File**: `api_server.py` → `health()`
- **Functionality**: Returns server status, active PyTorch compute device (`cuda` or `cpu`), GPU name, Torch dtype, and Redis/Celery queue connection status.
- **Frontend Status**: Disconnected (backend diagnostic endpoint).

### 2. `POST /generate` (Synchronous Generation)
- **Backend File**: `api_server.py` → `generate()`
- **Functionality**: Generates a MusicGen beat synchronously in the HTTP request thread and returns the WAV path and generation elapsed time.
- **Frontend Status**: Disconnected (`studio.html` uses `POST /generate/tracked` with SSE progress instead).

### 3. `POST /generate/async` & `GET /tasks/{task_id}` (Celery Queue Generation)
- **Backend File**: `api_server.py` → `generate_async()`, `get_task_status()`, `celery_worker.py` → `generate_beat_task()`
- **Functionality**: Dispatches beat generation to a standalone Celery worker via Redis / Fakeredis broker, allowing polling of task status (`PENDING`, `STARTED`, `PROGRESS`, `SUCCESS`, `FAILURE`).
- **Frontend Status**: Disconnected (frontend uses threaded SSE generation endpoint `POST /generate/tracked`).

### 4. Celery Worker Async Tasks (`celery_worker.py`)
- **Backend File**: `celery_worker.py`
  - `beatflow.generate_beat` → `generate_beat_task()`
  - `beatflow.separate_stems` → `separate_stems_task()`
  - `beatflow.analyze_audio` → `analyze_audio_task()`
- **Functionality**: Background asynchronous task queue execution for heavy GPU/CPU tasks.
- **Frontend Status**: Backend worker infrastructure only.

### 5. `PATCH /auth/me` (Update Bio / Username / Avatar)
- **Backend File**: `api_server.py` → `update_me()`
- **Functionality**: Authenticated endpoint to update user profile bio, avatar URL, and username in SQLite database.
- **Frontend Status**: Disconnected (`settings.html` profile save is currently client-side only).

### 6. `PATCH /auth/me/password` (Change Password)
- **Backend File**: `api_server.py` → `change_password()`
- **Functionality**: Authenticated endpoint verifying current Argon2 password and updating to new hashed password in SQLite database.
- **Frontend Status**: Disconnected (`settings.html` password form validates in JS without calling this API).

### 7. `GET /users/{username}` (Public User Profile)
- **Backend File**: `api_server.py` → `get_user_profile()`
- **Functionality**: Returns full user public profile details, list of public repositories, follower/following counts, and whether requesting user follows them.
- **Frontend Status**: Disconnected (available for standalone creator profile pages).

### 8. Direct CLI Beat Generator (`beat_generator.py`)
- **Backend File**: `beat_generator.py`
- **Functionality**: Standalone interactive terminal CLI script for generating MusicGen beats from 25 mood presets or custom prompts without launching the web server.
- **Frontend Status**: Standalone CLI script.

---

## 7. Complete Frontend-to-Backend Map

| # | Feature Name | Frontend Page | Trigger UI Element | Frontend File / Function | API Endpoint | Backend File & Handler | Database / AI Service |
|---|--------------|---------------|-------------------|--------------------------|--------------|------------------------|-----------------------|
| 1 | Register Account | `/ui/index.html` | "Create account" button | `index.html` → `doRegister()` | `POST /auth/register` | `api_server.py` → `register()` | SQLite `User`, `Repository`, Argon2 |
| 2 | Sign In / Login | `/ui/index.html` | "Sign in" button | `index.html` → `doSignIn()` | `POST /auth/login` | `api_server.py` → `login()` | SQLite `User`, JWT, Argon2 |
| 3 | Authenticated User | All pages | Page Load | `library.html`, `repo.html` | `GET /auth/me` | `api_server.py` → `me()` | SQLite `User` |
| 4 | Sign Out | All pages | "Sign out" button | `nav.js`, `studio.html` | None (Client-only) | None | `localStorage` |
| 5 | AI Beat Generation | `/ui/studio.html` | "Generate Beat" button | `studio.html` → `generateBeat()` | `POST /generate/tracked` | `api_server.py` → `generate_tracked()` | `facebook/musicgen-small` |
| 6 | Generation Progress | `/ui/studio.html` | Progress Bar | `studio.html` → `EventSource` | `GET /sse/progress/{task_id}` | `api_server.py` → `sse_progress()` | In-memory SSE stream |
| 7 | Prompt Enhancer | `/ui/studio.html` | "✨ Enhance" button | `studio.html` → `enhancePrompt()` | `POST /tools/enhance-prompt` | `api_server.py` → `enhance_prompt_endpoint()` | Acoustic Presets Engine |
| 8 | Hum to Beat | `/ui/studio.html` | "Convert Hum" button | `studio.html` → `submitHum()` | `POST /hum` | `api_server.py` → `hum_to_beat_endpoint()` | `facebook/musicgen-melody` |
| 9 | Beat Continuation | `/ui/studio.html` | "⏭ Continue" button | `studio.html` → `continueBeat()` | `POST /continue` | `api_server.py` → `continue_beat_endpoint()` | `facebook/musicgen-small` |
| 10 | Audio Upload | `/ui/studio.html` | "Upload File" input | `studio.html` → `uploadAudioFile()` | `POST /upload` | `api_server.py` → `upload_audio()` | Local Filesystem |
| 11 | Audio Analysis | `/ui/studio.html` | "Analyze Audio" button | `studio.html` → `analyzeAudio()` | `POST /analyze` | `api_server.py` → `analyze()` | Librosa (BPM, Key, Energy) |
| 12 | Stem Separation | `/ui/studio.html` | "Separate Stems" button | `studio.html` → `separateStems()` | `POST /separate` | `api_server.py` → `separate()` | Demucs `htdemucs` AI |
| 13 | DAW Multi-Track | `/ui/studio.html` | Mute/Solo/Pan/Rev/8D | `studio.html` → `initStemEditor()` | None (Client-only) | Static `/stems/*` files | Web Audio API Engine |
| 14 | Stem Mix Render | `/ui/studio.html` | "Export Mix" button | `studio.html` → `exportStemMix()` | None (Client-only) | None | `OfflineAudioContext` |
| 15 | AI Mastering | `/ui/studio.html` | "Master Audio" button | `studio.html` → `masterAudio()` | `POST /master` | `api_server.py` → `master_endpoint()` | PyLoudNorm (-14 LUFS) |
| 16 | MIDI Extraction | `/ui/studio.html` | "Export MIDI" button | `studio.html` → `exportMidi()` | `POST /tools/audio-to-midi` | `api_server.py` → `audio_to_midi_endpoint()` | Librosa `piptrack` + MIDI 1.0 |
| 17 | Stem Pack ZIP | `/ui/studio.html`, `/ui/repo.html` | "Download Stem Pack" | `studio.html` → `downloadStemPack()` | `GET /projects/{id}/commits/{cid}/export-stems` | `api_server.py` → `export_stems_zip()` | Python `zipfile` |
| 18 | Lyrics Generator | `/ui/studio.html` | "Generate Lyrics" button | `studio.html` → `generateLyrics()` | `POST /tools/generate-lyrics` | `api_server.py` → `generate_lyrics_endpoint()` | Lyrics Rule Engine |
| 19 | Teleprompter | `/ui/studio.html` | "Start Teleprompter" | `studio.html` → `toggleTeleprompter()` | None (Client-only) | None | DOM Interval Timer |
| 20 | Commit Beat | `/ui/studio.html` | "Save as Commit" button | `studio.html` → `saveToProject()` | `POST /projects/{id}/commit` | `api_server.py` → `create_commit()` | SQLite `Commit`, Librosa |
| 21 | Quick Save Library | `/ui/studio.html` | "Quick Save" button | `studio.html` → `saveToLibrary()` | `POST /library/save` | `api_server.py` → `save_to_library()` | SQLite `Commit`, `Repository` |
| 22 | Audio Visualizer | `/ui/studio.html` | Audio Playback | `visualizer.js` → `MusicVisualizer` | None (Client-only) | None | Web Audio `AnalyserNode` |
| 23 | User Library | `/ui/library.html` | Page Load | `library.html` → `loadLibrary()` | `GET /library` | `api_server.py` → `get_library()` | SQLite `Commit` |
| 24 | Add to Project | `/ui/library.html` | "Add to Project" button | `library.html` → `doAddToProject()` | `POST /projects/{id}/commit` | `api_server.py` → `create_commit()` | SQLite `Commit` |
| 25 | List My Projects | `/ui/projects.html`, `/ui/studio.html` | Page Load / Dropdowns | `projects.html` → `loadRepos()` | `GET /projects/mine` | `api_server.py` → `my_projects()` | SQLite `Repository` |
| 26 | Search Projects | `/ui/explore.html`, `/ui/index.html` | Search input & filter chips | `explore.html` → `loadExplore()` | `GET /projects/search` | `api_server.py` → `search_projects()` | SQLite `Repository`, `Commit` |
| 27 | Create Project | `/ui/projects.html`, `/ui/repo.html` | "+ New Project" button | `projects.html` → `createProject()` | `POST /projects` | `api_server.py` → `create_project()` | SQLite `Repository` |
| 28 | Get Project Detail | `/ui/repo.html` | Page Load | `repo.html` → `loadRepo()` | `GET /projects/{id}` | `api_server.py` → `get_project()` | SQLite `Repository`, `Commit` |
| 29 | Edit Project | `/ui/projects.html` | "Save Changes" in modal | `projects.html` → `saveEdit()` | `PATCH /projects/{id}` | `api_server.py` → `patch_project()` | SQLite `Repository` |
| 30 | Fork Project | `/ui/explore.html`, `/ui/repo.html` | "Fork" button | `explore.html` → `forkRepo()` | `POST /projects/{id}/fork` | `api_server.py` → `fork_project()` | SQLite `Repository` |
| 31 | Star / Unstar | `/ui/explore.html`, `/ui/repo.html` | "Star" button | `explore.html` → `toggleStar()` | `POST` / `DELETE /projects/{id}/star` | `api_server.py` → `star_project()`, `unstar_project()` | SQLite `Star`, `Repository` |
| 32 | Track Play Count | `/ui/repo.html` | Audio Playback | `repo.html` → `bumpPlay()` | `POST /projects/{id}/play` | `api_server.py` → `play_project()` | SQLite `Repository` |
| 33 | Visual Commit Tree | `/ui/project_tree.html` | Page Load | `project_tree.html` → `loadTree()` | `GET /projects/{id}/tree` | `api_server.py` → `commit_tree()` | SQLite `Commit` DAG |
| 34 | List Branches | `/ui/project_tree.html` | "Branches" button | `project_tree.html` → `loadBranches()` | `GET /projects/{id}/branches` | `api_server.py` → `list_branches()` | SQLite `Commit` |
| 35 | Audio Diff Compare | `/ui/repo.html`, `/ui/project_tree.html` | "Run Audio Diff" button | `repo.html` → `runAudioDiff()` | `GET /projects/{id}/diff/compare` | `api_server.py` → `compare_commits_full()` | Librosa CENS Similarity |
| 36 | Commit Comments | `/ui/repo.html` | "Post Comment" / "Delete" | `repo.html` → `submitComment()` | `GET`/`POST`/`DELETE .../comments` | `api_server.py` → `list_comments()`, `add_comment()`, `delete_comment()` | SQLite `Comment` |
| 37 | List Users / Search | `/ui/community.html` | Search input & Page Load | `community.html` → `runSearch()` | `GET /users` | `api_server.py` → `list_users()` | SQLite `User` |
| 38 | Follow / Unfollow | `/ui/community.html` | "Follow" button | `community.html` → `toggleFollow()` | `POST` / `DELETE /users/{u}/follow` | `api_server.py` → `follow_user()`, `unfollow_user()` | SQLite `Follow` |
