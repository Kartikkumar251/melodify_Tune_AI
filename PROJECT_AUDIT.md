# PROJECT AUDIT: Melodify Tune AI / BeatFlow

> Complete audit and mapping of existing features across Frontend, API, Backend, Database, and ML/AI subsystems.

---

## Architecture Overview & Route Map

```
FRONTEND PAGES (frontend/)
├── index.html          → Landing Page, Hero Generator, Quick Auth
├── studio.html         → Full AI Beat Studio & 4-Track DAW Workspace
├── dashboard.html      → Creator Hub, Quick Generator, Recent Beats, Stats
├── explore.html        → Public Feed, Trending Beats, Tag & Mood Filters
├── projects.html       → Repository Directory, Search, Project Creation
├── repo.html           → Version Control View, Commit History, Stem Packs, Comments
├── project_tree.html   → Interactive Git Audio Tree Visualization
├── library.html        → Personal Saved Beats, Stems, Offline Audio Stash
├── community.html      → Creator Directory, Leaderboards, Follow System
└── settings.html       → User Profile, Bio, Avatar, Security Settings

SHARED MODULES (frontend/)
├── nav.js              → Dynamic Navbar, Auth State, Quick Player, Global Shortcuts
└── visualizer.js       → Real-time 60fps Web Audio Spectrum & Oscillogram Engine

BACKEND SUBSYSTEMS (backend/)
├── api_server.py       → FastAPI REST API, SSE Progress Streaming, Audio Dispatcher
├── audio_processing.py → Librosa Harmonic Analysis, Demucs Stem Separation, Mastering, MIDI
├── auth.py             → JWT Bearer Security, Argon2 Password Hashing, User Dependency
├── database.py         → SQLAlchemy SQLite Engine (`beatflow.db`) & Session Manager
├── models.py           → ORM Schema (User, Repository, Commit, Stem, Star, Comment, Follow)
├── celery_worker.py    → Celery Background Task Processing (Redis Optional)
└── run_demucs.py       → Demucs Process Isolation Runner for Windows

MACHINE LEARNING & AUDIO SYNTHESIS (ml/)
├── beat_generator.py   → Audiocraft MusicGen CLI Interface & Pipeline Loader
└── setup_musicgen.bat  → Audiocraft / PyTorch / CUDA automated environment setup
```

---

## 1. Full-Stack Connected Features

---

### Feature: User Registration
- **Frontend Page**: `/` (`index.html`) & `/settings.html`
- **Button/UI**: "Sign Up" / "Create Account" modal button (`submitAuth()`)
- **Frontend File**: `frontend/index.html` & `frontend/settings.html`
- **API**: `POST /auth/register`
- **Backend**: `backend/api_server.py` → `register()`
- **Database/Service**: SQLite `User` table, `backend/auth.py` (Argon2 password hashing), auto-creates default private "My Beats" `Repository`
- **Flow**: User inputs username, email, and password → frontend sends JSON body → backend hashes password, saves new `User`, provisions default repository → returns JWT access token & user object → frontend saves token to `localStorage` and updates UI.

---

### Feature: User Login
- **Frontend Page**: `/` (`index.html`) & `/settings.html`
- **Button/UI**: "Sign In" button in Auth Modal (`submitAuth()`)
- **Frontend File**: `frontend/index.html` & `frontend/settings.html`
- **API**: `POST /auth/login`
- **Backend**: `backend/api_server.py` → `login()`
- **Database/Service**: SQLite `User` table, `backend/auth.py` (`verify_password`, `create_access_token`)
- **Flow**: User submits credentials → backend verifies hash → creates and returns signed JWT token → frontend stores token in `localStorage.token` and reloads auth state across pages.

---

### Feature: Current User Profile Fetch
- **Frontend Page**: All pages (`index.html`, `studio.html`, `dashboard.html`, `projects.html`, `repo.html`, `community.html`, `settings.html`, `library.html`)
- **Button/UI**: Automatic on page load (`loadCurrentUser()`, `fetchMe()`)
- **Frontend File**: `frontend/nav.js` & `frontend/settings.html`
- **API**: `GET /auth/me`
- **Backend**: `backend/api_server.py` → `me()`
- **Database/Service**: SQLite `User` table via `get_current_user` dependency
- **Flow**: Page loads → `nav.js` sends `Authorization: Bearer <token>` header → backend validates JWT token → returns user profile data → navbar renders user avatar, username, and sign-out buttons.

---

### Feature: Update User Profile & Bio
- **Frontend Page**: `/settings.html`
- **Button/UI**: "Save Profile" / "Update Info" button (`saveProfile()`)
- **Frontend File**: `frontend/settings.html`
- **API**: `PATCH /auth/me`
- **Backend**: `backend/api_server.py` → `update_me()`
- **Database/Service**: SQLite `User` table (`bio`, `avatar_url`)
- **Flow**: User updates avatar URL or bio text and clicks Save → frontend sends PATCH request → backend updates `User` record in database → returns updated user object → frontend displays success toast.

---

### Feature: Real-Time Tracked Beat Generation (with SSE Progress)
- **Frontend Page**: `/studio.html` & `/dashboard.html`
- **Button/UI**: "⚡ Generate Beat" button (`generateBeat()`)
- **Frontend File**: `frontend/studio.html` & `frontend/dashboard.html`
- **API**: `POST /generate/tracked` followed by `GET /sse/progress/{task_id}`
- **Backend**: `backend/api_server.py` → `generate_tracked()`, `sse_progress()`, `_generate()`, `_synthesize_algorithmic_beat()`
- **Database/Service**: AI Audiocraft MusicGen (`facebook/musicgen-small`) with fallback to algorithmic multi-layer audio synthesizer + `soundfile` WAV writer
- **Flow**: User configures prompt, BPM, duration, and key → clicks Generate → frontend triggers `POST /generate/tracked` receiving `task_id` → frontend opens `EventSource` on `/sse/progress/{task_id}` for live progress bar (0–100%) → backend background thread synthesizes audio → SSE emits `complete` with audio URL and metadata → frontend loads audio into waveform visualizer and multitrack player.

---

### Feature: Smart Prompt Suggestions / Quick Preset Chips
- **Frontend Page**: `/studio.html`
- **Button/UI**: Genre & Mood Chips (`⚡ Midnight Neon`, `☕ Rain Lo-Fi`, `🔥 Atlanta Trap`, `🇬🇧 UK Drill`, `💥 Drift Phonk`, `🤖 Cyberpunk`, `🌴 Afrobeats`, `🌌 Space Ambient`, `💜 Late Night R&B`)
- **Frontend File**: `frontend/studio.html` (`applyQuickKeyword()`, `fetchSmartSuggestion()`)
- **API**: `POST /tools/smart-suggest`
- **Backend**: `backend/api_server.py` → `smart_suggest_endpoint()`
- **Database/Service**: `PROMPT_KNOWLEDGE_BASE` dictionary in `backend/api_server.py`
- **Flow**: User clicks a preset pill → frontend calls smart suggest endpoint with keyword → backend looks up optimal prompt tags, recommended BPM, and musical key → returns suggestion JSON → frontend auto-populates prompt input, BPM slider, and key selector.

---

### Feature: Prompt Autocomplete Dropdown
- **Frontend Page**: `/studio.html`
- **Button/UI**: Typing into prompt textarea (`onPromptInput()`)
- **Frontend File**: `frontend/studio.html`
- **API**: `GET /tools/prompt-autocomplete?q={query}`
- **Backend**: `backend/api_server.py` → `prompt_autocomplete_endpoint()`
- **Database/Service**: In-memory keyword lookup from `PROMPT_KNOWLEDGE_BASE`
- **Flow**: User types in studio prompt input → debounce sends query to backend → backend filters relevant music tags and style prompts → returns matched list → frontend renders dropdown suggestions.

---

### Feature: AI Prompt Enhancer
- **Frontend Page**: `/studio.html`
- **Button/UI**: "✨ Enhance Prompt" button (`enhancePrompt()`)
- **Frontend File**: `frontend/studio.html`
- **API**: `POST /tools/enhance-prompt`
- **Backend**: `backend/api_server.py` → `enhance_prompt_endpoint()`
- **Database/Service**: Rule-based audio acoustic synthesis engine & studio sound tagging
- **Flow**: User enters raw idea (e.g., "dark trap beat") → clicks Enhance → backend expands text with studio acoustic descriptors (stereo width, mastering EQ, saturation, sub-bass harmonics) → frontend replaces prompt text with enhanced version.

---

### Feature: Audio Analysis (BPM, Key, Energy, LUFS Loudness)
- **Frontend Page**: `/studio.html`
- **Button/UI**: "🔍 Analyze Audio" button (`analyzeAudio()`)
- **Frontend File**: `frontend/studio.html`
- **API**: `POST /analyze`
- **Backend**: `backend/api_server.py` → `analyze()`, `backend/audio_processing.py` → `analyze_audio()`
- **Database/Service**: `librosa` (BPM tempo beat tracking, Chroma feature key extraction, RMS energy), `pyloudnorm` (Integrated LUFS)
- **Flow**: User provides audio URL or uploads audio file → frontend calls `/analyze` → backend loads audio via Librosa → computes exact tempo, musical key, loudness, and energy curve → returns JSON metrics → frontend displays analysis dashboard cards and waveform peaks.

---

### Feature: 4-Stem Audio Separation
- **Frontend Page**: `/studio.html`
- **Button/UI**: "✂ Separate Stems" button (`separateStems()`)
- **Frontend File**: `frontend/studio.html`
- **API**: `POST /separate`
- **Backend**: `backend/api_server.py` → `separate()`, `backend/audio_processing.py` → `separate_stems()`, `backend/run_demucs.py`
- **Database/Service**: Meta Demucs (`htdemucs`) deep learning model with spectral fallback filter + SQLite `Stem` table
- **Flow**: User clicks Separate Stems on current beat → backend runs Demucs model separating audio into 4 isolated WAV files (Drums, Bass, Other/Melody, Vocals) → saves stems to disk and database → returns stem URLs → frontend loads each track into the FL Studio-style multitrack mixer.

---

### Feature: AI Audio Mastering
- **Frontend Page**: `/studio.html`
- **Button/UI**: "🎚 AI Master" button (`masterAudio()`)
- **Frontend File**: `frontend/studio.html`
- **API**: `POST /master`
- **Backend**: `backend/api_server.py` → `master_endpoint()`, `backend/audio_processing.py` → `master_audio()`
- **Database/Service**: `pyloudnorm` (EBU R128 standard -14 LUFS normalization, 3-band parametric EQ, brickwall peak limiter)
- **Flow**: User clicks Master → backend applies multi-band acoustic equalization and true-peak limiting → writes mastered WAV to `mastered_outputs/` → returns mastered audio URL and LUFS stats → frontend updates main player with the polished track.

---

### Feature: Hum-to-Beat Melody Conditioning
- **Frontend Page**: `/studio.html`
- **Button/UI**: "🎤 Hum-to-Beat" / "Convert Hum" modal button (`submitHum()`)
- **Frontend File**: `frontend/studio.html`
- **API**: `POST /hum`
- **Backend**: `backend/api_server.py` → `hum_to_beat_endpoint()`, `backend/audio_processing.py` → `hum_to_beat()`
- **Database/Service**: Audiocraft MusicGen Melody conditioning / Librosa pitch detection
- **Flow**: User uploads or records voice humming → backend extracts pitch contour and feeds it as acoustic guidance into the generation pipeline → returns synthesized full-production beat matching the hummed melody.

---

### Feature: Audio-to-MIDI (.mid) Export
- **Frontend Page**: `/studio.html`
- **Button/UI**: "🎹 Export MIDI" button (`exportMidi()`)
- **Frontend File**: `frontend/studio.html`
- **API**: `POST /tools/audio-to-midi`
- **Backend**: `backend/api_server.py` → `audio_to_midi_endpoint()`, `backend/audio_processing.py` → `audio_to_midi()`
- **Database/Service**: `librosa` (onset detection, pitch detection), `mido` (Standard MIDI File Formatter)
- **Flow**: User clicks Export MIDI on active beat or stem → backend extracts notes and velocities → generates a standard `.mid` file → returns download URL → browser triggers file download for use in external DAWs (Ableton, FL Studio, Logic).

---

### Feature: Audio Continuation / Loop Extension
- **Frontend Page**: `/studio.html`
- **Button/UI**: "⏭ Extend Beat" button (`continueBeat()`)
- **Frontend File**: `frontend/studio.html`
- **API**: `POST /continue`
- **Backend**: `backend/api_server.py` → `continue_beat_endpoint()`, `backend/audio_processing.py` → `continue_audio()`
- **Database/Service**: Audiocraft audio prompt continuation / crossfade acoustic loop extender
- **Flow**: User selects duration to extend → backend takes tail of current track and generates seamless musical continuation → concatenates with equal-power crossfade → returns extended audio URL.

---

### Feature: Audio File Upload
- **Frontend Page**: `/studio.html`
- **Button/UI**: "📁 Upload Audio" file input (`uploadAudioFile()`)
- **Frontend File**: `frontend/studio.html`
- **API**: `POST /upload`
- **Backend**: `backend/api_server.py` → `upload_audio()`
- **Database/Service**: File system storage (`upload_tmp/` and `beat_outputs/`)
- **Flow**: User selects local WAV/MP3/FLAC/OGG file → frontend uploads multipart form data → backend validates audio format and saves file → returns accessible URL and duration → studio loads track into workspace.

---

### Feature: AI Lyrics & Synchronized Teleprompter
- **Frontend Page**: `/studio.html`
- **Button/UI**: "📝 Generate Lyrics" button (`generateLyrics()`) & "▶ Start Live Teleprompter" (`toggleTeleprompter()`)
- **Frontend File**: `frontend/studio.html`
- **API**: `POST /tools/generate-lyrics`
- **Backend**: `backend/api_server.py` → `generate_lyrics_endpoint()`
- **Database/Service**: Algorithmic song structure lyric engine (Verses, Chorus, Bridge, Outro) with BPM-timed timestamp cues
- **Flow**: User inputs topic/mood and clicks Generate Lyrics → backend builds rhythm-synced lyrics with line-by-line timestamps → frontend renders lyrics teleprompter widget that auto-scrolls and highlights lines in real time as audio plays.

---

### Feature: Save Beat to Project (Git Commit for Audio)
- **Frontend Page**: `/studio.html`
- **Button/UI**: "💾 Commit Beat to Project" modal button (`saveToProject()`)
- **Frontend File**: `frontend/studio.html`
- **API**: `POST /projects/{repo_id}/commit`
- **Backend**: `backend/api_server.py` → `create_commit()`
- **Database/Service**: SQLite `Commit` table (stores SHA-like hash, `audio_url`, `prompt`, `bpm`, `key`, `energy`, `parent_id`)
- **Flow**: User selects target project and enters commit message → frontend sends POST request → backend generates unique commit hash, links parent commit, records audio metadata in DB → returns new commit object → updates project version history.

---

### Feature: Save Beat to Personal Library
- **Frontend Page**: `/studio.html`
- **Button/UI**: "❤️ Save to Library" button (`saveToLibrary()`)
- **Frontend File**: `frontend/studio.html`
- **API**: `POST /library/save`
- **Backend**: `backend/api_server.py` → `save_to_library()`
- **Database/Service**: SQLite `Commit` table linked to user's private `library_repo_id`
- **Flow**: User clicks Save to Library → backend commits the beat into user's private system library repository → returns confirmation toast on frontend.

---

### Feature: Fetch Personal Library
- **Frontend Page**: `/library.html`
- **Button/UI**: Automatic on page load (`loadLibrary()`)
- **Frontend File**: `frontend/library.html`
- **API**: `GET /library`
- **Backend**: `backend/api_server.py` → `get_library()`
- **Database/Service**: SQLite `Commit` and `Repository` tables
- **Flow**: Library page loads → frontend requests user's library → backend queries all commits under user's private library repository → returns array of saved beats with stems → frontend renders beat cards with playable waveforms.

---

### Feature: List Public Projects
- **Frontend Page**: `/projects.html` & `/explore.html`
- **Button/UI**: Automatic on page load (`loadProjects()`)
- **Frontend File**: `frontend/projects.html` & `frontend/explore.html`
- **API**: `GET /projects`
- **Backend**: `backend/api_server.py` → `list_projects()`
- **Database/Service**: SQLite `Repository` table filtered by `is_public == True`
- **Flow**: Page loads → frontend fetches public projects → backend queries repositories with commit counts, owner details, and star counts → returns project list → frontend displays project cards.

---

### Feature: Search and Filter Projects
- **Frontend Page**: `/explore.html` & `/projects.html`
- **Button/UI**: Search input, Mood buttons, BPM sliders, Sort dropdown (`onSearchInput()`)
- **Frontend File**: `frontend/explore.html` & `frontend/projects.html`
- **API**: `GET /projects/search?q={q}&mood={mood}&bpm_min={min}&bpm_max={max}&sort={sort}`
- **Backend**: `backend/api_server.py` → `search_projects()`
- **Database/Service**: SQLite `Repository` and `Commit` tables with multi-clause filtering
- **Flow**: User adjusts search term, mood filter, or BPM range → frontend triggers search request → backend applies SQLAlchemy filters and sorting (trending, stars, newest) → returns matched projects → frontend updates project grid.

---

### Feature: Create New Project Repository
- **Frontend Page**: `/dashboard.html` & `/projects.html`
- **Button/UI**: "➕ New Project" modal submit button (`createProject()`)
- **Frontend File**: `frontend/dashboard.html` & `frontend/projects.html`
- **API**: `POST /projects`
- **Backend**: `backend/api_server.py` → `create_project()`
- **Database/Service**: SQLite `Repository` table (`name`, `description`, `is_public`, `owner_id`)
- **Flow**: User fills project name, description, and visibility toggle → submits modal → backend inserts new `Repository` → returns project record → frontend redirects to `/repo.html?id={id}`.

---

### Feature: Project Repository Detail & Revision History
- **Frontend Page**: `/repo.html?id={id}`
- **Button/UI**: Automatic on page load (`loadRepo()`)
- **Frontend File**: `frontend/repo.html`
- **API**: `GET /projects/{id}`
- **Backend**: `backend/api_server.py` → `get_project()`
- **Database/Service**: SQLite `Repository`, `Commit`, and `Stem` tables
- **Flow**: Page loads with project ID → backend retrieves repository details along with chronological commit history and stem associations → returns complete project hierarchy → frontend renders commit timeline, stem player, and project stats.

---

### Feature: Star / Unstar Project
- **Frontend Page**: `/repo.html`, `/projects.html`, `/explore.html`
- **Button/UI**: "⭐ Star" button (`toggleStar()`)
- **Frontend File**: `frontend/repo.html`, `frontend/projects.html`, `frontend/explore.html`
- **API**: `POST /projects/{id}/star` & `DELETE /projects/{id}/star`
- **Backend**: `backend/api_server.py` → `star_project()`, `unstar_project()`
- **Database/Service**: SQLite `Star` table & `Repository.star_count`
- **Flow**: User clicks Star button → frontend calls star/unstar endpoint → backend inserts/deletes `Star` row and increments/decrements `star_count` → returns updated count → frontend toggles star active state.

---

### Feature: Fork Project Repository
- **Frontend Page**: `/repo.html`
- **Button/UI**: "🍴 Fork Project" button (`forkRepo()`)
- **Frontend File**: `frontend/repo.html`
- **API**: `POST /projects/{id}/fork`
- **Backend**: `backend/api_server.py` → `fork_project()`
- **Database/Service**: SQLite `Repository` table (`parent_repo_id`) & `Commit` table
- **Flow**: User clicks Fork on another creator's project → backend clones repository metadata under current user, references `parent_repo_id`, and copies latest commits → returns new project ID → frontend navigates to the forked repository.

---

### Feature: Visual Git Audio Tree / Graph
- **Frontend Page**: `/project_tree.html?id={id}` & `/repo.html`
- **Button/UI**: "🌳 Git Tree View" link/tab (`loadTree()`)
- **Frontend File**: `frontend/project_tree.html`
- **API**: `GET /projects/{id}/tree`
- **Backend**: `backend/api_server.py` → `get_project_tree()`
- **Database/Service**: SQLite `Commit` table recursive graph traversal
- **Flow**: User opens tree view → backend traverses commit tree via `parent_id` relations → returns structured graph nodes and branch links → frontend Canvas / SVG renders interactive branching git tree with audio preview nodes.

---

### Feature: Audio Commit Comments System
- **Frontend Page**: `/repo.html`
- **Button/UI**: "Post Comment" button (`addComment()`) & Delete button (`deleteComment()`)
- **Frontend File**: `frontend/repo.html`
- **API**: `GET /projects/{repo_id}/commits/{commit_id}/comments`, `POST /projects/{repo_id}/commits/{commit_id}/comments`, `DELETE /comments/{comment_id}`
- **Backend**: `backend/api_server.py` → `list_comments()`, `add_comment()`, `delete_comment()`
- **Database/Service**: SQLite `Comment` table (`user_id`, `commit_id`, `text`, `created_at`)
- **Flow**: User types feedback on a specific audio commit and submits → backend validates auth, inserts `Comment` → returns comment object with user metadata → frontend appends new comment into conversation thread.

---

### Feature: Creator Directory & Follow System
- **Frontend Page**: `/community.html`
- **Button/UI**: "Follow" / "Unfollow" button (`toggleFollow()`) & Page load (`loadCreators()`)
- **Frontend File**: `frontend/community.html`
- **API**: `GET /users`, `GET /users/{username}`, `POST /users/{username}/follow`, `DELETE /users/{username}/follow`
- **Backend**: `backend/api_server.py` → `list_users()`, `get_user()`, `follow_user()`, `unfollow_user()`
- **Database/Service**: SQLite `User` and `Follow` tables
- **Flow**: Community page lists creators with stats → user clicks Follow → backend adds record to `Follow` table → returns updated follower count → frontend toggles button to "Following".

---

### Feature: Audio Revision Diff (Harmonic & Stem Comparison)
- **Frontend Page**: `/repo.html`
- **Button/UI**: "Compare Diff" button between two commit versions (`compareDiff()`)
- **Frontend File**: `frontend/repo.html`
- **API**: `GET /projects/{repo_id}/commits/{hash_a}/diff/{hash_b}`
- **Backend**: `backend/api_server.py` → `diff_commits()`
- **Database/Service**: SQLite `Commit` and `Stem` tables
- **Flow**: User selects two commit revisions to compare → backend computes BPM shift, key transposition, energy delta, and lists added/removed stems → returns structured diff JSON → frontend displays Git-style green/red audio change summary.

---

### Feature: Download Full Stem Pack (.ZIP)
- **Frontend Page**: `/studio.html` & `/repo.html`
- **Button/UI**: "📦 Download Stem Pack (.zip)" button (`downloadStemPack()`)
- **Frontend File**: `frontend/studio.html` & `frontend/repo.html`
- **API**: `GET /projects/{repo_id}/commits/{commit_id}/export-stems`
- **Backend**: `backend/api_server.py` → `export_stems_zip()`
- **Database/Service**: Python `zipfile` memory buffer streaming stem WAV files from disk
- **Flow**: User clicks Download Stem Pack → backend bundles Drums, Bass, Other, and Vocals WAV files into a zip archive with track metadata README → streams zip response → browser prompts download.

---

### Feature: Track Play Count Tracker
- **Frontend Page**: `/repo.html`, `/projects.html`, `/explore.html`
- **Button/UI**: Audio Play Trigger (`recordPlay()`)
- **Frontend File**: `frontend/repo.html`, `frontend/projects.html`, `frontend/explore.html`
- **API**: `POST /projects/{id}/play`
- **Backend**: `backend/api_server.py` → `record_play()`
- **Database/Service**: SQLite `Repository.play_count` field
- **Flow**: User plays a project's audio track → frontend fires fire-and-forget POST request → backend increments `play_count` in database → returns updated play tally.

---

## 2. Frontend-Only Features

---

### Feature: Real-Time Audio Visualizer Canvas Engine
- **Frontend Page**: `/studio.html`, `/index.html`, `/repo.html`, `/library.html`
- **Button/UI**: Automatic when any audio plays
- **Frontend File**: `frontend/visualizer.js`
- **API**: None (Client-side Web Audio API)
- **Backend**: N/A
- **Database/Service**: Browser `AudioContext`, `AnalyserNode`, Canvas 2D Context
- **Flow**: Audio playback connects to Web Audio Analyser node → canvas renders 60fps frequency bars, circular spectrum, and oscillogram wave animations with neon gradient shading.

---

### Feature: FL Studio-Style Multitrack DAW Mixer Controls
- **Frontend Page**: `/studio.html`
- **Button/UI**: Track Volume sliders, Pan knobs, Mute (`M`), Solo (`S`), Playhead scrubber, Master Volume slider
- **Frontend File**: `frontend/studio.html`
- **API**: None (Client-side Web Audio API)
- **Backend**: N/A
- **Database/Service**: Browser `AudioContext`, `GainNode`, `StereoPannerNode`
- **Flow**: User moves track sliders or clicks Mute/Solo → frontend adjusts Web Audio gain and panning nodes for individual stems in real-time without reloading audio.

---

### Feature: Offline Mixdown & WAV Audio Export
- **Frontend Page**: `/studio.html`
- **Button/UI**: "⬇ Export Mix (.wav)" button (`exportMixdown()`)
- **Frontend File**: `frontend/studio.html` (`audioBufferToWav()`)
- **API**: None (Client-side Web Audio API)
- **Backend**: N/A
- **Database/Service**: Browser `OfflineAudioContext`, RIFF PCM WAV binary encoder
- **Flow**: User clicks Export Mix → browser renders all active stems through `OfflineAudioContext` with their respective volume and pan settings → encodes PCM float buffer into 16-bit stereo WAV blob → triggers instant browser file download.

---

### Feature: Musical Scales & BPM Theory Reference Tooltips
- **Frontend Page**: `/studio.html`
- **Button/UI**: Changing Key / Scale / BPM selectors (`onKeyScaleChange()`, `onBpmHint()`)
- **Frontend File**: `frontend/studio.html`
- **API**: None (Client-side music theory dictionary)
- **Backend**: N/A
- **Database/Service**: In-memory music theory scale mappings
- **Flow**: User selects key (e.g., "F# Minor") or adjusts BPM slider → frontend calculates compatible modes, mood descriptions (e.g., "Dark, Melancholic, Drill"), and tempo category (e.g., "Midtempo Trap / Hip-Hop") and displays tooltip hints.

---

## 3. Backend-Only & Disconnected / Standalone Features

---

### Feature: Synchronous Direct Beat Generation
- **Frontend Page**: None (Replaced by tracked SSE endpoint `/generate/tracked`)
- **Button/UI**: N/A
- **Frontend File**: N/A
- **API**: `POST /generate`
- **Backend**: `backend/api_server.py` → `generate()`
- **Database/Service**: Audiocraft MusicGen / Algorithmic synthesizer
- **Flow**: Synchronous blocking generation endpoint. Returns generated audio URL directly once complete. Maintained for headless API client integrations.

---

### Feature: Celery Asynchronous Beat Generation & Task Status
- **Frontend Page**: None (Reserved for distributed background workers)
- **Button/UI**: N/A
- **Frontend File**: N/A
- **API**: `POST /generate/async` & `GET /tasks/{task_id}`
- **Backend**: `backend/api_server.py` → `generate_async()`, `get_task_status()`, `backend/celery_worker.py`
- **Database/Service**: Celery worker queue + Redis broker
- **Flow**: Dispatches generation payload to Celery queue → worker executes task asynchronously in background → client polls `/tasks/{task_id}` for completion.

---

### Feature: List Project Root Branches
- **Frontend Page**: None (Project detail currently uses unified `/tree` endpoint)
- **Button/UI**: N/A
- **Frontend File**: N/A
- **API**: `GET /projects/{repo_id}/branches`
- **Backend**: `backend/api_server.py` → `list_branches()`
- **Database/Service**: SQLite `Commit` table (`parent_id.is_(None)`)
- **Flow**: Returns list of all root-level commits that initiated independent timeline branches for a given project repository.

---

### Feature: Patch Project Repository Metadata
- **Frontend Page**: None (Settings currently updates user metadata; project edit form not yet wired)
- **Button/UI**: N/A
- **Frontend File**: N/A
- **API**: `PATCH /projects/{repo_id}`
- **Backend**: `backend/api_server.py` → `patch_project()`
- **Database/Service**: SQLite `Repository` table (`name`, `description`, `is_public`)
- **Flow**: Authenticated repository owner submits updated name, description, or visibility flag → backend updates database record and returns updated repository object.

---

### Feature: Standalone MusicGen Terminal CLI Generator
- **Frontend Page**: None (CLI utility)
- **Button/UI**: N/A
- **Frontend File**: N/A
- **API**: None
- **Backend / Script**: `ml/beat_generator.py`
- **Database/Service**: Audiocraft MusicGen model directly via terminal
- **Flow**: Developer runs `python ml/beat_generator.py` in terminal → interactively prompts for prompt text, BPM, and duration → runs MusicGen inference locally and saves output WAV file directly to disk.

---

## 4. Summary Matrix: Frontend Page to Backend Mapping

| Frontend Page / Component | Key UI Actions / Buttons | API Endpoints Called | Backend Handler Functions | Database Models & Services |
| :--- | :--- | :--- | :--- | :--- |
| **`index.html`** (Landing / Quick Studio) | Sign In / Sign Up, Quick Generate | `POST /auth/login`<br>`POST /auth/register`<br>`POST /generate/tracked` | `login()`<br>`register()`<br>`generate_tracked()` | `User`, `Repository`<br>MusicGen / Synth |
| **`studio.html`** (Studio & 4-Track DAW) | Generate Beat, Smart Presets, Stem Separation, AI Master, Export MIDI, Hum-to-Beat, Continue, Commit | `POST /generate/tracked`<br>`GET /sse/progress/{task_id}`<br>`POST /tools/smart-suggest`<br>`POST /tools/enhance-prompt`<br>`POST /analyze`<br>`POST /separate`<br>`POST /master`<br>`POST /hum`<br>`POST /tools/audio-to-midi`<br>`POST /continue`<br>`POST /projects/{id}/commit`<br>`POST /library/save` | `generate_tracked()`<br>`sse_progress()`<br>`smart_suggest_endpoint()`<br>`enhance_prompt_endpoint()`<br>`analyze()`<br>`separate()`<br>`master_endpoint()`<br>`hum_to_beat_endpoint()`<br>`audio_to_midi_endpoint()`<br>`continue_beat_endpoint()`<br>`create_commit()`<br>`save_to_library()` | MusicGen, Demucs, Librosa, Pyloudnorm, Mido, SQLite `Commit`, `Stem` |
| **`dashboard.html`** (Creator Hub) | New Project, Quick Gen, Recent Beats | `GET /auth/me`<br>`POST /projects`<br>`POST /generate/tracked` | `me()`<br>`create_project()`<br>`generate_tracked()` | `User`, `Repository`, `Commit` |
| **`explore.html`** (Public Feed) | Search, Filter Mood/BPM, Star, Play | `GET /projects`<br>`GET /projects/search`<br>`POST /projects/{id}/star`<br>`POST /projects/{id}/play` | `list_projects()`<br>`search_projects()`<br>`star_project()`<br>`record_play()` | `Repository`, `Star`, `Commit` |
| **`projects.html`** (Repositories) | Search Projects, Create Project, Star | `GET /projects`<br>`POST /projects`<br>`POST /projects/{id}/star` | `list_projects()`<br>`create_project()`<br>`star_project()` | `Repository`, `Star`, `User` |
| **`repo.html`** (Version Control & DAW) | Branch History, Fork, Commit Diff, Export Stems ZIP, Comments | `GET /projects/{id}`<br>`POST /projects/{id}/fork`<br>`GET /projects/{id}/tree`<br>`GET /projects/.../diff/...`<br>`GET /projects/.../export-stems`<br>`POST /projects/.../comments`<br>`DELETE /comments/{id}` | `get_project()`<br>`fork_project()`<br>`get_project_tree()`<br>`diff_commits()`<br>`export_stems_zip()`<br>`add_comment()`<br>`delete_comment()` | `Repository`, `Commit`, `Stem`, `Comment` |
| **`project_tree.html`** (Git Tree Visualizer) | Interactive Commit Branch Explorer | `GET /projects/{id}/tree`<br>`GET /projects/{id}` | `get_project_tree()`<br>`get_project()` | `Commit` recursive tree |
| **`library.html`** (My Saved Beats) | Browse Library Beats, Play, Stem Packs | `GET /library`<br>`GET /projects/.../export-stems` | `get_library()`<br>`export_stems_zip()` | `Commit`, `Stem`, `Repository` |
| **`community.html`** (Creator Network) | Browse Users, Follow / Unfollow | `GET /users`<br>`GET /users/{username}`<br>`POST /users/{username}/follow`<br>`DELETE /users/{username}/follow` | `list_users()`<br>`get_user()`<br>`follow_user()`<br>`unfollow_user()` | `User`, `Follow` |
| **`settings.html`** (Account Management) | Save Bio/Avatar, Sign Out | `GET /auth/me`<br>`PATCH /auth/me` | `me()`<br>`update_me()` | `User` |
| **`nav.js`** (Global Navbar) | Auth Status, Quick Player, Logout | `GET /auth/me` | `me()` | `User` |
