# Mistakes & Fixes Log

## M1 — Frontend served from `file://` → fetch blocked
**Symptom:** `Failed to fetch` on all API calls  
**Cause:** Browser restricts cross-origin fetch from `file://` to `http://localhost`  
**Fix:** Mounted the HTML folder via FastAPI `StaticFiles` at `/ui/`. Access via `http://localhost:8000/ui/index.html`

---

## M2 — Missing DB column `library_repo_id`
**Symptom:** Register returns 500 "Internal Server Error"  
**Cause:** `library_repo_id` column was added to the `User` ORM model *after* the database was first created, so the SQLite `users` table was missing it  
**Fix:** `ALTER TABLE users ADD COLUMN library_repo_id TEXT`

---

## M3 — Wrong token field name → auth guard kicks every page
**Symptom:** After login/register, clicking any nav link redirects back to index  
**Cause:** `doSignIn` used `d.access_token` and `d.username` but API returns `d.token` and `d.user.username`. `localStorage.setItem('bf_token', undefined)` stored the string `"undefined"` → JWT decode fails → auth guard fires  
**Fix:** Changed to `setAuth(d.token, d.user)` in both `doSignIn` and `doRegister`

---

## M4 — doRegister did a redundant second login call
**Symptom:** Extra latency on register; used `ld.access_token` (same wrong field)  
**Cause:** Manual second `/auth/login` call after `/auth/register`, even though register already returns `{token, user}`  
**Fix:** Removed second login call; use `d.token` / `d.user` directly from register response

---

## M5 — Dashboard nav missing username display
**Symptom:** Username never appears in the top-right nav on the dashboard page  
**Cause:** `dashboard.html` nav had no `id="nav-username"` span; `loadDashboard()` never set it  
**Fix:** Added `<span id="nav-username">` to dashboard nav and set it in `loadDashboard()` after `/auth/me` resolves

---

## M6 — Explore page crashes: `Cannot read properties of undefined (reading '0')`
**Symptom:** Explore page shows error instead of repo grid  
**Cause:** Code assumed `repo.owner` was an object `{username: "..."}` but `/projects/search` returns `owner` as a plain string (e.g. `"demo"`)  
**Fix:** `const ownerName = typeof repo.owner==='string' ? repo.owner : (repo.owner?.username||'unknown')`

---

## M7 � Audio Tools unusable unless a beat was generated in the same session
**Symptom:** Clicking "Analyze BPM / Key", "Separate Stems", or "AI Master" always shows "No audio loaded yet." unless a beat was generated in the current page session
**Cause:** All three tool functions guard on `_currentFilename`, which was only set by `addBeatCard()` � triggered only after in-session generation. No way to load an existing or external file.
**Fix (backend):** Added `POST /upload` endpoint � accepts multipart/form-data audio file, saves to OUTPUT_DIR with UUID name, returns {filename, audio_url}.
**Fix (frontend � studio.html):**
- Added "Audio Source" section above tool buttons with: session-beat dropdown (auto-populated on generation) + file upload button + loaded-file label
- `addBeatCard()` now also appends to the dropdown and updates the label
- New JS: `pickSessionBeat(filename)` and `uploadAudioFile(input)`
**Libraries:** librosa, demucs, soundfile already installed; pyloudnorm 0.2.0 newly installed
