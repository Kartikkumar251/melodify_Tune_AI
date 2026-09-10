# BeatFlow AI - Frontend Pages Blueprint

## Complete Frontend Architecture for Backend Integration

### **CRITICAL PAGES (Must Have)**

#### 1. **Authentication Pages**
- **Login Page** (`login.html`)
  - Email/password authentication
  - Remember me option
  - OAuth integration (Google, GitHub)
  - Connects to: `POST /auth/login` endpoint
  - Returns: JWT token stored in localStorage/cookies

- **Signup Page** (`signup.html`)
  - Create new account
  - Email verification flow
  - Password strength indicator
  - Connects to: `POST /auth/signup` endpoint
  - Triggers: Email verification task

- **Password Reset Page** (`password-reset.html`)
  - Email input form
  - Password reset token verification
  - New password form
  - Connects to: `POST /auth/forgot-password`, `POST /auth/reset-password`

- **Email Verification Page** (`verify-email.html`)
  - Verify email with token from link
  - Resend verification option
  - Connects to: `POST /auth/verify-email`

---

#### 2. **Project Editor (DAW) - MOST CRITICAL**
- **Main Editor Page** (`editor.html`)
  - **Central DAW Interface** with:
    - Waveform display (left panel) - shows audio timeline
    - Project file tree (sidebar) - loads from `GET /repositories/{repo_id}/commits`
    - Mixer controls (right panel)
    - Transport controls (Play, Pause, Record, Stop)
    - Timeline/cursor position
  
  - **Key Sections:**
    - Audio track canvas (Wavesurfer.js or similar)
    - Stem display (drums, bass, vocals, melody, other)
    - Commit/version list (loads from `GET /repositories/{repo_id}/history`)
    - Real-time stem separation visualization
  
  - **Interactions:**
    - Upload audio file → `POST /audio/upload`
    - Generate music → `POST /generate` (triggers Celery task)
    - Separate stems → `POST /separate-stems` (Demucs task)
    - Analyze audio (BPM, key) → `POST /analyze`
    - Create commit → `POST /commit`
    - Save changes → `PUT /repositories/{repo_id}`
  
  - **WebSocket Connection:**
    - Real-time task progress updates
    - Stem separation progress
    - Generation status
    - Collaborative editing updates

---

#### 3. **Project Management Pages**
- **Projects Dashboard** (`projects.html`) - ✅ Already created
  - List all user projects
  - Create new project button
  - Search/filter functionality
  - Connects to: `GET /repositories` (list all)

- **Project Details Page** (`project-detail.html`)
  - Project settings and metadata
  - Collaborators list
  - Commit history timeline
  - Project statistics (size, duration, stems count)
  - Delete/archive project options
  - Connects to: `GET /repositories/{repo_id}`, `GET /repositories/{repo_id}/collaborators`

- **Create New Project Modal** (embedded in multiple pages)
  - Project name, description, genre, BPM
  - Template selection
  - Connects to: `POST /repositories` endpoint

---

#### 4. **Music Generation Interface**
- **AI Generator Studio Page** (`generator.html`)
  - Text prompt input
  - Model selection (MusicGen variants)
  - Generation parameters:
    - Duration
    - Temperature (creativity)
    - Top-P sampling
    - Guidance scale
  - Liquid intensity slider (reference stem upload)
  - Previous generation history
  - Real-time generation progress with status updates
  
  - **Connects to:**
    - `POST /generate` (start generation task)
    - `GET /tasks/{task_id}` (check status via WebSocket)
    - `GET /artifacts/{artifact_id}` (download results)

---

#### 5. **Collaboration & Sharing**
- **Collaborators Page** (`collaborators.html`) - ✅ Already created
  - List team members with roles
  - Invite new collaborators
  - Manage permissions (Admin, Editor, Viewer)
  - Activity log
  - Connects to: `GET/POST /repositories/{repo_id}/collaborators`, `/invite`

- **Share/Invite Modal** (`share-modal.html`)
  - Generate public share links
  - Share with specific users
  - Set expiration time
  - Set edit/view permissions
  - Connects to: `POST /share/{repo_id}`, `GET /share/{share_token}`

- **Activity Feed Page** (`activity.html`)
  - User activity log
  - Commit history with diffs
  - Collaboration events
  - Comments/discussions
  - Connects to: `GET /repositories/{repo_id}/activity`, `/commits/{commit_id}/comments`

---

#### 6. **Discovery & Explore**
- **Explore Page** (`explore.html`) - ✅ Already created
  - Featured projects
  - Genre browsing
  - Trending projects
  - Search functionality
  - Fork functionality
  - Connects to: `GET /projects/explore`, `GET /projects/trending`, `POST /fork/{repo_id}`

- **Search Results Page** (`search-results.html`)
  - Filter by genre, BPM range, duration
  - Sort options (newest, trending, popular)
  - Project cards with preview
  - Connects to: `GET /search?q=query&filters=...`

- **Marketplace Page** (`marketplace.html`)
  - Preset templates
  - Sound packs for purchase
  - Featured artists
  - Sample packs for download
  - Connects to: `GET /marketplace/templates`, `/marketplace/samples`

---

#### 7. **Account & Settings**
- **Dashboard** (`dashboard.html`) - ✅ Already created
  - Quick statistics
  - Recent projects
  - Quick actions
  - Upcoming events
  - Storage usage
  - Connects to: `GET /user/stats`, `GET /repositories?limit=5`

- **Settings Page** (`settings.html`) - ✅ Already created
  - Profile editing
  - Preferences
  - Privacy & security
  - Billing & subscription
  - Two-factor authentication
  - Session management
  - Connects to: `PUT /user/profile`, `PUT /user/preferences`, `POST /auth/2fa`

- **User Profile Page** (`profile.html`)
  - Public user profile
  - User's public projects
  - Follower/following counts
  - Bio and links
  - Connects to: `GET /users/{user_id}`, `GET /users/{user_id}/projects`

---

#### 8. **Utility & Support Pages**
- **Notifications Center** (`notifications.html`)
  - All notifications history
  - Mark as read/unread
  - Filter by type
  - Connects to: `GET /notifications`, `PUT /notifications/{id}/read`

- **File Upload Modal** (`upload-modal.html`)
  - Drag-and-drop file upload
  - Progress indicator
  - Supported formats (MP3, WAV, FLAC, etc.)
  - Connects to: `POST /audio/upload` with multipart form-data

- **Error Pages**
  - 404 Not Found
  - 500 Server Error
  - Unauthorized (401)
  - Forbidden (403)

- **Loading States**
  - Page skeleton loaders
  - Progress indicators
  - Real-time status updates via WebSocket

---

### **OPTIONAL BUT RECOMMENDED PAGES**

#### 9. **Advanced Features**
- **Version Control/History Viewer** (`version-history.html`)
  - Visual tree of commits and branches
  - Diff viewer (audio waveform comparison)
  - Branch management (create, merge, delete)
  - Merge conflict resolution
  - Connects to: `GET /repositories/{repo_id}/history`, `POST /merge`, `POST /branch`

- **Mixing Console** (`mixer.html`)
  - Full mixer interface
  - Track levels, pan, mute, solo
  - Effects rack (EQ, reverb, compression)
  - Master output control
  - Connects to: `PUT /repositories/{repo_id}/stems/{stem_id}/mix`

- **Stem Separator Preview** (`stem-separator.html`)
  - Real-time stem separation visualization
  - Individual stem playback
  - Stem mixing playground
  - Export individual stems
  - Connects to: `GET /artifacts/{artifact_id}/stems`, `POST /export/{stem_id}`

- **Tutorial/Onboarding** (`onboarding.html`)
  - Interactive tutorial for new users
  - Feature walkthrough
  - Sample project creation
  - Links to documentation

---

### **API ENDPOINT MAPPING**

| Page | Method | Endpoint | Purpose |
|------|--------|----------|---------|
| Login | POST | `/auth/login` | User authentication |
| Signup | POST | `/auth/signup` | New account creation |
| Projects | GET | `/repositories` | List all user projects |
| Editor | GET | `/repositories/{repo_id}` | Load project data |
| Editor | POST | `/generate` | Start music generation |
| Editor | POST | `/separate-stems` | Start stem separation |
| Editor | POST | `/analyze` | Analyze audio (BPM, key) |
| Editor | POST | `/commit` | Create version commit |
| Generator | GET | `/tasks/{task_id}` | Check task status |
| Collaborators | GET/POST | `/repositories/{repo_id}/collaborators` | Manage team |
| Share | POST | `/share/{repo_id}` | Generate share link |
| Explore | GET | `/projects/explore` | Get featured projects |
| Dashboard | GET | `/user/stats` | Get user statistics |
| Settings | PUT | `/user/profile` | Update profile |
| Notifications | GET | `/notifications` | Fetch all notifications |

---

### **FRONTEND FEATURES REQUIRING BACKEND SUPPORT**

#### **Real-Time Updates (WebSocket)**
```javascript
// Connection needed for:
ws://server:8000/ws/{user_id}/{repo_id}

// Events:
- Task progress (generation, stem separation)
- Collaborative edits (other users editing same project)
- Notifications (new collaborators, comments)
- File upload progress
```

#### **Authentication Flow**
1. User logs in → JWT token returned
2. Token stored in localStorage
3. All API requests include: `Authorization: Bearer {token}`
4. Token refresh endpoint: `POST /auth/refresh`

#### **File Upload/Download**
- Upload audio: `POST /audio/upload` with multipart/form-data
- Download stems/results: `GET /artifacts/{id}/download`
- Stream audio playback: Audio `<source src={url}>`

---

### **IMPLEMENTATION PRIORITY**

**Phase 1 (Frontend MVP):**
1. ✅ Login/Signup
2. ✅ Dashboard
3. ✅ Projects List
4. 🔴 Main Editor (DAW)
5. 🔴 AI Generator Studio
6. ✅ Settings

**Phase 2 (Core Features):**
7. 🔴 Collaborators/Sharing
8. 🔴 Explore
9. 🔴 Version History
10. 🔴 Notifications

**Phase 3 (Polish & Advanced):**
11. Mixer Console
12. Stem Separator Preview
13. Marketplace
14. Onboarding

---

### **TECHNOLOGY STACK FOR FRONTEND**

- **HTML5:** Semantic markup
- **Tailwind CSS:** Styling (already configured)
- **JavaScript/TypeScript:** Interactivity
- **Wavesurfer.js:** Audio waveform display
- **Tone.js:** Audio playback and synthesis
- **WebSocket:** Real-time updates
- **Fetch/Axios:** API communication
- **Material Symbols:** Icons

---

### **KEY CHALLENGES TO ADDRESS**

1. **Audio Processing in Browser:**
   - Large audio files (streaming needed)
   - Real-time visualization
   - Playback synchronization

2. **Real-Time Collaboration:**
   - WebSocket connection management
   - Conflict resolution
   - User presence indicators

3. **Task Progress Updates:**
   - Long-running AI tasks (5-30 seconds)
   - Progress bars and ETA
   - Error handling and retries

4. **Large File Uploads:**
   - Chunked upload
   - Resume capability
   - Progress indication

---

## **SUMMARY**

**Minimum 9 pages needed** for full BeatFlow AI functionality:
- 2 Auth pages (Login, Signup)
- 1 Editor page (Critical - DAW interface)
- 1 Generator page (Critical - Music generation)
- 2 Project management pages
- 1 Collaborators page
- 1 Explore page
- 1 Settings page

**Plus supporting features:**
- WebSocket real-time updates
- File upload/download
- Audio playback controls
- Progress indicators
- Task status tracking
