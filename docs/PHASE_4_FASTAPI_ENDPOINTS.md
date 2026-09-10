# Phase 4: FastAPI Server & API Endpoints

## Overview
Build REST API endpoints for music generation, stem separation, version control, and repository management using FastAPI.

## Objectives
- ✅ Set up FastAPI app with CORS and middleware
- ✅ Create authentication endpoints
- ✅ Implement AI generation endpoints (async/polling)
- ✅ Create version control endpoints (commits, branches)
- ✅ Implement audio playback and download
- ✅ Set up WebSocket for real-time updates
- ✅ Add comprehensive error handling

## 1. Main Application File

### main.py
```python
from fastapi import FastAPI, HTTPException, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from models.database import init_db, get_db
import logging
from config import CeleryConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Lifecycle events
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("🚀 Starting BeatFlow AI Backend...")
    await init_db()
    logger.info("✅ Database initialized")
    yield
    # Shutdown
    logger.info("🛑 Shutting down BeatFlow AI Backend...")

# Create FastAPI app
app = FastAPI(
    title="BeatFlow AI Backend",
    description="AI-powered music generation and version control",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to specific domains in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="./static"), name="static")

# Health check
@app.get("/", tags=["Health"])
async def root():
    return {
        "message": "BeatFlow AI Backend",
        "status": "operational",
        "endpoints": {
            "docs": "/docs",
            "redoc": "/redoc"
        }
    }

@app.get("/health", tags=["Health"])
async def health_check():
    return {"status": "healthy"}

# Import and include routers (will be created in Phase 4)
from routes import generation, version_control, audio, repositories

app.include_router(generation.router, prefix="/api/ai", tags=["AI Generation"])
app.include_router(version_control.router, prefix="/api/vcs", tags=["Version Control"])
app.include_router(audio.router, prefix="/api/audio", tags=["Audio"])
app.include_router(repositories.router, prefix="/api/repositories", tags=["Repositories"])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=True
    )
```

## 2. Generation Endpoints

### routes/generation.py
```python
from fastapi import APIRouter, HTTPException, BackgroundTasks, Query
from pydantic import BaseModel, Field
from typing import Optional
from worker import generate_music_task, separate_stems_task, analyze_audio_task
from celery.result import AsyncResult
import logging
from uuid import UUID

logger = logging.getLogger(__name__)
router = APIRouter()

# Request/Response Models
class GenerationRequest(BaseModel):
    prompt: str = Field(..., description="Music description")
    bpm: int = Field(120, ge=60, le=300, description="Beats per minute")
    mood: str = Field("neutral", description="Musical mood/style")
    duration: int = Field(15, ge=5, le=30, description="Duration in seconds")
    repo_id: Optional[str] = None
    commit_id: Optional[str] = None

class GenerationResponse(BaseModel):
    task_id: str
    status: str = "pending"
    message: str

class TaskStatusResponse(BaseModel):
    task_id: str
    status: str
    result: Optional[dict] = None
    progress: Optional[dict] = None
    error: Optional[str] = None

class StemSeparationRequest(BaseModel):
    file_path: str
    output_dir: Optional[str] = None

class StemSeparationResponse(BaseModel):
    task_id: str
    status: str = "pending"

class AudioAnalysisResponse(BaseModel):
    bpm: int
    key_signature: str
    duration: float

# ============================================
# GENERATION ENDPOINTS
# ============================================

@router.post("/generate", response_model=GenerationResponse)
async def generate_music(request: GenerationRequest):
    """
    Generate music from text prompt
    
    Returns a task_id for polling the result.
    
    Example:
    ```
    POST /api/ai/generate
    {
        "prompt": "Upbeat electronic dance music",
        "bpm": 128,
        "mood": "energetic",
        "duration": 15
    }
    ```
    
    Then poll: GET /api/ai/status/{task_id}
    """
    try:
        logger.info(f"🎵 Generation request: {request.prompt}")
        
        # Offload to Celery worker
        task = generate_music_task.delay(
            prompt=request.prompt,
            bpm=request.bpm,
            mood=request.mood,
            duration=request.duration,
            repo_id=request.repo_id,
            commit_id=request.commit_id
        )
        
        logger.info(f"Task queued: {task.id}")
        
        return GenerationResponse(
            task_id=task.id,
            status="pending",
            message="Generation task queued. Poll the status endpoint to check progress."
        )
        
    except Exception as e:
        logger.error(f"Generation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status/{task_id}", response_model=TaskStatusResponse)
async def get_generation_status(task_id: str):
    """
    Check the status of a generation task
    
    Status values:
    - PENDING: Task is queued
    - STARTED: Task is running on GPU
    - PROGRESS: Partial completion (with progress dict)
    - SUCCESS: Completed (with result dict)
    - FAILURE: Failed (with error message)
    - RETRY: Task retried
    """
    try:
        task_result = AsyncResult(task_id)
        
        response = {
            "task_id": task_id,
            "status": task_result.status.lower(),
            "result": None,
            "progress": None,
            "error": None
        }
        
        if task_result.state == 'PROGRESS':
            response["progress"] = task_result.info
        elif task_result.state == 'SUCCESS':
            response["result"] = task_result.result
        elif task_result.state == 'FAILURE':
            response["error"] = str(task_result.info)
        
        return response
        
    except Exception as e:
        logger.error(f"Status check error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/hum-to-beat")
async def hum_to_beat(
    file_upload: UploadFile = File(...),
    bpm: int = Query(120, ge=60, le=300),
    mood: str = Query("neutral")
):
    """
    Generate music based on hummed melody
    
    The hummed audio is used as a conditioning guide for MusicGen-Melody
    """
    try:
        # Save uploaded file temporarily
        import tempfile
        import os
        from fastapi import UploadFile, File, Query
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            content = await file_upload.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        logger.info(f"🎤 Hum-to-beat: {file_upload.filename}")
        
        # Queue task (would need a special hum_to_beat_task in worker.py)
        task = generate_music_task.delay(
            prompt=f"Melody-guided, {mood} mood, {bpm} bpm",
            bpm=bpm,
            mood=mood,
            duration=15
        )
        
        # Clean up temp file after task is submitted
        os.unlink(tmp_path)
        
        return {
            "task_id": task.id,
            "status": "pending",
            "message": "Melody-to-beat generation queued"
        }
        
    except Exception as e:
        logger.error(f"Hum-to-beat error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================
# STEM SEPARATION ENDPOINTS
# ============================================

@router.post("/separate-stems", response_model=StemSeparationResponse)
async def separate_stems(request: StemSeparationRequest):
    """
    Separate audio file into stems (drums, bass, vocals, other)
    
    Example:
    ```
    POST /api/ai/separate-stems
    {
        "file_path": "/static/audio/gen_abc123.wav"
    }
    ```
    """
    try:
        logger.info(f"🎼 Stem separation: {request.file_path}")
        
        task = separate_stems_task.delay(
            file_path=request.file_path,
            output_dir=request.output_dir
        )
        
        return StemSeparationResponse(
            task_id=task.id,
            status="pending"
        )
        
    except Exception as e:
        logger.error(f"Stem separation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================
# ANALYSIS ENDPOINTS
# ============================================

@router.post("/analyze-audio")
async def analyze_audio(file_path: str = Query(...)):
    """
    Analyze audio file for BPM, key, duration
    
    Example:
    ```
    POST /api/ai/analyze-audio?file_path=/static/audio/gen_abc123.wav
    ```
    """
    try:
        logger.info(f"📊 Audio analysis: {file_path}")
        
        task = analyze_audio_task.delay(file_path=file_path)
        
        # Wait for analysis (should be fast)
        result = task.get(timeout=30)
        
        return result
        
    except Exception as e:
        logger.error(f"Analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================
# BATCH GENERATION (For Future)
# ============================================

@router.post("/batch-generate")
async def batch_generate(requests: list[GenerationRequest]):
    """
    Generate multiple variations at once
    
    Returns list of task IDs
    """
    try:
        task_ids = []
        for req in requests:
            task = generate_music_task.delay(
                prompt=req.prompt,
                bpm=req.bpm,
                mood=req.mood,
                duration=req.duration
            )
            task_ids.append(task.id)
        
        return {
            "task_ids": task_ids,
            "count": len(task_ids),
            "message": "Batch generation queued"
        }
        
    except Exception as e:
        logger.error(f"Batch generation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
```

## 3. Version Control Endpoints

### routes/version_control.py
```python
from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from models.database import get_db
from models.models import Repository, Commit, Stem, User
from models.schemas import CommitCreate, CommitResponse
from pydantic import BaseModel, Field
from typing import List, Optional
from uuid import UUID
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

# Models
class CommitRequest(BaseModel):
    parent_commit_id: Optional[UUID] = None
    message: str
    bpm: Optional[int] = None
    key_signature: Optional[str] = None
    generation_params: Optional[dict] = None

class BranchInfo(BaseModel):
    commit_id: UUID
    message: str
    timestamp: str
    author: Optional[str] = None

# ============================================
# COMMIT ENDPOINTS
# ============================================

@router.post("/commit/{repo_id}")
async def create_commit(
    repo_id: UUID,
    request: CommitRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Create a new commit (save current state)
    
    Example:
    ```
    POST /api/vcs/commit/{repo_id}
    {
        "parent_commit_id": "prev-commit-id",
        "message": "Added 808 bass",
        "bpm": 120,
        "key_signature": "C Major",
        "generation_params": {"prompt": "..."}
    }
    ```
    """
    try:
        # Verify repository exists
        result = await db.execute(
            select(Repository).where(Repository.id == repo_id)
        )
        repo = result.scalars().first()
        
        if not repo:
            raise HTTPException(status_code=404, detail="Repository not found")
        
        # Create commit
        commit = Commit(
            repo_id=repo_id,
            parent_commit_id=request.parent_commit_id,
            author_id=repo.owner_id,  # Use repo owner for now
            message=request.message,
            bpm=request.bpm,
            key_signature=request.key_signature,
            generation_params=request.generation_params
        )
        
        db.add(commit)
        await db.commit()
        await db.refresh(commit)
        
        logger.info(f"✅ Commit created: {commit.id}")
        
        return {
            "id": commit.id,
            "repo_id": repo_id,
            "parent_commit_id": request.parent_commit_id,
            "message": request.message,
            "timestamp": commit.created_at
        }
        
    except Exception as e:
        logger.error(f"Commit creation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/commit/{commit_id}")
async def get_commit(
    commit_id: UUID,
    db: AsyncSession = Depends(get_db)
):
    """
    Get commit details and its stems
    """
    try:
        result = await db.execute(
            select(Commit).where(Commit.id == commit_id)
        )
        commit = result.scalars().first()
        
        if not commit:
            raise HTTPException(status_code=404, detail="Commit not found")
        
        return {
            "id": commit.id,
            "message": commit.message,
            "bpm": commit.bpm,
            "key_signature": commit.key_signature,
            "render_url": commit.render_url,
            "timestamp": commit.created_at,
            "stems": [
                {
                    "type": stem.stem_type,
                    "url": stem.file_url,
                    "duration": stem.duration
                }
                for stem in commit.stems
            ]
        }
        
    except Exception as e:
        logger.error(f"Commit retrieval error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/history/{repo_id}")
async def get_commit_history(
    repo_id: UUID,
    db: AsyncSession = Depends(get_db)
):
    """
    Get commit history as a tree structure
    
    Returns the "Git tree" for visualization
    """
    try:
        result = await db.execute(
            select(Commit)
            .where(Commit.repo_id == repo_id)
            .order_by(Commit.created_at)
        )
        commits = result.scalars().all()
        
        if not commits:
            return {"commits": [], "root": None}
        
        # Build tree structure
        commit_map = {c.id: c for c in commits}
        tree = []
        
        for commit in commits:
            tree.append({
                "id": commit.id,
                "parent_id": commit.parent_commit_id,
                "message": commit.message,
                "timestamp": commit.created_at,
                "bpm": commit.bpm,
                "key_signature": commit.key_signature
            })
        
        return {
            "repo_id": repo_id,
            "commits": tree,
            "root": commits[0].id if commits else None
        }
        
    except Exception as e:
        logger.error(f"History retrieval error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================
# BRANCH/FORK ENDPOINTS
# ============================================

@router.post("/fork")
async def fork_repository(
    original_repo_id: UUID,
    fork_name: str,
    owner_id: UUID,
    db: AsyncSession = Depends(get_db)
):
    """
    Fork a repository from a specific commit
    
    Example:
    ```
    POST /api/vcs/fork
    {
        "original_repo_id": "xxx",
        "fork_name": "My Remix",
        "owner_id": "yyy"
    }
    ```
    """
    try:
        # Get original repository
        result = await db.execute(
            select(Repository).where(Repository.id == original_repo_id)
        )
        original_repo = result.scalars().first()
        
        if not original_repo:
            raise HTTPException(status_code=404, detail="Original repository not found")
        
        # Create new repository (fork)
        fork_repo = Repository(
            owner_id=owner_id,
            name=fork_name,
            description=f"Fork of {original_repo.name}",
            is_public=original_repo.is_public,
            genre=original_repo.genre
        )
        
        db.add(fork_repo)
        await db.flush()  # Get ID before commit
        
        # Copy commits and stems from original
        result = await db.execute(
            select(Commit).where(Commit.repo_id == original_repo_id)
        )
        original_commits = result.scalars().all()
        
        commit_mapping = {}  # Map old IDs to new IDs
        
        for original_commit in original_commits:
            new_commit = Commit(
                repo_id=fork_repo.id,
                parent_commit_id=commit_mapping.get(original_commit.parent_commit_id),
                author_id=owner_id,
                message=original_commit.message,
                bpm=original_commit.bpm,
                key_signature=original_commit.key_signature,
                render_url=original_commit.render_url,
                generation_params=original_commit.generation_params
            )
            db.add(new_commit)
            await db.flush()
            commit_mapping[original_commit.id] = new_commit.id
            
            # Copy stems
            for stem in original_commit.stems:
                new_stem = Stem(
                    commit_id=new_commit.id,
                    stem_type=stem.stem_type,
                    file_url=stem.file_url,
                    duration=stem.duration,
                    file_size=stem.file_size,
                    generation_params=stem.generation_params
                )
                db.add(new_stem)
        
        await db.commit()
        await db.refresh(fork_repo)
        
        logger.info(f"✅ Repository forked: {fork_repo.id}")
        
        return {
            "id": fork_repo.id,
            "name": fork_repo.name,
            "owner_id": owner_id,
            "description": fork_repo.description,
            "created_at": fork_repo.created_at
        }
        
    except Exception as e:
        await db.rollback()
        logger.error(f"Fork error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
```

## 4. Audio Endpoints

### routes/audio.py
```python
from fastapi import APIRouter, HTTPException, Query, File, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from pathlib import Path
import logging
import os

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get("/download/{filename}")
async def download_audio(filename: str):
    """
    Download audio file
    
    Example: GET /api/audio/download/gen_abc123.wav
    """
    try:
        file_path = Path(f"./static/audio/{filename}")
        
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found")
        
        if not str(file_path).startswith(os.path.abspath("./static/audio")):
            raise HTTPException(status_code=403, detail="Access denied")
        
        return FileResponse(
            file_path,
            media_type="audio/wav",
            filename=filename
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Download error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/upload")
async def upload_audio(file: UploadFile = File(...)):
    """
    Upload audio file for processing
    """
    try:
        filename = f"upload_{file.filename}"
        file_path = f"./static/audio/{filename}"
        
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        logger.info(f"✅ File uploaded: {filename}")
        
        return {
            "filename": filename,
            "path": f"/static/audio/{filename}",
            "size": len(content)
        }
        
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
```

## 5. Repository Endpoints

### routes/repositories.py
```python
from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from models.database import get_db
from models.models import Repository
from pydantic import BaseModel
from uuid import UUID
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

class RepositoryCreate(BaseModel):
    name: str
    description: str = ""
    is_public: bool = False
    genre: str = ""

class RepositoryResponse(BaseModel):
    id: UUID
    name: str
    description: str
    is_public: bool
    genre: str
    created_at: str

@router.post("/create")
async def create_repository(
    owner_id: UUID,
    request: RepositoryCreate,
    db: AsyncSession = Depends(get_db)
):
    """
    Create a new repository (project)
    """
    try:
        repo = Repository(
            owner_id=owner_id,
            name=request.name,
            description=request.description,
            is_public=request.is_public,
            genre=request.genre
        )
        
        db.add(repo)
        await db.commit()
        await db.refresh(repo)
        
        logger.info(f"✅ Repository created: {repo.id}")
        
        return {
            "id": repo.id,
            "name": repo.name,
            "owner_id": owner_id,
            "created_at": repo.created_at
        }
        
    except Exception as e:
        await db.rollback()
        logger.error(f"Repository creation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{repo_id}")
async def get_repository(
    repo_id: UUID,
    db: AsyncSession = Depends(get_db)
):
    """
    Get repository details
    """
    try:
        result = await db.execute(
            select(Repository).where(Repository.id == repo_id)
        )
        repo = result.scalars().first()
        
        if not repo:
            raise HTTPException(status_code=404, detail="Repository not found")
        
        return {
            "id": repo.id,
            "name": repo.name,
            "description": repo.description,
            "is_public": repo.is_public,
            "genre": repo.genre,
            "created_at": repo.created_at,
            "owner_id": repo.owner_id
        }
        
    except Exception as e:
        logger.error(f"Repository retrieval error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
```

## 6. Running the Server

```bash
# Terminal 3: Start FastAPI server
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Expected Output
```
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Application startup complete
✅ Database initialized
```

### Access API Documentation
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Checklist for Phase 4 Completion
- [ ] FastAPI app created with proper startup/shutdown
- [ ] CORS configured
- [ ] Static file mounting configured
- [ ] Generation endpoints implemented and tested
- [ ] Version control endpoints implemented
- [ ] Repository management endpoints implemented
- [ ] Audio endpoints implemented
- [ ] API documentation accessible at /docs
- [ ] Server runs without errors

## Next Steps
→ Proceed to **Phase 5: Integration & Testing**

