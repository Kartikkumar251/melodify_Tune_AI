# Phase 6: Optimization & Deployment

## Overview
Optimize the backend for performance, prepare for deployment, and create production-ready infrastructure.

## Objectives
- ✅ Optimize database queries
- ✅ Implement caching strategies
- ✅ Optimize AI model inference
- ✅ Set up monitoring and logging
- ✅ Configure production environment
- ✅ Create deployment scripts
- ✅ Document deployment procedures

## 1. Database Optimization

### 1.1 Add Database Indexes

### models/models.py (Add Indexes)
```python
from sqlalchemy import Index

class Repository(Base):
    __tablename__ = "repositories"
    # ... existing fields ...
    
    __table_args__ = (
        Index('idx_owner_id', 'owner_id'),  # For quick user lookups
        Index('idx_is_public', 'is_public'),  # For public repos
    )

class Commit(Base):
    __tablename__ = "commits"
    # ... existing fields ...
    
    __table_args__ = (
        Index('idx_repo_id', 'repo_id'),  # For commit history
        Index('idx_parent_id', 'parent_commit_id'),  # For tree traversal
        Index('idx_author_id', 'author_id'),  # For author lookups
    )

class Stem(Base):
    __tablename__ = "stems"
    # ... existing fields ...
    
    __table_args__ = (
        Index('idx_commit_id', 'commit_id'),  # For stem retrieval
        Index('idx_stem_type', 'stem_type'),  # For stem filtering
    )
```

### 1.2 Query Optimization

```python
# routes/version_control.py - Optimized queries

from sqlalchemy.orm import selectinload

@router.get("/history/{repo_id}")
async def get_commit_history(
    repo_id: UUID,
    db: AsyncSession = Depends(get_db)
):
    """Optimized with eager loading to prevent N+1 queries"""
    
    result = await db.execute(
        select(Commit)
        .where(Commit.repo_id == repo_id)
        .options(
            selectinload(Commit.author),
            selectinload(Commit.stems)
        )
        .order_by(Commit.created_at)
    )
    
    commits = result.unique().scalars().all()
    return commits
```

## 2. Caching Strategy

### 2.1 Redis Caching

### services/cache.py
```python
import redis.asyncio as redis
import json
import logging
from typing import Any, Optional
import os
from datetime import timedelta

logger = logging.getLogger(__name__)

class CacheManager:
    def __init__(self):
        self.redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self.client: Optional[redis.Redis] = None
        self.default_ttl = 3600  # 1 hour
    
    async def connect(self):
        """Connect to Redis"""
        self.client = await redis.from_url(self.redis_url, decode_responses=True)
        logger.info("✅ Connected to Redis cache")
    
    async def disconnect(self):
        """Disconnect from Redis"""
        if self.client:
            await self.client.close()
            logger.info("❌ Disconnected from Redis cache")
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        if not self.client:
            return None
        
        try:
            value = await self.client.get(key)
            if value:
                logger.info(f"Cache hit: {key}")
                return json.loads(value)
        except Exception as e:
            logger.warning(f"Cache get error: {str(e)}")
        
        return None
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None
    ) -> bool:
        """Set value in cache"""
        if not self.client:
            return False
        
        try:
            ttl = ttl or self.default_ttl
            await self.client.set(key, json.dumps(value), ex=ttl)
            logger.info(f"Cache set: {key} (TTL: {ttl}s)")
            return True
        except Exception as e:
            logger.warning(f"Cache set error: {str(e)}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete key from cache"""
        if not self.client:
            return False
        
        try:
            await self.client.delete(key)
            return True
        except Exception as e:
            logger.warning(f"Cache delete error: {str(e)}")
            return False
    
    async def clear(self) -> bool:
        """Clear all cache"""
        if not self.client:
            return False
        
        try:
            await self.client.flushdb()
            logger.info("Cache cleared")
            return True
        except Exception as e:
            logger.warning(f"Cache clear error: {str(e)}")
            return False

# Global cache instance
cache_manager = CacheManager()

# Dependency for FastAPI
async def get_cache() -> CacheManager:
    return cache_manager
```

### 2.2 Use Caching in Routes

```python
# routes/repositories.py - With caching

@router.get("/{repo_id}")
async def get_repository(
    repo_id: UUID,
    db: AsyncSession = Depends(get_db),
    cache: CacheManager = Depends(get_cache)
):
    """Get repository with caching"""
    
    cache_key = f"repo:{repo_id}"
    
    # Try cache first
    cached_repo = await cache.get(cache_key)
    if cached_repo:
        return cached_repo
    
    # Query database
    result = await db.execute(
        select(Repository).where(Repository.id == repo_id)
    )
    repo = result.scalars().first()
    
    if not repo:
        raise HTTPException(status_code=404, detail="Repository not found")
    
    repo_dict = {
        "id": repo.id,
        "name": repo.name,
        "description": repo.description,
        "is_public": repo.is_public,
        "genre": repo.genre,
        "created_at": repo.created_at,
        "owner_id": repo.owner_id
    }
    
    # Cache for 1 hour
    await cache.set(cache_key, repo_dict, ttl=3600)
    
    return repo_dict
```

### 2.3 Initialize Cache in Main

```python
# main.py - Add cache initialization

from services.cache import cache_manager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("🚀 Starting BeatFlow AI Backend...")
    await init_db()
    await cache_manager.connect()
    logger.info("✅ Backend initialized")
    yield
    # Shutdown
    logger.info("🛑 Shutting down...")
    await cache_manager.disconnect()
```

## 3. AI Model Optimization

### 3.1 Model Quantization

```python
# worker.py - Model quantization for faster inference

import torch
from torch.quantization import quantize_dynamic

def load_models_optimized():
    """Load and optimize models for inference"""
    
    task_logger.info("⏳ Loading optimized AI Models...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load MusicGen
    musicgen_model = MusicGen.get_pretrained(
        'facebook/musicgen-small',
        device=device
    )
    
    # Apply dynamic quantization for faster inference
    if device == "cpu":
        musicgen_model = torch.quantization.quantize_dynamic(
            musicgen_model,
            {torch.nn.Linear},
            dtype=torch.qint8
        )
    
    task_logger.info("✅ Models optimized")
    return musicgen_model
```

### 3.2 Batch Processing

```python
# worker.py - Support batch generation

@celery_app.task(bind=True)
def generate_music_batch(self, prompts: list):
    """Generate multiple tracks in one batch"""
    
    WorkerState.musicgen_model.set_generation_params(duration=15)
    
    # Generate all at once (more VRAM efficient than sequential)
    with torch.no_grad():
        wavs = WorkerState.musicgen_model.generate(
            descriptions=prompts,
            progress=False
        )
    
    results = []
    for i, wav in enumerate(wavs):
        output_filename = f"batch_gen_{self.request.id}_{i}"
        # Save each file
        audio_write(f"{STATIC_PATH}/audio/{output_filename}", wav.cpu(), ...)
        results.append({"filename": output_filename})
    
    return {"status": "success", "results": results}
```

### 3.3 VRAM Monitoring

```python
# services/monitoring.py

import torch
import logging

logger = logging.getLogger(__name__)

class VRAMMonitor:
    @staticmethod
    def get_usage():
        """Get current VRAM usage"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            total = torch.cuda.get_device_properties(0).total_memory / 1e9
            
            return {
                "allocated_gb": round(allocated, 2),
                "reserved_gb": round(reserved, 2),
                "total_gb": round(total, 2),
                "utilization_percent": round((allocated / total) * 100, 1),
                "available_gb": round((total - allocated) / 1, 2)
            }
        return None
    
    @staticmethod
    def log_usage():
        """Log VRAM usage"""
        usage = VRAMMonitor.get_usage()
        if usage:
            logger.info(
                f"VRAM: {usage['allocated_gb']}GB / {usage['total_gb']}GB "
                f"({usage['utilization_percent']}%)"
            )
```

## 4. Monitoring & Logging

### 4.1 Structured Logging

### config.py - Add logging config

```python
import logging
from logging.handlers import RotatingFileHandler
import json_logging

# Initialize JSON logging
json_logging.ENABLE_JSON_LOGGING = True
json_logging.init_flask(app, enable_json=True)

def setup_logging():
    """Configure structured logging"""
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # File handler with rotation
    file_handler = RotatingFileHandler(
        "logs/beatflow.log",
        maxBytes=10485760,  # 10MB
        backupCount=10
    )
    file_handler.setLevel(logging.DEBUG)
    
    # Formatters
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
```

### 4.2 Health Check Endpoint

```python
# routes/monitoring.py

from fastapi import APIRouter
from services.monitoring import VRAMMonitor
import os

router = APIRouter()

@router.get("/health/detailed")
async def detailed_health():
    """Detailed health check with system info"""
    
    import psutil
    import torch
    
    vram_usage = VRAMMonitor.get_usage()
    
    return {
        "status": "healthy",
        "redis": check_redis_health(),
        "database": await check_db_health(),
        "gpu": {
            "available": torch.cuda.is_available(),
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "memory": vram_usage
        },
        "cpu": {
            "percent": psutil.cpu_percent(),
            "cores": psutil.cpu_count()
        },
        "memory": {
            "percent": psutil.virtual_memory().percent,
            "used_gb": psutil.virtual_memory().used / 1e9
        }
    }
```

## 5. Production Environment Configuration

### .env.production
```
# Environment
ENVIRONMENT=production

# Database
DATABASE_URL=postgresql://user:password@db.production.com:5432/beatflow
SQLALCHEMY_ECHO=False

# Redis
REDIS_URL=redis://cache.production.com:6379/0

# Celery
CELERY_BROKER_URL=redis://cache.production.com:6379/0
CELERY_RESULT_BACKEND=redis://cache.production.com:6379/1

# API
API_HOST=0.0.0.0
API_PORT=8000
API_RELOAD=False

# AI
DEVICE=cuda
MUSICGEN_MODEL=facebook/musicgen-small

# Logging
LOG_LEVEL=INFO
LOG_FILE=/var/log/beatflow/app.log

# Storage
STATIC_PATH=/var/beatflow/static
S3_BUCKET=beatflow-production
S3_REGION=us-east-1

# CORS
CORS_ORIGINS=["https://beatflow.app"]
```

## 6. Docker Deployment

### Dockerfile
```dockerfile
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

WORKDIR /app

# Install dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create directories
RUN mkdir -p static/audio static/stems logs

# Expose ports
EXPOSE 8000 6379

# Run application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### docker-compose.yml
```yaml
version: '3.8'

services:
  db:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: beatflow
      POSTGRES_USER: beatflow
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U beatflow"]
      interval: 10s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5

  worker:
    build: .
    command: celery -A worker.celery_app worker --loglevel=info
    environment:
      - DATABASE_URL=postgresql://beatflow:${DB_PASSWORD}@db:5432/beatflow
      - REDIS_URL=redis://redis:6379/0
      - DEVICE=cuda
    depends_on:
      db:
        condition: service_healthy
      redis:
        condition: service_healthy
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  api:
    build: .
    command: uvicorn main:app --host 0.0.0.0 --port 8000
    environment:
      - DATABASE_URL=postgresql://beatflow:${DB_PASSWORD}@db:5432/beatflow
      - REDIS_URL=redis://redis:6379/0
    ports:
      - "8000:8000"
    depends_on:
      db:
        condition: service_healthy
      redis:
        condition: service_healthy
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 10s
      timeout: 5s
      retries: 5

volumes:
  postgres_data:
  redis_data:
```

## 7. Deployment Scripts

### scripts/deploy.sh
```bash
#!/bin/bash

set -e

echo "🚀 Starting BeatFlow AI Deployment..."

# Load environment
source .env.production

# Build images
echo "📦 Building Docker images..."
docker-compose build

# Start services
echo "🔄 Starting services..."
docker-compose up -d

# Run migrations
echo "🗄️  Running database migrations..."
docker-compose exec api alembic upgrade head

# Initialize cache
echo "💾 Initializing cache..."
docker-compose exec api python -c "from services.cache import cache_manager; import asyncio; asyncio.run(cache_manager.connect())"

# Verify deployment
echo "✅ Verifying deployment..."
docker-compose exec api curl http://localhost:8000/health

echo "🎉 Deployment complete!"
echo "API available at: http://localhost:8000"
echo "Docs available at: http://localhost:8000/docs"
```

### scripts/rollback.sh
```bash
#!/bin/bash

echo "↩️ Rolling back deployment..."

# Stop services
docker-compose down

# Restore previous version
git checkout HEAD~1
docker-compose up -d

echo "✅ Rollback complete"
```

## 8. Performance Benchmarks

### Benchmark Results Template

```
Performance Benchmarks
======================
Date: [DATE]
Environment: [Production/Staging]

Response Times:
- Health check: 10ms
- Generate music (API call): 50ms
- Get status: 20ms
- List commits: 100ms
- Create commit: 80ms

AI Performance:
- Music generation (15s): 12s (warmup: 35s)
- Stem separation (15s track): 25s
- Audio analysis: 3s

Database:
- Query: 50ms average
- Commit creation: 100ms
- History retrieval (50 commits): 200ms

Cache Hit Rate: 85%
Average API Response Time: 60ms
```

## 9. Monitoring Checklist

- [ ] Logging configured (file + console)
- [ ] Health checks implemented
- [ ] VRAM monitoring active
- [ ] Database connection pooling optimized
- [ ] Redis caching implemented
- [ ] Docker images built and tested
- [ ] Docker Compose stack verified
- [ ] Deployment scripts created and tested
- [ ] Rollback procedure documented
- [ ] Performance benchmarks documented

## Checklist for Phase 6 Completion
- [ ] Database indexes added and verified
- [ ] Query optimization completed
- [ ] Redis caching fully implemented
- [ ] Model optimization applied
- [ ] Structured logging configured
- [ ] Health check endpoints working
- [ ] Docker setup complete
- [ ] docker-compose stack tested
- [ ] Deployment script functional
- [ ] All services running smoothly
- [ ] Performance benchmarks recorded
- [ ] Documentation updated

## Next Steps
→ **Project Ready for Hackathon Showcase** 🚀

## Quick Start Production Deployment

```bash
# 1. Prepare environment
cp .env.production .env

# 2. Build and start
docker-compose up -d

# 3. Verify
curl http://localhost:8000/health

# 4. View logs
docker-compose logs -f api

# 5. Scale workers (if needed)
docker-compose up -d --scale worker=3
```

