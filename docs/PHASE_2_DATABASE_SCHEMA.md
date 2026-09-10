# Phase 2: Database Schema & ORM Configuration

## Overview
Design and implement the PostgreSQL database schema with SQLAlchemy ORM, establishing the "Git for Audio" version control system architecture.

## Objectives
- ✅ Design database schema following "Git for Audio" concept
- ✅ Set up PostgreSQL connection with SQLAlchemy
- ✅ Create ORM models for all entities
- ✅ Set up database migrations with Alembic
- ✅ Create database utilities and session management

## 1. Database Schema Design

### Entity Relationship Diagram

```
┌─────────────┐
│   Users     │
└──────┬──────┘
       │ 1:N
       ├─────────────────────┐
       │                     │
   ┌───▼──────────┐  ┌──────┴──────┐
   │Repositories  │  │ Commits     │
   └──────┬───────┘  └──────┬──────┘
          │ 1:N             │
          │                 │ 1:N
          │                 │
          │          ┌──────▼──────┐
          │          │   Stems     │
          │          ├─────────────┤
          │          │ drums       │
          │          │ bass        │
          │          │ vocals      │
          │          │ melody      │
          │          │ other       │
          │          └─────────────┘
          │
          └──────────────────┘
          (Commit has self-ref)
          parent_commit_id
```

### Table Specifications

#### 1. users
```sql
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    username VARCHAR(255) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

#### 2. repositories
```sql
CREATE TABLE repositories (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    owner_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    is_public BOOLEAN DEFAULT FALSE,
    genre VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(owner_id, name)
);
```

#### 3. commits
```sql
CREATE TABLE commits (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    repo_id UUID NOT NULL REFERENCES repositories(id) ON DELETE CASCADE,
    parent_commit_id UUID REFERENCES commits(id) ON DELETE SET NULL,
    author_id UUID NOT NULL REFERENCES users(id) ON DELETE SET NULL,
    message VARCHAR(500),
    bpm INTEGER,
    key_signature VARCHAR(10),
    render_url TEXT,
    generation_params JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

#### 4. stems
```sql
CREATE TABLE stems (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    commit_id UUID NOT NULL REFERENCES commits(id) ON DELETE CASCADE,
    stem_type VARCHAR(50) NOT NULL,
    file_url TEXT NOT NULL,
    duration FLOAT,
    file_size INTEGER,
    generation_params JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

## 2. SQLAlchemy ORM Models

### models/database.py
```python
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy import event
import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@localhost/beatflow")
# Convert to async URL
ASYNC_DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://")

engine = create_async_engine(
    ASYNC_DATABASE_URL,
    echo=os.getenv("SQLALCHEMY_ECHO", "False") == "True",
    pool_pre_ping=True,
    pool_size=20,
    max_overflow=10
)

AsyncSessionLocal = sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False
)

Base = declarative_base()

async def get_db():
    """Dependency for FastAPI"""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()

async def init_db():
    """Initialize database tables"""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

async def drop_db():
    """Drop all tables (for development)"""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
```

### models/models.py
```python
from sqlalchemy import Column, String, Integer, Float, Boolean, DateTime, ForeignKey, JSON, Enum, Text
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid
from .database import Base
from enum import Enum as PyEnum

class StemType(str, PyEnum):
    DRUMS = "drums"
    BASS = "bass"
    VOCALS = "vocals"
    MELODY = "melody"
    OTHER = "other"

class User(Base):
    __tablename__ = "users"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    username = Column(String(255), unique=True, nullable=False, index=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    repositories = relationship("Repository", back_populates="owner")
    commits = relationship("Commit", back_populates="author")

class Repository(Base):
    __tablename__ = "repositories"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    owner_id = Column(PG_UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    is_public = Column(Boolean, default=False)
    genre = Column(String(100))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    owner = relationship("User", back_populates="repositories")
    commits = relationship("Commit", back_populates="repository")

class Commit(Base):
    __tablename__ = "commits"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    repo_id = Column(PG_UUID(as_uuid=True), ForeignKey("repositories.id", ondelete="CASCADE"), nullable=False)
    parent_commit_id = Column(PG_UUID(as_uuid=True), ForeignKey("commits.id", ondelete="SET NULL"), nullable=True)
    author_id = Column(PG_UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    message = Column(String(500))
    bpm = Column(Integer)
    key_signature = Column(String(10))
    render_url = Column(Text)
    generation_params = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    repository = relationship("Repository", back_populates="commits")
    author = relationship("User", back_populates="commits")
    stems = relationship("Stem", back_populates="commit")
    children = relationship("Commit", remote_side=[id], backref="parent")

class Stem(Base):
    __tablename__ = "stems"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    commit_id = Column(PG_UUID(as_uuid=True), ForeignKey("commits.id", ondelete="CASCADE"), nullable=False)
    stem_type = Column(Enum(StemType), nullable=False)
    file_url = Column(Text, nullable=False)
    duration = Column(Float)
    file_size = Column(Integer)
    generation_params = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    commit = relationship("Commit", back_populates="stems")
```

### models/schemas.py
```python
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
from uuid import UUID
from enum import Enum

class StemType(str, Enum):
    DRUMS = "drums"
    BASS = "bass"
    VOCALS = "vocals"
    MELODY = "melody"
    OTHER = "other"

# User Schemas
class UserBase(BaseModel):
    username: str
    email: str

class UserCreate(UserBase):
    password: str

class UserResponse(UserBase):
    id: UUID
    created_at: datetime
    
    class Config:
        from_attributes = True

# Stem Schemas
class StemResponse(BaseModel):
    id: UUID
    stem_type: StemType
    file_url: str
    duration: Optional[float]
    file_size: Optional[int]
    
    class Config:
        from_attributes = True

# Commit Schemas
class CommitBase(BaseModel):
    message: Optional[str] = None
    bpm: Optional[int] = None
    key_signature: Optional[str] = None

class CommitCreate(CommitBase):
    parent_commit_id: Optional[UUID] = None
    generation_params: Optional[dict] = None

class CommitResponse(CommitBase):
    id: UUID
    repo_id: UUID
    parent_commit_id: Optional[UUID]
    author_id: Optional[UUID]
    render_url: Optional[str]
    created_at: datetime
    stems: List[StemResponse] = []
    
    class Config:
        from_attributes = True

# Repository Schemas
class RepositoryBase(BaseModel):
    name: str
    description: Optional[str] = None
    is_public: bool = False
    genre: Optional[str] = None

class RepositoryCreate(RepositoryBase):
    pass

class RepositoryResponse(RepositoryBase):
    id: UUID
    owner_id: UUID
    created_at: datetime
    commits: List[CommitResponse] = []
    
    class Config:
        from_attributes = True
```

## 3. Database Initialization

### Initialize on Application Startup

In `main.py`:
```python
from fastapi import FastAPI
from models.database import init_db

app = FastAPI(title="BeatFlow AI")

@app.on_event("startup")
async def startup():
    await init_db()
    print("✅ Database initialized")

@app.on_event("shutdown")
async def shutdown():
    pass
```

## 4. Alembic Setup (Optional - for Migrations)

```bash
alembic init alembic
```

### alembic/env.py - Configure for async
```python
from sqlalchemy.ext.asyncio import create_async_engine
# ... configure sqlalchemy.url and other settings
```

### Create Migration
```bash
alembic revision --autogenerate -m "Initial schema"
alembic upgrade head
```

## 5. Testing Database Connection

```python
# test_db_connection.py
import asyncio
from models.database import AsyncSessionLocal, init_db
from models.models import User

async def test_connection():
    await init_db()
    async with AsyncSessionLocal() as session:
        result = await session.execute("SELECT 1")
        print("✅ Database connection successful")

if __name__ == "__main__":
    asyncio.run(test_connection())
```

## Checklist for Phase 2 Completion
- [ ] PostgreSQL database created and running
- [ ] SQLAlchemy engine configured (async)
- [ ] All ORM models defined
- [ ] Pydantic schemas created
- [ ] Database initialization logic implemented
- [ ] Database connection tested
- [ ] Alembic migrations configured (optional)

## Next Steps
→ Proceed to **Phase 3: Celery Worker & AI Models Setup**

