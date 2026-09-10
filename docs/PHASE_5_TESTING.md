# Phase 5: Integration & Testing

## Overview
Integrate all backend components, run comprehensive tests, and validate the entire system works end-to-end.

## Objectives
- ✅ Set up testing framework (pytest)
- ✅ Create unit tests for models and services
- ✅ Create integration tests for API endpoints
- ✅ Create end-to-end workflow tests
- ✅ Validate AI task execution
- ✅ Performance and load testing
- ✅ Document all test results

## 1. Testing Setup

### requirements-dev.txt
```
pytest==7.4.3
pytest-asyncio==0.21.1
pytest-cov==4.1.0
httpx==0.25.1
faker==20.0.0
```

### Installation
```bash
pip install -r requirements-dev.txt
```

## 2. Test Configuration

### tests/conftest.py
```python
import pytest
import asyncio
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from httpx import AsyncClient
from main import app
from models.database import Base, get_db
from models.models import User, Repository, Commit, Stem
from uuid import uuid4
import os

# Use in-memory SQLite for testing
TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"

@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
async def test_db():
    """Create test database"""
    engine = create_async_engine(
        TEST_DATABASE_URL,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool
    )
    
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    TestingSessionLocal = sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False
    )
    
    async def override_get_db():
        async with TestingSessionLocal() as session:
            yield session
    
    app.dependency_overrides[get_db] = override_get_db
    
    yield TestingSessionLocal
    
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)

@pytest.fixture
async def client(test_db):
    """Create test client"""
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac

@pytest.fixture
async def test_user(test_db):
    """Create test user"""
    async with test_db() as session:
        user = User(
            id=uuid4(),
            username="testuser",
            email="test@example.com",
            password_hash="hashed_password"
        )
        session.add(user)
        await session.commit()
        return user

@pytest.fixture
async def test_repository(test_db, test_user):
    """Create test repository"""
    async with test_db() as session:
        repo = Repository(
            id=uuid4(),
            owner_id=test_user.id,
            name="Test Repo",
            description="Test Repository",
            is_public=True,
            genre="Electronic"
        )
        session.add(repo)
        await session.commit()
        return repo

@pytest.fixture
async def test_commit(test_db, test_repository, test_user):
    """Create test commit"""
    async with test_db() as session:
        commit = Commit(
            id=uuid4(),
            repo_id=test_repository.id,
            parent_commit_id=None,
            author_id=test_user.id,
            message="Initial commit",
            bpm=120,
            key_signature="C Major",
            render_url="/static/audio/gen_123.wav"
        )
        session.add(commit)
        await session.commit()
        return commit
```

## 3. Unit Tests

### tests/test_models.py
```python
import pytest
from models.models import User, Repository, Commit, Stem, StemType
from uuid import uuid4

@pytest.mark.asyncio
async def test_user_creation(test_db):
    """Test user model creation"""
    async with test_db() as session:
        user = User(
            username="johndoe",
            email="john@example.com",
            password_hash="hashed_pwd"
        )
        session.add(user)
        await session.commit()
        
        assert user.id is not None
        assert user.username == "johndoe"
        assert user.email == "john@example.com"

@pytest.mark.asyncio
async def test_repository_creation(test_db, test_user):
    """Test repository model creation"""
    async with test_db() as session:
        repo = Repository(
            owner_id=test_user.id,
            name="My Track",
            description="Chill lo-fi beat",
            genre="Lo-Fi"
        )
        session.add(repo)
        await session.commit()
        
        assert repo.id is not None
        assert repo.name == "My Track"
        assert repo.owner_id == test_user.id

@pytest.mark.asyncio
async def test_commit_creation(test_db, test_repository, test_user):
    """Test commit model creation"""
    async with test_db() as session:
        commit = Commit(
            repo_id=test_repository.id,
            parent_commit_id=None,
            author_id=test_user.id,
            message="Initial generation",
            bpm=120,
            key_signature="C Major"
        )
        session.add(commit)
        await session.commit()
        
        assert commit.id is not None
        assert commit.bpm == 120
        assert commit.parent_commit_id is None

@pytest.mark.asyncio
async def test_commit_tree(test_db, test_repository, test_user):
    """Test commit tree structure"""
    async with test_db() as session:
        # Create initial commit
        commit1 = Commit(
            repo_id=test_repository.id,
            author_id=test_user.id,
            message="Initial commit",
            bpm=120
        )
        session.add(commit1)
        await session.flush()
        
        # Create child commit
        commit2 = Commit(
            repo_id=test_repository.id,
            parent_commit_id=commit1.id,
            author_id=test_user.id,
            message="Modified drums",
            bpm=120
        )
        session.add(commit2)
        await session.commit()
        
        assert commit2.parent_commit_id == commit1.id

@pytest.mark.asyncio
async def test_stem_creation(test_db, test_commit):
    """Test stem model creation"""
    async with test_db() as session:
        stem = Stem(
            commit_id=test_commit.id,
            stem_type=StemType.DRUMS,
            file_url="/static/stems/drums.wav"
        )
        session.add(stem)
        await session.commit()
        
        assert stem.stem_type == StemType.DRUMS
```

## 4. API Integration Tests

### tests/test_api_generation.py
```python
import pytest
from httpx import AsyncClient

@pytest.mark.asyncio
async def test_health_check(client):
    """Test health check endpoint"""
    response = await client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

@pytest.mark.asyncio
async def test_generate_music_endpoint(client):
    """Test music generation endpoint"""
    response = await client.post(
        "/api/ai/generate",
        json={
            "prompt": "Upbeat electronic dance music",
            "bpm": 128,
            "mood": "energetic",
            "duration": 15
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "task_id" in data
    assert data["status"] == "pending"
    
    # Verify task_id format
    assert len(data["task_id"]) > 0

@pytest.mark.asyncio
async def test_get_status_endpoint(client):
    """Test status check endpoint"""
    # First generate
    gen_response = await client.post(
        "/api/ai/generate",
        json={
            "prompt": "Test music",
            "bpm": 120,
            "mood": "neutral"
        }
    )
    
    task_id = gen_response.json()["task_id"]
    
    # Then check status
    status_response = await client.get(f"/api/ai/status/{task_id}")
    
    assert status_response.status_code == 200
    assert status_response.json()["task_id"] == task_id
```

### tests/test_api_vcs.py
```python
import pytest
from uuid import uuid4

@pytest.mark.asyncio
async def test_create_repository(client, test_user):
    """Test repository creation"""
    response = await client.post(
        "/api/repositories/create",
        json={
            "name": "My New Track",
            "description": "A new beat",
            "is_public": True,
            "genre": "Hip-Hop"
        },
        params={"owner_id": str(test_user.id)}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "My New Track"
    assert data["owner_id"] == test_user.id

@pytest.mark.asyncio
async def test_create_commit(client, test_repository, test_user):
    """Test commit creation"""
    response = await client.post(
        f"/api/vcs/commit/{test_repository.id}",
        json={
            "message": "Added bass drum",
            "bpm": 120,
            "key_signature": "C Major"
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "Added bass drum"
    assert data["repo_id"] == test_repository.id

@pytest.mark.asyncio
async def test_get_commit_history(client, test_repository):
    """Test getting commit history"""
    response = await client.get(f"/api/vcs/history/{test_repository.id}")
    
    assert response.status_code == 200
    data = response.json()
    assert "commits" in data
    assert isinstance(data["commits"], list)

@pytest.mark.asyncio
async def test_fork_repository(client, test_repository, test_user):
    """Test repository forking"""
    new_owner_id = uuid4()
    
    response = await client.post(
        "/api/vcs/fork",
        json={
            "original_repo_id": str(test_repository.id),
            "fork_name": "My Remix",
            "owner_id": str(new_owner_id)
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "My Remix"
    assert data["owner_id"] == new_owner_id
```

## 5. End-to-End Workflow Tests

### tests/test_e2e_workflow.py
```python
import pytest
import asyncio

@pytest.mark.asyncio
async def test_complete_workflow(client, test_user):
    """
    Test complete workflow:
    1. Create repository
    2. Generate music
    3. Separate stems
    4. Create commit
    5. Get history
    """
    
    # Step 1: Create repository
    repo_response = await client.post(
        "/api/repositories/create",
        json={
            "name": "E2E Test Track",
            "description": "End-to-end test",
            "genre": "Electronic"
        },
        params={"owner_id": str(test_user.id)}
    )
    assert repo_response.status_code == 200
    repo_id = repo_response.json()["id"]
    
    # Step 2: Generate music
    gen_response = await client.post(
        "/api/ai/generate",
        json={
            "prompt": "Fast-paced electronic track",
            "bpm": 140,
            "mood": "intense",
            "duration": 15,
            "repo_id": str(repo_id)
        }
    )
    assert gen_response.status_code == 200
    task_id = gen_response.json()["task_id"]
    print(f"Generation task ID: {task_id}")
    
    # Step 3: Check generation status (in real scenario, would wait for completion)
    status_response = await client.get(f"/api/ai/status/{task_id}")
    assert status_response.status_code == 200
    
    # Step 4: Create commit
    commit_response = await client.post(
        f"/api/vcs/commit/{repo_id}",
        json={
            "message": "Initial generation",
            "bpm": 140,
            "key_signature": "D Major"
        }
    )
    assert commit_response.status_code == 200
    commit_id = commit_response.json()["id"]
    
    # Step 5: Get history
    history_response = await client.get(f"/api/vcs/history/{repo_id}")
    assert history_response.status_code == 200
    history = history_response.json()["commits"]
    assert len(history) > 0

@pytest.mark.asyncio
async def test_branching_workflow(client, test_user):
    """
    Test branching workflow:
    1. Create repository
    2. Create initial commit
    3. Create branch (via child commit)
    4. Verify both branches exist
    """
    
    # Create repo
    repo_response = await client.post(
        "/api/repositories/create",
        json={
            "name": "Branch Test",
            "genre": "Ambient"
        },
        params={"owner_id": str(test_user.id)}
    )
    repo_id = repo_response.json()["id"]
    
    # Create initial commit
    commit1_response = await client.post(
        f"/api/vcs/commit/{repo_id}",
        json={"message": "Initial"}
    )
    commit1_id = commit1_response.json()["id"]
    
    # Create two child commits (branches)
    commit2_response = await client.post(
        f"/api/vcs/commit/{repo_id}",
        json={
            "message": "Branch 1 modification",
            "parent_commit_id": str(commit1_id)
        }
    )
    commit2_id = commit2_response.json()["id"]
    
    commit3_response = await client.post(
        f"/api/vcs/commit/{repo_id}",
        json={
            "message": "Branch 2 modification",
            "parent_commit_id": str(commit1_id)
        }
    )
    commit3_id = commit3_response.json()["id"]
    
    # Verify history shows both branches
    history_response = await client.get(f"/api/vcs/history/{repo_id}")
    history = history_response.json()["commits"]
    
    assert len(history) == 3
    assert any(c["id"] == commit2_id for c in history)
    assert any(c["id"] == commit3_id for c in history)
```

## 6. Performance Testing

### tests/test_performance.py
```python
import pytest
import time

@pytest.mark.asyncio
async def test_large_history(client, test_user):
    """Test performance with large commit history"""
    
    # Create repo
    repo_response = await client.post(
        "/api/repositories/create",
        json={"name": "Performance Test"},
        params={"owner_id": str(test_user.id)}
    )
    repo_id = repo_response.json()["id"]
    
    # Create many commits
    parent_id = None
    for i in range(50):
        commit_response = await client.post(
            f"/api/vcs/commit/{repo_id}",
            json={
                "message": f"Commit {i}",
                "parent_commit_id": str(parent_id) if parent_id else None
            }
        )
        parent_id = commit_response.json()["id"]
    
    # Measure history retrieval time
    start_time = time.time()
    history_response = await client.get(f"/api/vcs/history/{repo_id}")
    elapsed_time = time.time() - start_time
    
    print(f"Retrieved 50 commits in {elapsed_time:.2f} seconds")
    assert elapsed_time < 5.0  # Should be fast

@pytest.mark.asyncio
async def test_concurrent_requests(client):
    """Test handling concurrent API requests"""
    import asyncio
    
    async def make_request():
        return await client.get("/health")
    
    # Make 100 concurrent requests
    tasks = [make_request() for _ in range(100)]
    results = await asyncio.gather(*tasks)
    
    # All should succeed
    assert all(r.status_code == 200 for r in results)
```

## 7. Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=models --cov=routes --cov-report=html

# Run specific test file
pytest tests/test_api_generation.py -v

# Run specific test
pytest tests/test_api_generation.py::test_health_check -v

# Run with detailed output
pytest tests/ -vv -s
```

## 8. Coverage Report

```bash
# Generate HTML coverage report
pytest tests/ --cov=. --cov-report=html

# View report
open htmlcov/index.html  # macOS
start htmlcov/index.html  # Windows
```

## 9. Mock Testing (For Celery)

### tests/test_celery_tasks.py
```python
import pytest
from unittest.mock import patch, MagicMock
from worker import generate_music_task, separate_stems_task

@pytest.mark.asyncio
async def test_generate_music_task_mock():
    """Test music generation task with mocking"""
    
    with patch('worker.WorkerState.musicgen_model') as mock_model:
        # Mock the model output
        mock_model.generate.return_value = MagicMock()
        
        # Would test task execution
        # Note: Full testing requires Redis/Celery running

@pytest.mark.asyncio  
async def test_separate_stems_task_mock():
    """Test stem separation with mocking"""
    
    with patch('subprocess.run') as mock_run:
        mock_run.return_value = MagicMock(returncode=0)
        
        # Would test task execution
```

## Checklist for Phase 5 Completion
- [ ] Pytest configured with async support
- [ ] Test database set up (SQLite in-memory)
- [ ] Unit tests for all models written and passing
- [ ] API endpoint tests written and passing
- [ ] End-to-end workflow tests written and passing
- [ ] Performance tests written and passing
- [ ] Coverage report generated (>70% recommended)
- [ ] All tests pass locally
- [ ] Test results documented

## Test Results Summary Template

```
Test Suite Results
==================
Date: [DATE]
Total Tests: [X]
Passed: [X]
Failed: [0]
Skipped: [0]
Coverage: [X%]

Models Tests:
- User model: ✅
- Repository model: ✅
- Commit model: ✅
- Stem model: ✅

API Tests:
- Generation endpoints: ✅
- VCS endpoints: ✅
- Audio endpoints: ✅
- Repository endpoints: ✅

Integration Tests:
- Complete workflow: ✅
- Branching workflow: ✅

Performance:
- Large history: ✅ (< 5s for 50 commits)
- Concurrent requests: ✅ (100 concurrent)
```

## Next Steps
→ Proceed to **Phase 6: Optimization & Deployment**

