"""
models.py — SQLAlchemy ORM Data Architecture for BeatFlow AI
Defines the "Git for Audio" relational graph:
  User ──< Repository ──< Commit (Self-referential tree) ──< Stem
"""
from __future__ import annotations
import uuid
from datetime import datetime, timezone
from typing import Optional, List

from sqlalchemy import (
    Column, String, Text, Float, Integer, Boolean,
    DateTime, ForeignKey, Enum, UniqueConstraint
)
from sqlalchemy.orm import relationship
from database import Base


def _now() -> datetime:
    """Return naive UTC timestamp."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _uuid() -> str:
    """Generate 36-char UUID string."""
    return str(uuid.uuid4())


# ── User Account Entity ───────────────────────────────────────────
class User(Base):
    __tablename__ = "users"

    id: str = Column(String(36), primary_key=True, default=_uuid)
    username: str = Column(String(50), unique=True, nullable=False, index=True)
    email: str = Column(String(120), unique=True, nullable=False, index=True)
    password_hash: str = Column(String(128), nullable=False)
    bio: str = Column(Text, default="")
    avatar_url: str = Column(String(256), default="")
    library_repo_id: Optional[str] = Column(String(36), nullable=True)   # Default personal workspace repo
    created_at: datetime = Column(DateTime, default=_now)
    is_active: bool = Column(Boolean, default=True)

    repositories = relationship("Repository", back_populates="owner", cascade="all, delete-orphan")
    starred_repos = relationship("Star", back_populates="user", cascade="all, delete-orphan")
    following = relationship("Follow", back_populates="follower", foreign_keys="[Follow.follower_id]", cascade="all, delete-orphan")
    followers = relationship("Follow", back_populates="followee", foreign_keys="[Follow.followee_id]", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<User username='{self.username}'>"


# ── Audio Repository Entity ───────────────────────────────────────
class Repository(Base):
    __tablename__ = "repositories"

    id: str = Column(String(36), primary_key=True, default=_uuid)
    owner_id: str = Column(String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    name: str = Column(String(100), nullable=False)
    description: str = Column(Text, default="")
    is_public: bool = Column(Boolean, default=True)
    forked_from: Optional[str] = Column(String(36), ForeignKey("repositories.id"), nullable=True)
    created_at: datetime = Column(DateTime, default=_now)
    updated_at: datetime = Column(DateTime, default=_now, onupdate=_now)

    play_count: int = Column(Integer, default=0)
    star_count: int = Column(Integer, default=0)

    owner = relationship("User", back_populates="repositories")
    commits = relationship("Commit", back_populates="repository",
                           cascade="all, delete-orphan",
                           order_by="Commit.created_at")
    fork_parent = relationship("Repository", remote_side="Repository.id", foreign_keys=[forked_from])

    def __repr__(self) -> str:
        return f"<Repository name='{self.name}' id='{self.id[:8]}'>"


# ── Audio Commit Node Entity ──────────────────────────────────────
class Commit(Base):
    """
    Represents an immutable version snapshot of an audio mix in the version tree.
    Supports branching via self-referential parent_id.
    """
    __tablename__ = "commits"

    id: str = Column(String(36), primary_key=True, default=_uuid)
    repository_id: str = Column(String(36), ForeignKey("repositories.id", ondelete="CASCADE"), nullable=False)
    parent_id: Optional[str] = Column(String(36), ForeignKey("commits.id"), nullable=True)   # None = root
    author_id: str = Column(String(36), ForeignKey("users.id"), nullable=False)

    # Audio Mix Payload
    message: str = Column(String(200), default="Beat update")
    prompt: str = Column(Text, default="")
    audio_url: str = Column(String(512), nullable=False)
    duration: float = Column(Float, default=0.0)

    # Acoustic & Music Intelligence Metadata
    bpm: Optional[float] = Column(Float, nullable=True)
    key: Optional[str] = Column(String(20), nullable=True)
    energy: Optional[float] = Column(Float, nullable=True)
    mood: Optional[str] = Column(String(80), nullable=True)
    model_used: str = Column(String(60), default="musicgen-small")
    elapsed_sec: float = Column(Float, default=0.0)

    # Human-readable commit hash
    commit_hash: str = Column(String(8), default=lambda: uuid.uuid4().hex[:8], unique=True)
    created_at: datetime = Column(DateTime, default=_now)

    repository = relationship("Repository", back_populates="commits")
    author = relationship("User")
    stems = relationship("Stem", back_populates="commit", cascade="all, delete-orphan")
    comments = relationship("Comment", back_populates="commit", cascade="all, delete-orphan",
                            order_by="Comment.created_at")
    parent = relationship("Commit", remote_side="Commit.id", foreign_keys=[parent_id])
    children = relationship("Commit", back_populates="parent",
                            foreign_keys="[Commit.parent_id]")

    def __repr__(self) -> str:
        return f"<Commit hash='{self.commit_hash}' msg='{self.message[:25]}'>"


# ── Audio Stem Track Entity ───────────────────────────────────────
class Stem(Base):
    """
    Individual separated instrument stems (drums, bass, vocals, other) for a commit.
    """
    __tablename__ = "stems"

    id: str = Column(String(36), primary_key=True, default=_uuid)
    commit_id: str = Column(String(36), ForeignKey("commits.id", ondelete="CASCADE"), nullable=False)
    type: str = Column(
        Enum("drums", "bass", "vocals", "other", "full_mix", name="stem_type"),
        nullable=False
    )
    audio_url: str = Column(String(512), nullable=False)
    file_size: int = Column(Integer, default=0)
    created_at: datetime = Column(DateTime, default=_now)

    commit = relationship("Commit", back_populates="stems")

    def __repr__(self) -> str:
        return f"<Stem type='{self.type}' commit='{self.commit_id[:8]}'>"


# ── Social Graph Entities ──────────────────────────────────────────
class Star(Base):
    """Tracks project starring / favorites."""
    __tablename__ = "stars"
    __table_args__ = (UniqueConstraint("user_id", "repo_id", name="uq_star"),)

    id: str = Column(String(36), primary_key=True, default=_uuid)
    user_id: str = Column(String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    repo_id: str = Column(String(36), ForeignKey("repositories.id", ondelete="CASCADE"), nullable=False)
    created_at: datetime = Column(DateTime, default=_now)

    user = relationship("User", back_populates="starred_repos")
    repository = relationship("Repository")


class Follow(Base):
    """Tracks user follow graph."""
    __tablename__ = "follows"
    __table_args__ = (UniqueConstraint("follower_id", "followee_id", name="uq_follow"),)

    id: str = Column(String(36), primary_key=True, default=_uuid)
    follower_id: str = Column(String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    followee_id: str = Column(String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    created_at: datetime = Column(DateTime, default=_now)

    follower = relationship("User", back_populates="following", foreign_keys=[follower_id])
    followee = relationship("User", back_populates="followers", foreign_keys=[followee_id])


class Comment(Base):
    """Feedback and discussion comments on commits."""
    __tablename__ = "comments"

    id: str = Column(String(36), primary_key=True, default=_uuid)
    commit_id: str = Column(String(36), ForeignKey("commits.id", ondelete="CASCADE"), nullable=False)
    author_id: str = Column(String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    body: str = Column(Text, nullable=False)
    created_at: datetime = Column(DateTime, default=_now)

    commit = relationship("Commit", back_populates="comments")
    author = relationship("User")
