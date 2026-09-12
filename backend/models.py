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
    clone_licenses = relationship("CloneLicense", back_populates="creator", cascade="all, delete-orphan")
    clone_purchases = relationship("CloneTransaction", back_populates="cloning_user", foreign_keys="[CloneTransaction.cloning_user_id]", cascade="all, delete-orphan")
    clone_sales = relationship("CloneTransaction", back_populates="original_creator", foreign_keys="[CloneTransaction.original_creator_id]", cascade="all, delete-orphan")

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
    clone_license = relationship("CloneLicense", back_populates="repository", uselist=False, cascade="all, delete-orphan")
    clone_transactions = relationship("CloneTransaction", back_populates="original_repository", foreign_keys="[CloneTransaction.original_repository_id]", cascade="all, delete-orphan")
    activities = relationship("ActivityEvent", back_populates="repository", cascade="all, delete-orphan", order_by="ActivityEvent.created_at.desc()")

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


# ── Clone Licensing & Monetization Entities ───────────────────────
class CloneLicense(Base):
    """
    Defines how other creators can clone/fork a repository or track:
    - mode="free_support": Free clone after completing creator support (Star + Fork)
    - mode="paid": Paid clone with a specified fee in INR (₹)
    """
    __tablename__ = "clone_licenses"

    id: str = Column(String(36), primary_key=True, default=_uuid)
    repository_id: str = Column(String(36), ForeignKey("repositories.id", ondelete="CASCADE"), nullable=False, unique=True)
    commit_id: Optional[str] = Column(String(36), ForeignKey("commits.id", ondelete="SET NULL"), nullable=True)
    creator_id: str = Column(String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    mode: str = Column(String(30), default="free_support")  # "free_support" | "paid"
    price: float = Column(Float, default=0.0)
    currency: str = Column(String(10), default="INR")
    is_active: bool = Column(Boolean, default=True)
    created_at: datetime = Column(DateTime, default=_now)
    updated_at: datetime = Column(DateTime, default=_now, onupdate=_now)

    repository = relationship("Repository", back_populates="clone_license")
    creator = relationship("User", back_populates="clone_licenses")
    commit = relationship("Commit")
    transactions = relationship("CloneTransaction", back_populates="license", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<CloneLicense mode='{self.mode}' price={self.price} repo='{self.repository_id[:8]}'>"


class CloneTransaction(Base):
    """
    Audit log of all clone events (both Paid and Free Support unlocks).
    Tracks creative lineage, revenue splits, and attribution.
    """
    __tablename__ = "clone_transactions"

    id: str = Column(String(36), primary_key=True, default=_uuid)
    license_id: Optional[str] = Column(String(36), ForeignKey("clone_licenses.id", ondelete="SET NULL"), nullable=True)
    original_repository_id: str = Column(String(36), ForeignKey("repositories.id", ondelete="CASCADE"), nullable=False)
    original_commit_id: Optional[str] = Column(String(36), ForeignKey("commits.id", ondelete="SET NULL"), nullable=True)
    original_creator_id: str = Column(String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    cloning_user_id: str = Column(String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    clone_repository_id: Optional[str] = Column(String(36), ForeignKey("repositories.id", ondelete="SET NULL"), nullable=True)
    clone_commit_id: Optional[str] = Column(String(36), ForeignKey("commits.id", ondelete="SET NULL"), nullable=True)

    license_mode: str = Column(String(30), default="free_support")  # "paid" | "free_support"
    total_amount: float = Column(Float, default=0.0)
    creator_amount: float = Column(Float, default=0.0)
    platform_fee: float = Column(Float, default=0.0)
    currency: str = Column(String(10), default="INR")
    status: str = Column(String(30), default="completed")  # "completed" | "demo_completed" | "pending"
    created_at: datetime = Column(DateTime, default=_now)

    license = relationship("CloneLicense", back_populates="transactions")
    original_repository = relationship("Repository", back_populates="clone_transactions", foreign_keys=[original_repository_id])
    clone_repository = relationship("Repository", foreign_keys=[clone_repository_id])
    original_creator = relationship("User", back_populates="clone_sales", foreign_keys=[original_creator_id])
    cloning_user = relationship("User", back_populates="clone_purchases", foreign_keys=[cloning_user_id])
    original_commit = relationship("Commit", foreign_keys=[original_commit_id])

    def __repr__(self) -> str:
        return f"<CloneTransaction mode='{self.license_mode}' amt={self.total_amount} cloner='{self.cloning_user_id[:8]}'>"


# ── Project Activity & Change Timeline Entity ─────────────────────
class ActivityEvent(Base):
    """
    Chronological event log for project lifecycle actions:
    creation, commits, stem separation, mastering, hum-to-beat, forks, clones, stars, etc.
    """
    __tablename__ = "activity_events"

    id: str = Column(String(36), primary_key=True, default=_uuid)
    repository_id: str = Column(String(36), ForeignKey("repositories.id", ondelete="CASCADE"), nullable=False, index=True)
    user_id: Optional[str] = Column(String(36), ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True)
    commit_id: Optional[str] = Column(String(36), ForeignKey("commits.id", ondelete="SET NULL"), nullable=True, index=True)

    event_type: str = Column(String(50), nullable=False, index=True)
    title: str = Column(String(200), nullable=False)
    description: str = Column(Text, default="")
    metadata_json: Optional[str] = Column(Text, nullable=True)  # Serialized JSON dict of metrics, tags, deltas
    created_at: datetime = Column(DateTime, default=_now, index=True)

    repository = relationship("Repository", back_populates="activities")
    user = relationship("User")
    commit = relationship("Commit")

    def __repr__(self) -> str:
        return f"<ActivityEvent type='{self.event_type}' repo='{self.repository_id[:8]}' title='{self.title[:20]}'>"

