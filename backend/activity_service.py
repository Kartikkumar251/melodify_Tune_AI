"""
activity_service.py — Unified Project Activity & Change Timeline Service for Melodify
Provides centralized event recording, querying, formatting, and historical synchronization.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import or_

from models import ActivityEvent, Repository, Commit, User, CloneTransaction, Star, Comment

logger = logging.getLogger("activity_service")


# Standardized Event Type Definitions
class EventType:
    PROJECT_CREATED       = "PROJECT_CREATED"
    COMMIT_CREATED        = "COMMIT_CREATED"
    AUDIO_ADDED           = "AUDIO_ADDED"
    AUDIO_UPDATED         = "AUDIO_UPDATED"
    AI_GENERATED          = "AI_GENERATED"
    STEM_SEPARATED        = "STEM_SEPARATED"
    MASTER_APPLIED        = "MASTER_APPLIED"
    HUM_TO_BEAT_GENERATED = "HUM_TO_BEAT_GENERATED"
    AUDIO_CONTINUED       = "AUDIO_CONTINUED"
    FCA_OPTIMIZED         = "FCA_OPTIMIZED"
    MIDI_GENERATED        = "MIDI_GENERATED"
    LYRICS_GENERATED      = "LYRICS_GENERATED"
    FORK_CREATED          = "FORK_CREATED"
    CLONE_CREATED         = "CLONE_CREATED"
    STAR_ADDED            = "STAR_ADDED"
    COMMENT_ADDED         = "COMMENT_ADDED"
    EXPORT_CREATED        = "EXPORT_CREATED"
    PROJECT_UPDATED       = "PROJECT_UPDATED"


def log_activity(
    db: Session,
    repo_id: str,
    event_type: str,
    title: str,
    description: str = "",
    user_id: Optional[str] = None,
    commit_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Optional[ActivityEvent]:
    """
    Safely record an activity event on a project/repository.
    Fails gracefully if any unexpected error occurs so core business operations are never interrupted.
    """
    if not repo_id:
        return None

    try:
        meta_str = json.dumps(metadata) if metadata else None
        event = ActivityEvent(
            repository_id=repo_id,
            user_id=user_id,
            commit_id=commit_id,
            event_type=event_type,
            title=title,
            description=description or "",
            metadata_json=meta_str,
        )
        db.add(event)
        db.commit()
        db.refresh(event)
        return event
    except Exception as e:
        logger.warning(f"[ActivityService] Failed to log activity {event_type} on repo {repo_id}: {e}")
        try:
            db.rollback()
        except Exception:
            pass
        return None


def format_event_dict(event: ActivityEvent) -> Dict[str, Any]:
    """
    Formats an ActivityEvent into a rich, structured JSON response with author details,
    commit link info, and parsed metadata.
    """
    meta: Dict[str, Any] = {}
    if event.metadata_json:
        try:
            meta = json.loads(event.metadata_json)
        except Exception:
            meta = {}

    # Determine visual category / dot theme
    category = "general"
    icon = "●"
    dot_color = "var(--green-accent)"

    et = event.event_type
    if et in (EventType.COMMIT_CREATED, EventType.AI_GENERATED):
        category = "commit"
        icon = "🎵"
        dot_color = "var(--green-bright)"
    elif et in (EventType.MASTER_APPLIED, EventType.STEM_SEPARATED, EventType.AUDIO_CONTINUED, EventType.HUM_TO_BEAT_GENERATED, EventType.FCA_OPTIMIZED, EventType.MIDI_GENERATED):
        category = "audio"
        icon = "⚡"
        dot_color = "var(--blue)"
    elif et in (EventType.FORK_CREATED, EventType.CLONE_CREATED):
        category = "lineage"
        icon = "🌿"
        dot_color = "var(--purple)"
    elif et in (EventType.STAR_ADDED, EventType.COMMENT_ADDED):
        category = "social"
        icon = "★" if et == EventType.STAR_ADDED else "💬"
        dot_color = "var(--orange)"
    elif et == EventType.EXPORT_CREATED:
        category = "export"
        icon = "📦"
        dot_color = "var(--yellow)"

    author_name = event.user.username if event.user else "System"
    avatar_url = event.user.avatar_url if event.user else ""

    commit_data = None
    if event.commit:
        commit_data = {
            "id": event.commit.id,
            "hash": event.commit.commit_hash,
            "message": event.commit.message,
            "audio_url": event.commit.audio_url,
            "bpm": event.commit.bpm,
            "key": event.commit.key,
            "duration": event.commit.duration,
        }

    return {
        "id": event.id,
        "repository_id": event.repository_id,
        "event_type": event.event_type,
        "category": category,
        "badge_class": f"dot-{category}",
        "icon": icon,
        "dot_color": dot_color,
        "title": event.title,
        "description": event.description,
        "user_id": event.user_id,
        "author": author_name,
        "username": author_name,
        "avatar_url": avatar_url,
        "commit_id": event.commit_id,
        "commit_hash": event.commit.commit_hash if event.commit else None,
        "commit": commit_data,
        "metadata": meta,
        "created_at": event.created_at.isoformat() if event.created_at else None,
        "time_ago": _relative_time(event.created_at) if event.created_at else "just now",
        "relative_time": _relative_time(event.created_at) if event.created_at else "just now",
    }


def _relative_time(dt: datetime) -> str:
    """Format datetime to human readable relative time."""
    if not dt:
        return ""
    diff = datetime.utcnow() - dt
    secs = int(diff.total_seconds())
    if secs < 60:
        return "just now"
    if secs < 3600:
        mins = secs // 60
        return f"{mins}m ago"
    if secs < 86400:
        hrs = secs // 3600
        return f"{hrs}h ago"
    days = secs // 86400
    if days < 30:
        return f"{days}d ago"
    return dt.strftime("%b %d, %Y")


def backfill_historical_activities(db: Session, repo: Repository) -> None:
    """
    If a repository currently has 0 activity events (e.g. created before activity logging),
    synthesizes initial activity events directly from actual database records (Project Creation,
    Commits, Forks, Stems, Stars) so historical projects display a complete timeline.
    """
    existing_count = db.query(ActivityEvent).filter(ActivityEvent.repository_id == repo.id).count()
    if existing_count > 0:
        return

    try:
        # 1. Project Created Event
        event_created = ActivityEvent(
            repository_id=repo.id,
            user_id=repo.owner_id,
            event_type=EventType.PROJECT_CREATED,
            title=f"Project Created — '{repo.name}'",
            description=repo.description or "Initialized audio workspace repository.",
            created_at=repo.created_at,
        )
        db.add(event_created)

        # If forked from upstream
        if repo.forked_from and repo.fork_parent:
            fork_event = ActivityEvent(
                repository_id=repo.id,
                user_id=repo.owner_id,
                event_type=EventType.FORK_CREATED,
                title=f"Forked from @{repo.fork_parent.owner.username if repo.fork_parent.owner else 'creator'}/{repo.fork_parent.name}",
                description="Cloned upstream audio commit history into new branch.",
                created_at=repo.created_at,
            )
            db.add(fork_event)

        # 2. Add events for existing commits
        for c in repo.commits:
            stem_names = [s.type for s in c.stems] if c.stems else []
            meta = {
                "bpm": c.bpm,
                "key": c.key,
                "mood": c.mood,
                "duration": c.duration,
                "stems": stem_names,
            }
            c_event = ActivityEvent(
                repository_id=repo.id,
                user_id=c.author_id,
                commit_id=c.id,
                event_type=EventType.COMMIT_CREATED,
                title=f"Committed \"{c.message}\"",
                description=f"Snapshot {c.commit_hash} · BPM: {c.bpm or 'N/A'} · Key: {c.key or 'N/A'}",
                metadata_json=json.dumps(meta),
                created_at=c.created_at,
            )
            db.add(c_event)

            # If commit has stems, add stem separation event
            if stem_names:
                stem_event = ActivityEvent(
                    repository_id=repo.id,
                    user_id=c.author_id,
                    commit_id=c.id,
                    event_type=EventType.STEM_SEPARATED,
                    title=f"Stem Separation ({len(stem_names)} Tracks)",
                    description=f"Isolated stems: {', '.join(stem_names)} for commit {c.commit_hash}",
                    metadata_json=json.dumps({"stems": stem_names}),
                    created_at=c.created_at,
                )
                db.add(stem_event)

        db.commit()
    except Exception as e:
        logger.warning(f"[ActivityService] Error backfilling activities for repo {repo.id}: {e}")
        try:
            db.rollback()
        except Exception:
            pass


def get_project_activities(
    db: Session,
    repo_id: str,
    limit: int = 50,
    offset: int = 0,
    event_type_filter: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], int]:
    """
    Fetch paginated, chronological activity events for a repository.
    """
    repo = db.query(Repository).filter(Repository.id == repo_id).first()
    if not repo:
        return [], 0

    # Ensure historical activities are populated if empty
    backfill_historical_activities(db, repo)

    query = db.query(ActivityEvent).filter(ActivityEvent.repository_id == repo_id)

    # Apply category / type filters
    if event_type_filter and event_type_filter != "all":
        if event_type_filter == "commits":
            query = query.filter(ActivityEvent.event_type.in_([EventType.COMMIT_CREATED, EventType.AI_GENERATED]))
        elif event_type_filter == "audio":
            query = query.filter(ActivityEvent.event_type.in_([
                EventType.MASTER_APPLIED,
                EventType.STEM_SEPARATED,
                EventType.HUM_TO_BEAT_GENERATED,
                EventType.AUDIO_CONTINUED,
                EventType.FCA_OPTIMIZED,
                EventType.MIDI_GENERATED,
                EventType.AUDIO_ADDED,
            ]))
        elif event_type_filter in ("lineage", "forks", "clones"):
            query = query.filter(ActivityEvent.event_type.in_([EventType.FORK_CREATED, EventType.CLONE_CREATED]))
        elif event_type_filter == "social":
            query = query.filter(ActivityEvent.event_type.in_([EventType.STAR_ADDED, EventType.COMMENT_ADDED]))
        else:
            query = query.filter(ActivityEvent.event_type == event_type_filter)

    total_count = query.count()
    events = query.order_by(ActivityEvent.created_at.desc(), ActivityEvent.id.desc()).offset(offset).limit(limit).all()

    formatted = [format_event_dict(e) for e in events]
    return formatted, total_count
