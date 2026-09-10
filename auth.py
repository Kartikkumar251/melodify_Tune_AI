"""
auth.py — Authentication & JWT Security Services for BeatFlow AI
Handles password hashing (Argon2), JWT token generation, verification, and FastAPI user auth dependencies.
"""
from __future__ import annotations
import os
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any

from jose import JWTError, jwt
from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session

from database import get_db

# ── Security Configuration ──────────────────────────────────────────
SECRET_KEY: str = os.getenv("JWT_SECRET", "beatflow-secret-changeme-in-prod")
ALGORITHM: str = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7   # 7 days validity

_ph = PasswordHasher()
_oauth2 = OAuth2PasswordBearer(tokenUrl="/auth/login", auto_error=False)


# ── Password Hashing Helpers ───────────────────────────────────────
def hash_password(plain_password: str) -> str:
    """Hash plaintext password with Argon2id."""
    return _ph.hash(plain_password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify plaintext password against stored Argon2 hash."""
    try:
        return _ph.verify(hashed_password, plain_password)
    except (VerifyMismatchError, Exception):
        return False


# ── JWT Token Utilities ────────────────────────────────────────────
def create_access_token(user_id: str, username: str) -> str:
    """Generate signed JWT access token for authenticated session."""
    expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    payload = {
        "sub": user_id,
        "username": username,
        "exp": expire
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str) -> Dict[str, Any]:
    """Decode and validate signature/expiration of a JWT token."""
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError:
        return {}


# ── FastAPI User Dependencies ──────────────────────────────────────
def get_current_user(
    token: Optional[str] = Depends(_oauth2),
    db: Session = Depends(get_db),
):
    """
    FastAPI dependency: Resolves authenticated User instance.
    Raises HTTP 401 if missing or invalid.
    """
    from models import User

    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )
    payload = decode_token(token)
    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token"
        )
    user = db.query(User).filter(User.id == user_id, User.is_active.is_(True)).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User account not found or deactivated"
        )
    return user


def get_current_user_optional(
    token: Optional[str] = Depends(_oauth2),
    db: Session = Depends(get_db),
):
    """
    FastAPI dependency: Returns user if authenticated, otherwise None for public routes.
    """
    from models import User

    if not token:
        return None
    payload = decode_token(token)
    user_id = payload.get("sub")
    if not user_id:
        return None
    return db.query(User).filter(User.id == user_id, User.is_active.is_(True)).first()
