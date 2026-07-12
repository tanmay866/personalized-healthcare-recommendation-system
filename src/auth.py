"""
User management for the Streamlit app, backed by SQLite (see ``db.py``).

Provides signup/login with salted password hashes, Admin/User roles, per-user
health profiles, and an activity log that powers the analytics dashboard
(prediction history, disease popularity trends).

A production system would add JWT-based sessions and a hosted database — the
storage layer in ``db.py`` speaks plain SQL, so moving to PostgreSQL is a
connection-string change, not a rewrite.

Default admin account (seeded on first run): admin / admin123
"""

from __future__ import annotations

import hashlib
import secrets
from datetime import datetime, timezone

from sqlalchemy.exc import IntegrityError

import db

ROLES = ("Admin", "User")


def _hash(password: str, salt: str) -> str:
    return hashlib.sha256((salt + password).encode()).hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _seed_admin() -> None:
    """Create the default admin account if it doesn't exist yet."""
    if db.get_user("admin") is None:
        salt = secrets.token_hex(8)
        try:
            db.insert_user(
                username="admin",
                name="Administrator",
                salt=salt,
                password_hash=_hash("admin123", salt),
                role="Admin",
                created=_now(),
            )
        except IntegrityError:
            pass  # another concurrent request seeded it first — fine


def register_user(username: str, name: str, password: str) -> tuple[bool, str]:
    """Create a regular user account. Returns (ok, message)."""
    _seed_admin()
    username = username.strip().lower()
    if not username.isalnum() or len(username) < 3:
        return False, "Username must be alphanumeric, 3+ characters."
    if len(password) < 6:
        return False, "Password must be at least 6 characters."
    if db.get_user(username) is not None:
        return False, "Username already exists."
    salt = secrets.token_hex(8)
    try:
        db.insert_user(
            username=username,
            name=name.strip() or username,
            salt=salt,
            password_hash=_hash(password, salt),
            role="User",
            created=_now(),
        )
    except IntegrityError:
        # Race safety: two simultaneous signups can both pass the pre-check;
        # the PRIMARY KEY constraint is the final arbiter of uniqueness.
        return False, "Username already exists."
    return True, "Account created — you can log in now."


def verify_user(username: str, password: str) -> dict | None:
    """Return {username, name, role} on valid credentials, else None."""
    _seed_admin()
    u = db.get_user(username.strip().lower())
    if u and _hash(password, u["salt"]) == u["password_hash"]:
        return {"username": u["username"], "name": u["name"], "role": u["role"]}
    return None


def get_profile(username: str) -> dict:
    return db.get_profile(username)


def update_profile(username: str, profile: dict) -> None:
    db.set_profile(username, profile)


def list_users() -> list[dict]:
    """Admin view: all users without secrets."""
    _seed_admin()
    return db.all_users()


def log_event(username: str, event_type: str, detail: dict) -> None:
    db.insert_event(username, event_type, detail, _now())


def get_events(username: str | None = None) -> list[dict]:
    """All events, or one user's events (newest first)."""
    return db.fetch_events(username)
