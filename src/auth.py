"""
Lightweight user management for the Streamlit app.

Provides signup/login with salted password hashes, Admin/User roles, per-user
health profiles, and an activity log that powers the analytics dashboard
(prediction history, disease popularity trends).

Storage is simple JSON on disk — appropriate for a demo app. A production
system would use a real database + JWT-based sessions (see README "Future
Enhancements").

Default admin account (seeded on first run): admin / admin123
"""

from __future__ import annotations

import hashlib
import json
import secrets
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
USERS_PATH = DATA / "users.json"
ACTIVITY_PATH = DATA / "user_activity.json"

ROLES = ("Admin", "User")


# --------------------------------------------------------------------------- #
# Storage helpers
# --------------------------------------------------------------------------- #
def _load(path: Path, default):
    if path.exists():
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError:
            return default
    return default


def _save(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))


def _hash(password: str, salt: str) -> str:
    return hashlib.sha256((salt + password).encode()).hexdigest()


# --------------------------------------------------------------------------- #
# Users
# --------------------------------------------------------------------------- #
def _users() -> dict:
    users = _load(USERS_PATH, {})
    if "admin" not in users:  # seed a default admin on first run
        salt = secrets.token_hex(8)
        users["admin"] = {
            "name": "Administrator",
            "salt": salt,
            "password_hash": _hash("admin123", salt),
            "role": "Admin",
            "profile": {},
            "created": datetime.now(timezone.utc).isoformat(),
        }
        _save(USERS_PATH, users)
    return users


def register_user(username: str, name: str, password: str) -> tuple[bool, str]:
    """Create a regular user account. Returns (ok, message)."""
    username = username.strip().lower()
    if not username.isalnum() or len(username) < 3:
        return False, "Username must be alphanumeric, 3+ characters."
    if len(password) < 6:
        return False, "Password must be at least 6 characters."
    users = _users()
    if username in users:
        return False, "Username already exists."
    salt = secrets.token_hex(8)
    users[username] = {
        "name": name.strip() or username,
        "salt": salt,
        "password_hash": _hash(password, salt),
        "role": "User",
        "profile": {},
        "created": datetime.now(timezone.utc).isoformat(),
    }
    _save(USERS_PATH, users)
    return True, "Account created — you can log in now."


def verify_user(username: str, password: str) -> dict | None:
    """Return the user record on valid credentials, else None."""
    users = _users()
    u = users.get(username.strip().lower())
    if u and _hash(password, u["salt"]) == u["password_hash"]:
        return {"username": username.strip().lower(), "name": u["name"], "role": u["role"]}
    return None


def get_profile(username: str) -> dict:
    return _users().get(username, {}).get("profile", {})


def update_profile(username: str, profile: dict) -> None:
    users = _users()
    if username in users:
        users[username]["profile"] = profile
        _save(USERS_PATH, users)


def list_users() -> list[dict]:
    """Admin view: all users without secrets."""
    return [
        {"username": k, "name": v["name"], "role": v["role"], "created": v.get("created", "")}
        for k, v in _users().items()
    ]


# --------------------------------------------------------------------------- #
# Activity log (powers analytics: history, popularity trends)
# --------------------------------------------------------------------------- #
def log_event(username: str, event_type: str, detail: dict) -> None:
    events = _load(ACTIVITY_PATH, [])
    events.append(
        {
            "user": username,
            "type": event_type,
            "detail": detail,
            "ts": datetime.now(timezone.utc).isoformat(),
        }
    )
    _save(ACTIVITY_PATH, events)


def get_events(username: str | None = None) -> list[dict]:
    """All events, or one user's events (newest first)."""
    events = _load(ACTIVITY_PATH, [])
    if username:
        events = [e for e in events if e["user"] == username]
    return sorted(events, key=lambda e: e["ts"], reverse=True)
