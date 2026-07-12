"""
SQLite database layer for user management and activity tracking.

A real relational store (tables, schema, parameterized SQL) replacing the
earlier JSON files. SQLite ships with Python, so there are no extra
dependencies — and because everything below speaks plain SQL through one
``_connect()`` helper, swapping in PostgreSQL/MySQL later is a matter of
changing the connection, not the queries.

Schema
------
users(username PK, name, salt, password_hash, role, profile_json, created)
activity(id PK AUTOINCREMENT, username, type, detail_json, ts)
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "app.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    username      TEXT PRIMARY KEY,
    name          TEXT NOT NULL,
    salt          TEXT NOT NULL,
    password_hash TEXT NOT NULL,
    role          TEXT NOT NULL DEFAULT 'User',
    profile_json  TEXT NOT NULL DEFAULT '{}',
    created       TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS activity (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    username    TEXT NOT NULL,
    type        TEXT NOT NULL,
    detail_json TEXT NOT NULL DEFAULT '{}',
    ts          TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_activity_user ON activity(username);
CREATE INDEX IF NOT EXISTS idx_activity_ts   ON activity(ts);

CREATE TABLE IF NOT EXISTS feedback (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT NOT NULL,
    disease  TEXT NOT NULL,
    drug     TEXT NOT NULL,
    vote     INTEGER NOT NULL CHECK (vote IN (-1, 1)),
    ts       TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_feedback_arm ON feedback(disease, drug);
"""


def _connect() -> sqlite3.Connection:
    """Open a connection (one per call — safe across Streamlit's threads)."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    return conn


# --------------------------------------------------------------------------- #
# Users
# --------------------------------------------------------------------------- #
def get_user(username: str) -> dict | None:
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE username = ?", (username,)
        ).fetchone()
    return dict(row) if row else None


def insert_user(
    username: str, name: str, salt: str, password_hash: str, role: str, created: str
) -> None:
    with _connect() as conn:
        conn.execute(
            "INSERT INTO users (username, name, salt, password_hash, role, created) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (username, name, salt, password_hash, role, created),
        )


def get_profile(username: str) -> dict:
    row = get_user(username)
    return json.loads(row["profile_json"]) if row else {}


def set_profile(username: str, profile: dict) -> None:
    with _connect() as conn:
        conn.execute(
            "UPDATE users SET profile_json = ? WHERE username = ?",
            (json.dumps(profile), username),
        )


def all_users() -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT username, name, role, created FROM users ORDER BY created"
        ).fetchall()
    return [dict(r) for r in rows]


# --------------------------------------------------------------------------- #
# Activity
# --------------------------------------------------------------------------- #
def insert_event(username: str, event_type: str, detail: dict, ts: str) -> None:
    with _connect() as conn:
        conn.execute(
            "INSERT INTO activity (username, type, detail_json, ts) VALUES (?, ?, ?, ?)",
            (username, event_type, json.dumps(detail), ts),
        )


# --------------------------------------------------------------------------- #
# Recommendation feedback (powers the RL bandit)
# --------------------------------------------------------------------------- #
def insert_feedback(username: str, disease: str, drug: str, vote: int, ts: str) -> None:
    with _connect() as conn:
        conn.execute(
            "INSERT INTO feedback (username, disease, drug, vote, ts) VALUES (?, ?, ?, ?, ?)",
            (username, disease, drug, vote, ts),
        )


def feedback_counts(disease: str) -> dict[str, tuple[int, int]]:
    """Per-drug (upvotes, downvotes) for a disease."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT drug, "
            "SUM(CASE WHEN vote = 1 THEN 1 ELSE 0 END) AS ups, "
            "SUM(CASE WHEN vote = -1 THEN 1 ELSE 0 END) AS downs "
            "FROM feedback WHERE disease = ? GROUP BY drug",
            (disease,),
        ).fetchall()
    return {r["drug"]: (int(r["ups"]), int(r["downs"])) for r in rows}


def fetch_events(username: str | None = None, limit: int = 500) -> list[dict]:
    query = "SELECT username, type, detail_json, ts FROM activity"
    params: tuple = ()
    if username:
        query += " WHERE username = ?"
        params = (username,)
    query += " ORDER BY ts DESC LIMIT ?"
    params += (limit,)
    with _connect() as conn:
        rows = conn.execute(query, params).fetchall()
    return [
        {
            "user": r["username"],
            "type": r["type"],
            "detail": json.loads(r["detail_json"]),
            "ts": r["ts"],
        }
        for r in rows
    ]
