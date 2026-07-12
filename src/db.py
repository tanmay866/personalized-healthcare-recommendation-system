"""
Database layer for user management, activity tracking and feedback.

Dialect-agnostic via SQLAlchemy Core: runs on SQLite by default (zero setup)
and on PostgreSQL — hosted or local — by setting one environment variable:

    DATABASE_URL=postgresql://user:pass@host:5432/dbname   # e.g. Neon/Supabase
    (unset)                                                # -> sqlite data/app.db

Nothing else in the codebase changes between backends: the schema is declared
once with SQLAlchemy Table objects (which render the right DDL per dialect)
and all queries go through the same expression API.

Schema
------
users(username PK, name, salt, password_hash, role, profile_json, created)
activity(id PK, username, type, detail_json, ts)
feedback(id PK, username, disease, drug, vote, ts)
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

from sqlalchemy import (
    CheckConstraint,
    Column,
    Engine,
    Index,
    Integer,
    MetaData,
    Table,
    Text,
    case,
    create_engine,
    func,
    select,
)

ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_SQLITE = f"sqlite:///{ROOT / 'data' / 'app.db'}"

metadata = MetaData()

users = Table(
    "users",
    metadata,
    Column("username", Text, primary_key=True),
    Column("name", Text, nullable=False),
    Column("salt", Text, nullable=False),
    Column("password_hash", Text, nullable=False),
    Column("role", Text, nullable=False, server_default="User"),
    Column("profile_json", Text, nullable=False, server_default="{}"),
    Column("created", Text, nullable=False),
)

activity = Table(
    "activity",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("username", Text, nullable=False),
    Column("type", Text, nullable=False),
    Column("detail_json", Text, nullable=False, server_default="{}"),
    Column("ts", Text, nullable=False),
    Index("idx_activity_user", "username"),
    Index("idx_activity_ts", "ts"),
)

feedback = Table(
    "feedback",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("username", Text, nullable=False),
    Column("disease", Text, nullable=False),
    Column("drug", Text, nullable=False),
    Column("vote", Integer, CheckConstraint("vote IN (-1, 1)"), nullable=False),
    Column("ts", Text, nullable=False),
    Index("idx_feedback_arm", "disease", "drug"),
)


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    """Create the engine once, honoring DATABASE_URL, and ensure the schema."""
    url = os.environ.get("DATABASE_URL", _DEFAULT_SQLITE)
    # Some providers (Heroku-style) hand out postgres:// which SQLAlchemy
    # no longer accepts — normalize it.
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql://", 1)

    kwargs = {"pool_pre_ping": True}
    if url.startswith("sqlite"):
        Path(ROOT / "data").mkdir(parents=True, exist_ok=True)
        # Streamlit runs scripts in worker threads; allow cross-thread use.
        kwargs["connect_args"] = {"check_same_thread": False}

    engine = create_engine(url, **kwargs)
    metadata.create_all(engine)
    return engine


# --------------------------------------------------------------------------- #
# Users
# --------------------------------------------------------------------------- #
def get_user(username: str) -> dict | None:
    with get_engine().connect() as conn:
        row = conn.execute(
            select(users).where(users.c.username == username)
        ).mappings().first()
    return dict(row) if row else None


def insert_user(
    username: str, name: str, salt: str, password_hash: str, role: str, created: str
) -> None:
    with get_engine().begin() as conn:
        conn.execute(
            users.insert().values(
                username=username, name=name, salt=salt,
                password_hash=password_hash, role=role, created=created,
            )
        )


def get_profile(username: str) -> dict:
    row = get_user(username)
    return json.loads(row["profile_json"]) if row else {}


def set_profile(username: str, profile: dict) -> None:
    with get_engine().begin() as conn:
        conn.execute(
            users.update()
            .where(users.c.username == username)
            .values(profile_json=json.dumps(profile))
        )


def all_users() -> list[dict]:
    with get_engine().connect() as conn:
        rows = conn.execute(
            select(users.c.username, users.c.name, users.c.role, users.c.created)
            .order_by(users.c.created)
        ).mappings().all()
    return [dict(r) for r in rows]


# --------------------------------------------------------------------------- #
# Activity
# --------------------------------------------------------------------------- #
def insert_event(username: str, event_type: str, detail: dict, ts: str) -> None:
    with get_engine().begin() as conn:
        conn.execute(
            activity.insert().values(
                username=username, type=event_type,
                detail_json=json.dumps(detail), ts=ts,
            )
        )


def fetch_events(username: str | None = None, limit: int = 500) -> list[dict]:
    q = select(activity.c.username, activity.c.type, activity.c.detail_json, activity.c.ts)
    if username:
        q = q.where(activity.c.username == username)
    q = q.order_by(activity.c.ts.desc()).limit(limit)
    with get_engine().connect() as conn:
        rows = conn.execute(q).mappings().all()
    return [
        {
            "user": r["username"],
            "type": r["type"],
            "detail": json.loads(r["detail_json"]),
            "ts": r["ts"],
        }
        for r in rows
    ]


# --------------------------------------------------------------------------- #
# Recommendation feedback (powers the RL bandit)
# --------------------------------------------------------------------------- #
def insert_feedback(username: str, disease: str, drug: str, vote: int, ts: str) -> None:
    with get_engine().begin() as conn:
        conn.execute(
            feedback.insert().values(
                username=username, disease=disease, drug=drug, vote=vote, ts=ts,
            )
        )


def feedback_counts(disease: str) -> dict[str, tuple[int, int]]:
    """Per-drug (upvotes, downvotes) for a disease."""
    ups = func.sum(case((feedback.c.vote == 1, 1), else_=0)).label("ups")
    downs = func.sum(case((feedback.c.vote == -1, 1), else_=0)).label("downs")
    q = (
        select(feedback.c.drug, ups, downs)
        .where(feedback.c.disease == disease)
        .group_by(feedback.c.drug)
    )
    with get_engine().connect() as conn:
        rows = conn.execute(q).mappings().all()
    return {r["drug"]: (int(r["ups"]), int(r["downs"])) for r in rows}
