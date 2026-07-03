"""
Research database (Vercel Postgres / Neon).

Structured, queryable home for the data we actually want to model later:
per-session metadata and the post-interview Likert battery. The zip archives
(full transcripts/logs) still go to Vercel Blob; this DB is the tidy layer for
"predict the respondent's answers from the conversation".

Connection (any one of these, checked in order):
  POSTGRES_URL            -> Vercel Postgres pooled URL (preferred on Vercel)
  POSTGRES_PRISMA_URL     -> Vercel Postgres pooled URL (alt name)
  DATABASE_URL            -> generic Postgres URL (local / other hosts)

If none is set, every function here is a no-op, so the app runs fine with no DB.

Schema is created on first use (idempotent). Two tables:
  interview_sessions  -> one row per finished session (metadata + agent stats)
  likert_responses    -> one row per Likert item answered (tidy/long format)
  open_responses      -> one row per open-ended free-text answer
"""
import json
import logging
import os
import threading
from datetime import datetime
from typing import List, Optional

logger = logging.getLogger("research_db")

_engine = None
_engine_lock = threading.Lock()
_schema_ready = False


def _database_url() -> Optional[str]:
    url = (os.getenv("POSTGRES_URL") or os.getenv("POSTGRES_PRISMA_URL")
           or os.getenv("DATABASE_URL") or "").strip()
    if not url:
        return None
    # SQLAlchemy needs the postgresql(+driver) scheme, not the bare "postgres://".
    if url.startswith("postgres://"):
        url = "postgresql://" + url[len("postgres://"):]
    return url


def is_configured() -> bool:
    return _database_url() is not None


def _get_engine():
    global _engine
    if _engine is not None:
        return _engine
    url = _database_url()
    if not url:
        return None
    with _engine_lock:
        if _engine is None:
            from sqlalchemy import create_engine
            # pool_pre_ping guards against dropped serverless-Postgres connections.
            _engine = create_engine(url, pool_pre_ping=True, pool_recycle=280,
                                    future=True)
    return _engine


_DDL = [
    """
    CREATE TABLE IF NOT EXISTS interview_sessions (
        id                    BIGSERIAL PRIMARY KEY,
        user_id               TEXT NOT NULL,
        session_id            INTEGER NOT NULL,
        conversation_type     TEXT,
        topic                 TEXT,
        started_at            TIMESTAMPTZ,
        ended_at              TIMESTAMPTZ,
        num_user_turns        INTEGER,
        num_interviewer_turns INTEGER,
        end_reason            TEXT,
        engagement_stats      JSONB,
        closer_stats          JSONB,
        context_bias          JSONB,
        archive_url           TEXT,
        created_at            TIMESTAMPTZ NOT NULL DEFAULT now(),
        UNIQUE (user_id, session_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS likert_responses (
        id             BIGSERIAL PRIMARY KEY,
        user_id        TEXT NOT NULL,
        session_id     INTEGER NOT NULL,
        item_key       TEXT NOT NULL,
        item_text      TEXT,
        scale_min      INTEGER,
        scale_max      INTEGER,
        response       INTEGER,
        response_label TEXT,
        created_at     TIMESTAMPTZ NOT NULL DEFAULT now()
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS open_responses (
        id          BIGSERIAL PRIMARY KEY,
        user_id     TEXT NOT NULL,
        session_id  INTEGER NOT NULL,
        item_key    TEXT NOT NULL,
        item_text   TEXT,
        response    TEXT,
        created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_likert_user_session ON likert_responses (user_id, session_id)",
    "CREATE INDEX IF NOT EXISTS idx_open_user_session ON open_responses (user_id, session_id)",
]


def _ensure_schema(engine) -> bool:
    global _schema_ready
    if _schema_ready:
        return True
    try:
        from sqlalchemy import text
        with engine.begin() as conn:
            for stmt in _DDL:
                conn.execute(text(stmt))
        _schema_ready = True
        return True
    except Exception as e:
        logger.warning(f"research_db schema init failed: {e}")
        return False


def record_session(user_id: str, session_id, *, conversation_type: str = None,
                   topic: str = None, started_at: datetime = None,
                   ended_at: datetime = None, num_user_turns: int = None,
                   num_interviewer_turns: int = None, end_reason: str = None,
                   engagement_stats: dict = None, closer_stats: dict = None,
                   context_bias: list = None, archive_url: str = None) -> None:
    """Upsert one session's metadata. Best-effort; never raises."""
    engine = _get_engine()
    if engine is None or not _ensure_schema(engine):
        return
    try:
        from sqlalchemy import text
        stmt = text("""
            INSERT INTO interview_sessions
                (user_id, session_id, conversation_type, topic, started_at, ended_at,
                 num_user_turns, num_interviewer_turns, end_reason,
                 engagement_stats, closer_stats, context_bias, archive_url)
            VALUES
                (:user_id, :session_id, :conversation_type, :topic, :started_at, :ended_at,
                 :num_user_turns, :num_interviewer_turns, :end_reason,
                 CAST(:engagement_stats AS JSONB), CAST(:closer_stats AS JSONB),
                 CAST(:context_bias AS JSONB), :archive_url)
            ON CONFLICT (user_id, session_id) DO UPDATE SET
                conversation_type = EXCLUDED.conversation_type,
                topic = EXCLUDED.topic,
                started_at = EXCLUDED.started_at,
                ended_at = EXCLUDED.ended_at,
                num_user_turns = EXCLUDED.num_user_turns,
                num_interviewer_turns = EXCLUDED.num_interviewer_turns,
                end_reason = EXCLUDED.end_reason,
                engagement_stats = EXCLUDED.engagement_stats,
                closer_stats = EXCLUDED.closer_stats,
                context_bias = EXCLUDED.context_bias,
                archive_url = EXCLUDED.archive_url
        """)
        with engine.begin() as conn:
            conn.execute(stmt, {
                "user_id": user_id, "session_id": int(session_id),
                "conversation_type": conversation_type, "topic": topic,
                "started_at": started_at, "ended_at": ended_at,
                "num_user_turns": num_user_turns,
                "num_interviewer_turns": num_interviewer_turns,
                "end_reason": end_reason,
                "engagement_stats": json.dumps(engagement_stats) if engagement_stats is not None else None,
                "closer_stats": json.dumps(closer_stats) if closer_stats is not None else None,
                "context_bias": json.dumps(context_bias) if context_bias is not None else None,
                "archive_url": archive_url,
            })
    except Exception as e:
        logger.warning(f"record_session failed for {user_id}/{session_id}: {e}")


def record_likert(user_id: str, session_id, responses: List[dict]) -> None:
    """Insert a batch of Likert answers. Each dict: item_key, item_text,
    scale_min, scale_max, response, response_label. Best-effort; never raises."""
    engine = _get_engine()
    if engine is None or not _ensure_schema(engine) or not responses:
        return
    try:
        from sqlalchemy import text
        stmt = text("""
            INSERT INTO likert_responses
                (user_id, session_id, item_key, item_text, scale_min, scale_max,
                 response, response_label)
            VALUES
                (:user_id, :session_id, :item_key, :item_text, :scale_min, :scale_max,
                 :response, :response_label)
        """)
        rows = [{
            "user_id": user_id, "session_id": int(session_id),
            "item_key": r.get("item_key"), "item_text": r.get("item_text"),
            "scale_min": r.get("scale_min"), "scale_max": r.get("scale_max"),
            "response": r.get("response"), "response_label": r.get("response_label"),
        } for r in responses]
        with engine.begin() as conn:
            conn.execute(stmt, rows)
    except Exception as e:
        logger.warning(f"record_likert failed for {user_id}/{session_id}: {e}")


def record_open(user_id: str, session_id, responses: List[dict]) -> None:
    """Insert open-ended free-text answers. Each dict: item_key, item_text,
    response. Best-effort; never raises."""
    engine = _get_engine()
    if engine is None or not _ensure_schema(engine) or not responses:
        return
    try:
        from sqlalchemy import text
        stmt = text("""
            INSERT INTO open_responses (user_id, session_id, item_key, item_text, response)
            VALUES (:user_id, :session_id, :item_key, :item_text, :response)
        """)
        rows = [{
            "user_id": user_id, "session_id": int(session_id),
            "item_key": r.get("item_key"), "item_text": r.get("item_text"),
            "response": r.get("response"),
        } for r in responses]
        with engine.begin() as conn:
            conn.execute(stmt, rows)
    except Exception as e:
        logger.warning(f"record_open failed for {user_id}/{session_id}: {e}")
