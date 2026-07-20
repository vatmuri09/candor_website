"""Postgres storage for session metadata and post-interview survey answers.

The full transcripts/zips go to Vercel Blob; this is the structured table layer.
Connection URL comes from POSTGRES_URL, POSTGRES_PRISMA_URL, or DATABASE_URL (in
that order). If none is set, everything here quietly does nothing so the app still
runs. Tables are created on first use: interview_sessions, likert_responses, and
open_responses.
"""
import json
import logging
import os
import threading
from datetime import datetime
from typing import List, Optional

logger = logging.getLogger("research_db")


class StorageUnavailable(Exception):
    """Raised when a live-session read fails for a transient reason (dropped
    serverless-Postgres connection, timeout, etc.) as opposed to the row simply
    not existing. Callers should treat this as retryable, NOT as "session gone."
    """


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
        transcript            JSONB,
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
    # Live web sessions: the working state we reload between turns so the app can
    # run on serverless (no in-memory session dict). One row per active session.
    """
    CREATE TABLE IF NOT EXISTS web_sessions (
        token       TEXT PRIMARY KEY,
        meta        JSONB,
        state       JSONB,
        files       BYTEA,
        outbox      JSONB NOT NULL DEFAULT '[]',
        status      TEXT NOT NULL DEFAULT 'in_progress',
        updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
    )
    """,
    # Approved background context, cached per topic so it can be reused across
    # conversations. One row per topic key (see context_store).
    """
    CREATE TABLE IF NOT EXISTS topic_context (
        topic_key   TEXT PRIMARY KEY,
        topic       TEXT,
        context     JSONB NOT NULL,
        approved    BOOLEAN NOT NULL DEFAULT false,
        updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
    )
    """,
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
                   context_bias: list = None, transcript: list = None,
                   archive_url: str = None) -> None:
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
                 engagement_stats, closer_stats, context_bias, transcript, archive_url)
            VALUES
                (:user_id, :session_id, :conversation_type, :topic, :started_at, :ended_at,
                 :num_user_turns, :num_interviewer_turns, :end_reason,
                 CAST(:engagement_stats AS JSONB), CAST(:closer_stats AS JSONB),
                 CAST(:context_bias AS JSONB), CAST(:transcript AS JSONB), :archive_url)
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
                transcript = EXCLUDED.transcript,
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
                "transcript": json.dumps(transcript) if transcript is not None else None,
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


# ---- live web sessions (working state kept between turns) ----

def save_web_session(token: str, meta: dict, state: dict, files: bytes,
                     outbox: list, status: str) -> bool:
    """Store/replace a live session's working state. Returns True on success."""
    engine = _get_engine()
    if engine is None or not _ensure_schema(engine):
        return False
    try:
        from sqlalchemy import text
        stmt = text("""
            INSERT INTO web_sessions (token, meta, state, files, outbox, status, updated_at)
            VALUES (:token, CAST(:meta AS JSONB), CAST(:state AS JSONB), :files,
                    CAST(:outbox AS JSONB), :status, now())
            ON CONFLICT (token) DO UPDATE SET
                meta = EXCLUDED.meta, state = EXCLUDED.state, files = EXCLUDED.files,
                outbox = EXCLUDED.outbox, status = EXCLUDED.status, updated_at = now()
        """)
        with engine.begin() as conn:
            conn.execute(stmt, {"token": token, "meta": json.dumps(meta),
                                "state": json.dumps(state), "files": files,
                                "outbox": json.dumps(outbox), "status": status})
        return True
    except Exception as e:
        logger.warning(f"save_web_session failed for {token}: {e}")
        return False


def read_and_clear_outbox(token: str):
    """Return (messages, status) for a token and empty its outbox. Cheap poll path.

    Returns None only when the row genuinely does not exist. Raises
    StorageUnavailable on a transient DB failure so the caller can keep the
    session alive and retry instead of declaring it expired.
    """
    engine = _get_engine()
    if engine is None or not _ensure_schema(engine):
        return None
    try:
        from sqlalchemy import text
        with engine.begin() as conn:
            row = conn.execute(
                text("SELECT outbox, status FROM web_sessions WHERE token = :t"),
                {"t": token},
            ).fetchone()
            if row is None:
                return None
            outbox, status = row
            if outbox:
                conn.execute(
                    text("UPDATE web_sessions SET outbox = '[]' WHERE token = :t"),
                    {"t": token},
                )
        return (outbox or []), status
    except Exception as e:
        logger.warning(f"read_and_clear_outbox failed for {token}: {e}")
        raise StorageUnavailable(str(e)) from e


def get_web_session(token: str):
    """Return (meta, state, files) for a token, or None if not found.

    Raises StorageUnavailable on a transient DB failure (vs. None for a token
    that truly isn't there) so a turn isn't wrongly reported as expired.
    """
    engine = _get_engine()
    if engine is None or not _ensure_schema(engine):
        return None
    try:
        from sqlalchemy import text
        with engine.begin() as conn:
            row = conn.execute(
                text("SELECT meta, state, files FROM web_sessions WHERE token = :t"),
                {"t": token},
            ).fetchone()
        if row is None:
            return None
        meta, state, files = row
        files = bytes(files) if files is not None else b""
        return meta, state, files
    except Exception as e:
        logger.warning(f"get_web_session failed for {token}: {e}")
        raise StorageUnavailable(str(e)) from e


# ---- admin read helpers ----

def list_sessions(limit: int = 200) -> list:
    """All finished sessions, newest first, for the admin list page."""
    engine = _get_engine()
    if engine is None or not _ensure_schema(engine):
        return []
    try:
        from sqlalchemy import text
        with engine.begin() as conn:
            rows = conn.execute(text("""
                SELECT id, user_id, session_id, conversation_type, topic,
                       num_user_turns, end_reason, created_at
                FROM interview_sessions
                ORDER BY created_at DESC
                LIMIT :lim
            """), {"lim": limit}).mappings().all()
        return [dict(r) for r in rows]
    except Exception as e:
        logger.warning(f"list_sessions failed: {e}")
        return []


def get_session_detail(row_id: int):
    """One session's full record + its Likert / open answers, for the admin page."""
    engine = _get_engine()
    if engine is None or not _ensure_schema(engine):
        return None
    try:
        from sqlalchemy import text
        with engine.begin() as conn:
            session = conn.execute(
                text("SELECT * FROM interview_sessions WHERE id = :id"),
                {"id": row_id}).mappings().fetchone()
            if session is None:
                return None
            session = dict(session)
            likert = conn.execute(text("""
                SELECT item_text, response, response_label FROM likert_responses
                WHERE user_id = :u AND session_id = :s ORDER BY id
            """), {"u": session["user_id"], "s": session["session_id"]}).mappings().all()
            openr = conn.execute(text("""
                SELECT item_text, response FROM open_responses
                WHERE user_id = :u AND session_id = :s ORDER BY id
            """), {"u": session["user_id"], "s": session["session_id"]}).mappings().all()
        return {"session": session, "likert": [dict(r) for r in likert],
                "open": [dict(r) for r in openr]}
    except Exception as e:
        logger.warning(f"get_session_detail failed for {row_id}: {e}")
        return None


def delete_web_session(token: str) -> None:
    engine = _get_engine()
    if engine is None or not _ensure_schema(engine):
        return
    try:
        from sqlalchemy import text
        with engine.begin() as conn:
            conn.execute(text("DELETE FROM web_sessions WHERE token = :t"), {"t": token})
    except Exception as e:
        logger.warning(f"delete_web_session failed for {token}: {e}")


# ---- per-topic background context ----

def get_topic_context(topic_key: str):
    """Return the cached context dict for a topic key, or None."""
    engine = _get_engine()
    if engine is None or not _ensure_schema(engine):
        return None
    try:
        from sqlalchemy import text
        with engine.begin() as conn:
            row = conn.execute(
                text("SELECT context FROM topic_context WHERE topic_key = :k"),
                {"k": topic_key},
            ).fetchone()
        return row[0] if row is not None else None
    except Exception as e:
        logger.warning(f"get_topic_context failed for {topic_key}: {e}")
        return None


def save_topic_context(topic_key: str, topic: str, context: dict,
                       approved: bool = True) -> bool:
    """Store/replace the cached context for a topic key. Returns True on success."""
    engine = _get_engine()
    if engine is None or not _ensure_schema(engine):
        return False
    try:
        from sqlalchemy import text
        stmt = text("""
            INSERT INTO topic_context (topic_key, topic, context, approved, updated_at)
            VALUES (:k, :topic, CAST(:context AS JSONB), :approved, now())
            ON CONFLICT (topic_key) DO UPDATE SET
                topic = EXCLUDED.topic, context = EXCLUDED.context,
                approved = EXCLUDED.approved, updated_at = now()
        """)
        with engine.begin() as conn:
            conn.execute(stmt, {"k": topic_key, "topic": topic,
                                "context": json.dumps(context), "approved": approved})
        return True
    except Exception as e:
        logger.warning(f"save_topic_context failed for {topic_key}: {e}")
        return False
