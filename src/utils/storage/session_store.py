"""Save and reload a live interview between web requests.

The app used to keep every running interview in a memory dict and a background
thread. That doesn't work on a serverless host where each request is a fresh
process, so instead we snapshot a session after every turn and rebuild it on the
next request.

Two backends:
  - Postgres (via research_db) when POSTGRES_URL is set. This is what Vercel uses.
    We store the small JSON state plus a zip of the session's files (FAISS banks,
    agenda) in one row keyed by the session token.
  - Local disk otherwise. The bank/agenda files already live under LOGS_DIR /
    DATA_DIR, so we only need to keep the JSON state next to them. Handy for
    running the app on your laptop.
"""
import io
import json
import os
import zipfile
from pathlib import Path
from typing import Callable, Optional, Tuple

from src.utils.storage import research_db


def _logs_dir() -> Path:
    return Path(os.getenv("LOGS_DIR", "logs"))


def _data_dir() -> Path:
    return Path(os.getenv("DATA_DIR", "data"))


def _use_db() -> bool:
    return research_db.is_configured()


# ---- zipping the per-user files (only needed for the Postgres backend) ----

def _zip_user_files(user_id: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for root, prefix in ((_logs_dir() / user_id, "logs"),
                             (_data_dir() / user_id, "data")):
            if root.exists():
                for path in root.rglob("*"):
                    if path.is_file():
                        z.write(path, arcname=f"{prefix}/{path.relative_to(root)}")
    return buf.getvalue()


def _unzip_user_files(user_id: str, blob: bytes) -> None:
    if not blob:
        return
    roots = {"logs": _logs_dir() / user_id, "data": _data_dir() / user_id}
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        for name in z.namelist():
            top, _, rel = name.partition("/")
            root = roots.get(top)
            if root is None or not rel:
                continue
            dest = root / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(z.read(name))


# ---- local-disk fallback ----

def _local_dir(token: str) -> Path:
    return _data_dir() / "_web_sessions" / token


def _save_local(token: str, meta: dict, state: dict, outbox: list, status: str) -> None:
    d = _local_dir(token)
    d.mkdir(parents=True, exist_ok=True)
    (d / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    (d / "state.json").write_text(json.dumps(state), encoding="utf-8")
    (d / "outbox.json").write_text(json.dumps({"outbox": outbox, "status": status}),
                                   encoding="utf-8")


def _load_local(token: str) -> Optional[Tuple[dict, dict]]:
    d = _local_dir(token)
    if not (d / "state.json").exists():
        return None
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    state = json.loads((d / "state.json").read_text(encoding="utf-8"))
    return meta, state


# ---- public API ----

def save(token: str, session, meta: dict, outbox: list, status: str) -> None:
    """Snapshot a session after a turn. meta must let us rebuild it later."""
    state = session.to_state()  # this also writes the bank/agenda files to disk
    if _use_db():
        research_db.save_web_session(token, meta, state,
                                     _zip_user_files(meta["user_id"]), outbox, status)
    else:
        _save_local(token, meta, state, outbox, status)


def fetch_messages(token: str):
    """Return (messages, status) waiting for the client and clear them. Cheap poll."""
    if _use_db():
        return research_db.read_and_clear_outbox(token)
    d = _local_dir(token)
    path = d / "outbox.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    messages, status = data.get("outbox", []), data.get("status", "in_progress")
    if messages:
        data["outbox"] = []
        path.write_text(json.dumps(data), encoding="utf-8")
    return messages, status


def load(token: str, builder: Callable[[dict], "object"]):
    """Rebuild a session from its snapshot; returns (session, meta) or None.

    builder(meta) constructs a fresh InterviewSession; we call it after the files
    are back on disk so the banks and agenda load correctly, then restore the
    conversation state on top.
    """
    if _use_db():
        record = research_db.get_web_session(token)
        if record is None:
            return None
        meta, state, files = record
        _unzip_user_files(meta["user_id"], files)
    else:
        record = _load_local(token)
        if record is None:
            return None
        meta, state = record

    session = builder(meta)
    session.load_state(state)
    return session, meta


def delete(token: str) -> None:
    if _use_db():
        research_db.delete_web_session(token)
    else:
        d = _local_dir(token)
        for name in ("meta.json", "state.json"):
            (d / name).unlink(missing_ok=True)
