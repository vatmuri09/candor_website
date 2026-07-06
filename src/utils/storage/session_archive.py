"""Zips up a finished interview and uploads it to storage.

STORAGE_BACKEND picks where it goes:
  auto   - Google Drive if set up, else Vercel Blob, else nothing (stays on disk)
  vercel - Vercel Blob only
  drive  - Google Drive only
  both   - every backend that's configured

Never raises, so a failed upload can't break a finished session.
"""
import io
import logging
import os
import threading
import zipfile
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from src.utils.storage import vercel_blob, drive_export

logger = logging.getLogger("session_archive")


def _zip_dir(src_dir: Path) -> Optional[io.BytesIO]:
    """Zip a directory into an in-memory buffer. Returns None if empty/missing."""
    if not src_dir.exists() or not any(src_dir.rglob("*")):
        return None
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in src_dir.rglob("*"):
            if path.is_file():
                zf.write(path, arcname=str(path.relative_to(src_dir)))
    buf.seek(0)
    return buf


def _build_archive(user_id: str, extra_dirs: Optional[list]) -> Optional[io.BytesIO]:
    """Zip the user's logs (+ any extra dirs) into a single buffer."""
    logs_dir = Path(os.getenv("LOGS_DIR", "logs")) / user_id
    buf = _zip_dir(logs_dir)

    for extra in (extra_dirs or []):
        extra_path = Path(extra)
        extra_buf = _zip_dir(extra_path)
        if extra_buf is None:
            continue
        if buf is None:
            buf = extra_buf
            continue
        merged = io.BytesIO()
        with zipfile.ZipFile(merged, "w", zipfile.ZIP_DEFLATED) as out:
            for source, prefix in ((buf, "logs"), (extra_buf, extra_path.name)):
                with zipfile.ZipFile(source, "r") as zin:
                    for item in zin.namelist():
                        out.writestr(f"{prefix}/{item}", zin.read(item))
        merged.seek(0)
        buf = merged
    return buf


def _selected_backends() -> List[str]:
    """Which backends to upload to, given env + what's actually configured."""
    mode = os.getenv("STORAGE_BACKEND", "auto").strip().lower()
    if mode == "vercel":
        return ["vercel"] if vercel_blob.is_configured() else []
    if mode == "drive":
        return ["drive"] if drive_export.is_configured() else []
    if mode == "both":
        return ([b for b, ok in (("vercel", vercel_blob.is_configured()),
                                 ("drive", drive_export.is_configured())) if ok])
    # auto: keep the existing setup — prefer Google Drive, fall back to Vercel Blob.
    if drive_export.is_configured():
        return ["drive"]
    if vercel_blob.is_configured():
        return ["vercel"]
    return []


def _do_upload(name: str, buf: io.BytesIO, backends: List[str]) -> None:
    for backend in backends:
        # Each backend consumes the buffer from the start.
        buf.seek(0)
        if backend == "vercel":
            vercel_blob.upload(name, buf)
        elif backend == "drive":
            drive_export._upload(name, buf)  # reuse Drive's uploader


def archive_session(user_id: str, session_id, extra_dirs: Optional[list] = None,
                    async_upload: bool = True) -> None:
    """Zip a user's interview data and upload it to the configured backend(s).

    Best-effort, never raises. No-op if no backend is configured.
    """
    backends = _selected_backends()
    if not backends:
        return
    try:
        buf = _build_archive(user_id, extra_dirs)
        if buf is None:
            logger.info(f"No interview files to archive for user {user_id}")
            return
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"{user_id}_session_{session_id}_{stamp}.zip"
        if async_upload:
            threading.Thread(target=_do_upload, args=(name, buf, backends), daemon=True).start()
        else:
            _do_upload(name, buf, backends)
    except Exception as e:
        logger.warning(f"archive_session failed for {user_id} (data still saved locally): {e}")
