"""
Best-effort Google Drive archival of finished interview sessions.

How it works:
  1. Create a Google Cloud *service account* and download its JSON key.
  2. In Google Drive, create a folder and share it with the service account's
     email (found in the JSON as "client_email"), giving it Editor access.
  3. Set two environment variables on the server:
       GOOGLE_SERVICE_ACCOUNT_JSON  -> the full JSON key (as a string) OR a path to it
       GDRIVE_FOLDER_ID             -> the id from the shared folder's URL
         (drive.google.com/drive/folders/<THIS_IS_THE_ID>)

If either variable is missing, or the google client libraries aren't installed,
archival is silently skipped — the interview data always remains on local disk,
so nothing here can break a running interview.
"""

import io
import os
import json
import zipfile
import logging
import threading
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_SCOPES = ["https://www.googleapis.com/auth/drive"]


def is_configured() -> bool:
    """True if the env vars needed for Drive upload are present."""
    return bool(os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON") and os.getenv("GDRIVE_FOLDER_ID"))


def _load_credentials():
    """Build service-account credentials from env. Returns None on any problem."""
    raw = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON", "").strip()
    if not raw:
        return None

    try:
        # Accept either the raw JSON string or a path to a JSON file.
        if raw.startswith("{"):
            info = json.loads(raw)
        elif os.path.exists(raw):
            with open(raw, "r", encoding="utf-8") as f:
                info = json.load(f)
        else:
            logger.warning("GOOGLE_SERVICE_ACCOUNT_JSON is neither JSON nor a valid path; skipping Drive export")
            return None

        from google.oauth2 import service_account  # lazy import
        return service_account.Credentials.from_service_account_info(info, scopes=_SCOPES)
    except Exception as e:
        logger.warning(f"Could not build Drive credentials: {e}")
        return None


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


def _upload(name: str, buf: io.BytesIO) -> None:
    """Upload an in-memory zip to the configured Drive folder."""
    creds = _load_credentials()
    folder_id = os.getenv("GDRIVE_FOLDER_ID", "").strip()
    if creds is None or not folder_id:
        return

    try:
        from googleapiclient.discovery import build  # lazy import
        from googleapiclient.http import MediaIoBaseUpload

        service = build("drive", "v3", credentials=creds, cache_discovery=False)
        media = MediaIoBaseUpload(buf, mimetype="application/zip", resumable=False)
        metadata = {"name": name, "parents": [folder_id]}
        created = service.files().create(
            body=metadata, media_body=media, fields="id",
            supportsAllDrives=True,
        ).execute()
        logger.info(f"Uploaded interview archive '{name}' to Drive (file id {created.get('id')})")
    except Exception as e:
        logger.warning(f"Drive upload of '{name}' failed (data still saved locally): {e}")


def archive_session(user_id: str, session_id, extra_dirs: Optional[list] = None,
                    async_upload: bool = True) -> None:
    """Zip a user's interview data and upload it to Drive. Best-effort, never raises.

    Args:
        user_id: the interview user id (its folder under LOGS_DIR is archived).
        session_id: session number, used only to name the archive.
        extra_dirs: optional extra directories to include (e.g. DATA_DIR/user_id).
        async_upload: run the upload on a background thread so callers don't block.
    """
    if not is_configured():
        return

    try:
        from datetime import datetime
        logs_dir = Path(os.getenv("LOGS_DIR", "logs")) / user_id
        buf = _zip_dir(logs_dir)

        # Fold in any extra directories under distinct prefixes.
        for extra in (extra_dirs or []):
            extra_path = Path(extra)
            extra_buf = _zip_dir(extra_path)
            if extra_buf is None:
                continue
            if buf is None:
                buf = extra_buf
            else:
                # Merge: append extra files under a subfolder in the same archive.
                merged = io.BytesIO()
                with zipfile.ZipFile(merged, "w", zipfile.ZIP_DEFLATED) as out:
                    for source, prefix in ((buf, "logs"), (extra_buf, extra_path.name)):
                        with zipfile.ZipFile(source, "r") as zin:
                            for item in zin.namelist():
                                out.writestr(f"{prefix}/{item}", zin.read(item))
                merged.seek(0)
                buf = merged

        if buf is None:
            logger.info(f"No interview files to archive for user {user_id}")
            return

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"{user_id}_session_{session_id}_{stamp}.zip"

        if async_upload:
            threading.Thread(target=_upload, args=(name, buf), daemon=True).start()
        else:
            _upload(name, buf)
    except Exception as e:
        logger.warning(f"archive_session failed for {user_id} (data still saved locally): {e}")
