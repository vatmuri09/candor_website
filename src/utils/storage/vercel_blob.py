"""
Vercel Blob upload backend.

Replaces Google Drive as the destination for finished-interview zip archives.
Uses the Vercel Blob REST API directly (a single PUT with the store's
read-write token) so we don't pull in an extra SDK — `requests` is already a
dependency.

Env:
  BLOB_READ_WRITE_TOKEN   -> the store's read/write token (Vercel dashboard >
                             Storage > your Blob store > ".env.local" tab, or
                             `vercel env pull`). Required.
  VERCEL_BLOB_PREFIX      -> optional key prefix/folder inside the store
                             (default "interviews").
  VERCEL_BLOB_API_VERSION -> optional override of the x-api-version header
                             (default "7").

Docs: https://vercel.com/docs/storage/vercel-blob  (REST usage mirrors the
`@vercel/blob` `put()` call).
"""
import io
import logging
import os
from typing import Optional

import requests

logger = logging.getLogger("vercel_blob")

_BASE_URL = "https://blob.vercel-storage.com"


def _token() -> str:
    return os.getenv("BLOB_READ_WRITE_TOKEN", "").strip()


def is_configured() -> bool:
    """True if the Blob store token is present."""
    return bool(_token())


def _pathname(name: str) -> str:
    prefix = os.getenv("VERCEL_BLOB_PREFIX", "interviews").strip().strip("/")
    return f"{prefix}/{name}" if prefix else name


def upload(name: str, buf: io.BytesIO, content_type: str = "application/zip") -> Optional[str]:
    """Upload an in-memory buffer to Vercel Blob. Returns the public URL or None.

    Best-effort: logs and returns None on any failure so callers never break a
    finished session over an archival hiccup (data is still on local disk).
    """
    token = _token()
    if not token:
        return None

    pathname = _pathname(name)
    url = f"{_BASE_URL}/{pathname}"
    data = buf.getvalue() if isinstance(buf, io.BytesIO) else buf
    headers = {
        "authorization": f"Bearer {token}",
        "x-api-version": os.getenv("VERCEL_BLOB_API_VERSION", "7"),
        "x-content-type": content_type,
        # Keep the exact key we asked for (don't append a random suffix) so the
        # archive is addressable by session; keys are already unique (timestamp).
        "x-add-random-suffix": "0",
    }
    try:
        resp = requests.put(url, data=data, headers=headers, timeout=60)
        resp.raise_for_status()
        blob_url = None
        try:
            blob_url = resp.json().get("url")
        except Exception:
            pass
        logger.info(f"Uploaded interview archive '{pathname}' to Vercel Blob ({blob_url})")
        return blob_url
    except Exception as e:
        logger.warning(f"Vercel Blob upload of '{pathname}' failed (data still saved locally): {e}")
        return None
