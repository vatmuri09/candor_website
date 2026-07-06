"""Vercel entry point.

Vercel looks for a WSGI `app` in this file. We just point it at the Flask app.
On Vercel the filesystem is read-only except /tmp, so the working files (FAISS
banks, logs, agenda) go there; the durable copy lives in Postgres (see
session_store) and the Blob archive.
"""
import os
import sys

# Make the repo root importable (this file lives in /api).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("LOGS_DIR", "/tmp/logs")
os.environ.setdefault("DATA_DIR", "/tmp/data")

from src.main_flask import app  # noqa: E402

# Vercel's Python runtime serves this WSGI callable.
application = app
