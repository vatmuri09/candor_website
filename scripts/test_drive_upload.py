#!/usr/bin/env python3
"""
Verify that a service account can upload a file into a Drive folder.

Usage:
    python scripts/test_drive_upload.py /path/to/service_account.json <FOLDER_ID>

Prints SUCCESS with a file link, or the exact Drive API error so we know
whether we need to switch approaches (e.g. the "storageQuotaExceeded" error
that personal service accounts hit when writing into a My Drive folder).
"""
import io
import sys

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload

SCOPES = ["https://www.googleapis.com/auth/drive"]


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)

    key_path, folder_id = sys.argv[1], sys.argv[2]

    creds = service_account.Credentials.from_service_account_info(
        __import__("json").load(open(key_path)), scopes=SCOPES
    )
    print(f"Service account: {creds.service_account_email}")
    print(f"Target folder:   {folder_id}\n")

    service = build("drive", "v3", credentials=creds, cache_discovery=False)

    data = io.BytesIO(b"candor drive upload test")
    media = MediaIoBaseUpload(data, mimetype="text/plain", resumable=False)
    metadata = {"name": "candor_test_upload.txt", "parents": [folder_id]}

    try:
        created = service.files().create(
            body=metadata, media_body=media,
            fields="id, webViewLink", supportsAllDrives=True,
        ).execute()
        print("SUCCESS ✅  Uploaded test file.")
        print("  File id:  ", created.get("id"))
        print("  View:     ", created.get("webViewLink"))
        print("\nThe service-account + folder setup works. You can wire it into the app.")
    except Exception as e:
        print("FAILED ❌")
        print(f"  {type(e).__name__}: {e}")
        msg = str(e)
        if "storageQuota" in msg or "quota" in msg.lower():
            print("\n-> This is the service-account-has-no-storage problem.")
            print("   We'll switch to OAuth (files owned by your own account) instead.")


if __name__ == "__main__":
    main()
