# Deploying the interview web app on Vercel

Each interview **turn now runs to completion inside a single request**, and the
session is saved to **Vercel Postgres** between turns (there's no long-running
server holding it in memory). That makes the app fit Vercel's serverless model.
Working files (FAISS banks, agenda, logs) go to `/tmp`; the durable copy lives in
Postgres, and finished interviews are archived to Vercel Blob.

The user flow: visitor lands on `/` → picks a **conversation type** → chats with
the AI interviewer. The OpenAI key and all storage stay server-side. Admins read
past conversations at `/admin` (password protected).

---

## 1. Push the code to GitHub

```bash
# from the project root
git remote add origin https://github.com/<your-org-or-user>/<repo>.git
git push -u origin main
```

---

## 2. Deploy on Vercel

1. Create the storage first: in the Vercel dashboard → **Storage** →
   - **Postgres** (Neon) — create a database. Vercel adds `POSTGRES_URL` to the
     project automatically when you link it.
   - **Blob** — create a store and copy its `BLOB_READ_WRITE_TOKEN`.
2. **Add New… → Project**, import the GitHub repo. Vercel detects
   [`vercel.json`](vercel.json) and the Python function at
   [`api/index.py`](api/index.py); no build settings needed. Pick the **Pro**
   plan so functions can run up to `maxDuration: 300`.
3. Set the environment variables (Project → Settings → Environment Variables):

   | Variable | Value |
   | --- | --- |
   | `OPENAI_API_KEY` | your OpenAI key (**required**) |
   | `POSTGRES_URL` | from the linked Postgres store (**required** — this is where live sessions + transcripts are stored) |
   | `BLOB_READ_WRITE_TOKEN` | from the Blob store (for zip archives) |
   | `STORAGE_BACKEND` | `vercel` |
   | `ADMIN_PASSWORD` | any password — gates `/admin` |
   | `FLASK_SECRET_KEY` | any long random string |
   | `LOGS_DIR` | `/tmp/logs` |
   | `DATA_DIR` | `/tmp/data` |
   | `MODEL_NAME` | e.g. `gpt-4.1-mini`, or a GPT-5 model like `gpt-5-mini` |
   | `EMBEDDING_BACKEND` | `openai` (keeps the function small — no torch) |

   Optional model overrides (`AGENDA_MANAGER_MODEL_NAME`,
   `EXPLORATION_PLANNER_MODEL_NAME`, etc.) default to `MODEL_NAME`.
4. **Deploy.** When it's live, open the URL — you'll land on the conversation
   picker. `<url>/health` returns `{"status": "healthy", "db": true, ...}` once
   Postgres is wired up.
5. Visit `<url>/admin`, sign in with `ADMIN_PASSWORD`, and you'll see each
   finished interview with its transcript, Likert answers, and agent stats.

> **Note:** the whole app is served by the single Python function in
> `api/index.py`; `vercel.json` rewrites every path to it. The Postgres tables
> (`interview_sessions`, `likert_responses`, `open_responses`, `web_sessions`)
> are created automatically on first use.

---

## 3. (Optional) Archive finished interviews to Google Drive

Each finished interview is uploaded as a timestamped `.zip`. Two auth options —
**OAuth is recommended** for a personal / edu account, because uploaded files are
owned by *your* account and use its real storage quota. (A service account has
**no storage quota** and can only upload into a *Shared Drive*, so it fails with
`storageQuotaExceeded` when pointed at a normal My Drive folder.)

Common setup:
- In Google Cloud Console → **APIs & Services** → enable the **Google Drive API**.
- In Drive, create a folder; its id is the last part of the URL:
  `drive.google.com/drive/folders/`**`<GDRIVE_FOLDER_ID>`**.

### Option A — OAuth (recommended)
1. **APIs & Services → Credentials → Create credentials → OAuth client ID.**
   If prompted, configure the consent screen (**External**, add your email as a
   **Test user**). Application type: **Desktop app**. Download the client secret JSON.
2. On your laptop, run the one-time helper (opens a browser — log in with the
   account whose Drive should own the files):
   ```bash
   python scripts/get_oauth_token.py /path/to/client_secret.json
   ```
3. It prints three values. In Render set them plus the folder id:
   - `GOOGLE_OAUTH_CLIENT_ID`, `GOOGLE_OAUTH_CLIENT_SECRET`, `GOOGLE_OAUTH_REFRESH_TOKEN`
   - `GDRIVE_FOLDER_ID`

### Option B — Service account (only with a Shared Drive)
1. **Credentials → Create credentials → Service account**; download its JSON key.
2. Create a **Shared Drive** (Team Drive), add the account's `client_email` as
   **Content manager**, and use a folder inside it for `GDRIVE_FOLDER_ID`.
3. In Render set `GOOGLE_SERVICE_ACCOUNT_JSON` (entire JSON) and `GDRIVE_FOLDER_ID`.

Either way: if the vars are unset or wrong, upload is skipped and data still
stays on disk — it never breaks an interview. Verify a setup with
`python scripts/test_drive_upload.py <key.json> <folder_id>` (service account) or
by finishing a test interview.

---

## 4. Adding or editing conversation types

Presets live in `CONVERSATION_TYPES` in [`src/main_flask.py`](src/main_flask.py).
Each points at a topic-plan JSON in `data/configs/` (same shape as `topics.json`).
Add an entry + a JSON file, redeploy, and the new card appears on the landing
page automatically. `custom` lets users type their own topic (no file needed).

---

## Local run

```bash
pip install -r requirements.txt
cp .env_sample .env   # then set OPENAI_API_KEY and the dirs
python -m src.main_flask --port 8080
# open http://localhost:8080
```
