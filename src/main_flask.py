"""Flask web app for the interview.

Each interview turn now runs to completion inside a single request and the
session state is saved to storage between turns (see session_store), so the app
can run on a serverless host like Vercel instead of a long-lived server.
"""

from flask import (Flask, request, jsonify, render_template, Response,
                   redirect, url_for, session as flask_session)
from flask_cors import CORS
import asyncio
import os
import uuid
import argparse
import time
import logging
import secrets
import json
from functools import wraps
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Must run before any src.* import: several modules (e.g. session_agenda.py)
# read env vars like LOGS_DIR/DATA_DIR at MODULE-IMPORT time, not lazily inside
# functions. Loading .env after those imports silently leaves them None unless
# the vars already exist as real shell/process env vars.
load_dotenv(override=True)

from src.utils.speech.text_to_speech import create_tts_engine
from src.utils.speech.speech_to_text import create_stt_engine
from src.interview_session.interview_session import InterviewSession
from src.utils.storage import session_archive, research_db, session_store

START_TIME = time.time()


class AppConfig:
    def __init__(self):
        self.host = "0.0.0.0"
        self.port = 5000
        self.debug = False
        self.restart = False
        self.max_turns = None
        self.additional_context_path = None


config = AppConfig()

TTS_PROVIDER = os.getenv('TTS_PROVIDER', 'openai')
TTS_VOICE = os.getenv('TTS_VOICE', 'alloy')
tts_engine = create_tts_engine(provider=TTS_PROVIDER, voice=TTS_VOICE)
stt_engine = create_stt_engine()

app = Flask(__name__, static_folder='web/static', template_folder='web/templates')
app.secret_key = os.getenv('FLASK_SECRET_KEY', secrets.token_hex(32))
CORS(app)

ADMIN_PASSWORD = os.getenv('ADMIN_PASSWORD')

# =============================================================================
# CONVERSATION TYPES
# =============================================================================

def _config_path(filename: str) -> str:
    disk_path = os.path.join(os.path.dirname(os.getenv('DATA_DIR', 'data')), 'configs', filename)
    if os.path.exists(disk_path):
        return disk_path
    return os.path.join('data', 'configs', filename)


CONVERSATION_TYPES = {
    "ai_workforce": {
        "label": "AI in the Workforce", "emoji": "🤖",
        "blurb": "A research interview about how you use AI tools in your day-to-day work.",
        "description": "Understanding the impact of AI in the workforce",
        "plan_file": "topics.json",
    },
    "career_story": {
        "label": "Career & Work Story", "emoji": "💼",
        "blurb": "Talk through your career journey, motivations, and where you're headed.",
        "description": "Exploring your career journey, work, and professional growth",
        "plan_file": "topics_career.json",
    },
    "life_background": {
        "label": "Life & Background", "emoji": "🌱",
        "blurb": "Share your story — roots, formative experiences, and reflections.",
        "description": "Exploring your life story, background, and personal experiences",
        "plan_file": "topics_life.json",
    },
    "custom": {
        "label": "Custom Topic", "emoji": "✨",
        "blurb": "Tell us what you'd like to be interviewed about and we'll take it from there.",
        "description": None,
        "plan_file": "topics_general.json",
    },
}

DEFAULT_CONVERSATION_TYPE = "ai_workforce"


# =============================================================================
# ADMIN-CREATED INTERVIEWS (custom question sets shared via a link)
# =============================================================================
# Small JSON registry mapping a share token -> {title, plan_file, created_at,
# questions}. The actual topic/subtopic plan for each is a normal topics.json
# file, generated from the admin's question list and saved next to the presets.

_REGISTRY_PATH = os.path.join('data', 'configs', '_custom_interviews.json')


def _registry_use_db() -> bool:
    return research_db.is_configured()


def _load_registry() -> dict:
    """Local-disk fallback registry, used only when no Postgres is configured
    (e.g. running on your laptop). On Vercel research_db is always used instead —
    the filesystem there is read-only/ephemeral, see _materialize_custom_plan."""
    if os.path.exists(_REGISTRY_PATH):
        try:
            with open(_REGISTRY_PATH, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _save_registry(registry: dict) -> None:
    os.makedirs(os.path.dirname(_REGISTRY_PATH), exist_ok=True)
    with open(_REGISTRY_PATH, 'w', encoding='utf-8') as f:
        json.dump(registry, f, indent=2)


def create_custom_interview(title: str, opening_question: str) -> str:
    """Save an admin-authored topic + opening question. Returns the share token.

    Stored as a 1-element list in the `questions` column so the existing
    schema/storage helpers don't need to change; `_materialize_custom_plan`
    uses that single string as the deterministic opener while the rest of the
    plan comes from the generic topic-agnostic scaffold (topics_general.json)
    — the same one the participant-facing "Custom Topic" preset uses — so the
    agenda/exploration agents probe live from there instead of the admin
    having to write out every question.
    """
    token = uuid.uuid4().hex[:10]
    questions = [opening_question]
    if _registry_use_db():
        research_db.save_custom_interview(token, title, questions)
    else:
        registry = _load_registry()
        registry[token] = {"title": title, "questions": questions,
                           "created_at": time.time()}
        _save_registry(registry)
    return token


def get_custom_interview(token: str) -> Optional[dict]:
    if _registry_use_db():
        return research_db.get_custom_interview(token)
    return _load_registry().get(token)


def list_custom_interviews() -> list:
    if _registry_use_db():
        return research_db.list_custom_interviews()
    registry = _load_registry()
    interviews = [{"token": t, **v} for t, v in registry.items()]
    interviews.sort(key=lambda x: x.get("created_at", 0), reverse=True)
    return interviews


def _materialize_custom_plan(link_token: str, plan_path: str) -> bool:
    """(Re)write the topics.json-shaped plan file for a link-based interview.

    The plan is the same generic, topic-agnostic scaffold used by the
    participant-facing "Custom Topic" preset (topics_general.json) — its
    subtopics apply to any subject and drive the agenda/exploration agents'
    live probing. The only admin input baked in is the opening question,
    which overwrites the scaffold's first subtopic so it's what the
    interviewer deterministically opens on (see
    Interviewer._first_planned_subtopic).

    Called on EVERY session build (not just creation) because the source of
    truth is the registry (Postgres or local JSON), and the plan file itself is
    working state — on Vercel it lives under DATA_DIR (/tmp), which does not
    persist across serverless invocations. Cheap (one lookup + one file write),
    so re-materializing every turn is fine.
    """
    entry = get_custom_interview(link_token)
    if entry is None:
        return False
    opening_question = (entry.get("questions") or [""])[0]
    with open(_config_path("topics_general.json"), 'r', encoding='utf-8') as f:
        plan = json.load(f)
    if plan and plan[0].get("subtopics"):
        plan[0]["subtopics"][0] = opening_question
    os.makedirs(os.path.dirname(plan_path), exist_ok=True)
    with open(plan_path, 'w', encoding='utf-8') as f:
        json.dump(plan, f, indent=2)
    return True


def resolve_conversation(conversation_type, custom_description=None, link_token=None):
    """Return (interview_description, interview_plan_path) for a chosen type."""
    if conversation_type == 'link' and link_token:
        entry = get_custom_interview(link_token)
        if entry:
            # Must be written per-session (see _materialize_custom_plan), so this
            # has to live under DATA_DIR like session_store/context_store do —
            # _config_path's fallback for a not-yet-written file resolves to the
            # read-only deployment bundle on Vercel.
            plan_path = os.path.join(os.getenv('DATA_DIR', 'data'), 'configs',
                                     f"custom_{link_token}.json")
            return entry["title"], plan_path
        # Fall through to default if the token is unknown/stale.
    preset = CONVERSATION_TYPES.get(conversation_type or DEFAULT_CONVERSATION_TYPE,
                                    CONVERSATION_TYPES[DEFAULT_CONVERSATION_TYPE])
    description = preset["description"]
    if description is None:  # custom
        description = (custom_description or "").strip() or "A topic of the participant's choosing"
    return description, _config_path(preset["plan_file"])


# =============================================================================
# LOGGING
# =============================================================================

if not app.debug:
    os.makedirs(os.getenv('LOGS_DIR', 'logs'), exist_ok=True)
    file_handler = RotatingFileHandler(
        os.path.join(os.getenv('LOGS_DIR', 'logs'), 'flask_app.log'),
        maxBytes=10485760, backupCount=5)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'))
    file_handler.setLevel(logging.INFO)
    app.logger.addHandler(file_handler)
    app.logger.setLevel(logging.INFO)


# =============================================================================
# BUILDING AND RUNNING A SESSION
# =============================================================================

def session_meta(user_id, conversation_type, custom_description, link_token=None):
    """The bit of info we need to rebuild a session on the next request."""
    description, plan_path = resolve_conversation(conversation_type, custom_description, link_token)
    return {
        "user_id": user_id,
        "conversation_type": conversation_type,
        "custom_description": custom_description,
        "interview_description": description,
        "interview_plan_path": plan_path,
        "link_token": link_token,
    }


def build_session(meta: dict) -> InterviewSession:
    """Construct a fresh InterviewSession from stored meta.

    Called on EVERY turn (see session_store.load), not just at creation — so for
    a link-based interview we must re-materialize the plan file here every time,
    not just once. See _materialize_custom_plan.
    """
    if meta.get("link_token"):
        _materialize_custom_plan(meta["link_token"], meta["interview_plan_path"])
    return InterviewSession(
        interaction_mode='api',
        user_config={"user_id": meta["user_id"], "enable_voice": False, "restart": False},
        interview_config={
            "enable_voice": False,
            "interview_description": meta["interview_description"],
            "interview_plan_path": meta["interview_plan_path"],
            "interview_evaluation": os.getenv('COMPLETION_METRIC'),
            "additional_context_path": config.additional_context_path,
            "initial_user_portrait_path": os.getenv('USER_PORTRAIT_PATH'),
        },
        max_turns=config.max_turns,
    )


def _as_messages(texts) -> list:
    """Wrap interviewer reply strings as the message dicts the frontend expects."""
    out = []
    for t in texts:
        out.append({"id": f"msg_{uuid.uuid4().hex}", "role": "Interviewer",
                    "content": t, "timestamp": time.time()})
    return out


def _load_or_response(token):
    """Load a session for a request.

    Returns (session, meta, None) on success, or (None, None, response) where
    response is a ready-to-return (json, status) tuple:
      - 503 when storage is transiently unavailable (client should retry)
      - 400 when the token genuinely isn't a live session
    """
    try:
        loaded = session_store.load(token, build_session)
    except research_db.StorageUnavailable:
        return None, None, (jsonify({
            'success': False, 'retryable': True,
            'error': 'Storage temporarily unavailable, please retry'}), 503)
    if loaded is None:
        return None, None, (jsonify({
            'success': False, 'error': 'Invalid or expired session'}), 400)
    session, meta = loaded
    return session, meta, None


def _status_for(session) -> str:
    return "in_progress" if session.session_in_progress else "completed"


def _finalize(token, session, meta) -> None:
    """When a session ends: archive its files and record its metadata (once)."""
    try:
        data_dir = os.path.join(os.getenv('DATA_DIR', 'data'), meta["user_id"])
        session_archive.archive_session(
            user_id=meta["user_id"], session_id=getattr(session, 'session_id', 0),
            extra_dirs=[data_dir] if os.path.isdir(data_dir) else None)
    except Exception as e:
        app.logger.warning(f"archive failed for {meta['user_id']}: {e}")
    try:
        record_session_metadata(session, meta)
    except Exception as e:
        app.logger.warning(f"record_session failed for {meta['user_id']}: {e}")


def record_session_metadata(session, meta) -> None:
    """Write one session's metadata row to the research DB."""
    if not research_db.is_configured():
        return
    monitor = getattr(session, 'engagement_monitor', None)
    closer = getattr(session, 'conversation_closer', None)
    interviewer = getattr(session, '_interviewer', None)
    probe_monitor = getattr(session, 'probe_quality_monitor', None)
    chat = getattr(session, 'chat_history', []) or []
    num_user = len([m for m in chat if getattr(m, 'role', None) == 'User'])
    num_interviewer = len([m for m in chat if getattr(m, 'role', None) == 'Interviewer'])
    transcript = [{"role": getattr(m, 'role', None), "content": getattr(m, 'content', '')}
                  for m in chat]
    research_db.record_session(
        user_id=meta["user_id"], session_id=getattr(session, 'session_id', 0),
        conversation_type=meta.get("conversation_type"),
        topic=meta.get("interview_description"),
        num_user_turns=num_user, num_interviewer_turns=num_interviewer,
        end_reason=getattr(closer, 'state', None),
        engagement_stats=monitor.stats() if monitor is not None else None,
        closer_stats=closer.stats() if closer is not None else None,
        guardrail_stats=getattr(interviewer, 'guardrail_stats', None),
        probe_quality_stats=probe_monitor.stats if probe_monitor is not None else None,
        # NOTE: the DB column is still named context_bias; we now record the
        # researched/approved background context there.
        context_bias=([getattr(session, 'retrieved_context')]
                      if getattr(session, 'retrieved_context', None) else None),
        transcript=transcript)


# =============================================================================
# PUBLIC PAGES
# =============================================================================

@app.route('/')
def index():
    types = [{"key": key, **{k: v for k, v in cfg.items()
                             if k not in ("description", "plan_file")}}
             for key, cfg in CONVERSATION_TYPES.items()]
    return render_template('select.html', conversation_types=types)


@app.route('/chat')
def unified_chat():
    return render_template('chat.html')


# =============================================================================
# INTERVIEW API
# =============================================================================

@app.route('/api/start-session', methods=['POST'])
def start_session():
    """Start a new interview: build the session and research background context.

    The interview does NOT open yet — we return a researched briefing for the
    participant to approve (see /api/approve-context), then that approved context
    seeds the conversation.
    """
    data = request.get_json(silent=True) or {}
    conversation_type = data.get('conversation_type') or DEFAULT_CONVERSATION_TYPE
    custom_description = data.get('custom_description')
    link_token = data.get('link_token')
    user_id = (data.get('user_id') or '').strip() or f"web_{uuid.uuid4().hex[:16]}"

    token = str(uuid.uuid4())
    meta = session_meta(user_id, conversation_type, custom_description, link_token)
    session = build_session(meta)

    try:
        context = asyncio.run(session.research_context())
    except Exception as e:
        app.logger.warning(f"context research failed for {user_id}: {e}")
        context = {"topic": meta["interview_description"], "context": "",
                   "sources": [], "ok": False, "error": str(e)}
    session_store.save(token, session, meta, [], "awaiting_context")

    app.logger.info(f"Session started (awaiting context approval): {token} | "
                    f"user {user_id} | type {conversation_type}")
    return jsonify({
        'success': True, 'session_token': token,
        'session_id': session.session_id, 'user_id': user_id,
        'conversation_type': conversation_type, 'was_existing': False,
        'needs_context_approval': True,
        'interview_description': meta["interview_description"],
        'context': context,
    })


@app.route('/api/research-context', methods=['POST'])
def research_context():
    """Re-run topic research for an existing (not-yet-started) session.

    Powers the popup's "Search again" button. Bypasses the per-topic cache so the
    operator can get a fresh briefing.
    """
    data = request.json or {}
    token = data.get('session_token')
    session, meta, err = _load_or_response(token)
    if err is not None:
        return err
    try:
        result = asyncio.run(
            session.context_research_agent.research_topic(meta["interview_description"]))
        context = result.to_dict()
        context["from_cache"] = False
        session.retrieved_context = context
    except Exception as e:
        app.logger.warning(f"re-research failed for {meta['user_id']}: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500
    session_store.save(token, session, meta, [], "awaiting_context")
    return jsonify({'success': True, 'context': context})


@app.route('/api/approve-context', methods=['POST'])
def approve_context():
    """Approve (optionally edited) background context and open the interview."""
    data = request.json or {}
    token = data.get('session_token')
    approved_text = (data.get('context') or '').strip()

    session, meta, err = _load_or_response(token)
    if err is not None:
        return err

    # Idempotent: if the interview already opened, just replay the outbox.
    if session.chat_history:
        return jsonify({'success': True, 'messages': [], 'already_started': True})

    if not approved_text:
        approved_text = (session.retrieved_context or {}).get('context', '')

    opening = asyncio.run(session.start(approved_context=approved_text))
    outbox = _as_messages(opening)
    status = _status_for(session)
    session_store.save(token, session, meta, outbox, status)

    app.logger.info(f"Context approved, interview opened: {token} | user {meta['user_id']}")
    return jsonify({'success': True, 'messages': outbox})


@app.route('/api/send-message', methods=['POST'])
def send_message():
    """Run one interview turn for the given user message."""
    data = request.json or {}
    token = data.get('session_token')
    user_message = data.get('message')

    session, meta, err = _load_or_response(token)
    if err is not None:
        return err

    if not session.session_in_progress:
        return jsonify({'success': False, 'error': 'Session has ended',
                        'session_completed': True}), 400

    replies = asyncio.run(session.run_one_turn(user_message))
    outbox = _as_messages(replies)
    status = _status_for(session)
    if status == "completed":
        _finalize(token, session, meta)
    session_store.save(token, session, meta, outbox, status)

    return jsonify({'success': True, 'messages': outbox,
                    'session_completed': status == "completed"})


@app.route('/api/get-messages', methods=['GET'])
def get_messages():
    """Poll for interviewer messages produced by the last turn."""
    token = request.args.get('session_token')
    try:
        result = session_store.fetch_messages(token)
    except research_db.StorageUnavailable:
        # Transient DB blip on a poll — keep the session alive and let the client
        # try again on its next tick rather than declaring the interview dead.
        return jsonify({'success': True, 'messages': [], 'session_active': True,
                        'transient': True})
    if result is None:
        return jsonify({'success': False, 'error': 'Invalid or expired session'}), 400

    messages, status = result
    done = status == "completed"
    if done:
        end_id = f"system_end_{token}"
        if not any(m.get('id') == end_id for m in messages):
            messages.append({
                'id': end_id, 'role': 'Interviewer',
                'content': "Session has been completed! Thank you for your participation "
                           "in our interview! Your responses have been recorded!",
                'timestamp': time.time()})

    return jsonify({'success': True, 'messages': messages,
                    'session_active': not done, 'session_completed': done})


@app.route('/api/acknowledge-messages', methods=['POST'])
def acknowledge_messages():
    # Messages are cleared from the outbox as soon as they're fetched, so there's
    # nothing to do here. Kept so the existing frontend keeps working.
    return jsonify({'success': True})


@app.route('/api/send-voice', methods=['POST'])
def send_voice():
    """Transcribe an audio answer and run it through as a normal turn."""
    token = request.form.get('session_token')
    audio_file = request.files.get('audio')
    if not audio_file:
        return jsonify({'success': False, 'error': 'No audio file provided'}), 400
    if stt_engine is None:
        return jsonify({'success': False, 'error': 'Voice input is not enabled'}), 503

    session, meta, err = _load_or_response(token)
    if err is not None:
        return err

    tmp = Path(os.getenv('DATA_DIR', 'data')) / f"temp_audio_{uuid.uuid4().hex}.wav"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    audio_file.save(tmp)
    try:
        text = stt_engine.transcribe(str(tmp))
    finally:
        tmp.unlink(missing_ok=True)

    replies = asyncio.run(session.run_one_turn(text))
    outbox = _as_messages(replies)
    status = _status_for(session)
    if status == "completed":
        _finalize(token, session, meta)
    session_store.save(token, session, meta, outbox, status)

    return jsonify({'success': True, 'transcribed_text': text, 'messages': outbox})


@app.route('/api/get-voice-response', methods=['GET'])
def get_voice_response():
    """Synthesize speech for a piece of text passed by the client."""
    text = request.args.get('text')
    if not text:
        return jsonify({'success': False, 'error': 'text required'}), 400
    if tts_engine is None:
        return jsonify({'success': False, 'error': 'TTS not enabled'}), 503
    out_path = Path(os.getenv('DATA_DIR', 'data')) / f"temp_speech_{uuid.uuid4().hex}.mp3"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        tts_engine.text_to_speech(text=text, output_path=str(out_path))
        audio = out_path.read_bytes()
    finally:
        out_path.unlink(missing_ok=True)
    return Response(audio, mimetype='audio/mpeg')


@app.route('/api/end-session', methods=['POST'])
def end_session():
    """End the interview early at the user's request."""
    data = request.json or {}
    token = data.get('session_token')
    session, meta, err = _load_or_response(token)
    if err is not None:
        return err

    session.end_session()
    _finalize(token, session, meta)
    session_store.save(token, session, meta, [], "completed")

    return jsonify({'success': True, 'session_id': session.session_id,
                    'user_id': meta["user_id"]})


# =============================================================================
# POST-INTERVIEW SURVEY
# =============================================================================

_BATTERY_CACHE = None


def _load_battery() -> dict:
    global _BATTERY_CACHE
    if _BATTERY_CACHE is not None:
        return _BATTERY_CACHE
    # Must fall back to the bundled repo path like _config_path does: on
    # Vercel, DATA_DIR is /tmp/... (read/write but ephemeral) and nothing ever
    # writes likert_battery.json there — the file only ships in the deployed,
    # read-only repo bundle at data/configs/likert_battery.json. Using
    # DATA_DIR directly (with no fallback) silently produced an empty battery
    # in production, so the survey never rendered (showSurvey() treats zero
    # items as "no survey" and skips straight to finishAfterSurvey()).
    path = os.getenv('LIKERT_BATTERY_PATH', _config_path('likert_battery.json'))
    try:
        with open(path, 'r', encoding='utf-8') as f:
            _BATTERY_CACHE = json.load(f)
    except Exception as e:
        app.logger.warning(f"Could not load Likert battery from {path}: {e}")
        _BATTERY_CACHE = {"title": "", "scale": {}, "likert_items": [], "open_items": []}
    return _BATTERY_CACHE


@app.route('/api/survey/battery', methods=['GET'])
def survey_battery():
    return jsonify({'success': True, 'battery': _load_battery()})


@app.route('/api/survey/submit', methods=['POST'])
def survey_submit():
    """Store the post-interview Likert + open-ended answers."""
    data = request.json or {}
    token = data.get('session_token')
    session, meta, err = _load_or_response(token)
    if err is not None:
        return err
    session_id = getattr(session, 'session_id', 0)

    battery = _load_battery()
    scale = battery.get('scale', {})
    scale_min, scale_max = scale.get('min'), scale.get('max')
    labels = scale.get('labels', {})
    likert_in = data.get('likert', {}) or {}
    open_in = data.get('open', {}) or {}

    likert_rows, open_rows = [], []
    for item in battery.get('likert_items', []):
        key = item.get('key')
        if key not in likert_in or likert_in[key] in (None, ''):
            continue
        try:
            val = int(likert_in[key])
        except (TypeError, ValueError):
            continue
        likert_rows.append({'item_key': key, 'item_text': item.get('text'),
                            'scale_min': scale_min, 'scale_max': scale_max,
                            'response': val, 'response_label': labels.get(str(val))})
    for item in battery.get('open_items', []):
        key = item.get('key')
        text = (open_in.get(key) or '').strip()
        if text:
            open_rows.append({'item_key': key, 'item_text': item.get('text'), 'response': text})

    try:
        record_session_metadata(session, meta)
        research_db.record_likert(meta["user_id"], session_id, likert_rows)
        research_db.record_open(meta["user_id"], session_id, open_rows)
    except Exception as e:
        app.logger.warning(f"survey DB write failed for {meta['user_id']}: {e}")

    try:
        logs_dir = os.path.join(os.getenv('LOGS_DIR', 'logs'), meta["user_id"])
        os.makedirs(logs_dir, exist_ok=True)
        with open(os.path.join(logs_dir, f'survey_session_{session_id}.json'), 'w',
                  encoding='utf-8') as f:
            json.dump({'likert': likert_rows, 'open': open_rows,
                       'submitted_at': time.time()}, f, indent=2)
        _finalize(token, session, meta)  # re-archive so the survey is in the zip
    except Exception as e:
        app.logger.warning(f"survey file write failed for {meta['user_id']}: {e}")

    return jsonify({'success': True,
                    'recorded': {'likert': len(likert_rows), 'open': len(open_rows)}})


# =============================================================================
# ADMIN — view stored conversations (password protected)
# =============================================================================

def admin_required(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if not ADMIN_PASSWORD:
            # TEMP: no ADMIN_PASSWORD set -> admin area open to anyone with the URL.
            return fn(*args, **kwargs)
        if not flask_session.get('is_admin'):
            return redirect(url_for('admin_login'))
        return fn(*args, **kwargs)
    return wrapper


@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    if not ADMIN_PASSWORD:
        return redirect(url_for('admin_home'))
    error = None
    if request.method == 'POST':
        if request.form.get('password') == ADMIN_PASSWORD:
            flask_session['is_admin'] = True
            return redirect(url_for('admin_home'))
        error = "Wrong password."
    return render_template('admin_login.html', error=error)


@app.route('/admin/logout')
def admin_logout():
    flask_session.pop('is_admin', None)
    return redirect(url_for('admin_login'))


# Guardrail counters whose non-zero presence means a regeneration attempt
# could not fix the problem (the respondent-facing turn still tripped it).
_SEVERE_GUARDRAIL_KEYS = (
    "regeneration_failed", "near_duplicate_regen_failed", "depth_cap_regen_failed",
)
# Counters that mean *something* fired but was successfully corrected.
_FLAGGED_GUARDRAIL_KEYS = (
    "affirmation", "closing", "midsentence_service", "stance", "advice", "no_question",
    "near_duplicate", "depth_cap_triggered", "repeated_leadin",
)


def _worst_severity(guardrail_stats: Optional[dict], probe_quality_stats: Optional[dict]) -> str:
    """One-word severity label for the admin session list (SPEC.md priority #4)."""
    guardrail_stats = guardrail_stats or {}
    probe_quality_stats = probe_quality_stats or {}
    if any(guardrail_stats.get(k, 0) for k in _SEVERE_GUARDRAIL_KEYS):
        return "severe"
    if probe_quality_stats.get("flat_streaks_flagged", 0):
        return "flat-probing"
    if any(guardrail_stats.get(k, 0) for k in _FLAGGED_GUARDRAIL_KEYS):
        return "flagged"
    if not guardrail_stats and not probe_quality_stats:
        return "unknown"
    return "clean"


@app.route('/admin')
@admin_required
def admin_home():
    sessions = research_db.list_sessions()
    for s in sessions:
        s["severity"] = _worst_severity(s.get("guardrail_stats"), s.get("probe_quality_stats"))
    return render_template('admin.html', sessions=sessions)


@app.route('/admin/interviews', methods=['GET', 'POST'])
@admin_required
def admin_interviews():
    error = None
    new_link = None
    if request.method == 'POST':
        title = (request.form.get('title') or '').strip()
        opening_question = (request.form.get('opening_question') or '').strip()
        if not title or not opening_question:
            error = "Give the interview a topic and an opening question."
        else:
            token = create_custom_interview(title, opening_question)
            new_link = url_for('interview_link', token=token, _external=True)

    interviews = list_custom_interviews()
    for it in interviews:
        it["link"] = url_for('interview_link', token=it["token"], _external=True)

    return render_template('admin_interviews.html', interviews=interviews,
                           error=error, new_link=new_link)


@app.route('/i/<token>')
def interview_link(token):
    """Public entry point for an admin-created, shareable interview link."""
    entry = get_custom_interview(token)
    if entry is None:
        return render_template('link_start.html', title=None, token=token), 404
    return render_template('link_start.html', title=entry["title"], token=token)


@app.route('/admin/session/<int:row_id>')
@admin_required
def admin_session(row_id):
    detail = research_db.get_session_detail(row_id)
    if detail is None:
        return "Session not found.", 404
    return render_template('admin_session.html', **detail)


# =============================================================================
# HEALTH
# =============================================================================

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'uptime_seconds': round(time.time() - START_TIME, 2),
                    'db': research_db.is_configured(), 'tts_provider': TTS_PROVIDER})


# =============================================================================
# MAIN (local dev only; on Vercel the WSGI app is imported directly)
# =============================================================================

def parse_arguments():
    parser = argparse.ArgumentParser(description='Flask Interview Session Web Application')
    parser.add_argument('--port', type=int, default=5000)
    parser.add_argument('--host', type=str, default='0.0.0.0')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--additional_context_path', default=None)
    parser.add_argument('--max_turns', type=int, default=None)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    config.host = args.host
    config.port = args.port
    config.debug = args.debug
    config.max_turns = args.max_turns
    config.additional_context_path = args.additional_context_path

    print(f"Interview server on http://{config.host}:{config.port}  (db={research_db.is_configured()})")
    app.run(host=config.host, port=config.port, debug=config.debug,
            use_reloader=False, threaded=True)
