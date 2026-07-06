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

from src.utils.speech.text_to_speech import create_tts_engine
from src.utils.speech.speech_to_text import create_stt_engine
from src.interview_session.interview_session import InterviewSession
from src.utils.storage import session_archive, research_db, session_store

load_dotenv(override=True)

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


def resolve_conversation(conversation_type, custom_description=None):
    """Return (interview_description, interview_plan_path) for a chosen type."""
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

def session_meta(user_id, conversation_type, custom_description):
    """The bit of info we need to rebuild a session on the next request."""
    description, plan_path = resolve_conversation(conversation_type, custom_description)
    return {
        "user_id": user_id,
        "conversation_type": conversation_type,
        "custom_description": custom_description,
        "interview_description": description,
        "interview_plan_path": plan_path,
    }


def build_session(meta: dict) -> InterviewSession:
    """Construct a fresh InterviewSession from stored meta."""
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
        context_bias=getattr(session, 'context_bias_reports', None),
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
    """Start a new interview: build the session, generate the opening question."""
    data = request.get_json(silent=True) or {}
    conversation_type = data.get('conversation_type') or DEFAULT_CONVERSATION_TYPE
    custom_description = data.get('custom_description')
    user_id = (data.get('user_id') or '').strip() or f"web_{uuid.uuid4().hex[:16]}"

    token = str(uuid.uuid4())
    meta = session_meta(user_id, conversation_type, custom_description)
    session = build_session(meta)

    opening = asyncio.run(session.start())
    outbox = _as_messages(opening)
    status = _status_for(session)
    session_store.save(token, session, meta, outbox, status)

    app.logger.info(f"Session started: {token} | user {user_id} | type {conversation_type}")
    return jsonify({
        'success': True, 'session_token': token,
        'session_id': session.session_id, 'user_id': user_id,
        'conversation_type': conversation_type, 'was_existing': False,
    })


@app.route('/api/send-message', methods=['POST'])
def send_message():
    """Run one interview turn for the given user message."""
    data = request.json or {}
    token = data.get('session_token')
    user_message = data.get('message')

    loaded = session_store.load(token, build_session)
    if loaded is None:
        return jsonify({'success': False, 'error': 'Invalid or expired session'}), 400
    session, meta = loaded

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
    result = session_store.fetch_messages(token)
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

    loaded = session_store.load(token, build_session)
    if loaded is None:
        return jsonify({'success': False, 'error': 'Invalid or expired session'}), 400
    session, meta = loaded

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
    loaded = session_store.load(token, build_session)
    if loaded is None:
        return jsonify({'success': False, 'error': 'Invalid or expired session'}), 400
    session, meta = loaded

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
    path = os.getenv('LIKERT_BATTERY_PATH',
                     os.path.join(os.getenv('DATA_DIR', 'data'), 'configs', 'likert_battery.json'))
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
    loaded = session_store.load(token, build_session)
    if loaded is None:
        return jsonify({'success': False, 'error': 'Invalid or expired session'}), 400
    session, meta = loaded
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
        if not flask_session.get('is_admin'):
            return redirect(url_for('admin_login'))
        return fn(*args, **kwargs)
    return wrapper


@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    if not ADMIN_PASSWORD:
        return "Admin area is not configured (set ADMIN_PASSWORD).", 503
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


@app.route('/admin')
@admin_required
def admin_home():
    sessions = research_db.list_sessions()
    return render_template('admin.html', sessions=sessions)


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
