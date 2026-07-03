"""
Flask Web Application for Interview Session
Supports both text and voice input/output with authentication
"""

from flask import Flask, request, jsonify, render_template, Response, redirect, url_for
from flask_cors import CORS
import traceback
import asyncio
import threading
import os
import uuid
import argparse
import time
import logging
import secrets
import json
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Dict, Optional
from dotenv import load_dotenv

# Your backend imports
from src.utils.speech.text_to_speech import TextToSpeechBase, create_tts_engine
from src.utils.speech.audio_player import AudioPlayerBase, create_audio_player
from src.utils.speech.speech_to_text import create_stt_engine
from src.interview_session.interview_session import InterviewSession
from src.utils.storage import session_archive, research_db

load_dotenv(override=True)

# =============================================================================
# CONFIGURATION
# =============================================================================

SESSION_TIMEOUT_SECONDS = 3600  # 1 hour
START_TIME = time.time()

class AppConfig:
    """Application configuration"""
    def __init__(self):
        self.default_user_id = "web_user"
        self.host = "0.0.0.0"
        self.port = 5000
        self.debug = False
        self.restart = False
        self.max_turns = None
        self.additional_context_path = None

config = AppConfig()

# TTS/STT engines
TTS_PROVIDER = os.getenv('TTS_PROVIDER', 'openai')
TTS_VOICE = os.getenv('TTS_VOICE', 'alloy')
tts_engine: TextToSpeechBase = create_tts_engine(provider=TTS_PROVIDER, voice=TTS_VOICE)
stt_engine = create_stt_engine()

# =============================================================================
# FLASK APP SETUP
# =============================================================================

app = Flask(__name__,
            static_folder='web/static',
            template_folder='web/templates')

app.secret_key = os.getenv('FLASK_SECRET_KEY', secrets.token_hex(32))
CORS(app)

# =============================================================================
# CONVERSATION TYPES
# =============================================================================
# Each preset maps a user-facing choice to an interview description + a topic
# plan (a JSON file in data/configs). "custom" lets the visitor supply their
# own topic; it reuses a broad open-ended plan and swaps in their description.

def _config_path(filename: str) -> str:
    """Resolve a config file, preferring the mounted-disk copy if present."""
    disk_path = os.path.join(os.path.dirname(os.getenv('DATA_DIR', 'data')), 'configs', filename)
    if os.path.exists(disk_path):
        return disk_path
    return os.path.join('data', 'configs', filename)

CONVERSATION_TYPES = {
    "ai_workforce": {
        "label": "AI in the Workforce",
        "emoji": "🤖",
        "blurb": "A research interview about how you use AI tools in your day-to-day work.",
        "description": "Understanding the impact of AI in the workforce",
        "plan_file": "topics.json",
    },
    "career_story": {
        "label": "Career & Work Story",
        "emoji": "💼",
        "blurb": "Talk through your career journey, motivations, and where you're headed.",
        "description": "Exploring your career journey, work, and professional growth",
        "plan_file": "topics_career.json",
    },
    "life_background": {
        "label": "Life & Background",
        "emoji": "🌱",
        "blurb": "Share your story — roots, formative experiences, and reflections.",
        "description": "Exploring your life story, background, and personal experiences",
        "plan_file": "topics_life.json",
    },
    "custom": {
        "label": "Custom Topic",
        "emoji": "✨",
        "blurb": "Tell us what you'd like to be interviewed about and we'll take it from there.",
        "description": None,          # filled in from the user's text at start
        "plan_file": "topics_general.json",
    },
}

DEFAULT_CONVERSATION_TYPE = "ai_workforce"

def resolve_conversation(conversation_type: Optional[str], custom_description: Optional[str] = None):
    """Return (interview_description, interview_plan_path) for a chosen type."""
    preset = CONVERSATION_TYPES.get(conversation_type or DEFAULT_CONVERSATION_TYPE,
                                    CONVERSATION_TYPES[DEFAULT_CONVERSATION_TYPE])
    description = preset["description"]
    if description is None:  # custom
        description = (custom_description or "").strip() or "A topic of the participant's choosing"
    return description, _config_path(preset["plan_file"])

# =============================================================================
# LOGGING SETUP
# =============================================================================

if not app.debug:
    os.makedirs('logs', exist_ok=True)
    file_handler = RotatingFileHandler(
        'logs/flask_app.log',
        maxBytes=10485760,  # 10MB
        backupCount=10
    )
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
    ))
    file_handler.setLevel(logging.INFO)
    app.logger.addHandler(file_handler)
    app.logger.setLevel(logging.INFO)
    app.logger.info('Interview application startup')

# =============================================================================
# ASYNC EVENT LOOP MANAGEMENT
# =============================================================================

loop = asyncio.new_event_loop()

def start_background_loop(loop):
    """Run async event loop in background thread."""
    asyncio.set_event_loop(loop)
    loop.run_forever()

threading.Thread(target=start_background_loop, args=(loop,), daemon=True).start()

def run_async_task(coro):
    """Submit coroutine to background loop."""
    return asyncio.run_coroutine_threadsafe(coro, loop)

# =============================================================================
# SESSION MANAGEMENT
# =============================================================================

class SessionWrapper:
    def __init__(self, session_token: str, interview_session: InterviewSession,
                 user_id: str, conversation_type: Optional[str] = None):
        self.session_token = session_token
        self.interview_session = interview_session
        self.user_id = user_id
        self.conversation_type = conversation_type
        self.created_at = time.time()
        self.archived = False  # guard so we only export to Drive once

active_sessions: Dict[str, SessionWrapper] = {}
last_messages_by_session: Dict[str, Dict[str, str]] = {}
session_audio_cache: Dict[str, Dict[str, object]] = {}

def create_interview_session(user_id: str, conversation_type: Optional[str] = None,
                             custom_description: Optional[str] = None) -> tuple[InterviewSession, str]:
    """Create an interview session for an anonymous web user + chosen topic."""
    session_token = str(uuid.uuid4())

    interview_description, interview_plan_path = resolve_conversation(
        conversation_type, custom_description
    )

    interview_session = InterviewSession(
        interaction_mode='api',
        user_config={
            "user_id": user_id,
            "enable_voice": False,
            "restart": config.restart
        },
        interview_config={
            "enable_voice": False,
            "interview_description": interview_description,
            "interview_plan_path": interview_plan_path,
            "interview_evaluation": os.getenv('COMPLETION_METRIC'),
            "additional_context_path": config.additional_context_path,
            "initial_user_portrait_path": os.getenv('USER_PORTRAIT_PATH'),
        },
        max_turns=config.max_turns
    )
    
    wrapper = SessionWrapper(
        session_token=session_token,
        interview_session=interview_session,
        user_id=user_id,
        conversation_type=conversation_type,
    )
    active_sessions[session_token] = wrapper
    
    session_loop = asyncio.new_event_loop()
    def _start_loop(l):
        asyncio.set_event_loop(l)
        l.run_forever()
    t = threading.Thread(target=_start_loop, args=(session_loop,), daemon=True)
    t.start()

    wrapper.loop = session_loop
    wrapper.loop_thread = t
    asyncio.run_coroutine_threadsafe(interview_session.run(), session_loop)
    
    return interview_session, session_token

def get_session(session_token: str) -> Optional[InterviewSession]:
    wrapper = active_sessions.get(session_token)
    return wrapper.interview_session if wrapper is not None else None

def get_session_wrapper(session_token: str) -> Optional[SessionWrapper]:
    return active_sessions.get(session_token)

def archive_session_wrapper(wrapper: Optional[SessionWrapper]) -> None:
    """Best-effort export of a finished session's data to storage (once)."""
    if not wrapper or wrapper.archived:
        return
    wrapper.archived = True
    try:
        session = wrapper.interview_session
        data_dir = os.path.join(os.getenv('DATA_DIR', 'data'), wrapper.user_id)
        session_archive.archive_session(
            user_id=wrapper.user_id,
            session_id=getattr(session, 'session_id', 0),
            extra_dirs=[data_dir] if os.path.isdir(data_dir) else None,
        )
    except Exception as e:
        app.logger.warning(f"Session archive failed for {wrapper.user_id}: {e}")

    # Record structured session metadata to the research DB (best-effort).
    try:
        record_session_metadata(wrapper)
    except Exception as e:
        app.logger.warning(f"DB record_session failed for {wrapper.user_id}: {e}")


def record_session_metadata(wrapper: SessionWrapper) -> None:
    """Write one session's metadata row to the research DB (Vercel Postgres)."""
    if not research_db.is_configured():
        return
    session = wrapper.interview_session
    monitor = getattr(session, 'engagement_monitor', None)
    closer = getattr(session, 'conversation_closer', None)
    chat = getattr(session, 'chat_history', []) or []
    num_user = len([m for m in chat if getattr(m, 'role', None) == 'User'])
    num_interviewer = len([m for m in chat if getattr(m, 'role', None) == 'Interviewer'])
    research_db.record_session(
        user_id=wrapper.user_id,
        session_id=getattr(session, 'session_id', 0),
        conversation_type=wrapper.conversation_type,
        topic=getattr(session, '_interview_description', None),
        num_user_turns=num_user,
        num_interviewer_turns=num_interviewer,
        end_reason=getattr(closer, 'state', None),
        engagement_stats=monitor.stats() if monitor is not None else None,
        closer_stats=closer.stats() if closer is not None else None,
        context_bias=getattr(session, 'context_bias_reports', None),
    )

# =============================================================================
# PAGE ROUTES - PUBLIC (no login; the user just picks a conversation type)
# =============================================================================

@app.route('/')
def index():
    """Landing page: pick a conversation type, then start."""
    types = [
        {"key": key, **{k: v for k, v in cfg.items() if k != "description" and k != "plan_file"}}
        for key, cfg in CONVERSATION_TYPES.items()
    ]
    return render_template('select.html', conversation_types=types)

@app.route('/chat')
def unified_chat():
    """Unified chat interface. Session is created via /api/start-session."""
    return render_template('chat.html')

# =============================================================================
# API ENDPOINTS - PROTECTED (REQUIRE LOGIN)
# Public endpoints keyed by an opaque session_token (no login).
# =============================================================================

@app.route('/api/start-session', methods=['POST'])
def start_session():
    """Initialize a new interview session for an anonymous web visitor.

    Body (JSON, all optional):
        conversation_type: one of CONVERSATION_TYPES keys (default ai_workforce)
        custom_description: free text, used when conversation_type == "custom"
        user_id: pass an existing id to resume; otherwise a fresh one is minted
    """
    data = request.get_json(silent=True) or {}
    conversation_type = data.get('conversation_type') or DEFAULT_CONVERSATION_TYPE
    custom_description = data.get('custom_description')

    # Anonymous identity: reuse a client-supplied id (resume) or mint a new one.
    user_id = (data.get('user_id') or '').strip() or f"web_{uuid.uuid4().hex[:16]}"

    # If this user already has a live session, hand it back rather than duplicate.
    for token, wrapper in active_sessions.items():
        if wrapper.user_id == user_id and wrapper.interview_session.session_in_progress:
            app.logger.info(f"Returning existing session {token} for {user_id}")
            return jsonify({
                'success': True,
                'session_token': token,
                'session_id': wrapper.interview_session.session_id,
                'user_id': user_id,
                'conversation_type': wrapper.conversation_type,
                'message': 'Using existing session',
                'was_existing': True
            })

    interview_session, session_token = create_interview_session(
        user_id=user_id,
        conversation_type=conversation_type,
        custom_description=custom_description,
    )

    app.logger.info(f"Session created: {session_token} | user {user_id} | type {conversation_type}")
    print(f"[Session] Created session {session_token} for {user_id} (type: {conversation_type})")

    return jsonify({
        'success': True,
        'session_token': session_token,
        'session_id': interview_session.session_id,
        'user_id': user_id,
        'conversation_type': conversation_type,
        'message': 'Session started successfully',
        'was_existing': False
    })

@app.route('/api/send-message', methods=['POST'])
def send_message():
    """Send a text message to the interview session"""
    data = request.json
    session_token = data.get('session_token')
    user_message = data.get('message')

    session = get_session(session_token)
    if not session:
        return jsonify({
            'success': False,
            'error': 'Invalid or expired session'
        }), 400

    if not session.session_in_progress:
        return jsonify({
            'success': False,
            'error': 'Session has ended',
            'session_completed': True
        }), 400

    wrapper = get_session_wrapper(session_token)
    if wrapper and hasattr(wrapper, 'loop'):
        wrapper.loop.call_soon_threadsafe(wrapper.interview_session.user.add_user_message, user_message)
    else:
        session.user.add_user_message(user_message)

    bot_reply = wait_for_agent_response(session)

    return jsonify({
        'success': True,
        'message': 'Message sent successfully'
    })

@app.route('/api/send-voice', methods=['POST'])
def send_voice():
    """Send a voice message to the interview session"""
    session_token = request.form.get('session_token')
    audio_file = request.files.get('audio')

    if not audio_file:
        return jsonify({
            'success': False,
            'error': 'No audio file provided'
        }), 400

    session = get_session(session_token)
    if not session:
        return jsonify({
            'success': False,
            'error': 'Invalid or expired session'
        }), 400

    # Check if STT is available
    if stt_engine is None:
        return jsonify({
            'success': False,
            'error': 'Voice input is not enabled on this server (PyAudio not available)'
        }), 503

    temp_audio_path = Path(f"temp_audio_{uuid.uuid4().hex}.wav")
    audio_file.save(temp_audio_path)

    try:
        transcribed_text = transcribe_audio_to_text(temp_audio_path)

        wrapper = get_session_wrapper(session_token)
        if wrapper and hasattr(wrapper, 'loop'):
            wrapper.loop.call_soon_threadsafe(wrapper.interview_session.user.add_user_message, transcribed_text)
        else:
            session.user.add_user_message(transcribed_text)

        return jsonify({
            'success': True,
            'transcribed_text': transcribed_text,
            'message': 'Voice message processed successfully'
        })
    finally:
        if temp_audio_path.exists():
            temp_audio_path.unlink()

@app.route('/api/get-messages', methods=['GET'])
def get_messages():
    """Get new messages from the session (polling endpoint)"""
    session_token = request.args.get('session_token')

    session = get_session(session_token)
    if not session:
        print(f"[get_messages] Invalid session_token={session_token}")
        return jsonify({
            'success': False,
            'error': 'Invalid or expired session',
            'active_sessions_count': len(active_sessions)
        }), 400

    messages = []
    if session.user:
        if hasattr(session.user, 'get_new_messages'):
            messages = session.user.get_new_messages() or []
        elif hasattr(session.user, '_message_buffer'):
            lock = getattr(session.user, '_lock', None)
            if lock:
                with lock:
                    messages = list(getattr(session.user, '_message_buffer', []))
            else:
                messages = list(getattr(session.user, '_message_buffer', []))
        elif hasattr(session.user, 'get_and_clear_messages'):
            messages = session.user.get_and_clear_messages() or []

    is_session_done = session.session_completed
    
    if not is_session_done:
        current_turns = getattr(session, 'turns', 0)
        max_turns = getattr(session, 'max_turns', float('inf'))
        if current_turns is not None and max_turns is not None and current_turns >= max_turns:
            is_session_done = True
        elif not session.session_in_progress and len(session.chat_history) > 0:
            is_session_done = True

    if is_session_done:
        # Session finished on its own -> archive its data to Drive (best-effort, once).
        archive_session_wrapper(get_session_wrapper(session_token))
        end_msg_id = f"system_end_{session_token}"
        if not any(m.get('id') == end_msg_id for m in messages):
            messages.append({
                'id': end_msg_id,
                'role': 'Interviewer',
                'content': "Session has been completed! Thank you for your participation in our interview! Your responses have been recorded!",
                'timestamp': time.time()
            })

    return jsonify({
        'success': True,
        'messages': messages,
        'session_active': session.session_in_progress,
        'session_completed': is_session_done
    })

@app.route('/api/acknowledge-messages', methods=['POST'])
def acknowledge_messages():
    """Mark messages as acknowledged by the client"""
    data = request.json
    session_token = data.get('session_token')
    message_ids = data.get('message_ids', [])

    session = get_session(session_token)
    if not session:
        return jsonify({
            'success': False,
            'error': 'Invalid or expired session'
        }), 400

    if session.user and hasattr(session.user, '_message_buffer'):
        lock = getattr(session.user, '_lock', None)
        if lock:
            with lock:
                buffer = getattr(session.user, '_message_buffer', [])
                session.user._message_buffer = [
                    m for m in buffer 
                    if m.get('id') not in message_ids
                ]
        
    return jsonify({'success': True})

@app.route('/api/get-voice-response', methods=['GET'])
def get_voice_response():
    """Get the latest interviewer message as voice audio"""
    session_token = request.args.get('session_token')
    message_id = request.args.get('message_id')

    session = get_session(session_token)
    if not session:
        return jsonify({
            'success': False,
            'error': 'Invalid or expired session'
        }), 400

    if not message_id:
        return jsonify({
            'success': False,
            'error': 'message_id required'
        }), 400

    target_msg = None
    for m in session.chat_history:
        if hasattr(m, 'id') and m.id == message_id:
            target_msg = m
            break
    
    if not target_msg:
        return jsonify({
            'success': False,
            'error': 'Message not found'
        }), 404

    cache = session_audio_cache.setdefault(session_token, {})
    entry = cache.get(message_id)

    # Check cache entry status
    if isinstance(entry, dict):
        # New format: {'status': 'pending'|'ready'|'failed', 'data': bytes|None, 'error': str|None}
        status = entry.get('status')
        if status == 'ready' and entry.get('data'):
            return Response(entry['data'], mimetype='audio/mpeg')
        elif status == 'failed':
            return jsonify({
                'success': False,
                'error': f"TTS generation failed: {entry.get('error', 'Unknown error')}"
            }), 500
        elif status == 'pending':
            return ('', 202)  # Still generating
    elif isinstance(entry, (bytes, bytearray)):
        # Legacy format support
        return Response(entry, mimetype='audio/mpeg')
    elif entry == 'pending':
        # Legacy format support
        return ('', 202)

    # No cache entry exists - start generation
    cache[message_id] = {'status': 'pending', 'data': None, 'error': None, 'timestamp': time.time()}
    wrapper = get_session_wrapper(session_token)

    if not wrapper or not hasattr(wrapper, 'loop'):
        cache[message_id] = {
            'status': 'failed',
            'data': None,
            'error': 'Session loop not available'
        }
        return jsonify({
            'success': False,
            'error': 'Session not properly initialized'
        }), 500

    async def _synth_and_cache():
        try:
            text = target_msg.content
            app.logger.info(f"Starting TTS generation for message {message_id}")

            def _blocking_tts():
                out_path = Path(f"temp_speech_{uuid.uuid4().hex}.mp3")
                try:
                    tts_engine.text_to_speech(text=text, output_path=str(out_path))
                    data = out_path.read_bytes()
                    app.logger.info(f"TTS generated {len(data)} bytes for message {message_id}")
                    return data
                finally:
                    if out_path.exists():
                        out_path.unlink(missing_ok=True)

            audio_bytes = await wrapper.loop.run_in_executor(None, _blocking_tts)

            # Update cache with success
            cache[message_id] = {
                'status': 'ready',
                'data': audio_bytes,
                'error': None,
                'timestamp': time.time()
            }
            app.logger.info(f"TTS cached successfully for message {message_id}")

        except Exception as e:
            error_msg = str(e)
            app.logger.error(f"TTS error for message {message_id}: {error_msg}", exc_info=True)

            # Update cache with failure
            cache[message_id] = {
                'status': 'failed',
                'data': None,
                'error': error_msg,
                'timestamp': time.time()
            }

    # Schedule async TTS generation
    future = asyncio.run_coroutine_threadsafe(_synth_and_cache(), wrapper.loop)

    # Log if scheduling failed
    try:
        future.result(timeout=0.1)  # Quick check if it errored immediately
    except Exception:
        pass  # Will complete async

    return ('', 202)  # Tell client to poll again

@app.route('/api/end-session', methods=['POST'])
def end_session():
    """End the interview session - background tasks will complete gracefully"""
    data = request.json
    session_token = data.get('session_token')

    wrapper = get_session_wrapper(session_token)
    if not wrapper:
        return jsonify({
            'success': False,
            'error': 'Invalid or expired session'
        }), 400

    session = wrapper.interview_session

    # End the session (marks as not in progress, stops new tasks)
    session.end_session()

    # Archive the completed interview to Drive (best-effort, once).
    archive_session_wrapper(wrapper)

    app.logger.info(f"Session {session_token} ended by user, background tasks will complete")

    # Don't remove from active_sessions yet - let background tasks complete
    # They'll be cleaned up when session.run() completes or by timeout cleanup

    return jsonify({
        'success': True,
        'message': 'Session ending, background tasks will complete shortly',
        'session_id': session.session_id,
        'user_id': session.user_id
    })

# =============================================================================
# POST-INTERVIEW SURVEY (fixed validated Likert battery)
# =============================================================================

_BATTERY_CACHE = None


def _load_battery() -> dict:
    """Load the fixed post-interview battery from config (cached)."""
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
    """Return the fixed post-interview questionnaire for the frontend to render."""
    return jsonify({'success': True, 'battery': _load_battery()})


@app.route('/api/survey/submit', methods=['POST'])
def survey_submit():
    """Persist a respondent's post-interview Likert + open-ended answers.

    Body: { session_token, likert: {item_key: value, ...}, open: {item_key: text, ...} }
    Answers are written both to the research DB (tidy rows) and to the session's
    log dir (so they travel with the zip archive).
    """
    data = request.json or {}
    session_token = data.get('session_token')
    wrapper = get_session_wrapper(session_token)
    if not wrapper:
        return jsonify({'success': False, 'error': 'Invalid or expired session'}), 400

    session = wrapper.interview_session
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
        likert_rows.append({
            'item_key': key, 'item_text': item.get('text'),
            'scale_min': scale_min, 'scale_max': scale_max,
            'response': val, 'response_label': labels.get(str(val)),
        })
    for item in battery.get('open_items', []):
        key = item.get('key')
        text = (open_in.get(key) or '').strip()
        if not text:
            continue
        open_rows.append({'item_key': key, 'item_text': item.get('text'), 'response': text})

    # Persist to the research DB (best-effort).
    try:
        record_session_metadata(wrapper)  # ensure the parent session row exists
        research_db.record_likert(wrapper.user_id, session_id, likert_rows)
        research_db.record_open(wrapper.user_id, session_id, open_rows)
    except Exception as e:
        app.logger.warning(f"survey DB write failed for {wrapper.user_id}: {e}")

    # Persist to the session log dir so it travels with the zip archive.
    try:
        logs_dir = os.path.join(os.getenv('LOGS_DIR', 'logs'), wrapper.user_id)
        os.makedirs(logs_dir, exist_ok=True)
        with open(os.path.join(logs_dir, f'survey_session_{session_id}.json'), 'w', encoding='utf-8') as f:
            json.dump({'likert': likert_rows, 'open': open_rows,
                       'submitted_at': time.time()}, f, indent=2)
    except Exception as e:
        app.logger.warning(f"survey file write failed for {wrapper.user_id}: {e}")

    # Re-archive so the survey answers are included in the uploaded zip.
    try:
        wrapper.archived = False
        archive_session_wrapper(wrapper)
    except Exception as e:
        app.logger.warning(f"re-archive after survey failed for {wrapper.user_id}: {e}")

    return jsonify({'success': True, 'recorded': {'likert': len(likert_rows), 'open': len(open_rows)}})


@app.route('/api/session-status', methods=['GET'])
def session_status():
    """Get current session status including background task progress"""
    session_token = request.args.get('session_token')

    wrapper = get_session_wrapper(session_token)
    if not wrapper:
        return jsonify({
            'success': False,
            'error': 'Invalid or expired session'
        }), 400

    session = wrapper.interview_session

    # Get background task count if available
    background_tasks_count = 0
    if hasattr(session, '_background_tasks'):
        try:
            import asyncio
            # Try to get count safely
            if hasattr(session, '_background_tasks_lock'):
                # Can't acquire lock in sync context, just get len
                background_tasks_count = len(session._background_tasks)
        except:
            background_tasks_count = 0

    return jsonify({
        'success': True,
        'session_active': session.session_in_progress,
        'session_completed': session.session_completed,
        'background_tasks_remaining': background_tasks_count,
        'message_count': len(session.chat_history),
        'session_id': session.session_id,
        'user_id': session.user_id
    })

@app.route('/api/debug-session', methods=['GET'])
def debug_session():
    """Development-only: return session internals"""
    session_token = request.args.get('session_token')
    if not session_token:
        return jsonify({'success': False, 'error': 'session_token required'}), 400

    wrapper = get_session_wrapper(session_token)
    if not wrapper:
        return jsonify({'success': False, 'error': 'Invalid or expired session', 'active_sessions_count': len(active_sessions)}), 400

    session = wrapper.interview_session

    last_msgs = []
    for m in session.chat_history[-20:]:
        last_msgs.append({
            'id': getattr(m, 'id', None),
            'role': getattr(m, 'role', None),
            'content': getattr(m, 'content', None),
            'timestamp': getattr(m, 'timestamp', None).isoformat() if getattr(m, 'timestamp', None) else None,
        })

    user_buffer = []
    user = session.user
    if hasattr(user, '_message_buffer'):
        try:
            lock = getattr(user, '_lock', None)
            if lock:
                lock.acquire()
            user_buffer = list(getattr(user, '_message_buffer', []))
        finally:
            if lock:
                lock.release()

    return jsonify({
        'success': True,
        'session_id': session.session_id,
        'session_active': session.session_in_progress,
        'session_completed': session.session_completed,
        'message_count': len(session.chat_history),
        'chat_history': last_msgs,
        'user_buffer': user_buffer,
        'active_sessions_count': len(active_sessions)
    })

@app.route('/process_audio', methods=['POST'])
def process_audio():
    """Compatibility route for older speech_chat.html template"""
    session_token = request.form.get('session_token')
    audio_file = request.files.get('audio')

    if not audio_file:
        return jsonify({'success': False, 'error': 'No audio file provided'}), 400

    # Reuse the session's own user_id, or mint an anonymous one.
    existing = get_session_wrapper(session_token) if session_token else None
    user_id = existing.user_id if existing else f"web_{uuid.uuid4().hex[:16]}"

    if not session_token:
        interview_session, session_token = create_interview_session(user_id=user_id)
    else:
        interview_session = get_session(session_token)
        if not interview_session:
            interview_session, session_token = create_interview_session(user_id=user_id)

    temp_audio_path = Path(f"temp_audio_{uuid.uuid4().hex}.wav")
    audio_file.save(temp_audio_path)

    try:
        transcribed_text = transcribe_audio_to_text(temp_audio_path)
        wrapper = get_session_wrapper(session_token)
        if wrapper and hasattr(wrapper, 'loop'):
            wrapper.loop.call_soon_threadsafe(
                wrapper.interview_session.user.add_user_message, 
                transcribed_text
            )
        else:
            interview_session.user.add_user_message(transcribed_text)

        bot_reply = wait_for_agent_response(interview_session, timeout=15.0)
        
        last_messages_by_session[session_token] = {
            'user_message': transcribed_text,
            'bot_reply': bot_reply or ''
        }

        if bot_reply:
            out_path = Path(f"temp_speech_{uuid.uuid4().hex}.mp3")
            generate_speech_from_text(bot_reply, out_path)
            audio_bytes = out_path.read_bytes()
            out_path.unlink(missing_ok=True)
            return Response(audio_bytes, mimetype='audio/mpeg')

        return jsonify({
            'success': True, 
            'user_message': transcribed_text, 
            'bot_reply': bot_reply
        }), 200

    finally:
        if temp_audio_path.exists():
            temp_audio_path.unlink()

@app.route('/get_last_messages', methods=['GET'])
def get_last_messages():
    """Get last messages for session"""
    session_token = request.args.get('session_token')
    if not session_token:
        return jsonify({'success': False, 'error': 'session_token required'}), 400

    msgs = last_messages_by_session.get(session_token, {})
    return jsonify({
        'success': True,
        'user_message': msgs.get('user_message', ''),
        'bot_reply': msgs.get('bot_reply', '')
    })

# =============================================================================
# HEALTH CHECK - NOT PROTECTED (for monitoring)
# =============================================================================

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint - no login required for monitoring"""
    current_time = time.time()
    session_ages = []
    for wrapper in active_sessions.values():
        age_minutes = (current_time - wrapper.created_at) / 60
        session_ages.append(age_minutes)
    
    avg_age = sum(session_ages) / len(session_ages) if session_ages else 0
    
    return jsonify({
        'status': 'healthy',
        'active_sessions': len(active_sessions),
        'avg_session_age_minutes': round(avg_age, 2),
        'tts_provider': TTS_PROVIDER,
        'tts_voice': TTS_VOICE,
        'uptime_seconds': round(current_time - START_TIME, 2)
    })

# =============================================================================
# VOICE PROCESSING UTILITIES
# =============================================================================

def generate_speech_from_text(text: str, output_path: Path) -> Path:
    """Generate speech audio from text using TTS"""
    global tts_engine
    if tts_engine is None:
        raise NotImplementedError("TTS engine is not configured")
    os.makedirs(os.path.dirname(str(output_path)) or '.', exist_ok=True)
    result_path = tts_engine.text_to_speech(text=text, output_path=str(output_path))
    return Path(result_path)

def transcribe_audio_to_text(audio_path: Path) -> str:
    """Transcribe audio file to text using speech recognition"""
    global stt_engine
    return stt_engine.transcribe(str(audio_path))

def wait_for_agent_response(session, timeout: float = 60.0, poll_interval: float = 0.5):
    """Wait for the Interviewer/Agent to produce an output"""
    start_time = None
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            start_time = None
        else:
            start_time = loop.time()
    except Exception:
        start_time = None

    elapsed = 0.0
    import time as _time

    while elapsed < timeout:
        try:
            msgs = []
            if hasattr(session.user, 'get_and_clear_messages'):
                msgs = session.user.get_and_clear_messages() or []
            interviewer_msgs = [m for m in msgs if m.get('role') == 'Interviewer']
            if interviewer_msgs:
                return interviewer_msgs[-1].get('content')
        except Exception:
            pass
        _time.sleep(poll_interval)
        elapsed += poll_interval
    return None

# =============================================================================
# SESSION CLEANUP
# =============================================================================

def cleanup_old_sessions():
    """Remove sessions older than timeout threshold and clean stale audio cache"""
    current_time = time.time()
    to_remove = []

    for token, wrapper in list(active_sessions.items()):
        age = current_time - wrapper.created_at
        if age > SESSION_TIMEOUT_SECONDS:
            to_remove.append(token)
            session_audio_cache.pop(token, None)
            last_messages_by_session.pop(token, None)

    for token in to_remove:
        wrapper = active_sessions.pop(token, None)
        if wrapper:
            print(f"[Cleanup] Removed session {token} (age: {age/60:.1f}min, user: {wrapper.user_id})")

    if to_remove:
        print(f"[Cleanup] Removed {len(to_remove)} old sessions. Active: {len(active_sessions)}")

    # Clean up stale pending audio cache entries (older than 5 minutes)
    audio_cleaned = 0
    for session_token, cache in list(session_audio_cache.items()):
        for message_id, entry in list(cache.items()):
            if isinstance(entry, dict):
                entry_age = current_time - entry.get('timestamp', current_time)
                # Remove pending entries older than 5 minutes (likely failed)
                if entry.get('status') == 'pending' and entry_age > 300:
                    cache.pop(message_id, None)
                    audio_cleaned += 1
                # Remove failed entries older than 1 hour
                elif entry.get('status') == 'failed' and entry_age > 3600:
                    cache.pop(message_id, None)
                    audio_cleaned += 1

    if audio_cleaned > 0:
        print(f"[Cleanup] Removed {audio_cleaned} stale audio cache entries")

def start_cleanup_thread():
    """Start background thread for session cleanup"""
    def cleanup_loop():
        while True:
            time.sleep(300)  # Every 5 minutes
            try:
                cleanup_old_sessions()
            except Exception as e:
                print(f"[Cleanup] Error: {e}")
    
    t = threading.Thread(target=cleanup_loop, daemon=True, name="SessionCleanup")
    t.start()
    print("[Cleanup] Started session cleanup thread")

# =============================================================================
# MAIN
# =============================================================================

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Flask Interview Session Web Application'
    )
    parser.add_argument('--user-id', type=str, help='Default user ID')
    parser.add_argument('--port', type=int, default=5000)
    parser.add_argument('--host', type=str, default='0.0.0.0')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--additional_context_path', default=None)
    parser.add_argument('--restart', action='store_true', default=False)
    parser.add_argument('--max_turns', type=int, default=None)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()

    if args.restart and args.user_id:
        os.system(f"rm -rf {os.getenv('LOGS_DIR')}/{args.user_id}")
        os.system(f"rm -rf {os.getenv('DATA_DIR')}/{args.user_id}")
        print(f"Cleared data for user {args.user_id}")
    
    config.default_user_id = args.user_id if args.user_id else "web_guest"
    config.host = args.host
    config.port = args.port
    config.debug = args.debug
    config.restart = args.restart
    config.max_turns = args.max_turns
    config.additional_context_path = args.additional_context_path
    
    start_cleanup_thread()
    
    print("\n" + "="*70)
    print("Flask Interview Session Server - Multi-User Mode")
    print("="*70)
    print(f"🌐 Host:              {config.host}")
    print(f"🔌 Port:              {config.port}")
    print(f"🐛 Debug:             {config.debug}")
    print(f"🔐 Authentication:    Enabled")
    print(f"🧹 Session Cleanup:   Every 5 minutes (timeout: {SESSION_TIMEOUT_SECONDS/60:.0f} min)")
    print(f"🗣️  TTS Provider:      {TTS_PROVIDER} ({TTS_VOICE})")
    print("="*70)
    print(f"\n📍 Login at: http://{config.host}:{config.port}/login")
    print(f"📊 Health check: http://{config.host}:{config.port}/health")
    print("="*70 + "\n")
    
    if not config.debug:
        print("⚠️  For production, use: gunicorn -w 2 --threads 4 -b 0.0.0.0:8080 flask_app:app\n")
    
    app.run(
        host=config.host,
        port=config.port,
        debug=config.debug,
        use_reloader=False,
        threaded=True
    )