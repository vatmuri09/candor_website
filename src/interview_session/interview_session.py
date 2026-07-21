import asyncio
import os
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, TypedDict
import signal

import time
from tiktoken import get_encoding

from src.agents.base_agent import BaseAgent
from src.interview_session.session_models import Message, MessageType, Participant
from src.agents.interviewer.interviewer import Interviewer, InterviewerConfig, TTSConfig
from src.agents.agenda_manager.agenda_manager import AgendaManager, AgendaManagerConfig
from src.agents.exploration_planner.exploration_planner import ExplorationPlanner, ExplorationPlannerConfig
from src.agents.engagement.engagement_monitor import EngagementMonitor
from src.agents.engagement.probe_quality_monitor import ProbeQualityMonitor
from src.agents.conversation_closer.conversation_closer import ConversationCloser
from src.agents.context.context_research import ContextResearchAgent, ContextResearchResult
from src.agents.user.user_agent import UserAgent
from src.content.session_agenda.session_agenda import SessionAgenda
from src.utils.data_process import save_feedback_to_csv
from src.utils.logger.session_logger import SessionLogger, setup_logger
from src.utils.logger.evaluation_logger import EvaluationLogger
from src.interview_session.user.user import User
from src.interview_session.user.dummy_participant import UserDummyParticipant
from src.content.memory_bank.memory_bank_vector_db import VectorMemoryBank
from src.content.memory_bank.memory import Memory
from src.content.question_bank.question_bank_vector_db import QuestionBankVectorDB
from src.utils.token_tracker import TokenUsageTracker




class UserConfig(TypedDict, total=False):
    """Configuration for user settings.
    """
    user_id: str
    enable_voice: bool
    report_style: str


class InterviewConfig(TypedDict, total=False):
    """Configuration for interview settings."""
    enable_voice: bool
    interview_description: str
    interview_plan_path: str
    interview_evaluation: str
    additional_context_path: str
    initial_user_portrait_path: str

class BankConfig(TypedDict, total=False):
    """Configuration for memory and question banks."""
    memory_bank_type: str  # "vector_db", "graph_rag", etc.
    historical_question_bank_type: str  # "vector_db", "graph", "semantic", etc.


class InterviewSession:

    def __init__(self, interaction_mode: str = 'terminal', user_config: UserConfig = {},
                 interview_config: InterviewConfig = {}, bank_config: BankConfig = {},
                 use_baseline: Optional[bool] = None, max_turns: Optional[int] = None):
        """Initialize the interview session.

        Args:
            interaction_mode: How to interact with user 
                Options: 'terminal', 'agent', or 'api'
            user_config: User configuration dictionary
                user_id: User identifier (default: 'default_user')
                enable_voice: Enable voice input (default: False)
            interview_config: Interview configuration dictionary
                enable_voice: Enable voice output (default: False)
            bank_config: Bank configuration dictionary
                memory_bank_type: Type of memory bank 
                    Options: "vector_db", etc.
                historical_question_bank_type: Type of question bank 
                    Options: "vector_db", etc.
            use_baseline: Whether to use baseline prompt (default: read from .env)
            max_turns: Optional maximum number of turns before ending session
                      If None, session continues until manually ended
        """

        # Set the baseline mode for all agents
        if use_baseline is not None:
            # Set the class variable directly to affect all agent instances
            BaseAgent.use_baseline = use_baseline
        else:
            BaseAgent.use_baseline = \
                os.getenv("USE_BASELINE_PROMPT", "false").lower() == "true"
        
        # User setup
        self.user_id = user_config.get("user_id", "default_user")
        self._initial_additional_context_path = interview_config.get("additional_context_path", None)
        self._interview_description = interview_config.get("interview_description", "any topic")

        # Session agenda setup
        self.session_agenda = SessionAgenda.get_last_session_agenda(self.user_id,
                                                                    initial_user_portrait_path=interview_config.get('initial_user_portrait_path'),
                                                                    interview_plan_path=interview_config.get('interview_plan_path'),
                                                                    interview_description=self._interview_description,
                                                                    interview_evaluation=interview_config.get('interview_evaluation'))
        self.session_id = self.session_agenda.session_id + 1

        # Memory bank setup
        memory_bank_type = bank_config.get("memory_bank_type", "vector_db")
        if memory_bank_type == "vector_db":
            self.memory_bank = VectorMemoryBank.load_from_file(self.user_id)
            self.memory_bank.set_session_id(self.session_id)
        else:
            raise ValueError(f"Unknown memory bank type: {memory_bank_type}")

        # Question bank setup
        historical_question_bank_type = \
            bank_config.get("historical_question_bank_type", "vector_db")
        if historical_question_bank_type == "vector_db":
            self.historical_question_bank = \
                QuestionBankVectorDB.load_from_file(
                    self.user_id)
            self.historical_question_bank.set_session_id(self.session_id)
            self.proposed_question_bank = QuestionBankVectorDB()
        else:
            raise ValueError(
                f"Unknown question bank type: {historical_question_bank_type}")

        # Logger setup
        setup_logger(self.user_id, self.session_id,
                     console_output_files=["execution_log"])
        EvaluationLogger.setup_logger(self.user_id, self.session_id)

        # Token usage tracking setup
        self.token_tracker = TokenUsageTracker(
            session_id=str(self.session_id),
            user_id=self.user_id
        )
        # Set the class variable so all agents can access it
        BaseAgent.token_tracker = self.token_tracker

        # Chat history
        self.chat_history: list[Message] = []

        # Background tasks spawned during a turn (notifications, agenda/planner
        # work). We keep references so a synchronous turn can wait for them all
        # to finish before responding. See _spawn / _drain / run_one_turn.
        self._bg_tasks: set = set()

        # Session states signals
        self.interaction_mode = interaction_mode
        self.session_in_progress = True
        self.session_completed = False
        self._session_timeout = False
        self.max_turns = max_turns

        # Counter for user messages
        self._user_message_count = 0

        # Last message timestamp tracking for session timeout
        self._last_message_time = datetime.now()
        self._last_user_message = None
        self.timeout_minutes = int(os.getenv("SESSION_TIMEOUT_MINUTES", 10))

        # User in the interview session
        if interaction_mode == 'agent':
            self.user: User = UserAgent(
                user_id=self.user_id, interview_session=self, 
                config=user_config)
        elif interaction_mode == 'terminal':
            self.user: User = User(user_id=self.user_id, interview_session=self,
                                   enable_voice_input=user_config \
                                   .get("enable_voice", False))
        elif interaction_mode == 'api':
            self.user: User = UserDummyParticipant(user_id=self.user_id, interview_session=self) # No direct user interface for API mode
        else:
            raise ValueError(f"Invalid interaction_mode: {interaction_mode}")

        # Agents in the interview session
        self._interviewer: Interviewer = Interviewer(
            config=InterviewerConfig(
                user_id=self.user_id,
                tts=TTSConfig(enabled=interview_config.get(
                    "enable_voice", False)),
                interview_description=self._interview_description,
            ),
            interview_session=self
        )
        # AgendaManager config with optional dedicated model
        scribe_config = AgendaManagerConfig(user_id=self.user_id)

        # Use dedicated scribe model if configured
        scribe_model = os.getenv("AGENDA_MANAGER_MODEL_NAME")
        if scribe_model:
            scribe_config["model_name"] = scribe_model
            # Pass base_url if configured (for vLLM)
            scribe_base_url = os.getenv("AGENDA_MANAGER_VLLM_BASE_URL")
            if scribe_base_url:
                scribe_config["base_url"] = scribe_base_url

        self.agenda_manager = AgendaManager(
            config=scribe_config,
            interview_session=self
        )
        
        # ExplorationPlanner config
        # TODO: Tune exploration planner parameters
        planner_config = ExplorationPlannerConfig(
                user_id=self.user_id,
                turn_trigger=int(os.getenv("EXPLORATION_PLANNER_TURN_TRIGGER", "3")),
                num_rollouts=int(os.getenv("EXPLORATION_PLANNER_NUM_ROLLOUTS", "3")),
                rollout_horizon=int(os.getenv("EXPLORATION_PLANNER_ROLLOUT_HORIZON", "3")),
                max_strategic_questions=int(os.getenv("EXPLORATION_PLANNER_MAX_QUESTIONS", "5")),
                alpha=float(os.getenv("EXPLORATION_PLANNER_ALPHA", "0.5")),  # Coverage weight
                beta=float(os.getenv("EXPLORATION_PLANNER_BETA", "0.3")),   # Cost penalty
                gamma=float(os.getenv("EXPLORATION_PLANNER_GAMMA", "0.2"))   # Emergence reward
        )
        
        # Use dedicated planner model if configured
        planner_model = os.getenv("EXPLORATION_PLANNER_MODEL_NAME")
        if planner_model:
            planner_config["model_name"] = planner_model
            # Pass base_url if configured (for vLLM)
            planner_base_url = os.getenv("EXPLORATION_PLANNER_VLLM_BASE_URL")
            if planner_base_url:
                planner_config["base_url"] = planner_base_url
        
        self.exploration_planner: ExplorationPlanner = ExplorationPlanner(
            config=planner_config,
            interview_session=self
        )

        # Engagement monitor + conversation closer (the two new bots). These are
        # consulted by the Interviewer at the top of each turn rather than being
        # pub/sub subscribers, so there is still exactly one interviewer turn.
        monitor_model = os.getenv("ENGAGEMENT_MONITOR_MODEL_NAME")
        monitor_cfg = {"user_id": self.user_id}
        if monitor_model:
            monitor_cfg["model_name"] = monitor_model
        self.engagement_monitor = EngagementMonitor(
            config=monitor_cfg, interview_session=self
        )
        # Rule-based, no-LLM-call depth-probing signal (SPEC.md priority #3):
        # flags a streak of interviewer turns that never push for a reason,
        # concrete example, number, or contrast. Read by admin surfacing only;
        # does not steer the interviewer.
        self.probe_quality_monitor = ProbeQualityMonitor()
        self.conversation_closer = ConversationCloser(
            topic_name=self._interview_description,
            interview_session=self,
        )

        # Context research agent: researches background about the interview topic
        # (via web search) and produces a briefing that seeds the interview. The
        # briefing is shown to the participant for approval before the interview
        # starts, then reused across conversations on the same topic.
        context_model = os.getenv("CONTEXT_SEARCH_MODEL_NAME")
        context_cfg = {}
        if context_model:
            context_cfg["model_name"] = context_model
        self.context_research_agent = ContextResearchAgent(
            config=context_cfg, interview_session=self
        )
        # The researched briefing (dict) offered for approval, and the text the
        # participant approved to seed the interview with.
        self.retrieved_context: Optional[dict] = None
        self.approved_context: Optional[str] = None

        # Subscriptions of participants to each other
        self._subscriptions: Dict[str, List[Participant]] = {
            # Subscribers of Interviewer: Note-taker and User (in following code)
            "Interviewer": [self.agenda_manager],
            # Subscribers of User: Interviewer, AgendaManager, and ExplorationPlanner
            "User": [self._interviewer, self.agenda_manager, self.exploration_planner]
        }

        # User participant for terminal interaction
        if self.user:
            self._subscriptions["Interviewer"].append(self.user)

        # User API participant for backend API interaction
        # self.api_participant = None
        # if interaction_mode == 'api':
        #     self.api_participant = UserDummyParticipant(interview_session=self)
        #     self._subscriptions["Interviewer"].append(self.api_participant)
        #     self._subscriptions["User"].append(self.api_participant)

        # Shutdown signal handler - only for agent mode
        if interaction_mode == 'agent':
            self._setup_signal_handlers()
        
        SessionLogger.log_to_file(
            "execution_log", f"[INIT] Interview session initialized")
        SessionLogger.log_to_file(
            "execution_log", f"[INIT] User ID: {self.user_id}")
        SessionLogger.log_to_file(
            "execution_log", f"[INIT] Session ID: {self.session_id}")
        SessionLogger.log_to_file(
            "execution_log", f"[INIT] Use baseline: {BaseAgent.use_baseline}")
        
        self.tokenizer = get_encoding("cl100k_base")

    def _spawn(self, coro):
        """Start a background task and keep a reference so a turn can wait on it."""
        task = asyncio.create_task(coro)
        self._bg_tasks.add(task)
        task.add_done_callback(self._bg_tasks.discard)
        return task

    async def _notify_participants(self, message: Message):
        """Notify subscribers asynchronously"""
        # Gets subscribers for the user that sent the message.
        subscribers = self._subscriptions.get(message.role, [])
        SessionLogger.log_to_file(
            "execution_log", 
            (
                f"[NOTIFY] Notifying {len(subscribers)} subscribers "
                f"for message from {message.role}"
            )
        )

        # Create independent tasks for each subscriber
        tasks = []
        for sub in subscribers:
            if self.session_in_progress:
                task = self._spawn(sub.on_message(message))
                tasks.append(task)
        
        # Allow tasks to run concurrently without waiting for each other
        await asyncio.sleep(0)  # Explicitly yield control

        # Special handling for user messages after notifying participants
        if message.role == "User":
            self._last_user_message = message
            self._user_message_count += 1

            # Update the turn counter for token tracking
            BaseAgent.current_turn = self._user_message_count

            # Save token usage snapshot every turn
            snapshot_path = self.token_tracker.save_snapshot()
            SessionLogger.log_to_file(
                "execution_log",
                f"[TOKEN_TRACKING] Saved token usage snapshot to {snapshot_path}",
                log_level="info"
            )

            # Check if max turns reached
            if self.max_turns is not None and \
                    self._user_message_count >= self.max_turns:
                SessionLogger.log_to_file(
                    "execution_log",
                    f"[TURNS] Maximum turns ({self.max_turns}) reached. "
                    f"Ending session."
                )
                self.session_in_progress = False
                # Save final token usage summary
                final_summary_path = self.token_tracker.save_final_summary()
                SessionLogger.log_to_file(
                    "execution_log",
                    f"[TOKEN_TRACKING] Saved final token usage summary to {final_summary_path}",
                    log_level="info"
                )
            elif self.session_agenda.all_core_topics_completed():
                SessionLogger.log_to_file(
                    "execution_log",
                    f"[TOPICS] All topics for this session have been completed. "
                    f"Ending session."
                )
                self.session_in_progress = False
                # Save final token usage summary
                final_summary_path = self.token_tracker.save_final_summary()
                SessionLogger.log_to_file(
                    "execution_log",
                    f"[TOKEN_TRACKING] Saved final token usage summary to {final_summary_path}",
                    log_level="info"
                )

    def add_message_to_chat_history(self, role: str, content: str = "", 
                                    message_type: str = MessageType.CONVERSATION,
                                    metadata: dict = {}):
        """Add a message to the chat history"""

        # Reject messages if session is not in progress
        if not self.session_in_progress:
            return

        # Set fixed content for skip and like messages
        if message_type == MessageType.SKIP:
            content = "Skip the question"
        elif message_type == MessageType.LIKE:
            content = "Like the question"

        # Create message object
        message = Message(
            id=str(uuid.uuid4()),
            type=message_type,
            role=role,
            content=content,
            timestamp=datetime.now(),
            metadata=metadata,
        )

        if role == "User":
            self._last_message_time = message.timestamp
        elif role == "Interviewer" and self._last_user_message is not None:
            self._last_user_message = None
        
        # Log feedback
        if message_type != MessageType.CONVERSATION:
            save_feedback_to_csv(
                self.chat_history[-1], message, self.user_id, self.session_id)

        # Notify participants if message is a skip or conversation
        if message_type == MessageType.SKIP or \
              message_type == MessageType.CONVERSATION:
            
            # Add message to chat history
            self.chat_history.append(message)
            SessionLogger.log_to_file(
                "chat_history", f"{message.role}: {message.content}")
            
            # Notify participants
            self._spawn(self._notify_participants(message))


        SessionLogger.log_to_file(
            "execution_log", 
            (
                f"[CHAT_HISTORY] {message.role}'s message has been added "
                f"to chat history."
            )
        )

    async def _drain(self):
        """Wait until every background task from this turn has finished.

        A turn kicks off a cascade of tasks (notify the interviewer, which in turn
        notifies the agenda manager and the user buffer, etc.). We keep gathering
        them until nothing is left running and the agenda manager / exploration
        planner are both idle.
        """
        while True:
            pending = [t for t in list(self._bg_tasks) if not t.done()]
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
                continue
            # Let any just-scheduled task get going, then check the slow agents.
            await asyncio.sleep(0)
            if (self.agenda_manager.processing_in_progress or
                    self.exploration_planner.processing_in_progress):
                await asyncio.sleep(0.05)
                continue
            if not any(not t.done() for t in self._bg_tasks):
                break

    def _collect_interviewer_messages(self) -> List[str]:
        """Pull any interviewer messages that were buffered during the turn."""
        if hasattr(self.user, "get_and_clear_messages"):
            msgs = self.user.get_and_clear_messages() or []
            return [m["content"] for m in msgs if m.get("role") == "Interviewer"]
        return []

    async def research_context(self) -> dict:
        """Research background context for this interview's topic.

        Returns a briefing dict (see ContextResearchResult.to_dict) to show the
        participant for approval. Reuses a cached, previously-approved briefing for
        the same topic when one exists (see utils.storage.context_store).
        """
        from src.utils.storage import context_store

        cached = context_store.get(self._interview_description)
        if cached and cached.get("context"):
            cached["from_cache"] = True
            self.retrieved_context = cached
            SessionLogger.log_to_file(
                "execution_log",
                f"[CONTEXT_RESEARCH] reusing cached context for "
                f"{self._interview_description!r}"
            )
            return cached

        result = await self.context_research_agent.research_topic(
            self._interview_description)
        data = result.to_dict()
        data["from_cache"] = False
        self.retrieved_context = data
        return data

    async def start(self, approved_context: Optional[str] = None) -> List[str]:
        """Seed the agenda (optionally with approved context) and open the interview.

        Run once per session. If approved_context is given it seeds the agenda and
        is cached for reuse on this topic; otherwise we fall back to any configured
        additional_context_path.
        """
        if approved_context is not None:
            self.approved_context = approved_context
            self._cache_approved_context(approved_context)

        await self.agenda_manager.augment_session_agenda(
            additional_context_path=self._initial_additional_context_path,
            approved_context=self.approved_context)
        self.session_in_progress = True
        # Interviewer opens the conversation.
        await self._interviewer.on_message(None)
        await self._drain()
        return self._collect_interviewer_messages()

    def _cache_approved_context(self, approved_context: str) -> None:
        """Persist the approved briefing for reuse across conversations."""
        try:
            from src.utils.storage import context_store
            record = dict(self.retrieved_context or {})
            record["context"] = approved_context
            record["topic"] = self._interview_description
            record["approved"] = True
            context_store.put(self._interview_description, record, approved=True)
        except Exception as e:
            SessionLogger.log_to_file(
                "execution_log", f"[CONTEXT_RESEARCH] cache write failed: {e}")

    async def run_one_turn(self, user_text: str) -> List[str]:
        """Feed one user message through the pipeline and return the reply."""
        if not self.session_in_progress:
            return []
        self.add_message_to_chat_history(role="User", content=user_text)
        await self._drain()
        return self._collect_interviewer_messages()

    # ---- state save/restore so a session can survive between web requests ----

    def _agents_for_state(self) -> dict:
        """The agents whose event stream we need to keep between turns."""
        return {
            "interviewer": self._interviewer,
            "agenda_manager": self.agenda_manager,
            "exploration_planner": self.exploration_planner,
            "engagement_monitor": self.engagement_monitor,
            "context_research_agent": self.context_research_agent,
        }

    def to_state(self) -> dict:
        """Dump the in-memory conversation state to a plain dict (JSON-safe).

        The FAISS banks, the session agenda and the strategic state are saved to
        their own files (see save_files); this covers everything else so a fresh
        InterviewSession can be rebuilt and pick up exactly where it left off.
        """
        self.save_files()
        monitor, closer, interviewer = (
            self.engagement_monitor, self.conversation_closer, self._interviewer)
        last_sig = monitor.last_signal
        return {
            "session_id": self.session_id,
            "session_in_progress": self.session_in_progress,
            "session_completed": self.session_completed,
            "user_message_count": self._user_message_count,
            "max_turns": self.max_turns,
            "interview_description": self._interview_description,
            "chat_history": [m.model_dump(mode="json") for m in self.chat_history],
            "retrieved_context": self.retrieved_context,
            "approved_context": self.approved_context,
            "event_streams": {
                name: [e.model_dump(mode="json") for e in agent.event_stream]
                for name, agent in self._agents_for_state().items()
            },
            "engagement_monitor": {
                "answer_history": monitor.answer_history,
                "sentiments": monitor.sentiments,
                "streak": monitor.streak,
                "turns_since_llm_check": monitor._turns_since_llm_check,
                "last_signal": last_sig.__dict__ if last_sig else None,
            },
            "conversation_closer": {
                "state": closer.state,
                "last_offer_turn": closer._last_offer_turn,
                "close_offers": closer.close_offers,
                "pivots": closer.pivots,
                "quality_collapse_ends": closer.quality_collapse_ends,
                "stance_collapse_streak": closer._stance_collapse_streak,
            },
            "interviewer": {
                "guardrail_stats": interviewer.guardrail_stats,
                "pending_directive_note": interviewer._pending_directive_note,
            },
            "probe_quality_monitor": self.probe_quality_monitor.to_state(),
        }

    def load_state(self, state: dict) -> None:
        """Restore a session dict produced by to_state onto this fresh session."""
        from src.agents.engagement.engagement_monitor import QualitySignal

        self.session_id = state["session_id"]
        self.session_in_progress = state["session_in_progress"]
        self.session_completed = state["session_completed"]
        self._user_message_count = state.get("user_message_count", 0)
        self.max_turns = state.get("max_turns")
        self.retrieved_context = state.get("retrieved_context")
        self.approved_context = state.get("approved_context")
        self.chat_history = [Message(**m) for m in state.get("chat_history", [])]

        streams = state.get("event_streams", {})
        for name, agent in self._agents_for_state().items():
            agent.event_stream = [BaseAgent.Event(**e) for e in streams.get(name, [])]

        mon = state.get("engagement_monitor", {})
        self.engagement_monitor.answer_history = mon.get("answer_history", [])
        self.engagement_monitor.sentiments = mon.get("sentiments", [])
        self.engagement_monitor.streak = mon.get("streak", 0)
        self.engagement_monitor._turns_since_llm_check = mon.get("turns_since_llm_check", 0)
        if mon.get("last_signal"):
            self.engagement_monitor.last_signal = QualitySignal(**mon["last_signal"])

        clo = state.get("conversation_closer", {})
        self.conversation_closer.state = clo.get("state", self.conversation_closer.ACTIVE)
        self.conversation_closer._last_offer_turn = clo.get("last_offer_turn", 0)
        self.conversation_closer.close_offers = clo.get("close_offers", 0)
        self.conversation_closer.pivots = clo.get("pivots", 0)
        self.conversation_closer.quality_collapse_ends = clo.get("quality_collapse_ends", 0)
        self.conversation_closer._stance_collapse_streak = clo.get("stance_collapse_streak", 0)

        itv = state.get("interviewer", {})
        self._interviewer.guardrail_stats = itv.get("guardrail_stats",
                                                     self._interviewer.guardrail_stats)
        self._interviewer._pending_directive_note = itv.get("pending_directive_note", "")

        self.probe_quality_monitor.load_state(state.get("probe_quality_monitor"))

    def save_files(self) -> None:
        """Persist the FAISS banks, agenda and strategic state to disk."""
        try:
            self.memory_bank.save_to_file(self.user_id)
            self.historical_question_bank.save_to_file(self.user_id)
            self.session_agenda.save(save_type="original")
            self.exploration_planner.strategic_state.save_to_file(self.user_id)
        except Exception as e:
            SessionLogger.log_to_file("execution_log", f"[STATE] save_files warning: {e}")

    async def run(self):
        """Run the interview session"""
        # Augment session agenda with existing profile if applicable
        await self.agenda_manager.augment_session_agenda(additional_context_path=self._initial_additional_context_path)

        SessionLogger.log_to_file(
            "execution_log", f"[RUN] Starting interview session")
        self.session_in_progress = True

        # In-interview Processing
        try:
            # Interviewer initiate the conversation (if not in API mode)
            if self.user is not None:
                await self._interviewer.on_message(None)

            # Monitor the session for completion and timeout
            while self.session_in_progress or \
                self.agenda_manager.processing_in_progress or \
                self.exploration_planner.processing_in_progress:
                await asyncio.sleep(0.1)

                # Check for timeout
                if datetime.now() - self._last_message_time \
                        > timedelta(minutes=self.timeout_minutes):
                    SessionLogger.log_to_file(
                        "execution_log", 
                        (
                            f"[TIMEOUT] Session timed out after "
                            f"{self.timeout_minutes} minutes of inactivity"
                        )
                    )
                    self.session_in_progress = False
                    self._session_timeout = True
                    break

        except Exception as e:
            SessionLogger.log_to_file(
                "execution_log", f"[RUN] Unexpected error: {str(e)}")
            raise e

        # Post-interview Processing
        finally:
            self.session_in_progress = False

            # Save memory bank
            self.memory_bank.save_to_file(self.user_id)
            SessionLogger.log_to_file(
                "execution_log", f"[COMPLETED] Memory bank saved")

            # Save historical question bank
            self.historical_question_bank.save_to_file(self.user_id)
            SessionLogger.log_to_file(
                "execution_log", f"[COMPLETED] Question bank saved")

            self.session_completed = True
            SessionLogger.log_to_file(
                "execution_log", f"[COMPLETED] Session completed")

    async def get_session_memories(self, include_processed=True) -> List[Memory]:
        """Get memories added during this session
        
        Args:
            include_processed: If True, returns all memories from the session
                              If False, returns only the unprocessed memories
        """
        return await self.agenda_manager.get_session_memories(
            clear_processed=False, 
            wait_for_processing=True,
            include_processed=include_processed
        )

    def end_session(self):
        """End the session without triggering report update"""
        self.session_in_progress = False

        # Save final token usage summary
        if hasattr(self, 'token_tracker'):
            final_summary_path = self.token_tracker.save_final_summary()
            SessionLogger.log_to_file(
                "execution_log",
                f"[TOKEN_TRACKING] Saved final token usage summary to {final_summary_path}",
                log_level="info"
            )

    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, self._signal_handler)

    def _signal_handler(self):
        """Handle shutdown signals"""
        self.session_in_progress = False
        SessionLogger.log_to_file(
            "execution_log", f"[SIGNAL] Shutdown signal received")
        SessionLogger.log_to_file(
            "execution_log", f"[SIGNAL] Waiting for interview session to finish...")
    
    def set_db_session_id(self, db_session_id: int):
        """Set the database session ID. Used for server mode"""
        self.db_session_id = db_session_id

    def get_db_session_id(self) -> int:
        """Get the database session ID. Used for server mode"""
        return self.db_session_id
        