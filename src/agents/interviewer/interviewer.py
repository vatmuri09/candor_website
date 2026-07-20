import os
import re
from typing import TYPE_CHECKING, TypedDict



from src.agents.base_agent import BaseAgent
from src.agents.interviewer.prompts import get_prompt
from src.agents.interviewer.tools import RespondToUser
from src.agents.shared.memory_tools import Recall
from src.agents.shared.anti_sycophancy import (
    inspect_turn, sanitize_interviewer_turn, regen_reminder,
)

# Fallback extractor used only when tool-call XML fails to parse. We must never
# leak <tool_calls>, <thinking>, or <subtopic_id> markup into the transcript.
_RESPONSE_TAG_RE = re.compile(r"<response>(.*?)</response>", re.IGNORECASE | re.DOTALL)
_THINKING_TAG_RE = re.compile(r"<thinking>.*?</thinking>", re.IGNORECASE | re.DOTALL)
_ANY_XML_TAG_RE = re.compile(r"<[^>]+>")

# How many consecutive turns may target the same subtopic_id (Rule-1 follow-ups
# drilling into one thread) before the next prompt gets a hard directive to move on.
# Tracked in code (Interviewer._same_subtopic_streak), not by asking the model to
# count its own recent questions.
_DEPTH_CAP_THRESHOLD = 2

# Near-duplicate detection across the whole session's interviewer questions.
# No API call: takes the max of unigram-Jaccard and bigram-Jaccard over
# stopword-filtered tokens. Bigrams catch paraphrases like "outcomes came from
# X" vs "outcomes did X bring"; unigrams catch verbatim near-repeats.
_REPEAT_SIM_THRESHOLD = 0.55

_TOKEN_RE = re.compile(r"[a-zA-Z0-9']+")
_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "at", "by",
    "with", "from", "as", "is", "was", "are", "were", "be", "been", "have",
    "has", "had", "you", "your", "yours", "yourself", "we", "us", "our",
    "i", "me", "my", "it", "its", "this", "that", "these", "those", "do",
    "did", "does", "can", "could", "would", "should", "any", "some", "how",
    "what", "when", "where", "why", "which", "who", "about", "into", "out",
    "then", "than", "there", "here", "if", "so", "just", "also", "please",
}


def _content_tokens(text: str) -> list:
    if not text:
        return []
    toks = [t.lower() for t in _TOKEN_RE.findall(text)]
    return [t for t in toks if t not in _STOPWORDS and len(t) > 2]


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _question_similarity(candidate: str, prior: str) -> float:
    ct = _content_tokens(candidate)
    pt = _content_tokens(prior)
    if len(ct) < 3 or len(pt) < 3:
        return 0.0
    uni = _jaccard(set(ct), set(pt))
    cb = set(zip(ct, ct[1:]))
    pb = set(zip(pt, pt[1:]))
    bi = _jaccard(cb, pb)
    return max(uni, bi)


def _near_duplicate_of(candidate: str, priors: list) -> str | None:
    """Return the prior question the candidate near-duplicates, else None."""
    best_sim, best_prior = 0.0, None
    for prior in priors:
        s = _question_similarity(candidate, prior)
        if s > best_sim:
            best_sim, best_prior = s, prior
    return best_prior if best_sim >= _REPEAT_SIM_THRESHOLD else None



def _salvage_response_text(raw: str) -> str:
    """Pull a usable question out of a malformed tool-call response.

    Prefer the contents of <response>...</response>; if that's missing, strip
    <thinking> blocks and any other XML tags and return what's left. Callers
    still run this through the sycophancy sanitizer before sending.
    """
    if not raw:
        return ""
    m = _RESPONSE_TAG_RE.search(raw)
    if m and m.group(1).strip():
        return m.group(1).strip()
    stripped = _THINKING_TAG_RE.sub("", raw)
    stripped = _ANY_XML_TAG_RE.sub("", stripped)
    return stripped.strip()
from src.utils.llm.prompt_utils import format_prompt
from src.interview_session.session_models import Participant, Message

from src.utils.logger.session_logger import SessionLogger
from src.utils.constants.colors import GREEN, RESET


if TYPE_CHECKING:
    from src.interview_session.interview_session import InterviewSession



class TTSConfig(TypedDict, total=False):
    """Configuration for text-to-speech."""
    enabled: bool
    provider: str  # e.g. 'openai'
    voice: str     # e.g. 'alloy'


class InterviewerConfig(TypedDict, total=False):
    """Configuration for the Interviewer agent."""
    user_id: str
    tts: TTSConfig
    interview_description: str


class Interviewer(BaseAgent, Participant):
    '''Inherits from BaseAgent and Participant. Participant is a class that all agents in the interview session inherit from.'''

    def __init__(self, config: InterviewerConfig, interview_session: 'InterviewSession'):
        BaseAgent.__init__(
            self, name="Interviewer",
            description="The agent that holds the interview and asks questions.",
            config=config)
        Participant.__init__(
            self, title="Interviewer",
            interview_session=interview_session)

        self.interview_description = config.get("interview_description")
        self.tools = {
            "recall": Recall(memory_bank=self.interview_session.memory_bank),
            "respond_to_user": RespondToUser(
                tts_config=config.get("tts", {}),
                base_path= \
                    f"{os.getenv('DATA_DIR', 'data')}/{config.get('user_id')}/",
                on_response=self._handle_response,
                on_turn_complete=lambda: setattr(
                    self, '_turn_to_respond', False)
            ),
        }

        self._turn_to_respond = False
        # A one-turn note the closer can add to the prompt (e.g. "change topic").
        self._pending_directive_note = ""
        # Code-tracked (not model-self-reported) count of consecutive turns that
        # targeted the same subtopic_id, and which subtopic that is. Used to force
        # a hard stop on over-mining a single thread instead of asking the model to
        # count its own recent questions in prose.
        self._same_subtopic_streak = 0
        self._last_subtopic_id = ""
        # Text of every question asked while the current same-subtopic streak has
        # been building. Used to catch the case a subtopic_id change alone can't:
        # the model relabels its follow-up under a different (valid) subtopic_id
        # while still narratively re-treading the same incident/story.
        self._streak_question_texts: list = []
        # Counts of how often each guardrail fired, saved with the session.
        self.guardrail_stats = {
            "affirmation": 0, "closing": 0, "midsentence_service": 0,
            "stance": 0, "advice": 0, "no_question": 0,
            "regenerated": 0, "regeneration_failed": 0,
            "near_duplicate": 0, "near_duplicate_regenerated": 0,
            "near_duplicate_regen_failed": 0,
            "depth_cap_triggered": 0, "depth_cap_regenerated": 0,
            "depth_cap_regen_failed": 0,
        }

    async def _handle_response(self, response: str, subtopic_id: str = "") -> str:
        """Clean up the interviewer's reply and add it to the chat history."""
        # Depth-cap enforcement needs to know whether the cap was ALREADY active
        # (i.e. this candidate was generated under a [DEPTH CAP ...] directive)
        # before we update the streak for this turn.
        cap_was_active = self._same_subtopic_streak >= _DEPTH_CAP_THRESHOLD
        clean = await self._enforce_non_affirming(response, cap_was_active)

        self.interview_session.add_message_to_chat_history(
            role=self.title,
            content=clean,
            metadata={'subtopic_id': str(subtopic_id)},
        )
        self.add_event(sender=self.name, tag="message",
                       content=clean)

        # Code-computed streak, not model self-report: update AFTER this turn is
        # final so next turn's prompt reflects the true count.
        sid = str(subtopic_id or "")
        if sid and sid == self._last_subtopic_id:
            self._same_subtopic_streak += 1
            self._streak_question_texts.append(clean)
        else:
            self._same_subtopic_streak = 1 if sid else 0
            self._streak_question_texts = [clean] if sid else []
        self._last_subtopic_id = sid

        return clean

    async def _enforce_non_affirming(self, response: str, cap_was_active: bool = False) -> str:
        """Sanitize + guardrail a generated turn. Returns respondent-safe text."""
        inspection = inspect_turn(response)
        for f in inspection.flags:
            self.guardrail_stats[f] = self.guardrail_stats.get(f, 0) + 1

        clean_text = inspection.clean_text
        if inspection.needs_regeneration:
            for v in inspection.violations:
                self.guardrail_stats[v] = self.guardrail_stats.get(v, 0) + 1
            SessionLogger.log_to_file(
                "execution_log",
                f"[GUARDRAIL] Interviewer turn tripped {inspection.violations}; regenerating. "
                f"Draft: {clean_text!r}"
            )

            regenerated = await self._regenerate_question(
                clean_text, inspection.violations
            )
            if regenerated:
                self.guardrail_stats["regenerated"] += 1
                regen_clean, _ = sanitize_interviewer_turn(regenerated)
                # If the regeneration still has no question, fall back rather than loop.
                recheck = inspect_turn(regen_clean)
                if not recheck.violations:
                    clean_text = recheck.clean_text
                else:
                    SessionLogger.log_to_file(
                        "execution_log",
                        f"[GUARDRAIL] Regeneration still tripped {recheck.violations}; using it anyway."
                    )
                    clean_text = regen_clean
            else:
                self.guardrail_stats["regeneration_failed"] += 1

        # Second-pass check: near-duplicate of recent interviewer questions? If
        # so, force ONE more regeneration with an explicit "already asked" note.
        # The STAR-grind failure in midlife_career_pivot showed the model happily
        # re-asks the same slot until told the exact question it repeated.
        dup_of = self._find_recent_duplicate(clean_text)
        if dup_of:
            self.guardrail_stats["near_duplicate"] += 1
            SessionLogger.log_to_file(
                "execution_log",
                f"[GUARDRAIL] Near-duplicate of prior question. Regenerating.\n"
                f"  candidate: {clean_text!r}\n  prior:     {dup_of!r}"
            )
            regen2 = await self._regenerate_non_duplicate(clean_text, dup_of)
            if regen2:
                regen2_clean, _ = sanitize_interviewer_turn(regen2)
                if _near_duplicate_of(regen2_clean, [dup_of]) is None:
                    self.guardrail_stats["near_duplicate_regenerated"] += 1
                    clean_text = regen2_clean
                else:
                    self.guardrail_stats["near_duplicate_regen_failed"] += 1
            else:
                self.guardrail_stats["near_duplicate_regen_failed"] += 1

        # Third pass: the depth cap was active for this candidate (it was generated
        # under a [DEPTH CAP ...] directive telling the model to leave this thread).
        # We deliberately do NOT try to lexically verify whether it complied —
        # "still the same underlying incident, different technical facet" is a
        # semantic judgment (e.g. "backup audio feed setup" vs "intercom
        # coordination" share almost no vocabulary despite being the same
        # troubleshooting scenario), and a token-overlap check either misses real
        # overlaps or false-positives on shared domain words. Instead: the streak
        # crossing the threshold is itself the fully deterministic trigger, and we
        # unconditionally force one regeneration attempt naming the specific prior
        # thread questions to avoid, rather than pretend a regex can grade the
        # result.
        if cap_was_active and self._streak_question_texts:
            self.guardrail_stats["depth_cap_triggered"] += 1
            SessionLogger.log_to_file(
                "execution_log",
                f"[GUARDRAIL] Depth cap active (streak={self._same_subtopic_streak} "
                f"on subtopic {self._last_subtopic_id}); forcing off-thread "
                f"regeneration.\n  candidate: {clean_text!r}"
            )
            regen3 = await self._regenerate_off_thread(
                clean_text, self._streak_question_texts, self._last_subtopic_id
            )
            if regen3:
                regen3_clean, _ = sanitize_interviewer_turn(regen3)
                self.guardrail_stats["depth_cap_regenerated"] += 1
                clean_text = regen3_clean
            else:
                self.guardrail_stats["depth_cap_regen_failed"] += 1

        return clean_text

    async def _regenerate_off_thread(self, draft: str, thread_texts: list, subtopic_id: str) -> str:
        """Ask the model to leave an over-mined thread entirely, not just reword it."""
        prior_list = "\n".join(f"  - {t}" for t in thread_texts[-6:])
        prompt = (
            "You are a strictly non-affirming research interviewer.\n"
            f"The topic is: {self.interview_description}.\n\n"
            f"You just drafted: {draft!r}\n\n"
            "This is your 4th+ consecutive question narrowing into the SAME "
            f"incident/story/mechanism (subtopic {subtopic_id}). Your last several "
            f"questions on this thread were:\n{prior_list}\n\n"
            "Write the next question about a COMPLETELY DIFFERENT subtopic from the "
            "interview plan — not another angle on this same incident, tool, or "
            "mechanism, even if worded differently. Do not reuse any of the specific "
            "nouns from the thread above. Output ONE plain, open-ended question — no "
            "preamble, no quotes, no tool tags."
        )
        try:
            out = await self.call_engine_async(prompt)
            return (out or "").strip()
        except Exception as e:
            SessionLogger.log_to_file(
                "execution_log",
                f"[GUARDRAIL] Depth-cap regeneration call failed: {e}"
            )
            return ""

    def _find_recent_duplicate(self, candidate: str) -> str | None:
        """Return the prior interviewer message that near-duplicates candidate.

        Checks the WHOLE session's interviewer messages, not just a short
        lookback window. This is pure string/token comparison (no LLM call),
        so there's no cost reason to cap it — and a short window misses
        duplicates that recur after the conversation has moved through other
        subtopics and circled back (e.g. re-asking turn 4's question at turn
        14, well outside a 6-message window).
        """
        priors = self.get_event_stream_str(
            [{"sender": "Interviewer", "tag": "message"}], as_list=True
        )
        return _near_duplicate_of(candidate, priors)

    async def _regenerate_non_duplicate(self, draft: str, prior: str) -> str:
        """Ask the model to produce a genuinely different question."""
        prompt = (
            "You are a strictly non-affirming research interviewer.\n"
            f"The topic is: {self.interview_description}.\n\n"
            f"You just drafted: {draft!r}\n"
            f"But you already asked (recently): {prior!r}\n\n"
            "Rewrite the next question so it targets a DIFFERENT dimension of "
            "the topic — a new subtopic, a different angle on the same one, or "
            "a specific detail the respondent has NOT yet been asked about. "
            "Do NOT rephrase the prior question. Do NOT ask about the same "
            "aspect. Output ONE plain, open-ended question — no preamble, no "
            "quotes, no tool tags."
        )
        try:
            out = await self.call_engine_async(prompt)
            return (out or "").strip()
        except Exception as e:
            SessionLogger.log_to_file(
                "execution_log",
                f"[GUARDRAIL] Duplicate-regeneration call failed: {e}"
            )
            return ""

    async def _regenerate_question(self, bad_draft: str, violations: list) -> str:
        """Ask the model to rewrite a bad turn as a single plain question."""
        recent = self.get_event_stream_str(
            [
                {"sender": "Interviewer", "tag": "message"},
                {"sender": "User", "tag": "message"},
            ],
            as_list=True,
        )
        recent_ctx = "\n".join(recent[-6:])
        prompt = (
            "You are a strictly non-affirming research interviewer.\n"
            f"The topic is: {self.interview_description}.\n\n"
            "Recent conversation:\n"
            f"{recent_ctx}\n\n"
            f"You drafted this next turn: \"{bad_draft}\"\n"
            f"{regen_reminder(violations)}\n\n"
            "Output ONLY the single rewritten question, with no preamble, no quotes, "
            "no acknowledgment, and no tool tags."
        )
        try:
            out = await self.call_engine_async(prompt)
            return (out or "").strip()
        except Exception as e:
            SessionLogger.log_to_file(
                "execution_log", f"[GUARDRAIL] Regeneration call failed: {e}"
            )
            return ""

    async def _run_conversation_director(self, message: Message) -> bool:
        """Let the EngagementMonitor + ConversationCloser steer this turn.

        Returns True if a scripted turn was emitted (or the session ended) and the
        interviewer should NOT generate a normal question this turn.
        """
        session = self.interview_session
        monitor = getattr(session, "engagement_monitor", None)
        closer = getattr(session, "conversation_closer", None)
        if monitor is None or closer is None:
            return False  # subagents not wired (e.g. eval harness) -> normal flow

        user_answer = message.content if (message and message.role == "User") else None
        if user_answer is not None:
            monitor.observe(user_answer)

        turn_count = len([m for m in session.chat_history if m.role == "User"])
        transcript_tail = "\n".join(
            self.get_event_stream_str(
                [{"sender": "Interviewer", "tag": "message"},
                 {"sender": "User", "tag": "message"}],
                as_list=True,
            )[-8:]
        )

        try:
            directive = await closer.direct(user_answer, monitor, transcript_tail, turn_count)
        except Exception as e:
            SessionLogger.log_to_file("execution_log", f"[CLOSER] direct() failed: {e}")
            return False

        if directive.action == "normal":
            # A pivot note (respondent chose to continue) rides along with the prompt.
            self._pending_directive_note = directive.resume_note or ""
            return False

        SessionLogger.log_to_file(
            "execution_log",
            f"[CLOSER] action={directive.action} reason={directive.reason}"
        )

        if directive.action == "end_now":
            # Optionally deliver a brief closing line, then end the session.
            if directive.text:
                await self._handle_response(directive.text)
            self._turn_to_respond = False
            session.end_session()
            return True

        # scripted_offer or scripted_wind_down: deliver the scripted line as this turn.
        self._turn_to_respond = False
        await self._handle_response(directive.text)
        return True

    async def on_message(self, message: Message):

        if message:
            SessionLogger.log_to_file(
                "execution_log",
                f"[NOTIFY] Interviewer received message from {message.role}"
            )
            self.add_event(sender=message.role, tag="message",
                           content=message.content)

        # Consult the engagement monitor + conversation closer BEFORE generating.
        # If they dictate a scripted turn (check-in, wind-down, or hard end), we
        # emit that instead of a normal question.
        if await self._run_conversation_director(message):
            return

        self._turn_to_respond = True
        iterations = 0

        while self._turn_to_respond and iterations < self._max_consideration_iterations:
            prompt = self._get_prompt()
            self.add_event(sender=self.name, tag="llm_prompt", content=prompt)
            response = await self.call_engine_async(prompt)
            print(f"{GREEN}Interviewer:\n{response}{RESET}")
            try:
                await self.handle_tool_calls_async(response)
            except Exception as e:
                salvaged = _salvage_response_text(response)
                SessionLogger.log_to_file(
                    "execution_log",
                    f"[TOOL_PARSE] {e}; salvaged {len(salvaged)} chars of response text."
                )
                if salvaged:
                    await self._handle_response(salvaged)
                    # RespondToUser normally clears _turn_to_respond; on the salvage
                    # path we must clear it explicitly or on_message loops forever.
                    self._turn_to_respond = False
                # If nothing usable came back, drop this iteration and let the loop
                # regenerate. Never surface raw <tool_calls> XML to the respondent.

            iterations += 1
            if iterations >= self._max_consideration_iterations:
                self.add_event(
                    sender="system",
                    tag="error",
                    content=f"Exceeded maximum number of consideration "
                    f"iterations ({self._max_consideration_iterations})"
                )

    def _first_planned_subtopic(self) -> str:
        """The first subtopic in the interview plan — the deterministic starting
        point for the opening question. Same plan always yields the same opener."""
        return self._first_planned_subtopic_pair()[1]

    def _first_planned_subtopic_id(self) -> str:
        """The subtopic_id matching _first_planned_subtopic's description, so the
        opener's tool call attaches to the real plan entry instead of a guessed
        slug (which silently fails to record in the session agenda)."""
        return self._first_planned_subtopic_pair()[0]

    def _first_planned_subtopic_pair(self) -> tuple[str, str]:
        try:
            topics = self.interview_session.session_agenda \
                .interview_topic_manager.get_all_topics()
            for topic in topics:
                for subtopic in topic.required_subtopics.values():
                    if subtopic.description:
                        return subtopic.subtopic_id, subtopic.description
        except Exception:
            pass
        return "", "the person's background and relationship to this topic"

    def _get_prompt(self):
        '''Gets the prompt for the interviewer. '''
        # Get user portrait and last meeting summary from session agenda
        user_portrait_str = self.interview_session.session_agenda \
            .get_user_portrait_str()
        last_meeting_summary_str = (
            self.interview_session.session_agenda
            .get_last_meeting_summary_str()
        )
        research_briefing_str = (
            self.interview_session.session_agenda
            .get_research_briefing_str()
        )

        # Get chat history from event stream where these are the senders
        chat_history_events = self.get_event_stream_str(
            [
                {"sender": "Interviewer", "tag": "message"},
                {"sender": "User", "tag": "message"},
                {"sender": "system", "tag": "recall"},
            ],
            as_list=True
        )

        recent_events = chat_history_events[-self._max_events_len:] if \
            len(chat_history_events) > self._max_events_len else chat_history_events
        current_events = recent_events[-2:] if len(recent_events) >= 2 else recent_events

        all_interviewer_messages = self.get_event_stream_str(
            [{"sender": "Interviewer", "tag": "message"}],
            as_list=True
        )
        recent_interviewer_messages = all_interviewer_messages[-5:] if \
            len(all_interviewer_messages) >= 5 else all_interviewer_messages

        # Start with all available tools
        tools_set = set(self.tools.keys())
        
        if self.use_baseline:
            # For baseline mode, remove recall tool
            tools_set.discard("recall")

        # Create format parameters based on prompt type
        format_params = {
            "user_portrait": user_portrait_str,
            "interview_description": self.interview_description,
            "last_meeting_summary": last_meeting_summary_str,
            "research_briefing": research_briefing_str or "(no research briefing available)",
            "chat_history": '\n'.join(recent_events),
            "current_events": '\n'.join(current_events),
            "recent_interviewer_messages": '\n'.join(
                [msg for msg in recent_interviewer_messages]),
            "tool_descriptions": self.get_tools_description(list(tools_set)),
            # Deterministic first question: always the first subtopic in the plan.
            "opening_subtopic": self._first_planned_subtopic(),
            "opening_subtopic_id": self._first_planned_subtopic_id(),
        }

        # Only add questions_and_notes for normal mode
        if not self.use_baseline:
            questions_and_notes_str = self.interview_session.session_agenda \
                .get_questions_and_notes_str(
                    hide_answered="all", active_topics_only=True
                )
            format_params["questions_and_notes"] = questions_and_notes_str

            # Get strategic question suggestions from ExplorationPlanner (only if not stale)
            # Staleness is checked before formatting to avoid unnecessary work
            if self._should_include_strategic_questions():
                strategic_questions_str = self._format_strategic_questions()
                format_params["strategic_questions"] = strategic_questions_str

        # Use the baseline prompt if enabled
        if len(all_interviewer_messages) == 0 and len(last_meeting_summary_str) == 0:
            main_prompt = get_prompt("introduction")
        elif len(all_interviewer_messages) == 0:
            main_prompt = get_prompt("introduction_continue_session")
        elif self.use_baseline:
            main_prompt = get_prompt("baseline")
        else:
            main_prompt = get_prompt("normal")

            # Remove STRATEGIC_QUESTIONS section from template if stale
            if not self.use_baseline and not self._should_include_strategic_questions():
                # Remove the {STRATEGIC_QUESTIONS} line to exclude the section entirely
                main_prompt = main_prompt.replace("\n{STRATEGIC_QUESTIONS}\n", "\n")
                # Don't provide strategic_questions key in format_params (already omitted above)

        prompt = format_prompt(main_prompt, format_params)

        # One-turn directive from the ConversationCloser (e.g. mandatory pivot on resume).
        # Separate field from the depth-cap directive below: the closer's "normal"
        # path unconditionally overwrites _pending_directive_note (even to "") every
        # turn, so it cannot safely share state with a directive that must survive
        # turns the closer doesn't touch.
        if self._pending_directive_note:
            prompt = f"{prompt}\n{self._pending_directive_note}"
            self._pending_directive_note = ""

        # Code-computed depth cap: if the last N turns in a row targeted the same
        # subtopic, hard-block Rule 1 for it next turn rather than trusting the
        # model to notice on its own.
        if self._same_subtopic_streak >= _DEPTH_CAP_THRESHOLD:
            prompt = (
                f"{prompt}\n\n[DEPTH CAP — you have asked "
                f"{self._same_subtopic_streak} consecutive questions targeting "
                f"subtopic {self._last_subtopic_id}. RULE 1 IS NOT AVAILABLE this "
                "turn, even if the respondent's last answer contains an unprobed "
                "concrete noun. You MUST apply RULE 2 (deepen a different, "
                "ungrounded subtopic) or RULE 3 (move to a new subtopic) and target "
                f"a subtopic_id other than {self._last_subtopic_id}.]"
            )

        # ExplorationPlanner's utility-scored coverage gaps constrain Rule 3, not
        # just suggest to it. Turns the planner's ranked, structured output
        # (subtopic_id + priority, derived from real is_covered state — see
        # exploration_planner.py's _calculate_hypothetical_utility) into an actual
        # limit on the choice set instead of an appendix the model can ignore.
        rule3_directive = self._rule3_constraint_directive()
        if rule3_directive:
            prompt = f"{prompt}{rule3_directive}"

        return prompt

    def _rule3_constraint_directive(self) -> str:
        """When ExplorationPlanner's suggestions are fresh, constrain which
        subtopic_ids RULE 3 may target to the top utility-scored ones."""
        if self.use_baseline or not self._should_include_strategic_questions():
            return ""
        suggestions = self.interview_session.exploration_planner \
            .strategic_state.strategic_question_suggestions
        if not suggestions:
            return ""
        top_ids, seen = [], set()
        for s in sorted(suggestions, key=lambda x: x.get('priority', 0), reverse=True):
            sid = s.get('subtopic_id')
            if sid and sid not in seen:
                seen.add(sid)
                top_ids.append(sid)
            if len(top_ids) >= 5:
                break
        if not top_ids:
            return ""
        ids_str = ", ".join(top_ids)
        return (
            f"\n\n[RULE 3 CONSTRAINT — ExplorationPlanner has utility-scored the "
            f"current coverage gaps. If RULE 3 applies this turn (moving to a new "
            f"subtopic), you MUST target one of: {ids_str}. Do not pick a "
            "different subtopic_id under Rule 3 while these remain uncovered — "
            "they were chosen to maximize expected value for this interview. "
            "This constraint does not apply to Rule 1 or Rule 2.]"
        )

    def _format_strategic_questions(self) -> str:
        """
        Format strategic question suggestions from ExplorationPlanner.

        Returns formatted string with strategic questions or empty state message.
        Handles case where suggestions may be stale (from 3-5 turns ago).
        """
        # Access strategic state from ExplorationPlanner
        strategic_state = self.interview_session.exploration_planner.strategic_state
        suggestions = strategic_state.strategic_question_suggestions

        if not suggestions:
            return "No strategic question suggestions available yet. Use coverage-based heuristics to select questions from the topics list."

        # Get top rollout if available
        top_rollout = None
        if strategic_state.rollout_predictions:
            top_rollout = strategic_state.rollout_predictions[0]

        formatted_lines = []

        # Add top rollout context if available
        if top_rollout:
            formatted_lines.append("**Highest-Utility Conversation Path Predicted:**")
            formatted_lines.append(f"Utility Score: {top_rollout.utility_score:.3f} (Higher is better)")
            formatted_lines.append(f"- Expected new subtopics covered: {top_rollout.expected_coverage_delta}")
            formatted_lines.append(f"- Emergence potential: {top_rollout.emergence_potential:.2f}")
            formatted_lines.append(f"- Cost (turns): {top_rollout.cost_estimate}")
            formatted_lines.append("")
            formatted_lines.append("The questions below are optimized to align with this high-utility path.")
            formatted_lines.append("")

        # Format suggestions by priority (high to low)
        sorted_suggestions = sorted(suggestions, key=lambda x: x.get('priority', 0), reverse=True)

        formatted_lines.append("**Strategic Question Suggestions (sorted by priority):**")
        formatted_lines.append("")
        for i, suggestion in enumerate(sorted_suggestions, 1):
            formatted_lines.append(f"{i}. **{suggestion['content']}**")
            formatted_lines.append(f"   - Target: Subtopic {suggestion['subtopic_id']}")
            formatted_lines.append(f"   - Strategy: {suggestion['strategy_type']}")
            formatted_lines.append(f"   - Priority: {suggestion['priority']}/10")
            formatted_lines.append(f"   - Reasoning: {suggestion['reasoning']}")
            formatted_lines.append("")  # Blank line between suggestions

        return "\n".join(formatted_lines)

    def _should_include_strategic_questions(self) -> bool:
        """
        Determine if strategic questions should be included in the prompt.

        Strategic questions become stale after exceeding the rollout horizon.
        Only include them if they are fresh (within horizon + buffer).

        Returns:
            bool: True if strategic questions should be included, False if stale
        """
        strategic_state = self.interview_session.exploration_planner.strategic_state

        # If no suggestions exist, don't include
        if not strategic_state.strategic_question_suggestions:
            return False

        # Calculate current turn (count User messages)
        current_turn = len([
            m for m in self.interview_session.chat_history
            if m.role == "User"
        ])

        # Get last planning turn from strategic state
        last_planning_turn = strategic_state.last_planning_turn

        # If planning hasn't run yet (turn 0), don't include
        if last_planning_turn == 0:
            return False

        # Get rollout horizon from exploration planner
        rollout_horizon = self.interview_session.exploration_planner.rollout_horizon

        # Calculate staleness: questions are stale if we're beyond horizon + buffer
        # Buffer of 2 turns accounts for: 1) planning completes after trigger, 2) grace period
        staleness_threshold = last_planning_turn + rollout_horizon + 2

        # Include questions only if NOT stale
        is_fresh = current_turn <= staleness_threshold

        if not is_fresh:
            SessionLogger.log_to_file(
                "execution_log",
                f"[NOTIFY] (Interviewer) Strategic questions are stale "
                f"(last_planning_turn={last_planning_turn}, current_turn={current_turn}, "
                f"threshold={staleness_threshold}). Excluding from prompt."
            )

        return is_fresh
