"""Decides when to wrap up the interview.

Before each interviewer turn this looks at the latest answer and the engagement
monitor and picks one of: keep going, offer to wrap up, ask a final "anything to
add?" then end, or (if things have gone really badly) end right away. It's a
small state machine and doesn't send its own messages.

Closer 2.0 additions:
- LLM-generated close lines (thread-aware, name a real loose end + a concrete
  respondent detail), with the canned fallbacks used only when the LLM path
  fails or the InterviewTracker has no state yet.
- `quality_collapse` path that ends early if the tracker's respondent stance
  drops for `stance_collapse_streak` consecutive updates.
"""
import os
import re
from dataclasses import dataclass
from typing import Optional

from src.agents.base_agent import BaseAgent
from src.agents.engagement.engagement_monitor import EngagementMonitor
from src.utils.logger.session_logger import SessionLogger

DISENGAGE_THRESHOLD = 3

_CONTINUE_SIGNALS = re.compile(
    r"\b(yes\b|yeah\b|sure\b|ok\b|okay\b|fine\b|go ahead|continue|"
    r"more questions?|keep going|alright\b|why not|sounds good|"
    r"i'?m (open|okay|fine)|let'?s (continue|keep going))\b",
    re.IGNORECASE,
)
_STOP_SIGNALS = re.compile(
    r"\b(no\b|nope|stop|done|that'?s (all|enough|it)|"
    r"i'?m (good|done|fine|finished|all set|wiped|tired|beat|exhausted|spent)|"
    r"wrap (it|this )?up|finish|end|"
    r"enough|quit|nothing (more|else)|let'?s (stop|end|wrap)|call it)\b",
    re.IGNORECASE,
)
# Conservative: bare "no" excluded so a negation during a normal answer doesn't end it.
_VOLUNTEERED_STOP = re.compile(
    r"\b("
    r"that'?s (all|it|enough)( for me| for now| i('?ve| have) got)?"
    r"|i'?m (done|finished|all set)( here| now| with this)?"
    r"|i'?m good,? thanks"
    r"|(let'?s|can we|i'?d like to|i want to|i'?d rather) (stop|wrap (it|this )?up|finish|end (this|it|the interview)|be done)"
    r"|nothing (else|more)( to (add|say))?"
    r"|i (don'?t|do not) want to (continue|go on|keep going)"
    r"|i (have|need) to (go|run|leave|head out)"
    r"|wrap (it|this) up"
    r"|stop the interview"
    r")\b",
    re.IGNORECASE,
)


def _parse_continue_intent(text: str) -> str:
    stop = bool(_STOP_SIGNALS.search(text or ""))
    go = bool(_CONTINUE_SIGNALS.search(text or ""))
    if stop and not go:
        return "stop"
    if go and not stop:
        return "continue"
    return "unclear"


@dataclass
class Directive:
    action: str                     # normal | scripted_offer | scripted_wind_down | end_now
    text: Optional[str] = None      # scripted interviewer text (offer / final / closing)
    resume_note: str = ""           # appended to interviewer prompt when respondent continues
    end_after_answer: bool = False  # wind-down: end once the respondent's next answer arrives
    reason: str = ""


class ConversationCloser:
    # states
    ACTIVE = "active"
    AWAITING_CLOSE_REPLY = "awaiting_close_reply"
    WINDING_DOWN = "winding_down"
    ENDED = "ended"

    def __init__(self, topic_name: str, config: dict = None,
                 interview_session=None):
        self.topic_name = (topic_name or "this").strip()
        self.state = self.ACTIVE
        cfg = config or {}
        self.interview_session = interview_session
        # Length check-in: after this many respondent turns, offer to continue/wrap.
        # 0 disables. Re-offer no more often than every `recheck_interval` turns.
        self.soft_turn_budget = int(cfg.get("soft_turn_budget",
                                            os.getenv("CLOSER_SOFT_TURN_BUDGET", "14")))
        self.recheck_interval = int(cfg.get("recheck_interval",
                                            os.getenv("CLOSER_RECHECK_INTERVAL", "6")))
        self._last_offer_turn = 0
        self.close_offers = 0
        self.pivots = 0
        self.quality_collapse_ends = 0
        # Optional writer for LLM-generated close/wrap-up lines. Instantiated
        # lazily so the closer still works in eval harnesses that don't wire it.
        self._writer: Optional["_CloseLineWriter"] = None
        # Stance-collapse detector (fed by tracker.respondent_stance)
        self._stance_collapse_streak = 0
        self.stance_collapse_threshold = float(
            os.getenv("CLOSER_STANCE_COLLAPSE_THRESHOLD", "0.25"))
        self.stance_collapse_streak = int(
            os.getenv("CLOSER_STANCE_COLLAPSE_STREAK", "3"))

    # The fixed lines we fall back on when offering to wrap up or ending.
    def _close_offer(self) -> str:
        return (f"We've covered a good amount on {self.topic_name}. Would you like to "
                f"keep going with a few more questions, or would you prefer to wrap up?")

    def _length_checkin(self) -> str:
        return (f"We've been talking for a while now. Would you still like to continue, "
                f"or would you rather wrap up here?")

    def _natural_final(self) -> str:
        return ("Before we finish — is there anything else you'd want to add that we "
                "haven't covered?")

    def _severe_closing(self) -> str:
        return "Okay — we'll stop the interview here. Thank you for your time."

    def _resume_note(self, tracker=None) -> str:
        base = (" [MANDATORY DIMENSION CHANGE: The respondent has agreed to continue. "
                "Ask about a completely new, unexplored dimension of the topic. Do not "
                "reference the last several exchanges.")
        # Feed the tracker's actual loose ends in so the LLM has concrete alternatives.
        if tracker is not None:
            ends = tracker.loose_ends() if hasattr(tracker, "loose_ends") else []
            if ends:
                bullets = "\n".join(
                    f"   - {le.get('thread','').strip()}"
                    for le in ends[:4] if le.get("thread")
                )
                if bullets:
                    base += (
                        " Prefer picking up one of these unexplored threads:\n"
                        f"{bullets}"
                    )
        base += "]"
        return base

    async def direct(self, user_answer: Optional[str], monitor: EngagementMonitor,
                     transcript_tail: str, turn_count: int) -> Directive:
        """Decide the next interviewer action. Called at the top of each turn."""

        # Very first turn (interviewer opens): nothing to close on.
        if user_answer is None:
            return Directive(action="normal")

        tracker = self._tracker()

        # We already asked the final question last turn; this answer ends it.
        if self.state == self.WINDING_DOWN:
            self.state = self.ENDED
            return Directive(action="end_now", reason="wind_down_complete")

        # We asked "continue or wrap up?" last turn; interpret the reply.
        if self.state == self.AWAITING_CLOSE_REPLY:
            intent = _parse_continue_intent(user_answer)
            if _VOLUNTEERED_STOP.search(user_answer or "") or intent == "stop":
                self.state = self.WINDING_DOWN
                final_line = await self._compose_final_question(user_answer, transcript_tail, tracker) \
                             or self._natural_final()
                return Directive(action="scripted_wind_down", text=final_line,
                                 end_after_answer=True, reason="declined_offer")
            # continue or unclear -> resume with a mandatory pivot to a fresh dimension
            self.state = self.ACTIVE
            self.pivots += 1
            return Directive(action="normal", resume_note=self._resume_note(tracker),
                             reason="accepted_offer")

        # --- state == ACTIVE ---

        # 1) Respondent explicitly volunteered a stop mid-interview.
        if _VOLUNTEERED_STOP.search(user_answer or ""):
            self.state = self.WINDING_DOWN
            final_line = await self._compose_final_question(user_answer, transcript_tail, tracker) \
                         or self._natural_final()
            return Directive(action="scripted_wind_down", text=final_line,
                             end_after_answer=True, reason="volunteered_stop")

        # 2) Quality-collapse early exit (stance-based). If the tracker reports
        # respondent cooperativeness/specificity persistently in the tank, we
        # end gracefully instead of grinding out more turns. Separate from the
        # monitor's severe-breakdown path, which is affect-based.
        if self._stance_collapse_triggered(tracker):
            self.state = self.ENDED
            self.quality_collapse_ends += 1
            closing = await self._compose_natural_close(user_answer, transcript_tail, tracker,
                                                       reason_hint="quality_collapse") \
                      or self._severe_closing()
            return Directive(action="end_now", text=closing,
                             reason="quality_collapse")

        # 3) LLM breakdown gate (throttled; only when cheap signals already look bad).
        if monitor.should_consider_llm_check():
            verdict = await monitor.diagnose_breakdown(transcript_tail)
            if verdict.severity == "severe":
                self.state = self.ENDED
                closing = await self._compose_natural_close(user_answer, transcript_tail, tracker,
                                                           reason_hint="severe_breakdown") \
                          or self._severe_closing()
                return Directive(action="end_now", text=closing,
                                 reason=f"severe_breakdown: {verdict.reason}")
            if verdict.severity == "mild":
                self.state = self.AWAITING_CLOSE_REPLY
                self.close_offers += 1
                self._last_offer_turn = turn_count
                offer = await self._compose_offer(transcript_tail, tracker,
                                                  reason_hint="mild_breakdown") \
                        or self._close_offer()
                return Directive(action="scripted_offer", text=offer,
                                 reason=f"mild_breakdown: {verdict.reason}")

        # 4) Disengagement streak -> offer to close.
        if monitor.streak >= DISENGAGE_THRESHOLD:
            self.state = self.AWAITING_CLOSE_REPLY
            self.close_offers += 1
            self._last_offer_turn = turn_count
            offer = await self._compose_offer(transcript_tail, tracker,
                                              reason_hint="disengagement_streak") \
                    or self._close_offer()
            return Directive(action="scripted_offer", text=offer,
                             reason="disengagement_streak")

        # 5) Length check-in -> "we've talked a while, continue?"
        if (self.soft_turn_budget and turn_count >= self.soft_turn_budget
                and turn_count - self._last_offer_turn >= self.recheck_interval):
            self.state = self.AWAITING_CLOSE_REPLY
            self.close_offers += 1
            self._last_offer_turn = turn_count
            offer = await self._compose_offer(transcript_tail, tracker,
                                              reason_hint="length_checkin") \
                    or self._length_checkin()
            return Directive(action="scripted_offer", text=offer,
                             reason="length_checkin")

        return Directive(action="normal")

    # ---- LLM composition helpers ------------------------------------------------

    def _tracker(self):
        return getattr(self.interview_session, "interview_tracker", None) \
            if self.interview_session else None

    def _ensure_writer(self) -> Optional["_CloseLineWriter"]:
        if self._writer is not None:
            return self._writer
        try:
            self._writer = _CloseLineWriter(topic=self.topic_name)
            return self._writer
        except Exception as e:
            SessionLogger.log_to_file(
                "execution_log", f"[CLOSER] writer init failed: {e}"
            )
            return None

    async def _compose_offer(self, transcript_tail: str, tracker,
                             reason_hint: str) -> str:
        writer = self._ensure_writer()
        if writer is None:
            return ""
        loose_ends = tracker.loose_ends() if tracker else []
        return await writer.write_offer(
            transcript_tail=transcript_tail,
            loose_ends=loose_ends,
            reason_hint=reason_hint,
        )

    async def _compose_final_question(self, user_answer: str, transcript_tail: str,
                                      tracker) -> str:
        writer = self._ensure_writer()
        if writer is None:
            return ""
        loose_ends = tracker.loose_ends() if tracker else []
        return await writer.write_final_question(
            user_answer=user_answer,
            transcript_tail=transcript_tail,
            loose_ends=loose_ends,
        )

    async def _compose_natural_close(self, user_answer: str, transcript_tail: str,
                                     tracker, reason_hint: str) -> str:
        writer = self._ensure_writer()
        if writer is None:
            return ""
        return await writer.write_close(
            transcript_tail=transcript_tail,
            reason_hint=reason_hint,
        )

    def _stance_collapse_triggered(self, tracker) -> bool:
        if tracker is None:
            return False
        stance = getattr(tracker.state, "respondent_stance", None) or {}
        coop = float(stance.get("cooperativeness", 0.5))
        spec = float(stance.get("specificity", 0.5))
        thr = self.stance_collapse_threshold
        # Both low -> increment streak, else reset. Streak persists across turns
        # via self._stance_collapse_streak; when it hits threshold, fire once.
        if coop <= thr and spec <= thr:
            self._stance_collapse_streak += 1
        else:
            self._stance_collapse_streak = 0
        return self._stance_collapse_streak >= self.stance_collapse_streak

    def stats(self) -> dict:
        return {"close_offers": self.close_offers, "pivots": self.pivots,
                "quality_collapse_ends": self.quality_collapse_ends,
                "final_state": self.state}


# ---- LLM writer for close-related lines ----------------------------------------

class _CloseLineWriter(BaseAgent):
    """Tiny agent that writes offer/final/close lines. Kept out of the state
    machine so ConversationCloser stays deterministic and testable."""

    def __init__(self, topic: str):
        cfg = {}
        model = os.getenv("CLOSER_WRITER_MODEL_NAME")
        if model:
            cfg["model_name"] = model
        BaseAgent.__init__(
            self, name="CloserWriter",
            description="Writes interviewer-voice close/offer/final lines.",
            config=cfg,
        )
        self.topic = topic

    async def write_offer(self, transcript_tail: str, loose_ends: list,
                          reason_hint: str) -> str:
        prompt = (
            "You write a SINGLE line for a research interviewer. Voice: neutral, "
            "non-affirming, not warm-fuzzy. No thanks, no praise, no advice.\n\n"
            f"Interview topic: {self.topic}\n"
            f"Trigger reason: {reason_hint}\n\n"
            "Recent conversation (most recent last):\n"
            f"{transcript_tail}\n\n"
            "Loose ends the tracker has flagged (may be empty):\n"
            f"{_format_loose_ends(loose_ends)}\n\n"
            "Write a one-sentence question offering the respondent the choice to "
            "keep going or wrap up. If there is a specific unexplored thread worth "
            "naming, mention it briefly. PREFER threads tagged [respondent] — the "
            "respondent already hinted at these and naming them lands harder. Only "
            "reach for a [briefing] thread if no respondent-sourced one exists. "
            "Format: \"You mentioned X but we didn't get into <specifics> — do you "
            "want to say more, or wrap up here?\" Otherwise ask cleanly. No preamble, "
            "no quotes, no lists. Output ONLY the line."
        )
        return await self._call(prompt, label="offer")

    async def write_final_question(self, user_answer: str, transcript_tail: str,
                                   loose_ends: list) -> str:
        prompt = (
            "You write a SINGLE closing question for a research interviewer, said "
            "AFTER the respondent has agreed to end. Voice: neutral, non-affirming.\n\n"
            f"Interview topic: {self.topic}\n\n"
            "Recent conversation:\n"
            f"{transcript_tail}\n\n"
            f"Respondent's most recent turn: {user_answer!r}\n\n"
            "Loose ends the tracker has flagged (may be empty):\n"
            f"{_format_loose_ends(loose_ends)}\n\n"
            "Write ONE final open-ended question. If there is a genuine loose end, "
            "name it: \"Before we finish — you mentioned X earlier; anything else "
            "on that you want to add?\" If no clean loose end, ask the generic "
            "\"anything we haven't covered you'd want to add?\" Do NOT thank. Do NOT "
            "recap. Output ONLY the question."
        )
        return await self._call(prompt, label="final_question")

    async def write_close(self, transcript_tail: str, reason_hint: str) -> str:
        prompt = (
            "You write the interviewer's LAST line. Voice: neutral, brief.\n\n"
            f"Trigger reason: {reason_hint}\n"
            f"Interview topic: {self.topic}\n\n"
            "Recent conversation:\n"
            f"{transcript_tail}\n\n"
            "Write 1–2 sentences ending the interview. Do NOT thank effusively. "
            "Do NOT summarize what the respondent said. One quiet close. Output ONLY the line(s)."
        )
        return await self._call(prompt, label="close")

    async def _call(self, prompt: str, label: str) -> str:
        try:
            raw = (await self.call_engine_async(prompt) or "").strip()
        except Exception as e:
            SessionLogger.log_to_file("execution_log",
                                     f"[CLOSER] writer.{label} failed: {e}")
            return ""
        # Strip stray leading/trailing quotes the model sometimes adds.
        return raw.strip().strip('"').strip("'").strip()


def _format_loose_ends(loose_ends: list) -> str:
    if not loose_ends:
        return "(none)"
    # Respondent-sourced loose ends are stronger — the respondent already hinted
    # at these, so naming them lands harder than surfacing a briefing fact they
    # never mentioned. Sort them to the top.
    def _rank(le: dict) -> int:
        return 0 if (le.get("source") or "").lower() == "respondent" else 1
    ranked = sorted(loose_ends[:12], key=_rank)
    lines = []
    for le in ranked[:6]:
        thread = (le.get("thread") or "").strip()
        why = (le.get("why_worth_pulling") or "").strip()
        src = (le.get("source") or "").strip().lower()
        tag = f" [{src}]" if src in ("respondent", "briefing") else ""
        if thread:
            lines.append(f"- {thread}{tag}" + (f" — {why}" if why else ""))
    return "\n".join(lines) if lines else "(none)"
