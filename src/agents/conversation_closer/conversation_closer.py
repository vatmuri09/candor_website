"""Decides when to wrap up the interview.

Before each interviewer turn this looks at the latest answer and the engagement
monitor and picks one of: keep going, offer to wrap up, ask a final "anything to
add?" then end, or (if things have gone really badly) end right away. It's a
small state machine and doesn't send its own messages.
"""
import os
import re
from dataclasses import dataclass
from typing import Optional

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

    def __init__(self, topic_name: str, config: dict = None):
        self.topic_name = (topic_name or "this").strip()
        self.state = self.ACTIVE
        cfg = config or {}
        # Length check-in: after this many respondent turns, offer to continue/wrap.
        # 0 disables. Re-offer no more often than every `recheck_interval` turns.
        self.soft_turn_budget = int(cfg.get("soft_turn_budget",
                                            os.getenv("CLOSER_SOFT_TURN_BUDGET", "14")))
        self.recheck_interval = int(cfg.get("recheck_interval",
                                            os.getenv("CLOSER_RECHECK_INTERVAL", "6")))
        self._last_offer_turn = 0
        self.close_offers = 0
        self.pivots = 0

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

    def _resume_note(self) -> str:
        return (" [MANDATORY DIMENSION CHANGE: The respondent has agreed to continue. "
                "Ask about a completely new, unexplored dimension of the topic. Do not "
                "reference the last several exchanges.]")

    async def direct(self, user_answer: Optional[str], monitor: EngagementMonitor,
                     transcript_tail: str, turn_count: int) -> Directive:
        """Decide the next interviewer action. Called at the top of each turn."""

        # Very first turn (interviewer opens): nothing to close on.
        if user_answer is None:
            return Directive(action="normal")

        # We already asked the final question last turn; this answer ends it.
        if self.state == self.WINDING_DOWN:
            self.state = self.ENDED
            return Directive(action="end_now", reason="wind_down_complete")

        # We asked "continue or wrap up?" last turn; interpret the reply.
        if self.state == self.AWAITING_CLOSE_REPLY:
            intent = _parse_continue_intent(user_answer)
            if _VOLUNTEERED_STOP.search(user_answer or "") or intent == "stop":
                self.state = self.WINDING_DOWN
                return Directive(action="scripted_wind_down", text=self._natural_final(),
                                 end_after_answer=True, reason="declined_offer")
            # continue or unclear -> resume with a mandatory pivot to a fresh dimension
            self.state = self.ACTIVE
            self.pivots += 1
            return Directive(action="normal", resume_note=self._resume_note(),
                             reason="accepted_offer")

        # --- state == ACTIVE ---

        # 1) Respondent explicitly volunteered a stop mid-interview.
        if _VOLUNTEERED_STOP.search(user_answer or ""):
            self.state = self.WINDING_DOWN
            return Directive(action="scripted_wind_down", text=self._natural_final(),
                             end_after_answer=True, reason="volunteered_stop")

        # 2) LLM breakdown gate (throttled; only when cheap signals already look bad).
        if monitor.should_consider_llm_check():
            verdict = await monitor.diagnose_breakdown(transcript_tail)
            if verdict.severity == "severe":
                self.state = self.ENDED
                return Directive(action="end_now", text=self._severe_closing(),
                                 reason=f"severe_breakdown: {verdict.reason}")
            if verdict.severity == "mild":
                self.state = self.AWAITING_CLOSE_REPLY
                self.close_offers += 1
                self._last_offer_turn = turn_count
                return Directive(action="scripted_offer", text=self._close_offer(),
                                 reason=f"mild_breakdown: {verdict.reason}")

        # 3) Disengagement streak -> offer to close.
        if monitor.streak >= DISENGAGE_THRESHOLD:
            self.state = self.AWAITING_CLOSE_REPLY
            self.close_offers += 1
            self._last_offer_turn = turn_count
            return Directive(action="scripted_offer", text=self._close_offer(),
                             reason="disengagement_streak")

        # 4) Length check-in -> "we've talked a while, continue?"
        if (self.soft_turn_budget and turn_count >= self.soft_turn_budget
                and turn_count - self._last_offer_turn >= self.recheck_interval):
            self.state = self.AWAITING_CLOSE_REPLY
            self.close_offers += 1
            self._last_offer_turn = turn_count
            return Directive(action="scripted_offer", text=self._length_checkin(),
                             reason="length_checkin")

        return Directive(action="normal")

    def stats(self) -> dict:
        return {"close_offers": self.close_offers, "pivots": self.pivots,
                "final_state": self.state}
