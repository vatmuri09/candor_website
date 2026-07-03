"""
EngagementMonitor — quality-of-conversation monitor (bot #1 of the two new bots).

Ported and extended from candor/agents/quality_monitor.py. Combines cheap,
deterministic engagement signals with VADER sentiment and an optional LLM
"is this conversation broken" gate:

  Deterministic (free, every turn):
    - dismissal phrases ("i don't know", "whatever", "not really", ...)
    - lexical repetition vs. the previous answer (Jaccard >= 0.42)
    - relative deterioration: answer < 0.5x the respondent's OWN running median
      length, once a baseline of >=3 answers exists (NOT an absolute cutoff —
      terse-but-engaged answers must not be punished; see candor batch-test notes)

  NLP (free, every turn):
    - VADER compound sentiment, and a short rolling trend so a slide from
      positive -> negative registers even when no dismissal phrase is present.

  LLM gate (paid, throttled): diagnose_breakdown() asks the model whether the
  conversation has degraded to the point it should end (respondent hostile,
  checked-out, nonsensical, or refusing). Only called when cheap signals already
  look bad, so cost stays near zero on healthy interviews.

The monitor holds per-session state (streak, answer history, sentiment trend)
and is *consulted* by the Interviewer/Closer rather than posting its own turns.
"""
import os
import re
import statistics
from dataclasses import dataclass, field
from typing import List

from src.agents.base_agent import BaseAgent
from src.agents.engagement.sentiment import sentiment_compound
from src.utils.logger.session_logger import SessionLogger

_DISMISSAL = re.compile(
    r"\b(i don'?t (really )?know|not really|i don'?t care|whatever|i guess|"
    r"doesn'?t (really )?matter|not sure|no idea|hard to say|"
    r"can'?t say|i already (said|told you)|same (as before|thing)|"
    r"nothing (really|much|new)|i dunno|not much|not a lot|"
    r"not interested|who cares|i (couldn'?t|can'?t) say|"
    r"i'?m not sure|beats me|no clue)\b",
    re.IGNORECASE,
)

REPETITION_THRESHOLD = 0.42
DETERIORATION_RATIO = 0.5
MIN_HISTORY_FOR_BASELINE = 3
NEGATIVE_SENTIMENT = -0.35          # single very-negative answer
SENTIMENT_SLIDE = -0.30            # drop between recent-mean and this answer


@dataclass
class QualitySignal:
    level: str                      # "good" | "flagged" | "declining" | "disengaged"
    triggers: List[str]
    streak: int
    sentiment: float
    sentiment_trend: float          # this answer minus recent mean


@dataclass
class BreakdownVerdict:
    broken: bool = False
    severity: str = "none"          # none | mild | severe
    reason: str = ""


def _jaccard(a: str, b: str) -> float:
    wa, wb = set(a.lower().split()), set(b.lower().split())
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


class EngagementMonitor(BaseAgent):
    """Quality-of-conversation monitor. One instance per interview session."""

    def __init__(self, config: dict = None, interview_session=None):
        BaseAgent.__init__(
            self, name="EngagementMonitor",
            description="Monitors respondent engagement and conversation quality.",
            config=config or {},
        )
        self.interview_session = interview_session
        self.answer_history: List[str] = []
        self.sentiments: List[float] = []
        self.streak = 0
        self.last_signal: QualitySignal = None
        self._turns_since_llm_check = 0

    # ---- deterministic + sentiment assessment (called every user turn) ----
    def observe(self, answer: str) -> QualitySignal:
        answer = answer or ""
        triggers = []
        n_words = len(answer.split())

        sent = sentiment_compound(answer)
        recent_mean = statistics.mean(self.sentiments[-3:]) if self.sentiments else 0.0
        trend = sent - recent_mean

        if _DISMISSAL.search(answer):
            triggers.append("dismissal")
        if self.answer_history and _jaccard(answer, self.answer_history[-1]) >= REPETITION_THRESHOLD:
            triggers.append("repetition")
        if len(self.answer_history) >= MIN_HISTORY_FOR_BASELINE:
            baseline = statistics.median(len(h.split()) for h in self.answer_history)
            if baseline > 0 and n_words < DETERIORATION_RATIO * baseline:
                triggers.append("deteriorating")
        if sent <= NEGATIVE_SENTIMENT:
            triggers.append("negative_sentiment")
        elif len(self.sentiments) >= 2 and trend <= SENTIMENT_SLIDE:
            triggers.append("sentiment_slide")

        # Update state AFTER computing (so repetition/baseline use prior turns).
        self.answer_history.append(answer)
        self.sentiments.append(sent)

        if not triggers:
            self.streak = 0
            level = "good"
        else:
            self.streak += 1
            level = ("disengaged" if self.streak >= 3
                     else "declining" if self.streak >= 2 else "flagged")

        self.last_signal = QualitySignal(
            level=level, triggers=triggers, streak=self.streak,
            sentiment=sent, sentiment_trend=trend,
        )
        if triggers:
            SessionLogger.log_to_file(
                "execution_log",
                f"[ENGAGEMENT] level={level} streak={self.streak} "
                f"triggers={triggers} sentiment={sent:.2f} trend={trend:.2f}"
            )
        return self.last_signal

    def should_consider_llm_check(self) -> bool:
        """Only spend an LLM call when cheap signals already look bad."""
        if self.last_signal is None:
            return False
        self._turns_since_llm_check += 1
        bad = self.last_signal.level in ("declining", "disengaged")
        very_negative = self.last_signal.sentiment <= NEGATIVE_SENTIMENT
        # Throttle: at most once every 2 turns even when bad.
        return (bad or very_negative) and self._turns_since_llm_check >= 2

    # ---- LLM breakdown gate (throttled) ----
    async def diagnose_breakdown(self, transcript_tail: str) -> BreakdownVerdict:
        """Ask the model whether the conversation has broken down badly enough to end.

        Returns severity: 'severe' -> end now, 'mild' -> offer to close, 'none' -> continue.
        """
        self._turns_since_llm_check = 0
        prompt = (
            "You are a neutral research supervisor auditing an interview transcript.\n"
            "Decide whether the conversation has degraded so badly that it should be "
            "ended early. Signs of severe breakdown: the respondent is hostile or abusive, "
            "has completely checked out (repeated 'I don't know'/'whatever' with no content), "
            "is answering nonsensically, is trying to derail or troll, or explicitly refuses "
            "to continue. A respondent who is simply terse, or giving short but real answers, "
            "is NOT a breakdown.\n\n"
            f"Recent transcript:\n{transcript_tail}\n\n"
            "Respond with EXACTLY one line in this format:\n"
            "SEVERITY=<none|mild|severe>; REASON=<short reason>"
        )
        try:
            raw = (await self.call_engine_async(prompt) or "").strip()
        except Exception as e:
            SessionLogger.log_to_file("execution_log", f"[ENGAGEMENT] breakdown gate failed: {e}")
            return BreakdownVerdict()

        sev = "none"
        m = re.search(r"SEVERITY\s*=\s*(none|mild|severe)", raw, re.IGNORECASE)
        if m:
            sev = m.group(1).lower()
        reason = ""
        rm = re.search(r"REASON\s*=\s*(.+)", raw, re.IGNORECASE)
        if rm:
            reason = rm.group(1).strip()[:200]
        SessionLogger.log_to_file("execution_log", f"[ENGAGEMENT] breakdown gate: {raw!r}")
        return BreakdownVerdict(broken=sev in ("mild", "severe"), severity=sev, reason=reason)

    def stats(self) -> dict:
        return {
            "turns_observed": len(self.answer_history),
            "final_streak": self.streak,
            "mean_sentiment": round(statistics.mean(self.sentiments), 3) if self.sentiments else 0.0,
            "min_sentiment": round(min(self.sentiments), 3) if self.sentiments else 0.0,
        }
