"""Helpers to keep the interviewer from being sycophantic.

We strip praise/affirmations off the front of a generated turn and flag turns
that don't ask a question, state an opinion, or give advice. It's all plain regex
so it runs every turn without an extra LLM call.
"""
import re
from dataclasses import dataclass, field

# Evaluative adjectives an interviewer should never apply to an answer.
_EVAL_ADJ = (
    r"understandable|great|good|nice|wonderful|fantastic|amazing|excellent|"
    r"awesome|fascinating|interesting|insightful|impressive|admirable|inspiring|"
    r"valuable|important|helpful|reasonable|fair|valid|true|correct|right|"
    r"thoughtful|remarkable|incredible|beautiful|lovely|delightful|powerful|"
    r"compelling|refreshing|profound|brilliant|smart|wise|meaningful|healthy|"
    r"perfect|solid|clear|cool|sensible|efficient|sound"
)

# Leading evaluative openers: "That's great.", "What a fascinating point.",
# "Absolutely!", "I love that.", "That sounds like a great...", etc.
_NOUN = (
    r"(?: (point|question|observation|example|insight|choice|one|stuff|work|"
    r"thinking|answer|response|attitude|approach|mindset|perspective|routine|story))?"
)
_AFFIRMATION_OPENERS = re.compile(
    r"^\s*("
    r"that('?s| is| was| sounds( like)?| must (have )?(be|been)) (a |an |really |very |so |quite |such )*"
    rf"(?:{_EVAL_ADJ})[^.!?]*"
    r"|what (a|an) (really |very |truly )?(?:" + _EVAL_ADJ + r")[^.!?]*"
    r"|(?:i |we )?(really )?(love|like|appreciate|admire|respect) (that|hearing|your|how|what)[^.!?]*"
    r"|(?:i'?m|we'?re) (so |really |very )?(glad|happy|thrilled|impressed|excited)[^.!?]*"
    r"|absolutely|certainly|definitely|totally|for sure|of course"
    r"|(?:wow|amazing|incredible|impressive|fascinating|interesting|lovely|beautiful|"
    rf"nice|great|awesome|excellent|perfect|good|cool){_NOUN}"
    r"|indeed|absolutely right|so true|well said"
    r"|(?:it'?s|that'?s) (great|good|nice|wonderful|impressive|refreshing) (to (hear|know)|that)[^.!?]*"
    r"|(?:kudos|congrats|congratulations|hats off|good on you)[^.!?]*"
    r"|i (can|could) (see|tell|imagine) (why|that|how)[^.!?]*"
    r"|that makes (a lot of )?sense[^.!?]*"
    r"|i hear you|i see|i understand|understood|got it|gotcha|noted|fair enough|makes sense|sounds good"
    r")"
    r"[.,!:—-]*\s*",
    re.IGNORECASE,
)

# Service-register / closing pleasantries.
_CLOSING_LANGUAGE = re.compile(
    r"^\s*("
    r"thank(s| you)( so much| very much)?( for (sharing|that|your (time|answer|honesty|openness)|being (so )?open))?[^.!?]*"
    r"|you'?re (welcome|very welcome)[^.!?]*"
    r"|(feel free|don'?t hesitate|please feel welcome) to[^.!?]*"
    r"|have a (great|good|wonderful|nice|lovely) (day|one|rest of)[^.!?]*"
    r"|take (care|breaks|your time)[^.!?]*"
    r"|(i )?(really )?appreciate (you|your|it|that)[^.!?]*"
    r"|it'?s been (a )?(great|wonderful|nice|real) (pleasure|talking|chatting)[^.!?]*"
    r")"
    r"[.,!:—-]*\s*",
    re.IGNORECASE,
)

# A short, permitted neutral acknowledgment (kept, never flagged).
_NEUTRAL_ACK = re.compile(r"^\s*(okay|ok|alright|right|so|now|i see)[.,]?\s*", re.IGNORECASE)


def _strip_repeatedly(pattern: re.Pattern, text: str, max_iters: int = 4) -> tuple[str, bool]:
    """Strip a leading pattern, possibly chained (e.g. 'Great! Thanks for sharing.')."""
    fired = False
    cleaned = text
    for _ in range(max_iters):
        new = pattern.sub("", cleaned, count=1).strip()
        if new == cleaned:
            break
        cleaned = new
        fired = True
    return (cleaned if cleaned else text), fired


def sanitize_interviewer_turn(text: str) -> tuple[str, list]:
    """Strip sycophantic openers and closing pleasantries. Returns (clean, flags).

    Interleaves the two strippers so combos like "Thanks for sharing — that's a
    fascinating point." are fully removed regardless of order.
    """
    if not text:
        return text, []
    flags: list[str] = []
    cleaned = text.strip()
    for _ in range(4):
        cleaned, aff = _strip_repeatedly(_AFFIRMATION_OPENERS, cleaned, 1)
        if aff and "affirmation" not in flags:
            flags.append("affirmation")
        cleaned, clo = _strip_repeatedly(_CLOSING_LANGUAGE, cleaned, 1)
        if clo and "closing" not in flags:
            flags.append("closing")
        if not aff and not clo:
            break
    return (cleaned if cleaned.strip() else text), flags


# First-person stance / opinion — the interviewer must not editorialize.
_STANCE = re.compile(
    r"\b("
    r"i (think|believe|feel|reckon|suspect|figure|would (say|argue|contend)|'?d (say|argue))\b"
    r"|in my (opinion|view|experience|mind)"
    r"|personally,?\s"
    r"|if you ask me"
    r"|my (take|view|opinion|sense|stance) (is|would be)"
    r"|(i'?m|i am) (a (big )?)?(fan|supporter|believer|proponent|critic|skeptic) of"
    r"|i (agree|disagree) (with|that)"
    r"|(honestly|frankly),? i"
    r"|the (truth|reality|fact) is( that)?"
    r"|it'?s (clear|obvious|true) that"
    r")\b",
    re.IGNORECASE,
)

# Advice / recommendation / instruction directed at the respondent.
_ADVICE = re.compile(
    r"\b("
    r"you should|you ought to|you (could|might want to|may want to) (try|consider|think about|look into)"
    r"|(my|a) (advice|recommendation|suggestion) (is|would be)"
    r"|i('?d| would) (recommend|suggest|advise|encourage you)"
    r"|(try|consider) (to |)(doing|thinking|talking|reaching|setting|making)"
    r"|(it (might|would|could) (be )?(help|be worth|be good)|it helps) (to|if you)"
    r"|have you (tried|considered) (?!.*\?)"  # advice framed as a rhetorical question
    r"|the best (way|thing) (to|for you|would be)"
    r"|what i('?d| would) do is"
    r")\b",
    re.IGNORECASE,
)

_QUESTION_WORD = re.compile(
    r"\b(what|why|how|when|where|which|who|whom|whose|can you|could you|would you|"
    r"do you|did you|have you|has|is there|are there|was there|were there|tell me|"
    r"describe|walk me through|talk me through|share)\b",
    re.IGNORECASE,
)


def contains_question(text: str) -> bool:
    """True if the turn actually asks something (a '?' or an interrogative frame)."""
    if not text:
        return False
    if "?" in text:
        return True
    # Some legitimate prompts end without '?': "Tell me about...", "Describe..."
    last = text.strip().split(".")[-1] if "." in text else text
    return bool(_QUESTION_WORD.search(text)) and bool(_QUESTION_WORD.search(last or text))


def detect_stance(text: str) -> bool:
    return bool(_STANCE.search(text or ""))


def detect_advice(text: str) -> bool:
    return bool(_ADVICE.search(text or ""))


@dataclass
class TurnInspection:
    clean_text: str
    flags: list = field(default_factory=list)          # sycophancy strips that fired
    violations: list = field(default_factory=list)     # hard guardrail failures
    needs_regeneration: bool = False


def inspect_turn(text: str) -> TurnInspection:
    """Sanitize a turn, then run guardrails. If a hard violation remains, the
    caller should regenerate once (see REGEN_REMINDER) before using clean_text.
    """
    clean, flags = sanitize_interviewer_turn(text)
    violations = []
    if not contains_question(clean):
        violations.append("no_question")
    if detect_stance(clean):
        violations.append("stance")
    if detect_advice(clean):
        violations.append("advice")
    return TurnInspection(
        clean_text=clean,
        flags=flags,
        violations=violations,
        needs_regeneration=bool(violations),
    )


# Appended to the prompt when regenerating a turn that tripped a guardrail.
REGEN_REMINDER = (
    "\n\n[GUARDRAIL — your previous draft violated the interviewer stance "
    "({violations}). Rewrite it as ONE neutral, open-ended question only. "
    "Do NOT evaluate, affirm, praise, or thank the respondent. Do NOT state your "
    "own opinion, belief, or stance. Do NOT give advice or recommendations. Do NOT "
    "answer any question they asked you — if they asked you something, simply ask "
    "your next question. Output only the question.]"
)


def regen_reminder(violations: list) -> str:
    return REGEN_REMINDER.format(violations=", ".join(violations) or "non-affirming stance")
