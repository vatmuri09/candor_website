"""
ContextBiasAgent — evaluates compiled context/source material for bias.

Task: when an interview is seeded with external context (the additional-context
file, or, in future, retrieved sources), that material can smuggle slant into the
interviewer's questions — a confound for political-science data. This agent
produces a bias report so the slant is measured and attributable, WITHOUT
scrubbing the substance. Vikram's constraint: mitigate bias, don't erase the
interesting/meaty details.

Two layers:

  1. Deterministic lexical report (free, no network): VADER polarity, a
     subjectivity estimate, and hits against curated loaded-language / partisan /
     hedging lexicons. Good for a quick, reproducible numeric signal and for
     flagging obviously slanted material even with no LLM available.

  2. LLM analysis (paid, optional): separates established fact from contested
     framing, estimates a lean, lists loaded phrasings, and returns a
     `neutralized_context` that keeps every substantive detail but attributes or
     de-editorializes the framing. This is the version safe to feed the
     interviewer; the original is preserved alongside for provenance.

On why not an off-the-shelf classifier: robust political-bias models (e.g. HF
transformer classifiers) pull in torch and are heavy for a CPU web deploy, and
still don't do the fact-vs-framing separation we actually need. The lexical layer
covers the cheap signal; the LLM covers the nuanced separation.
"""
import json
import re
from dataclasses import dataclass, field, asdict
from typing import List, Optional

from src.agents.base_agent import BaseAgent
from src.agents.engagement.sentiment import sentiment_compound
from src.utils.logger.session_logger import SessionLogger

# Curated lexicons (small, transparent, editable). Not exhaustive — a cheap prior.
_LOADED_TERMS = re.compile(
    r"\b(radical|extremist|regime|thug|terrorist|patriot|freedom[- ]loving|"
    r"woke|snowflake|libtard|rino|fascist|nazi|communist|socialist agenda|"
    r"far[- ]left|far[- ]right|mainstream media|deep state|witch hunt|hoax|"
    r"invasion|flood of|handout|elite|establishment|dangerous|catastrophic|"
    r"disastrous|shocking|slammed|blasted|destroyed|owned|debunked|"
    r"so[- ]called|allegedly|reportedly|claims to|failed|corrupt|scandal[- ]ridden)\b",
    re.IGNORECASE,
)
_SUBJECTIVE = re.compile(
    r"\b(clearly|obviously|undoubtedly|certainly|of course|everyone knows|"
    r"the truth is|the fact is|no one can deny|it'?s obvious|shamefully|"
    r"outrageous|disgraceful|heroic|brave|cowardly|beautiful|terrible|"
    r"amazing|horrific|tragic|wonderful|awful|best|worst|must|should|"
    r"deserve|unfair|unjust|righteous|evil|good|bad)\b",
    re.IGNORECASE,
)
_HEDGES = re.compile(
    r"\b(some say|critics argue|supporters claim|it is believed|arguably|"
    r"may|might|could|possibly|perhaps|reportedly|allegedly|it seems|"
    r"according to (some|reports|sources))\b",
    re.IGNORECASE,
)


@dataclass
class LexicalBiasReport:
    polarity: float = 0.0              # VADER compound, [-1, 1]
    subjectivity: float = 0.0         # subjective-marker density, [0, ~1]
    loaded_terms: List[str] = field(default_factory=list)
    subjective_markers: List[str] = field(default_factory=list)
    hedges: List[str] = field(default_factory=list)
    n_words: int = 0

    @property
    def slant_score(self) -> float:
        """0 (neutral) .. 1 (heavily slanted): blend of polarity magnitude,
        subjectivity, and loaded-term density."""
        loaded_density = len(self.loaded_terms) / max(1, self.n_words / 100)
        return round(min(1.0, 0.5 * abs(self.polarity) + 0.3 * self.subjectivity
                         + 0.2 * min(1.0, loaded_density)), 3)


def lexical_bias_report(text: str) -> LexicalBiasReport:
    text = text or ""
    words = re.findall(r"[A-Za-z']+", text)
    n = len(words)
    loaded = [m.group(0).lower() for m in _LOADED_TERMS.finditer(text)]
    subj = [m.group(0).lower() for m in _SUBJECTIVE.finditer(text)]
    hedges = [m.group(0).lower() for m in _HEDGES.finditer(text)]
    subjectivity = min(1.0, (len(subj) + len(loaded)) / max(1, n / 25))
    return LexicalBiasReport(
        polarity=round(sentiment_compound(text), 3),
        subjectivity=round(subjectivity, 3),
        loaded_terms=sorted(set(loaded)),
        subjective_markers=sorted(set(subj)),
        hedges=sorted(set(hedges)),
        n_words=n,
    )


class ContextBiasAgent(BaseAgent):
    """Analyzes compiled context for bias. One-shot per piece of source material."""

    def __init__(self, config: dict = None, interview_session=None):
        BaseAgent.__init__(
            self, name="ContextBiasAgent",
            description="Evaluates compiled context/source material for bias.",
            config=config or {},
        )
        self.interview_session = interview_session

    async def analyze_llm(self, text: str) -> Optional[dict]:
        """LLM fact-vs-framing separation. Returns None on failure."""
        if not text or not text.strip():
            return None
        prompt = (
            "You are a neutral media-bias analyst for a political-science study.\n"
            "Analyze the CONTEXT below. Your job is to make slant visible and to "
            "produce a neutral version that KEEPS ALL substantive, interesting detail "
            "(numbers, events, specific claims, competing viewpoints) while removing or "
            "attributing editorializing. Do NOT blandify away meaningful content.\n\n"
            "Return ONLY a JSON object with these keys:\n"
            '  "estimated_lean": one of "left","center-left","center","center-right","right","unclear"\n'
            '  "confidence": 0.0-1.0\n'
            '  "established_facts": [short strings — claims stated neutrally and widely verifiable]\n'
            '  "contested_framings": [short strings — claims/framings that are opinion or disputed]\n'
            '  "loaded_language": [exact loaded words/phrases found]\n'
            '  "balance_note": one sentence on whether multiple sides are represented\n'
            '  "neutralized_context": a rewrite that preserves all substantive detail but '
            "attributes or removes slant (keep it roughly the same length; do not shorten by cutting facts)\n\n"
            f"CONTEXT:\n{text}\n"
        )
        try:
            raw = (await self.call_engine_async(prompt) or "").strip()
        except Exception as e:
            SessionLogger.log_to_file("execution_log", f"[CONTEXT_BIAS] LLM analyze failed: {e}")
            return None
        # Be tolerant of code fences / stray prose around the JSON.
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if not m:
            SessionLogger.log_to_file("execution_log", f"[CONTEXT_BIAS] no JSON in response: {raw[:200]!r}")
            return None
        try:
            return json.loads(m.group(0))
        except Exception as e:
            SessionLogger.log_to_file("execution_log", f"[CONTEXT_BIAS] JSON parse failed: {e}")
            return None

    async def report(self, text: str, use_llm: bool = True) -> dict:
        """Full report: deterministic lexical layer always, LLM layer if enabled."""
        lex = lexical_bias_report(text)
        out = {"lexical": asdict(lex), "slant_score": lex.slant_score, "llm": None}
        if use_llm:
            out["llm"] = await self.analyze_llm(text)
        SessionLogger.log_to_file(
            "execution_log",
            f"[CONTEXT_BIAS] slant={lex.slant_score} polarity={lex.polarity} "
            f"loaded={lex.loaded_terms} lean={(out['llm'] or {}).get('estimated_lean')}"
        )
        return out
