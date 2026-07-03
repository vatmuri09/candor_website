"""
Lightweight sentiment / engagement NLP for the EngagementMonitor.

Uses NLTK's VADER (already a project dependency). VADER is a lexicon+rule model
tuned for short, informal text — a good fit for chat answers and, unlike a neural
model, free and instant per turn. The lexicon is lazily loaded and, if missing on
a fresh server, downloaded once; if that fails we fall back to a tiny built-in
polarity heuristic so the interview never breaks over a missing data file.
"""
import re
import threading

_analyzer = None
_lock = threading.Lock()
_fallback = False

# Minimal fallback lexicon (only used if VADER can't be loaded at all).
_POS = {"good", "great", "love", "enjoy", "enjoyed", "happy", "excited", "fun",
        "interesting", "glad", "like", "helpful", "nice", "yes", "definitely"}
_NEG = {"bad", "hate", "boring", "bored", "tired", "annoyed", "annoying",
        "frustrated", "pointless", "stupid", "waste", "whatever", "no", "dont",
        "worse", "worst", "sad", "angry", "sick", "done"}


def _get_analyzer():
    global _analyzer, _fallback
    if _analyzer is not None or _fallback:
        return _analyzer
    with _lock:
        if _analyzer is not None or _fallback:
            return _analyzer
        try:
            from nltk.sentiment.vader import SentimentIntensityAnalyzer
            try:
                _analyzer = SentimentIntensityAnalyzer()
            except LookupError:
                import nltk
                nltk.download("vader_lexicon", quiet=True)
                _analyzer = SentimentIntensityAnalyzer()
        except Exception:
            _fallback = True
            _analyzer = None
        return _analyzer


def _fallback_score(text: str) -> float:
    words = re.findall(r"[a-z']+", (text or "").lower())
    if not words:
        return 0.0
    pos = sum(w in _POS for w in words)
    neg = sum(w in _NEG for w in words)
    if pos == neg:
        return 0.0
    return max(-1.0, min(1.0, (pos - neg) / max(1, pos + neg)))


def sentiment_compound(text: str) -> float:
    """Return a compound polarity in [-1, 1] (VADER's compound, or heuristic)."""
    analyzer = _get_analyzer()
    if analyzer is None:
        return _fallback_score(text)
    try:
        return analyzer.polarity_scores(text or "")["compound"]
    except Exception:
        return _fallback_score(text)
