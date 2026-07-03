"""
SCRATCH / EXPLORATORY — conversation simulation vs. actual divergence.

Task 8: take the "planning" the ExplorationPlanner already does one step further —
actually *simulate* how the conversation could go and measure how close the
simulation lands to what really happened. Two uses:
  * validate the planner's world-model (are its rollouts realistic?), and
  * a first cut at the Task-4 goal: "predict the respondent's answers from the
    conversation so far."

This is deliberately OFFLINE and OPT-IN (not in the live request path) because
it costs one LLM call per simulated turn. Keep it as a research tool; run it on
logged sessions, not in production.

How it works, per User turn in a logged transcript:
  1. Take the conversation prefix up to (and including) the interviewer's
     question.
  2. Ask the model to role-play the respondent and predict their next answer,
     seeded by a short persona inferred from earlier answers.
  3. Score predicted vs. actual answer with embedding cosine similarity
     (EmbeddingService) and token-Jaccard. divergence = 1 - similarity.

Usage:
  python -m evaluation.simulate_conversation <chat_history.log> [--max-turns 5]
  python -m evaluation.simulate_conversation <log> --no-llm   # free; scores
      the trivial "predict = previous answer" baseline to exercise the metric.

Env: OPENAI_API_KEY (for --llm mode and openai embeddings). Use
EMBEDDING_BACKEND=noop to force the free token-Jaccard-only path.
"""
import argparse
import re
import sys
from dataclasses import dataclass
from typing import List, Optional

_LINE = re.compile(r"^\d{4}-\d\d-\d\d[ T].*?- INFO - (Interviewer|User): (.*)$")


@dataclass
class Turn:
    role: str
    text: str


def parse_transcript(path: str) -> List[Turn]:
    """Parse a chat_history.log into ordered Interviewer/User turns.

    Continuation lines (multi-line messages) are appended to the current turn.
    """
    turns: List[Turn] = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n")
            m = _LINE.match(line)
            if m:
                turns.append(Turn(role=m.group(1), text=m.group(2).strip()))
            elif turns and line.strip():
                turns[-1].text += "\n" + line.strip()
    return turns


def _tokens(text: str) -> set:
    return set(re.findall(r"[a-z0-9']+", (text or "").lower()))


def jaccard(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


class Similarity:
    """Embedding cosine similarity with a token-Jaccard fallback."""

    def __init__(self):
        self._svc = None
        try:
            from src.content.embeddings.embedding_service import EmbeddingService
            self._svc = EmbeddingService()
        except Exception as e:
            print(f"[sim] embeddings unavailable ({e}); using token-Jaccard only", file=sys.stderr)

    def cosine(self, a: str, b: str) -> Optional[float]:
        if self._svc is None:
            return None
        try:
            import numpy as np
            va, vb = self._svc.get_embedding(a), self._svc.get_embedding(b)
            na, nb = np.linalg.norm(va), np.linalg.norm(vb)
            if na == 0 or nb == 0:
                return None
            return float(np.dot(va, vb) / (na * nb))
        except Exception:
            return None

    def score(self, a: str, b: str) -> float:
        cos = self.cosine(a, b)
        return cos if cos is not None else jaccard(a, b)


def infer_persona(prior_user_turns: List[str]) -> str:
    """Cheap persona seed from the respondent's earlier answers (no LLM)."""
    if not prior_user_turns:
        return "a thoughtful interview participant"
    joined = " ".join(prior_user_turns)[-1200:]
    return f"the same person who has been answering, whose earlier replies were: \"{joined}\""


def simulate_answer(engine, prefix: List[Turn], persona: str) -> str:
    """One LLM call: role-play the respondent's next answer to the last question."""
    from src.utils.llm.engines import invoke_engine
    convo = "\n".join(f"{t.role}: {t.text}" for t in prefix)
    prompt = (
        "You are simulating a research-interview RESPONDENT (not the interviewer). "
        f"You are {persona}. Read the conversation and write ONLY the respondent's "
        "next answer to the interviewer's most recent question. Be realistic and "
        "consistent with the earlier answers; 1-4 sentences; no preamble.\n\n"
        f"{convo}\n\nUser:"
    )
    resp = invoke_engine(engine, prompt)
    return (getattr(resp, "content", None) or str(resp)).strip()


def run(path: str, max_turns: int, use_llm: bool) -> None:
    turns = parse_transcript(path)
    if not turns:
        print("No turns parsed — is this a chat_history.log?", file=sys.stderr)
        return

    sim = Similarity()
    engine = None
    if use_llm:
        try:
            import os
            from src.utils.llm.engines import get_engine
            engine = get_engine(model_name=os.getenv("MODEL_NAME", "gpt-4.1-mini"))
        except Exception as e:
            print(f"[sim] could not init engine ({e}); falling back to --no-llm", file=sys.stderr)
            use_llm = False

    divergences = []
    scored = 0
    prior_user: List[str] = []
    print(f"\n{'turn':>4} | {'sim':>5} | {'divg':>5} | mode")
    print("-" * 48)

    for i, turn in enumerate(turns):
        if turn.role != "User":
            continue
        # Need at least one interviewer question before this answer.
        prefix = turns[:i]
        if not any(t.role == "Interviewer" for t in prefix):
            prior_user.append(turn.text)
            continue

        actual = turn.text
        if use_llm and engine is not None:
            predicted = simulate_answer(engine, prefix, infer_persona(prior_user))
            mode = "llm"
        else:
            # Free baseline: predict the respondent repeats their previous answer.
            predicted = prior_user[-1] if prior_user else ""
            mode = "baseline"

        s = sim.score(predicted, actual)
        divergences.append(1.0 - s)
        scored += 1
        print(f"{scored:>4} | {s:>5.2f} | {1.0 - s:>5.2f} | {mode}")
        if scored <= 3 or use_llm:
            print(f"       predicted: {predicted[:110]!r}")
            print(f"       actual:    {actual[:110]!r}")

        prior_user.append(turn.text)
        if scored >= max_turns:
            break

    if divergences:
        avg = sum(divergences) / len(divergences)
        print("-" * 48)
        print(f"scored {len(divergences)} turns | mean divergence = {avg:.3f} "
              f"(0 = simulation matched actual, 1 = unrelated)")


def main():
    ap = argparse.ArgumentParser(description="Simulate a conversation and measure divergence from the actual log.")
    ap.add_argument("transcript", help="path to a chat_history.log")
    ap.add_argument("--max-turns", type=int, default=5, help="cap simulated turns (cost control)")
    ap.add_argument("--no-llm", action="store_true", help="skip LLM; score the free repeat-previous baseline")
    args = ap.parse_args()
    run(args.transcript, args.max_turns, use_llm=not args.no_llm)


if __name__ == "__main__":
    main()
