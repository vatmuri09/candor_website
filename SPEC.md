# SparkMe Revamp — Spec (2026-07-20)

Working spec for the current sprint: fix conversation quality, make the admin
panel show *whether* it's actually probing, and lay groundwork for
current-events interview content. Scope is "next few hours," so this is
ordered by leverage, not completeness.

## 1. What the logs actually show

Read `logs/05aa3fad-9c16-4e0b-b1a7-49815cdfb563/execution_logs/session_1/chat_history.log`
(a live session from 2026-06-29, AI-in-workforce topic) turn by turn. Concrete
defects, not vibes:

- **Near-duplicate questions.** Six consecutive interviewer turns re-asked the
  same question about "autonomy impacting workflow/client interactions" with
  cosmetic rewording:
  > "...how does working independently influence your role and workflow..."
  > "...how do you think this autonomy impacts the overall quality of service..."
  > "...could you share more about how this autonomy specifically impacts your
  > workflow and client interactions..."

  The user gave essentially the same answer four times because the question
  never moved. This is the single biggest quality failure in the transcript.
- **No depth-probing.** Every follow-up stays at the same level of abstraction
  as the prior answer — it restates what the user said and asks a slightly
  reworded version of "tell me more," never "why," never a concrete example,
  never a number, never a contrasting case. A real interviewer would have
  drilled into "extensive manual correction" (what does that look like
  concretely? how much time does it add?) instead of pivoting away.
- **Duplicate opening turns.** The log shows two full greeting+intro messages
  back to back at the very start of the session (12:38 and 12:43), suggesting
  a double-session-start or reload bug independent of the LLM's own repetition.

**What's already been done about it:** commit `f835a90` (2026-07-17, after
this log) added a non-LLM near-duplicate guardrail to
[interviewer.py](src/agents/interviewer/interviewer.py) — Jaccard similarity
(unigram + bigram) over the last 6 questions, threshold 0.55, with a
regeneration attempt on trip. This is real progress but:
1. It's unvalidated — no session log exists from after the fix landed.
2. It's a symptom patch (blocks repeats after the fact) not a cause fix (why
   does the interviewer *want* to ask the same thing again — is the agenda
   manager not advancing subtopic coverage, or is the model just not planning
   ahead?).
3. It does nothing for the "no depth" problem, which is a content/prompting
   issue, not a repetition issue — a guardrail that only catches literal
   near-duplicates will pass through six *different* shallow questions just as
   easily as one repeated one.

**Also found, orthogonal but real:** `logs/flask_app.log` shows a
`ModuleNotFoundError: No module named 'psycopg2'` inside `save_web_session` on
a local run. `requirements.txt` does list `psycopg2-binary==2.9.10`, so this
was a local venv drift, not a real dependency gap — but worth a canary check
against the actual Vercel deployment (Postgres storage per current setup)
before trusting session persistence there.

## 2. Priorities for this session (ordered)

1. **Validate the near-dup guardrail against a fresh run** — run one live
   session (terminal or web), confirm `guardrail_stats["near_duplicate"]`
   fires and regenerations actually diverge, not just reword.
2. **Fix root cause of repetition**: check whether `AgendaManager` /
   `ExplorationPlanner` are correctly marking subtopics covered after a
   sufficient answer, or whether the interviewer keeps re-selecting the same
   subtopic. This is in [agenda_manager.py](src/agents/agenda_manager/agenda_manager.py)
   (`_update_subtopic_coverage`) and [exploration_planner.py](src/agents/exploration_planner/exploration_planner.py).
3. **Add a depth/probe-quality signal** — new lightweight check (rule-based
   first, LLM-judge second) that flags interviewer turns which just restate
   the prior answer without adding a "why / concrete example / number /
   contrast" ask. Surface this the same way `guardrail_stats` already is.
4. **Admin: surface quality signals, not just transcripts.** `admin_session.html`
   currently renders only the raw transcript — none of `guardrail_stats`,
   `EngagementMonitor` signals (`QualitySignal`, `BreakdownVerdict`), or the
   new depth-probe flag are shown anywhere in the admin UI today. This is the
   fastest way to answer "is it actually probing" at a glance across sessions.
5. **Question/topic authoring in admin** — deferred per your note; revisit
   once (3) and (4) are solid.
6. **Current-events content** — build on the existing
   [context_research.py](src/agents/context/context_research.py) web-search
   agent (already wired to an approval popup per commit `d0e4bcc`) rather than
   a new agent. Extend it to produce dated, falsifiable current-events probes
   (e.g. "did you watch any World Cup matches this week," anchored to an
   actual fixture list) instead of generic topic summaries.

## 3. New/changed agents ("subagents")

| Agent | Status | Change needed |
|---|---|---|
| `Interviewer` | guardrail added, unvalidated | validate; add depth-probe check alongside near-dup check |
| `AgendaManager` | existing | audit subtopic-coverage update logic for the repetition root cause |
| `ExplorationPlanner` | existing | confirm it's actually being consulted every 3-5 turns and not silently no-op'ing |
| `EngagementMonitor` | existing, signals unused downstream | wire `QualitySignal`/`BreakdownVerdict` into admin view |
| `ContextResearchAgent` | existing | extend for dated current-events queries, not just static topic background |
| New: **ProbeQualityMonitor** (proposed) | doesn't exist | rule-based first pass (turn restates prior answer / no new information request), same shape as `EngagementMonitor` so it slots into the same guardrail_stats plumbing |

## 4. Admin panel spec (minimum for "can I tell if it's working")

On `admin_session.html`, per session, add:
- Guardrail stat bar: counts for `near_duplicate`, `near_duplicate_regen_failed`,
  `no_question`, `advice`, `stance`, `affirmation`, and the new depth-probe flag.
- Inline transcript annotation: badge each interviewer turn that tripped a
  guardrail, with the guardrail name, directly next to that turn (not just an
  aggregate count) so quality issues are visible in context.
- Engagement trend: plot or list `EngagementMonitor` signal per turn
  (good/flagged/declining/disengaged) so a reviewer can see where the
  respondent checked out.

On `admin.html` (session list), add a column summarizing worst guardrail
severity per session so bad sessions are triageable without opening each one.

## 5. Current-events content (deferred detail, scoping only)

- Reuse `ContextResearchAgent.research_topic()` — it already does live search
  + source extraction + human approval before use.
- New use: admin enters a topic like "2026 World Cup" or a specific news
  event; agent researches it, extracts 3-5 dated, checkable facts (fixture
  results, dates, named events), and those become anchor points the
  interviewer can reference concretely ("did you catch the match on
  [date]") instead of open-ended "what's on your mind about the news."
- Explicitly out of scope for this pass: swapping the default topic bank —
  `topics.json` stays AI-in-workforce until you decide otherwise.

## 6. Out of scope for "next few hours"

- Replacing/merging the topic bank (deferred per your answer).
- Full admin question-authoring UI (deferred per your answer).
- Vercel/Postgres deploy debugging beyond a sanity check — no evidence yet
  that it's broken there specifically, only locally.
