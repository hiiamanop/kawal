# KAWAL Harness Monitor Console — Design

Date: 2026-11-26

## Intent

Turn the KAWAL X harness from a research CLI into an operator-facing monitoring
console. A government operator should be able to scan public X complaints, see
which ones are getting attention now, inspect the evidence and trust decision,
monitor the case, and prepare a forwarding packet for the relevant municipal
department.

This is a prototype with a human in the loop. The console does not send official
tickets, reply to citizens, or contact agencies automatically.

## Goals

- Scan live X search results and enrich each post with the existing harness
  pipeline.
- Show a ranked feed of complaints based on engagement, recency, risk, and trust.
- Show the full audit trail for each case: Triage, Extraction, Verifier,
  Arbiter, evidence, source intent, location quality, and cache provenance.
- Recommend a municipal department from the existing category mapping.
- Track operator status: new, monitoring, ready to forward, forwarded, dismissed.
- Generate a forwarding packet as Markdown and JSON for human review.
- Provide a replay mode for offline demos and tests.
- Replace the legacy WhatsApp dashboard with a ChatGPT-style local UI.
- Keep core tests offline and deterministic.

## Non-goals

- No automatic dispatch to government systems, email, WhatsApp, or police.
- No citizen replies or public responses.
- No multi-user authentication or role management in phase 1.
- No public internet deployment; default bind is `127.0.0.1`.
- No full social-media firehose; X search is a sampled view, not a complete feed.
- No geocoding or map view in phase 1.
- No semantic clustering of duplicate complaints in phase 1.
- No change to the legacy WhatsApp pipeline internals; only its dashboard is
  replaced.

## Assumptions

- The primary user is a trusted local operator running the prototype.
- Live data comes from Agent Reach X search with a valid X session.
- Replay mode uses the committed `fixtures/x_posts` directory.
- Rule-based agents are the default analysis path; LLM analysis is optional and
  budgeted.
- Department names come from `services.core.policy.resolve_department_name`.
- UI copy is Indonesian; code, tests, and documentation are English.

## Approaches Considered

1. **Feed-first console with a ChatGPT-style shell** (selected)
   - Left sidebar, ranked feed, detail panel, status actions, bottom query box.
   - Best fit for monitoring many cases while keeping a familiar ChatGPT look.
   - Uses FastAPI and local HTML/CSS/JS, matching the existing server stack.
2. **Chat-first assistant**
   - Operator asks a question and receives complaint cards in a conversation.
   - Closer to ChatGPT literally, but weak for queues, filters, and status work.
3. **Single-page React/Svelte app**
   - More polished interactions, but adds a build toolchain and dependencies
     that the repository does not currently use.

The selected approach keeps the existing FastAPI/uvicorn stack and avoids a
frontend build step.

## Architecture

### Runtime

- FastAPI application served by uvicorn.
- `scripts/run_monitor.py` replaces `scripts/run_dashboard.py`.
- Default bind: `127.0.0.1:8088`.
- No PostgreSQL requirement for the monitor console.

### New modules

```text
services/monitor/
├── __init__.py
├── app.py            FastAPI app and HTTP routes
├── config.py         default queries and scoring constants
├── models.py         MonitorCase, ScanRequest, ForwardPacket
├── scoring.py        viral score and ranking
├── routing.py        department recommendation
├── scan.py           live/replay scan orchestration
├── store.py          SQLite persistence and audit log
└── forwarding.py     Markdown/JSON forwarding packet
```

The HTML, CSS, and JavaScript live under `services/monitor/static/` and
`services/monitor/templates/`. No CDN is required; the UI must work offline.

### Data flow

```text
operator query
  -> AgentReachXSource (live) or ReplayXSource (demo)
  -> AgentReach returns SocialPost items with raw engagement data
  -> HarnessOrchestrator with rule agents by default
  -> MonitorCase: post + case + engagement + viral score + department
  -> SQLite store
  -> feed and detail UI
```

### Persistence

SQLite is used through the Python standard library. Default database path:

```text
runs/monitor/monitor.db
```

Override with `KAWAL_MONITOR_DB`.

Tables:

- `monitor_cases`: post and case JSON, full `HarnessRun` JSON including trace,
  cost, and cache provenance, extracted filter columns, engagement metrics,
  viral score, recommended department, operator status, analysis status,
  notes, first_seen_at, last_seen_at, updated_at.
- `monitor_case_queries`: case-to-query association, so a case can be linked to
  every query that returned it.
- `monitor_scans`: scan id, query, source mode, analysis mode, counts, errors,
  started_at, finished_at.
- `monitor_actions`: append-only audit log for status changes and forwarding.

Only one scan may run at a time. A second request receives HTTP 409.

### Scan behavior

- A scan processes one query with a configurable limit, default 10.
- Live scans call `AgentReachXSource().fetch(query, limit)`.
- Replay scans read a fixture directory and filter by the query only for display;
  replay is intended for demos and tests.
- Existing `post_id` values are updated rather than duplicated. Engagement and
  `last_seen_at` are refreshed; human status, notes, and the original analysis
  are preserved.
- New posts are analyzed with rule agents by default. LLM mode is opt-in and
  uses the existing budget guard.
- The scan returns counts, errors, and the number of new versus updated cases.
- A failed Agent Reach call returns a clear error state; it does not create
  fake cases.
- If analysis returns `FAILED` or `BUDGET_EXCEEDED`, the post is stored with
  `analysis_status` set accordingly, nullable score and department, and is
  hidden from the default complaint feed. It remains visible under `Errors`.
- Phase 1 does not use the semantic cache in the monitor scan. The full
  `HarnessRun` is still stored; the detail panel shows cache provenance only
  when a run actually contains it.

Default queries:

- `jalan rusak bandung`
- `trotoar rusak bandung`
- `sampah menumpuk bandung`
- `banjir bandung`
- `lampu jalan mati bandung`
- `pungli bandung`

The operator can type a custom query.

### Viral score

The score is a transparent heuristic in the range 0-100:

```text
engagement_raw  = likes + 0.05 * views
engagement_norm = min(1, log1p(engagement_raw) / log1p(10000))
recency_norm    = max(0, 1 - hours_since_post / 72)
risk_norm       = URGENT 1.0, HIGH 0.75, MEDIUM 0.5, LOW 0.25
trust_norm      = ACCEPTED 1.0, NEEDS_REVIEW 0.6, REJECTED 0.0

viral_score = round(100 * (
    0.45 * engagement_norm +
    0.25 * recency_norm +
    0.20 * risk_norm +
    0.10 * trust_norm
))
```

Missing engagement values count as zero. The weights live in `config.py` so they
can be tuned without changing the scoring code.

### Rule category coverage

The rule path is extended from three categories to all 12 municipal complaint
categories before the monitor UI is built:

```text
ROAD, DRAINAGE_FLOOD, WASTE, CLEAN_WATER, CIVIL_ADMIN, HEALTH_SERVICE,
PUBLIC_ORDER, TRANSPORTATION, FIRE_RESCUE, SOCIAL_AFFAIRS, EDUCATION,
PARKS_HOUSING
```

- `RuleTriageAgent` gets a category cue lexicon for every category.
- `RuleExtractionAgent` gets object and location cues for every category.
- `RuleVerifierAgent` checks category consistency for every category.
- Existing risk cues remain shared; category-specific escalation cues may be
  added where needed.
- Ambiguous multi-category text is not forced into a confident category. The
  Verifier should surface the conflict so the case becomes `NEEDS_REVIEW`.
- Out-of-taxonomy complaints must not be silently assigned to `ROAD`; they
  become `NEEDS_REVIEW` with an explicit reason code.
- The rule-vs-LLM evaluation is rerun after the expansion because existing rule
  metrics will change.

### Department routing

- Use `resolve_department_name(case.triage.category)`.
- Display the department as a recommendation, never as an official assignment.
- Phase 1 does not route by kelurahan or authority unit; the operator must verify
  the jurisdiction.
- Emergency or criminal escalation remains a human decision.

### Status workflow

Allowed transitions:

```text
NEW              -> MONITORING, READY_TO_FORWARD, DISMISSED
MONITORING       -> READY_TO_FORWARD, DISMISSED
READY_TO_FORWARD -> FORWARDED, MONITORING, DISMISSED
FORWARDED        -> terminal
DISMISSED        -> terminal
```

- `POST /forward` generates the packet and moves a `NEW` or `MONITORING` case
  to `READY_TO_FORWARD`. It never sends anything.
- `PATCH status=FORWARDED` is allowed only from `READY_TO_FORWARD`, after the
  operator confirms they sent the packet through their own channel.
- `FORWARDED` and `DISMISSED` cannot be reopened in phase 1.
- Feed-card actions match the state: `NEW`/`MONITORING` show `Monitor` and
  `Detail`; `READY_TO_FORWARD` shows `Forward` and `Mark forwarded`;
  `FORWARDED`/`DISMISSED` are read-only archive views.

Every transition writes a `monitor_actions` row with old status, new status,
note, actor (`local-operator`), and timestamp.

### Forwarding packet

`POST /api/monitor/cases/{case_id}/forward` creates:

```text
runs/monitor/forwards/<case_id>-<timestamp>.md
runs/monitor/forwards/<case_id>-<timestamp>.json
```

The packet contains:

- original post text, author, URL, timestamp, and engagement;
- category, risk, location, evidence spans, source intent, trust decision;
- recommended department;
- operator note;
- a clear `DRAFT — NOT SENT` warning.

The UI can download the packet. Marking the case `FORWARDED` is a separate
operator action after they send it through their own channel.

## API

```text
GET    /                              HTML console
GET    /api/monitor/cases             feed, filters: status, trust, category, query
GET    /api/monitor/cases/{case_id}   full case and audit trail
PATCH  /api/monitor/cases/{case_id}   status and operator note
POST   /api/monitor/scan              one live or replay scan
POST   /api/monitor/cases/{case_id}/forward
GET    /api/monitor/stats             counts and last scan
GET    /api/monitor/health            app, database, and Agent Reach status
```

Request and response bodies use Pydantic models in `services/monitor/models.py`.
Errors use stable codes and human-readable messages.

Filter contract:

- `status` is the operator workflow status (`NEW`, `MONITORING`,
  `READY_TO_FORWARD`, `FORWARDED`, `DISMISSED`).
- `trust` is the harness trust status (`ACCEPTED`, `NEEDS_REVIEW`,
  `REJECTED`).
- `query` matches every case linked through `monitor_case_queries`, not only the
  last scan that returned it.
- The default feed is complaints with `trust` in (`ACCEPTED`, `NEEDS_REVIEW`)
  and `status` not `DISMISSED`, ordered by viral score.
- `analysis_status` (`OK`, `FAILED`, `BUDGET_EXCEEDED`) is a separate filter
  used by the `Errors` view.

## UI

### Visual direction

ChatGPT-like, not a copy of ChatGPT branding:

- neutral light theme, system fonts, generous whitespace;
- left sidebar with a soft gray background;
- main content max width around 900px;
- rounded cards with subtle borders, no gradients or neon;
- clear typographic hierarchy and compact badges;
- responsive layout that collapses the sidebar on narrow screens.

### Layout

```text
┌────────────────────┬──────────────────────────────────────────────┐
│ KAWAL Monitor      │  Ringkasan hari ini                          │
│                    │  [total] [complaint] [review] [siap kirim]   │
│ [+ Scan sekarang]  │                                              │
│                    │  Keluhan viral                               │
│ Live Feed          │  ┌────────────────────────────────────────┐  │
│ Monitoring         │  │ score  category  risk  department      │  │
│ Siap Diteruskan    │  │ author · time · engagement             │  │
│ Forwarded          │  │ excerpt ...                            │  │
│ Archive            │  │ [Monitor] [Detail] [Siap Diteruskan]   │  │
│                    │  └────────────────────────────────────────┘  │
│ Filter             │                                              │
│ Kategori           │                                              │
│ Risiko             │                                              │
│ Status             │  ┌────────────────────────────────────────┐  │
│                    │  │ Tanya atau scan keluhan...       [Scan]│  │
└────────────────────┴──────────────────────────────────────────────┘
```

The bottom composer accepts a custom query and starts a scan. It is not a
general chatbot in phase 1.

### Feed card

- viral score and rank;
- category badge, risk badge, trust status;
- recommended department;
- author, handle, posted time, likes/views;
- text excerpt and location if available;
- primary actions: `Detail`, `Monitor`, `Siap Diteruskan`.

Default feed shows complaints only, ordered by viral score. A filter can show
needs-review, rejected, or all.

### Detail panel

- original post and link;
- media thumbnails;
- engagement and viral score breakdown;
- Triage, Extraction, Verifier, and Arbiter outputs;
- evidence spans and reason codes;
- source intent and location quality;
- cache provenance;
- trace timeline;
- status actions and operator note;
- forwarding packet action.

### States

- initial loading;
- empty feed with a `Scan sekarang` prompt;
- scan in progress;
- Agent Reach disconnected or X session expired;
- rate-limited;
- no new posts;
- database error;
- case not found.

## Security and privacy

- Bind to `127.0.0.1` by default. The app is not safe for public deployment
  without authentication and TLS.
- Do not expose environment variables or API keys in the UI.
- Keep the existing PII fail-closed guard before any LLM call.
- Show only public X content.
- No automatic forwarding, ticketing, or citizen contact.
- Forwarding packets are drafts and are stored locally.
- The SQLite database and forwarding packets are local sensitive data; document
  that they should not be committed.

## Testing

- Unit tests for viral scoring, ranking, routing, status transitions, and
  forwarding packet content.
- API tests with temporary SQLite and replay fixtures; no live X calls.
- HTML smoke tests for the console shell and key sections.
- Replay scan test with a small fixture directory.
- Error tests for Agent Reach failure, duplicate scan, and invalid status
  transition.
- Remove or replace legacy dashboard tests; keep the legacy WhatsApp pipeline
  code untouched.
- Add one positive and one ambiguous case per category to the rule-path tests.

## Migration and cleanup

- Replace `services/dashboard/app.py` with the monitor console.
- Remove the legacy WhatsApp dashboard routes, including `/track` and
  `/api/tickets`, and their tests.
- Replace `scripts/run_dashboard.py` with `scripts/run_monitor.py`.
- Update `README.md`, `docs/testing.md`, and `docs/` references.
- Keep `runs/monitor/` and forwarding packets out of git.

## Selected decisions

### Legacy `/track` removal

The legacy `/track` and `/api/tickets` routes are removed. No PostgreSQL shim
and no compatibility page are kept. The legacy WhatsApp pipeline code remains
untouched, but links it emits to `http://localhost:8088/track/<ticket>` will
return 404 after the monitor console replaces the old dashboard. This is
accepted because that pipeline is experimental and not the current target.

### Rule path covers all 12 categories

The rule path is extended to all 12 categories before the UI is considered
complete. The console must not ship with a ROAD/WASTE/DRAINAGE-only feed while
its default queries include public order, transportation, or other categories.

## Limitations

- X search is a sample, not a complete national feed.
- Engagement counts can be stale and can be manipulated.
- Viral score is a heuristic, not a measure of truth or severity.
- Department routing is category-based and needs human jurisdiction checks.
- Rule cue coverage is finite; out-of-taxonomy complaints must go to review, not
  be forced into a category.
- Live mode depends on Agent Reach and a valid X session.
- Phase 1 has no authentication, clustering, geocoding, or automatic dispatch.
- The console must not be presented as an official government system.
