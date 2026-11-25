# KAWAL Harness X — Milestone 1 Design

**Status:** Draft for approval (revision 2)
**Date:** 2026-11-25
**Scope:** X only via Agent-Reach, one initial category, one initial city, role-based multi-agent, case artifact as output. No WhatsApp, Threads, government ticket system, or citizen replies in this milestone.

---

## 1. Intent

KAWAL stops being a "complete complaint system" and becomes a **research harness** that can be exercised end to end.

The problem is not "the model is not smart enough." The problem is that there is no single, explicit path from raw input to an inspectable output. This milestone defines that path:

> One X post -> one structured case, processed by multiple agents that check each other, with recorded cost and decision trace.

Core principles:

- **Do not trust a single agent.** Important decisions require an independent check.
- **Role-based, not swarm.** Agents never call each other; a deterministic orchestrator controls the sequence.
- **Agent-agnostic.** Every role satisfies the same contract and can be filled by a local model, a cloud LLM, or plain rules.
- **Cost-aware.** Every agent call records model, cost, latency, and prompt version. Extra verification happens only when it is worth the cost.
- **Replay-first.** Tests do not depend on X, live login, or network availability.

---

## 2. Goals

1. Fetch one X post through Agent-Reach and save it as a fixture.
2. Convert that post into a schema-validated `ComplaintCase`.
3. Run at least two independent agents for "is this a complaint?" and "what category?".
4. Record every step, model, cost, latency, and decision reason.
5. Provide one deterministic replay command and one automated test.
6. Provide thin ports/adapters so sources and outputs can be replaced without changing the core.

## 3. Non-goals

- Threads, WhatsApp, Facebook, Instagram, or any other channel.
- Integration with a real government ticket system.
- Replies, mentions, DMs, or notifications to citizens.
- Handling all categories or all cities at once.
- Production dashboard, multi-tenancy, or user authentication.
- Large-scale quality evaluation; that is a later milestone.
- Autonomous swarms or agents that choose their own workflow.

---

## 4. Constraints

- **Agent-Reach is not installed** on the development machine. X requires a browser login, performed by a human.
- Agent-Reach's X channel is **experimental and read-only**. Threads search is not supported.
- X search results may change, be rate-limited, or fail. Fixtures are the source of truth for tests.
- There is no ground-truth label in this milestone. Quality evaluation starts after the fixture and output shape are stable.
- Only public posts are processed. No DMs or private data. PII handling is still required for logs and fixtures.
- The committed milestone fixture must be a real X post captured through Agent-Reach.
- Remote model cost must be visible. Local models count as zero cost, but latency and CPU usage are still recorded.

---

## 5. Approaches Considered

| Approach | Pros | Cons | Decision |
|---|---|---|---|
| **Replay-first + thin ports** | Deterministic, fast to green, research-friendly, no platform dependency | Not a full live demo | **Chosen** |
| Live-first (direct X) | Most realistic demo | Fragile login, rate limits, hard to reproduce | Rejected for milestone 1 |
| Large framework-first (plugin registry, many adapters) | Strong "harness" story | Abstraction before one working path | Rejected; only two ports are kept |
| Multi-agent voting/judge | May improve accuracy | Expensive, hard to audit | Rejected for milestone 1 |
| Multi-agent swarm | Flexible | Not accountable for public complaints | Rejected |
| **Role-based multi-agent + verifier** | Independent, measurable, evaluable | Verification cost must be controlled | **Chosen** |

---

## 6. Architecture

```text
+---------------------+
| SourcePort          |
| - ReplayXSource     |  fixture JSON
| - AgentReachXSource |  live X (spike)
+----------+----------+
           | SocialPost
           v
+---------------------+
| Normalizer          |  deterministic
+----------+----------+
           | HarnessRun (draft)
           v
+---------------------+
| Orchestrator        |  deterministic state machine
|  TriageAgent        |
|  ExtractionAgent    |
|  VerifierAgent      |
|  ArbiterAgent       |  optional; conflict only
|  CaseWriter         |  deterministic
+----------+----------+
           | HarnessRun + Trace + CostReport
           v
+---------------------+
| RunSink             |
| - FileRunSink       |  milestone 1
| - SimulatorTicketSink | next milestone
| - GovernmentTicketSink | later
+---------------------+
```

Important rules:

- Agents have no database, broker, or side-effect access. They receive an `AgentTask` and return an `AgentResult`.
- Only the orchestrator may call agents, choose the order, and stop the run.
- `CaseWriter` never calls a model. It assembles the final output from validated agent results.
- Adapters live at the edge. The core never imports Agent-Reach, HTTP clients, or model SDKs.

---

## 7. Canonical Models

These live in a new module, `contracts/harness.py`, to avoid disturbing the legacy contracts.

```python
class SocialPost(BaseModel):
    post_id: str
    platform: Literal["x"]
    url: str
    author_handle: str
    text: str
    media_urls: tuple[str, ...] = ()
    posted_at: datetime | None = None
    collected_at: datetime
    raw: dict[str, Any] = {}

class TriageOutput(BaseModel):
    is_complaint: bool
    category: Category
    risk: RiskLevel
    confidence: float
    reason_codes: tuple[str, ...]

class ExtractionOutput(BaseModel):
    location: str | None
    object: str | None
    time: str | None
    evidence_spans: tuple[str, ...]
    confidence: float

class VerificationOutput(BaseModel):
    is_complaint: bool
    category: Category
    evidence_supported: bool
    issues: tuple[str, ...]
    confidence: float

class ArbitrationOutput(BaseModel):
    is_complaint: bool
    category: Category
    rationale: str
    confidence: float

class AgentUsage(BaseModel):
    model: str
    provider: str
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    cost_source: Literal["provider", "estimated", "local"]
    latency_ms: float

class AgentResult(BaseModel):
    agent_id: str
    role: str
    status: Literal["OK", "INVALID", "UNAVAILABLE", "BUDGET_EXCEEDED"]
    output: dict[str, Any] | None
    usage: AgentUsage
    prompt_version: str
    error: str | None = None

class ComplaintCase(BaseModel):
    case_id: str
    post: SocialPost
    triage: TriageOutput
    extraction: ExtractionOutput
    verification: VerificationOutput
    arbitration: ArbitrationOutput | None = None
    trust_decision: Literal["AGREED", "NEEDS_REVIEW", "REJECTED"]
    status: Literal["ACCEPTED", "NEEDS_REVIEW", "REJECTED"]
    reason_codes: tuple[str, ...]
    created_at: datetime

class HarnessRun(BaseModel):
    run_id: str
    post: SocialPost | None = None
    status: Literal[
        "COMPLETED", "NEEDS_REVIEW", "REJECTED", "BUDGET_EXCEEDED", "FAILED"
    ]
    case: ComplaintCase | None = None
    trace: tuple["TraceEvent", ...] = ()
    cost: "CostReport"
```

A `ComplaintCase` only exists when triage, extraction, and verification all produced valid outputs. A run that stops earlier is represented by `HarnessRun` with `case=None`, plus trace and cost.

`case_id` is derived deterministically from `platform + post_id`, so replay does not create duplicate cases.

---

## 8. Ports and Adapters

### 8.1 SourcePort

```python
class SourcePort(Protocol):
    def fetch(self, query: str, limit: int) -> list[SocialPost]: ...
```

Implementations:

- `ReplayXSource(path)` reads one or more recorded JSON files. Used by tests and evaluation.
- `AgentReachXSource(query, limit)` calls the Agent-Reach CLI, parses its JSON, and normalizes the result to `SocialPost`. Used for the spike and live demo, not CI.

`AgentReachXSource` must never be called from unit tests. Tests use `ReplayXSource`.

### 8.2 Agent

```python
class Agent(Protocol):
    agent_id: str
    role: str
    def estimate_cost(self, task: AgentTask) -> float: ...
    def run(self, task: AgentTask) -> AgentResult: ...
```

`AgentTask` carries:

- `role`
- `source_text` and allowed `source_media`
- `context` (allowed categories, extraction data for the verifier, policy context)
- `deadline_ms`
- `budget_remaining_usd`

No agent receives full case state, credentials, or another agent's private reasoning. The orchestrator validates each output against the canonical schema after the call.

Implementations:

- `RuleTriageAgent`, `RuleExtractionAgent`, `RuleVerifierAgent`, and `RuleArbiterAgent` are deterministic rule-based implementations. They are the milestone 1 default and keep replay tests independent of a model provider.
- `PricedStubAgent` is a test double with explicit estimated and actual cost.
- LLM-backed implementations can satisfy the same contract later. The orchestrator does not know which implementation is active.

### 8.3 RunSink

```python
class RunSink(Protocol):
    def write(self, run: HarnessRun) -> RunReceipt: ...
```

Milestone 1 implementation:

- `FileRunSink` writes `run.json`, `case.json` when present, `trace.jsonl`, and `cost.json` to `runs/<run_id>/`.

Future implementations (not built now):

- `SimulatorTicketSink`
- `GovernmentTicketSink`

---

## 9. Agent Roles and Trust Rules

### TriageAgent

- Input: post text and media metadata.
- Output: `TriageOutput`.
- Task: is this a complaint, what category, what risk, confidence.
- Initial implementation: rules/regex or a local model. A cheap LLM is allowed.

### ExtractionAgent

- Input: post text.
- Output: `ExtractionOutput`.
- Task: location, object, time, evidence spans.
- Initial implementation: NER/regex; LLM only when ambiguous.

### VerifierAgent

- Input: source post and `ExtractionOutput`. **It must not see `TriageOutput`**, to avoid anchoring.
- Output: `VerificationOutput`.
- Task: independently decide `is_complaint` and `category`, and check whether extraction fields are supported by the text.
- Implementation must differ from TriageAgent: a different model/provider or a different implementation class. The same model with a reworded prompt is weak independence and is not sufficient for the primary decision.

### ArbiterAgent (optional)

- Called only when TriageAgent and VerifierAgent disagree on `is_complaint` or `category`, and budget remains.
- Output: `ArbitrationOutput`.
- The Arbiter is **explanatory only**. It never converts disagreement into `ACCEPTED`. Any disagreement becomes `NEEDS_REVIEW` for a human, regardless of what the Arbiter concludes.

### CaseWriter (deterministic)

- Never calls a model.
- Validates schemas, assembles `ComplaintCase`, computes the evidence hash, and writes trace and cost.
- A schema failure produces `NEEDS_REVIEW`; it is never silently repaired.

### Trust rules

- `is_complaint` and `category` must be independently produced by TriageAgent and VerifierAgent.
- `location`, `object`, and `time` may come from ExtractionAgent plus a deterministic validator that requires the span to exist in the source text.
- If extraction fields conflict, VerifierAgent records `issues`.
- An unresolved conflict becomes `NEEDS_REVIEW`. No silent guessing.
- If the Verifier is unavailable, the run cannot become `ACCEPTED`.
- `ACCEPTED` requires TriageAgent and VerifierAgent to agree. The Arbiter may explain a disagreement but cannot authorize `ACCEPTED`.

---

## 10. Orchestrator and States

Deterministic state machine:

```text
RECEIVED
  -> NORMALIZED
  -> TRIAGED
  -> EXTRACTED
  -> VERIFIED
  -> ACCEPTED | NEEDS_REVIEW | REJECTED
  -> WRITTEN
```

Run-level terminal states:

- `COMPLETED` — case written.
- `NEEDS_REVIEW` — case written with unresolved conflict or missing evidence.
- `REJECTED` — clear non-complaint.
- `BUDGET_EXCEEDED` — run stopped before completion; trace and cost still written.
- `FAILED` — unexpected operational error; trace written.

Rules:

- Budget is checked before every agent call.
- An agent failure or invalid output never reaches `CaseWriter` as if it were valid.
- `NEEDS_REVIEW` still produces a run artifact; it is not an error.
- `REJECTED` is only for clear non-complaints, never for operational failures.
- Every transition writes a `TraceEvent`.

---

## 11. Budget and Trace

### Budget contract

- Budget is **per run**, configured by `--max-cost-usd` or a config file. Default: `0.05 USD`.
- Before each call, the orchestrator estimates worst-case cost from the agent's `max_input_tokens`, `max_output_tokens`, and a local price table.
- If `estimated_cost > remaining_budget`, the run stops with `BUDGET_EXCEEDED`; no call is made.
- After the call, actual usage is recorded. If the provider does not report cost, the estimate is used and marked `cost_source="estimated"`.
- Local and rule-based agents use `cost_usd=0`, `cost_source="local"`, but still record latency.
- The budget-path test uses a `PricedStubAgent` with a known estimate; it never depends on a real API.

### CostReport

`CostReport` includes:

- total cost per run,
- cost per role,
- latency per role,
- number of calls,
- number of arbiter calls,
- budget status.

### Trace

Trace is JSON Lines with at least:

```json
{
  "run_id": "...",
  "case_id": "...",
  "event": "agent.completed",
  "role": "triage",
  "agent_id": "...",
  "model": "...",
  "prompt_version": "...",
  "input_hash": "...",
  "output_hash": "...",
  "cost_usd": 0.0,
  "latency_ms": 0,
  "at": "..."
}
```

Raw citizen text is not written to application logs. Full text exists only in fixtures and local run files.

---

## 12. Data Scope for Milestone 1

- Platform: **X only**.
- Initial category: **road damage**.
- Initial city/jurisdiction: **Bandung** (one test jurisdiction).
- Language: Indonesian.
- Media: text; image URLs may be stored but are not analyzed in this milestone.
- Location: extracted from text; geocoding is not required.
- The committed fixture is a real X post captured through Agent-Reach.

Other categories and cities remain in the design but are not tested in this milestone.

---

## 13. End-to-End Flows

### Main replay flow

1. `ReplayXSource` reads `fixtures/x_posts/x_road_bandung_001.json`.
2. Normalizer creates a `SocialPost` and deterministic `case_id`.
3. TriageAgent returns `TriageOutput`.
4. ExtractionAgent returns `ExtractionOutput`.
5. VerifierAgent returns `VerificationOutput`.
6. If they agree, CaseWriter writes a `ComplaintCase` with status `ACCEPTED`.
7. `FileRunSink` writes `run.json`, `case.json`, `trace.jsonl`, and `cost.json`.
8. The test asserts category, status, location, trace count, and total cost.

### Conflict flow

1. The same fixture runs with a fake VerifierAgent that deliberately disagrees.
2. The orchestrator calls ArbiterAgent if budget allows, only to record an explanation.
3. The run becomes `NEEDS_REVIEW` and the case is still written.
4. The test asserts that `ACCEPTED` is impossible while agents disagree, even when the Arbiter agrees with one side.

### Budget flow

1. Budget is set very low.
2. The orchestrator stops before a call that would exceed it.
3. Run status is `BUDGET_EXCEEDED`; no case is written; trace and cost are written.

### Live smoke test (manual, not CI)

1. Install Agent-Reach.
2. Log in to X.
3. Run `agent-reach twitter search "jalan rusak bandung" --limit 5`.
4. Save one result as a fixture.
5. Run the harness with `AgentReachXSource`.
6. Record the result and cost.

If live search fails, the milestone is blocked until a real X post is captured. The failure is recorded as an external dependency, not converted into a synthetic fixture.

---

## 14. Acceptance Criteria

1. At least one real X post is captured through Agent-Reach, saved as a fixture, and can be read by `ReplayXSource`.
2. One command runs the full flow to `run.json`, `case.json`, `trace.jsonl`, and `cost.json`.
3. `ComplaintCase` validates against the schema and has a deterministic `case_id`.
4. TriageAgent and VerifierAgent are independent; any disagreement yields `NEEDS_REVIEW`, never automatic `ACCEPTED`, even if the Arbiter explains the conflict.
5. Every agent call records model, cost, latency, and prompt version.
6. Replaying twice does not create duplicate cases.
7. Automated tests cover the main path, conflict path, and budget path.
8. No calls to WhatsApp, Threads, government systems, or citizen-reply services.
9. `AgentReachXSource` performs one live X search and saves the result as the committed fixture. This is required for milestone completion.

---

## 15. Resolved Decisions

### 15.1 Arbiter authority

**Decision: Option B (conservative).** When TriageAgent and VerifierAgent disagree, the Arbiter may explain the conflict but never authorizes `ACCEPTED`. The run becomes `NEEDS_REVIEW` for a human.

### 15.2 First fixture provenance

**Decision: Option B (real capture).** The committed fixture must be one real post captured live from X through Agent-Reach. Human login is a hard dependency. A synthetic fixture may be used only in unit tests, never as the milestone fixture.

---

## 16. Builder Decisions

- Model/provider for Triage, Extraction, Verifier, and Arbiter.
- Prompt text and confidence thresholds.
- Fixture format and storage location.
- Whether a dashboard is used in this milestone or JSON files are enough.
- Retry/timeout details per adapter.
- The exact X post and the first category/city.
- PII redaction details for fixtures and reports.

## 17. Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Agent-Reach X fails or login expires | Fixture-first; live only as a smoke test |
| Verification cost grows | Per-run budget; Arbiter only on conflict |
| Agents anchor on each other | Verifier does not see TriageOutput; different implementation/model |
| Inconsistent output | Schema validation + `NEEDS_REVIEW`, no silent fallback |
| Public data contains PII | Redact in trace/log; keep fixtures local |
| Too many roles at the start | Start with three required roles + one optional; measure first |

---

## 18. Definition of Done

One command, one fixture, one case, one trace, one cost report, and one test that fails if any step breaks.
