# KAWAL Harness X Milestone 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a deterministic X-only harness that turns one public post into a schema-validated complaint case using role-based multi-agent verification, recorded cost, and a replayable trace.

**Architecture:** Thin ports at the edges (`SourcePort`, `Agent`, `RunSink`) surround a deterministic orchestrator. Rule-based agents provide a runnable baseline; the same `Agent` contract can later host LLM-backed implementations. The default path is fixture-driven and deterministic; a live Agent-Reach source is an optional smoke-test adapter.

**Tech Stack:** Python 3.11+, Pydantic v2, pytest, standard library subprocess/urllib. No new runtime dependency.

**Spec:** `docs/superpowers/specs/2026-11-25-kawal-harness-x-design.md`

## Global Constraints

- Code and documentation are written in English.
- X only. No WhatsApp, Threads, Facebook, Instagram, government ticket system, or citizen replies.
- One initial category (`ROAD`) and one initial city (`Bandung`).
- `ACCEPTED` requires TriageAgent and VerifierAgent to agree. The Arbiter never authorizes `ACCEPTED`; disagreement becomes `NEEDS_REVIEW`.
- Every agent call records model, provider, cost, cost source, latency, and prompt version.
- Budget is per run. The budget check happens before a call, using a worst-case estimate.
- No raw citizen text in application logs. Full text lives only in fixtures and local run files.
- The committed milestone fixture must be a real X post captured through Agent-Reach. Synthetic fixtures are test-only.
- Do not create a pull request. Work stays on `feat/kawal-harness-x-m1` until review.

## Review Focus

- **Empty or whitespace-only post:** the run must end `FAILED` with a trace, not crash.
- **Non-complaint post:** Triage and Verifier both reject; the case ends `REJECTED`, not `ACCEPTED`.
- **Missing location:** agents agree it is a complaint but extraction has no location; the case must be `NEEDS_REVIEW`.
- **Verifier disagrees on category:** even if the Arbiter explains, the case must stay `NEEDS_REVIEW`.
- **Budget exhausted before Triage:** no `ComplaintCase` exists; `HarnessRun.case` is `None` and trace/cost are still written.
- **Replay twice:** the same post produces the same deterministic `case_id` and no duplicate logical case.

---

### Task 1: Canonical Contracts and Ports

**Files:**
- Create: `contracts/harness.py`
- Create: `services/harness/__init__.py`
- Create: `services/harness/ports.py`
- Test: `tests/test_harness_contracts.py`

**Interfaces:**
- Produces: `SocialPost`, `TriageOutput`, `ExtractionOutput`, `VerificationOutput`, `ArbitrationOutput`, `AgentUsage`, `AgentResult`, `ComplaintCase`, `HarnessRun`, `TraceEvent`, `CostReport`, `RunReceipt`
- Produces: `SourcePort`, `Agent`, `RunSink`, `AgentTask`
- Produces: `deterministic_case_id(post: SocialPost) -> str`

- [ ] **Step 1: Write failing contract tests**

Test names and assertions:
- `test_case_id_is_deterministic` — two calls with the same post return the same string; a different `post_id` returns a different string.
- `test_complaint_case_requires_verification` — constructing `ComplaintCase` without `verification` raises `ValidationError`.
- `test_harness_run_allows_missing_case` — `HarnessRun(status="BUDGET_EXCEEDED", case=None, cost=CostReport(...))` validates.
- `test_agent_result_rejects_unknown_status` — an invalid status raises `ValidationError`.

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_harness_contracts.py -v`
Expected: FAIL with import errors for `contracts.harness`.

- [ ] **Step 3: Implement contracts and ports**

In `contracts/harness.py`, define the Pydantic models from the spec. Use `ConfigDict(extra="forbid", frozen=True)` where practical. `AgentResult.output` is `dict[str, Any] | None`. `HarnessRun.cost` is required.

In `services/harness/ports.py`, define:

```python
class AgentTask(BaseModel):
    role: str
    source_text: str
    source_media: tuple[str, ...] = ()
    context: dict[str, Any] = {}
    deadline_ms: int = 5000
    budget_remaining_usd: float = 0.0

class SourcePort(Protocol):
    def fetch(self, query: str, limit: int) -> list[SocialPost]: ...

class Agent(Protocol):
    agent_id: str
    role: str
    def estimate_cost(self, task: AgentTask) -> float: ...
    def run(self, task: AgentTask) -> AgentResult: ...

class RunSink(Protocol):
    def write(self, run: HarnessRun) -> RunReceipt: ...
```

`deterministic_case_id` uses `sha256(f"{post.platform}:{post.post_id}".encode())` and returns `f"case-x-{digest[:16]}"`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_harness_contracts.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add contracts/harness.py services/harness/__init__.py services/harness/ports.py tests/test_harness_contracts.py
git commit -m "feat: add harness contracts and ports"
```

---

### Task 2: Source Adapters

**Files:**
- Create: `services/harness/sources.py`
- Create: `tests/fixtures/x_posts/synthetic_road_bandung_001.json`
- Test: `tests/test_harness_sources.py`

**Interfaces:**
- Consumes: `SocialPost`, `SourcePort`
- Produces: `ReplayXSource(path: str | Path)`
- Produces: `AgentReachXSource(binary: str | None = None, runner: Callable[..., subprocess.CompletedProcess[str]] | None = None, timeout_seconds: float = 60.0)`

- [ ] **Step 1: Write failing source tests**

Test names and assertions:
- `test_replay_source_reads_fixture` — returns one `SocialPost` with `platform="x"`, the fixture `post_id`, non-empty `text`, and `collected_at`.
- `test_agent_reach_source_parses_items` — a fake runner returns `{"items": [{"id": "1", "text": "...", "url": "...", "author": "..."}]}`; the adapter maps it to one `SocialPost`.
- `test_agent_reach_source_reports_need_setup` — a runner raising `FileNotFoundError` raises `AgentReachSourceError` with code `NEED_SETUP`.

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_harness_sources.py -v`
Expected: FAIL with import errors for `services.harness.sources`.

- [ ] **Step 3: Implement adapters and fixture**

`ReplayXSource.fetch` ignores `query`/`limit`, reads the JSON file, and returns a one-item list. It accepts a raw Agent-Reach item or a normalized harness fixture with a `post` key.

`AgentReachXSource` calls `[binary, "twitter", "search", query, "--limit", str(limit)]`, parses JSON, accepts an `items` list or a single item, and maps `id`, `text`, `url`, `author`/`username`, `created_at`/`timestamp`, and media URLs. Define `AgentReachSourceError(RuntimeError)` with `code`.

The synthetic fixture is for tests only:

```json
{
  "post": {
    "post_id": "synthetic-road-bandung-001",
    "platform": "x",
    "url": "https://x.com/example/status/1",
    "author_handle": "warga_bandung",
    "text": "Min, jalan di Jl. Kopo Bandung rusak parah, ada lubang besar. Tolong segera diperbaiki.",
    "media_urls": [],
    "posted_at": "2026-11-20T08:00:00+00:00",
    "collected_at": "2026-11-25T00:00:00+00:00"
  }
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_harness_sources.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add services/harness/sources.py tests/fixtures/x_posts/synthetic_road_bandung_001.json tests/test_harness_sources.py
git commit -m "feat: add replay and Agent-Reach X sources"
```

---

### Task 3: Rule-Based Agents

**Files:**
- Create: `services/harness/agents.py`
- Test: `tests/test_harness_agents.py`

**Interfaces:**
- Consumes: `Agent`, `AgentTask`, `AgentResult`, `TriageOutput`, `ExtractionOutput`, `VerificationOutput`, `ArbitrationOutput`
- Produces: `RuleTriageAgent`, `RuleExtractionAgent`, `RuleVerifierAgent`, `RuleArbiterAgent`, `PricedStubAgent`

- [ ] **Step 1: Write failing agent tests**

Test names and assertions:
- `test_triage_detects_road_complaint` — road/lubang text → `is_complaint=True`, `category=ROAD`, `confidence >= 0.5`.
- `test_triage_rejects_greeting` — “selamat pagi” → `is_complaint=False`.
- `test_extraction_finds_street_and_object` — text with “Jl. Kopo” and “lubang” → `location` contains “Kopo”, `object` is not `None`.
- `test_verifier_agrees_with_clear_complaint` — same text → `is_complaint=True`, `category=ROAD`, `evidence_supported=True`.
- `test_verifier_flags_unsupported_location` — extraction location not present in source text → `evidence_supported=False`, `issues` non-empty.
- `test_arbiter_never_authorizes_acceptance` — arbiter output has no field that can change trust; orchestrator behavior is tested in Task 4.
- `test_priced_stub_reports_usage` — cost, latency, and model fields are populated.

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_harness_agents.py -v`
Expected: FAIL with import errors for `services.harness.agents`.

- [ ] **Step 3: Implement rule agents**

Heuristics:
- `RuleTriageAgent`: complaint cues (`lapor`, `aduan`, `tolong`, `min`, `rusak`, `lubang`, `banjir`, `sampah`); road cues (`jalan`, `jl`, `lubang`, `aspal`, `berlubang`); risk `HIGH` if hazard cues (`bahaya`, `parah`, `korban`, `kecelakaan`), else `MEDIUM`.
- `RuleExtractionAgent`: regex for street (`\b(?:Jl\.?|Jalan)\s+([A-Z][\w\-]+(?:\s+[A-Z][\w\-]+){0,3})`), location labels (`Kelurahan`, `Kecamatan`), objects (`lubang`, `aspal`, `jalan`, `trotoar`), time cues (`kemarin`, `pagi`, `sore`, `minggu`). `evidence_spans` are exact substrings from the source.
- `RuleVerifierAgent`: independent rule set; requires an explicit complaint cue + road damage cue; recomputes category from damage cues; checks every extraction evidence span is in `source_text`. It must not receive or read a `TriageOutput`.
- `RuleArbiterAgent`: returns a rationale describing the disagreement. It is explanatory only.
- `PricedStubAgent`: configurable fixed `estimate`, fixed `AgentUsage`, and return output; used by budget tests.

All rule agents return `AgentResult` with `cost_source="local"`, `cost_usd=0.0`, `latency_ms` measured with `time.perf_counter()`, and `prompt_version="rule-v1"`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_harness_agents.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add services/harness/agents.py tests/test_harness_agents.py
git commit -m "feat: add role-based rule agents"
```

---

### Task 4: Budget, Trace, and Orchestrator

**Files:**
- Create: `services/harness/budget.py`
- Create: `services/harness/trace.py`
- Create: `services/harness/orchestrator.py`
- Test: `tests/test_harness_orchestrator.py`

**Interfaces:**
- Consumes: the Task 1 contracts and Task 3 agents.
- Produces: `BudgetTracker(max_cost_usd: float)`, `TraceRecorder`, `HarnessOrchestrator(triage, extraction, verifier, arbiter=None, budget=None, max_verification_attempts=1)`
- Produces: `HarnessOrchestrator.run(post: SocialPost) -> HarnessRun`

- [ ] **Step 1: Write failing orchestrator tests**

Test names and assertions:
- `test_agreement_produces_accepted_case` — clear road fixture → `run.status == "COMPLETED"`, `run.case.status == "ACCEPTED"`, `trust_decision == "AGREED"`.
- `test_non_complaint_produces_rejected_case` — greeting → `run.status == "REJECTED"`, `case.status == "REJECTED"`.
- `test_verifier_disagreement_stays_needs_review` — fake verifier says `WASTE`; arbiter says `ROAD`; `case.status == "NEEDS_REVIEW"`, never `ACCEPTED`.
- `test_missing_location_stays_needs_review` — extraction returns `location=None`; status `NEEDS_REVIEW`.
- `test_budget_exhaustion_before_triage_has_no_case` — priced triage estimate exceeds budget; `run.status == "BUDGET_EXCEEDED"`, `case is None`, trace non-empty.
- `test_empty_post_fails_with_trace` — empty text → `FAILED`, `case is None`.
- `test_verifier_unavailable_has_no_case` — verifier returns `UNAVAILABLE`; `case is None`, status `NEEDS_REVIEW`.
- `test_arbiter_does_not_flip_disagreement` — arbiter agrees with triage; status remains `NEEDS_REVIEW`.

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_harness_orchestrator.py -v`
Expected: FAIL with import errors for `services.harness.orchestrator`.

- [ ] **Step 3: Implement budget, trace, and orchestrator**

`BudgetTracker`:
- `remaining_usd`, `total_spent_usd`, `calls`.
- `can_afford(estimate) -> bool`.
- `record(usage)` raises if it would exceed the limit.

`TraceRecorder`:
- `add(event: str, **fields) -> None`.
- `events -> tuple[TraceEvent, ...]`.
- `CostReport` totals from `AgentUsage` values; include `budget_status`.

`HarnessOrchestrator.run` sequence:
1. Validate non-empty text; else `FAILED`.
2. Build `SocialPost` and deterministic `case_id`.
3. Before each agent: `estimate_cost`; if not affordable → `BUDGET_EXCEEDED` with `case=None`, trace, cost.
4. Run Triage; invalid/unavailable → `FAILED` or `NEEDS_REVIEW` with `case=None`.
5. Run Extraction; invalid → `NEEDS_REVIEW` with `case=None`.
6. Run Verifier with `source_text` and `ExtractionOutput` only; if `UNAVAILABLE` → `NEEDS_REVIEW`, `case=None`.
7. Compare `is_complaint` and `category`:
   - both `False` and categories equal → case `REJECTED`.
   - both `True`, categories equal, location present, evidence spans supported → case `ACCEPTED`.
   - both `True`, categories equal, location missing → case `NEEDS_REVIEW`.
   - disagreement → call Arbiter if budget allows (explanation only); case `NEEDS_REVIEW`.
8. Build `ComplaintCase` only when triage, extraction, and verification outputs exist.
9. Compute `CostReport`; return `HarnessRun`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_harness_orchestrator.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add services/harness/budget.py services/harness/trace.py services/harness/orchestrator.py tests/test_harness_orchestrator.py
git commit -m "feat: add budgeted harness orchestrator"
```

---

### Task 5: File Sink and CLI

**Files:**
- Create: `services/harness/sinks.py`
- Create: `scripts/run_harness_x.py`
- Test: `tests/test_harness_e2e.py`

**Interfaces:**
- Consumes: `HarnessOrchestrator`, `ReplayXSource`, `FileRunSink`
- Produces: `FileRunSink(root: str | Path)`
- Produces CLI: `python scripts/run_harness_x.py --fixture PATH --out DIR --max-cost-usd FLOAT`

- [ ] **Step 1: Write failing E2E tests**

Test names and assertions:
- `test_e2e_writes_run_case_trace_and_cost` — run on the synthetic fixture; files `run.json`, `case.json`, `trace.jsonl`, `cost.json` exist; `case.status == "ACCEPTED"`.
- `test_e2e_replay_is_idempotent` — run twice; both `case.json` files have the same `case_id`; no more than two run directories.
- `test_cli_returns_zero_and_prints_case_id` — invoke `main([...])`; return code `0`; stdout contains the case id.

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_harness_e2e.py -v`
Expected: FAIL with import errors for `services.harness.sinks` or missing CLI.

- [ ] **Step 3: Implement sink and CLI**

`FileRunSink.write` creates `root/<run_id>/`, writes:
- `run.json` — full `HarnessRun` with `model_dump(mode="json")`,
- `case.json` — only when `run.case` is not `None`,
- `trace.jsonl` — one JSON object per event,
- `cost.json` — `CostReport`.

CLI behavior:
- `--fixture` required for replay.
- `--query` optional for live Agent-Reach; `--limit` default 5.
- `--out` default `runs`.
- `--max-cost-usd` default `0.05`.
- `--live-source` switches to `AgentReachXSource`; otherwise `ReplayXSource`.
- Build rule agents by default. Print `run_id`, `case_id`, `status`, and output directory.
- Return `0` for `COMPLETED`, `NEEDS_REVIEW`, or `REJECTED`; return `2` for `FAILED` or `BUDGET_EXCEEDED`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_harness_e2e.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add services/harness/sinks.py scripts/run_harness_x.py tests/test_harness_e2e.py
git commit -m "feat: add harness file sink and CLI"
```

---

### Task 6: Full Validation and Documentation

**Files:**
- Create: `docs/harness-x-m1.md`
- Modify: `README.md` only if a short pointer is needed.
- Test: entire suite.

- [ ] **Step 1: Add documentation**

Document:
- what milestone 1 does and does not do,
- the commands (replay, live source, budget),
- Agent-Reach installation/login steps,
- fixture format,
- trust rules and the conservative Arbiter decision,
- how to inspect `run.json`, `case.json`, `trace.jsonl`, `cost.json`,
- the fact that the committed fixture must be a real X post.

- [ ] **Step 2: Run the full validation**

Run: `pytest tests/test_harness_contracts.py tests/test_harness_sources.py tests/test_harness_agents.py tests/test_harness_orchestrator.py tests/test_harness_e2e.py -v`
Expected: PASS.

Run the CLI on the synthetic fixture:
`python scripts/run_harness_x.py --fixture tests/fixtures/x_posts/synthetic_road_bandung_001.json --out /tmp/kawal-harness-run`
Expected: prints a `case_id` and writes four files.

- [ ] **Step 3: Commit**

```bash
git add docs/harness-x-m1.md README.md
git commit -m "docs: document harness X milestone 1"
```

---

### Task 7: Real X Fixture Capture (External Dependency)

**Files:**
- Create: `fixtures/x_posts/<real-post-id>.json`
- Modify: `tests/test_harness_e2e.py` only if the test should point at the real fixture (keep synthetic for CI; add a separate manual smoke test).

- [ ] **Step 1: Install Agent-Reach**

Follow the repository’s official install instructions. Do not install the PyPI package named `agent-reach`.

- [ ] **Step 2: Log in to X**

The human operator runs the login flow. This cannot be automated by the harness.

- [ ] **Step 3: Capture one post**

Run:
`agent-reach twitter search "jalan rusak bandung" --limit 5`

Save one raw result as `fixtures/x_posts/<real-post-id>.json`.

- [ ] **Step 4: Run the harness against the real fixture**

Run:
`python scripts/run_harness_x.py --fixture fixtures/x_posts/<real-post-id>.json --out runs/real-001`

Verify that `case.json`, `trace.jsonl`, and `cost.json` are written.

- [ ] **Step 5: Commit**

```bash
git add fixtures/x_posts/<real-post-id>.json
git commit -m "test: add real X fixture for harness milestone 1"
```

---

## Self-Review

- Spec coverage: sources, ports, agents, trust rules, budget, trace, sink, CLI, acceptance criteria, and the real-fixture dependency each map to a task.
- Type consistency: `AgentTask`, `AgentResult`, `ComplaintCase`, `HarnessRun`, and `RunSink` are defined once in Task 1 and consumed by later tasks.
- Review focus: empty post, non-complaint, missing location, verifier disagreement, budget exhaustion, and replay idempotency each have a named test in Tasks 3–5.
- Real X capture is isolated in Task 7 because it depends on human login; the code and synthetic tests can be completed before it.
