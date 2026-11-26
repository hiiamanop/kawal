# KAWAL Harness Monitor Console Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the legacy WhatsApp dashboard with a ChatGPT-style local console for monitoring live X complaints, inspecting trust decisions, and preparing department forwarding packets.

**Architecture:** FastAPI serves a local HTML/CSS/JS console backed by SQLite. A scan service reads live X or replay fixtures, runs the existing harness, scores cases by engagement/recency/risk/trust, maps categories to departments, and stores operator workflow state. No automatic sending.

**Tech Stack:** Python 3.15, FastAPI, uvicorn, sqlite3, vanilla HTML/CSS/JS, pytest.

**Spec:** `docs/superpowers/specs/2026-11-26-kawal-harness-monitor-ui-design.md`

**Prerequisite:** complete `docs/superpowers/plans/2026-11-26-rule-category-coverage.md` first.

## Global Constraints

- Default bind: `127.0.0.1:8088`.
- No PostgreSQL for the monitor console.
- No automatic tickets, emails, WhatsApp messages, or citizen replies.
- Forwarding packets are local drafts marked `DRAFT — NOT SENT`.
- No CDN or frontend build step.
- UI copy is Indonesian; code, tests, and documentation are English.
- Only one scan may run at a time.
- Core tests must not call live X.
- Do not modify legacy WhatsApp pipeline internals; only remove its dashboard.
- `runs/` and `runs/monitor/` stay out of git.

## Review Focus

- Live X failure or rate limit must show an error and create no fake cases.
- A post returned by multiple queries must remain one case, preserve operator state, and refresh engagement only.
- A failed or budget-exceeded analysis must be stored as an error and hidden from the default complaint feed.
- Invalid status transitions must not change state or write a misleading audit row.
- Forwarding must produce draft files only and never perform network sends.
- The console must render offline without external CSS or JS CDNs.

---

### Task 1: Monitor models and SQLite store

**Files:**
- Create: `services/monitor/__init__.py`
- Create: `services/monitor/config.py`
- Create: `services/monitor/models.py`
- Create: `services/monitor/store.py`
- Test: `tests/test_monitor_store.py`

**Interfaces:**
- Produces: `OperatorStatus`, `AnalysisStatus`, `MonitorCase`, `ScanRequest`, `ScanResult`, `CaseUpdate`, `ForwardPacket`, `MonitorStats` in `services/monitor/models.py`.
- Produces: `MonitorStore(db_path: str | Path)` with `create_schema()`, `upsert_case(case, query) -> bool`, `get_case(case_id) -> MonitorCase | None`, `list_cases(...) -> list[MonitorCase]`, `update_case(case_id, update) -> MonitorCase`, `record_scan(result) -> None`, `stats() -> MonitorStats`, `list_actions(case_id) -> list[dict]`, `list_case_queries(case_id) -> list[dict]`.
- Produces: `InvalidTransition` exception.

Test helpers for this plan live in `tests/test_monitor_helpers.py`: `_run(text) -> tuple[SocialPost, HarnessRun]`, `_monitor_case(post, run, **overrides) -> MonitorCase`, `_complaint_case(...) -> ComplaintCase`, `_store_with_case(tmp_path, **overrides) -> MonitorStore`, `_write_fixture(dir, post_id, text) -> Path`, and `_request(dir, query) -> ScanRequest`.

- [ ] **Step 1: Write failing model/store tests**

```python
def test_upsert_case_creates_once_and_preserves_operator_state(tmp_path):
    store = MonitorStore(tmp_path / "monitor.db")
    store.create_schema()
    post, run = _run("Min, jalan di Jl. Kopo rusak parah.")
    case = _monitor_case(post, run)
    assert store.upsert_case(case, "jalan rusak") is True
    store.update_case(case.case_id, CaseUpdate(status="MONITORING", note="Cek lokasi"))
    case2 = case.model_copy(update={"last_seen_at": case.last_seen_at + timedelta(hours=1)})
    assert store.upsert_case(case2, "trotoar rusak") is False
    saved = store.get_case(case.case_id)
    assert saved.operator_status == "MONITORING"
    assert saved.notes == "Cek lokasi"
    assert {row["query"] for row in store.list_case_queries(case.case_id)} == {"jalan rusak", "trotoar rusak"}

def test_invalid_status_transition_is_rejected(tmp_path):
    store = _store_with_case(tmp_path)
    with pytest.raises(InvalidTransition):
        store.update_case("case-x-1", CaseUpdate(status="FORWARDED"))

def test_failed_analysis_is_hidden_from_default_feed(tmp_path):
    store = _store_with_case(tmp_path, analysis_status="FAILED", viral_score=None, department=None)
    assert store.list_cases() == []
    assert len(store.list_cases(analysis_status="FAILED")) == 1

def test_stats_and_actions(tmp_path):
    store = _store_with_case(tmp_path)
    store.update_case("case-x-1", CaseUpdate(status="READY_TO_FORWARD", note="Siap"))
    assert store.stats().ready_to_forward == 1
    assert [a["new_status"] for a in store.list_actions("case-x-1")][-1] == "READY_TO_FORWARD"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_monitor_store.py -v`
Expected: FAIL with missing module.

- [ ] **Step 3: Implement models, config, and store**

`models.py` defines:

```python
OperatorStatus = Literal["NEW", "MONITORING", "READY_TO_FORWARD", "FORWARDED", "DISMISSED"]
AnalysisStatus = Literal["OK", "FAILED", "BUDGET_EXCEEDED"]

class MonitorCase(BaseModel):
    case_id: str
    post: SocialPost
    run: HarnessRun
    viral_score: int | None = Field(default=None, ge=0, le=100)
    department: str | None = None
    operator_status: OperatorStatus = "NEW"
    analysis_status: AnalysisStatus = "OK"
    notes: str = ""
    first_seen_at: datetime
    last_seen_at: datetime
    updated_at: datetime
```

`store.py` uses `sqlite3` with JSON payload columns. Tables:

```text
monitor_cases(case_id TEXT PRIMARY KEY, post_json TEXT, run_json TEXT,
              case_json TEXT, viral_score INTEGER, department TEXT,
              trust_status TEXT, operator_status TEXT, analysis_status TEXT,
              notes TEXT, first_seen_at TEXT, last_seen_at TEXT, updated_at TEXT)
monitor_case_queries(case_id TEXT, query TEXT, seen_at TEXT,
                     PRIMARY KEY(case_id, query))
monitor_scans(scan_id TEXT PRIMARY KEY, query TEXT, source_mode TEXT,
              analysis_mode TEXT, new_cases INTEGER, updated_cases INTEGER,
              failed INTEGER, errors_json TEXT, started_at TEXT, finished_at TEXT)
monitor_actions(id INTEGER PRIMARY KEY AUTOINCREMENT, case_id TEXT,
                old_status TEXT, new_status TEXT, note TEXT, actor TEXT,
                at TEXT)
```

Allowed transitions are the matrix in the spec. `update_case` writes an action only when status changes; note-only updates do not write a status action. Upsert preserves `operator_status`, `notes`, and `first_seen_at`, and links every query to the case.

Default `list_cases()` returns only `analysis_status == "OK"`, `trust_status` in (`ACCEPTED`, `NEEDS_REVIEW`), and `operator_status != "DISMISSED"`. Passing `analysis_status` or `trust` explicitly overrides that view for the Errors/Archive filters.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_monitor_store.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add services/monitor/__init__.py services/monitor/config.py services/monitor/models.py services/monitor/store.py tests/test_monitor_store.py
git commit -m "feat: add monitor models and sqlite store"
```

---

### Task 2: Viral scoring, department routing, and forwarding packets

**Files:**
- Create: `services/monitor/scoring.py`
- Create: `services/monitor/routing.py`
- Create: `services/monitor/forwarding.py`
- Test: `tests/test_monitor_scoring.py`
- Test: `tests/test_monitor_forwarding.py`

**Interfaces:**
- Consumes: `MonitorCase`, `ComplaintCase`, `SocialPost`.
- Produces: `viral_score(post: SocialPost, case: ComplaintCase | None, now: datetime) -> int`
- Produces: `recommend_department(case: ComplaintCase) -> str`
- Produces: `write_forward_packet(case: MonitorCase, note: str, output_dir: Path) -> ForwardPacket`

- [ ] **Step 1: Write failing scoring/routing tests**

```python
def test_viral_score_uses_engagement_recency_risk_and_trust():
    now = datetime(2026, 11, 26, 12, 0, tzinfo=timezone.utc)
    post = _post(likes=100, views=1000, posted_at=now - timedelta(hours=1))
    case = _complaint_case(risk="HIGH", status="ACCEPTED")
    assert viral_score(post, case, now) == 74

def test_missing_engagement_is_zero_not_an_error():
    now = datetime(2026, 11, 26, 12, 0, tzinfo=timezone.utc)
    assert viral_score(_post(likes=None, views=None, posted_at=now), _complaint_case(), now) >= 0

def test_department_routing_uses_category_mapping():
    assert recommend_department(_complaint_case(category=Category.WASTE)) == "Dinas Lingkungan Hidup (DLH)"
    assert recommend_department(_complaint_case(category=Category.ROAD)) == "Dinas Sumber Daya Air dan Bina Marga (DSDABM)"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_monitor_scoring.py -v`
Expected: FAIL with missing module.

- [ ] **Step 3: Implement scoring and routing**

Use the exact formula and weights from the spec:

```text
engagement_raw  = likes + 0.05 * views
engagement_norm = min(1, log1p(engagement_raw) / log1p(10000))
recency_norm    = max(0, 1 - hours_since_post / 72)
risk_norm       = URGENT 1.0, HIGH 0.75, MEDIUM 0.5, LOW 0.25
trust_norm      = ACCEPTED 1.0, NEEDS_REVIEW 0.6, REJECTED 0.0
viral_score = round(100 * (0.45 * engagement_norm + 0.25 * recency_norm + 0.20 * risk_norm + 0.10 * trust_norm))
```

Put weights in `config.py`. `recommend_department` calls `resolve_department_name(case.triage.category)`.

- [ ] **Step 4: Write failing forwarding test**

```python
def test_forward_packet_is_a_local_draft(tmp_path):
    case = _monitor_case_with_note()
    packet = write_forward_packet(case, "Mohon diteruskan ke DSDABM.", tmp_path)
    md = Path(packet.markdown_path).read_text()
    data = json.loads(Path(packet.json_path).read_text())
    assert "DRAFT — NOT SENT" in md
    assert case.case_id in md
    assert data["case_id"] == case.case_id
    assert data["department"] == case.department
```

- [ ] **Step 5: Run forwarding test to verify it fails**

Run: `pytest tests/test_monitor_forwarding.py -v`
Expected: FAIL.

- [ ] **Step 6: Implement forwarding**

Write both files under `output_dir / f"{case.case_id}-{timestamp}.md"` and `.json`. The Markdown includes post URL, text, author, posted time, engagement, category, risk, location, trust status, department, operator note, and the `DRAFT — NOT SENT` line. The JSON is the machine-readable equivalent. Do not perform network calls.

- [ ] **Step 7: Run tests**

Run: `pytest tests/test_monitor_scoring.py tests/test_monitor_forwarding.py -v`
Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add services/monitor/scoring.py services/monitor/routing.py services/monitor/forwarding.py tests/test_monitor_scoring.py tests/test_monitor_forwarding.py
git commit -m "feat: add monitor scoring routing and forwarding"
```

---

### Task 3: Live/replay scan service

**Files:**
- Create: `services/monitor/scan.py`
- Test: `tests/test_monitor_scan.py`

**Interfaces:**
- Consumes: `MonitorStore`, `MonitorCase`, `ScanRequest`, `ScanResult`.
- Produces: `run_scan(request: ScanRequest, store: MonitorStore, *, source=None, orchestrator=None) -> ScanResult`
- Produces: `build_source(request: ScanRequest) -> SourcePort`
- Produces: `build_orchestrator(analysis_mode: str) -> HarnessOrchestrator`

- [ ] **Step 1: Write failing replay scan tests**

```python
def test_replay_scan_stores_and_ranks_cases(tmp_path):
    fixture_dir = _write_fixture(tmp_path, "post-1", "Min, jalan di Jl. Kopo rusak parah.")
    store = _store(tmp_path)
    request = ScanRequest(query="jalan rusak", limit=10, source_mode="replay", fixture_dir=str(fixture_dir))
    result = run_scan(request, store)
    assert result.new_cases == 1
    assert result.cases[0].department == "Dinas Sumber Daya Air dan Bina Marga (DSDABM)"
    assert result.cases[0].viral_score is not None

def test_second_scan_updates_without_duplicating(tmp_path):
    store = _store(tmp_path)
    first = run_scan(_request(tmp_path, "jalan rusak"), store)
    store.update_case(first.cases[0].case_id, CaseUpdate(status="MONITORING", note="Pantau"))
    second = run_scan(_request(tmp_path, "trotoar rusak"), store)
    assert second.updated_cases == 1
    saved = store.get_case(first.cases[0].case_id)
    assert saved.operator_status == "MONITORING"
    assert saved.notes == "Pantau"

def test_failed_analysis_is_stored_as_error(tmp_path):
    result = run_scan(_request(tmp_path, "jalan rusak"), store, orchestrator=FailingOrchestrator())
    assert result.failed == 1
    assert result.cases[0].analysis_status == "FAILED"
    assert result.cases[0].viral_score is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_monitor_scan.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement scan**

`build_source` returns `AgentReachXSource()` for live and `ReplayXSource(fixture_dir)` for replay. `build_orchestrator` returns the four rule agents, or the corresponding LLM agents when `analysis_mode == "llm"` and an API key is configured; if LLM is requested without a key, raise `ValueError` before scanning. Use `OpenRouterChatClient()` plus `LLMTriageAgent(client)`, `LLMExtractionAgent(client)`, `LLMVerifierAgent(client)`, and `LLMArbiterAgent(client)`, matching `scripts/compare_harness_agents.py`.

`run_scan`:

- fetch posts from the source;
- run the orchestrator for each post;
- use `deterministic_case_id(post)` when `run.case is None`;
- set `analysis_status` from `run.status`: `COMPLETED`/`NEEDS_REVIEW`/`REJECTED` -> `OK`, `FAILED` -> `FAILED`, `BUDGET_EXCEEDED` -> `BUDGET_EXCEEDED`;
- compute score/department only when a case exists;
- call `store.upsert_case(case, request.query)` and count new vs updated;
- catch per-post exceptions as failed cases, but do not invent a `ComplaintCase`;
- return `ScanResult` with errors.

- [ ] **Step 4: Add source failure test**

```python
def test_source_failure_returns_error_without_cases(tmp_path):
    with pytest.raises(SourceError):
        run_scan(_request(tmp_path, "jalan rusak"), _store(tmp_path), source=FailingSource())
```

- [ ] **Step 5: Run tests**

Run: `pytest tests/test_monitor_scan.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add services/monitor/scan.py tests/test_monitor_scan.py
git commit -m "feat: add monitor scan service"
```

---

### Task 4: FastAPI monitor API

**Files:**
- Create: `services/monitor/app.py`
- Create: `services/monitor/static/app.css`
- Create: `services/monitor/static/app.js`
- Create: `services/monitor/templates/index.html`
- Test: `tests/test_monitor_api.py`

**Interfaces:**
- Consumes: `MonitorStore`, `run_scan`, `write_forward_packet`.
- Produces: `create_monitor_app(db_path: str | Path | None = None) -> FastAPI`

- [ ] **Step 1: Write failing API tests**

```python
def test_index_renders_console_shell(tmp_path):
    client = TestClient(create_monitor_app(db_path=tmp_path / "monitor.db"))
    response = client.get("/")
    assert response.status_code == 200
    assert "KAWAL Monitor" in response.text
    assert "Keluhan viral" in response.text
    assert "Scan sekarang" in response.text

def test_legacy_tracking_route_is_gone(tmp_path):
    client = TestClient(create_monitor_app(db_path=tmp_path / "monitor.db"))
    assert client.get("/track/TKT-1").status_code == 404
    assert client.get("/api/tickets").status_code == 404

def test_scan_then_update_then_forward(tmp_path):
    fixture_dir = _write_fixture(tmp_path, "post-1", "Min, jalan di Jl. Kopo rusak parah.")
    client = TestClient(create_monitor_app(db_path=tmp_path / "monitor.db"))
    scan = client.post(
        "/api/monitor/scan",
        json={
            "query": "jalan rusak",
            "limit": 10,
            "source_mode": "replay",
            "analysis_mode": "rule",
            "fixture_dir": str(fixture_dir),
        },
    )
    assert scan.status_code == 200
    case = client.get("/api/monitor/cases").json()["cases"][0]
    update = client.patch(f"/api/monitor/cases/{case['case_id']}", json={"status": "MONITORING", "note": "Pantau"})
    assert update.status_code == 200
    forward = client.post(f"/api/monitor/cases/{case['case_id']}/forward", json={"note": "Teruskan"})
    assert forward.status_code == 200
    assert "DRAFT" in Path(forward.json()["markdown_path"]).read_text()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_monitor_api.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement the app**

Routes:

```text
GET    /
GET    /static/*
GET    /api/monitor/cases
GET    /api/monitor/cases/{case_id}
PATCH  /api/monitor/cases/{case_id}
POST   /api/monitor/scan
POST   /api/monitor/cases/{case_id}/forward
GET    /api/monitor/stats
GET    /api/monitor/health
```

Use a module-level `threading.Lock` per app for scan concurrency. If locked, return HTTP 409. Catch source errors and return HTTP 502 with a stable code. Catch `InvalidTransition` and return HTTP 409. Return 404 for missing cases. `health` only checks local app/database and whether `agent-reach` is on `PATH`; it must not call X.

- [ ] **Step 4: Run API tests**

Run: `pytest tests/test_monitor_api.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add services/monitor/app.py services/monitor/static/app.css services/monitor/static/app.js services/monitor/templates/index.html tests/test_monitor_api.py
git commit -m "feat: add monitor api and console shell"
```

---

### Task 5: Build the ChatGPT-style console UI

**Files:**
- Modify: `services/monitor/templates/index.html`
- Modify: `services/monitor/static/app.css`
- Modify: `services/monitor/static/app.js`
- Test: `tests/test_monitor_api.py`

**Interfaces:**
- Consumes: monitor API endpoints from Task 4.
- Produces: no new Python API.

- [ ] **Step 1: Extend HTML smoke test**

```python
def test_console_has_feed_detail_and_composer(tmp_path):
    html = TestClient(create_monitor_app(db_path=tmp_path / "monitor.db")).get("/").text
    for text in ["Keluhan viral", "Tanya atau scan keluhan", "Detail", "Siap Diteruskan", "Monitoring"]:
        assert text in html
    assert "cdn.tailwindcss.com" not in html
```

- [ ] **Step 2: Implement the UI**

Layout requirements:

- left sidebar: brand, `Scan sekarang`, navigation buckets, category/risk/status filters;
- main area: summary cards and ranked complaint feed;
- detail slide-over: post text, source link, engagement, analysis panels, trace, status actions, forwarding;
- bottom composer: query input and scan button;
- states: loading, empty, scan error, rate limit, no Agent Reach session;
- light neutral ChatGPT-like theme, system fonts, rounded cards, no gradients/neon/CDN.

`app.js` loads `/api/monitor/stats` and `/api/monitor/cases`, renders cards, opens detail, patches status, posts scans, and downloads forwarding files. Escape HTML before injecting post text.

- [ ] **Step 3: Run UI/API tests**

Run: `pytest tests/test_monitor_api.py -v`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add services/monitor/templates/index.html services/monitor/static/app.css services/monitor/static/app.js tests/test_monitor_api.py
git commit -m "feat: build monitor console ui"
```

---

### Task 6: Replace the legacy dashboard and runner

**Files:**
- Delete: `services/dashboard/app.py`
- Delete: `services/dashboard/__init__.py`
- Delete: `scripts/run_dashboard.py`
- Delete: `tests/test_dashboard.py`
- Create: `scripts/run_monitor.py`
- Modify: `.gitignore`
- Modify: `README.md`
- Modify: `docs/testing.md`

- [ ] **Step 1: Add the runner**

`scripts/run_monitor.py` parses:

```text
--host (default 127.0.0.1)
--port (default 8088)
--db-path (default runs/monitor/monitor.db)
```

It calls `create_monitor_app(db_path=...)` and `uvicorn.run`.

- [ ] **Step 2: Remove legacy dashboard routes and tests**

Delete the legacy dashboard package, runner, and test file. Verify no import remains:

Run: `grep -R "create_dashboard_app\|run_dashboard\|services.dashboard" services scripts tests`
Expected: no matches.

- [ ] **Step 3: Update gitignore and docs**

Add `runs/` to `.gitignore`. Update README layout/status and replace the old dashboard command with:

```bash
python scripts/run_monitor.py --port 8088
```

Update `docs/testing.md` with the actual monitor test command and counts.

- [ ] **Step 4: Run the app smoke test**

Run: `pytest tests/test_monitor_api.py tests/test_monitor_store.py tests/test_monitor_scan.py tests/test_monitor_scoring.py tests/test_monitor_forwarding.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "refactor: replace legacy dashboard with monitor console"
```

---

### Task 7: Full validation and PR

**Files:**
- Modify: `docs/testing.md` and README only if counts changed.

- [ ] **Step 1: Run core CI**

Run: `bash scripts/ci.sh`
Expected: PASS; record the exact counts.

- [ ] **Step 2: Run research lane**

Run: `bash scripts/research_ci.sh`
Expected: PASS; quarantine remains empty.

- [ ] **Step 3: Check whitespace**

Run: `git diff --check`
Expected: no output.

- [ ] **Step 4: Update counts**

If counts changed, update `README.md` and `docs/testing.md` with the exact output.

- [ ] **Step 5: Commit and push**

```bash
git add README.md docs/testing.md
git commit -m "docs: update monitor console validation counts"
git push -u origin feat/harness-monitor-ui
```

- [ ] **Step 6: Open PR**

Create a PR against `main` summarizing the rule-coverage prerequisite, monitor console, tests, and the live/replay limitations. Do not merge.
