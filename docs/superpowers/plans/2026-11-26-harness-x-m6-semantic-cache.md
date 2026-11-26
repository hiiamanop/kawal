# Harness X M6 Semantic Cache Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a versioned pgvector-backed semantic cache that skips the LLM on safe exact or high-confidence repeated complaints, while preserving the harness trust guards and measuring false reuse and cost savings.

**Architecture:** A `CacheFirstRunner` wraps the existing `HarnessOrchestrator`. It always runs free deterministic Triage, Extraction, and Verifier guards, then performs an exact or pgvector lookup. A hit produces a new case for the current post; a miss or failed guard delegates to the unchanged inner orchestrator. Only accepted, rule-consistent, PII-free cases are recorded. The legacy WhatsApp cache stays untouched.

**Tech Stack:** Python 3.11+, Pydantic 2, NumPy, psycopg 3, PostgreSQL 16 + pgvector, pytest, existing cassette/LLM harness.

**Spec:** `docs/superpowers/specs/2026-11-26-harness-x-m6-semantic-cache-design.md`

## Global Constraints

- X-only harness. Do not change WhatsApp pipeline behavior.
- Cache only `status=ACCEPTED`, `trust_decision=AGREED`, `location_quality=SPECIFIC`, `source_intent=CITIZEN_COMPLAINT`, rule/LLM category agreement, and PII-free payloads.
- Never store the full complaint body in the cache.
- Reuse requires matching namespace, analysis version, embedding model, category, and source intent.
- Deterministic source-intent, location, Triage, Extraction, and Verifier guards run on every cache hit.
- Semantic reuse is disabled by default with the deterministic fallback embedder.
- Core CI must stay offline. Postgres tests are marked `integration`.
- Use existing `Category`, `RiskLevel`, `SourceIntent`, `LocationQuality`, and `ComplaintCase` types.
- Code and docs in English. No new Python runtime dependency; pgvector is a database extension.
- Run `bash scripts/ci.sh` before committing/pushing. Create PR only when green.
- Never print, commit, or expose `.env` secrets.

## Review Focus

- **Different incidents at a generic place:** `Stasiun Bandung` and `Stasiun Cimahi` must not be treated as the same location.
- **Rule/LLM category disagreement:** an accepted LLM case whose category differs from deterministic Triage/Verifier must not be cached.
- **PII in a rule-accepted case:** a case accepted by the rule path must still be refused by the cache when it contains NIK, phone, email, or NPWP.
- **Stale versions:** a cache entry from a different analysis version or embedding model must never be reused.
- **Non-citizen content:** news, opinions, questions, and announcements that share complaint keywords must fall back, not reuse.

---

### Task 1: pgvector Migration and Compose Image

**Files:**
- Create: `infra/migrations/014_m6_harness_semantic_cache.sql`
- Create: `supabase/migrations/20261126000000_m6_harness_semantic_cache.sql`
- Modify: `docker-compose.yml:74`
- Test: `tests/test_harness_cache_migration.py`

**Interfaces:**
- Consumes: existing `vector` and `pgcrypto` extensions.
- Produces: table `harness_case_cache`, HNSW index, and metadata indexes used by Task 5.

- [ ] **Step 1: Write the failing migration tests**

```python
def test_migration_exists_in_both_trees():
    assert INFRA_MIGRATION.is_file()
    assert SUPABASE_MIGRATION.is_file()


def test_migration_uses_pgvector_and_hnsw():
    sql = INFRA_MIGRATION.read_text(encoding="utf-8").lower()
    assert "harness_case_cache" in sql
    assert "embedding vector(768)" in sql
    assert "using hnsw" in sql
    assert "vector_cosine_ops" in sql
    assert "record_guards jsonb" in sql


def test_compose_uses_pgvector_image():
    compose = COMPOSE.read_text(encoding="utf-8")
    assert "pgvector/pgvector:pg16" in compose
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_harness_cache_migration.py -q`
Expected: FAIL because the migration files and image change do not exist.

- [ ] **Step 3: Create both migrations**

Use the schema and indexes from spec section 8.1. Both files must be byte-for-byte equivalent except for the filename. Include `CREATE EXTENSION IF NOT EXISTS vector;` and `pgcrypto`.

- [ ] **Step 4: Change the Compose Postgres image**

Change `image: postgres:16-alpine` to `image: pgvector/pgvector:pg16`. Do not change ports, volume, or healthcheck.

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_harness_cache_migration.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add infra/migrations/014_m6_harness_semantic_cache.sql supabase/migrations/20261126000000_m6_harness_semantic_cache.sql docker-compose.yml tests/test_harness_cache_migration.py
git commit -m "feat: add harness semantic cache migration"
```

---

### Task 2: Embedder Identity and Fallback Detection

**Files:**
- Modify: `services/intelligence/embedder.py`
- Test: `tests/test_harness_embedder.py`

**Interfaces:**
- Produces: `TextEmbedder.model_id: str`, `TextEmbedder.is_fallback: bool`; `HarnessAnalysisCache` uses these to version and gate semantic reuse.

- [ ] **Step 1: Write the failing tests**

```python
def test_deterministic_embedder_identity():
    embedder = DeterministicFallbackEmbedder(dim=768)
    assert embedder.model_id == "deterministic-hash-768-v1"
    assert embedder.is_fallback is True


def test_indobert_missing_artifact_reports_fallback(tmp_path):
    embedder = IndoBertEmbedder(model_path=tmp_path / "missing")
    assert embedder.is_fallback is True
    assert embedder.model_id == "indobert-fallback-768"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_harness_embedder.py -q`
Expected: FAIL with `AttributeError` for `model_id`.

- [ ] **Step 3: Implement the identity fields**

Add class attributes to `TextEmbedder`: `model_id = "unknown"`, `is_fallback = False`. Set the deterministic values in `DeterministicFallbackEmbedder.__init__`. In `IndoBertEmbedder.__init__`, set `model_id` to the chosen artifact directory name on success and `indobert-fallback-768` on fallback; set `is_fallback` accordingly.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_harness_embedder.py tests/test_semantic_cache.py -q -m "not research"`
Expected: PASS for the new tests and existing non-research cache tests.

- [ ] **Step 5: Commit**

```bash
git add services/intelligence/embedder.py tests/test_harness_embedder.py
git commit -m "feat: version embedder identity"
```

---

### Task 3: Cache Models, Hash Helpers, and In-Memory Store

**Files:**
- Create: `services/harness/cache.py`
- Create: `services/harness/cache_store.py`
- Test: `tests/test_harness_cache_store.py`

**Interfaces:**
- Produces:
  - `CachedAnalysis`, `CacheEntry`, `CacheQuery`, `CacheCandidate`, `CacheLookupResult`, `CacheRecordResult`
  - `HarnessCacheStore` protocol
  - `compute_text_hash(text: str) -> str`
  - `compute_cache_key(namespace, analysis_version, embedding_model, text_hash) -> str`
  - `InMemoryHarnessCacheStore`
- The store methods are:
  - `find_exact(namespace, analysis_version, embedding_model, text_hash) -> CacheEntry | None`
  - `find_similar(query: CacheQuery) -> list[CacheCandidate]`
  - `save(entry: CacheEntry) -> None`
  - `increment_hit(cache_key: str) -> None`
  - `reset(namespace: str, analysis_version: str) -> int`

- [ ] **Step 1: Write the failing store tests**

Use a helper that returns a valid 768-dim entry. Assert:

```python
def test_in_memory_exact_lookup_is_scoped_by_version():
    store = InMemoryHarnessCacheStore()
    store.save(_entry(text_hash="a" * 64, analysis_version="v1"))
    assert store.find_exact("harness-x", "v1", "deterministic-hash-768-v1", "a" * 64) is not None
    assert store.find_exact("harness-x", "v2", "deterministic-hash-768-v1", "a" * 64) is None


def test_in_memory_semantic_lookup_filters_category_and_source_intent():
    store = InMemoryHarnessCacheStore()
    store.save(_entry(category=Category.ROAD, source_intent="CITIZEN_COMPLAINT"))
    store.save(_entry(cache_key="b" * 64, category=Category.WASTE, source_intent="CITIZEN_COMPLAINT"))
    results = store.find_similar(
        CacheQuery(
            namespace="harness-x",
            analysis_version="v1",
            embedding_model="deterministic-hash-768-v1",
            embedding=tuple([1.0] + [0.0] * 767),
            category=Category.ROAD,
            source_intent="CITIZEN_COMPLAINT",
            min_similarity=0.9,
            limit=5,
        )
    )
    assert [candidate.entry.category for candidate in results] == [Category.ROAD]


def test_in_memory_reset_only_removes_matching_namespace_and_version():
    store = InMemoryHarnessCacheStore()
    store.save(_entry(namespace="harness-x", analysis_version="v1"))
    store.save(_entry(cache_key="c" * 64, namespace="other", analysis_version="v1"))
    assert store.reset("harness-x", "v1") == 1
    assert store.count == 1


def test_cache_entry_has_no_full_post_body_field():
    entry = _entry()
    assert "raw_text" not in entry.model_dump()
    assert "post_text" not in entry.model_dump()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_harness_cache_store.py -q`
Expected: FAIL because the modules do not exist.

- [ ] **Step 3: Implement models and helpers**

Use frozen Pydantic models with `extra="forbid"`. `CacheEntry.embedding` must be exactly 768 floats. `analysis` is a validated `CachedAnalysis`, not a free-form dict. `record_guards` is `dict[str, object]`.

- [ ] **Step 4: Implement the in-memory store**

Use a dict keyed by `cache_key`, plus a hash index keyed by `(namespace, analysis_version, embedding_model, text_hash)`. Use NumPy cosine similarity. `increment_hit` returns a new entry via `model_copy` with updated `hit_count` and `last_hit_at`.

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_harness_cache_store.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add services/harness/cache.py services/harness/cache_store.py tests/test_harness_cache_store.py
git commit -m "feat: add harness cache models and memory store"
```

---

### Task 4: Cache Engine Lookup, Record Gates, and PII Checks

**Files:**
- Modify: `services/harness/cache.py`
- Modify: `services/harness/cache_store.py:1-120`
- Test: `tests/test_harness_cache_engine.py`

**Interfaces:**
- Consumes: Task 3 models and in-memory store; Task 2 embedder identity.
- Produces: `HarnessAnalysisCache` with:
  - `lookup(text, category, source_intent) -> CacheLookupResult`
  - `record(case, rule_triage, rule_verification, run_id, location_entities) -> CacheRecordResult`
  - `increment_hit(entry) -> None`
  - `reset() -> int`
  - `embedding_model`, `namespace`, `analysis_version`

- [ ] **Step 1: Write the failing engine tests**

```python
def test_record_stores_accepted_rule_consistent_case(cache, accepted_case, rule_outputs):
    result = cache.record(
        case=accepted_case,
        rule_triage=rule_outputs.triage,
        rule_verification=rule_outputs.verification,
        run_id="run-1",
        location_entities=("kopo",),
    )
    assert result.stored is True
    assert cache.lookup(accepted_case.post.text, Category.ROAD, "CITIZEN_COMPLAINT").hit


def test_record_rejects_rule_category_disagreement(cache, accepted_case, rule_outputs):
    result = cache.record(
        case=accepted_case,
        rule_triage=rule_outputs.triage.model_copy(update={"category": Category.DRAINAGE_FLOOD}),
        rule_verification=rule_outputs.verification,
        run_id="run-1",
        location_entities=("kopo",),
    )
    assert result.stored is False
    assert "CACHE_RULE_DISAGREEMENT" in result.reason_codes


def test_record_rejects_pii(cache, pii_case, rule_outputs):
    result = cache.record(
        case=pii_case,
        rule_triage=rule_outputs.triage,
        rule_verification=rule_outputs.verification,
        run_id="run-1",
        location_entities=("kopo",),
    )
    assert result.stored is False
    assert "CACHE_PII_BLOCKED" in result.reason_codes


def test_semantic_lookup_is_disabled_for_fallback_by_default(cache):
    cache.record(...)
    result = cache.lookup("jalan rusak lain di kopo", Category.ROAD, "CITIZEN_COMPLAINT")
    assert result.hit is False
    assert "SEMANTIC_DISABLED_FALLBACK" in result.reason_codes
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_harness_cache_engine.py -q`
Expected: FAIL because `HarnessAnalysisCache` does not exist.

- [ ] **Step 3: Implement lookup**

Exact lookup first. If no exact and `embedder.is_fallback` and `allow_fallback_semantic=False`, return a miss with `SEMANTIC_DISABLED_FALLBACK`. Otherwise embed and query with the configured threshold and exact metadata filters.

- [ ] **Step 4: Implement record gates**

Refuse with `CACHE_NOT_ACCEPTED`, `CACHE_NOT_SPECIFIC`, `CACHE_NOT_CITIZEN`, `CACHE_RULE_NOT_COMPLAINT`, `CACHE_RULE_DISAGREEMENT`, `CACHE_EVIDENCE_UNSUPPORTED`, or `CACHE_PII_BLOCKED` as applicable. Scan `case.model_dump_json()` and the supplied location entities/spans with `contains_pii`. Build `record_guards` from the decisions.

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_harness_cache_engine.py tests/test_harness_cache_store.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add services/harness/cache.py services/harness/cache_store.py tests/test_harness_cache_engine.py
git commit -m "feat: add cache record gates and lookups"
```

---

### Task 5: PgVector Store

**Files:**
- Modify: `services/harness/cache_store.py`
- Test: `tests/test_harness_cache_pgvector.py`
- Test: `tests/test_harness_cache_pgvector_integration.py`

**Interfaces:**
- Produces: `PgVectorHarnessCacheStore(connection_factory)` implementing `HarnessCacheStore`.
- SQL uses `embedding <=> %s::vector`, `1 - (embedding <=> %s::vector)`, and parameterized metadata filters.

- [ ] **Step 1: Write failing unit tests with a mocked connection**

```python
def test_pgvector_find_similar_uses_cosine_operator():
    store, cur = _mock_store()
    cur.fetchall.return_value = []
    store.find_similar(_query())
    sql = cur.execute.call_args.args[0].lower()
    assert "<=>" in sql
    assert "::vector" in sql
    assert "namespace = %s" in sql


def test_pgvector_save_upserts_without_raw_text():
    store, cur = _mock_store()
    store.save(_entry())
    sql = cur.execute.call_args.args[0].lower()
    assert "insert into harness_case_cache" in sql
    assert "on conflict (cache_key)" in sql
    assert "raw_text" not in sql
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_harness_cache_pgvector.py -q`
Expected: FAIL because the class does not exist.

- [ ] **Step 3: Implement the store**

Serialize embeddings as a pgvector literal string. Parse JSONB columns and vector values in `_row_to_entry`. Commit writes. Fail closed with an empty result on database errors, matching the legacy store pattern.

- [ ] **Step 4: Add an integration test**

Mark it `@pytest.mark.integration`; skip when `KAWAL_TEST_DATABASE_URL` is absent. Assume the compose database already has migration 014 applied; do not apply migrations inside the test. Save an entry, find it exactly, find it semantically, reset the namespace, and assert it is gone.

- [ ] **Step 5: Run tests**

Run: `pytest tests/test_harness_cache_pgvector.py -q`
Expected: PASS. Optionally run the integration test against the compose Postgres on `127.0.0.1:54322`.

- [ ] **Step 6: Commit**

```bash
git add services/harness/cache_store.py tests/test_harness_cache_pgvector.py tests/test_harness_cache_pgvector_integration.py
git commit -m "feat: add pgvector cache store"
```

---

### Task 6: Cache Provenance in Runs and Batch Summaries

**Files:**
- Modify: `contracts/harness.py`
- Modify: `services/harness/batch.py`
- Test: `tests/test_harness_cache_provenance.py`

**Interfaces:**
- Produces: `CacheProvenance` and `HarnessRun.cache: CacheProvenance | None`.
- `BatchItemSummary` gains `cache_attempted`, `cache_hit`, `cache_reused`, `cache_match_type`, `cache_similarity`, `llm_calls`.

- [ ] **Step 1: Write failing tests**

```python
def test_harness_run_accepts_cache_provenance():
    run = _run(cache=CacheProvenance(attempted=True, hit=True, reused=True, match_type="EXACT", similarity=1.0, recorded=True))
    assert run.cache.reused is True


def test_batch_summary_copies_cache_fields():
    summary = HarnessBatchRunner._summarize(_post(), _run(cache=...))
    assert summary.cache_hit is True
    assert summary.cache_match_type == "EXACT"


def test_batch_summary_defaults_to_no_cache():
    summary = HarnessBatchRunner._summarize(_post(), _run(cache=None))
    assert summary.cache_attempted is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_harness_cache_provenance.py -q`
Expected: FAIL because the fields do not exist.

- [ ] **Step 3: Implement the contract and batch fields**

Add `recorded: bool = False` to `CacheProvenance`. `llm_calls` in the batch summary is computed from trace events whose `event == "agent.completed"` and `provider != "local"`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_harness_cache_provenance.py tests/test_harness_batch.py tests/test_harness_contracts.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add contracts/harness.py services/harness/batch.py tests/test_harness_cache_provenance.py
git commit -m "feat: expose cache provenance in runs"
```

---

### Task 7: Cache-First Runner and Reuse Guards

**Files:**
- Create: `services/harness/cache_router.py`
- Test: `tests/test_harness_cache_router.py`

**Interfaces:**
- Produces:
  - `normalize_location(value: str) -> str`
  - `locations_overlap(current: str | None, cached: Sequence[str]) -> bool`
  - `CacheFirstRunner(inner, cache, rule_triage=None, rule_extraction=None, rule_verifier=None)`
  - `CacheFirstRunner.run(post, run_id=None) -> HarnessRun`

- [ ] **Step 1: Write the failing location tests**

```python
def test_generic_place_tokens_do_not_overlap():
    assert locations_overlap("Stasiun Bandung", ["stasiun cimahi"]) is False
    assert locations_overlap("Stasiun Bandung", ["stasiun bandung"]) is True


def test_street_prefixes_are_normalized():
    assert locations_overlap("Jl. Merdeka", ["jalan merdeka"]) is True
    assert locations_overlap("Terusan Buah Batu", ["buah batu"]) is True
```

- [ ] **Step 2: Write failing router tests**

```python
def test_exact_cache_hit_skips_inner_orchestrator(cache, accepted_case, rule_outputs):
    inner = ExplodingInner()
    runner = CacheFirstRunner(inner, cache, rule_triage=RuleTriageAgent(), rule_extraction=RuleExtractionAgent(), rule_verifier=RuleVerifierAgent())
    cache.record(...)
    run = runner.run(accepted_case.post)
    assert run.status == "COMPLETED"
    assert run.cache.reused is True
    assert run.cost.total_cost_usd == 0.0


def test_location_mismatch_falls_back(cache, accepted_case, rule_outputs):
    inner = RecordingInner()
    run = CacheFirstRunner(inner, cache, ...).run(_post("Jl. Cimahi rusak"))
    assert run.cache.reused is False
    assert inner.called is True
```

Also test source-intent mismatch, rule category mismatch, evidence mismatch, risk elevation when the current rule Triage reports a higher risk than the cached Triage, and `inner` accepted run recording.

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/test_harness_cache_router.py -q`
Expected: FAIL because the router does not exist.

- [ ] **Step 4: Implement location helpers and guards**

Use the generic-token list from the spec. Require exact normalized equality, two significant token overlaps, or one distinctive token of at least four characters. Never count `stasiun`, `terminal`, `pasar`, `kecamatan`, `kelurahan`, `kota`, or `kabupaten` as distinctive.

- [ ] **Step 5: Implement the runner**

Run rule Triage, Extraction, and Verifier first. On a hit, require source intent `CITIZEN_COMPLAINT`, both rule agents complaint, category equality, `SPECIFIC` location, location overlap, current evidence support, and exact span presence for exact matches. Build the cached case with cached triage category, the higher of cached/current risk, current extraction/verification, and `CACHE_REUSED` reason codes. On a miss or failed guard, delegate to the inner orchestrator and record accepted inner cases.

- [ ] **Step 6: Run tests to verify they pass**

Run: `pytest tests/test_harness_cache_router.py tests/test_harness_orchestrator.py -q`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add services/harness/cache_router.py tests/test_harness_cache_router.py
git commit -m "feat: add cache-first harness runner"
```

---

### Task 8: Cache Metrics, Shared LLM CLI, and Evaluation CLI

**Files:**
- Create: `services/harness/cache_metrics.py`
- Create: `scripts/harness_llm_cli.py`
- Create: `scripts/evaluate_harness_cache.py`
- Modify: `scripts/compare_harness_agents.py`
- Test: `tests/test_harness_cache_metrics.py`
- Test: `tests/test_harness_cache_cli.py`

**Interfaces:**
- Produces:
  - `CacheModeMetrics`, `CacheEvaluationReport`, `score_cache_evaluation(warmup_runs, eval_runs, labels)`
  - `scripts.harness_llm_cli.add_llm_arguments(parser)` and `build_chat_client(args)`
  - `scripts.evaluate_harness_cache.main(argv=None) -> int`
- CLI flags: `--fixture`, `--labels`, `--out`, `--analysis-version`, `--cache-store`, `--cache-dsn`, `--cache-namespace`, `--cache-reset`, `--threshold`, `--passes`, `--inner rule|llm`, `--repeat-accepted`, plus the shared LLM/cassette flags.

- [ ] **Step 1: Write failing metrics tests**

```python
def test_cache_metrics_counts_hits_and_cost_saved():
    report = score_cache_evaluation([_accepted_run(cost=0.002, recorded=True)], [_accepted_cache_run()], _labels())
    assert report.cache_eligible == 1
    assert report.exact_hits == 1
    assert report.cache_eligible_hit_rate == 1.0
    assert report.cost_saved_usd == pytest.approx(0.002)


def test_false_reuse_counts_only_reused_runs():
    report = score_cache_evaluation([], [_reused_non_complaint_run()], _labels())
    assert report.false_reuse == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_harness_cache_metrics.py -q`
Expected: FAIL because the metrics module does not exist.

- [ ] **Step 3: Implement metrics**

Count `llm_calls` from trace events with `event == "agent.completed"` and `provider != "local"`. `cache_eligible` is the number of warmup runs with `run.cache.recorded is True`. `false_reuse` counts eval runs with `run.cache.reused is True` and a status not in the label's `acceptable_statuses`.

- [ ] **Step 4: Extract the shared LLM CLI helper**

Move the existing `--model`, price, reasoning, cassette, and replay logic from `compare_harness_agents.py` into `scripts/harness_llm_cli.py`. Keep `compare_harness_agents.py` behavior and tests unchanged.

- [ ] **Step 5: Implement the evaluation CLI**

Build the inner orchestrator from `--inner`. `rule` uses rule agents and needs no API key. `llm` uses LLM agents and the shared chat client. Warm up the cache with pass 1, then evaluate pass 2. `--repeat-accepted N` builds a duplicate stream by repeating accepted labeled posts before the warmup. Write `cache_report.json` and `cache_report.md`.

- [ ] **Step 6: Write the CLI test**

Run the CLI with `--inner rule` on `tests/fixtures/x_posts`, a temporary labels file, `--cache-store memory`, `--passes 2`, and `--out tmp_path`. Assert exit code 0, `cache_eligible >= 1`, `exact_hits >= 1`, `false_reuse == 0`, and both report files exist.

- [ ] **Step 7: Run tests**

Run: `pytest tests/test_harness_cache_metrics.py tests/test_harness_cache_cli.py tests/test_harness_compare.py -q`
Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add services/harness/cache_metrics.py scripts/harness_llm_cli.py scripts/evaluate_harness_cache.py scripts/compare_harness_agents.py tests/test_harness_cache_metrics.py tests/test_harness_cache_cli.py
git commit -m "feat: add cache evaluation and metrics"
```

---

### Task 9: Evaluation Report, Docs, and Final CI

**Files:**
- Create: `docs/harness-x-m6.md`
- Create: `docs/evaluations/2026-11-26-harness-x-m6-cache.md`
- Modify: `docs/testing.md`
- Modify: `TODO.md`
- Test: `tests/test_harness_cache_e2e.py`

**Interfaces:**
- Consumes: all prior tasks.
- Produces: documented cache behavior, measured results, and a green core CI.

- [ ] **Step 1: Add an end-to-end test**

Use an in-memory store, a rule inner orchestrator, one accepted synthetic post, and two passes. Assert the first pass records and the second pass returns `run.cache.match_type == "EXACT"`, `run.cache.reused is True`, and zero LLM cost.

- [ ] **Step 2: Run the test**

Run: `pytest tests/test_harness_cache_e2e.py -q`
Expected: PASS.

- [ ] **Step 3: Run the real evaluation**

Run the CLI on `fixtures/x_posts` with `--inner llm` and `--passes 2` using the local OpenRouter key. This is a live run, expected to cost about $0.004; do not print the key. Save the report.

Do not use the M5 cassette for this two-pass run: the same non-cacheable prompts occur twice, and the M5 cassette stores one response per prompt. The live run records the second pass without cassette exhaustion.

Also run with `--repeat-accepted 3` to measure a duplicate stream. Record both results in the evaluation doc, including cache-eligible and overall hit rates, cost saved, and false reuse.

- [ ] **Step 4: Write the docs**

`docs/harness-x-m6.md` must cover architecture, schema, cache gates, location matching, versioning, CLI usage, metrics, and limits. Update `docs/testing.md` with the new core test count and the integration lane instructions.

- [ ] **Step 5: Run full validation**

```bash
bash scripts/ci.sh
bash scripts/research_ci.sh
git diff --check
```

Expected: core CI green, research lane green with the existing quarantine, no whitespace errors.

- [ ] **Step 6: Update TODO and commit**

Mark all M6 tasks completed with `todo_write`. Commit:

```bash
git add docs/harness-x-m6.md docs/evaluations/2026-11-26-harness-x-m6-cache.md docs/testing.md TODO.md tests/test_harness_cache_e2e.py
git commit -m "docs: report harness semantic cache results"
```

---

## Self-Review

- Spec coverage: migration, store, engine, guards, provenance, metrics, CLI, evaluation, docs, and PII all have tasks.
- Type consistency: `CacheEntry`, `CacheQuery`, `CacheCandidate`, `HarnessAnalysisCache`, `CacheFirstRunner`, and `score_cache_evaluation` are defined before later tasks use them.
- Review Focus covered: generic location tokens in Task 7; rule/LLM disagreement and PII in Task 4; stale versions in Tasks 3–4; non-citizen content in Task 7.
- No implementation step depends on an undefined type or function.
- No acceptance criterion is claimed as “all 18 posts hit”; the plan reports cache-eligible and overall rates separately.
