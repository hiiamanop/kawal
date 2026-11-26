# KAWAL Harness X — M6 Semantic Cache Design

**Status:** draft for approval
**Date:** 2026-11-26
**Scope:** cache-first routing for the X-only harness, backed by pgvector with an
in-memory fallback for CI.

---

## 1. Intent

M6 reduces LLM cost by reusing a previously accepted analysis when a new X post
is an exact or high-confidence semantic match. Cache reuse must not weaken the
existing trust model: deterministic guards still run, and a cache hit is never
accepted without current evidence and metadata checks.

The research question is:

> Can a versioned semantic cache reduce LLM calls and cost on repeated or
> near-duplicate complaints without increasing false accepts?

## 2. Goals

- Store accepted harness analyses in a pgvector-backed cache.
- Add an in-memory vector store with the same contract for tests and CI.
- Route exact and high-confidence semantic hits away from the LLM.
- Run current deterministic Triage and Verifier guards on every cache hit.
- Reject reuse across namespace, analysis version, embedding model, category,
  source intent, or location mismatch.
- Record cache provenance in the run, trace, batch summary, and evaluation.
- Measure hit rate, LLM calls saved, cost saved, and false reuse.
- Keep the legacy WhatsApp `case_knowledge_bank` cache untouched.

## 3. Non-goals

- Caching rejected or `NEEDS_REVIEW` cases.
- Reusing analysis across tenants, namespaces, model versions, prompt versions,
  policy versions, or embedding models.
- Replacing deterministic source-intent or location guards with vector search.
- Using FAISS, SQLite, or a local-only index as the production store.
- Tuning or retraining the IndoBERT embedder.
- Automatic cache writes from unverified runs.

## 4. Anti-goals

M6 fails if any of these happen:

- A non-complaint or different-location incident is accepted from cache.
- A cache hit produces `ACCEPTED` without current rule Triage and rule Verifier
  agreement.
- A stale model, prompt, policy, or schema version is reused.
- Raw complaint text or PII is stored in the cache table.
- The cache bypasses the deterministic location or source-intent guards.

## 5. Current State

The repository already has a semantic cache for the legacy WhatsApp pipeline:

- `services/intelligence/semantic_cache.py`
- `case_knowledge_bank`
- `FLOAT8[]` embeddings plus a PL/pgSQL cosine function
- duplicate-ticket and causal-reuse semantics

That cache is tied to WhatsApp `cases` and `tenants`, and its table has no
pgvector index. Harness X has different needs: versioned multi-agent analysis,
no ticket side effects, and evaluation provenance. Reusing the legacy table
would couple two different lifecycles and risks breaking the WhatsApp pipeline.

## 6. Approaches Considered

### 6.1 Dedicated harness table + pgvector (chosen)

Create `harness_case_cache` with `vector(768)`, HNSW, versioned metadata, and a
JSONB analysis payload. Keep the legacy cache unchanged. Use an in-memory store
for CI.

Trade-off: one more table and a second cache module, but clean ownership,
no foreign-key coupling, and straightforward versioning.

### 6.2 Extend `case_knowledge_bank`

Add pgvector and analysis columns to the existing table.

Trade-off: fewer tables, but couples harness evaluation to WhatsApp cases,
tenants, and duplicate-ticket semantics. Migration risk is high for little gain.

### 6.3 Local-only vector index

Use NumPy, FAISS, or SQLite without Postgres.

Trade-off: easy to run, but it is not the requested production vector DB and
does not provide the same index and operational path.

## 7. Architecture

```text
SocialPost
  -> CacheFirstRunner
       -> deterministic source-intent guard
       -> rule Triage + rule Extraction + rule Verifier
       -> exact hash lookup
       -> pgvector semantic lookup
       -> reuse guards
            pass -> cached analysis -> new ComplaintCase -> HarnessRun
            fail -> inner HarnessOrchestrator (LLM path)
  -> if inner run is ACCEPTED -> record case in cache
```

The cache is a wrapper around `HarnessOrchestrator`, not a replacement for it.
On a miss or failed guard, the existing orchestrator runs unchanged.

## 8. Cache Store

### 8.1 Table

```sql
CREATE TABLE harness_case_cache (
    cache_key TEXT PRIMARY KEY,
    namespace TEXT NOT NULL,
    analysis_version TEXT NOT NULL,
    embedding_model TEXT NOT NULL,
    text_hash CHAR(64) NOT NULL,
    source_post_id TEXT NOT NULL,
    source_run_id TEXT NOT NULL,
    embedding vector(768) NOT NULL,
    category TEXT NOT NULL,
    rule_category TEXT NOT NULL,
    source_intent TEXT NOT NULL,
    location_quality TEXT NOT NULL,
    location_entities JSONB NOT NULL,
    evidence_spans JSONB NOT NULL,
    analysis_payload JSONB NOT NULL,
    record_guards JSONB NOT NULL,
    trust_decision TEXT NOT NULL,
    status TEXT NOT NULL,
    hit_count INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
    last_hit_at TIMESTAMPTZ
);
```

Indexes:

- unique `(namespace, analysis_version, embedding_model, text_hash)`;
- HNSW `embedding vector_cosine_ops` with `m=16`, `ef_construction=64`;
- lookup index `(namespace, analysis_version, embedding_model, category,
  source_intent)`.

The table stores no full post body. `text_hash` is the normalized SHA-256 hash.
`location_entities` holds normalized location strings; `evidence_spans` holds
short verbatim fragments only when the record gate finds no PII. The JSONB
payload holds validated Triage, Extraction, Verifier, and optional Arbitration
outputs for audit, plus the guard results that allowed recording.

If any PII pattern is found in the case payload, spans, or location entities,
the case is not cached. This keeps the no-PII invariant even when the rule path
accepted a post that would have been blocked from LLM egress.

### 8.2 Store contract

```python
class HarnessCacheStore(Protocol):
    def find_exact(self, key: CacheKey) -> CacheEntry | None: ...
    def find_similar(self, query: CacheQuery) -> list[CacheCandidate]: ...
    def save(self, entry: CacheEntry) -> None: ...
    def increment_hit(self, cache_key: str) -> None: ...
```

Implementations:

- `InMemoryHarnessCacheStore` using NumPy cosine similarity.
- `PgVectorHarnessCacheStore` using psycopg and pgvector operators.

Both use the same Pydantic entry and candidate models.

## 9. Embeddings

The cache uses the existing `TextEmbedder` contract. Add `model_id` to each
embedder so entries cannot be reused across embedding versions:

- deterministic fallback: `deterministic-hash-768-v1`;
- IndoBERT: the resolved artifact name plus a fingerprint of its config,
  tokenizer, and weight metadata and edge bytes, or `indobert-fallback-768`.

The cache rejects an embedder whose `model_id` is still `unknown`.

Semantic reuse is disabled by default when the embedder is the deterministic
fallback. Exact reuse still works. Tests may explicitly enable fallback semantic
reuse to exercise guard logic. Production semantic reuse requires a real
IndoBERT artifact.

## 10. Cache Key and Versioning

A cache entry is valid only when all of these match:

- namespace, default `harness-x`;
- analysis version, configured per evaluation;
- embedding model id;
- category;
- source intent.

The analysis version is a required caller-supplied string. It must change when
the LLM model, prompts, policy, or output schema changes. The cache does not
guess compatibility from timestamps.

The `text_hash` is computed from normalized text: lowercase, collapsed
whitespace, trimmed. Exact lookup uses this hash. `cache_key` is the SHA-256 of
`namespace`, `analysis_version`, `embedding_model`, and `text_hash`, joined with
a stable separator. This makes duplicate writes idempotent.

## 11. Reuse Policy

Only accepted cases are recorded, and only when the deterministic rule agents
agree with the accepted analysis:

- `status == ACCEPTED`;
- `trust_decision == AGREED`;
- `location_quality == SPECIFIC`;
- `source_intent == CITIZEN_COMPLAINT`;
- Triage and Verifier both say `is_complaint=True`;
- Verifier says `evidence_supported=True`;
- current rule Triage category equals the accepted category;
- current rule Verifier category equals the accepted category;
- current rule Verifier says `evidence_supported=True`;
- no PII pattern is present in the payload, spans, or locations.

The rule agreement gate is deliberate. A case where the LLM accepts one
category and the rule agents report another is not cacheable: the cache must not
reuse a decision that the deterministic guards would reject on a later pass.
Such cases remain full-pipeline cases.

On every lookup, the cache-first runner runs current deterministic agents first:

1. `assess_source_intent(post.text)`
2. `RuleTriageAgent`
3. `RuleExtractionAgent`
4. `RuleVerifierAgent` on the current extraction

Reuse requires all of the following on the current post:

- source intent is `CITIZEN_COMPLAINT`;
- rule Triage and rule Verifier both say complaint;
- their categories match each other and the cached category;
- current location quality is `SPECIFIC`;
- current location overlaps the cached location entities;
- current evidence is supported by the current text;
- cached entry is `ACCEPTED` and `AGREED`;
- cache namespace/version/embedding model match.

Location overlap is not raw token intersection. Locations are normalized by
lowercasing, removing punctuation, removing street prefixes such as `jl` and
`jalan`, and dropping generic place words such as `stasiun`, `terminal`, `pasar`,
`kecamatan`, `kelurahan`, `kota`, and `kabupaten`. Reuse is allowed only when the
normalized location is equal, at least two significant tokens overlap, or one
distinctive token of at least four characters overlaps. This rejects
`Stasiun Bandung` vs `Stasiun Cimahi`, while accepting `Jl. Merdeka` vs
`Jalan Merdeka` and `Terusan Buah Batu` vs `Buah Batu`.

Exact reuse additionally requires the cached evidence spans to be present in the
current text. Semantic reuse requires a similarity at or above the configured
threshold, default `0.92`.

If any guard fails, the runner records `CACHE_REJECTED` with a reason code and
delegates to the full orchestrator. The LLM path remains authoritative on a
failed guard.

## 12. Result Construction on a Hit

A cache hit creates a new case for the current post:

- `case_id` is the deterministic id of the current post;
- `post` is the current post;
- `triage` uses the cached category and risk, elevated to the current rule
  Triage risk if the current risk is higher;
- `extraction` is the current rule extraction, not the cached extraction;
- `verification` is the current rule verification, not the cached verification;
- `arbitration` is `None`;
- `location_quality` and `source_intent` are recomputed for the current post;
- `trust_decision` is `AGREED`;
- `status` is `ACCEPTED`;
- reason codes include `CACHE_REUSED`, `CACHE_EXACT` or `CACHE_SEMANTIC`, and
  `CACHE_GUARDS_PASSED`.

This keeps the decision grounded in the current post while using the cache to
skip the LLM. The cached Verifier output is provenance, not current evidence.

## 13. Provenance

Add an optional `cache` field to `HarnessRun`:

```python
class CacheProvenance(BaseModel):
    attempted: bool
    hit: bool
    reused: bool
    match_type: Literal["EXACT", "SEMANTIC", "NONE"]
    similarity: float
    source_case_id: str | None
    recorded: bool
    reason_codes: tuple[str, ...]
```

Add cache fields to `BatchItemSummary`:

- `cache_attempted`
- `cache_hit`
- `cache_reused`
- `cache_match_type`
- `cache_similarity`
- `llm_calls`

The trace records `cache.lookup`, `cache.hit`, `cache.rejected`,
`cache.recorded`, and the guard reason codes.

## 14. Metrics

New cache metrics:

- cache lookups, exact hits, semantic hits, rejected hits, misses;
- overall hit rate and cache-eligible hit rate;
- false reuse count and rate;
- LLM calls saved;
- cost saved in USD;
- latency saved in ms;
- decision accuracy and accepted precision for the cache path.

`cache-eligible` means the post was accepted by the full pipeline and passed the
record gate. Reporting both rates prevents a small cacheable subset from being
presented as a whole-dataset hit rate.

False reuse means a cache-accepted case whose ground-truth label does not accept
`COMPLETED`. The existing label `acceptable_statuses` remains the source of
truth.

## 15. Evaluation

### 15.1 18-post two-pass evaluation

Use the 18 real X fixtures in two passes:

1. First pass warms the cache from accepted full-pipeline results.
2. Second pass sends the same posts again.

Only accepted, rule-consistent posts can be cached. In the current labels, three
posts are accepted; after the rule-agreement gate, the cacheable count may be
lower. The report must show:

- exact hits / cache-eligible posts;
- cache-eligible posts / all 18 posts;
- exact hits / all 18 posts;
- LLM calls on pass 2, including every non-cacheable post;
- false reuse and decision equality for the cacheable posts.

The M5 cassette can serve pass 1, so this evaluation needs no new API spend. It
proves correctness and false-reuse safety, not large savings.

### 15.2 Duplicate stream

Add a separate synthetic `CacheScenario` fixture: repeat the accepted complaint
texts several times, interleaved with non-complaints. This measures realistic
repeat-report savings without pretending the 18-post dataset is duplicate-heavy.

Expected: the first occurrence misses, later exact copies hit, non-complaints do
not hit, and the final decisions match the labels. Hit rate is reported over the
accepted stream, not over the 18-post ground-truth set.

### 15.3 Semantic near-duplicates

Use a small curated `CacheScenario` fixture with a seed complaint and
paraphrased queries at the same location. Each scenario declares the expected
hit and expected decision. This is a synthetic extension, not part of the
18-post X ground-truth dataset.

The semantic test must run with a real embedder for production numbers. The
deterministic fallback may be used only to exercise the routing and guard tests;
semantic reuse is disabled by default with that fallback.

### 15.4 False-reuse suite

Seed the cache with accepted complaints, then run all 13 non-complaint X
fixtures and a set of different-location complaints, including the
`Stasiun Bandung` vs `Stasiun Cimahi` case. No cache path may accept them.

## 16. CLI and Operations

Add `scripts/evaluate_harness_cache.py`:

```bash
python scripts/evaluate_harness_cache.py \
  --fixture fixtures/x_posts \
  --labels fixtures/x_labels.json \
  --analysis-version harness-x-m6-v1 \
  --cache-namespace harness-x-eval \
  --cache-reset \
  --threshold 0.92 \
  --passes 2 \
  --out runs/m6
```

It runs baseline and cache-first paths, scores both with ground-truth labels,
and writes `cache_report.json` and `cache_report.md`.

For Postgres:

```bash
--cache-store postgres \
--cache-dsn postgresql://postgres:postgres@127.0.0.1:54322/postgres
```

`--cache-reset` deletes only rows for the selected namespace and analysis
version before pass 1. Without it, a Postgres run is not reproducible. The
in-memory store is empty per process. The default for tests is `memory`. The
migration is applied by the existing migration flow, not by the evaluation
script.

The compose Postgres image changes from `postgres:16-alpine` to
`pgvector/pgvector:pg16`; the existing `infra/initdb/01-extensions.sql` already
expects the vector extension. Add the migration to both
`infra/migrations/014_m6_harness_semantic_cache.sql` and a timestamped
`supabase/migrations` file. The integration target is the compose service on
port `54322`.

## 17. Testing

Core tests must run offline with no Postgres:

- in-memory exact and semantic lookup;
- namespace, version, category, and embedding-model isolation;
- threshold behaviour;
- accepted-only recording;
- exact reuse bypasses the inner orchestrator;
- semantic reuse bypasses the inner orchestrator only when all guards pass;
- location mismatch, source-intent mismatch, category mismatch, stale version,
  and evidence mismatch all fall back;
- generic place tokens do not create false location overlap;
- LLM/rule category disagreement prevents recording;
- PII in a case prevents recording;
- namespace reset removes only the selected namespace/version;
- cache provenance and batch fields;
- metrics for overall hit rate, cache-eligible hit rate, cost saved, and false
  reuse;
- both migration trees contain `vector(768)`, HNSW, and the expected indexes;
- no full post body is serialized into a cache entry.

Postgres integration tests may be marked `integration` and are not part of the
core gate.

## 18. Acceptance Criteria

1. Migration 014 creates `harness_case_cache` with pgvector and HNSW in both
   migration trees.
2. Compose uses a pgvector-enabled Postgres image, and the integration lane can
   apply migration 014.
3. In-memory and pgvector stores implement the same contract.
4. Exact and high-confidence semantic hits can skip the inner orchestrator.
5. Every hit runs current source-intent, Triage, Extraction, and Verifier guards.
6. Any guard failure falls back to the full pipeline.
7. Only accepted, rule-consistent, PII-free cases are recorded.
8. Cache provenance is visible in `HarnessRun`, trace, batch summary, and report.
9. The 18-post two-pass evaluation reports cache-eligible and overall hit rates,
   shows no false reuse, and does not claim whole-dataset savings.
10. The synthetic duplicate stream shows measurable cost savings.
11. Non-complaint and different-location suites show zero cache false accepts.
12. Core CI is green; research/integration tests remain separated.
13. Documentation covers the schema, guard rules, metrics, and operational
    limits.

## 19. Risks

- **False reuse from different incidents at similar locations.** Mitigated by
  accepted-only caching, category/source/location/evidence guards, a high
  threshold, and a false-reuse suite.
- **Rule/LLM category disagreement.** Mitigated by refusing to record any case
  where the rule agents do not agree with the accepted category.
- **Weak fallback embeddings.** Mitigated by disabling semantic reuse when the
  deterministic fallback is active.
- **Stale analysis after model changes.** Mitigated by required analysis and
  embedding versions.
- **pgvector index drift.** Mitigated by migration tests in both trees and a
  query plan check in the integration lane.
- **PII in cache.** Mitigated by storing text hashes and short evidence spans
  only, scanning the record payload for PII, and refusing to cache on any match.
- **Dataset too small for threshold calibration.** The initial threshold is a
  conservative default; semantic evaluation is marked as synthetic until a
  larger real near-duplicate set exists.
- **Non-reproducible persistent evaluation.** Mitigated by namespace-scoped
  reset and versioned cache keys.
