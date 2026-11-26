# Harness X — M6 Semantic Cache

M6 adds a versioned semantic cache to the X-only harness. It can skip the LLM
for exact or high-confidence repeated complaints while keeping the existing
deterministic trust guards authoritative.

The legacy WhatsApp cache (`case_knowledge_bank`) is unchanged.

## Why

The 18-post X evaluation showed that most cost comes from LLM Triage and
Verifier calls. Repeated reports of the same incident are common in city
complaint streams. Cache-first routing avoids paying for the same analysis
again when the new post is safely equivalent.

The cache is not an authority. It is a prior that must pass current guards.

## Architecture

```text
SocialPost
  -> CacheFirstRunner
       -> source-intent guard
       -> rule Triage / Extraction / Verifier
       -> exact hash lookup
       -> pgvector semantic lookup
       -> reuse guards
            allowed -> new case from cached triage + current evidence
            rejected -> inner HarnessOrchestrator
  -> accepted inner result -> record in cache
```

The inner `HarnessOrchestrator` is unchanged. `CacheFirstRunner` wraps it and
implements the same `run(post, run_id=None) -> HarnessRun` shape.

## Storage

Production store: `harness_case_cache` in PostgreSQL with pgvector.

The migration is `infra/migrations/014_m6_harness_semantic_cache.sql`, mirrored
to `supabase/migrations/20261126000000_m6_harness_semantic_cache.sql`.
Compose uses `pgvector/pgvector:pg16`.

The table stores:

- cache key, namespace, analysis version, embedding model;
- normalized text hash, source post id, source run id;
- `vector(768)` embedding;
- category, rule category, source intent, location quality;
- normalized location entities and short evidence spans;
- validated analysis payload;
- record guard results;
- hit count and timestamps.

It does **not** store the full complaint body.

Indexes:

- unique `(namespace, analysis_version, embedding_model, text_hash)`;
- HNSW `embedding vector_cosine_ops` with `m=16`, `ef_construction=64`;
- metadata lookup index for namespace, version, model, category, and source
  intent.

CI uses `InMemoryHarnessCacheStore`. Production uses
`PgVectorHarnessCacheStore`.

## What gets cached

Only cases that pass all record gates:

- `status=ACCEPTED`;
- `trust_decision=AGREED`;
- `location_quality=SPECIFIC`;
- `source_intent=CITIZEN_COMPLAINT`;
- rule Triage and Verifier both say complaint;
- rule Triage and Verifier categories equal the accepted category;
- Verifier says evidence is supported;
- at least one location entity exists;
- no PII in the case payload or location entities;
- no stored free-text field contains the full complaint body;
- no stored free-text field exceeds 200 characters.

A case where the LLM accepted one category and the rule agents reported another
is deliberately not cached.

## Reuse guards

Every lookup runs current deterministic agents first:

1. source-intent assessment;
2. rule Triage;
3. rule Extraction;
4. rule Verifier on the current extraction.

A hit is reusable only when:

- current source intent is a citizen complaint;
- no strong news, opinion, question, or announcement cue is present, even when
  the post also contains citizen words such as `lapor` or `tolong`;
- both rule agents say complaint;
- rule categories equal the cached category;
- current location is `SPECIFIC`;
- current location overlaps the cached location;
- current evidence is supported;
- exact hits contain every cached evidence span in the current text;
- semantic hits have current rule evidence spans;
- namespace, analysis version, and embedding model match.

If any guard fails, the runner records `CACHE_REJECTED` and runs the full
orchestrator. The LLM path remains authoritative on a failed guard.

## Location matching

Locations are lowercased, stripped of punctuation, and stripped of prefixes such
as `jl` and `jalan`. Generic place words such as `stasiun`, `terminal`, `pasar`,
`kecamatan`, `kelurahan`, `kota`, and `kabupaten` are removed.

Reuse requires exact normalized equality, two significant token overlaps, or
one distinctive token of at least four characters. This rejects
`Stasiun Bandung` vs `Stasiun Cimahi` while accepting
`Jl. Merdeka` vs `Jalan Merdeka`.

## Versioning

A cache entry is valid only for the same:

- namespace (`harness-x` or an evaluation namespace);
- analysis version;
- embedding model id.

The analysis version must change when the model, prompt, policy, or result schema
changes. The cache does not infer compatibility from timestamps.

The deterministic fallback embedder disables semantic reuse by default. Exact
reuse still works. Production semantic reuse requires an IndoBERT artifact.

## Provenance

`HarnessRun.cache` contains:

- whether lookup was attempted;
- hit/reused flags;
- exact or semantic match;
- similarity;
- source case id;
- whether the run was recorded;
- reason codes.

Batch summaries expose the same fields plus `llm_calls`. Trace events include
`cache.lookup`, `cache.hit`, `cache.rejected`, and `cache.recorded`.

## CLI

```bash
python scripts/evaluate_harness_cache.py \
  --fixture fixtures/x_posts \
  --labels fixtures/x_labels.json \
  --out runs/m6 \
  --analysis-version harness-x-m6-v1 \
  --cache-store memory \
  --cache-namespace m6-eval \
  --cache-reset \
  --threshold 0.92 \
  --passes 2 \
  --inner llm
```

`--inner rule` runs offline and needs no API key. `--inner llm` supports the
shared cassette flags from `scripts/harness_llm_cli.py`.

For Postgres:

```bash
--cache-store postgres \
--cache-dsn postgresql://postgres:postgres@127.0.0.1:54322/postgres
```

`--cache-reset` deletes only the selected namespace and analysis version.

## Metrics

The report includes:

- cache-eligible posts, exact hits, semantic hits, rejected hits, misses;
- overall hit rate and cache-eligible hit rate;
- false reuse and false reuse rate;
- baseline versus cached decision accuracy;
- LLM calls saved;
- cost and latency saved.

Baseline is a separate no-cache pass. Warmup is not used as the cost baseline.
The saved cost and latency compare baseline with the evaluation pass; warmup
cost is reported separately by the run artifacts and is not folded into the
saved value.

## Evaluation

See
`docs/evaluations/2026-11-26-harness-x-m6-cache.md`.

The live 18-post run produced:

- 2 cache-eligible posts;
- 2 exact hits;
- 0 false reuse;
- 36 to 32 LLM calls;
- $0.000536 to $0.000475 cost.

The duplicate stream, which keeps all 15 non-complaints and repeats the two
cacheable complaints three times, produced:

- 6 exact hits out of 24 posts;
- 0 false reuse;
- 48 to 36 LLM calls;
- $0.000738 to $0.000514 cost.

The 18-post number is intentionally small: only accepted, rule-consistent cases
are cacheable. The duplicate stream measures the savings when repeats actually
exist while still exercising non-complaint posts.

## Limits

- Live evaluation used exact matching because the deterministic fallback
  disables semantic matching. Semantic routing is covered by tests with the
  fallback explicitly enabled.
- Semantic thresholds are not calibrated against a large real near-duplicate
  set.
- Postgres integration is marked `integration` and is not part of core CI.
- Cache entries are treated as immutable during lookup. Re-recording the same
  cache key upserts a fresh analysis; stale entries are invalidated by version.
- The cache does not replace deterministic location or source-intent guards.
- The table stores short evidence spans and embeddings. Treat it as sensitive
  local data.
