# Testing

The repository has two test lanes.

## Core CI

```bash
bash scripts/ci.sh
```

This runs tests that do not require PostgreSQL, Redpanda, trained model
artifacts, or generated thesis reports. The script also unsets
`KAWAL_TEST_DATABASE_URL`, `OPENROUTER_API_KEY`, and
`KAWAL_OPENROUTER_API_KEY` so a local `.env` cannot accidentally turn the core
lane into an integration or live-model run. GitHub Actions uses the same script
for the `test` job.

Last local result after M6:

```text
685 passed, 42 skipped, 85 deselected
```

## Research and integration lane

```bash
bash scripts/research_ci.sh
```

This runs research and integration tests while deselecting the explicit
blockers in [`tests/quarantine.txt`](../tests/quarantine.txt). It unsets the
OpenRouter key because these tests must stay offline. Last local result:

```text
52 passed, 4 skipped, 29 quarantined
```

The third skip is the pgvector integration test when no test database is
configured.

To see the full, unquarantined lane and its current failures:

```bash
python -m pytest -q -m "research or integration"
```

### Quarantine groups

| Group | Tests | Blocker |
|---|---:|---|
| Dashboard / concurrency | 2 | PostgreSQL is not running on `127.0.0.1:5432` or `:54322` |
| M3 dataset export | 16 | Missing `artifacts/m3_synthetic`; generator also fails split anti-leak and category-coverage checks |
| M3 dataset generator | 4 | Real generator defects: category coverage, family diversity, lexicon anchors, cross-seed leakage |
| M3 independent holdout | 4 | Missing `artifacts/m3_independent_holdout`; coverage and contamination checks also fail |
| Semantic cache | 2 | IndoBERT artifact is missing, so the deterministic fallback cannot meet real-model thresholds |
| Thesis tables | 1 | Missing `artifacts/thesis_evaluation_report.json` |

The quarantine is not a fix. It keeps the research lane honest and runnable
while the blockers stay visible. Remove a line from `tests/quarantine.txt` only
after that test passes in a clean checkout and its blocker is resolved. Do not
add core-lane tests to the quarantine; core tests must stay green.

## M6 cache integration

The pgvector path is covered by `tests/test_harness_cache_pgvector_integration.py`
and marked `integration`. It skips unless `KAWAL_TEST_DATABASE_URL` is set.

```bash
podman-compose up -d postgres
podman exec -i kawal-postgres psql -U postgres -d postgres -v ON_ERROR_STOP=1 \
  -f - < infra/migrations/014_m6_harness_semantic_cache.sql
KAWAL_TEST_DATABASE_URL=postgresql://postgres:postgres@127.0.0.1:54322/postgres \
  pytest -q tests/test_harness_cache_pgvector_integration.py
podman-compose down
```

The integration test also checks that the planner selects
`ix_harness_cache_hnsw` when sequential scans are disabled. The compose
Postgres image is `docker.io/pgvector/pgvector:pg16`; the legacy
`postgres:16-alpine` image does not provide the `vector` extension.

A local `--inner rule` smoke run against Postgres produced 3 cache-eligible
posts, 3 exact hits, and 0 false reuse on the 18-post fixture set.

## Markers

- `integration` — needs PostgreSQL, Redpanda, or another live dependency.
- `research` — needs M3 artifacts, a trained IndoBERT model, or generated
  thesis reports.

When adding a test that needs one of those dependencies, mark it explicitly
instead of letting it fail in the core CI lane. If it is blocked by a known
missing artifact or defect, add it to `tests/quarantine.txt` with a reason.
