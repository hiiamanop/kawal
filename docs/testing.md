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

Last local result after the monitor console:

```text
731 passed, 41 skipped, 85 deselected
```

### Monitor console

The monitor console tests use temporary SQLite databases and replay fixtures; they
do not call live X.

```bash
pytest tests/test_monitor_api.py tests/test_monitor_store.py tests/test_monitor_scan.py \
  tests/test_monitor_scoring.py tests/test_monitor_forwarding.py -q
```

Last local result: `16 passed`.

Run the console locally with:

```bash
python scripts/run_monitor.py --port 8088
```

It binds to `127.0.0.1`; live scans require an Agent Reach X session.

## Research and integration lane

```bash
bash scripts/research_ci.sh
```

This runs research and integration tests while deselecting explicit blockers
listed in [`tests/quarantine.txt`](../tests/quarantine.txt). It unsets the
OpenRouter key because these tests must stay offline. Last local result:

```text
78 passed, 7 skipped, 0 quarantined
```

The skips are missing prerequisites: PostgreSQL, generated M3 artifacts,
thesis artifacts, and the pgvector integration database. The generator covers
all 12 categories with 12 families each (144 families; 96/24/24 across
train/dev/test) and passes the split and corpus anti-leak audits.

To run the full lane without quarantine filtering:

```bash
python -m pytest -q -m "research or integration"
```

### Quarantine groups

There are no quarantined tests. `tests/quarantine.txt` stays as a guard list for
real content or model gaps. Do not add core-lane tests to it; core tests must
stay green.

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
