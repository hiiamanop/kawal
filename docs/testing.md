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

Last local result after M5.2–M5.5:

```text
611 passed, 43 skipped, 83 deselected
```

## Research and integration lane

```bash
bash scripts/research_ci.sh
```

This runs research and integration tests while deselecting the explicit
blockers in [`tests/quarantine.txt`](../tests/quarantine.txt). It unsets the
OpenRouter key because these tests must stay offline. Last local result:

```text
52 passed, 2 skipped, 29 quarantined
```

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

## Markers

- `integration` — needs PostgreSQL, Redpanda, or another live dependency.
- `research` — needs M3 artifacts, a trained IndoBERT model, or generated
  thesis reports.

When adding a test that needs one of those dependencies, mark it explicitly
instead of letting it fail in the core CI lane. If it is blocked by a known
missing artifact or defect, add it to `tests/quarantine.txt` with a reason.
