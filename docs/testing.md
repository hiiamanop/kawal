# Testing

The repository has two test lanes.

## Core CI

```bash
bash scripts/ci.sh
```

This runs tests that do not require PostgreSQL, Redpanda, trained model
artifacts, or generated thesis reports. GitHub Actions uses the same script
for the `test` job.

## Research and integration suite

```bash
pytest -q -m "research or integration"
```

This lane requires local infrastructure and/or generated research artifacts.
It is currently not green in a clean checkout and is intentionally excluded
from the PR gate until those artifacts and baseline failures are addressed.

## Markers

- `integration` — needs PostgreSQL, Redpanda, or another live dependency.
- `research` — needs M3 artifacts, a trained IndoBERT model, or generated
  thesis reports.

When adding a test that needs one of those dependencies, mark it explicitly
instead of letting it fail in the core CI lane.
