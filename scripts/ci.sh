#!/usr/bin/env bash
set -euo pipefail

# Core CI must not depend on local integration credentials or a live model.
unset KAWAL_TEST_DATABASE_URL
unset OPENROUTER_API_KEY
unset KAWAL_OPENROUTER_API_KEY

# Core CI: tests that do not require Postgres, Redpanda, trained artifacts,
# or generated thesis reports.
python -m pytest -q -m "not integration and not research" "$@"
