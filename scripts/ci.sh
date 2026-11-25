#!/usr/bin/env bash
set -euo pipefail

# Core CI: tests that do not require Postgres, Redpanda, trained artifacts,
# or generated thesis reports.
python -m pytest -q -m "not integration and not research" "$@"
