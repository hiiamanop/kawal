#!/usr/bin/env bash
set -euo pipefail

# Research and integration tests are not part of the PR gate. This script runs
# the green remainder and deselects the explicit blockers in tests/quarantine.txt.
unset OPENROUTER_API_KEY
unset KAWAL_OPENROUTER_API_KEY

deselect=()
while IFS= read -r line; do
  [[ -z "${line}" || "${line}" == \#* ]] && continue
  deselect+=(--deselect "${line}")
done < tests/quarantine.txt

python -m pytest -q -m "research or integration" \
  ${deselect[@]+"${deselect[@]}"} "$@"
