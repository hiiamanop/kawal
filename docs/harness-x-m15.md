# KAWAL Harness X — M1.5 Live Batch and Evaluation

M1.5 extends the milestone 1 harness with live X search, batch processing, and a
small evaluation report.

## What changed

- `AgentReachXSource` extracts the numeric post id from `.../status/<id>` when
  Agent-Reach search results do not include an explicit `id`.
- `ReplayXSource` accepts a directory of JSON fixtures.
- `HarnessBatchRunner` processes many posts, writes per-case artifacts, and
  produces `summary.json` and `summary.md`.
- `scripts/run_harness_x_batch.py` runs a replay or live batch and can save the
  captured posts as reusable fixtures.

## Batch output

Each batch is written to `runs/batch-<batch_id>/`:

- `summary.json` — machine-readable evaluation summary;
- `summary.md` — human-readable status/category/cost table;
- one directory per case with `run.json`, `case.json`, `trace.jsonl`, and
  `cost.json`.

The summary counts `COMPLETED`/accepted, `NEEDS_REVIEW`, `REJECTED`, `FAILED`,
and `BUDGET_EXCEEDED`, plus category distribution, total cost, and total
latency.

## Commands

### Replay a directory of fixtures

```bash
python scripts/run_harness_x_batch.py \
  --fixture tests/fixtures/x_posts \
  --out runs/batch-replay \
  --batch-id replay-001
```

### Live X search

```bash
PATH="$HOME/.local/bin:$PATH" \
KAWAL_AGENT_REACH_BIN="$HOME/.local/bin/agent-reach" \
python scripts/run_harness_x_batch.py \
  --live-source \
  --query "jalan rusak bandung" \
  --limit 10 \
  --out runs/batch-live \
  --save-fixtures fixtures/x_posts/live-001
```

Live search requires an active X session in Agent-Reach. It is an experimental,
read-only path. Save the results as fixtures so tests and later evaluations do
not depend on the live platform.

### Test

```bash
pytest tests/test_harness_batch.py tests/test_harness_batch_cli.py \
  tests/test_harness_sources.py -q
```

## Current limits

- The agents remain deterministic rule-based baselines.
- The evaluation report measures harness outcomes and cost, not ground-truth
  complaint quality.
- Live X search is not part of the core CI lane.
