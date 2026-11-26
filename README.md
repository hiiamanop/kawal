# KAWAL

KAWAL is a research prototype for processing Indonesian citizen complaints with
auditable, multi-agent decisions. The current focus is an X-only harness that
compares rule-based and LLM-backed agents on recorded public posts.

The repository still contains an earlier WhatsApp pipeline and its dashboard,
simulator, outbox, and policy components. That pipeline is experimental and is
not the current evaluation target.

## Status

| Area | State |
|---|---|
| Core CI | `686 passed, 42 skipped, 85 deselected` |
| Research/integration lane | Green with 4 explicit content/model gaps quarantined |
| Harness X | M1 through M6 implemented |
| WhatsApp pipeline | Experimental, still present, not the current focus |
| Production readiness | Not production-ready |

The harness is useful for reproducible evaluation, not for live municipal
dispatch. It does not create real government tickets, send replies to citizens,
or replace human review.

## What the X harness does

```text
recorded X post
  -> ReplayXSource
  -> deterministic Triage / Extraction / Verifier
  -> source-intent and location guards
  -> trust decision
       ACCEPTED only when agents agree and guards pass
       NEEDS_REVIEW on disagreement or missing evidence
       REJECTED for non-actionable or non-citizen content
  -> FileRunSink trace, case, cost, and cache provenance
```

The LLM path follows the same orchestrator contract:

```text
LLM Triage -> rule or LLM Extraction -> LLM Verifier -> rule or LLM Arbiter
```

The trust rules are deliberately conservative:

- Triage and Verifier disagreement always becomes `NEEDS_REVIEW`.
- The Arbiter explains a conflict; it never authorizes `ACCEPTED`.
- A complaint must have a `SPECIFIC` location to be accepted.
- News, opinions, questions, and announcements are not actionable complaints.
- LLM output is validated against Pydantic schemas before it reaches a case.
- LLM calls retry once on empty or invalid output and record the attempt count.
- Prompt injection and PII are blocked before remote egress.

## Semantic cache (M6)

M6 adds a versioned cache for accepted analyses:

- exact hash lookup;
- pgvector HNSW lookup for high-confidence semantic matches;
- in-memory store for offline tests;
- source, location, category, evidence, and version guards on every reuse;
- cache provenance in runs and batch summaries;
- cost, hit-rate, and false-reuse metrics.

Only accepted, rule-consistent, PII-free cases are cached. A cache hit still runs
current deterministic Triage, Extraction, and Verifier guards. If any guard
fails, the full pipeline runs.

The live 18-post evaluation produced 2 exact hits, 0 false reuse, and reduced LLM
calls from 36 to 32. A duplicate stream that keeps the 15 non-complaints produced
6 exact hits out of 24 posts and reduced LLM calls from 48 to 36. A five-scenario
sweep with base IndoBERT found threshold 0.85 gives perfect precision and recall;
the default 0.92 missed one paraphrase. See
[docs/evaluations/2026-11-26-harness-x-m6-cache.md](docs/evaluations/2026-11-26-harness-x-m6-cache.md).

## Repository layout

```text
kawal/
├── contracts/              Pydantic request, result, and case schemas
├── docs/                   design notes, milestone docs, evaluations
├── fixtures/               recorded X posts and ground-truth labels
├── infra/migrations/       PostgreSQL migrations, including pgvector
├── scripts/                harness runners, comparison, scoring, legacy workers
├── services/harness/       X harness agents, orchestrator, cache, sinks
├── services/intelligence/  embedders and legacy semantic cache
├── services/core/          legacy WhatsApp decision pipeline
├── services/intake/        legacy WhatsApp intake and assembly
├── services/dashboard/     legacy inspector dashboard
├── tests/                  core, research, and integration tests
├── docker-compose.yml      PostgreSQL/pgvector, Redpanda, OPA, simulator
└── requirements*.txt       runtime, dev, ML, and simulator dependencies
```

## Quickstart

### Prerequisites

- Linux, macOS, or WSL2
- Python 3.11 or newer
- Docker or Podman with Compose for the optional infrastructure services

### Install

```bash
git clone https://github.com/hiiamanop/kawal.git
cd kawal
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
```

Install `requirements-ml.txt` only if you need the local model stack:

```bash
pip install -r requirements-ml.txt
```

### Configuration

Core tests run without a `.env` file or an API key. The script unsets the
OpenRouter key so a local secret cannot turn core CI into a live-model run.

For live LLM evaluation:

```bash
cp .env.example .env
```

Set `OPENROUTER_API_KEY` in `.env`. Never commit `.env`. The harness also accepts
`KAWAL_OPENROUTER_MODEL`; the recorded evaluations used `tencent/hy4-flash` with
`--no-reasoning`.

## Run the X harness

Rule-only single post:

```bash
python scripts/run_harness_x.py \
  --fixture fixtures/x_posts/x_mgt_radio_2026159892885832085.json \
  --out runs/demo
```

Batch over recorded posts:

```bash
python scripts/run_harness_x_batch.py \
  --fixture fixtures/x_posts \
  --limit 18 \
  --out runs/batch
```

Rule versus LLM comparison:

```bash
python scripts/compare_harness_agents.py \
  --fixture fixtures/x_posts \
  --limit 18 \
  --model tencent/hy4-flash \
  --input-price-per-million 0.035 \
  --output-price-per-million 0.1405 \
  --out runs/compare \
  --no-reasoning
```

Score a comparison against ground-truth labels:

```bash
python scripts/score_harness_comparison.py \
  --comparison runs/compare/comparison-<id>/comparison.json \
  --labels fixtures/x_labels.json
```

Record or replay an LLM cassette:

```bash
# record
python scripts/compare_harness_agents.py ... --cassette runs/cassette.json

# replay without an API key
python scripts/compare_harness_agents.py ... \
  --cassette runs/cassette.json \
  --replay-cassette
```

Evaluate the semantic cache:

```bash
python scripts/evaluate_harness_cache.py \
  --fixture fixtures/x_posts \
  --labels fixtures/x_labels.json \
  --out runs/m6 \
  --analysis-version harness-x-m6-v1 \
  --cache-store memory \
  --cache-namespace m6-eval \
  --cache-reset \
  --passes 2 \
  --inner rule
```

Use `--inner llm` for the live LLM path. Use `--cache-store postgres` with
`--cache-dsn` for pgvector. The compose Postgres image is
`docker.io/pgvector/pgvector:pg16`; run migration
`infra/migrations/014_m6_harness_semantic_cache.sql` before a Postgres evaluation.

Live X collection through Agent-Reach is optional:

```bash
python scripts/run_harness_x_batch.py \
  --live-source \
  --query "jalan rusak bandung" \
  --limit 5
```

Recorded fixtures are the reproducible path. Live search depends on the local
Agent-Reach session and the upstream platform.

## Test

Core, offline lane:

```bash
bash scripts/ci.sh
```

Research and integration lane with the documented quarantine:

```bash
bash scripts/research_ci.sh
```

The full, unquarantined research lane currently has 4 known failures:

```bash
pytest -q -m "research or integration"
```

The blockers are listed in [tests/quarantine.txt](tests/quarantine.txt). They
cover extended-category family diversity and the legacy semantic-cache model
threshold. Missing Postgres, M3, and thesis artifacts
now produce skips instead of failures.

## Legacy WhatsApp pipeline

The worker scripts, compose services, and dashboard for the earlier pipeline are
still present for local experiments:

```bash
docker compose up -d postgres redpanda opa ticket-simulator
python scripts/run_all_workers.py
python scripts/run_dashboard.py --port 8088
```

This path is not covered by the core CI gate and has not been verified as a
working end-to-end service in this checkout. Do not treat it as production
software. The current README makes no claim about live government integration or
verified throughput for it.

## Documentation

- [Harness X M6](docs/harness-x-m6.md)
- [Testing lanes and quarantine](docs/testing.md)
- [M6 design](docs/superpowers/specs/2026-11-26-harness-x-m6-semantic-cache-design.md)
- [M6 implementation plan](docs/superpowers/plans/2026-11-26-harness-x-m6-semantic-cache.md)
- [Security policy](docs/SECURITY.md)
- [Evaluation artifacts](docs/evaluations/)

## License

No license file is tracked in this checkout. Add a license before
redistributing the code outside this repository.
