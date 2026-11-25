# KAWAL Harness X — Milestone 1

This document describes the runnable X-only harness milestone. The design is in
`docs/superpowers/specs/2026-11-25-kawal-harness-x-design.md`; the implementation
plan is in `docs/superpowers/plans/2026-11-25-kawal-harness-x-m1.md`.

## What it does

It takes one public X post, normalizes it into a `SocialPost`, runs a role-based
multi-agent pipeline, and writes one auditable case:

```text
X fixture or Agent-Reach search
  -> ReplayXSource or AgentReachXSource
  -> TriageAgent
  -> ExtractionAgent
  -> VerifierAgent (independent; does not see TriageOutput)
  -> optional ArbiterAgent (explanation only)
  -> CaseWriter / HarnessOrchestrator
  -> run.json + case.json + trace.jsonl + cost.json
```

The milestone intentionally excludes WhatsApp, Threads, real government ticket
systems, citizen replies, multiple categories, and multiple cities.

## Trust rules

- `ACCEPTED` requires TriageAgent and VerifierAgent to agree on `is_complaint` and
  `category`.
- A disagreement always becomes `NEEDS_REVIEW`. The Arbiter may explain the
  conflict, but it cannot authorize `ACCEPTED`.
- A missing location or unsupported evidence also becomes `NEEDS_REVIEW`.
- The Verifier receives only the source post and extraction output. It never sees
  the Triage output.
- Every agent call records model, provider, cost, cost source, latency, and prompt
  version.

## Quick start

Use a virtual environment with the repository requirements. The harness itself
uses only Pydantic and the standard library.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
```

Run the deterministic replay:

```bash
python scripts/run_harness_x.py \
  --fixture tests/fixtures/x_posts/synthetic_road_bandung_001.json \
  --out runs/synthetic-001 \
  --run-id synthetic-001
```

The command prints a JSON summary and writes:

- `runs/synthetic-001/run.json`
- `runs/synthetic-001/case.json`
- `runs/synthetic-001/trace.jsonl`
- `runs/synthetic-001/cost.json`

Set a budget:

```bash
python scripts/run_harness_x.py \
  --fixture tests/fixtures/x_posts/synthetic_road_bandung_001.json \
  --out runs/budget-test \
  --max-cost-usd 0.001
```

## Fixture format

A normalized fixture has one `post` object:

```json
{
  "post": {
    "post_id": "synthetic-road-bandung-001",
    "platform": "x",
    "url": "https://x.com/example/status/1",
    "author_handle": "warga_bandung",
    "text": "Min, jalan di Jl. Kopo Bandung rusak parah, ada lubang besar.",
    "media_urls": [],
    "posted_at": "2026-11-20T08:00:00+00:00",
    "collected_at": "2026-11-25T00:00:00+00:00"
  }
}
```

`ReplayXSource` also accepts an Agent-Reach-shaped payload with an `items` list.

## Live Agent-Reach source

Agent-Reach is an external, read-only CLI. Install it using the official
instructions from the upstream repository; do not install the unrelated PyPI
package named `agent-reach`.

The X channel is experimental and requires a human login for search:

```bash
agent-reach channels
agent-reach twitter search "jalan rusak bandung" --limit 5
```

If search returns `NEED_LOGIN`, a public post can still be captured through the
read-only web backend without a session:

```bash
agent-reach read "https://x.com/<account>/status/<id>"
```

The committed milestone fixture was captured with this public read path. Live
search remains a separate smoke test until an X session is available.

Then run the harness directly against the live source:

```bash
python scripts/run_harness_x.py \
  --live-source \
  --query "jalan rusak bandung" \
  --limit 5 \
  --out runs/live-001
```

The live path is a smoke test. Automated tests use `ReplayXSource` and do not
require X credentials.

## Milestone fixture requirement

The committed milestone fixture must be one real X post captured through
Agent-Reach. The synthetic fixture in `tests/fixtures/x_posts/` exists only for
unit and E2E tests. Capture the real post, save it under
`fixtures/x_posts/<post-id>.json`, and run:

```bash
python scripts/run_harness_x.py \
  --fixture fixtures/x_posts/<post-id>.json \
  --out runs/real-001
```

## Current limitations

- Agents are deterministic rule-based baselines. They are a harness baseline,
  not a claim about LLM quality.
- Image URLs are stored but not analyzed.
- Location extraction is text-only; geocoding is not used.
- The output is a case artifact, not a government ticket.
- The live Agent-Reach path depends on login state and platform availability.

## Tests

```bash
pytest \
  tests/test_harness_contracts.py \
  tests/test_harness_sources.py \
  tests/test_harness_agents.py \
  tests/test_harness_orchestrator.py \
  tests/test_harness_e2e.py -v
```

## Extending the harness

Implement the `Agent` protocol from `services/harness/ports.py`:

```python
class Agent(Protocol):
    agent_id: str
    role: str
    def estimate_cost(self, task: AgentTask) -> float: ...
    def run(self, task: AgentTask) -> AgentResult: ...
```

The orchestrator does not import a provider SDK. A model-backed agent can be
added without changing the source adapters, sink, or trust rules.
