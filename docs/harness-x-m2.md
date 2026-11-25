# KAWAL Harness X — M2 Agent-Agnostic LLM

M2 compares the deterministic rule agents with LLM-backed Triage and Verifier
agents on the same X fixtures. The orchestrator, source adapter, sink, and trust
rules do not change.

## What is compared

| Role | Rule path | LLM path |
|---|---|---|
| Triage | `RuleTriageAgent` | `LLMTriageAgent` |
| Extraction | `RuleExtractionAgent` | `RuleExtractionAgent` |
| Verifier | `RuleVerifierAgent` | `LLMVerifierAgent` |
| Arbiter | `RuleArbiterAgent` | `RuleArbiterAgent` |

The Verifier receives only the source post and extraction output. It never sees
the Triage output. `ACCEPTED` still requires the Triage and Verifier decisions to
agree; the Arbiter remains explanatory only.

M3 adds optional `--llm-extraction` and `--llm-arbiter` flags so the LLM path can
use `LLMExtractionAgent` and `LLMArbiterAgent` as well. See
`docs/evaluations/2026-11-25-harness-x-m3-aggregate.md` for the six-post live
result.

## Guardrails

- A simple PII check blocks remote egress for 16-digit NIK values, Indonesian
  phone numbers, and email addresses. The LLM agent returns `UNAVAILABLE` with
  `PII_EGRESS_BLOCKED`; the rule path remains available for comparison.
- LLM output must parse as JSON and validate against `TriageOutput` or
  `VerificationOutput`. Invalid output becomes `INVALID`.
- Budget is checked before the call using configured max tokens and prices.
- API keys are read from the environment and never written to reports or logs.
- Raw post text is not written to application logs.

## Running the comparison

The live path needs an OpenRouter key:

```bash
export OPENROUTER_API_KEY="..."
python scripts/compare_harness_agents.py \
  --fixture fixtures/x_posts \
  --limit 10 \
  --model google/gemini-2.5-flash \
  --input-price-per-million 0.30 \
  --output-price-per-million 2.50 \
  --out runs/m2 \
  --comparison-id m2-001
```

For reasoning models that consume the output budget without returning content
(for example `tencent/hy4-flash`), add `--no-reasoning`.

Prices are used for the pre-call budget estimate and for cost calculation when
the provider does not return a cost. Use the current OpenRouter prices for the
model you choose.

The command writes:

- `runs/m2/comparison-m2-001/comparison.json`
- `runs/m2/comparison-m2-001/comparison.md`
- `runs/m2/comparison-m2-001/rule/...`
- `runs/m2/comparison-m2-001/llm/...`

The report includes agreement rate, rule/LLM status counts, total cost, total
latency, and per-post decisions.

## Ground-truth scoring

The comparison report can be scored against `fixtures/x_labels.json`:

```bash
python scripts/score_harness_comparison.py \
  --comparison runs/m2/comparison-m2-001/comparison.json \
  --labels fixtures/x_labels.json
```

This writes `metrics.json` and `metrics.md` next to the comparison report. The
metrics cover complaint precision/recall, category accuracy, decision accuracy,
location accuracy, false accepts/rejects, review rate, and cost per correct
decision.

A complaint can only be accepted when its location is `SPECIFIC`; `MISSING` and
`VAGUE` locations force `NEEDS_REVIEW` on both paths.

Before the final decision, source text is classified as `CITIZEN_COMPLAINT`,
`NEWS_REPORT`, `OPINION`, `QUESTION`, `ANNOUNCEMENT`, or `UNKNOWN`. Clear
non-citizen sources force `REJECTED` when Triage says the post is a complaint;
`UNKNOWN` continues through the normal pipeline. See
`docs/evaluations/2026-11-25-harness-x-m5-actionability.md`.

## Offline tests

Tests use a fake chat client and do not need a key or network:

```bash
pytest tests/test_harness_llm.py tests/test_harness_compare.py -q
```

The tests cover valid output, invalid JSON, PII blocking, verifier prompt
independence, comparison agreement, comparison disagreement, and the missing-key
CLI path.

## Current limits

- Extraction and Arbiter remain rule-based in M2.
- A live LLM comparison requires credentials and is not part of core CI.
- Cost estimation depends on the prices passed on the command line when the
  provider does not report a cost.
- Prompt quality is not optimized in this milestone; the purpose is to prove the
  agent-agnostic contract and measurement path.
