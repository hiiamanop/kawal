# Harness X — M5.2–M5.5 Hardening (2026-11-26)

This evaluation covers four follow-up milestones:

- **M5.2** — LLM retry and output reliability.
- **M5.3** — prompt-injection and PII hardening.
- **M5.4** — reproducible LLM cassettes.
- **M5.5** — research-test quarantine and documentation.

## M5.2 — LLM reliability

Each LLM agent now has two attempts by default. It retries:

- empty or invalid provider responses,
- transient transport failures,
- invalid JSON,
- invalid schema.

A retry uses a repair instruction with the error code, not the raw previous
model output. Token usage, cost, and latency are aggregated across attempts.
`AgentResult.attempts` records how many calls were made, and the orchestrator
writes it to the run trace.

Offline tests cover invalid JSON exhaustion, invalid JSON followed by success,
empty response followed by success, and preflight blocking without any client
call.

## M5.3 — security hardening

- PII checks run on the source text and on the fully rendered prompt. NIK, NPWP,
  phone numbers, and email addresses trigger `PII_EGRESS_BLOCKED`.
- Known injection phrases trigger `PROMPT_INJECTION_BLOCKED` before egress.
- Source text, extraction output, triage output, and verification output are
  wrapped in nonce-derived data blocks. The system prompt states that content
  inside those blocks is untrusted data, never instructions.
- The retry prompt never repeats the rejected model output verbatim.

Offline tests cover English and Indonesian injection patterns, PII in source and
context, redaction helpers, and unique data-block delimiters.

## M5.4 — cassette record and replay

The comparison CLI accepts:

```bash
--cassette PATH
--replay-cassette
--cassette-overwrite
```

A cassette stores request hashes, model and provider metadata, token limits,
responses, usage, and recorded errors. It does not store raw prompts. Replay
requires no API key and reproduces retry behaviour because client errors are
recorded as entries.

Live recording on the 18-post dataset produced **57 entries** and no recorded
errors. Replay produced the same decisions, categories, costs, and latencies.
The only difference was the evaluation timestamp.

## Metrics

| Metric | M5.1 Rule | M5.2 Rule | M5.1 LLM | M5.2 LLM |
|---|---:|---:|---:|---:|
| Decision accuracy | 0.944 | **0.944** | 0.889 | **0.889** |
| Accepted precision | 1.000 | **1.000** | 1.000 | **1.000** |
| Complaint precision | 0.500 | **0.500** | 0.455 | **0.455** |
| Complaint recall | 0.800 | **0.800** | 1.000 | **1.000** |
| Category accuracy | 0.750 | **0.750** | 1.000 | **1.000** |
| Location accuracy | 0.389 | **0.389** | 0.833 | **0.833** |
| Review rate | 0.056 | **0.056** | 0.222 | **0.222** |
| False accepts | 0 | **0** | 0 | **0** |
| Cost USD | 0.000 | **0.000** | 0.001196 | **0.001968** |
| Latency ms | 0.9 | **0.9** | 107,343 | **66,798** |

The decisions are unchanged. The LLM cost rose by about 65% because the hardened
prompts include a security preamble and data-block delimiters; latency fell by
about 38% in this run. All 57 live calls succeeded on the first attempt, so the
cost increase is not retry cost.

## M5.5 — research suite quarantine

Core CI remains green:

```text
611 passed, 43 skipped, 83 deselected
```

The research lane now has an explicit quarantine:

```bash
bash scripts/research_ci.sh
```

Result:

```text
52 passed, 2 skipped, 29 quarantined
```

The full unquarantined lane still has 29 failures. They are recorded in
`tests/quarantine.txt` and grouped as follows:

| Group | Count | Blocker |
|---|---:|---|
| Dashboard / concurrency | 2 | PostgreSQL is not running |
| M3 dataset export | 16 | Missing artifacts; generator anti-leak and coverage defects |
| M3 dataset generator | 4 | Category coverage, family diversity, lexicon anchors, leakage |
| M3 independent holdout | 4 | Missing artifacts; coverage and contamination checks |
| Semantic cache | 2 | Trained IndoBERT artifact is missing |
| Thesis tables | 1 | Missing evaluation report |

The quarantine is visible debt, not a fix. Tests should be removed from the list
only after they pass in a clean checkout.

## Validation

```bash
bash scripts/ci.sh
bash scripts/research_ci.sh
python scripts/compare_harness_agents.py ... --cassette ... 
python scripts/compare_harness_agents.py ... --cassette ... --replay-cassette
```

Artifacts:

- `2026-11-26-harness-x-m5-hardening-cassette.json`
- `2026-11-26-harness-x-m5-hardening-comparison.json`
- `2026-11-26-harness-x-m5-hardening-metrics.json`

## Limits

- No live malicious post was sent through the LLM path; injection blocking is
  validated offline.
- Injection detection is a deterministic cue list, not a complete classifier.
- Retry has no exponential backoff and is capped at two attempts.
- The cassette is a local JSON file and is not encrypted.
- The M3 generator and holdout defects remain; only their tests are quarantined.
