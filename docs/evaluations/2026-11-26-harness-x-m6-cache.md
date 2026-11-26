# Harness X — M6 Semantic Cache Evaluation (2026-11-26)

This evaluation measures cache-first routing on the 18 real X fixtures and on a
synthetic duplicate stream.

## Setup

- Model: `tencent/hy4-flash`
- Prices: $0.035/M input, $0.1405/M output
- Reasoning disabled with `--no-reasoning`
- LLM path: LLM Triage + LLM Verifier, rule Extraction + rule Arbiter
- Embedder: deterministic fallback
- Cache store: in-memory
- Exact match only in the live run; semantic reuse is disabled with the fallback
  embedder by default
- Labels: `fixtures/x_labels.json`

## 18-post two-pass evaluation

Only accepted, rule-consistent cases are cacheable. Three posts were accepted by
the LLM path, but one was excluded because its accepted category did not match
the deterministic rule category. Two posts were therefore cacheable.

| Metric | Baseline (no cache) | Cache-first |
|---|---:|---:|
| Posts | 18 | 18 |
| Accepted | 3 | 3 |
| Decision accuracy | 0.944 | 0.944 |
| False accepts | 0 | 0 |
| LLM calls | 36 | 32 |
| Cost USD | 0.001066 | 0.000509 |
| Latency ms | 45,890 | 39,047 |

Cache-specific:

| Metric | Value |
|---|---:|
| Cache-eligible posts | 2 |
| Exact hits | 2 |
| Semantic hits | 0 |
| Overall hit rate | 0.111 |
| Cache-eligible hit rate | 1.000 |
| False reuse | 0 |
| LLM calls saved | 4 |
| Cost saved USD | 0.000557 |
| Latency saved ms | 6,843 |

The decisions are unchanged. The cache removed the LLM calls for the two
cacheable posts; the other 16 posts still ran the full pipeline.

## Duplicate stream

The duplicate stream repeats accepted labeled posts three times. Two posts were
cacheable, so six of the nine evaluation posts matched exactly. The third
accepted post was not cacheable because of the rule/LLM category mismatch.

| Metric | Baseline (no cache) | Cache-first |
|---|---:|---:|
| Posts | 9 | 9 |
| Accepted | 9 | 9 |
| Decision accuracy | 1.000 | 1.000 |
| False accepts | 0 | 0 |
| LLM calls | 18 | 6 |
| Cost USD | 0.000281 | 0.000067 |
| Latency ms | 20,336 | 5,788 |

Cache-specific:

| Metric | Value |
|---|---:|
| Cache-eligible posts | 6 |
| Exact hits | 6 |
| Overall hit rate | 0.667 |
| Cache-eligible hit rate | 1.000 |
| False reuse | 0 |
| LLM calls saved | 12 |
| Cost saved USD | 0.000214 |
| Latency saved ms | 14,548 |

The duplicate stream shows the intended savings: once an accepted case is
cached, repeated exact reports cost zero LLM calls.

## Interpretation

- The record gate is conservative. One of three accepted LLM cases was not
  cacheable because deterministic Triage/Verifier categories disagreed with the
  accepted category.
- No cache false reuse occurred in either run.
- The 18-post dataset is not duplicate-heavy, so its overall hit rate is low by
  design. The cache-eligible hit rate is 100%, and the duplicate stream shows
  what happens when repeats exist.
- The cache does not change the decision accuracy of the baseline.

## Limitations

- Semantic matching was not exercised live because the deterministic fallback
  disables it. Unit tests exercise the semantic path with the fallback
  explicitly enabled, but a real IndoBERT threshold calibration is still needed.
- The Postgres/pgvector integration test is marked `integration` and was not run
  in this evaluation unless a local migrated database was available.
- The dataset is small and class-imbalanced.
- Evidence spans and embeddings are sensitive local data even though the full
  post body is not stored.

## Artifacts

- `2026-11-26-harness-x-m6-cache-18.json`
- `2026-11-26-harness-x-m6-cache-18.md`
- `2026-11-26-harness-x-m6-cache-duplicate.json`
- `2026-11-26-harness-x-m6-cache-duplicate.md`
