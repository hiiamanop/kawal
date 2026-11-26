# Harness X — M6 Semantic Cache Evaluation (2026-11-26)

This evaluation measures cache-first routing on the 18 real X fixtures and on a
synthetic duplicate stream that keeps the original non-complaints.

Cost savings compare the separate no-cache baseline with the evaluation pass.
Warmup cost is not included in the saved values.

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
| Decision accuracy | 0.944 | 0.889 |
| False accepts | 0 | 0 |
| LLM calls | 36 | 32 |
| Cost USD | 0.000536 | 0.000475 |
| Latency ms | 47,090 | 47,780 |

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
| Cost saved USD | 0.000061 |
| Latency saved ms | -690 |

The decision-accuracy difference comes from LLM nondeterminism on the 16
non-cacheable posts between the independent baseline and cache passes. The two
cache hits themselves had zero false reuse. The latency difference is noise at
this cost scale.

## Duplicate stream

The duplicate stream repeats the two cacheable accepted posts three times and
keeps all 15 non-complaints. That gives 24 evaluation posts: 6 cacheable repeats
and 18 non-cacheable posts.

| Metric | Baseline (no cache) | Cache-first |
|---|---:|---:|
| Posts | 24 | 24 |
| Accepted | 9 | 9 |
| Decision accuracy | 0.958 | 0.958 |
| False accepts | 0 | 0 |
| LLM calls | 48 | 36 |
| Cost USD | 0.000738 | 0.000514 |
| Latency ms | 73,819 | 58,759 |

Cache-specific:

| Metric | Value |
|---|---:|
| Cache-eligible posts | 6 |
| Exact hits | 6 |
| Semantic hits | 0 |
| Overall hit rate | 0.250 |
| Cache-eligible hit rate | 1.000 |
| False reuse | 0 |
| LLM calls saved | 12 |
| Cost saved USD | 0.000224 |
| Latency saved ms | 15,060 |

The duplicate stream shows the intended savings: once an accepted case is
cached, repeated exact reports cost zero LLM calls. Non-complaints still run the
full pipeline and are not reused.

## Safety checks

- The cache records only accepted, rule-consistent, PII-free cases.
- A full post body in an evidence span or other free-text field is rejected.
- News, opinion, question, and announcement cues block reuse even when the text
  also contains citizen words such as `lapor` or `tolong`.
- Current rule Triage, Extraction, and Verifier guards run on every hit.
- Stale analysis versions and embedding model identities cannot reuse entries.
- The 13 non-complaint X fixtures were seeded into the cache and replayed in the
  test suite; none was reused.

## Limitations

- Semantic matching was not exercised live because the deterministic fallback
  disables it. Unit tests exercise the semantic path with the fallback
  explicitly enabled, but a real IndoBERT threshold calibration is still needed.
- The Postgres/pgvector path was validated separately: the integration test,
  HNSW planner check, and a `--inner rule` smoke run passed. The LLM evaluation
  above used the in-memory store.
- The dataset is small and class-imbalanced.
- Live LLM decisions can vary between independent passes on non-cacheable posts.
- Evidence spans and embeddings are sensitive local data even though the full
  post body is not stored.

## Artifacts

- `2026-11-26-harness-x-m6-cache-18.json`
- `2026-11-26-harness-x-m6-cache-18.md`
- `2026-11-26-harness-x-m6-cache-duplicate.json`
- `2026-11-26-harness-x-m6-cache-duplicate.md`
