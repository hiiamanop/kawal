# Harness Cache Evaluation

- Namespace: `m6-eval-dup`
- Analysis version: `harness-x-m6-v1`
- Threshold: 0.92
- Posts in evaluation pass: 9
- Cache-eligible posts: 6

## Cache

| Metric | Value |
|---|---:|
| Exact hits | 6 |
| Semantic hits | 0 |
| Rejected hits | 0 |
| Misses | 3 |
| Overall hit rate | 0.667 |
| Cache-eligible hit rate | 1.000 |
| False reuse | 0 |
| False reuse rate | 0.000 |

## Baseline vs cached

| Metric | Baseline | Cached |
|---|---:|---:|
| Accepted | 9 | 9 |
| Decision accuracy | 1.000 | 1.000 |
| False accepts | 0 | 0 |
| LLM calls | 18 | 6 |
| Cost USD | 0.000281 | 0.000067 |
| Latency ms | 20336.0 | 5787.8 |

- LLM calls saved: 12
- Cost saved USD: 0.000214
- Latency saved ms: 14548.2
