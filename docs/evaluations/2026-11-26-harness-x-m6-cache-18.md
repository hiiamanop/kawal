# Harness Cache Evaluation

- Namespace: `m6-eval-18`
- Analysis version: `harness-x-m6-v1`
- Threshold: 0.92
- Posts in evaluation pass: 18
- Cache-eligible posts: 2

## Cache

| Metric | Value |
|---|---:|
| Exact hits | 2 |
| Semantic hits | 0 |
| Rejected hits | 0 |
| Misses | 16 |
| Overall hit rate | 0.111 |
| Cache-eligible hit rate | 1.000 |
| False reuse | 0 |
| False reuse rate | 0.000 |

## Baseline vs cached

| Metric | Baseline | Cached |
|---|---:|---:|
| Accepted | 3 | 3 |
| Decision accuracy | 0.944 | 0.944 |
| False accepts | 0 | 0 |
| LLM calls | 36 | 32 |
| Cost USD | 0.001066 | 0.000509 |
| Latency ms | 45889.7 | 39047.1 |

- LLM calls saved: 4
- Cost saved USD: 0.000557
- Latency saved ms: 6842.6
