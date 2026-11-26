# Harness Cache Evaluation

- Namespace: `m6-eval-dup`
- Analysis version: `harness-x-m6-v1`
- Threshold: 0.92
- Posts in evaluation pass: 24
- Cache-eligible posts: 6

## Cache

| Metric | Value |
|---|---:|
| Exact hits | 6 |
| Semantic hits | 0 |
| Rejected hits | 0 |
| Misses | 18 |
| Overall hit rate | 0.250 |
| Cache-eligible hit rate | 1.000 |
| False reuse | 0 |
| False reuse rate | 0.000 |

## Baseline vs evaluation pass

The baseline is a separate no-cache pass. Warmup cost is not included in
the saved values below.

| Metric | Baseline | Cached |
|---|---:|---:|
| Accepted | 9 | 9 |
| Decision accuracy | 0.958 | 0.958 |
| False accepts | 0 | 0 |
| LLM calls | 48 | 36 |
| Cost USD | 0.000738 | 0.000514 |
| Latency ms | 73818.6 | 58759.0 |

- LLM calls saved: 12
- Cost saved USD: 0.000224
- Latency saved ms: 15059.6
