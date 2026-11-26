# Cache Scenario Calibration

- Scenarios: 5
- Best threshold: 0.85

| Threshold | Precision | Recall | Accuracy | TP | FP | FN | TN |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.80 | 1.000 | 1.000 | 1.000 | 2 | 0 | 0 | 3 |
| 0.85 | 1.000 | 1.000 | 1.000 | 2 | 0 | 0 | 3 |
| 0.90 | 1.000 | 0.500 | 0.800 | 1 | 0 | 1 | 3 |
| 0.92 | 1.000 | 0.500 | 0.800 | 1 | 0 | 1 | 3 |
| 0.95 | 1.000 | 0.500 | 0.800 | 1 | 0 | 1 | 3 |

## Scenarios at best threshold

| Scenario | Expected | Actual | Match | Similarity |
|---|---|---|---|---:|
| road-kopo-paraphrase | True | True | SEMANTIC | 0.860 |
| road-cihampelas-paraphrase | True | True | SEMANTIC | 0.970 |
| road-kopo-different-location | False | False | NONE | 0.000 |
| road-kopo-waste-query | False | False | NONE | 0.000 |
| road-kopo-opinion | False | False | NONE | 0.000 |
