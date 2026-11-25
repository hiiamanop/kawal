# Harness X — M4 Ground-Truth Evaluation (2026-11-25)

This evaluation adds a location-specificity guardrail and scores the rule and LLM
paths against six hand-labeled X fixtures.

## Guardrail

A complaint can only be `ACCEPTED` when the extracted location is `SPECIFIC`.
`MISSING` and `VAGUE` locations force `NEEDS_REVIEW`, even when the Triage and
Verifier agents agree. The assessment is deterministic and applies to both paths.

## Setup

- Model: `tencent/hy4-flash`, reasoning disabled
- Prices: $0.035 / million input tokens, $0.1405 / million output tokens
- Fixtures: six real X posts with labels in `fixtures/x_labels.json`
- Rule path: all four roles rule-based
- LLM path: `LLMTriageAgent` + `LLMExtractionAgent` + `LLMVerifierAgent` + `LLMArbiterAgent`

## Per-post result

| Post | Label | Rule | LLM |
|---|---|---|---|
| `1807221515089506334` | complaint / ROAD / vague location / review | `NEEDS_REVIEW / ROAD` | `NEEDS_REVIEW / ROAD` |
| `2076110894149652629` | complaint / ROAD / specific / review | `REJECTED / ROAD` | `NEEDS_REVIEW / ROAD` |
| `931149896203169792` | complaint / ROAD / specific / accept | `COMPLETED / DRAINAGE_FLOOD` | `COMPLETED / ROAD` |
| `2026159892885832085` | complaint / ROAD / specific / accept | `COMPLETED / ROAD` | `COMPLETED / ROAD` |
| `2125560260849193400` | non-complaint / reject | `REJECTED / ROAD` | `NEEDS_REVIEW / DRAINAGE_FLOOD` |
| `2116307635217838493` | complaint / DRAINAGE_FLOOD / specific / accept | `COMPLETED / DRAINAGE_FLOOD` | `COMPLETED / DRAINAGE_FLOOD` |

## Metrics

| Metric | Rule | LLM |
|---|---:|---:|
| Complaint precision | 1.000 | 0.833 |
| Complaint recall | 0.800 | 1.000 |
| Category accuracy | 0.750 | 1.000 |
| Decision accuracy | 0.833 | 0.833 |
| Location accuracy | 0.667 | 1.000 |
| Review rate | 0.167 | 0.500 |
| False accepts | 0 | 0 |
| False rejects | 1 | 0 |
| Cost USD | 0.000000 | 0.000541 |
| Cost per correct USD | 0.000000 | 0.000090 |

## Interpretation

- The location guardrail removed the false accept from the previous run: both
  paths now have zero false accepts.
- The rule path is precise but rejects the ambiguous Dago accident report
  (`false_rejects = 1`) and misclassifies the Stasiun Bandung post as drainage.
- The LLM path has perfect category and location accuracy after the guardrail,
  but it sends the weather remark to review, so its precision is lower and its
  review rate is 50%.
- Status agreement is 50%; the ground-truth metrics are more meaningful than
  raw rule-vs-LLM agreement.

## Limitations

- Six labeled posts; labels are subjective and not independently double-annotated.
- One model, one run.
- The weather and accident posts are ambiguous by design; their labels are
  documented in `fixtures/x_labels.json`.
- Live X search remains unavailable; fixtures were captured earlier.

Machine-readable metrics: `2026-11-25-harness-x-m4-labels.json`.
Machine-readable comparison: `2026-11-25-harness-x-m4-comparison.json`.
