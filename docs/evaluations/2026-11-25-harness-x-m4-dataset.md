# Harness X — M4.3 Dataset Expansion (2026-11-25)

This evaluation expands the labeled X dataset from six to eighteen posts. Twelve
new public posts cover news reports, opinions, schedules, and transit questions;
they are intentional non-complaint controls.

## Dataset

- 18 real public X posts in `fixtures/x_posts/`
- 5 complaints and 13 non-complaints
- Labels in `fixtures/x_labels.json`
- Ground-truth fields: complaint decision, category, location quality, and
  acceptable run status.

## Setup

- Model: `tencent/hy4-flash`, reasoning disabled
- Prices: $0.035 / million input tokens, $0.1405 / million output tokens
- LLM path: Triage + Extraction + Verifier + Arbiter
- Rule path: all four roles rule-based

## Metrics

| Metric | Rule | LLM |
|---|---:|---:|
| Complaint precision | 0.500 | 0.417 |
| Complaint recall | 0.800 | 1.000 |
| Category accuracy | 0.750 | 1.000 |
| Decision accuracy | 0.722 | 0.500 |
| Location accuracy | 0.389 | 0.833 |
| False accepts | 1 | 4 |
| Cost USD | 0.000000 | 0.001904 |

- Rule status distribution: 4 accepted / 4 review / 10 rejected.
- LLM status distribution: 7 accepted / 7 review / 4 rejected.
- Raw rule-vs-LLM agreement: 3/18 (16.7%).
- LLM latency: 76.7 seconds total.

## False accepts

| Post | Path | Label |
|---|---|---|
| `2121593107276030456` | Rule | Police news report about storm damage in Pasirjambu |
| `1513890410447441930` | LLM | Traffic photo and commentary about Nagreg |
| `1998677956323848429` | LLM | Opinion about Bandung elevation data |
| `2122148769735360583` | LLM | News report about a road repair and alternative route |
| `2119723067408879971` | LLM | News report about a Damri garage fire |

## Interpretation

- The rule path is more conservative and has the better decision accuracy on this
  noisy dataset. It still accepts one news report about storm damage because the
  text contains road and hazard cues.
- The LLM path has perfect category accuracy and strong location accuracy, but
  it over-accepts news, opinions, and commentary. Its complaint precision drops
  to 41.7%.
- Raw rule-vs-LLM agreement is misleading because both paths can be wrong in
  different ways. Ground-truth decision accuracy and false accepts are the
  relevant metrics.
- The location-accuracy metric for the rule path is depressed by non-complaint
  posts whose location is absent or vague; it should be interpreted alongside
  complaint precision.

## Limitations

- 18 posts, subjective labels, one model, one run.
- The added controls are mostly news/opinion, so the dataset is not class-balanced.
- Labels are not independently double-annotated.
- Live X search remains unavailable; fixtures were captured from public search
  snapshots.

Machine-readable metrics: `2026-11-25-harness-x-m4-dataset.json`.
Machine-readable comparison: `2026-11-25-harness-x-m4-dataset-comparison.json`.
