# Harness X — M5.1 Actionable-Complaint Guardrail (2026-11-25)

This evaluation adds a deterministic source-intent guardrail that separates
citizen complaints from news reports, opinions, announcements, and questions.

## Guardrail

Source text is classified as one of:

- `CITIZEN_COMPLAINT`
- `NEWS_REPORT`
- `OPINION`
- `QUESTION`
- `ANNOUNCEMENT`
- `UNKNOWN`

Citizen-request cues take priority. Announcements, opinions, questions, and news
reports force `REJECTED` when the Triage agent says the post is a complaint.
`UNKNOWN` is left to the normal Triage/Verifier pipeline.

The guardrail is deterministic and runs before the final decision. It does not
call a model and adds no API cost.

## Result on the 18-post dataset

| Metric | M4.3 Rule | M5.1 Rule | M4.3 LLM | M5.1 LLM |
|---|---:|---:|---:|---:|
| Decision accuracy | 0.722 | **0.944** | 0.500 | **0.889** |
| False accepts | 1 | **0** | 4 | **0** |
| Accepted precision | n/a | **1.000** | n/a | **1.000** |
| Review rate | n/a | 0.056 | n/a | 0.222 |
| Complaint recall | 0.800 | 0.800 | 1.000 | 1.000 |
| Category accuracy | 0.750 | 0.750 | 1.000 | 1.000 |

- Rule status distribution: 3 accepted / 1 review / 14 rejected.
- LLM status distribution: 3 accepted / 4 review / 11 rejected.
- Raw rule-vs-LLM agreement: 6/18 (33.3%).
- LLM cost: $0.001196; LLM latency: 107.3 seconds.

## Interpretation

- The guardrail removes every false accept on this dataset, for both paths.
- Rule decision accuracy rises from 72.2% to 94.4%; LLM rises from 50% to 88.9%.
- The remaining rule error is the ambiguous Dago accident report, which the rule
  rejects while the label allows review.
- The LLM still sends four posts to review. Two are non-actionable sources where
  Triage and Verifier disagree, so the source guard does not short-circuit them.
  This is conservative and avoids false accepts.
- `complaint_precision` remains low because it measures the Triage
  `is_complaint` flag before the source guard. `accepted_precision` is 1.0 for
  both paths: every accepted case is a real complaint.

## Limitations

- The guardrail is a cue-based deterministic rule; it may overfit these 18 posts.
- Source-intent expectations are encoded in tests, but there is no independent
  human source-type annotation.
- The dataset is small and not class-balanced.
- A real complaint phrased as a question with a problem cue is intentionally left
  as `UNKNOWN` rather than auto-rejected, to avoid false negatives.

Machine-readable metrics: `2026-11-25-harness-x-m5-actionability.json`.
Machine-readable comparison: `2026-11-25-harness-x-m5-actionability-comparison.json`.
