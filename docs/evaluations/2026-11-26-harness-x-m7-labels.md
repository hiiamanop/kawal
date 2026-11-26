# Harness X — M7 Label Metadata Audit (2026-11-26)

M7 was planned as a dataset expansion and label-quality pass. The expansion part
is blocked by the current X upstream state. This document records what was
improved, what is still blocked, and what should not be claimed.

## Dataset state

- 18 recorded X fixtures.
- 5 labeled complaints: 3 `COMPLETED`, 2 `NEEDS_REVIEW`.
- 13 non-complaints: 12 `REJECTED` plus one additional review case.
- Categories: 4 `ROAD`, 1 `DRAINAGE_FLOOD`.

The set is small and class-imbalanced. It is not a representative sample of X
traffic.

## What changed

Every label now carries a `source_type`:

- `CITIZEN_COMPLAINT`: 4
- `NEWS_REPORT`: 5
- `OPINION`: 2
- `QUESTION`: 2
- `ANNOUNCEMENT`: 3
- `UNKNOWN`: 2

The comparison scorer now reports source-intent accuracy against this field.
The metric is useful for catching classifier regressions, but it is not
independent human ground truth: the source types were assigned by the same
project that wrote the classifier. A 1.0 score means the classifier matches the
recorded annotation, not that both are correct.

## Expansion blocker

`agent-reach twitter search` and `agent-reach twitter tweets` return
`UPSTREAM_BROKEN` on 2026-11-26. Web search found mostly news posts that are
already in the dataset or unrelated traffic commentary. No new fixture was added
just to increase the count, because that would mix fabricated or unverifiable
posts into a ground-truth dataset.

The correct next step is to capture new public posts when the X channel is
healthy, then label them with the same schema and a second reviewer.

## Validation

- Every existing label validates with `source_type`.
- `score_harness_comparison.py` reports source-intent accuracy.
- The core suite covers label coverage and source-type presence.

## Artifacts

- `fixtures/x_labels.json`
- `services/harness/metrics.py`
- `scripts/score_harness_comparison.py`
