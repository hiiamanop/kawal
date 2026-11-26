# Harness X — M7 Label Metadata Audit (2026-11-26)

M7 was planned as a dataset expansion and label-quality pass. Live X search
remains blocked, but public X posts can still be verified and captured through
the syndication endpoint. This document records what was improved, what is still
blocked, and what should not be claimed.

## Dataset state

- 23 recorded X fixtures.
- 6 labeled complaints: 3 `COMPLETED`, 3 `NEEDS_REVIEW`.
- 17 non-complaints, all `REJECTED`.
- Categories: 5 `ROAD`, 1 `DRAINAGE_FLOOD`.

The set is small and class-imbalanced. It is not a representative sample of X
traffic.

## What changed

Every label now carries a `source_type`:

- `CITIZEN_COMPLAINT`: 5
- `NEWS_REPORT`: 7
- `OPINION`: 2
- `QUESTION`: 2
- `ANNOUNCEMENT`: 4
- `UNKNOWN`: 3

The comparison scorer now reports source-intent accuracy against this field.
The metric is useful for catching classifier regressions, but it is not
independent human ground truth: the source types were assigned by the same
project that wrote the classifier. A 1.0 score means the classifier matches the
recorded annotation, not that both are correct.

Source-intent cues were extended for real phrases from the new posts:
`tewas terperosok`, `satpol pp`, `ditata`, `penataan trotoar`, and
`benerin atuh`.

## M7b public snapshot expansion

Five public X posts were added on 2026-11-26:

- `2067364825471361417` — news report about an ojol fatality at a drain hole on
  Jalan Pasteur.
- `2066315447985910003` — news report about a garage demolished from the trotoar
  on Jalan Ambon.
- `2109917602961354968` — government announcement about trotoar and drainage
  work near Pasar Kiaracondong.
- `2071102573483053232` — ambiguous video commentary about the trotoar in
  Dipati Ukur after enforcement.
- `1925164524103151818` — citizen complaint about the poor state of Bandung
  trotoar, with no specific location.

Each post was verified through `cdn.syndication.twimg.com`; the fixture records
the public snapshot method. No post was invented and no unavailable text was
reconstructed.

## Expansion blocker

`agent-reach twitter search` and `agent-reach twitter tweets` still return
`UPSTREAM_BROKEN` on 2026-11-26. The dedicated browser profile has no X login, so
live search cannot be restored without a human login. The public snapshot path
is a fallback, not a replacement for live search.

The new labels were assigned by the same project that wrote the classifier. A
second reviewer still needs to confirm the five additions before they are treated
as independent human ground truth.

## Validation

- Every existing label validates with `source_type`.
- The source-intent heuristic matches all 23 labeled fixtures.
- A rule-only batch on the 23 fixtures produced 3 accepted, 1 needs-review, and
  19 rejected cases with no failures.
- `score_harness_comparison.py` reports source-intent accuracy.
- The core suite covers label coverage and source-type presence.

## Artifacts

- `fixtures/x_labels.json`
- `services/harness/metrics.py`
- `scripts/score_harness_comparison.py`
