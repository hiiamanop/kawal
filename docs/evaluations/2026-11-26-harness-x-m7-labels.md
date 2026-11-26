# Harness X — M7 Label Metadata Audit (2026-11-26)

M7 was planned as a dataset expansion and label-quality pass. Live X search
remains blocked, but public X posts can still be verified and captured through
the syndication endpoint. This document records what was improved, what is still
blocked, and what should not be claimed.

## Dataset state

- 29 recorded X fixtures.
- 8 labeled complaints: 3 `COMPLETED`, 5 `NEEDS_REVIEW`.
- 21 non-complaints, all `REJECTED`.
- Categories: 7 `ROAD`, 1 `DRAINAGE_FLOOD`.

The set is small and class-imbalanced. It is not a representative sample of X
traffic.

## What changed

Every label now carries a `source_type`:

- `CITIZEN_COMPLAINT`: 7
- `NEWS_REPORT`: 11
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
`tewas terperosok`, `satpol pp`, `ditata`, `penataan trotoar`,
`benerin atuh`, `diabaikan`, `pungli merajalela`,
`warga minta pemerintah`, `reporter:`, `editor:`, `protes warga`, and
`turun tangan`.

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

## M7b live search expansion

After the Chromium cookie import restored the Agent Reach X session, live search
was used for a second batch of four posts:

- `2060576242433450181` — citizen complaint about damaged trotoar and ignored
  pedestrian rights, with no specific street.
- `2107732996271001795` — citizen complaint listing road, traffic-light,
  lighting, and extortion problems in Bandung, with no specific location.
- `2120782368684556339` — news report about severe road damage on
  Cihideung-Gombong, Bandung Barat.
- `2122292838826398059` — news report about a fallen tree damaging a car at
  Jalan Gelap Nyawang, Bandung.
- `2061727987591245863` — news report about a waste pile at TPS Cijambe
  spilling onto the road.
- `2054402389671543238` — news report about a waste protest and cleanup in
  Ngamprah, Bandung Barat.

The fixtures record the live-search source and the syndication verification. The
two long posts use the full text returned by live search; the syndication
endpoint truncates long posts.

## Live search status

Live X search is available again on 2026-11-26 after the Chromium cookie import.
The dedicated Agent Reach window must stay open while capturing. If it is
closed, search can fail with `Detached while handling command`; reopening it
with `agent-reach login twitter` restores access without importing cookies
again.

The new labels were assigned by the same project that wrote the classifier. A
second reviewer still needs to confirm the eleven additions before they are
treated as independent human ground truth.

## Validation

- Every existing label validates with `source_type`.
- The source-intent heuristic matches all 29 labeled fixtures.
- A rule-only batch on the 29 fixtures produced 3 accepted, 3 needs-review, and
  23 rejected cases with no failures.
- `score_harness_comparison.py` reports source-intent accuracy.
- The core suite covers label coverage and source-type presence.

## Artifacts

- `fixtures/x_labels.json`
- `services/harness/metrics.py`
- `scripts/score_harness_comparison.py`
