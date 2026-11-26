# Research Suite Cleanup (2026-11-26)

Before this pass, the unquarantined research/integration lane had **29 failures**.
After the pass:

- **73 passed**
- **8 skipped** for missing prerequisites
- **4 quarantined** for real content/model gaps

## Fixed

- Generator split audit: added common location words (`pintu`, `barat`, cardinal
  directions) and eliminated four duplicate jurisdiction strings across families.
- Corpus anti-leak: heldout hidden facts are filtered out of the DAPT corpus.
- Canonical spans: generated trajectories now carry NER spans for LOC, OBJ, and
  TIME when the extractor recognizes them.
- Lexicon validation: 15 legacy templates now contain a valid category anchor.
- Stale expectations: split counts now match the 720-scenario dataset
  (`432/144/144`, `54/18/18`).
- Prerequisite handling: missing PostgreSQL, M3 artifacts, holdout artifacts,
  and thesis artifacts now skip with a reason instead of failing.
- NER pattern coverage: social-aid objects such as `bantuan sembako` and
  `data bansos` are recognized.

## Remaining quarantine

- Six extended categories still have 3 families instead of 12.
- Base IndoBERT reaches 0.758 similarity on the legacy semantic pair, below the
  0.85/0.88 thresholds.

These are content/model work, not infrastructure failures.
