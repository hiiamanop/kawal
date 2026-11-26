# Research Suite Cleanup (2026-11-26)

Before this pass, the unquarantined research/integration lane had **29 failures**.
After the pass:

- **78 passed**
- **8 skipped** for missing prerequisites
- **0 quarantined**

## Fixed

- Generator split audit: added common location words (`pintu`, `barat`, cardinal
  directions) and eliminated four duplicate jurisdiction strings across families.
- Corpus anti-leak: heldout hidden facts are filtered out of the DAPT corpus.
- Canonical spans: generated trajectories now carry NER spans for LOC, OBJ, and
  TIME when the extractor recognizes them.
- Lexicon validation: 15 legacy templates now contain a valid category anchor.
- Stale expectations: split counts now match the 720-scenario dataset
  (`480/120/120` trajectories, `96/24/24` families).
- Prerequisite handling: missing PostgreSQL, M3 artifacts, holdout artifacts,
  and thesis artifacts now skip with a reason instead of failing.
- NER pattern coverage: social-aid objects such as `bantuan sembako` and
  `data bansos` are recognized.

## M8b family diversity

- Added 54 curated families, nine for each previously under-covered category:
  PUBLIC_ORDER, TRANSPORTATION, FIRE_RESCUE, SOCIAL_AFFAIRS, EDUCATION, and
  PARKS_HOUSING.
- Every curated family has a unique anchor and issue. A global 3-5 gram check
  against every existing anchor found no overlap, so no quarantine exception was
  needed.
- Split audit passes for seeds 42, 101, and 2024.
- Independent holdout objects were rephrased where they shared n-grams with the
  new source anchors. The holdout audit still rejects any source overlap.
- Removed a pre-existing cross-category traffic-light duplicate by giving the
  TRANSPORTATION family its own `lampu lalu lintas mati` issue. The split audit
  was not weakened to hide it.

## Quarantine cleared

`tests/quarantine.txt` is empty. The two former family-diversity failures now
pass against the 144-family dataset.
