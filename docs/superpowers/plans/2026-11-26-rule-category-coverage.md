# Rule Category Coverage Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the deterministic harness agents from three categories to all 12 municipal complaint categories without weakening the conservative trust model.

**Architecture:** Add a shared category lexicon module, then make Triage, Extraction, and Verifier use it. The orchestrator passes all categories to every agent and forces `NEEDS_REVIEW` for unknown or ambiguous category assignments.

**Tech Stack:** Python 3.15, Pydantic v2, pytest, existing harness agents.

**Spec:** `docs/superpowers/specs/2026-11-26-kawal-harness-monitor-ui-design.md`

## Global Constraints

- All 12 `Category` values must be supported: `ROAD`, `DRAINAGE_FLOOD`, `WASTE`, `CLEAN_WATER`, `CIVIL_ADMIN`, `HEALTH_SERVICE`, `PUBLIC_ORDER`, `TRANSPORTATION`, `FIRE_RESCUE`, `SOCIAL_AFFAIRS`, `EDUCATION`, `PARKS_HOUSING`.
- Out-of-taxonomy complaints must become `NEEDS_REVIEW`, not silently accepted as `ROAD`.
- Ambiguous multi-category complaints must become `NEEDS_REVIEW`.
- The rule path stays offline and zero-cost.
- Do not weaken existing trust rules, PII guards, or `Triage`/`Verifier` independence.
- Code and documentation are English; UI copy may be Indonesian later.

## Review Focus

- A complaint with no category cue must not be accepted as `ROAD`.
- A complaint with two strong category cues must not silently pick one and accept.
- The LLM Triage and Verifier contexts must receive all 12 allowed categories.
- A category-specific extraction object must not overturn a clear category without review.
- Existing clear ROAD complaints must remain accepted with the same behavior.

---

### Task 1: Category lexicon module

**Files:**
- Create: `services/harness/category_lexicon.py`
- Test: `tests/test_harness_category_lexicon.py`

**Interfaces:**
- Produces: `match_category(text: str) -> tuple[Category | None, bool]`
- Produces: `match_object(text: str) -> tuple[str | None, Category | None]`

- [ ] **Step 1: Write the failing tests**

```python
import pytest
from contracts.models import Category
from services.harness.category_lexicon import match_category, match_object

CASES = [
    ("Min, jalan di Jl. Kopo rusak parah, ada lubang besar.", Category.ROAD),
    ("Banjir dan drainase mampet di Cihampelas.", Category.DRAINAGE_FLOOD),
    ("Sampah menumpuk di TPS Cijambe.", Category.WASTE),
    ("Pipa PDAM bocor, air keruh.", Category.CLEAN_WATER),
    ("Pengurusan kartu keluarga terhambat di kelurahan.", Category.CIVIL_ADMIN),
    ("Puskesmas kekurangan obat, antrean panjang.", Category.HEALTH_SERVICE),
    ("Pungli parkir liar di trotoar pasar.", Category.PUBLIC_ORDER),
    ("Angkot ngetem dan lampu lalu lintas mati.", Category.TRANSPORTATION),
    ("Kebakaran rumah dan pohon tumbang.", Category.FIRE_RESCUE),
    ("Bansos belum cair untuk lansia terlantar.", Category.SOCIAL_AFFAIRS),
    ("Atap sekolah rusak dan biaya seragam mahal.", Category.EDUCATION),
    ("Lampu taman mati dan bangku rusak.", Category.PARKS_HOUSING),
]

@pytest.mark.parametrize("text,expected", CASES)
def test_match_category_covers_all_categories(text, expected):
    category, ambiguous = match_category(text)
    assert category == expected
    assert ambiguous is False

def test_match_category_flags_ambiguity():
    category, ambiguous = match_category("Pungli parkir liar di trotoar yang rusak.")
    assert category in {Category.PUBLIC_ORDER, Category.ROAD}
    assert ambiguous is True

def test_match_category_returns_none_for_unknown_text():
    assert match_category("Tolong tindak lanjuti laporan warga.") == (None, False)

def test_match_object_returns_longest_object_and_category():
    assert match_object("Ada lubang besar di jalan") == ("lubang", Category.ROAD)
    assert match_object("Sampah menumpuk di TPS") == ("sampah", Category.WASTE)
    assert match_object("Pipa bocor di depan rumah") == ("pipa bocor", Category.CLEAN_WATER)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_harness_category_lexicon.py -v`
Expected: FAIL with `ModuleNotFoundError: services.harness.category_lexicon`.

- [ ] **Step 3: Implement the lexicon**

Add module-level `TRIAGE_CUES: dict[Category, tuple[str, ...]]` and `OBJECT_CUES: dict[Category, tuple[str, ...]]`.

Triage cue minimums:

```text
ROAD: jalan, jalan rusak, lubang, aspal, trotoar, bahu jalan, jembatan
DRAINAGE_FLOOD: banjir, drainase, gorong-gorong, selokan, parit, genangan
WASTE: sampah, limbah, tps, tumpukan sampah
CLEAN_WATER: pdam, pipa air, pipa bocor, air keruh, kran mati, leding
CIVIL_ADMIN: ktp, kartu keluarga, akta, kependudukan, disdukcapil
HEALTH_SERVICE: puskesmas, rumah sakit, dokter, obat, bidan, ambulans
PUBLIC_ORDER: pungli, parkir liar, pedagang kaki lima, ketertiban, balap liar
TRANSPORTATION: angkot, bus, halte, lampu lalu lintas, rambu, zebra cross
FIRE_RESCUE: kebakaran, pemadam, pohon tumbang, evakuasi, tabung gas
SOCIAL_AFFAIRS: bansos, bantuan sosial, lansia, anak terlantar, panti
EDUCATION: sekolah, guru, kelas, buku, seragam, laboratorium
PARKS_HOUSING: taman, bangku, lampu taman, rumah susun, rusun, bangunan
```

`match_category` scores cues by matched cue length, picks the highest score, and returns `(category, ambiguous)`; `ambiguous` is true when cues from more than one category match. Return `(None, False)` when no cue matches.

`match_object` scans `OBJECT_CUES` and returns the longest matched object cue and its category; return `(None, None)` when no object matches.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_harness_category_lexicon.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add services/harness/category_lexicon.py tests/test_harness_category_lexicon.py
git commit -m "feat: add harness category lexicon"
```

---

### Task 2: Wire all categories into the rule agents and orchestrator

**Files:**
- Modify: `services/harness/agents.py`
- Modify: `services/harness/orchestrator.py:65-70`, `:146-155`, `:195-264`
- Test: `tests/test_harness_agents.py`
- Test: `tests/test_harness_orchestrator.py`
- Test: `tests/test_harness_e2e.py`

**Interfaces:**
- Consumes: `match_category`, `match_object` from Task 1.
- Produces: `RuleTriageAgent` reason codes `CATEGORY_UNKNOWN` and `AMBIGUOUS_CATEGORY`.
- Produces: `HarnessOrchestrator` passes every `Category.value` in `allowed_categories`.

- [ ] **Step 1: Write failing agent tests**

Add to `tests/test_harness_agents.py`:

```python
@pytest.mark.parametrize("text,expected", [
    ("Min, jalan di Jl. Kopo rusak parah.", Category.ROAD),
    ("Banjir dan drainase mampet.", Category.DRAINAGE_FLOOD),
    ("Sampah menumpuk di TPS.", Category.WASTE),
    ("Pipa PDAM bocor, air keruh.", Category.CLEAN_WATER),
    ("Kartu keluarga belum diproses.", Category.CIVIL_ADMIN),
    ("Puskesmas kekurangan obat.", Category.HEALTH_SERVICE),
    ("Pungli parkir liar di trotoar.", Category.PUBLIC_ORDER),
    ("Angkot ngetem dan lampu lalu lintas mati.", Category.TRANSPORTATION),
    ("Kebakaran rumah dan pohon tumbang.", Category.FIRE_RESCUE),
    ("Bansos belum cair untuk lansia.", Category.SOCIAL_AFFAIRS),
    ("Atap sekolah rusak.", Category.EDUCATION),
    ("Lampu taman mati dan bangku rusak.", Category.PARKS_HOUSING),
])
def test_triage_covers_all_categories(text, expected):
    result = RuleTriageAgent().run(_task(text))
    assert result.output is not None
    assert result.output["category"] == expected.value
```

Add tests asserting:

```python
def test_triage_marks_unknown_category():
    result = RuleTriageAgent().run(_task("Tolong tindak lanjuti laporan warga."))
    assert result.output is not None
    assert "CATEGORY_UNKNOWN" in result.output["reason_codes"]

def test_triage_marks_ambiguous_category():
    result = RuleTriageAgent().run(_task("Pungli parkir liar di trotoar yang rusak."))
    assert result.output is not None
    assert "AMBIGUOUS_CATEGORY" in result.output["reason_codes"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_harness_agents.py -v`
Expected: FAIL for the new category cases.

- [ ] **Step 3: Implement agent wiring**

In `RuleTriageAgent.run`:

- Use `category, ambiguous = match_category(lower)`.
- If `category is None`, set `category = Category.ROAD` and append `CATEGORY_UNKNOWN`.
- If `ambiguous`, append `AMBIGUOUS_CATEGORY`.
- Lower confidence to `0.55` for unknown/ambiguous categories.
- Keep existing complaint and hazard behavior.

In `RuleExtractionAgent.run`:

- Use `object_value, _ = match_object(lower)` instead of the three-cue tuple.
- Keep street/label extraction and evidence behavior unchanged.

In `RuleVerifierAgent.run`:

- Use `category, _ = match_category(lower)` and `object_value, object_category = match_object(lower)`.
- If `object_category is not None` and differs from `category`, use `object_category` and append issue `CATEGORY_CONFLICT`.
- If `category is None`, use `Category.ROAD` and append issue `CATEGORY_UNKNOWN`.
- Keep evidence checks unchanged.

- [ ] **Step 4: Write failing orchestrator tests**

Add to `tests/test_harness_orchestrator.py`:

```python
def test_orchestrator_passes_all_categories_to_agents(monkeypatch):
    seen: list[list[str]] = []
    original_triage = RuleTriageAgent.run
    original_verifier = RuleVerifierAgent.run

    def capture_triage(self, task):
        seen.append(list(task.context.get("allowed_categories", [])))
        return original_triage(self, task)

    def capture_verifier(self, task):
        seen.append(list(task.context.get("allowed_categories", [])))
        return original_verifier(self, task)

    monkeypatch.setattr(RuleTriageAgent, "run", capture_triage)
    monkeypatch.setattr(RuleVerifierAgent, "run", capture_verifier)
    post = SocialPost(post_id="all-categories", text="Jl. Kopo rusak parah, tolong diperbaiki.")
    _orchestrator().run(post)
    assert set(seen[0]) == {category.value for category in Category}
    assert set(seen[1]) == {category.value for category in Category}
```

Add end-to-end tests:

```python
def test_unknown_category_requires_review(tmp_path):
    post = SocialPost(post_id="unknown-cat", text="Tolong tindak lanjut laporan warga di Bandung.")
    run = _orchestrator().run(post)
    assert run.status == "NEEDS_REVIEW"
    assert run.case is not None
    assert "CATEGORY_UNKNOWN" in run.case.reason_codes

def test_ambiguous_category_requires_review():
    post = SocialPost(post_id="ambiguous-cat", text="Pungli parkir liar di trotoar yang rusak parah di Jl. Kopo.")
    run = _orchestrator().run(post)
    assert run.status == "NEEDS_REVIEW"
    assert run.case is not None
    assert "AMBIGUOUS_CATEGORY" in run.case.reason_codes
```

- [ ] **Step 5: Run tests to verify they fail**

Run: `pytest tests/test_harness_orchestrator.py::test_orchestrator_passes_all_categories_to_agents tests/test_harness_orchestrator.py::test_unknown_category_requires_review tests/test_harness_orchestrator.py::test_ambiguous_category_requires_review -v`
Expected: FAIL.

- [ ] **Step 6: Implement orchestrator wiring**

In `HarnessOrchestrator.run`:

- Replace both `"allowed_categories": ["ROAD"]` values with `[category.value for category in Category]`.
- After parsing triage, set `category_review = any(code in triage.reason_codes for code in ("CATEGORY_UNKNOWN", "AMBIGUOUS_CATEGORY"))`.
- In the final decision block, include the triage category reason codes in `reason_codes` and force `NEEDS_REVIEW` when `category_review` is true.
- Do not change the existing source-intent, location, evidence, or disagreement behavior.

- [ ] **Step 7: Run tests to verify they pass**

Run: `pytest tests/test_harness_agents.py tests/test_harness_orchestrator.py tests/test_harness_e2e.py -v`
Expected: PASS, including the existing clear-ROAD acceptance tests.

- [ ] **Step 8: Commit**

```bash
git add services/harness/agents.py services/harness/orchestrator.py tests/test_harness_agents.py tests/test_harness_orchestrator.py tests/test_harness_e2e.py
git commit -m "feat: cover all complaint categories in rule agents"
```

---

### Task 3: Regression validation and documentation

**Files:**
- Modify: `docs/testing.md` (only if test counts change)
- Test: full core and research lanes

- [ ] **Step 1: Run the focused harness tests**

Run: `pytest tests/test_harness_agents.py tests/test_harness_category_lexicon.py tests/test_harness_orchestrator.py tests/test_harness_e2e.py -q`
Expected: PASS.

- [ ] **Step 2: Run core CI**

Run: `bash scripts/ci.sh`
Expected: PASS with the updated test count.

- [ ] **Step 3: Run research lane**

Run: `bash scripts/research_ci.sh`
Expected: PASS; quarantine remains empty.

- [ ] **Step 4: Update test-count documentation**

Update `docs/testing.md` and the README status row if the core test count changed. Do not rewrite historical evaluation counts.

- [ ] **Step 5: Commit**

```bash
git add docs/testing.md README.md
git commit -m "docs: update validation counts for category coverage"
```
