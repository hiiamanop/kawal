from __future__ import annotations

import hashlib
import importlib.util
import json
from collections import Counter
from pathlib import Path
import re
from typing import Any

from contracts.models import (
    Category,
    ComplaintTrajectory,
    DatasetSplit,
    DecisionMode,
)
from services.dataset.splits import audit_family_splits, _is_common_language_ngram

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "generate_m3_hifi.py"


def load_generator() -> Any:
    spec = importlib.util.spec_from_file_location("generate_m3_hifi", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def records(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def texts(record: dict[str, Any]) -> list[str]:
    return [bubble["text"] for turn in record["turns"] for bubble in turn["bubbles"]]


def extract_non_common_ngrams(texts_seq: list[str], n: int) -> set[tuple[str, ...]]:
    ngs: set[tuple[str, ...]] = set()
    for text in texts_seq:
        toks = re.findall(r"[a-zA-Z0-9]+", text.lower())
        for i in range(len(toks) - n + 1):
            ng = tuple(toks[i : i + n])
            if not _is_common_language_ngram(ng):
                ngs.add(ng)
    return ngs


def test_generator_writes_matching_ood_manifest_checksum(tmp_path: Path) -> None:
    module = load_generator()
    module.generate_hifi_dataset(tmp_path, seed=42)
    actual_sha256 = hashlib.sha256((tmp_path / "ood_canary.jsonl").read_bytes()).hexdigest()
    manifest = json.loads((tmp_path / "hifi_manifest.json").read_text(encoding="utf-8"))
    assert manifest["artifacts"]["ood_canary"]["sha256"] == actual_sha256
    assert manifest["datasets"]["ood_canary"]["sha256"] == actual_sha256
    manifest_v2 = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest_v2["artifacts"]["ood_canary.jsonl"]["sha256"] == actual_sha256


def test_generator_is_synthetic_and_structurally_complete(tmp_path: Path) -> None:
    module = load_generator()
    module.generate_hifi_dataset(tmp_path, seed=42)
    train, dev, test, ood = (records(tmp_path / f"{name}.jsonl") for name in ("train", "dev", "test", "ood_canary"))
    assert [len(x) for x in (train, dev, test, ood)] == [2880, 576, 864, 864]
    trajectories = records(tmp_path / "trajectories.jsonl")
    assert len(trajectories) == 4320
    assert {r["category"] for r in train + dev + test + ood} == {c.value for c in Category}
    assert {r["split"] for r in train} == {DatasetSplit.TRAIN.value}
    assert {r["split"] for r in dev} == {DatasetSplit.DEV.value}
    assert {r["split"] for r in test + ood} == {DatasetSplit.TEST.value}
    assert all(r["world_truth"]["is_synthetic"] and not r["world_truth"]["real_world"] for r in train + dev + test + ood)

    train_fams = {r["family_id"] for r in train}
    dev_fams = {r["family_id"] for r in dev}
    test_fams = {r["family_id"] for r in test}
    ood_fams = {r["family_id"] for r in ood}
    assert not (train_fams & dev_fams)
    assert not (train_fams & test_fams)
    assert not (dev_fams & test_fams)
    assert not (train_fams & ood_fams)
    assert not (dev_fams & ood_fams)
    assert not (test_fams & ood_fams)


def test_precise_per_category_counts(tmp_path: Path) -> None:
    module = load_generator()
    module.generate_hifi_dataset(tmp_path, seed=42)
    expected_counts = {"train": 240, "dev": 48, "test": 72, "ood_canary": 72}
    for name, expected in expected_counts.items():
        recs = records(tmp_path / f"{name}.jsonl")
        counts = Counter(r["category"] for r in recs)
        assert len(counts) == 12
        for cat in Category:
            assert counts[cat.value] == expected, f"{name} {cat.value} count mismatch"


def test_concept_representation_all_52_in_train_dev_test(tmp_path: Path) -> None:
    module = load_generator()
    module.generate_hifi_dataset(tmp_path, seed=42)
    train_recs = records(tmp_path / "train.jsonl")
    dev_recs = records(tmp_path / "dev.jsonl")
    test_recs = records(tmp_path / "test.jsonl")
    ood_recs = records(tmp_path / "ood_canary.jsonl")

    for cat in Category:
        tr_concepts = {r["world_truth"]["concept_id"] for r in train_recs if r["category"] == cat.value}
        de_concepts = {r["world_truth"]["concept_id"] for r in dev_recs if r["category"] == cat.value}
        te_concepts = {r["world_truth"]["concept_id"] for r in test_recs if r["category"] == cat.value}
        ood_concepts = {r["world_truth"]["concept_id"] for r in ood_recs if r["category"] == cat.value}

        assert tr_concepts == set(range(52)), f"Train missing concepts in {cat.value}"
        assert te_concepts == set(range(52)), f"Test missing concepts in {cat.value}"
        assert ood_concepts == set(range(52)), f"OOD missing concepts in {cat.value}"
        assert len(de_concepts) == 48, f"Dev count mismatch in {cat.value}"
        assert tr_concepts | de_concepts | te_concepts == set(range(52))

    for recs in (train_recs, dev_recs, test_recs, ood_recs):
        for r in recs:
            assert "concept_id" in r["world_truth"]
            full_text = " ".join(texts(r))
            assert not re.search(r"\b(?:concept|concept_id|family|city12|split|seed)[_-]?\d*\b", full_text, re.I)


def test_all_labels_balanced_per_category_and_split(tmp_path: Path) -> None:
    module = load_generator()
    module.generate_hifi_dataset(tmp_path, seed=42)
    for name in ("train", "dev", "test", "ood_canary"):
        recs = records(tmp_path / f"{name}.jsonl")
        for cat in Category:
            subset = [r for r in recs if r["category"] == cat.value]
            for key, options in (("intent", module.INTENTS), ("risk", module.RISKS), ("completeness", module.COMPLETENESS)):
                counts = Counter(r["world_truth"][key] for r in subset)
                assert set(counts.keys()) == set(options)
                assert max(counts.values()) - min(counts.values()) <= 1


def test_shared_semantic_label_vocab_across_splits(tmp_path: Path) -> None:
    module = load_generator()
    module.generate_hifi_dataset(tmp_path, seed=42)
    train_text = " ".join(" ".join(texts(r)).lower() for r in records(tmp_path / "train.jsonl"))
    dev_text = " ".join(" ".join(texts(r)).lower() for r in records(tmp_path / "dev.jsonl"))
    test_text = " ".join(" ".join(texts(r)).lower() for r in records(tmp_path / "test.jsonl"))

    for key, pool in module.LABEL_POOLS.items():
        for label, phrases in pool.items():
            # Check that shared vocab phrases appear across train, dev, and test
            assert any(phrase.lower() in train_text for phrase in phrases[:3])
            assert any(phrase.lower() in dev_text for phrase in phrases[:3])
            assert any(phrase.lower() in test_text for phrase in phrases[:3])


def test_internal_family_split_and_ngram_audit(tmp_path: Path) -> None:
    module = load_generator()
    module.generate_hifi_dataset(tmp_path, seed=42)
    internal = [
        ComplaintTrajectory.model_validate_json(line)
        for line in (tmp_path / "trajectories.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    result = audit_family_splits(internal)
    assert result.passed, f"Family split audit failed: {result.violations[:10]}"
    assert len(result.cross_split_text_duplicates) == 0
    assert len(result.oracle_leakages) == 0


def test_ood_explicit_ngram_audit_against_internal(tmp_path: Path) -> None:
    module = load_generator()
    module.generate_hifi_dataset(tmp_path, seed=42)
    internal_recs = records(tmp_path / "trajectories.jsonl")
    ood_recs = records(tmp_path / "ood_canary.jsonl")

    internal_texts = [text for r in internal_recs for text in texts(r)]
    ood_texts = [text for r in ood_recs for text in texts(r)]

    for n in (5, 6, 7):
        in_ng = extract_non_common_ngrams(internal_texts, n)
        od_ng = extract_non_common_ngrams(ood_texts, n)
        overlap = in_ng & od_ng
        assert not overlap, f"OOD vs internal overlap at {n}-gram: {list(overlap)[:5]}"


def test_canonical_obj_boundary_delimiters_exact(tmp_path: Path) -> None:
    module = load_generator()
    module.generate_hifi_dataset(tmp_path, seed=42)
    for name in ("train", "dev", "test", "ood_canary"):
        for record in records(tmp_path / f"{name}.jsonl"):
            for turn in record["turns"]:
                for bubble in turn["bubbles"]:
                    text = bubble["text"]
                    for span in bubble["canonical_spans"]:
                        start, end, label = span[0], span[1], span[2]
                        if label == "OBJ":
                            post = text[end:]
                            assert any(post.startswith(delimiter) for delimiter in module.O_BOUNDARY_DELIMITERS), (
                                f"OBJ span in {record['scenario_id']} not followed by standard delimiter: {post[:20]!r}"
                            )


def test_style_family_representation_across_splits(tmp_path: Path) -> None:
    module = load_generator()
    module.generate_hifi_dataset(tmp_path, seed=42)
    for name in ("train", "dev", "test"):
        recs = records(tmp_path / f"{name}.jsonl")
        styles = Counter(r["world_truth"]["style_family"] for r in recs)
        assert set(styles.keys()) == set(module.STYLE_FAMILIES), f"{name} missing style_family representation"
        assert max(styles.values()) - min(styles.values()) == 0, f"{name} style families unbalanced"


def test_risk_metadata_and_evidence_policy(tmp_path: Path) -> None:
    module = load_generator()
    module.generate_hifi_dataset(tmp_path, seed=42)
    for name in ("train", "dev", "test", "ood_canary"):
        recs = records(tmp_path / f"{name}.jsonl")
        for r in recs:
            wt = r["world_truth"]
            risk = wt["risk"]
            kind = wt["risk_evidence_kind"]
            if risk == "LOW":
                assert kind == "cosmetic_minor"
            elif risk == "MEDIUM":
                assert kind == "operational_disruption"
            elif risk == "HIGH":
                assert kind == "safety_hazard"
            elif risk == "URGENT":
                assert kind == "immediate_emergency"

            text = r["turns"][0]["bubbles"][0]["text"].lower()
            if risk == "LOW":
                for bad in module.UNSAFE_WORDS_IN_LOW:
                    assert bad not in text, f"Unsafe word '{bad}' found in LOW risk text: {text}"
                orig_issue = module.ISSUES[Category(r["category"])][wt["concept_id"]].lower()
                assert orig_issue not in text, f"Original severe issue occurred unchanged in LOW: {orig_issue}"
            elif risk == "URGENT":
                has_trigger = any(trig in text for trig in module.URGENT_TRIGGER_WORDS)
                assert has_trigger, f"URGENT missing immediate trigger keyword: {text}"


def test_single_act_intents(tmp_path: Path) -> None:
    module = load_generator()
    module.generate_hifi_dataset(tmp_path, seed=42)
    for name in ("train", "dev", "test", "ood_canary"):
        recs = records(tmp_path / f"{name}.jsonl")
        for r in recs:
            intent = r["world_truth"]["intent"]
            text = r["turns"][0]["bubbles"][0]["text"].lower()
            first_sent = text.split(".")[0]
            if intent == "COMPLAINT":
                assert "?" not in first_sent
                assert not any(q in first_sent for q in ("kapan", "apakah", "bagaimana", "bolehkah"))
                assert not any(f in first_sent for f in ("usul", "saran", "rekomendasi", "sebaiknya", "alangkah"))
            elif intent == "INQUIRY":
                assert not any(c in first_sent for c in ("mengeluhkan", "mengganggu", "merugikan", "meresahkan", "komplain"))
                assert not any(f in first_sent for f in ("usul", "saran", "rekomendasi", "sebaiknya", "alangkah"))
            elif intent == "FEEDBACK":
                assert "?" not in first_sent
                assert not any(q in first_sent for q in ("kapan", "apakah", "bagaimana", "bolehkah"))
                assert not any(c in first_sent for c in ("mengeluhkan", "mengganggu", "merugikan", "meresahkan", "komplain"))


def test_location_completeness_spans_pure(tmp_path: Path) -> None:
    module = load_generator()
    module.generate_hifi_dataset(tmp_path, seed=42)
    for name in ("train", "dev", "test", "ood_canary"):
        recs = records(tmp_path / f"{name}.jsonl")
        for r in recs:
            comp = r["location_completeness"]
            loc_spans = [s for s in r["canonical_spans"] if s[2] == "LOC"]
            text = r["turns"][0]["bubbles"][0]["text"]
            loc_texts = [text[s[0]:s[1]] for s in loc_spans]
            if comp == "SUFFICIENT":
                assert len(loc_texts) == 5, f"SUFFICIENT must have full address spans: {loc_texts}"
            elif comp == "INCOMPLETE":
                assert len(loc_texts) == 1, f"INCOMPLETE must be kelurahan only: {loc_texts}"
                assert "kelurahan" in loc_texts[0].lower() or "kel." in loc_texts[0].lower()
                assert not any(w in loc_texts[0].lower() for w in ("catatan", "mohon", "detail", "info"))
            elif comp == "AMBIGUOUS":
                assert len(loc_texts) == 1, f"AMBIGUOUS must be landmark only: {loc_texts}"
                assert "kelurahan" not in loc_texts[0].lower()
                assert "atau" not in loc_texts[0].lower()
                assert not any(w in loc_texts[0].lower() for w in ("catatan", "mohon", "detail", "info"))


def test_artifacts_directory_is_valid_and_matches_checksums() -> None:
    artifacts_dir = ROOT / "artifacts" / "m3_hifi"
    manifest = json.loads((artifacts_dir / "hifi_manifest.json").read_text(encoding="utf-8"))
    actual_ood_sha256 = hashlib.sha256((artifacts_dir / "ood_canary.jsonl").read_bytes()).hexdigest()
    assert manifest["artifacts"]["ood_canary"]["sha256"] == actual_ood_sha256
    assert manifest["datasets"]["ood_canary"]["sha256"] == actual_ood_sha256

    trajs = [
        ComplaintTrajectory.model_validate_json(line)
        for line in (artifacts_dir / "trajectories.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(trajs) == 4320
    audit_res = audit_family_splits(trajs)
    assert audit_res.passed, f"Audit failed: {audit_res.violations[:10]}"
