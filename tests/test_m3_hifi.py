from __future__ import annotations

import hashlib
import importlib.util
import json
from collections import Counter
from pathlib import Path
import re
from typing import Any

from contracts.models import Category, ComplaintTrajectory, DatasetSplit, DecisionMode
from services.dataset.splits import _is_common_language_ngram, audit_family_splits

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
        for index in range(len(toks) - n + 1):
            ng = tuple(toks[index : index + n])
            if not _is_common_language_ngram(ng):
                ngs.add(ng)
    return ngs


def test_generator_writes_matching_ood_manifest_checksum(tmp_path: Path) -> None:
    module = load_generator()
    module.generate_hifi_dataset(tmp_path, seed=42)
    actual_sha256 = hashlib.sha256((tmp_path / "ood_canary.jsonl").read_bytes()).hexdigest()
    manifest = json.loads((tmp_path / "hifi_manifest.json").read_text(encoding="utf-8"))
    assert manifest["artifacts"]["ood_canary"]["sha256"] == actual_sha256


def test_generator_is_synthetic_and_structurally_complete(tmp_path: Path) -> None:
    module = load_generator()
    module.generate_hifi_dataset(tmp_path, seed=7)
    train, dev, test, ood = (records(tmp_path / f"{name}.jsonl") for name in ("train", "dev", "test", "ood_canary"))
    assert [len(values) for values in (train, dev, test, ood)] == [2880, 576, 864, 864]
    for values, expected in ((train, 240), (dev, 48), (test, 72), (ood, 72)):
        counts = Counter(record["category"] for record in values)
        assert counts == {category.value: expected for category in Category}
    all_records = train + dev + test + ood
    assert all(record["world_truth"]["is_synthetic"] for record in all_records)
    assert all(not record["world_truth"]["real_world"] for record in all_records)
    assert not {record["family_id"] for record in train + dev + test} & {record["family_id"] for record in ood}


def test_internal_family_split_and_ngram_audit(tmp_path: Path) -> None:
    module = load_generator()
    module.generate_hifi_dataset(tmp_path, seed=42)
    internal = [ComplaintTrajectory.model_validate_json(line) for line in (tmp_path / "trajectories.jsonl").read_text(encoding="utf-8").splitlines()]
    result = audit_family_splits(internal)
    assert result.passed, result.violations[:10]


def test_all_concepts_are_represented_in_internal_splits(tmp_path: Path) -> None:
    module = load_generator()
    module.generate_hifi_dataset(tmp_path, seed=12)
    expected = set(range(52))
    for name in ("train", "test"):
        values = records(tmp_path / f"{name}.jsonl")
        for category in Category:
            concepts = {record["world_truth"]["concept_id"] for record in values if record["category"] == category.value}
            assert concepts == expected


def test_label_balance_per_category_and_split(tmp_path: Path) -> None:
    module = load_generator()
    module.generate_hifi_dataset(tmp_path, seed=17)
    for name in ("train", "dev", "test", "ood_canary"):
        values = records(tmp_path / f"{name}.jsonl")
        for category in Category:
            category_values = [record["world_truth"] for record in values if record["category"] == category.value]
            for label, expected_values in (("intent", module.INTENTS), ("risk", module.RISKS), ("completeness", module.COMPLETENESS)):
                counts = Counter(value[label] for value in category_values)
                assert set(counts) == set(expected_values)
                assert max(counts.values()) - min(counts.values()) <= 1


def test_ood_is_independent_at_five_to_seven_grams(tmp_path: Path) -> None:
    module = load_generator()
    module.generate_hifi_dataset(tmp_path, seed=29)
    internal = records(tmp_path / "trajectories.jsonl")
    ood = records(tmp_path / "ood_canary.jsonl")
    internal_text = [text for record in internal for text in texts(record)]
    ood_text = [text for record in ood for text in texts(record)]
    for size in (5, 6, 7):
        assert not extract_non_common_ngrams(internal_text, size) & extract_non_common_ngrams(ood_text, size)


def test_visible_text_has_no_system_identifiers_or_raw_enums(tmp_path: Path) -> None:
    module = load_generator()
    module.generate_hifi_dataset(tmp_path, seed=31)
    forbidden = re.compile(r"\b(?:" + "|".join(category.value.lower() for category in Category) + r")\b|_|concept_id|family_id|scenario_id|city12", re.I)
    for name in ("train", "dev", "test", "ood_canary"):
        for record in records(tmp_path / f"{name}.jsonl"):
            for text in texts(record):
                assert not forbidden.search(text)


def test_canonical_spans_are_exact_nonoverlapping_and_have_o_boundary(tmp_path: Path) -> None:
    module = load_generator()
    module.generate_hifi_dataset(tmp_path, seed=37)
    for name in ("train", "dev", "test", "ood_canary"):
        for record in records(tmp_path / f"{name}.jsonl"):
            for bubble in (bubble for turn in record["turns"] for bubble in turn["bubbles"]):
                text = bubble["text"]
                spans = bubble["canonical_spans"]
                occupied: list[tuple[int, int]] = []
                for start, end, label in spans:
                    assert 0 <= start < end <= len(text)
                    assert text[start:end].strip() == text[start:end]
                    assert not any(start < right and end > left for left, right in occupied)
                    occupied.append((start, end))
                    if label == "OBJ":
                        assert any(text[end:].startswith(delimiter) for delimiter in module.O_BOUNDARY_DELIMITERS)


def test_risk_metadata_and_single_act_intents(tmp_path: Path) -> None:
    module = load_generator()
    module.generate_hifi_dataset(tmp_path, seed=43)
    kind_by_risk = {
        "LOW": "cosmetic_minor",
        "MEDIUM": "operational_disruption",
        "HIGH": "safety_hazard",
        "URGENT": "immediate_emergency",
    }
    for name in ("train", "dev", "test", "ood_canary"):
        for record in records(tmp_path / f"{name}.jsonl"):
            truth = record["world_truth"]
            text = " ".join(texts(record)).lower()
            assert truth["risk_evidence_kind"] == kind_by_risk[truth["risk"]]
            assert truth["intent"] in module.INTENTS
            if truth["risk"] == "LOW":
                assert not any(word in text for word in module.UNSAFE_WORDS_IN_LOW)
            if truth["risk"] == "URGENT":
                assert any(word in text for word in module.URGENT_TRIGGER_WORDS)


def test_location_completeness_is_grounded_in_spans(tmp_path: Path) -> None:
    module = load_generator()
    module.generate_hifi_dataset(tmp_path, seed=47)
    for name in ("train", "dev", "test", "ood_canary"):
        for record in records(tmp_path / f"{name}.jsonl"):
            loc_count = sum(label == "LOC" for bubble in (bubble for turn in record["turns"] for bubble in turn["bubbles"]) for _, _, label in bubble["canonical_spans"])
            completeness = record["world_truth"]["completeness"]
            assert (loc_count >= 5) if completeness == "SUFFICIENT" else (loc_count == 1)


def test_manifest_checksums_and_determinism(tmp_path: Path) -> None:
    module = load_generator()
    module.generate_hifi_dataset(tmp_path / "first", seed=5101)
    module.generate_hifi_dataset(tmp_path / "second", seed=5101)
    files = ("train.jsonl", "dev.jsonl", "test.jsonl", "trajectories.jsonl", "ood_canary.jsonl", "corpus.txt", "corpus_train.txt")
    for name in files:
        assert (tmp_path / "first" / name).read_bytes() == (tmp_path / "second" / name).read_bytes()
