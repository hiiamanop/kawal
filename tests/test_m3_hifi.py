from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import re
from typing import Any

from contracts.models import Category, Completeness, DatasetSplit, DecisionMode

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


def test_generator_writes_matching_ood_manifest_checksum(tmp_path: Path) -> None:
    module = load_generator()
    module.generate_hifi_dataset(tmp_path, seed=42)
    actual_sha256 = hashlib.sha256((tmp_path / "ood_canary.jsonl").read_bytes()).hexdigest()
    manifest = json.loads((tmp_path / "hifi_manifest.json").read_text(encoding="utf-8"))
    assert manifest["artifacts"]["ood_canary"]["sha256"] == actual_sha256
    assert manifest["datasets"]["ood_canary"]["sha256"] == actual_sha256


def test_generator_is_synthetic_and_structurally_complete(tmp_path: Path) -> None:
    module = load_generator()
    module.generate_hifi_dataset(tmp_path, seed=42)
    train, dev, test, ood = (records(tmp_path / f"{name}.jsonl") for name in ("train", "dev", "test", "ood_canary"))
    assert [len(x) for x in (train, dev, test, ood)] == [600, 132, 132, 360]
    assert {r["category"] for r in train + dev + test} == {c.value for c in Category}
    assert {r["split"] for r in train} == {DatasetSplit.TRAIN.value}
    assert {r["split"] for r in dev} == {DatasetSplit.DEV.value}
    assert {r["split"] for r in test + ood} == {DatasetSplit.TEST.value}
    assert all(r["world_truth"]["is_synthetic"] and not r["world_truth"]["real_world"] for r in train + dev + test + ood)
    assert not {r["family_id"] for r in train + dev + test} & {r["family_id"] for r in ood}


def test_no_split_identification_markers_or_cross_split_text_leakage(tmp_path: Path) -> None:
    module = load_generator()
    module.generate_hifi_dataset(tmp_path, seed=7)
    split_records = {name: records(tmp_path / f"{name}.jsonl") for name in ("train", "dev", "test")}
    forbidden = re.compile(r"\b(?:satu|dua|temuan|perkara)\b", re.I)
    normalized: dict[str, set[str]] = {}
    for name, values in split_records.items():
        normalized[name] = set()
        for record in values:
            text = " ".join(texts(record))
            assert not forbidden.search(text)
            normalized[name].add(re.sub(r"\s+", " ", text.lower()).strip())
    assert not normalized["train"] & normalized["dev"]
    assert not normalized["train"] & normalized["test"]
    assert not normalized["dev"] & normalized["test"]


def test_semantic_intent_is_single_act_and_risk_is_intrinsic(tmp_path: Path) -> None:
    module = load_generator()
    module.generate_hifi_dataset(tmp_path, seed=13)
    intent_framing = {
        "train": module.SPLIT_DATA["train"]["intents"],
        "dev": module.SPLIT_DATA["dev"]["intents"],
        "test": module.SPLIT_DATA["test"]["intents"],
        "ood": module.SPLIT_DATA["ood"]["intents"],
    }
    for record in records(tmp_path / "trajectories.jsonl") + records(tmp_path / "ood_canary.jsonl"):
        text = " ".join(texts(record)).lower()
        split = "ood" if record["world_truth"]["provenance"] == module.PROVENANCE_SYNTHETIC_INDEPENDENT else record["split"]
        intent = record["world_truth"]["intent"]
        assert intent_framing[split][intent].lower() in text
        assert module.SPLIT_DATA[split]["risks"][record["world_truth"]["risk"]].lower() in text


def test_completeness_and_actions_match_observable_information(tmp_path: Path) -> None:
    module = load_generator()
    module.generate_hifi_dataset(tmp_path, seed=19)
    for record in records(tmp_path / "trajectories.jsonl") + records(tmp_path / "ood_canary.jsonl"):
        truth = record["world_truth"]
        text = " ".join(texts(record)).lower()
        action = record["turns"][0]["expected_action"]["allowed_actions"]
        split = "ood" if truth["provenance"] == module.PROVENANCE_SYNTHETIC_INDEPENDENT else record["split"]
        if truth["completeness"] == Completeness.SUFFICIENT.value:
            assert all(value.lower() in text for value in record["observable_facts"][1:4])
            assert action == [DecisionMode.EXECUTE.value]
        elif truth["completeness"] == Completeness.INCOMPLETE.value:
            assert module.SPLIT_DATA[split]["completeness"]["INCOMPLETE"].lower() in text
            assert action == [DecisionMode.REQUEST_CLARIFICATION.value]
            assert record["turns"][0]["expected_action"]["missing"] == ["location_detail"]
        else:
            assert "atau" in text and module.SPLIT_DATA[split]["completeness"]["AMBIGUOUS"].lower() in text
            assert action == [DecisionMode.RE_EVALUATE.value]


def test_canonical_spans_are_exact_nonoverlapping_and_label_nondeterministic(tmp_path: Path) -> None:
    module = load_generator()
    module.generate_hifi_dataset(tmp_path, seed=23)
    values = records(tmp_path / "trajectories.jsonl")
    pairs = {(r["world_truth"]["intent"], r["world_truth"]["risk"], r["world_truth"]["completeness"]) for r in values}
    assert len(pairs) == 36
    for record in values + records(tmp_path / "ood_canary.jsonl"):
        for turn in record["turns"]:
            for bubble in turn["bubbles"]:
                spans = bubble["canonical_spans"]
                assert [span[2] for span in spans] and {span[2] for span in spans} >= {"OBJ", "LOC", "TIME"}
                assert spans == sorted(spans, key=lambda span: (span[0], span[1]))
                for start, end, label in spans:
                    assert bubble["text"][start:end] == next((fact for fact in record["observable_facts"] if bubble["text"][start:end] == fact), bubble["text"][start:end]) or label == "LOC"
                    assert 0 <= start < end <= len(bubble["text"])


def test_manifest_checksums_and_determinism(tmp_path: Path) -> None:
    module = load_generator()
    first, second = tmp_path / "one", tmp_path / "two"
    module.generate_hifi_dataset(first, seed=31)
    module.generate_hifi_dataset(second, seed=31)
    for path in first.iterdir():
        if path.name in {"manifest.json", "dataset_manifest.json"}:
            continue
        assert (second / path.name).read_bytes() == path.read_bytes()
    for line in (first / "sha256sums.txt").read_text().splitlines():
        digest, name = line.split()
        assert hashlib.sha256((first / name).read_bytes()).hexdigest() == digest
    manifest = json.loads((first / "manifest.json").read_text())
    assert manifest["ood"]["withheld_from_train"] is True
    assert manifest["artifacts"]["ood_canary.jsonl"]["sha256"]
