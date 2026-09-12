from __future__ import annotations

import json
from pathlib import Path
import socket
from typing import Any
import pytest

from contracts.models import Category, ComplaintTrajectory, DatasetSplit
from scripts import DatasetAuditError
from scripts.evaluate_m3 import EvaluationConfig, run_evaluate_m3
from scripts.generate_m3_dataset import (
    ALL_SCENARIO_TEMPLATES,
    audit_corpus_anti_leak,
    build_arg_parser,
    generate_and_export_m3_dataset,
    main,
)
from scripts.train_dapt import load_corpus, validate_corpus_text
from scripts.train_multitask import load_or_generate_dataset
from scripts.train_ner import extract_trajectory_ner_samples
from services.dataset.generator import load_trajectories_from_jsonl
from services.dataset.splits import audit_family_splits
from services.ml.manifest import ArtifactManifest, compute_file_sha256


@pytest.fixture(autouse=True)
def enforce_no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure that all tests run strictly offline without network access."""
    def _blocked_connect(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("Network access is strictly forbidden in no-network tests.")

    monkeypatch.setattr(socket.socket, "connect", _blocked_connect)


def test_no_network_dataset_generation(tmp_path: Path) -> None:
    """Test that dataset generation and export completes with zero network access."""
    out_dir = tmp_path / "offline_m3"
    result = generate_and_export_m3_dataset(
        output_dir=out_dir,
        seed=101,
        num_scenarios=72,
        corpus_target_lines=300,
        audit=True,
    )
    assert result["status"] == "SUCCESS"
    assert (out_dir / "manifest.json").is_file()
    assert (out_dir / "dataset_manifest.json").is_file()
    assert (out_dir / "corpus.txt").is_file()


def test_deterministic_generation_reproducibility(tmp_path: Path) -> None:
    """Test that two runs with the same seed generate bit-for-bit identical outputs."""
    dir1 = tmp_path / "run1"
    dir2 = tmp_path / "run2"

    res1 = generate_and_export_m3_dataset(
        output_dir=dir1,
        seed=42,
        num_scenarios=72,
        corpus_target_lines=300,
    )
    res2 = generate_and_export_m3_dataset(
        output_dir=dir2,
        seed=42,
        num_scenarios=72,
        corpus_target_lines=300,
    )

    for fname in ("train.jsonl", "dev.jsonl", "test_synthetic.jsonl", "corpus.txt", "sha256sums.txt"):
        file1 = dir1 / fname
        file2 = dir2 / fname
        assert file1.is_file() and file2.is_file()
        assert compute_file_sha256(file1) == compute_file_sha256(file2)
        assert file1.read_bytes() == file2.read_bytes()

    assert res1["counts"] == res2["counts"]


def test_seed_sensitivity_different_hashes(tmp_path: Path) -> None:
    """Test that different seeds produce distinct outputs."""
    dir1 = tmp_path / "seed_42"
    dir2 = tmp_path / "seed_43"

    generate_and_export_m3_dataset(output_dir=dir1, seed=42, num_scenarios=72, corpus_target_lines=200)
    generate_and_export_m3_dataset(output_dir=dir2, seed=43, num_scenarios=72, corpus_target_lines=200)

    hash1 = compute_file_sha256(dir1 / "train.jsonl")
    hash2 = compute_file_sha256(dir2 / "train.jsonl")
    assert hash1 != hash2


def test_active_m3_synthetic_split_anti_leak_audit() -> None:
    """Audit the exported dataset in artifacts/m3_synthetic for zero-leakage guarantee."""
    dataset_dir = Path("artifacts/m3_synthetic")
    assert dataset_dir.is_dir(), "artifacts/m3_synthetic must exist"

    all_file = dataset_dir / "trajectories.jsonl"
    assert all_file.is_file(), "trajectories.jsonl must exist"

    trajectories = load_trajectories_from_jsonl(all_file)
    assert len(trajectories) == 720

    audit = audit_family_splits(trajectories)
    assert audit.passed is True
    assert len(audit.violations) == 0
    assert len(audit.oracle_leakages) == 0
    assert len(audit.cross_split_text_duplicates) == 0
    assert len(audit.family_overlap) == 0

    assert audit.split_trajectories["train"] == 480
    assert audit.split_trajectories["dev"] == 120
    assert audit.split_trajectories["test"] == 120

    assert audit.split_families["train"] == 48
    assert audit.split_families["dev"] == 12
    assert audit.split_families["test"] == 12


def test_stratified_category_coverage() -> None:
    """Ensure that all 6 categories are represented in each split."""
    dataset_dir = Path("artifacts/m3_synthetic")
    all_categories = set(Category)

    for split_fname in ("train.jsonl", "dev.jsonl", "test_synthetic.jsonl"):
        split_file = dataset_dir / split_fname
        trajectories = load_trajectories_from_jsonl(split_file)
        cats = {t.category for t in trajectories}
        assert cats == all_categories, f"Split {split_fname} missing categories: {all_categories - cats}"


def test_multitask_heads_label_coverage_in_all_splits() -> None:
    """Verify that every train/dev/test split contains all classes expected by multitask heads."""
    dataset_dir = Path("artifacts/m3_synthetic")
    expected_intents = {"COMPLAINT", "INQUIRY", "FEEDBACK"}
    expected_categories = {c.value for c in Category}
    expected_risks = {"LOW", "MEDIUM", "HIGH", "URGENT"}
    expected_completeness = {"SUFFICIENT", "INCOMPLETE", "AMBIGUOUS"}

    for split_fname in ("train.jsonl", "dev.jsonl", "test_synthetic.jsonl"):
        split_file = dataset_dir / split_fname
        assert split_file.is_file(), f"{split_fname} missing"
        trajectories = load_trajectories_from_jsonl(split_file)

        # 1. Trajectory-level verification
        intents = {str(t.world_truth.get("intent")) for t in trajectories}
        categories = {t.category.value for t in trajectories}
        risks = {str(t.world_truth.get("risk")) for t in trajectories}
        completeness = {str(t.world_truth.get("completeness")) for t in trajectories}

        assert intents == expected_intents, f"{split_fname} missing intents: {expected_intents - intents}"
        assert categories == expected_categories, f"{split_fname} missing categories: {expected_categories - categories}"
        assert risks == expected_risks, f"{split_fname} missing risks: {expected_risks - risks}"
        assert completeness == expected_completeness, f"{split_fname} missing completeness: {expected_completeness - completeness}"

        # 2. Multitask sample extractor verification
        from scripts.train_multitask import extract_trajectory_multitask_samples
        samples = extract_trajectory_multitask_samples(trajectories)
        s_intents = {s["intent"] for s in samples}
        s_categories = {s["category"] for s in samples}
        s_risks = {s["risk"] for s in samples}
        s_completeness = {s["completeness"] for s in samples}

        assert s_intents == expected_intents
        assert s_categories == expected_categories
        assert s_risks == expected_risks
        assert s_completeness == expected_completeness


def test_dataset_manifest_multitask_head_coverage() -> None:
    """Verify that dataset_manifest.json accurately records full multitask head distributions."""
    dataset_dir = Path("artifacts/m3_synthetic")
    dataset_manifest_path = dataset_dir / "dataset_manifest.json"
    assert dataset_manifest_path.is_file()

    with open(dataset_manifest_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    splits = meta.get("splits", {})
    for split_name in ("train", "dev", "test"):
        split_meta = splits.get(split_name, {})
        assert "intents" in split_meta, f"Missing intents in {split_name} manifest"
        assert "risks" in split_meta, f"Missing risks in {split_name} manifest"
        assert "completeness" in split_meta, f"Missing completeness in {split_name} manifest"

        assert set(split_meta["intents"].keys()) == {"COMPLAINT", "INQUIRY", "FEEDBACK"}
        assert set(split_meta["categories"].keys()) == {c.value for c in Category}
        assert set(split_meta["risks"].keys()) == {"LOW", "MEDIUM", "HIGH", "URGENT"}
        assert set(split_meta["completeness"].keys()) == {"SUFFICIENT", "INCOMPLETE", "AMBIGUOUS"}


def test_synthetic_heldout_clear_labeling() -> None:
    """Verify that synthetic held-out evaluation data is explicitly labeled everywhere."""
    dataset_dir = Path("artifacts/m3_synthetic")

    test_synth_file = dataset_dir / "test_synthetic.jsonl"
    test_file = dataset_dir / "test.jsonl"
    assert test_synth_file.is_file()
    assert test_file.is_file()
    assert compute_file_sha256(test_synth_file) == compute_file_sha256(test_file)

    test_trajectories = load_trajectories_from_jsonl(test_synth_file)
    assert len(test_trajectories) == 120
    for t in test_trajectories:
        assert t.split == DatasetSplit.TEST
        assert t.world_truth.get("synthetic_heldout") is True
        assert t.world_truth.get("heldout_type") == "synthetic"
        assert t.world_truth.get("is_synthetic") is True
        assert t.world_truth.get("dataset_origin") == "synthetic"

    dataset_manifest_path = dataset_dir / "dataset_manifest.json"
    with open(dataset_manifest_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    heldout = meta.get("heldout", {})
    assert heldout.get("status") == "synthetic"
    assert heldout.get("is_synthetic") is True
    assert heldout.get("label") == "SYNTHETIC_HELDOUT"
    assert heldout.get("file") == "test_synthetic.jsonl"
    assert heldout.get("trajectories_count") == 120

    manifest_path = dataset_dir / "manifest.json"
    manifest = ArtifactManifest.from_file(manifest_path)
    assert "synthetic-heldout-test" in manifest.artifacts
    item = manifest.artifacts["synthetic-heldout-test"]
    assert item.task == "synthetic-heldout"
    assert item.path == "test_synthetic.jsonl"


def test_dapt_corpus_quality_and_anti_leak() -> None:
    """Verify that DAPT corpus is cleansed, PII-free, and contains no test leaks."""
    dataset_dir = Path("artifacts/m3_synthetic")
    corpus_path = dataset_dir / "corpus.txt"
    assert corpus_path.is_file()

    corpus_lines = load_corpus(corpus_path, pii_scrubbing=True)
    assert len(corpus_lines) >= 1500

    test_trajectories = load_trajectories_from_jsonl(dataset_dir / "test_synthetic.jsonl")
    dev_trajectories = load_trajectories_from_jsonl(dataset_dir / "dev.jsonl")

    corpus_violations = audit_corpus_anti_leak(
        corpus_lines=corpus_lines,
        heldout_trajectories=test_trajectories + dev_trajectories,
    )
    assert len(corpus_violations) == 0, f"Corpus anti-leak failed: {corpus_violations[:3]}"


def test_artifact_manifest_validation() -> None:
    """Validate ArtifactManifest schema and sha256 checksums."""
    dataset_dir = Path("artifacts/m3_synthetic")
    manifest = ArtifactManifest.from_file(dataset_dir / "manifest.json")
    results = manifest.validate_all(dataset_dir)
    assert len(results) >= 7
    for name, res in results.items():
        assert res.is_valid is True, f"Artifact {name} failed validation: {res.error_message}"
        assert res.file_exists is True


def test_sha256sums_file_matches_actual_files() -> None:
    """Verify that sha256sums.txt contains exact hashes for all exported files."""
    dataset_dir = Path("artifacts/m3_synthetic")
    sha_file = dataset_dir / "sha256sums.txt"
    assert sha_file.is_file()

    checked_count = 0
    with open(sha_file, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            expected_sha, fname = stripped.split()
            target_path = dataset_dir / fname
            assert target_path.is_file(), f"File {fname} in sha256sums.txt does not exist"
            actual_sha = compute_file_sha256(target_path)
            assert actual_sha == expected_sha, f"Hash mismatch for {fname}"
            checked_count += 1

    assert checked_count >= 8


def test_integration_with_m3_components() -> None:
    """Verify compatibility with train_dapt, train_multitask, train_ner, and evaluate_m3."""
    dataset_dir = Path("artifacts/m3_synthetic")

    corpus = load_corpus(dataset_dir / "corpus.txt")
    assert len(corpus) >= 1500

    train_trajs = load_or_generate_dataset(str(dataset_dir / "train.jsonl"), seed=42)
    assert len(train_trajs) == 480

    ner_samples = extract_trajectory_ner_samples(train_trajs)
    assert len(ner_samples) > 0
    assert any(len(s.get("spans", [])) > 0 for s in ner_samples)

    eval_config = EvaluationConfig(
        dataset_path=str(dataset_dir / "trajectories.jsonl"),
        seed=42,
    )
    eval_report = run_evaluate_m3(eval_config)
    assert eval_report.status == "PASSED"
    assert eval_report.split_audit["passed"] is True


def test_cli_execution_and_help(tmp_path: Path) -> None:
    """Test CLI argument parsing and execution."""
    parser = build_arg_parser()
    assert parser is not None

    test_out = tmp_path / "cli_m3"
    exit_code = main(
        [
            "--output-dir",
            str(test_out),
            "--seed",
            "77",
            "--num-scenarios",
            "72",
            "--corpus-lines",
            "250",
        ]
    )
    assert exit_code == 0
    assert (test_out / "manifest.json").is_file()
    assert (test_out / "sha256sums.txt").is_file()
    assert (test_out / "test_synthetic.jsonl").is_file()


def test_generator_template_count_validation(tmp_path: Path) -> None:
    """Test that generator rejects fewer than 6 scenario templates."""
    few_templates = ALL_SCENARIO_TEMPLATES[:3]
    with pytest.raises(ValueError, match="At least 6 scenario templates"):
        generate_and_export_m3_dataset(
            output_dir=tmp_path / "invalid",
            templates=few_templates,
        )


def test_exported_dataset_grounded_risk_and_ambiguous_evidence() -> None:
    """Verify that artifacts/m3_synthetic has grounded risk, ambiguous evidence, and semantic intents."""
    dataset_dir = Path("artifacts/m3_synthetic")
    trajectories = load_trajectories_from_jsonl(dataset_dir / "trajectories.jsonl")
    assert len(trajectories) == 720

    for traj in trajectories:
        all_text = " ".join(b.text.lower() for turn in traj.turns for b in turn.bubbles)
        risk = str(traj.world_truth.get("risk"))
        comp = str(traj.world_truth.get("completeness"))
        intent = str(traj.world_truth.get("intent"))

        # Risk grounding
        if risk == "LOW":
            assert any(
                w in all_text
                for w in (
                    "belum parah",
                    "tidak ada bahaya",
                    "tidak mendesak",
                    "tidak darurat",
                    "blm parah",
                    "gak ada bahaya",
                    "gak mendesak",
                    "gak darurat",
                    "blm darurat",
                )
            ), f"LOW risk ungrounded in: {all_text}"
        elif risk == "MEDIUM":
            assert any(
                w in all_text
                for w in (
                    "makin rusak",
                    "semakin rusak",
                    "makin rsk",
                    "semakin rsk",
                    "belum darurat",
                    "blm darurat",
                    "belum sangat parah",
                    "blm sangat parah",
                    "blm bgt parah",
                )
            ), f"MEDIUM risk ungrounded in: {all_text}"
        elif risk == "HIGH":
            assert any(
                w in all_text
                for w in ("sangat parah", "bahaya", "mendesak", "segera ditangani", "sgr ditangani")
            ), f"HIGH risk ungrounded in: {all_text}"
        elif risk == "URGENT":
            assert any(
                w in all_text
                for w in ("sangat darurat", "darurat parah", "bahaya sekali", "sekarang", "skrg", "darurat sangat mendesak")
            ), f"URGENT risk ungrounded in: {all_text}"

        # Completeness grounding & ambiguous evidence
        if comp == "AMBIGUOUS":
            assert traj.location_completeness == "AMBIGUOUS"
            assert traj.turns[0].expected_action.missing == ("location_ambiguity",)
            b2_text = traj.turns[0].bubbles[1].text.lower()
            assert any(
                w in b2_text
                for w in ("antara", "atau", "seberang", "samping", "belakang", "depan")
            )
        elif comp == "INCOMPLETE":
            assert traj.location_completeness == "INCOMPLETE"
            assert traj.turns[0].expected_action.missing == ("jurisdiction",)
        elif comp == "SUFFICIENT":
            assert traj.location_completeness == "SUFFICIENT"
            assert traj.turns[-1].expected_action.missing == ()

        # Intent semantic language
        if intent == "COMPLAINT":
            assert any(w in all_text for w in ("lapor", "laporkan", "laporan"))
        elif intent == "INQUIRY":
            assert any(w in all_text for w in ("informasi", "info", "bagaimana", "keterangan"))
        elif intent == "FEEDBACK":
            assert any(w in all_text for w in ("terima kasih", "makasih", "penanganan"))


def test_exported_dataset_scale_and_family_diversity() -> None:
    """Verify that artifacts/m3_synthetic has >=720 scenarios and >=12 families per category."""
    dataset_dir = Path("artifacts/m3_synthetic")
    all_file = dataset_dir / "trajectories.jsonl"
    assert all_file.is_file()

    trajectories = load_trajectories_from_jsonl(all_file)
    assert len(trajectories) >= 720

    families = {t.family_id for t in trajectories}
    assert len(families) >= 72

    cat_to_fams: dict[Category, set[str]] = {c: set() for c in Category}
    for t in trajectories:
        cat_to_fams[t.category].add(t.family_id)

    for cat in Category:
        assert (
            len(cat_to_fams[cat]) >= 12
        ), f"Category {cat} has {len(cat_to_fams[cat])} exported families, expected >= 12"

    manifest_path = dataset_dir / "dataset_manifest.json"
    with open(manifest_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    splits = meta.get("splits", {})
    assert splits["train"]["trajectories_count"] == 480
    assert splits["dev"]["trajectories_count"] == 120
    assert splits["test"]["trajectories_count"] == 120
    assert splits["train"]["families_count"] == 48
    assert splits["dev"]["families_count"] == 12
    assert splits["test"]["families_count"] == 12


def test_exported_dataset_ner_evidence_density() -> None:
    """Verify that exported train trajectories contain robust compact OBJ, LOC, and TIME NER evidence."""
    dataset_dir = Path("artifacts/m3_synthetic")
    train_file = dataset_dir / "train.jsonl"
    assert train_file.is_file()

    trajectories = load_trajectories_from_jsonl(train_file)
    samples = extract_trajectory_ner_samples(trajectories)
    assert len(samples) > 0

    found_labels = set()
    for s in samples:
        for _, _, lbl in s.get("spans", []):
            found_labels.add(lbl)

    assert "OBJ" in found_labels
    assert "LOC" in found_labels
    assert "TIME" in found_labels


def test_exported_dataset_category_anchor_lexicon_integrity() -> None:
    """Verify that exported trajectories contain no raw enum/ID leaks."""
    dataset_dir = Path("artifacts/m3_synthetic")
    trajectories = load_trajectories_from_jsonl(dataset_dir / "trajectories.jsonl")

    category_enum_names = {c.value.lower() for c in Category}
    for traj in trajectories:
        for turn in traj.turns:
            for bubble in turn.bubbles:
                text_lower = bubble.text.lower()
                for c_name in category_enum_names:
                    assert (
                        c_name not in text_lower
                    ), f"Category enum {c_name} found leaked in bubble: {bubble.text}"


def test_exported_trajectories_category_anchor_composition() -> None:
    """Verify that exported artifacts contain unambiguous composed category cues and compact objects in reporter bubbles."""
    dataset_dir = Path("artifacts/m3_synthetic")
    trajectories = load_trajectories_from_jsonl(dataset_dir / "trajectories.jsonl")
    assert len(trajectories) == 720

    drainage_cues = ("drainase", "parit", "gorong", "limpasan", "banjir", "saluran", "genangan", "luapan")
    waste_cues = ("sampah", "limbah", "kotoran", "residu")
    clean_water_cues = ("air bersih", "pdam", "kran", "leding", "perpipaan", "meteran")

    for traj in trajectories:
        b1_text = traj.turns[0].bubbles[0].text.lower()
        if traj.category == Category.DRAINAGE_FLOOD:
            assert any(c in b1_text for c in drainage_cues), f"Drainage cue missing in {traj.scenario_id}: {b1_text}"
        elif traj.category == Category.WASTE:
            assert any(c in b1_text for c in waste_cues), f"Waste cue missing in {traj.scenario_id}: {b1_text}"
        elif traj.category == Category.CLEAN_WATER:
            assert any(c in b1_text for c in clean_water_cues), f"Clean water cue missing in {traj.scenario_id}: {b1_text}"



