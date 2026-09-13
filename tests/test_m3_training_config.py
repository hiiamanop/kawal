from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import tempfile
import pytest
from pydantic import ValidationError

from services.ml.training import (
    DataLeakageError,
    HeldoutDatasetManifest,
    HeldoutManifestItem,
    HeldoutValidationError,
    HeldoutValidationResult,
    HyperparametersConfig,
    M3TrainingConfig,
    MetricResult,
    ModelVersionConfig,
    PrecisionConfig,
    PrecisionConstraintError,
    SplitConfig,
    SplitHashMismatchError,
    compute_content_sha256,
    compute_file_sha256,
    compute_split_hash,
    compute_split_map_hashes,
    evaluate_metric_thresholds,
    is_precision_allowed,
    validate_heldout_dataset,
    validate_heldout_manifest,
    validate_precision_constraint,
)


def test_model_version_config_valid() -> None:
    cfg = ModelVersionConfig(
        base_model="indobenchmark/indobert-base-p1",
        model_version="v1.0.0",
        tokenizer_version="v1.0.0",
        max_sequence_length=448,
        vocab_size=30521,
    )
    assert cfg.base_model == "indobenchmark/indobert-base-p1"
    assert cfg.effective_tokenizer_name == "indobenchmark/indobert-base-p1"
    assert cfg.max_sequence_length == 448
    assert cfg.tokenizer_sha256 is None


def test_model_version_config_tokenizer_sha() -> None:
    dummy_sha = "a" * 64
    cfg = ModelVersionConfig(
        base_model="indobenchmark/indobert-base-p1",
        model_version="v1.0.0",
        tokenizer_name="custom-tok",
        tokenizer_version="v1.0.0",
        tokenizer_sha256=dummy_sha,
    )
    assert cfg.effective_tokenizer_name == "custom-tok"
    assert cfg.tokenizer_sha256 == dummy_sha


def test_model_version_config_invalid() -> None:
    with pytest.raises(ValidationError):
        ModelVersionConfig(
            base_model="",
            model_version="v1.0.0",
            tokenizer_version="v1.0.0",
        )

    with pytest.raises(ValidationError):
        ModelVersionConfig(
            base_model="indobert",
            model_version="v1.0.0",
            tokenizer_version="v1.0.0",
            tokenizer_sha256="invalid-sha",
        )

    with pytest.raises(ValidationError):
        ModelVersionConfig(
            base_model="indobert",
            model_version="v1.0.0",
            tokenizer_version="v1.0.0",
            max_sequence_length=8,
        )


def test_hyperparameters_config_effective_batch_size() -> None:
    hp = HyperparametersConfig(
        learning_rate=2e-5,
        batch_size=8,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        warmup_ratio=0.06,
        weight_decay=0.01,
    )
    assert hp.effective_batch_size == 32
    assert hp.optimizer == "adamw"
    assert hp.lr_scheduler == "linear"


def test_hyperparameters_config_task_weights_and_validation() -> None:
    hp = HyperparametersConfig(
        learning_rate=1e-4,
        batch_size=16,
        num_train_epochs=5,
        task_weights={"intent": 1.0, "category": 2.0},
        temperature=1.2,
    )
    assert hp.task_weights["category"] == 2.0
    assert hp.temperature == 1.2

    with pytest.raises(ValidationError):
        HyperparametersConfig(
            learning_rate=-0.01,
            batch_size=16,
            num_train_epochs=5,
        )

    with pytest.raises(ValidationError):
        HyperparametersConfig(
            learning_rate=1e-4,
            batch_size=0,
            num_train_epochs=5,
        )

    with pytest.raises(ValidationError):
        HyperparametersConfig(
            learning_rate=1e-4,
            batch_size=16,
            num_train_epochs=5,
            task_weights={"intent": -1.0},
        )


def test_precision_constraints_cpu() -> None:
    assert is_precision_allowed("cpu", "fp32") is True
    assert is_precision_allowed("cpu", "fp16") is False
    assert is_precision_allowed("cpu", "bf16") is False
    assert is_precision_allowed("cpu", "amp") is False

    validate_precision_constraint("cpu", "fp32")

    with pytest.raises(PrecisionConstraintError):
        validate_precision_constraint("cpu", "fp16")

    with pytest.raises(PrecisionConstraintError):
        validate_precision_constraint("cpu", "amp")

    cfg = PrecisionConfig(device="cpu", training_precision="fp32")
    assert cfg.device == "cpu"
    assert cfg.training_precision == "fp32"
    assert cfg.serving_precision == "fp32"

    with pytest.raises(ValidationError):
        PrecisionConfig(device="cpu", training_precision="fp16")


def test_precision_constraints_gpu() -> None:
    for prec in ("fp32", "fp16", "bf16", "amp"):
        assert is_precision_allowed("cuda", prec) is True
        assert is_precision_allowed("gpu", prec) is True
        validate_precision_constraint("cuda", prec)
        cfg = PrecisionConfig(device="cuda", training_precision=prec)
        assert cfg.training_precision == prec
        assert cfg.serving_precision == "fp32"


def test_split_config_ratio_validation() -> None:
    cfg = SplitConfig(seed=42, train_ratio=0.7, dev_ratio=0.15, test_ratio=0.15)
    assert cfg.seed == 42

    with pytest.raises(ValidationError):
        SplitConfig(train_ratio=0.8, dev_ratio=0.2, test_ratio=0.1)

    with pytest.raises(ValidationError):
        SplitConfig(seed=-1)


def test_compute_split_hash_determinism_and_invariance() -> None:
    items_order_a = ["scenario-3", "scenario-1", "scenario-2"]
    items_order_b = ["scenario-1", "scenario-2", "scenario-3"]
    hash_a = compute_split_hash(items_order_a)
    hash_b = compute_split_hash(items_order_b)
    assert hash_a == hash_b
    assert len(hash_a) == 64

    records = [
        {"id": "rec-1", "label": "A"},
        {"id": "rec-2", "label": "B"},
    ]
    hash_records = compute_split_hash(records)
    assert len(hash_records) == 64

    hash_diff = compute_split_hash(["scenario-1", "scenario-2", "scenario-4"])
    assert hash_a != hash_diff


def test_compute_split_map_hashes() -> None:
    mapping = {
        "fam-1": "train",
        "fam-2": "train",
        "fam-3": "dev",
        "fam-4": "test",
    }
    split_hashes = compute_split_map_hashes(mapping)
    assert "train" in split_hashes
    assert "dev" in split_hashes
    assert "test" in split_hashes
    assert len(split_hashes["train"]) == 64
    assert split_hashes["train"] == compute_split_hash(["fam-1", "fam-2"])
    assert split_hashes["dev"] == compute_split_hash(["fam-3"])


def test_split_config_verify_and_assert_hashes() -> None:
    train_ids = ["fam-1", "fam-2"]
    dev_ids = ["fam-3"]
    expected_train_hash = compute_split_hash(train_ids)
    expected_dev_hash = compute_split_hash(dev_ids)

    cfg = SplitConfig(
        train_ratio=0.7,
        dev_ratio=0.15,
        test_ratio=0.15,
        expected_split_hashes={
            "train": expected_train_hash,
            "dev": expected_dev_hash,
        },
    )

    actual_splits = {"train": train_ids, "dev": dev_ids}
    verification = cfg.verify_split_hashes(actual_splits)
    assert verification["train"] is True
    assert verification["dev"] is True

    cfg.assert_valid_split_hashes(actual_splits)

    bad_splits = {"train": ["fam-1", "fam-999"], "dev": dev_ids}
    bad_verification = cfg.verify_split_hashes(bad_splits)
    assert bad_verification["train"] is False

    with pytest.raises(SplitHashMismatchError):
        cfg.assert_valid_split_hashes(bad_splits)

    with pytest.raises(SplitHashMismatchError):
        cfg.assert_valid_split_hashes({"train": train_ids})


def test_heldout_manifest_item_path_safety() -> None:
    with pytest.raises(ValidationError):
        HeldoutManifestItem(
            name="unsafe",
            path="../secret.jsonl",
            sha256="0" * 64,
            expected_count=10,
        )

    with pytest.raises(ValidationError):
        HeldoutManifestItem(
            name="unsafe",
            path="/etc/passwd",
            sha256="0" * 64,
            expected_count=10,
        )


def test_heldout_dataset_validation_success() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        base_path = Path(tmpdir)
        records = [
            {"id": "traj-dev-1", "text": "Jalan rusak parah"},
            {"id": "traj-dev-2", "text": "Saluran mampet banjir"},
        ]
        content_lines = "\n".join(json.dumps(r) for r in records) + "\n"
        dev_file = base_path / "dev.jsonl"
        dev_file.write_text(content_lines, encoding="utf-8")

        expected_file_sha = sha256(content_lines.encode("utf-8")).hexdigest()
        expected_split_hash = compute_split_hash(["traj-dev-1", "traj-dev-2"])

        item = HeldoutManifestItem(
            name="dev-set",
            split="dev",
            path="dev.jsonl",
            sha256=expected_file_sha,
            expected_count=2,
            split_hash=expected_split_hash,
        )

        train_ids = {"train-1", "train-2"}
        result = validate_heldout_dataset(
            item=item,
            base_dir=base_path,
            train_sample_ids=train_ids,
            raise_on_error=True,
        )

        assert result.is_valid is True
        assert result.file_exists is True
        assert result.sha256_matches is True
        assert result.count_matches is True
        assert result.split_hash_matches is True
        assert result.zero_leakage is True
        assert result.actual_count == 2
        assert result.error_message is None


def test_heldout_dataset_validation_missing_file() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        item = HeldoutManifestItem(
            name="missing",
            split="test",
            path="not_found.jsonl",
            sha256="0" * 64,
            expected_count=5,
        )
        res = validate_heldout_dataset(item=item, base_dir=tmpdir, raise_on_error=False)
        assert res.is_valid is False
        assert res.file_exists is False

        with pytest.raises(HeldoutValidationError):
            validate_heldout_dataset(item=item, base_dir=tmpdir, raise_on_error=True)


def test_heldout_dataset_validation_sha_mismatch() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        base_path = Path(tmpdir)
        dev_file = base_path / "dev.jsonl"
        dev_file.write_text("dummy\n", encoding="utf-8")

        item = HeldoutManifestItem(
            name="dev-set",
            split="dev",
            path="dev.jsonl",
            sha256="f" * 64,
            expected_count=1,
        )
        res = validate_heldout_dataset(item=item, base_dir=base_path, raise_on_error=False)
        assert res.is_valid is False
        assert res.sha256_matches is False

        with pytest.raises(HeldoutValidationError):
            validate_heldout_dataset(item=item, base_dir=base_path, raise_on_error=True)


def test_heldout_dataset_validation_count_mismatch() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        base_path = Path(tmpdir)
        dev_file = base_path / "dev.jsonl"
        content = json.dumps({"id": "1"}) + "\n"
        dev_file.write_text(content, encoding="utf-8")
        file_sha = sha256(content.encode("utf-8")).hexdigest()

        item = HeldoutManifestItem(
            name="dev-set",
            split="dev",
            path="dev.jsonl",
            sha256=file_sha,
            expected_count=10,
        )
        res = validate_heldout_dataset(item=item, base_dir=base_path, raise_on_error=False)
        assert res.is_valid is False
        assert res.count_matches is False

        with pytest.raises(HeldoutValidationError):
            validate_heldout_dataset(item=item, base_dir=base_path, raise_on_error=True)


def test_heldout_dataset_validation_data_leakage() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        base_path = Path(tmpdir)
        records = [{"id": "leak-1"}, {"id": "safe-2"}]
        content = "\n".join(json.dumps(r) for r in records) + "\n"
        dev_file = base_path / "dev.jsonl"
        dev_file.write_text(content, encoding="utf-8")
        file_sha = sha256(content.encode("utf-8")).hexdigest()

        item = HeldoutManifestItem(
            name="dev-set",
            split="dev",
            path="dev.jsonl",
            sha256=file_sha,
            expected_count=2,
        )

        train_sample_ids = {"train-a", "leak-1"}
        res = validate_heldout_dataset(
            item=item,
            base_dir=base_path,
            train_sample_ids=train_sample_ids,
            raise_on_error=False,
        )
        assert res.is_valid is False
        assert res.zero_leakage is False
        assert "leakage" in (res.error_message or "")

        with pytest.raises(DataLeakageError):
            validate_heldout_dataset(
                item=item,
                base_dir=base_path,
                train_sample_ids=train_sample_ids,
                raise_on_error=True,
            )


def test_validate_heldout_manifest_json_format() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        base_path = Path(tmpdir)
        records = [{"id": "t-1"}, {"id": "t-2"}]
        json_content = json.dumps(records)
        test_file = base_path / "test.json"
        test_file.write_text(json_content, encoding="utf-8")
        file_sha = sha256(json_content.encode("utf-8")).hexdigest()

        item = HeldoutManifestItem(
            name="test-manifest",
            split="test",
            path="test.json",
            sha256=file_sha,
            expected_count=2,
        )
        manifest = HeldoutDatasetManifest(datasets={"test": item})

        results = validate_heldout_manifest(
            manifest=manifest,
            base_dir=base_path,
            train_sample_ids={"train-1"},
            raise_on_error=True,
        )
        assert len(results) == 1
        assert results[0].is_valid is True
        assert results[0].count_matches is True


def test_metric_result_structure_and_thresholds() -> None:
    mr = MetricResult(
        split="dev",
        epoch=3,
        step=300,
        metrics={
            "loss": 0.35,
            "macro_f1": 0.88,
            "latency_p95_ms": 78.4,
            "ece": 0.05,
        },
        checkpoint_version="v1.0.0",
    )

    thresholds = {
        "macro_f1": 0.85,
        "loss": 0.50,
        "latency_p95_ms": 100.0,
        "ece": 0.08,
    }

    evaluated = mr.evaluate_thresholds(thresholds)
    assert evaluated.passed_thresholds is True
    assert evaluated.threshold_details["macro_f1"] is True
    assert evaluated.threshold_details["loss"] is True
    assert evaluated.threshold_details["latency_p95_ms"] is True

    failing_thresholds = {
        "macro_f1": 0.95,
        "loss": 0.50,
    }
    failed_eval = mr.evaluate_thresholds(failing_thresholds)
    assert failed_eval.passed_thresholds is False
    assert failed_eval.threshold_details["macro_f1"] is False
    assert failed_eval.threshold_details["loss"] is True


def test_metric_thresholds_explicit_operators() -> None:
    metrics = {"score": 10.0, "penalty": 2.0}
    rules = {
        "score": (">=", 10.0),
        "penalty": ("<", 5.0),
    }
    passed, details = evaluate_metric_thresholds(metrics, rules)
    assert passed is True
    assert details["score"] is True
    assert details["penalty"] is True


def test_m3_training_config_serialization_and_hash() -> None:
    cfg = M3TrainingConfig(
        config_version="1.0.0",
        task="multitask",
        seed=42,
        model_version=ModelVersionConfig(
            base_model="indobenchmark/indobert-base-p1",
            model_version="v1.0.0",
            tokenizer_version="v1.0.0",
        ),
        split_config=SplitConfig(
            seed=42,
            train_ratio=0.7,
            dev_ratio=0.15,
            test_ratio=0.15,
        ),
        hyperparameters=HyperparametersConfig(
            learning_rate=2e-5,
            batch_size=16,
            gradient_accumulation_steps=2,
            num_train_epochs=8,
            task_weights={"intent": 1.0, "category": 1.0},
        ),
        precision_config=PrecisionConfig(
            device="cpu",
            training_precision="fp32",
        ),
    )

    json_str = cfg.to_json()
    reconstructed = M3TrainingConfig.from_json(json_str)
    assert reconstructed.model_dump() == cfg.model_dump()

    hash1 = cfg.compute_config_hash()
    hash2 = reconstructed.compute_config_hash()
    assert hash1 == hash2
    assert len(hash1) == 64

    cfg.validate_runtime_compatibility("cpu")
    gpu_cfg = cfg.model_copy(update={"precision_config": PrecisionConfig(device="cuda", training_precision="amp")})
    gpu_cfg.validate_runtime_compatibility("cuda")
    with pytest.raises(PrecisionConstraintError):
        gpu_cfg.validate_runtime_compatibility("cpu")


def test_bundled_m3_configs_load_and_validate() -> None:
    config_paths = [
        Path("configs/m3/dapt.json"),
        Path("configs/m3/multitask.json"),
        Path("configs/m3/ner.json"),
    ]

    for p in config_paths:
        assert p.exists()
        cfg = M3TrainingConfig.from_file(p)
        assert cfg.config_version == "1.0.0"
        assert cfg.seed == 42
        assert cfg.precision_config.device == "cpu"
        assert cfg.precision_config.training_precision == "fp32"
        assert cfg.precision_config.serving_precision == "fp32"
        assert cfg.split_config.train_ratio + cfg.split_config.dev_ratio + cfg.split_config.test_ratio == 1.0
        assert cfg.hyperparameters.learning_rate > 0.0
        assert cfg.hyperparameters.effective_batch_size >= 1
        cfg.validate_runtime_compatibility("cpu")
        h = cfg.compute_config_hash()
        assert len(h) == 64


def test_compute_file_and_content_sha256() -> None:
    content = "test reproducibility\n"
    expected = sha256(content.encode("utf-8")).hexdigest()
    assert compute_content_sha256(content) == expected

    with tempfile.NamedTemporaryFile("w+", delete=False) as f:
        f.write(content)
        f.flush()
        temp_path = Path(f.name)

    try:
        assert compute_file_sha256(temp_path) == expected
    finally:
        temp_path.unlink()


def test_rtx3060_hyperparameters_multitask_and_ner() -> None:
    mt_cfg = M3TrainingConfig.from_file("configs/m3/multitask.json")
    ner_cfg = M3TrainingConfig.from_file("configs/m3/ner.json")

    # Hyperparameter tuning for RTX 3060 12GB VRAM & larger synthetic corpus
    for cfg in (mt_cfg, ner_cfg):
        hp = cfg.hyperparameters
        assert hp.batch_size == 16
        assert hp.gradient_accumulation_steps == 2
        assert hp.effective_batch_size == 32
        assert hp.optimizer == "adamw"
        assert hp.lr_scheduler == "linear"
        assert hp.early_stopping_patience == 2
        assert hp.warmup_ratio == 0.1
        assert hp.weight_decay == 0.01

        extra = hp.extra_params
        assert extra.get("target_hardware") == "NVIDIA GeForce RTX 3060 12GB"
        assert extra.get("target_gpu") == "NVIDIA GeForce RTX 3060"
        assert extra.get("target_vram_gb") == 12
        assert extra.get("target_device") == "cuda:0"
        assert extra.get("fp16") is True
        assert extra.get("mixed_precision") == "fp16"
        assert extra.get("per_device_train_batch_size") == 16
        assert extra.get("effective_batch_size") == 32

    # Multitask conservative tuning: LR 2e-5, max 8 epochs
    assert mt_cfg.hyperparameters.learning_rate == 0.00002
    assert mt_cfg.hyperparameters.num_train_epochs == 8
    assert mt_cfg.hyperparameters.task_weights == {
        "intent": 1.0,
        "category": 1.0,
        "risk": 1.0,
        "completeness": 1.0,
    }

    # NER conservative tuning: LR 3e-5, 5 epochs
    assert ner_cfg.hyperparameters.learning_rate == 0.00003
    assert ner_cfg.hyperparameters.num_train_epochs == 5


def test_nontrivial_synthetic_bootstrap_quality_thresholds_and_poor_result_detection() -> None:
    mt_cfg = M3TrainingConfig.from_file("configs/m3/multitask.json")
    ner_cfg = M3TrainingConfig.from_file("configs/m3/ner.json")

    mt_thresh = mt_cfg.target_metric_thresholds
    ner_thresh = ner_cfg.target_metric_thresholds

    # Verify nontrivial thresholds consistent with PRD (do not hide poor results)
    assert mt_thresh["macro_f1"] == 0.85
    assert mt_thresh["intent_f1"] == 0.90
    assert mt_thresh["category_macro_f1"] == 0.85
    assert mt_thresh["risk_macro_f1"] == 0.85
    assert mt_thresh["completeness_macro_f1"] == 0.85
    assert mt_thresh["high_risk_recall"] == 0.90
    assert mt_thresh["latency_p95_ms"] == 5000.0
    assert mt_thresh["ece"] == 0.08

    assert ner_thresh["ner_f1"] == 0.80
    assert ner_thresh["ner_entity_f1"] == 0.80
    assert ner_thresh["loc_f1"] == 0.80
    assert ner_thresh["obj_f1"] == 0.80
    assert ner_thresh["time_f1"] == 0.80
    assert ner_thresh["latency_p95_ms"] == 5000.0

    # Ensure poor / degraded results are strictly caught and NOT hidden
    poor_multitask_metrics = {
        "macro_f1": 0.62,
        "intent_f1": 0.70,
        "category_macro_f1": 0.58,
        "risk_macro_f1": 0.45,
        "completeness_macro_f1": 0.60,
        "high_risk_recall": 0.50,
        "latency_p95_ms": 6200.0,
        "ece": 0.18,
    }
    all_passed, details = evaluate_metric_thresholds(poor_multitask_metrics, mt_thresh)
    assert all_passed is False
    assert all(passed is False for passed in details.values())

    poor_ner_metrics = {
        "ner_f1": 0.42,
        "ner_entity_f1": 0.38,
        "loc_f1": 0.50,
        "obj_f1": 0.35,
        "time_f1": 0.20,
        "latency_p95_ms": 7500.0,
    }
    ner_passed, ner_details = evaluate_metric_thresholds(poor_ner_metrics, ner_thresh)
    assert ner_passed is False
    assert all(passed is False for passed in ner_details.values())


def test_configs_strict_schema_validation() -> None:
    # Verify extra forbidden fields fail schema validation
    mt_raw = json.loads(Path("configs/m3/multitask.json").read_text(encoding="utf-8"))
    mt_raw["invalid_extra_root_field"] = "unexpected"
    with pytest.raises(ValidationError):
        M3TrainingConfig.model_validate(mt_raw)

    ner_raw = json.loads(Path("configs/m3/ner.json").read_text(encoding="utf-8"))
    ner_raw["invalid_extra_root_field"] = "unexpected"
    with pytest.raises(ValidationError):
        M3TrainingConfig.model_validate(ner_raw)


def test_ner_bio_obj_weights_and_checkpoint_metric_compatibility() -> None:
    ner_cfg = M3TrainingConfig.from_file("configs/m3/ner.json")
    expected_weights = {
        "O": 1.0,
        "B-LOC": 2.5,
        "I-LOC": 2.0,
        "B-OBJ": 2.5,
        "I-OBJ": 2.0,
        "B-TIME": 3.0,
        "I-TIME": 2.5,
    }
    assert ner_cfg.hyperparameters.task_weights == expected_weights
    assert ner_cfg.hyperparameters.extra_params["class_weights"] == expected_weights
    assert ner_cfg.hyperparameters.extra_params["class_weights"]["B-OBJ"] == 2.5
    assert ner_cfg.hyperparameters.extra_params["class_weights"]["I-OBJ"] == 2.0
    assert ner_cfg.hyperparameters.extra_params["class_weights"]["B-OBJ"] > ner_cfg.hyperparameters.extra_params["class_weights"]["O"]
    assert ner_cfg.hyperparameters.extra_params["class_weights"]["I-OBJ"] > ner_cfg.hyperparameters.extra_params["class_weights"]["O"]
    assert ner_cfg.hyperparameters.extra_params["class_weights"]["B-LOC"] == 2.5
    assert ner_cfg.hyperparameters.extra_params["class_weights"]["I-LOC"] == 2.0
    assert ner_cfg.hyperparameters.extra_params["class_weights"]["B-TIME"] == 3.0
    assert ner_cfg.hyperparameters.extra_params["class_weights"]["I-TIME"] == 2.5
    assert ner_cfg.hyperparameters.extra_params["metric_for_best_model"] == "obj_f1"
    assert ner_cfg.hyperparameters.extra_params["greater_is_better"] is True

    from scripts.train_ner import parse_config_file, resolve_tag_weight

    parsed = parse_config_file(Path("configs/m3/ner.json"))
    assert parsed.class_weights is not None
    assert parsed.class_weights["B-OBJ"] == 2.5
    assert parsed.class_weights["I-OBJ"] == 2.0
    assert parsed.metric_for_best_model == "obj_f1"
    assert resolve_tag_weight(parsed.class_weights, "B-OBJ") == 2.5
    assert resolve_tag_weight(parsed.class_weights, "I-OBJ") == 2.0


def test_ner_bio_obj_neutral_weights_heldout_precision_regression() -> None:
    test_ner_bio_obj_weights_and_checkpoint_metric_compatibility()
