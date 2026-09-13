from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest.mock as mock
import pytest
from pydantic import ValidationError

from contracts.models import Category, ComplaintTrajectory, DatasetSplit
from scripts import (
    DAPTConfig,
    DatasetAuditError,
    EvaluationConfig,
    M3EvaluationReport,
    MultitaskTrainConfig,
    NERTrainConfig,
    ONNXExportConfig,
    OptionalDependencyError,
    check_ml_dependencies,
    create_artifact_item,
    create_or_update_artifact_manifest,
    require_ml_dependencies,
    run_evaluate_m3,
    run_export_onnx,
    run_train_dapt,
    run_train_multitask,
    run_train_ner,
    set_deterministic_seed,
    validate_dataset_splits,
)
from scripts.train_dapt import (
    compute_dataset_lineage,
    load_corpus,
    parse_dapt_config_dict,
    validate_corpus_text,
)
from scripts.train_multitask import (
    BertForMultiTaskClassification,
    MultiTaskOutput,
    calibrate_temperatures,
    deduplicate_trajectory_multitask_samples,
    extract_trajectory_multitask_samples,
    get_multitask_model_class,
    parse_config_file as parse_multitask_config_file,
    parse_multitask_config_dict,
    train_transformers_multitask,
)
from scripts.train_ner import (
    WeightedNERTrainer,
    _pick_canonical,
    align_spans_to_bio_tags,
    align_spans_with_tokenizer,
    compute_ner_metrics,
    deduplicate_trajectory_ner_samples,
    extract_grounded_object_candidates,
    extract_trajectory_ner_samples,
    get_weighted_ner_trainer_class,
    parse_config_file as parse_ner_config_file,
    parse_ner_config_dict,
    resolve_overlapping_spans,
    resolve_tag_weight,
    train_transformers_ner,
)
from scripts.export_onnx import (
    export_torch_model_to_onnx,
    get_dynamic_axes_and_io_names,
    load_model_for_export,
    parse_onnx_config_dict,
    reconstruct_ner_bert_config,
    validate_onnx_artifact,
    validate_tokenizer_and_numeric_parity,
)
from services.dataset.generator import (
    export_trajectories_to_jsonl,
    generate_dataset,
    generate_trajectory,
)
from services.intelligence.calibration import TemperatureCalibrator, load_temperatures
from services.ml.manifest import ArtifactManifest, compute_file_sha256
from services.ml.runtime import LocalMLRuntime


# ---------------------------------------------------------------------------
# 1. Seed Pinning and Reproducibility Tests
# ---------------------------------------------------------------------------

def test_set_deterministic_seed() -> None:
    set_deterministic_seed(42)
    import random

    val1 = [random.random() for _ in range(5)]
    set_deterministic_seed(42)
    val2 = [random.random() for _ in range(5)]
    assert val1 == val2


def test_optional_dependency_error_structure() -> None:
    err = OptionalDependencyError("torch", purpose="model pre-training")
    assert err.package_name == "torch"
    assert "torch" in str(err)
    assert "requirements-ml.txt" in str(err)
    assert "model pre-training" in str(err)


def test_check_and_require_ml_dependencies() -> None:
    status = check_ml_dependencies("sys", "json", "non_existent_ml_pkg_xyz")
    assert status["sys"] is True
    assert status["json"] is True
    assert status["non_existent_ml_pkg_xyz"] is False

    with pytest.raises(OptionalDependencyError) as exc_info:
        require_ml_dependencies("non_existent_ml_pkg_xyz", purpose="testing")
    assert "non_existent_ml_pkg_xyz" in str(exc_info.value)


# ---------------------------------------------------------------------------
# 2. Configuration Validation Tests
# ---------------------------------------------------------------------------

def test_dapt_config_validation() -> None:
    # Valid config
    cfg = DAPTConfig(seed=123, max_seq_length=512, mlm_probability=0.15)
    assert cfg.seed == 123
    assert cfg.max_seq_length == 512

    # Invalid max_seq_length (< 16 or > 512)
    with pytest.raises(ValidationError):
        DAPTConfig(max_seq_length=10)
    with pytest.raises(ValidationError):
        DAPTConfig(max_seq_length=1024)

    # Invalid learning rate (<= 0)
    with pytest.raises(ValidationError):
        DAPTConfig(learning_rate=-1e-4)

    # Invalid mlm_probability (not in 0.0..1.0)
    with pytest.raises(ValidationError):
        DAPTConfig(mlm_probability=0.0)
    with pytest.raises(ValidationError):
        DAPTConfig(mlm_probability=1.5)


def test_multitask_config_validation() -> None:
    # Valid config
    cfg = MultitaskTrainConfig(learning_rate=3e-5, num_epochs=3)
    assert cfg.learning_rate == 3e-5

    # Invalid negative batch size
    with pytest.raises(ValidationError):
        MultitaskTrainConfig(batch_size=0)

    # Invalid loss weights (missing required head)
    with pytest.raises(ValidationError):
        MultitaskTrainConfig(loss_weights={"intent": 1.0, "category": 1.0})

    # Invalid negative loss weight
    with pytest.raises(ValidationError):
        MultitaskTrainConfig(
            loss_weights={
                "intent": 1.0,
                "category": -0.5,
                "risk": 1.0,
                "completeness": 1.0,
            }
        )


def test_ner_config_validation() -> None:
    # Valid tagset
    cfg = NERTrainConfig(tagset=("O", "B-LOC", "I-LOC"))
    assert "O" in cfg.tagset

    # Missing "O"
    with pytest.raises(ValidationError):
        NERTrainConfig(tagset=("B-LOC", "I-LOC"))

    # Duplicate tag
    with pytest.raises(ValidationError):
        NERTrainConfig(tagset=("O", "B-LOC", "B-LOC"))

    # B-tag without corresponding I-tag
    with pytest.raises(ValidationError):
        NERTrainConfig(tagset=("O", "B-LOC"))


def test_onnx_export_config_validation() -> None:
    # Valid config
    cfg = ONNXExportConfig(output_filename="model.onnx", opset=17, precision="fp32")
    assert cfg.opset == 17

    # Non-fp32 precision rejected (M3 requires strict CPU FP32)
    with pytest.raises(ValidationError):
        ONNXExportConfig(precision="fp16")  # type: ignore

    # Non .onnx filename
    with pytest.raises(ValidationError):
        ONNXExportConfig(output_filename="model.bin")

    # Directory traversal in output_filename rejected
    with pytest.raises(ValidationError):
        ONNXExportConfig(output_filename="../model.onnx")


# ---------------------------------------------------------------------------
# 3. Corpus Validation and PII Scrubbing Tests
# ---------------------------------------------------------------------------

def test_corpus_pii_scrubbing_and_minhash() -> None:
    clean_lines = [
        "Jalan berlubang parah di depan kantor kecamatan sukajadi bandung.",
        "Gorong gorong saluran air mampet meluap ke badan jalan raya.",
        "Tumpukan sampah liar di pinggir jalan alternatif cileunyi.",
    ]
    validated = validate_corpus_text(clean_lines, pii_scrubbing=True, minhash_threshold=0.85)
    assert len(validated) == 3

    # NIK leak detection
    nik_leaked = [
        "Laporan jalan rusak dari warga dengan NIK 3273011234560001 tolong segera diperbaiki."
    ]
    with pytest.raises(ValueError, match="PII violation.*NIK"):
        validate_corpus_text(nik_leaked, pii_scrubbing=True)

    # Phone number leak detection
    phone_leaked = ["Hubungi pelapor di nomor 081234567890 jika tim sudah tiba."]
    with pytest.raises(ValueError, match="PII violation.*Phone"):
        validate_corpus_text(phone_leaked, pii_scrubbing=True)

    # Email leak detection
    email_leaked = ["Kirim bukti perbaikan ke warga@example.com terima kasih."]
    with pytest.raises(ValueError, match="PII violation.*Email"):
        validate_corpus_text(email_leaked, pii_scrubbing=True)

    # MinHash deduplication of near-identical lines
    dup_lines = [
        "jalan berlubang parah di depan kantor kecamatan cibadak sukajadi",
        "jalan berlubang parah di depan kantor kecamatan cibadak sukajadi",
    ]
    deduped = validate_corpus_text(dup_lines, pii_scrubbing=False, minhash_threshold=0.85)
    assert len(deduped) == 1


# ---------------------------------------------------------------------------
# 4. Dataset Split Audit Validation Tests
# ---------------------------------------------------------------------------

def test_dataset_split_audit_clean_and_corrupt() -> None:
    # 1. Clean dataset
    clean_ds = generate_dataset(num_scenarios=36, seed=42)
    audit = validate_dataset_splits(clean_ds)
    assert audit.passed is True
    assert len(audit.violations) == 0
    assert len(audit.oracle_leakages) == 0
    assert len(audit.family_overlap) == 0

    # 2. Corrupt dataset with family leakage across splits
    t1 = generate_trajectory(scenario_id="scen-01", family_id="leak-family", seed=42)
    t2 = generate_trajectory(scenario_id="scen-02", family_id="leak-family", seed=42)
    # Force t1 into TRAIN and t2 into TEST
    t1_corrupt = t1.model_copy(update={"split": DatasetSplit.TRAIN})
    t2_corrupt = t2.model_copy(update={"split": DatasetSplit.TEST})

    corrupt_ds = [t1_corrupt, t2_corrupt]
    with pytest.raises(DatasetAuditError):
        validate_dataset_splits(corrupt_ds, allow_violations=False)


# ---------------------------------------------------------------------------
# 5. Missing Dependency Offline Guard Tests
# ---------------------------------------------------------------------------

def test_safe_optional_dependency_error_on_real_training() -> None:
    orig_import = __import__

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name in {"torch", "transformers", "seqeval", "onnx"}:
            raise ImportError(f"No module named '{name}'")
        return orig_import(name, *args, **kwargs)

    with mock.patch("builtins.__import__", side_effect=fake_import):
        dapt_cfg = DAPTConfig()
        with pytest.raises(OptionalDependencyError):
            run_train_dapt(dapt_cfg, dry_run=False, validate_only=False)

        multitask_cfg = MultitaskTrainConfig()
        with pytest.raises(OptionalDependencyError):
            run_train_multitask(multitask_cfg, dry_run=False, validate_only=False)

        ner_cfg = NERTrainConfig()
        with pytest.raises(OptionalDependencyError):
            run_train_ner(ner_cfg, dry_run=False, validate_only=False)

        onnx_cfg = ONNXExportConfig()
        with pytest.raises(OptionalDependencyError):
            run_export_onnx(onnx_cfg, dry_run=False, validate_only=False)


# ---------------------------------------------------------------------------
# 6. Artifact Directory and Hash Manifest Creation Tests
# ---------------------------------------------------------------------------

def test_manifest_creation_and_runtime_verification() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir)
        dummy_file = out_dir / "model.onnx"
        dummy_file.write_bytes(b"ONNX_DUMMY_BINARY_DATA_TEST_12345")

        item = create_artifact_item(
            file_path=dummy_file,
            name="test-onnx-model",
            version="v1.0.0",
            task="classification",
            precision="fp32",
            relative_to=out_dir,
        )
        assert item.name == "test-onnx-model"
        assert item.path == "model.onnx"
        assert item.size_bytes == len(b"ONNX_DUMMY_BINARY_DATA_TEST_12345")
        assert len(item.sha256) == 64

        manifest = create_or_update_artifact_manifest(
            output_dir=out_dir,
            items=[item],
            manifest_version="1.0.0",
        )
        assert "test-onnx-model" in manifest.artifacts

        # Check manifest file written to disk
        manifest_path = out_dir / "manifest.json"
        assert manifest_path.is_file()

        # Check LocalMLRuntime can load and inspect it
        runtime = LocalMLRuntime(manifest=manifest, artifacts_dir=out_dir, allow_network=False)
        assert runtime.manifest is not None
        validation = runtime.warmup()
        assert "test-onnx-model" in validation
        assert validation["test-onnx-model"].is_valid is True
        assert validation["test-onnx-model"].artifact_name == "test-onnx-model"


# ---------------------------------------------------------------------------
# 7. End-to-End Dry-Run Training Pipeline Tests
# ---------------------------------------------------------------------------

def test_end_to_end_m3_dry_run_pipeline() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        base_dir = Path(tmpdir)

        # 0. Generate dataset JSONL
        ds = generate_dataset(num_scenarios=36, seed=42)
        dataset_jsonl = base_dir / "dataset.jsonl"
        export_trajectories_to_jsonl(ds, dataset_jsonl)
        assert dataset_jsonl.is_file()

        # 1. DAPT dry-run
        dapt_dir = base_dir / "dapt"
        dapt_cfg = DAPTConfig(output_dir=str(dapt_dir), seed=42)
        dapt_manifest = run_train_dapt(dapt_cfg, dry_run=True)
        assert dapt_manifest is not None
        assert "indobert-dapt" in dapt_manifest.artifacts
        assert (dapt_dir / "model.safetensors").is_file()
        assert (dapt_dir / "manifest.json").is_file()

        # 2. Multi-Task dry-run with split audit
        multitask_dir = base_dir / "multitask"
        multitask_cfg = MultitaskTrainConfig(
            dataset_path=str(dataset_jsonl),
            output_dir=str(multitask_dir),
            seed=42,
            audit_splits=True,
        )
        multitask_manifest = run_train_multitask(multitask_cfg, dry_run=True)
        assert multitask_manifest is not None
        assert "indobert-multitask-fp32" in multitask_manifest.artifacts
        assert (multitask_dir / "temperatures.json").is_file()

        # 3. NER dry-run with split audit
        ner_dir = base_dir / "ner"
        ner_cfg = NERTrainConfig(
            dataset_path=str(dataset_jsonl),
            output_dir=str(ner_dir),
            seed=42,
            audit_splits=True,
        )
        ner_manifest = run_train_ner(ner_cfg, dry_run=True)
        assert ner_manifest is not None
        assert "indobert-ner-bio" in ner_manifest.artifacts
        assert (ner_dir / "tagset.json").is_file()

        # 4. ONNX export dry-run and validation
        onnx_dir = base_dir / "onnx"
        onnx_cfg = ONNXExportConfig(
            output_dir=str(onnx_dir),
            output_filename="model.onnx",
            task="multitask",
            precision="fp32",
            validate_output=True,
        )
        onnx_manifest = run_export_onnx(onnx_cfg, dry_run=True)
        assert onnx_manifest is not None
        assert "indobert-multitask-onnx-fp32" in onnx_manifest.artifacts
        assert (onnx_dir / "model.onnx").is_file()

        # 5. Validate-only check on the exported ONNX file
        run_export_onnx(onnx_cfg, validate_only=True)

        # 6. Evaluation on multi-task artifacts
        eval_output = base_dir / "evaluation_results.json"
        eval_cfg = EvaluationConfig(
            manifest_path=str(onnx_dir / "manifest.json"),
            dataset_path=str(dataset_jsonl),
            output_path=str(eval_output),
            seed=42,
            run_benchmark=True,
            target_p95_ms=100.0,
            benchmark_runs=15,
        )
        report: M3EvaluationReport = run_evaluate_m3(eval_cfg)
        assert report.status == "PASSED"
        assert report.all_checks_passed is True
        assert report.sla_conformance is True
        assert report.split_audit["passed"] is True
        assert report.benchmark is not None
        assert report.benchmark["sla_met"] is True
        assert eval_output.is_file()

        # Check report serialization and reloading
        loaded_report = M3EvaluationReport.model_validate_json(eval_output.read_text(encoding="utf-8"))
        assert loaded_report.seed == 42
        assert loaded_report.status == "PASSED"


# ---------------------------------------------------------------------------
# 8. Module and CLI Execution Tests
# ---------------------------------------------------------------------------

def test_cli_execution_as_modules() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        # 1. train_dapt CLI
        res = subprocess.run(
            [
                sys.executable,
                "-m",
                "scripts.train_dapt",
                "--output-dir",
                str(tmp_path / "dapt"),
                "--dry-run",
            ],
            capture_output=True,
            text=True,
        )
        assert res.returncode == 0
        assert (tmp_path / "dapt" / "manifest.json").is_file()

        # 2. train_multitask CLI
        res = subprocess.run(
            [
                sys.executable,
                "-m",
                "scripts.train_multitask",
                "--output-dir",
                str(tmp_path / "multitask"),
                "--dry-run",
            ],
            capture_output=True,
            text=True,
        )
        assert res.returncode == 0
        assert (tmp_path / "multitask" / "manifest.json").is_file()

        # 3. train_ner CLI
        res = subprocess.run(
            [
                sys.executable,
                "-m",
                "scripts.train_ner",
                "--output-dir",
                str(tmp_path / "ner"),
                "--dry-run",
            ],
            capture_output=True,
            text=True,
        )
        assert res.returncode == 0
        assert (tmp_path / "ner" / "manifest.json").is_file()

        # 4. export_onnx CLI
        res = subprocess.run(
            [
                sys.executable,
                "-m",
                "scripts.export_onnx",
                "--output-dir",
                str(tmp_path / "onnx"),
                "--dry-run",
            ],
            capture_output=True,
            text=True,
        )
        assert res.returncode == 0
        assert (tmp_path / "onnx" / "manifest.json").is_file()

        # 5. evaluate_m3 CLI
        res = subprocess.run(
            [
                sys.executable,
                "-m",
                "scripts.evaluate_m3",
                "--manifest-path",
                str(tmp_path / "onnx" / "manifest.json"),
                "--output-path",
                str(tmp_path / "eval.json"),
                "--seed",
                "42",
            ],
            capture_output=True,
            text=True,
        )
        assert res.returncode == 0
        assert (tmp_path / "eval.json").is_file()


def test_cli_execution_directly_as_scripts() -> None:
    repo_root_path = Path(__file__).resolve().parent.parent

    # Test invoking scripts directly (e.g. python scripts/train_dapt.py --dry-run)
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        script_file = repo_root_path / "scripts" / "train_dapt.py"
        res = subprocess.run(
            [
                sys.executable,
                str(script_file),
                "--output-dir",
                str(tmp_path / "dapt"),
                "--dry-run",
            ],
            capture_output=True,
            text=True,
        )
        assert res.returncode == 0


def test_cli_missing_dependency_exit_code_1() -> None:
    # When invoking without --dry-run in environment without torch, CLI must exit 1
    res = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; sys.modules['torch'] = None; from scripts.train_multitask import main; sys.exit(main())",
        ],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 1
    assert "Optional Dependency Error" in res.stderr
    assert "requirements-ml.txt" in res.stderr


def test_cli_config_file_loading() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        cfg_file = tmp_path / "config.json"
        cfg_dict = {
            "model_name_or_path": "indobenchmark/indobert-base-p1",
            "output_dir": str(tmp_path / "dapt_from_file"),
            "seed": 99,
            "max_seq_length": 256,
            "mlm_probability": 0.2,
            "learning_rate": 1e-4,
            "batch_size": 8,
            "num_epochs": 2,
            "warmup_ratio": 0.05,
            "weight_decay": 0.02,
            "fp16": False,
            "minhash_threshold": 0.8,
            "pii_scrubbing": True,
            "version": "v1.1.0",
        }
        cfg_file.write_text(json.dumps(cfg_dict), encoding="utf-8")

        res = subprocess.run(
            [
                sys.executable,
                "-m",
                "scripts.train_dapt",
                "--config",
                str(cfg_file),
                "--dry-run",
            ],
            capture_output=True,
            text=True,
        )
        assert res.returncode == 0
        manifest_file = tmp_path / "dapt_from_file" / "manifest.json"
        assert manifest_file.is_file()
        with open(manifest_file, "r") as f:
            data = json.load(f)
        assert data["artifacts"]["indobert-dapt"]["version"] == "v1.1.0"


def test_dapt_validate_only_returns_none() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir) / "dapt_val"
        cfg = DAPTConfig(output_dir=str(out_dir))
        res = run_train_dapt(cfg, validate_only=True)
        assert res is None
        assert not out_dir.exists()


def test_load_corpus_from_various_sources() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        # 1. From ComplaintTrajectory JSONL
        ds = generate_dataset(num_scenarios=5, seed=42)
        jsonl_path = tmp_path / "trajectories.jsonl"
        export_trajectories_to_jsonl(ds, jsonl_path)
        corpus_jsonl = load_corpus(jsonl_path, seed=42)
        assert len(corpus_jsonl) > 0
        assert all(isinstance(line, str) for line in corpus_jsonl)

        # 2. From generic JSONL {"text": "..."}
        generic_jsonl = tmp_path / "generic.jsonl"
        with generic_jsonl.open("w", encoding="utf-8") as f:
            f.write(json.dumps({"text": "Jalan raya di kawasan sudirman tergenang air banjir setinggi lutut"}) + "\n")
            f.write(json.dumps({"content": "Penerangan jalan umum mati di flyover pasupati bandung"}) + "\n")
        corpus_generic = load_corpus(generic_jsonl)
        assert len(corpus_generic) == 2

        # 3. From plain text file
        txt_path = tmp_path / "plain.txt"
        txt_path.write_text("Tumpukan sampah basah di pasar tradisional belum diangkut tiga hari\n", encoding="utf-8")
        corpus_txt = load_corpus(txt_path)
        assert len(corpus_txt) == 1

        # 4. Fallback to synthetic generator (corpus_path=None)
        corpus_synth = load_corpus(None, seed=42)
        assert len(corpus_synth) >= len(ds)

        # 5. Non-existent file raises FileNotFoundError
        with pytest.raises(FileNotFoundError):
            load_corpus(tmp_path / "non_existent.txt")


def test_dapt_config_parsing_from_m3_format() -> None:
    m3_cfg_path = Path("configs/m3/dapt.json")
    with m3_cfg_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    dapt_cfg = parse_dapt_config_dict(data)
    assert dapt_cfg.model_name_or_path == "indobenchmark/indobert-base-p1"
    assert dapt_cfg.batch_size == 8
    assert dapt_cfg.gradient_accumulation_steps == 4
    assert dapt_cfg.learning_rate == 0.00002
    assert dapt_cfg.fp16 is True
    assert dapt_cfg.local_files_only is False
    assert dapt_cfg.dataloader_num_workers == 2


def test_dapt_cli_with_m3_config_dry_run() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        out_dir = tmp_path / "dapt_m3_out"
        res = subprocess.run(
            [
                sys.executable,
                "-m",
                "scripts.train_dapt",
                "--config",
                "configs/m3/dapt.json",
                "--output-dir",
                str(out_dir),
                "--dry-run",
            ],
            capture_output=True,
            text=True,
        )
        assert res.returncode == 0
        assert (out_dir / "manifest.json").is_file()
        assert (out_dir / "model.safetensors").is_file()
        with open(out_dir / "manifest.json") as f:
            manifest_data = json.load(f)
        assert "indobert-dapt" in manifest_data["artifacts"]


def test_huggingface_dapt_execution_flow(monkeypatch: pytest.MonkeyPatch) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir) / "hf_dapt_out"

        mock_torch = mock.MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.utils.data.Dataset = object
        mock_torch.tensor = lambda x, **kw: x
        mock_torch.long = "long"

        mock_tokenizer_instance = mock.MagicMock()
        mock_tokenizer_instance.return_value = {
            "input_ids": [[101, 1000, 102]],
            "attention_mask": [[1, 1, 1]],
            "special_tokens_mask": [[1, 0, 1]],
        }

        def mock_save_pretrained(save_directory: str) -> None:
            p = Path(save_directory)
            p.mkdir(parents=True, exist_ok=True)
            (p / "tokenizer_config.json").write_text("{}", encoding="utf-8")
            (p / "vocab.txt").write_text("[PAD]\n", encoding="utf-8")

        mock_tokenizer_instance.save_pretrained.side_effect = mock_save_pretrained

        mock_model_instance = mock.MagicMock()

        mock_trainer_instance = mock.MagicMock()

        def mock_save_model(output_dir: str) -> None:
            p = Path(output_dir)
            p.mkdir(parents=True, exist_ok=True)
            (p / "model.safetensors").write_bytes(b"TRAINED_MOCK_SAFETENSORS_WEIGHTS" + b"\x00" * 64)
            (p / "config.json").write_text("{}", encoding="utf-8")

        mock_trainer_instance.save_model.side_effect = mock_save_model

        mock_transformers = mock.MagicMock()
        mock_transformers.AutoTokenizer.from_pretrained.return_value = mock_tokenizer_instance
        mock_transformers.AutoModelForMaskedLM.from_pretrained.return_value = mock_model_instance
        mock_transformers.Trainer.return_value = mock_trainer_instance

        monkeypatch.setitem(sys.modules, "torch", mock_torch)
        monkeypatch.setitem(sys.modules, "torch.utils", mock_torch.utils)
        monkeypatch.setitem(sys.modules, "torch.utils.data", mock_torch.utils.data)
        monkeypatch.setitem(sys.modules, "transformers", mock_transformers)

        config = DAPTConfig(
            model_name_or_path="indobenchmark/indobert-base-p1",
            output_dir=str(out_dir),
            seed=42,
            max_seq_length=448,
            learning_rate=2e-5,
            batch_size=8,
            gradient_accumulation_steps=4,
            num_epochs=3,
            fp16=True,
            local_files_only=True,
            dataloader_num_workers=2,
            version="v1.0.0",
        )

        manifest = run_train_dapt(config, dry_run=False, validate_only=False)

        assert manifest is not None
        assert "indobert-dapt" in manifest.artifacts
        assert manifest.artifacts["indobert-dapt"].precision == "fp16"
        assert (out_dir / "model.safetensors").is_file()
        assert (out_dir / "manifest.json").is_file()

        # Verify AutoTokenizer and AutoModelForMaskedLM called with local_files_only
        mock_transformers.AutoTokenizer.from_pretrained.assert_called_once_with(
            "indobenchmark/indobert-base-p1",
            local_files_only=True,
        )
        mock_transformers.AutoModelForMaskedLM.from_pretrained.assert_called_once_with(
            "indobenchmark/indobert-base-p1",
            local_files_only=True,
        )

        # Verify DataCollatorForLanguageModeling arguments
        mock_transformers.DataCollatorForLanguageModeling.assert_called_once_with(
            tokenizer=mock_tokenizer_instance,
            mlm=True,
            mlm_probability=0.15,
        )

        # Verify TrainingArguments suitable for RTX 3060 12GB fp16
        training_args_call = mock_transformers.TrainingArguments.call_args
        assert training_args_call is not None
        args_kwargs = training_args_call.kwargs
        assert args_kwargs["output_dir"] == str(out_dir)
        assert args_kwargs["per_device_train_batch_size"] == 8
        assert args_kwargs["gradient_accumulation_steps"] == 4
        assert args_kwargs["fp16"] is True
        assert args_kwargs["learning_rate"] == 2e-5
        assert args_kwargs["dataloader_num_workers"] == 2
        assert args_kwargs["report_to"] == "none"

        # Verify Trainer was trained and saved
        mock_trainer_instance.train.assert_called_once()
        mock_trainer_instance.save_model.assert_called_once_with(str(out_dir))
        mock_tokenizer_instance.save_pretrained.assert_called_once_with(str(out_dir))


# ---------------------------------------------------------------------------
# 9. NER Training & BIO Alignment Unit Tests
# ---------------------------------------------------------------------------

def test_ner_trajectory_span_extraction_and_alignment() -> None:
    ds = generate_dataset(num_scenarios=5, seed=42)
    samples = extract_trajectory_ner_samples(ds)
    assert len(samples) > 0

    # Ensure each sample contains valid text and spans
    for sample in samples:
        assert isinstance(sample["text"], str)
        assert len(sample["text"]) > 0
        assert isinstance(sample["spans"], list)
        for s, e, label in sample["spans"]:
            assert 0 <= s < e <= len(sample["text"])
            assert label in ("LOC", "OBJ", "TIME")

    # Test BIO tag alignment on first sample
    first_sample = samples[0]
    words, tags, offsets = align_spans_to_bio_tags(first_sample["text"], first_sample["spans"])
    assert len(words) == len(tags) == len(offsets)
    assert all(t in NERTrainConfig().tagset for t in tags)

    # Test resolve overlapping spans
    raw_spans = [(10, 25, "LOC"), (15, 30, "OBJ"), (40, 50, "TIME")]
    resolved = resolve_overlapping_spans(raw_spans)
    # Longest span (15 chars: 10..25 or 15..30) selected, non-overlapping (40..50) preserved
    assert len(resolved) == 2
    assert resolved[1] == (40, 50, "TIME")


def test_ner_tokenizer_alignment_mapping() -> None:
    mock_tok = mock.MagicMock()
    # Simulate tokenizer output with offset_mapping
    text = "Lapor jalan berlubang di Bandung"
    # Offsets: [CLS], "Lapor", "jalan", "berlubang", "di", "Bandung", [SEP]
    mock_tok.return_value = {
        "input_ids": [101, 1001, 1002, 1003, 1004, 1005, 102],
        "attention_mask": [1, 1, 1, 1, 1, 1, 1],
        "token_type_ids": [0, 0, 0, 0, 0, 0, 0],
        "offset_mapping": [(0, 0), (0, 5), (6, 11), (12, 21), (22, 24), (25, 32), (0, 0)],
    }
    label2id = {"O": 0, "B-OBJ": 1, "I-OBJ": 2, "B-LOC": 3, "I-LOC": 4}
    spans = [(6, 21, "OBJ"), (25, 32, "LOC")]

    encoding = align_spans_with_tokenizer(mock_tok, text, spans, label2id, max_seq_length=16)
    assert encoding["labels"][0] == -100  # [CLS]
    assert encoding["labels"][1] == 0  # "Lapor" -> O
    assert encoding["labels"][2] == 1  # "jalan" -> B-OBJ
    assert encoding["labels"][3] == 2  # "berlubang" -> I-OBJ
    assert encoding["labels"][4] == 0  # "di" -> O
    assert encoding["labels"][5] == 3  # "Bandung" -> B-LOC
    assert encoding["labels"][6] == -100  # [SEP]


def test_ner_no_full_complaint_clauses_as_obj() -> None:
    ds = generate_dataset(num_scenarios=36, seed=42)
    samples = extract_trajectory_ner_samples(ds)
    assert len(samples) > 0

    full_issue_clauses = {traj.observable_facts[0].strip().lower() for traj in ds if traj.observable_facts}
    assert len(full_issue_clauses) > 0

    obj_spans_found: list[str] = []
    for sample in samples:
        text = sample["text"]
        for s, e, label in sample["spans"]:
            if label == "OBJ":
                span_text = text[s:e].strip()
                obj_spans_found.append(span_text)
                # Ensure span is not an entire complaint clause
                assert span_text.lower() not in full_issue_clauses, (
                    f"Full complaint clause labeled as OBJ: '{span_text}'"
                )
                # Ensure noun-phrase is compact (max 7 words, e.g. 'stok obat rutin diabetes dan darah tinggi')
                words = span_text.split()
                assert len(words) <= 7, f"OBJ span exceeds compact noun phrase limit: '{span_text}'"
                assert len(span_text) <= 50, f"OBJ span length excessive: '{span_text}'"
                # Ensure predicate clauses with verbs are not labeled as OBJ
                for bad_predicate in ("tidak diangkut", "gak diangkut", "tidak berada", "tertunda tiga bulan", "terhenti total"):
                    assert bad_predicate not in span_text.lower(), (
                        f"Complaint clause verb phrase detected in OBJ: '{span_text}'"
                    )

    assert len(obj_spans_found) > 0
    avg_words = sum(len(s.split()) for s in obj_spans_found) / len(obj_spans_found)
    assert avg_words <= 4.0


def test_ner_preserves_loc_and_time_targets() -> None:
    ds = generate_dataset(num_scenarios=36, seed=42)
    samples = extract_trajectory_ner_samples(ds)

    loc_count = 0
    time_count = 0
    obj_count = 0

    for sample in samples:
        for _, _, label in sample["spans"]:
            if label == "LOC":
                loc_count += 1
            elif label == "TIME":
                time_count += 1
            elif label == "OBJ":
                obj_count += 1

    assert loc_count > 0, "No LOC targets extracted"
    assert time_count > 0, "No TIME targets extracted; temporal expressions were lost"
    assert obj_count > 0, "No OBJ targets extracted"

    # Specific check: samples containing 'dua minggu' must retain it as TIME, not overwritten by OBJ
    found_dua_minggu_as_time = False
    for sample in samples:
        text = sample["text"]
        for s, e, label in sample["spans"]:
            span_text = text[s:e].lower()
            if "dua minggu" in span_text:
                assert label == "TIME", f"'dua minggu' was tagged as {label} instead of TIME"
                found_dua_minggu_as_time = True
    assert found_dua_minggu_as_time, "'dua minggu' temporal target not found in dataset samples"


def test_ner_resolve_overlap_nested_spans_preserves_all_useful_labels() -> None:
    # 1. TIME nested inside broad OBJ span
    raw_spans = [(10, 50, "OBJ"), (20, 30, "TIME")]
    resolved = resolve_overlapping_spans(raw_spans)
    assert resolved == [(10, 20, "OBJ"), (20, 30, "TIME"), (30, 50, "OBJ")]

    # 2. LOC and TIME nested inside broad OBJ span
    raw_spans = [(0, 60, "OBJ"), (10, 25, "LOC"), (40, 50, "TIME")]
    resolved = resolve_overlapping_spans(raw_spans)
    assert resolved == [(0, 10, "OBJ"), (10, 25, "LOC"), (25, 40, "OBJ"), (40, 50, "TIME"), (50, 60, "OBJ")]

    # 3. Inner span at start
    raw_spans = [(10, 50, "OBJ"), (10, 25, "LOC")]
    resolved = resolve_overlapping_spans(raw_spans)
    assert resolved == [(10, 25, "LOC"), (25, 50, "OBJ")]

    # 4. Inner span at end
    raw_spans = [(10, 50, "OBJ"), (35, 50, "TIME")]
    resolved = resolve_overlapping_spans(raw_spans)
    assert resolved == [(10, 35, "OBJ"), (35, 50, "TIME")]

    # 5. Multiple disjoint inner spans of same label inside broad span replaces outer span
    raw_spans = [(10, 60, "OBJ"), (10, 25, "OBJ"), (40, 55, "OBJ")]
    resolved = resolve_overlapping_spans(raw_spans)
    assert resolved == [(10, 25, "OBJ"), (40, 55, "OBJ")]

    # 6. Nested resolution with text trimming
    text = "Lapor: ada sampah dua minggu menumpuk di jalan"
    raw_spans = [(11, 37, "OBJ"), (18, 28, "TIME")]
    resolved = resolve_overlapping_spans(raw_spans, text=text)
    assert resolved == [(11, 17, "OBJ"), (18, 28, "TIME"), (29, 37, "OBJ")]
    assert text[11:17] == "sampah"
    assert text[18:28] == "dua minggu"
    assert text[29:37] == "menumpuk"


def test_ner_bio_clean_exact_boundary_targets() -> None:
    text = "Lapor! Ada jalan berlubang di Bandung, tgl 12."
    spans = [(11, 26, "OBJ"), (30, 37, "LOC")]
    words, tags, offsets = align_spans_to_bio_tags(text, spans)

    assert len(words) == len(tags) == len(offsets)
    # Check exact offsets
    for w, (ws, we) in zip(words, offsets):
        assert text[ws:we] == w
        # Punctuation must be cleanly stripped from word boundaries
        assert not any(p in w for p in ("!", ",", "."))

    # Verify correct BIO sequence
    assert "B-OBJ" in tags
    assert "I-OBJ" in tags
    assert "B-LOC" in tags
    # Verify BIO consistency: no I- tag without prior B- or I- of same entity
    prev_tag = "O"
    for tag in tags:
        if tag.startswith("I-"):
            assert prev_tag in (f"B-{tag[2:]}", f"I-{tag[2:]}")
        prev_tag = tag

    # Verify adjacent separate entities of same type both receive B-
    adj_text = "sampah tps liar"
    adj_spans = [(0, 6, "OBJ"), (7, 10, "OBJ")]
    adj_words, adj_tags, _ = align_spans_to_bio_tags(adj_text, adj_spans)
    assert adj_words == ["sampah", "tps", "liar"]
    assert adj_tags == ["B-OBJ", "B-OBJ", "O"]


def test_ner_deterministic_generation_and_extraction() -> None:
    ds1 = generate_dataset(num_scenarios=10, seed=42)
    ds2 = generate_dataset(num_scenarios=10, seed=42)
    s1 = extract_trajectory_ner_samples(ds1)
    s2 = extract_trajectory_ner_samples(ds2)

    assert len(s1) == len(s2)
    assert s1 == s2


def test_ner_rtx3060_config_parsing() -> None:
    ner_cfg_path = Path("configs/m3/ner.json")
    assert ner_cfg_path.is_file()
    cfg = parse_ner_config_file(ner_cfg_path)
    assert cfg.model_name_or_path == "indobenchmark/indobert-base-p1"
    assert cfg.batch_size == 16
    assert cfg.gradient_accumulation_steps == 2
    assert cfg.fp16 is True
    assert cfg.max_seq_length == 448
    assert cfg.learning_rate == 0.00003
    assert cfg.dataloader_num_workers == 2
    assert cfg.local_files_only is True
    assert "B-LOC" in cfg.tagset
    assert "B-OBJ" in cfg.tagset


def test_ner_cli_with_m3_config_dry_run() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        out_dir = tmp_path / "ner_m3_out"
        res = subprocess.run(
            [
                sys.executable,
                "-m",
                "scripts.train_ner",
                "--config",
                "configs/m3/ner.json",
                "--output-dir",
                str(out_dir),
                "--dry-run",
            ],
            capture_output=True,
            text=True,
        )
        assert res.returncode == 0
        assert (out_dir / "manifest.json").is_file()
        assert (out_dir / "tagset.json").is_file()
        assert (out_dir / "model.safetensors").is_file()
        with open(out_dir / "manifest.json") as f:
            manifest_data = json.load(f)
        assert "indobert-ner-bio" in manifest_data["artifacts"]


def test_huggingface_ner_execution_flow(monkeypatch: pytest.MonkeyPatch) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir) / "hf_ner_out"

        mock_torch = mock.MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.utils.data.Dataset = object
        mock_torch.tensor = lambda x, **kw: x
        mock_torch.long = "long"

        mock_tokenizer_instance = mock.MagicMock()
        mock_tokenizer_instance.return_value = {
            "input_ids": [101, 1000, 102],
            "attention_mask": [1, 1, 1],
            "token_type_ids": [0, 0, 0],
            "offset_mapping": [(0, 0), (0, 4), (0, 0)],
        }

        def mock_save_pretrained(save_directory: str) -> None:
            p = Path(save_directory)
            p.mkdir(parents=True, exist_ok=True)
            (p / "tokenizer_config.json").write_text("{}", encoding="utf-8")

        mock_tokenizer_instance.save_pretrained.side_effect = mock_save_pretrained

        mock_model_instance = mock.MagicMock()
        mock_trainer_instance = mock.MagicMock()

        def mock_save_model(output_dir: str) -> None:
            p = Path(output_dir)
            (p / "model.safetensors").write_bytes(b"TRAINED_MOCK_NER_SAFETENSORS" + b"\x00" * 64)
            (p / "config.json").write_text("{}", encoding="utf-8")

        mock_trainer_instance.save_model.side_effect = mock_save_model

        mock_transformers = mock.MagicMock()
        mock_transformers.AutoTokenizer.from_pretrained.return_value = mock_tokenizer_instance
        mock_transformers.AutoModelForTokenClassification.from_pretrained.return_value = mock_model_instance
        mock_transformers.Trainer.return_value = mock_trainer_instance

        monkeypatch.setitem(sys.modules, "torch", mock_torch)
        monkeypatch.setitem(sys.modules, "torch.utils", mock_torch.utils)
        monkeypatch.setitem(sys.modules, "torch.utils.data", mock_torch.utils.data)
        monkeypatch.setitem(sys.modules, "transformers", mock_transformers)

        config = NERTrainConfig(
            model_name_or_path="indobenchmark/indobert-base-p1",
            output_dir=str(out_dir),
            seed=42,
            max_seq_length=448,
            learning_rate=3e-5,
            batch_size=16,
            gradient_accumulation_steps=2,
            num_epochs=5,
            fp16=True,
            local_files_only=True,
            dataloader_num_workers=2,
            version="v1.0.0",
        )

        manifest = run_train_ner(config, dry_run=False, validate_only=False)

        assert manifest is not None
        assert "indobert-ner-bio" in manifest.artifacts
        assert manifest.artifacts["indobert-ner-bio"].precision == "fp16"
        assert (out_dir / "model.safetensors").is_file()
        assert (out_dir / "tagset.json").is_file()
        assert (out_dir / "manifest.json").is_file()

        # Verify AutoTokenizer and AutoModelForTokenClassification called with local_files_only
        mock_transformers.AutoTokenizer.from_pretrained.assert_called_once_with(
            "indobenchmark/indobert-base-p1",
            local_files_only=True,
        )
        mock_transformers.AutoModelForTokenClassification.from_pretrained.assert_called_once_with(
            "indobenchmark/indobert-base-p1",
            num_labels=len(config.tagset),
            id2label=mock.ANY,
            label2id=mock.ANY,
            local_files_only=True,
        )

        # Verify TrainingArguments suitable for RTX 3060 12GB fp16
        training_args_call = mock_transformers.TrainingArguments.call_args
        assert training_args_call is not None
        args_kwargs = training_args_call.kwargs
        assert args_kwargs["output_dir"] == str(out_dir)
        assert args_kwargs["per_device_train_batch_size"] == 16
        assert args_kwargs["gradient_accumulation_steps"] == 2
        assert args_kwargs["fp16"] is True
        assert args_kwargs["learning_rate"] == 3e-5
        assert args_kwargs["dataloader_num_workers"] == 2
        assert args_kwargs["report_to"] == "none"

        # Verify Trainer was trained and saved
        mock_trainer_instance.train.assert_called_once()
        mock_trainer_instance.save_model.assert_called_once_with(str(out_dir))
        mock_tokenizer_instance.save_pretrained.assert_called_once_with(str(out_dir))


# ---------------------------------------------------------------------------
# 10. ONNX Export & Dynamic Axes Validation Tests
# ---------------------------------------------------------------------------

def test_onnx_dynamic_axes_and_io_names_all_tasks() -> None:
    # 1. NER
    ner_axes, ner_inputs, ner_outputs = get_dynamic_axes_and_io_names("ner")
    assert ner_inputs == ["input_ids", "attention_mask"]
    assert ner_outputs == ["logits"]
    assert ner_axes["input_ids"] == {0: "batch_size", 1: "sequence_length"}
    assert ner_axes["logits"] == {0: "batch_size", 1: "sequence_length"}

    # 2. Multi-Task
    mt_axes, mt_inputs, mt_outputs = get_dynamic_axes_and_io_names("multitask")
    assert mt_inputs == ["input_ids", "attention_mask"]
    assert "intent_logits" in mt_outputs
    assert "category_logits" in mt_outputs
    assert "risk_logits" in mt_outputs
    assert "completeness_logits" in mt_outputs
    assert mt_axes["intent_logits"] == {0: "batch_size"}

    # 3. Classification
    cls_axes, cls_inputs, cls_outputs = get_dynamic_axes_and_io_names("classification")
    assert cls_inputs == ["input_ids", "attention_mask"]
    assert cls_outputs == ["logits"]
    assert cls_axes["logits"] == {0: "batch_size"}


def test_onnx_artifact_validation_details() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        # 1. Non-existent file
        with pytest.raises(FileNotFoundError):
            validate_onnx_artifact(tmp_path / "non_existent.onnx")

        # 2. Empty file
        empty_file = tmp_path / "empty.onnx"
        empty_file.write_bytes(b"")
        with pytest.raises(ValueError, match="empty"):
            validate_onnx_artifact(empty_file)

        # 3. Skeleton payload
        skeleton_file = tmp_path / "skeleton.onnx"
        skeleton_file.write_bytes(b"\x08\x07\x12\nKAWAL_ONNX_TEST_PAYLOAD_123")
        res = validate_onnx_artifact(skeleton_file)
        assert res["is_skeleton"] is True
        assert res["size_bytes"] == len(b"\x08\x07\x12\nKAWAL_ONNX_TEST_PAYLOAD_123")
        assert len(res["sha256"]) == 64


def test_onnx_torch_export_simulation_no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir)
        target_onnx = out_dir / "exported_model.onnx"

        mock_torch = mock.MagicMock()
        mock_torch.float32 = "float32"
        mock_torch.long = "long"
        mock_torch.zeros.return_value = mock.MagicMock()
        mock_torch.ones.return_value = mock.MagicMock()

        monkeypatch.setitem(sys.modules, "torch", mock_torch)
        monkeypatch.setitem(sys.modules, "torch.onnx", mock_torch.onnx)

        mock_model = mock.MagicMock()

        # Run export
        result_path = export_torch_model_to_onnx(
            model=mock_model,
            target_path=target_onnx,
            task="ner",
            max_seq_length=448,
            opset=17,
        )
        assert result_path == target_onnx

        mock_model.eval.assert_called_once()
        mock_model.to.assert_called_once_with(dtype="float32")
        mock_torch.onnx.export.assert_called_once()

        export_call = mock_torch.onnx.export.call_args
        assert export_call is not None
        call_args, call_kwargs = export_call
        assert call_args[0] == mock_model
        assert call_args[2] == str(target_onnx)
        assert call_kwargs["opset_version"] == 17
        assert call_kwargs["do_constant_folding"] is True
        assert call_kwargs["dynamic_axes"]["input_ids"] == {0: "batch_size", 1: "sequence_length"}
        assert call_kwargs["dynamic_axes"]["logits"] == {0: "batch_size", 1: "sequence_length"}


def test_onnx_export_cli_ner_task_and_validate_only() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        out_dir = tmp_path / "onnx_ner_out"

        # 1. Export NER dry-run CLI
        res = subprocess.run(
            [
                sys.executable,
                "-m",
                "scripts.export_onnx",
                "--output-dir",
                str(out_dir),
                "--task",
                "ner",
                "--dry-run",
            ],
            capture_output=True,
            text=True,
        )
        assert res.returncode == 0
        onnx_file = out_dir / "model.onnx"
        assert onnx_file.is_file()

        with open(out_dir / "manifest.json") as f:
            manifest_data = json.load(f)
        assert "indobert-ner-onnx-fp32" in manifest_data["artifacts"]

        # 2. Validate-only CLI on exported file
        res_val = subprocess.run(
            [
                sys.executable,
                "-m",
                "scripts.export_onnx",
                "--output-dir",
                str(out_dir),
                "--validate-only",
            ],
            capture_output=True,
            text=True,
        )
        assert res_val.returncode == 0


# ---------------------------------------------------------------------------
# 11. Multi-Task Training and RTX 3060 FP16 Tests
# ---------------------------------------------------------------------------

def test_multitask_rtx3060_config_parsing() -> None:
    mt_cfg_path = Path("configs/m3/multitask.json")
    assert mt_cfg_path.is_file()
    cfg = parse_multitask_config_file(mt_cfg_path)
    assert cfg.model_name_or_path == "indobenchmark/indobert-base-p1"
    assert cfg.batch_size == 16
    assert cfg.gradient_accumulation_steps == 2
    assert cfg.fp16 is True
    assert cfg.max_seq_length == 448
    assert cfg.learning_rate == 0.00002
    assert cfg.num_epochs == 8
    assert cfg.dataloader_num_workers == 2
    assert cfg.local_files_only is True
    assert cfg.temperature_scaling is True
    assert cfg.loss_weights["intent"] == 1.0
    assert cfg.loss_weights["category"] == 1.0
    assert cfg.loss_weights["risk"] == 1.0
    assert cfg.loss_weights["completeness"] == 1.0


def test_multitask_cli_with_m3_config_dry_run() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        out_dir = tmp_path / "multitask_m3_out"
        res = subprocess.run(
            [
                sys.executable,
                "-m",
                "scripts.train_multitask",
                "--config",
                "configs/m3/multitask.json",
                "--output-dir",
                str(out_dir),
                "--dry-run",
            ],
            capture_output=True,
            text=True,
        )
        assert res.returncode == 0
        assert (out_dir / "manifest.json").is_file()
        assert (out_dir / "config.json").is_file()
        assert (out_dir / "temperatures.json").is_file()
        assert (out_dir / "model.safetensors").is_file()
        with open(out_dir / "manifest.json") as f:
            manifest_data = json.load(f)
        assert "indobert-multitask-fp32" in manifest_data["artifacts"]
        assert "multitask-temperatures" in manifest_data["artifacts"]

        with open(out_dir / "config.json") as f:
            cfg_data = json.load(f)
        assert cfg_data["architectures"] == ["BertForMultiTaskClassification"]
        assert "intent" in cfg_data["heads"]
        assert "category" in cfg_data["heads"]
        assert "risk" in cfg_data["heads"]
        assert "completeness" in cfg_data["heads"]


def test_multitask_sample_extraction() -> None:
    trajectories = generate_dataset(num_scenarios=6, seed=42)
    samples = extract_trajectory_multitask_samples(trajectories)
    assert len(samples) >= 6
    for s in samples:
        assert "text" in s and len(s["text"]) > 0
        assert s["intent"] in ("COMPLAINT", "INQUIRY", "FEEDBACK")
        assert s["category"] in (
            "ROAD",
            "DRAINAGE_FLOOD",
            "WASTE",
            "CLEAN_WATER",
            "CIVIL_ADMIN",
            "HEALTH_SERVICE",
        )
        assert s["risk"] in ("LOW", "MEDIUM", "HIGH", "URGENT")
        assert s["completeness"] in ("SUFFICIENT", "INCOMPLETE", "AMBIGUOUS")
        assert s["split"] in ("train", "dev", "test")
        assert "scenario_id" in s


def test_multitask_validate_only() -> None:
    cfg = MultitaskTrainConfig()
    result = run_train_multitask(cfg, validate_only=True)
    assert result is None


def test_huggingface_multitask_execution_flow(monkeypatch: pytest.MonkeyPatch) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir) / "hf_multitask_out"

        mock_torch = mock.MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.utils.data.Dataset = object
        mock_torch.tensor = lambda x, **kw: x
        mock_torch.long = "long"
        mock_torch.float = "float"

        mock_tokenizer_instance = mock.MagicMock()
        mock_tokenizer_instance.return_value = {
            "input_ids": [101, 1000, 102],
            "attention_mask": [1, 1, 1],
            "token_type_ids": [0, 0, 0],
        }

        def mock_save_pretrained(save_directory: str) -> None:
            p = Path(save_directory)
            p.mkdir(parents=True, exist_ok=True)
            (p / "tokenizer_config.json").write_text("{}", encoding="utf-8")

        mock_tokenizer_instance.save_pretrained.side_effect = mock_save_pretrained

        mock_encoder = mock.MagicMock()
        mock_encoder.config.hidden_size = 768
        mock_encoder.config.hidden_dropout_prob = 0.1

        mock_trainer_instance = mock.MagicMock()

        def mock_save_model(output_dir: str) -> None:
            p = Path(output_dir)
            (p / "model.safetensors").write_bytes(b"TRAINED_MOCK_MULTITASK_SAFETENSORS" + b"\x00" * 64)
            (p / "config.json").write_text("{}", encoding="utf-8")

        mock_trainer_instance.save_model.side_effect = mock_save_model

        mock_transformers = mock.MagicMock()
        mock_transformers.AutoTokenizer.from_pretrained.return_value = mock_tokenizer_instance
        mock_transformers.AutoModel.from_pretrained.return_value = mock_encoder
        mock_transformers.Trainer.return_value = mock_trainer_instance

        monkeypatch.setitem(sys.modules, "torch", mock_torch)
        monkeypatch.setitem(sys.modules, "torch.nn", mock_torch.nn)
        monkeypatch.setitem(sys.modules, "torch.utils", mock_torch.utils)
        monkeypatch.setitem(sys.modules, "torch.utils.data", mock_torch.utils.data)
        monkeypatch.setitem(sys.modules, "transformers", mock_transformers)

        config = MultitaskTrainConfig(
            model_name_or_path="indobenchmark/indobert-base-p1",
            output_dir=str(out_dir),
            seed=42,
            max_seq_length=448,
            learning_rate=2e-5,
            batch_size=16,
            gradient_accumulation_steps=2,
            num_epochs=8,
            fp16=True,
            local_files_only=True,
            dataloader_num_workers=2,
            version="v1.0.0",
        )

        manifest = run_train_multitask(config, dry_run=False, validate_only=False)

        assert manifest is not None
        assert "indobert-multitask-fp16" in manifest.artifacts
        assert manifest.artifacts["indobert-multitask-fp16"].precision == "fp16"
        assert "multitask-temperatures" in manifest.artifacts
        assert (out_dir / "model.safetensors").is_file()
        assert (out_dir / "manifest.json").is_file()
        assert (out_dir / "temperatures.json").is_file()

        mock_transformers.AutoTokenizer.from_pretrained.assert_called_once_with(
            "indobenchmark/indobert-base-p1",
            local_files_only=True,
        )
        mock_transformers.AutoModel.from_pretrained.assert_called_once_with(
            "indobenchmark/indobert-base-p1",
            local_files_only=True,
        )
        mock_trainer_instance.train.assert_called_once()
        mock_trainer_instance.save_model.assert_called_once()
        mock_tokenizer_instance.save_pretrained.assert_called_once()

        training_args_call = mock_transformers.TrainingArguments.call_args
        assert training_args_call is not None
        args_kwargs = training_args_call.kwargs
        assert args_kwargs["output_dir"] == str(out_dir)
        assert args_kwargs["per_device_train_batch_size"] == 16
        assert args_kwargs["gradient_accumulation_steps"] == 2
        assert args_kwargs["fp16"] is True
        assert args_kwargs["learning_rate"] == 2e-5
        assert args_kwargs["dataloader_num_workers"] == 2
        assert args_kwargs["report_to"] == "none"
        assert args_kwargs["load_best_model_at_end"] is False
        assert "eval_strategy" in args_kwargs or "evaluation_strategy" in args_kwargs
        assert args_kwargs.get("eval_strategy") == "epoch" or args_kwargs.get("evaluation_strategy") == "epoch"
        assert args_kwargs["label_names"] == [
            "intent_labels",
            "category_labels",
            "risk_labels",
            "completeness_labels",
        ]
        assert args_kwargs["remove_unused_columns"] is False


def test_multitask_output_compatibility_and_indexing() -> None:
    out = MultiTaskOutput(
        loss=0.75,
        intent_logits="mock_intent",
        category_logits="mock_category",
        risk_logits="mock_risk",
        completeness_logits="mock_comp",
    )
    assert out.loss == 0.75
    assert out["loss"] == 0.75
    assert out[0] == 0.75
    assert out[1] == "mock_intent"
    assert out.logits == ("mock_intent", "mock_category", "mock_risk", "mock_comp")
    assert out["logits"] == ("mock_intent", "mock_category", "mock_risk", "mock_comp")
    assert isinstance(out, dict)
    loss, intent, cat, risk, comp = out
    assert loss == 0.75
    assert intent == "mock_intent"


def test_multitask_model_forward_eval_loss_and_labels_recognition(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_torch = mock.MagicMock()
    mock_nn = mock.MagicMock()

    class MockModule:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass
        def __call__(self, *args: Any, **kwargs: Any) -> Any:
            return self.forward(*args, **kwargs)

    mock_nn.Module = MockModule
    mock_nn.Dropout.return_value = lambda x: x
    mock_nn.Linear.side_effect = lambda in_f, out_f: (lambda x: mock.MagicMock())

    class MockCrossEntropyLoss:
        def __call__(self, logits: Any, targets: Any) -> Any:
            loss_mock = mock.MagicMock()
            loss_mock.__add__ = lambda s, o: loss_mock
            loss_mock.__radd__ = lambda s, o: loss_mock
            loss_mock.__rmul__ = lambda s, o: loss_mock
            loss_mock.__mul__ = lambda s, o: loss_mock
            return loss_mock

    mock_nn.CrossEntropyLoss = MockCrossEntropyLoss

    mock_torch.nn = mock_nn
    monkeypatch.setitem(sys.modules, "torch", mock_torch)
    monkeypatch.setitem(sys.modules, "torch.nn", mock_nn)

    model_cls = get_multitask_model_class()
    mock_encoder = mock.MagicMock()
    mock_encoder_out = mock.MagicMock()
    mock_encoder_out.pooler_output = mock.MagicMock(device="cpu")
    mock_encoder.return_value = mock_encoder_out

    model = model_cls(encoder=mock_encoder)

    out1 = model(
        input_ids=mock.MagicMock(),
        attention_mask=mock.MagicMock(),
        intent_labels=mock.MagicMock(),
        category_labels=mock.MagicMock(),
        risk_labels=mock.MagicMock(),
        completeness_labels=mock.MagicMock(),
    )
    assert out1.loss is not None
    assert out1["loss"] is not None
    assert isinstance(out1, dict)
    assert len(out1.logits) == 4

    out2 = model(
        input_ids=mock.MagicMock(),
        attention_mask=mock.MagicMock(),
        labels={
            "intent_labels": mock.MagicMock(),
            "category_labels": mock.MagicMock(),
            "risk_labels": mock.MagicMock(),
            "completeness_labels": mock.MagicMock(),
        },
    )
    assert out2.loss is not None

    mock_labels_2d = mock.MagicMock()
    mock_labels_2d.dim.return_value = 2
    mock_labels_2d.size.return_value = 4
    mock_labels_2d.__getitem__.return_value = mock.MagicMock()
    out3 = model(
        input_ids=mock.MagicMock(),
        attention_mask=mock.MagicMock(),
        labels=mock_labels_2d,
    )
    assert out3.loss is not None

    out4 = model(
        input_ids=mock.MagicMock(),
        attention_mask=mock.MagicMock(),
        labels=(mock.MagicMock(), mock.MagicMock(), mock.MagicMock(), mock.MagicMock()),
    )
    assert out4.loss is not None

    out5 = model(
        input_ids=mock.MagicMock(),
        attention_mask=mock.MagicMock(),
    )
    assert out5.loss is None
    assert len(out5.logits) == 4


def test_multitask_model_save_pretrained(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_torch = mock.MagicMock()
    mock_nn = mock.MagicMock()

    class MockModule:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass
        def state_dict(self) -> dict[str, Any]:
            return {"encoder.weight": mock.MagicMock()}

    mock_nn.Module = MockModule
    mock_torch.nn = mock_nn
    monkeypatch.setitem(sys.modules, "torch", mock_torch)
    monkeypatch.setitem(sys.modules, "torch.nn", mock_nn)

    model_cls = get_multitask_model_class()
    model = model_cls(encoder=mock.MagicMock())

    with tempfile.TemporaryDirectory() as tmpdir:
        save_dir = Path(tmpdir)
        model.save_pretrained(save_dir)
        mock_torch.save.assert_called_once()
        assert mock_torch.save.call_args[0][1] == str(save_dir / "pytorch_model.bin")


def test_multitask_execution_flow_without_dev_set(monkeypatch: pytest.MonkeyPatch) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir) / "no_dev_mt"

        mock_torch = mock.MagicMock()
        mock_torch.__path__ = []
        mock_torch.cuda.is_available.return_value = False
        mock_torch.utils.data.Dataset = object
        mock_torch.tensor = lambda x, **kw: x
        mock_torch.long = "long"

        mock_nn = mock.MagicMock()
        mock_torch.nn = mock_nn

        mock_tokenizer_instance = mock.MagicMock()
        mock_tokenizer_instance.return_value = {
            "input_ids": [101, 1000, 102],
            "attention_mask": [1, 1, 1],
            "token_type_ids": [0, 0, 0],
        }

        mock_encoder = mock.MagicMock()
        mock_encoder.config.hidden_size = 768
        mock_encoder.config.hidden_dropout_prob = 0.1

        mock_trainer_instance = mock.MagicMock()

        def mock_save_model(output_dir: str) -> None:
            p = Path(output_dir)
            (p / "model.safetensors").write_bytes(b"TRAINED_MOCK_NO_DEV" + b"\x00" * 64)
            (p / "config.json").write_text("{}", encoding="utf-8")

        mock_trainer_instance.save_model.side_effect = mock_save_model

        mock_transformers = mock.MagicMock()
        mock_transformers.AutoTokenizer.from_pretrained.return_value = mock_tokenizer_instance
        mock_transformers.AutoModel.from_pretrained.return_value = mock_encoder
        mock_transformers.Trainer.return_value = mock_trainer_instance

        monkeypatch.setitem(sys.modules, "torch", mock_torch)
        monkeypatch.setitem(sys.modules, "torch.nn", mock_nn)
        monkeypatch.setitem(sys.modules, "torch.utils", mock_torch.utils)
        monkeypatch.setitem(sys.modules, "torch.utils.data", mock_torch.utils.data)
        monkeypatch.setitem(sys.modules, "transformers", mock_transformers)

        config = MultitaskTrainConfig(
            model_name_or_path="indobenchmark/indobert-base-p1",
            output_dir=str(out_dir),
            seed=42,
            max_seq_length=128,
            learning_rate=2e-5,
            batch_size=8,
            gradient_accumulation_steps=1,
            num_epochs=1,
            fp16=False,
            local_files_only=True,
            dataloader_num_workers=0,
            temperature_scaling=False,
        )
        samples = [
            {
                "text": "Jalan berlubang parah",
                "intent": "COMPLAINT",
                "category": "ROAD",
                "risk": "HIGH",
                "completeness": "SUFFICIENT",
                "split": "train",
            }
            for _ in range(4)
        ]

        manifest = train_transformers_multitask(config, samples, out_dir)
        assert manifest is not None

        training_args_call = mock_transformers.TrainingArguments.call_args
        assert training_args_call is not None
        args_kwargs = training_args_call.kwargs
        assert args_kwargs["load_best_model_at_end"] is False
        eval_strategy_val = args_kwargs.get("eval_strategy", args_kwargs.get("evaluation_strategy"))
        assert eval_strategy_val == "no"
        assert args_kwargs["label_names"] == [
            "intent_labels",
            "category_labels",
            "risk_labels",
            "completeness_labels",
        ]


def test_multitask_model_forward_and_export_modes(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_torch = mock.MagicMock()
    mock_nn = mock.MagicMock()

    class MockModule:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass
        def __call__(self, *args: Any, **kwargs: Any) -> Any:
            return self.forward(*args, **kwargs)

    mock_nn.Module = MockModule
    mock_torch.nn = mock_nn

    monkeypatch.setitem(sys.modules, "torch", mock_torch)
    monkeypatch.setitem(sys.modules, "torch.nn", mock_nn)

    model_cls = get_multitask_model_class()
    assert issubclass(model_cls, MockModule)

    mock_encoder = mock.MagicMock()
    mock_encoder_out = mock.MagicMock()
    mock_encoder_out.last_hidden_state = mock.MagicMock()
    mock_encoder.return_value = mock_encoder_out

    model = model_cls(encoder=mock_encoder)
    assert hasattr(model, "intent_head")
    assert hasattr(model, "category_head")
    assert hasattr(model, "risk_head")
    assert hasattr(model, "completeness_head")


# ---------------------------------------------------------------------------
# 12. CLI Argument Overrides & Local Checkpoint Selection Tests
# ---------------------------------------------------------------------------

def test_dapt_cli_config_overrides_model_and_paths() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        cached_model_dir = tmp_path / "cached_indobert"
        cached_model_dir.mkdir()
        corpus_file = tmp_path / "custom_corpus.txt"
        corpus_file.write_text("Ini adalah laporan aduan jalan berlubang di jalan sudirman.\n", encoding="utf-8")
        out_dir = tmp_path / "dapt_override_out"

        res = subprocess.run(
            [
                sys.executable,
                "-m",
                "scripts.train_dapt",
                "--config",
                "configs/m3/dapt.json",
                "--model-name-or-path",
                str(cached_model_dir),
                "--corpus-path",
                str(corpus_file),
                "--output-dir",
                str(out_dir),
                "--dry-run",
            ],
            capture_output=True,
            text=True,
        )
        assert res.returncode == 0
        cfg_file = out_dir / "config.json"
        assert cfg_file.is_file()
        with open(cfg_file, "r", encoding="utf-8") as f:
            cfg_data = json.load(f)
        assert cfg_data["base_model"] == str(cached_model_dir)
        assert cfg_data["local_files_only"] is True


def test_dapt_cli_dataset_path_alias_and_selectable_local_files() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        corpus_file = tmp_path / "alias_corpus.txt"
        corpus_file.write_text("Genangan air setinggi 30cm menutup jalan raya barat.\n", encoding="utf-8")
        out_dir = tmp_path / "dapt_alias_out"

        res = subprocess.run(
            [
                sys.executable,
                "-m",
                "scripts.train_dapt",
                "--config",
                "configs/m3/dapt.json",
                "--model-name-or-path",
                str(tmp_path),
                "--dataset-path",
                str(corpus_file),
                "--output-dir",
                str(out_dir),
                "--no-local-files-only",
                "--dry-run",
            ],
            capture_output=True,
            text=True,
        )
        assert res.returncode == 0
        with open(out_dir / "config.json", "r", encoding="utf-8") as f:
            cfg_data = json.load(f)
        assert cfg_data["base_model"] == str(tmp_path)
        assert cfg_data["local_files_only"] is False


def test_multitask_cli_config_overrides_model_and_paths() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        cached_model_dir = tmp_path / "cached_dapt_checkpoint"
        cached_model_dir.mkdir()
        dataset_file = tmp_path / "custom_multitask.jsonl"
        ds = generate_dataset(num_scenarios=2, seed=42)
        export_trajectories_to_jsonl(ds, dataset_file)
        out_dir = tmp_path / "mt_override_out"

        res = subprocess.run(
            [
                sys.executable,
                "-m",
                "scripts.train_multitask",
                "--config",
                "configs/m3/multitask.json",
                "--model-name-or-path",
                str(cached_model_dir),
                "--dataset-path",
                str(dataset_file),
                "--output-dir",
                str(out_dir),
                "--local-files-only",
                "--dry-run",
            ],
            capture_output=True,
            text=True,
        )
        assert res.returncode == 0
        cfg_file = out_dir / "config.json"
        assert cfg_file.is_file()
        with open(cfg_file, "r", encoding="utf-8") as f:
            cfg_data = json.load(f)
        assert cfg_data["base_model"] == str(cached_model_dir)
        assert cfg_data["local_files_only"] is True


def test_multitask_cli_no_local_files_only_override() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        out_dir = tmp_path / "mt_remote_out"
        res = subprocess.run(
            [
                sys.executable,
                "-m",
                "scripts.train_multitask",
                "--config",
                "configs/m3/multitask.json",
                "--model-name-or-path",
                "indobenchmark/indobert-base-p1",
                "--output-dir",
                str(out_dir),
                "--no-local-files-only",
                "--dry-run",
            ],
            capture_output=True,
            text=True,
        )
        assert res.returncode == 0
        with open(out_dir / "config.json", "r", encoding="utf-8") as f:
            cfg_data = json.load(f)
        assert cfg_data["base_model"] == "indobenchmark/indobert-base-p1"
        assert cfg_data["local_files_only"] is False


def test_ner_cli_config_overrides_model_and_paths() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        cached_model_dir = tmp_path / "cached_dapt_ner"
        cached_model_dir.mkdir()
        dataset_file = tmp_path / "custom_ner.jsonl"
        ds = generate_dataset(num_scenarios=2, seed=42)
        export_trajectories_to_jsonl(ds, dataset_file)
        out_dir = tmp_path / "ner_override_out"

        res = subprocess.run(
            [
                sys.executable,
                "-m",
                "scripts.train_ner",
                "--config",
                "configs/m3/ner.json",
                "--model-name-or-path",
                str(cached_model_dir),
                "--dataset-path",
                str(dataset_file),
                "--output-dir",
                str(out_dir),
                "--local-files-only",
                "--dry-run",
            ],
            capture_output=True,
            text=True,
        )
        assert res.returncode == 0
        cfg_file = out_dir / "config.json"
        assert cfg_file.is_file()
        with open(cfg_file, "r", encoding="utf-8") as f:
            cfg_data = json.load(f)
        assert cfg_data["base_model"] == str(cached_model_dir)
        assert cfg_data["local_files_only"] is True


def test_ner_cli_no_local_files_only_override() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        out_dir = tmp_path / "ner_remote_out"
        res = subprocess.run(
            [
                sys.executable,
                "-m",
                "scripts.train_ner",
                "--config",
                "configs/m3/ner.json",
                "--model-name-or-path",
                "indobenchmark/indobert-base-p1",
                "--output-dir",
                str(out_dir),
                "--no-local-files-only",
                "--dry-run",
            ],
            capture_output=True,
            text=True,
        )
        assert res.returncode == 0
        with open(out_dir / "config.json", "r", encoding="utf-8") as f:
            cfg_data = json.load(f)
        assert cfg_data["base_model"] == "indobenchmark/indobert-base-p1"
        assert cfg_data["local_files_only"] is False


def test_cli_preserves_config_values_when_no_explicit_override() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        # 1. DAPT
        out_dapt = tmp_path / "dapt_preserve"
        res_dapt = subprocess.run(
            [
                sys.executable,
                "-m",
                "scripts.train_dapt",
                "--config",
                "configs/m3/dapt.json",
                "--output-dir",
                str(out_dapt),
                "--dry-run",
            ],
            capture_output=True,
            text=True,
        )
        assert res_dapt.returncode == 0
        with open(out_dapt / "config.json", "r", encoding="utf-8") as f:
            dapt_meta = json.load(f)
        assert dapt_meta["base_model"] == "indobenchmark/indobert-base-p1"
        assert dapt_meta["local_files_only"] is False

        # 2. Multitask
        out_mt = tmp_path / "mt_preserve"
        res_mt = subprocess.run(
            [
                sys.executable,
                "-m",
                "scripts.train_multitask",
                "--config",
                "configs/m3/multitask.json",
                "--output-dir",
                str(out_mt),
                "--dry-run",
            ],
            capture_output=True,
            text=True,
        )
        assert res_mt.returncode == 0
        with open(out_mt / "config.json", "r", encoding="utf-8") as f:
            mt_meta = json.load(f)
        assert mt_meta["base_model"] == "indobenchmark/indobert-base-p1"
        assert mt_meta["local_files_only"] is True

        # 3. NER
        out_ner = tmp_path / "ner_preserve"
        res_ner = subprocess.run(
            [
                sys.executable,
                "-m",
                "scripts.train_ner",
                "--config",
                "configs/m3/ner.json",
                "--output-dir",
                str(out_ner),
                "--dry-run",
            ],
            capture_output=True,
            text=True,
        )
        assert res_ner.returncode == 0
        with open(out_ner / "config.json", "r", encoding="utf-8") as f:
            ner_meta = json.load(f)
        assert ner_meta["base_model"] == "indobenchmark/indobert-base-p1"
        assert ner_meta["local_files_only"] is True


def test_configs_precision_device_settings_rtx3060_and_cpu_serving() -> None:
    config_paths = [
        Path("configs/m3/dapt.json"),
        Path("configs/m3/multitask.json"),
        Path("configs/m3/ner.json"),
    ]

    for p in config_paths:
        assert p.is_file()
        with open(p, "r", encoding="utf-8") as f:
            raw = json.load(f)

        hp = raw["hyperparameters"]
        extra = hp["extra_params"]
        prec = raw["precision_config"]

        # RTX3060 GPU training specifications
        assert extra.get("target_gpu") == "NVIDIA GeForce RTX 3060"
        assert extra.get("target_hardware") == "NVIDIA GeForce RTX 3060 12GB"
        assert extra.get("target_vram_gb") == 12
        assert extra.get("target_device") == "cuda:0"
        assert extra.get("fp16") is True
        assert extra.get("mixed_precision") == "fp16"

        # CPU FP32 serving specifications
        assert prec["device"] == "cpu"
        assert prec["training_precision"] == "fp32"
        assert prec["serving_precision"] == "fp32"


# ---------------------------------------------------------------------------
# 13. Audit & Real-Run Robustness Tests (Transformers API, Datasets, Multitask ONNX, Fallback Saving)
# ---------------------------------------------------------------------------

def test_ner_bubble_samples_preserve_splits_and_metadata() -> None:
    ds = generate_dataset(num_scenarios=5, seed=42)
    samples = extract_trajectory_ner_samples(ds)
    bubble_samples = [s for s in samples if s.get("granularity") == "bubble"]
    full_samples = [s for s in samples if s.get("granularity") == "full"]

    assert len(bubble_samples) > 0
    assert len(full_samples) > 0

    for s in bubble_samples:
        assert "split" in s
        assert s["split"] in ("train", "dev", "test")
        assert "scenario_id" in s and len(s["scenario_id"]) > 0

    for s in full_samples:
        assert "split" in s
        assert s["split"] in ("train", "dev", "test")
        assert "scenario_id" in s and len(s["scenario_id"]) > 0

    train_samples = [s for s in samples if s.get("split") == "train"]
    dev_samples = [s for s in samples if s.get("split") == "dev"]
    assert any(s.get("granularity") == "bubble" for s in train_samples)
    assert any(s.get("granularity") == "full" for s in train_samples)
    assert len(train_samples) > len(full_samples)


def test_ner_load_dataset_from_json_array() -> None:
    from scripts.train_ner import load_or_generate_dataset

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        json_file = tmp_path / "trajectories.json"
        ds = generate_dataset(num_scenarios=3, seed=42)
        json_file.write_text(
            json.dumps([t.model_dump(mode="json") for t in ds]),
            encoding="utf-8",
        )
        loaded = load_or_generate_dataset(str(json_file), seed=42, audit_splits=False)
        assert len(loaded) == 3
        assert isinstance(loaded[0], ComplaintTrajectory)


def test_dapt_load_corpus_from_json_array_and_extra_fields() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        # JSON array of dicts with various text fields
        json_file = tmp_path / "corpus.json"
        records = [
            {"text": "Laporan jalan rusak berlubang parah di Jalan Kaliurang."},
            {"body": "Saluran drainase tersumbat sampah plastik di dekat pasar."},
            {"message": "Lampu penerangan jalan padam sejak kemarin malam."},
            {"sentence": "Pohon tumbang menutup akses jalan raya utama kota."},
        ]
        json_file.write_text(json.dumps(records), encoding="utf-8")
        corpus = load_corpus(str(json_file), pii_scrubbing=True)
        assert len(corpus) == 4
        assert any("jalan rusak" in c for c in corpus)
        assert any("drainase" in c for c in corpus)

        # JSON array of plain strings
        json_strings_file = tmp_path / "corpus_strings.json"
        strings = [
            "Genangan air setinggi 20 cm di perempatan lampu merah.",
            "Tumpukan sampah liar menimbulkan bau busuk menyengat.",
        ]
        json_strings_file.write_text(json.dumps(strings), encoding="utf-8")
        corpus_str = load_corpus(str(json_strings_file), pii_scrubbing=True)
        assert len(corpus_str) == 2


def test_onnx_parse_config_dict_bundled_m3_format() -> None:
    # 1. Parse NER bundled M3 config
    with open("configs/m3/ner.json", "r", encoding="utf-8") as f:
        ner_raw = json.load(f)
    onnx_ner_cfg = parse_onnx_config_dict(ner_raw)
    assert onnx_ner_cfg.task == "ner"
    assert onnx_ner_cfg.precision == "fp32"
    assert onnx_ner_cfg.max_seq_length == 448
    assert onnx_ner_cfg.local_files_only is True

    # 2. Parse Multitask bundled M3 config
    with open("configs/m3/multitask.json", "r", encoding="utf-8") as f:
        mt_raw = json.load(f)
    onnx_mt_cfg = parse_onnx_config_dict(mt_raw)
    assert onnx_mt_cfg.task == "multitask"
    assert onnx_mt_cfg.precision == "fp32"
    assert onnx_mt_cfg.model_path == "artifacts/multitask/model.safetensors"


def test_onnx_load_model_for_export_multitask_simulation(monkeypatch: pytest.MonkeyPatch) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        checkpoint_dir = tmp_path / "checkpoint"
        checkpoint_dir.mkdir()

        config_data = {
            "architectures": ["BertForMultiTaskClassification"],
            "base_model": "indobenchmark/indobert-base-p1",
            "hidden_size": 768,
            "heads": {
                "intent": {"num_labels": 3, "classes": ["COMPLAINT", "INQUIRY", "FEEDBACK"]},
                "category": {"num_labels": 6, "classes": ["ROAD", "DRAINAGE_FLOOD", "WASTE", "CLEAN_WATER", "CIVIL_ADMIN", "HEALTH_SERVICE"]},
                "risk": {"num_labels": 4, "classes": ["LOW", "MEDIUM", "HIGH", "URGENT"]},
                "completeness": {"num_labels": 3, "classes": ["SUFFICIENT", "INCOMPLETE", "AMBIGUOUS"]},
            },
        }
        (checkpoint_dir / "config.json").write_text(json.dumps(config_data), encoding="utf-8")
        (checkpoint_dir / "model.safetensors").write_bytes(b"MOCK_MULTITASK_WEIGHTS")

        mock_torch = mock.MagicMock()
        mock_nn = mock.MagicMock()

        class MockModule:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                pass
            def __call__(self, *args: Any, **kwargs: Any) -> Any:
                return None
            def load_state_dict(self, *args: Any, **kwargs: Any) -> Any:
                pass

        mock_nn.Module = MockModule
        mock_torch.nn = mock_nn

        mock_encoder = mock.MagicMock()
        mock_transformers = mock.MagicMock()
        mock_transformers.AutoModel.from_pretrained.return_value = mock_encoder

        mock_safetensors = mock.MagicMock()
        mock_safetensors.torch = mock.MagicMock()
        mock_safetensors.torch.load_file.return_value = {"encoder.weight": "mock"}

        monkeypatch.setitem(sys.modules, "torch", mock_torch)
        monkeypatch.setitem(sys.modules, "torch.nn", mock_nn)
        monkeypatch.setitem(sys.modules, "transformers", mock_transformers)
        monkeypatch.setitem(sys.modules, "safetensors", mock_safetensors)
        monkeypatch.setitem(sys.modules, "safetensors.torch", mock_safetensors.torch)

        loaded_model = load_model_for_export(checkpoint_dir, task="multitask", local_files_only=True)
        assert hasattr(loaded_model, "intent_head")
        assert hasattr(loaded_model, "category_head")
        assert hasattr(loaded_model, "risk_head")
        assert hasattr(loaded_model, "completeness_head")
        mock_safetensors.torch.load_file.assert_called_once_with(str(checkpoint_dir / "model.safetensors"))


def test_onnx_load_model_for_export_strict_state_dict_fails_on_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        checkpoint_dir = tmp_path / "checkpoint"
        checkpoint_dir.mkdir()

        config_data = {
            "architectures": ["BertForMultiTaskClassification"],
            "base_model": "indobenchmark/indobert-base-p1",
            "hidden_size": 768,
            "vocab_size": 50000,
            "heads": {
                "intent": {"num_labels": 3, "classes": ["COMPLAINT", "INQUIRY", "FEEDBACK"]},
                "category": {"num_labels": 6, "classes": ["ROAD", "DRAINAGE_FLOOD", "WASTE", "CLEAN_WATER", "CIVIL_ADMIN", "HEALTH_SERVICE"]},
                "risk": {"num_labels": 4, "classes": ["LOW", "MEDIUM", "HIGH", "URGENT"]},
                "completeness": {"num_labels": 3, "classes": ["SUFFICIENT", "INCOMPLETE", "AMBIGUOUS"]},
            },
        }
        (checkpoint_dir / "config.json").write_text(json.dumps(config_data), encoding="utf-8")
        (checkpoint_dir / "model.safetensors").write_bytes(b"MOCK_MULTITASK_WEIGHTS")

        mock_torch = mock.MagicMock()
        mock_nn = mock.MagicMock()

        class StrictFailModule:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                pass
            def __call__(self, *args: Any, **kwargs: Any) -> Any:
                return None
            def load_state_dict(self, state_dict: Any, strict: bool = False) -> Any:
                if strict:
                    raise RuntimeError("size mismatch for encoder.embeddings.word_embeddings.weight: copying a param with shape torch.Size([50000, 768]) from checkpoint, the shape in current model is torch.Size([30522, 768]).")

        mock_nn.Module = StrictFailModule
        mock_torch.nn = mock_nn

        mock_encoder = mock.MagicMock()
        mock_transformers = mock.MagicMock()
        mock_transformers.AutoModel.from_pretrained.return_value = mock_encoder

        mock_safetensors = mock.MagicMock()
        mock_safetensors.torch = mock.MagicMock()
        mock_safetensors.torch.load_file.return_value = {"encoder.embeddings.word_embeddings.weight": "shape_mismatch"}

        monkeypatch.setitem(sys.modules, "torch", mock_torch)
        monkeypatch.setitem(sys.modules, "torch.nn", mock_nn)
        monkeypatch.setitem(sys.modules, "transformers", mock_transformers)
        monkeypatch.setitem(sys.modules, "safetensors", mock_safetensors)
        monkeypatch.setitem(sys.modules, "safetensors.torch", mock_safetensors.torch)

        with pytest.raises(RuntimeError, match="size mismatch"):
            load_model_for_export(checkpoint_dir, task="multitask", local_files_only=True)


def test_onnx_torch_export_simulation_multitask_task(monkeypatch: pytest.MonkeyPatch) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir)
        target_onnx = out_dir / "exported_multitask.onnx"

        mock_torch = mock.MagicMock()
        mock_torch.float32 = "float32"
        mock_torch.long = "long"
        mock_torch.zeros.return_value = mock.MagicMock()
        mock_torch.ones.return_value = mock.MagicMock()

        monkeypatch.setitem(sys.modules, "torch", mock_torch)
        monkeypatch.setitem(sys.modules, "torch.onnx", mock_torch.onnx)

        mock_model = mock.MagicMock()

        result_path = export_torch_model_to_onnx(
            model=mock_model,
            target_path=target_onnx,
            task="multitask",
            max_seq_length=448,
            opset=17,
        )
        assert result_path == target_onnx

        mock_torch.onnx.export.assert_called_once()
        _, call_kwargs = mock_torch.onnx.export.call_args
        assert call_kwargs["output_names"] == [
            "intent_logits",
            "category_logits",
            "risk_logits",
            "completeness_logits",
        ]
        assert "intent_logits" in call_kwargs["dynamic_axes"]
        assert "category_logits" in call_kwargs["dynamic_axes"]
        assert "risk_logits" in call_kwargs["dynamic_axes"]
        assert "completeness_logits" in call_kwargs["dynamic_axes"]


def test_dapt_and_ner_fallback_weight_saving(monkeypatch: pytest.MonkeyPatch) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir) / "fallback_dapt"

        mock_torch = mock.MagicMock()
        mock_torch.cuda.is_available.return_value = False
        mock_torch.utils.data.Dataset = object
        mock_torch.tensor = lambda x, **kw: x
        mock_torch.long = "long"
        mock_torch.save = lambda obj, f: Path(f).write_bytes(b"MOCK_FALLBACK_WEIGHTS")

        mock_tokenizer_instance = mock.MagicMock()
        mock_tokenizer_instance.return_value = {
            "input_ids": [[101, 1000, 102]],
            "attention_mask": [[1, 1, 1]],
            "special_tokens_mask": [[1, 0, 1]],
        }

        mock_model_instance = mock.MagicMock()
        mock_model_instance.state_dict.return_value = {"weight": "mock"}

        mock_trainer_instance = mock.MagicMock()

        def mock_save_model_no_weights(output_dir: str) -> None:
            p = Path(output_dir)
            p.mkdir(parents=True, exist_ok=True)
            (p / "config.json").write_text("{}", encoding="utf-8")

        mock_trainer_instance.save_model.side_effect = mock_save_model_no_weights

        def mock_save_pretrained(save_directory: str) -> None:
            p = Path(save_directory)
            p.mkdir(parents=True, exist_ok=True)
            (p / "tokenizer_config.json").write_text("{}", encoding="utf-8")

        mock_tokenizer_instance.save_pretrained.side_effect = mock_save_pretrained

        mock_transformers = mock.MagicMock()
        mock_transformers.AutoTokenizer.from_pretrained.return_value = mock_tokenizer_instance
        mock_transformers.AutoModelForMaskedLM.from_pretrained.return_value = mock_model_instance
        mock_transformers.Trainer.return_value = mock_trainer_instance

        monkeypatch.setitem(sys.modules, "torch", mock_torch)
        monkeypatch.setitem(sys.modules, "torch.utils", mock_torch.utils)
        monkeypatch.setitem(sys.modules, "torch.utils.data", mock_torch.utils.data)
        monkeypatch.setitem(sys.modules, "transformers", mock_transformers)

        config = DAPTConfig(
            model_name_or_path="indobenchmark/indobert-base-p1",
            output_dir=str(out_dir),
            seed=42,
            max_seq_length=448,
            learning_rate=2e-5,
            batch_size=8,
            gradient_accumulation_steps=4,
            num_epochs=1,
            fp16=False,
            local_files_only=True,
            version="v1.0.0",
        )

        manifest = run_train_dapt(config, dry_run=False, validate_only=False)
        assert manifest is not None
        assert "indobert-dapt" in manifest.artifacts
        assert (out_dir / "manifest.json").is_file()


def test_onnx_export_cli_model_path_override_and_local_files_only() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        out_dir = tmp_path / "onnx_cli_out"
        res = subprocess.run(
            [
                sys.executable,
                "-m",
                "scripts.export_onnx",
                "--output-dir",
                str(out_dir),
                "--task",
                "multitask",
                "--local-files-only",
                "--dry-run",
            ],
            capture_output=True,
            text=True,
        )
        assert res.returncode == 0
        assert (out_dir / "model.onnx").is_file()
        assert (out_dir / "manifest.json").is_file()


def test_multitask_output_super_post_init_scalar_loss_regression() -> None:
    payload = {
        "loss": 0.42,
        "intent_logits": [0.1, 0.2],
        "category_logits": [0.3, 0.4],
        "risk_logits": [0.5, 0.6],
        "completeness_logits": [0.7, 0.8],
    }
    out = MultiTaskOutput(payload)
    assert not isinstance(out.loss, dict)
    assert out.loss == 0.42
    assert out["loss"] == 0.42
    assert out[0] == 0.42
    assert out.intent_logits == [0.1, 0.2]
    assert out["intent_logits"] == [0.1, 0.2]
    assert out.logits == ([0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8])
    loss, intent, cat, risk, comp = out
    assert loss == 0.42
    assert not isinstance(loss, dict)


def test_multitask_output_calls_super_post_init_directly() -> None:
    from dataclasses import dataclass

    called = []

    class MockBaseOutput(dict):
        def __post_init__(self) -> None:
            called.append(True)

    @dataclass
    class DerivedOutput(MockBaseOutput):
        loss: Any = None
        intent_logits: Any = None
        category_logits: Any = None
        risk_logits: Any = None
        completeness_logits: Any = None

        def __post_init__(self) -> None:
            super_post_init = getattr(super(), "__post_init__", None)
            if callable(super_post_init):
                super().__post_init__()

    DerivedOutput(loss=1.23)
    assert called == [True]


def test_train_transformers_multitask_trainer_processing_class_api(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_torch = mock.MagicMock()
    mock_nn = mock.MagicMock()

    class MockModule:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass
        def __call__(self, *args: Any, **kwargs: Any) -> Any:
            return self.forward(*args, **kwargs)

    mock_nn.Module = MockModule
    mock_torch.nn = mock_nn

    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir)

        mock_tokenizer_instance = mock.MagicMock()
        mock_tokenizer_instance.return_value = {
            "input_ids": [101, 1000, 102],
            "attention_mask": [1, 1, 1],
            "token_type_ids": [0, 0, 0],
        }

        mock_encoder = mock.MagicMock()
        mock_encoder.config.hidden_size = 768
        mock_encoder.config.hidden_dropout_prob = 0.1

        captured_trainer_kwargs: dict[str, Any] = {}

        class FakeTrainer457:
            def __init__(self, model: Any, args: Any = None, processing_class: Any = None, **kwargs: Any) -> None:
                captured_trainer_kwargs["model"] = model
                captured_trainer_kwargs["args"] = args
                captured_trainer_kwargs["processing_class"] = processing_class
                captured_trainer_kwargs.update(kwargs)

            def train(self) -> None:
                pass

            def save_model(self, output_dir: str) -> None:
                p = Path(output_dir)
                (p / "model.safetensors").write_bytes(b"MOCK" + b"\x00" * 64)
                (p / "config.json").write_text("{}", encoding="utf-8")

        mock_transformers = mock.MagicMock()
        mock_transformers.AutoTokenizer.from_pretrained.return_value = mock_tokenizer_instance
        mock_transformers.AutoModel.from_pretrained.return_value = mock_encoder
        mock_transformers.Trainer = FakeTrainer457

        monkeypatch.setitem(sys.modules, "torch", mock_torch)
        monkeypatch.setitem(sys.modules, "torch.nn", mock_nn)
        monkeypatch.setitem(sys.modules, "torch.utils", mock_torch.utils)
        monkeypatch.setitem(sys.modules, "torch.utils.data", mock_torch.utils.data)
        monkeypatch.setitem(sys.modules, "transformers", mock_transformers)

        config = MultitaskTrainConfig(
            model_name_or_path="indobenchmark/indobert-base-p1",
            output_dir=str(out_dir),
            seed=42,
            max_seq_length=128,
            learning_rate=2e-5,
            batch_size=8,
            num_epochs=1,
            fp16=False,
            local_files_only=True,
            dataloader_num_workers=0,
            temperature_scaling=False,
        )
        samples = [
            {
                "text": "Jalan berlubang parah",
                "intent": "COMPLAINT",
                "category": "ROAD",
                "risk": "HIGH",
                "completeness": "SUFFICIENT",
                "split": "train",
            }
            for _ in range(4)
        ]

        manifest = train_transformers_multitask(config, samples, out_dir)
        assert manifest is not None
        assert "processing_class" in captured_trainer_kwargs
        assert captured_trainer_kwargs["processing_class"] == mock_tokenizer_instance
        assert "tokenizer" not in captured_trainer_kwargs


def test_train_transformers_multitask_trainer_legacy_tokenizer_api(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_torch = mock.MagicMock()
    mock_nn = mock.MagicMock()

    class MockModule:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass
        def __call__(self, *args: Any, **kwargs: Any) -> Any:
            return self.forward(*args, **kwargs)

    mock_nn.Module = MockModule
    mock_torch.nn = mock_nn

    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir)

        mock_tokenizer_instance = mock.MagicMock()
        mock_tokenizer_instance.return_value = {
            "input_ids": [101, 1000, 102],
            "attention_mask": [1, 1, 1],
            "token_type_ids": [0, 0, 0],
        }

        mock_encoder = mock.MagicMock()
        mock_encoder.config.hidden_size = 768
        mock_encoder.config.hidden_dropout_prob = 0.1

        captured_trainer_kwargs: dict[str, Any] = {}

        class FakeTrainerLegacy:
            def __init__(self, model: Any, args: Any = None, tokenizer: Any = None, **kwargs: Any) -> None:
                captured_trainer_kwargs["model"] = model
                captured_trainer_kwargs["args"] = args
                captured_trainer_kwargs["tokenizer"] = tokenizer
                captured_trainer_kwargs.update(kwargs)

            def train(self) -> None:
                pass

            def save_model(self, output_dir: str) -> None:
                p = Path(output_dir)
                (p / "model.safetensors").write_bytes(b"MOCK" + b"\x00" * 64)
                (p / "config.json").write_text("{}", encoding="utf-8")

        mock_transformers = mock.MagicMock()
        mock_transformers.AutoTokenizer.from_pretrained.return_value = mock_tokenizer_instance
        mock_transformers.AutoModel.from_pretrained.return_value = mock_encoder
        mock_transformers.Trainer = FakeTrainerLegacy

        monkeypatch.setitem(sys.modules, "torch", mock_torch)
        monkeypatch.setitem(sys.modules, "torch.nn", mock_nn)
        monkeypatch.setitem(sys.modules, "torch.utils", mock_torch.utils)
        monkeypatch.setitem(sys.modules, "torch.utils.data", mock_torch.utils.data)
        monkeypatch.setitem(sys.modules, "transformers", mock_transformers)

        config = MultitaskTrainConfig(
            model_name_or_path="indobenchmark/indobert-base-p1",
            output_dir=str(out_dir),
            seed=42,
            max_seq_length=128,
            learning_rate=2e-5,
            batch_size=8,
            num_epochs=1,
            fp16=False,
            local_files_only=True,
            dataloader_num_workers=0,
            temperature_scaling=False,
        )
        samples = [
            {
                "text": "Jalan berlubang parah",
                "intent": "COMPLAINT",
                "category": "ROAD",
                "risk": "HIGH",
                "completeness": "SUFFICIENT",
                "split": "train",
            }
            for _ in range(4)
        ]

        manifest = train_transformers_multitask(config, samples, out_dir)
        assert manifest is not None
        assert "tokenizer" in captured_trainer_kwargs
        assert captured_trainer_kwargs["tokenizer"] == mock_tokenizer_instance
        assert "processing_class" not in captured_trainer_kwargs


def test_multitask_config_preserves_encoder_vocab_size_50000() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir) / "mt_vocab_test"
        cfg = MultitaskTrainConfig(
            model_name_or_path="indobenchmark/indobert-base-p1",
            output_dir=str(out_dir),
            seed=42,
        )
        run_train_multitask(cfg, dry_run=True, validate_only=False)
        meta_file = out_dir / "config.json"
        assert meta_file.is_file()
        with open(meta_file, "r", encoding="utf-8") as f:
            meta = json.load(f)
        assert meta.get("vocab_size") == 50000


def test_validate_tokenizer_and_numeric_parity_mock_behavior() -> None:
    # When dependencies are missing or model is a mock, gracefully handle
    mock_model = mock.MagicMock()
    mock_model._mock_return_value = True
    res = validate_tokenizer_and_numeric_parity(
        onnx_path=Path("dummy.onnx"),
        torch_model=mock_model,
        task="multitask",
    )
    assert res.get("verified") in (True, False)


def test_validate_tokenizer_and_numeric_parity_logic(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_np = mock.MagicMock()
    mock_np.max.return_value = 1.2e-5
    mock_np.allclose.return_value = True

    mock_ort = mock.MagicMock()
    mock_session = mock.MagicMock()
    mock_out = mock.MagicMock()
    mock_out.name = "intent_logits"
    mock_session.get_outputs.return_value = [mock_out]
    mock_session.run.return_value = [mock.MagicMock()]
    mock_ort.InferenceSession.return_value = mock_session

    mock_torch = mock.MagicMock()
    mock_torch.no_grad.return_value.__enter__ = mock.MagicMock()
    mock_torch.no_grad.return_value.__exit__ = mock.MagicMock()

    mock_tok_cls = mock.MagicMock()
    mock_tok_inst = mock.MagicMock()
    mock_tok_inst.get_vocab.return_value = {"[PAD]": 0, "[CLS]": 1, "kata": 30520}
    mock_tok_inst.vocab_size = 30521
    mock_tok_inst.return_value = {
        "input_ids": mock.MagicMock(min=mock.MagicMock(return_value=mock.MagicMock(item=lambda: 0)),
                                   max=mock.MagicMock(return_value=mock.MagicMock(item=lambda: 30520)),
                                   cpu=mock.MagicMock(return_value=mock.MagicMock(numpy=lambda: "ids"))),
        "attention_mask": mock.MagicMock(cpu=mock.MagicMock(return_value=mock.MagicMock(numpy=lambda: "mask"))),
    }
    mock_tok_cls.from_pretrained.return_value = mock_tok_inst

    monkeypatch.setitem(sys.modules, "numpy", mock_np)
    monkeypatch.setitem(sys.modules, "onnxruntime", mock_ort)
    monkeypatch.setitem(sys.modules, "torch", mock_torch)

    class DummyTorchModel:
        def __init__(self) -> None:
            self.encoder = mock.MagicMock()
            self.encoder.embeddings = mock.MagicMock()
            self.encoder.embeddings.word_embeddings = mock.MagicMock()
            self.encoder.embeddings.word_embeddings.weight = mock.MagicMock(shape=[50000, 768])
        def eval(self) -> None:
            pass
        def __call__(self, *args: Any, **kwargs: Any) -> Any:
            return mock_res

    mock_model = DummyTorchModel()

    # Out of range test: max token ID >= model_vocab_size (50000)
    mock_tok_inst.get_vocab.return_value = {"[PAD]": 0, "oov": 50001}
    with tempfile.TemporaryDirectory() as tmpdir:
        onnx_file = Path(tmpdir) / "test.onnx"
        onnx_file.write_bytes(b"dummy")

        mock_transformers = mock.MagicMock()
        mock_transformers.AutoTokenizer = mock_tok_cls
        monkeypatch.setitem(sys.modules, "transformers", mock_transformers)

        with pytest.raises(ValueError, match="out of range"):
            validate_tokenizer_and_numeric_parity(
                onnx_path=onnx_file,
                torch_model=mock_model,
                task="multitask",
            )


def test_ner_all_72_templates_span_coverage() -> None:
    from unittest.mock import patch
    import scripts.generate_m3_dataset as gen_m3
    from scripts.generate_m3_dataset import EXTENDED_TEMPLATES, generate_safe_trajectory
    from services.dataset.generator import DEFAULT_TEMPLATES

    all_templates = list(DEFAULT_TEMPLATES) + list(EXTENDED_TEMPLATES)
    expected_total = len(DEFAULT_TEMPLATES) + len(EXTENDED_TEMPLATES)
    assert len(all_templates) == expected_total
    assert expected_total > 0

    fallback_lexicons = {
        cat: {
            "anchors": ("penerangan jalan umum", "lampu pengatur lalu lintas"),
            "forbidden_terms": (),
        }
        for cat in Category
        if cat not in gen_m3.CATEGORY_ANCHOR_LEXICONS
    }

    bad_words = {
        "total", "tercemar", "tergeletak", "bertumpuk", "ditolak", "mangkir",
        "kabar mangkir", "alasan regulasi ditolak", "kebun warga bertumpuk",
    }

    with patch.dict(gen_m3.CATEGORY_ANCHOR_LEXICONS, fallback_lexicons):
        for idx, tmpl in enumerate(all_templates):
            traj = generate_safe_trajectory(
                template=tmpl,
                scenario_id=f"test-cov-{idx}",
                persona="STANDARD",
                noise_level="LOW",
                multi_turn=True,
                seed=42 + idx,
                split=DatasetSplit.TRAIN,
            )
            samples = extract_trajectory_ner_samples([traj])
            assert len(samples) >= 2

            b1 = samples[0]
            spans = b1["spans"]
            span_texts = [b1["text"][s:e].strip().lower() for s, e, _ in spans]

            for st in span_texts:
                assert st not in bad_words, f"Bad word labeled as entity in {tmpl.family_id}: '{st}'"
                assert len(st.split()) <= 7, f"Span exceeds compact phrase limit: '{st}'"

            all_labels = {lbl for _, _, lbl in spans}
            assert len(all_labels) > 0, f"No entities extracted for {tmpl.family_id}"


def test_ner_explicit_time_coverage() -> None:
    from scripts.train_ner import _extract_spans_for_text

    text = "Lapor kondisi saat ini belum parah, tolong segera ditangani sekarang juga di jam operasional tengah malam."
    traj = ComplaintTrajectory(
        scenario_id="t1",
        family_id="road-pothole-arterial",
        category=Category.ROAD,
        split=DatasetSplit.TRAIN,
        turns=(),
        world_truth={},
    )
    spans = _extract_spans_for_text(text, traj, [])
    time_spans = [text[s:e] for s, e, lbl in spans if lbl == "TIME"]

    assert "saat ini" in time_spans
    assert "sekarang juga" in time_spans
    assert "jam operasional" in time_spans
    assert "tengah malam" in time_spans


def test_ner_landmark_loc_integrity_without_inner_obj_carving() -> None:
    text = "Titik patokannya di sekitar seratus meter dari jembatan baru ciliwung"
    raw_spans = [
        (20, 69, "LOC"),
        (47, 55, "OBJ"),
    ]
    resolved = resolve_overlapping_spans(raw_spans, text=text)
    assert len(resolved) == 1
    assert resolved[0] == (20, 69, "LOC")
    assert text[resolved[0][0]:resolved[0][1]] == "sekitar seratus meter dari jembatan baru ciliwung"


def test_ner_compound_noun_phrase_preservation() -> None:
    text = "Lapor: pipa induk air bersih pecah dan semburan air membanjiri"
    raw_spans = [
        (7, 28, "OBJ"),
        (7, 17, "OBJ"),
        (18, 28, "OBJ"),
        (39, 51, "OBJ"),
    ]
    resolved = resolve_overlapping_spans(raw_spans, text=text)
    span_texts = [text[s:e] for s, e, _ in resolved]
    assert "pipa induk air bersih" in span_texts
    assert "semburan air" in span_texts
    assert "pipa induk" not in span_texts
    assert "air bersih" not in span_texts


def test_ner_class_weights_and_training_controls_parsing() -> None:
    ner_cfg_path = Path("configs/m3/ner.json")
    cfg = parse_ner_config_file(ner_cfg_path)
    assert cfg.class_weights is not None
    assert cfg.class_weights.get("O") == 1.0
    assert cfg.class_weights.get("B-LOC") == 2.5
    assert cfg.class_weights.get("I-LOC") == 2.0
    assert cfg.class_weights.get("B-OBJ") == 2.5
    assert cfg.class_weights.get("I-OBJ") == 2.0
    assert cfg.class_weights.get("B-TIME") == 3.0
    assert cfg.class_weights.get("I-TIME") == 2.5
    assert cfg.metric_for_best_model == "obj_f1"
    assert cfg.greater_is_better is True
    assert cfg.early_stopping_patience == 2


def test_calibrate_temperatures_with_explicit_logits_and_held_improvement() -> None:
    """Verify calibrate_temperatures fits per-head temperatures from logits/labels and improves held ECE/NLL."""
    import random

    heads_classes = {
        "intent": 3,
        "category": 6,
        "risk": 4,
        "completeness": 3,
    }

    def generate_data(seed: int, n_samples: int) -> tuple[dict[str, list[list[float]]], dict[str, list[int]]]:
        split_logits: dict[str, list[list[float]]] = {}
        split_labels: dict[str, list[int]] = {}
        for idx_h, (head, k) in enumerate(heads_classes.items()):
            rng = random.Random(seed + idx_h * 101)
            h_logits: list[list[float]] = []
            h_labels: list[int] = []
            for _ in range(n_samples):
                y = rng.randrange(k)
                h_labels.append(y)
                is_correct = rng.random() < 0.76
                pred_y = y if is_correct else (y + 1 + rng.randrange(k - 1)) % k
                row = [round(rng.uniform(-0.25, 0.25), 4) for _ in range(k)]
                row[pred_y] += 3.85
                h_logits.append(row)
            split_logits[head] = h_logits
            split_labels[head] = h_labels
        return split_logits, split_labels

    dev_logits, dev_labels = generate_data(seed=42, n_samples=120)
    held_logits, held_labels = generate_data(seed=999, n_samples=120)

    fitted_temps = calibrate_temperatures(dev_logits=dev_logits, dev_labels=dev_labels)

    dummy_defaults = {"intent": 1.05, "category": 1.12, "risk": 1.08, "completeness": 1.02}
    for head in heads_classes:
        assert head in fitted_temps
        # Verify no dummy default values were returned
        assert fitted_temps[head] != dummy_defaults[head]
        assert fitted_temps[head] > 1.0

    calibrator = TemperatureCalibrator(temperatures=fitted_temps)
    for head in heads_classes:
        logits = held_logits[head]
        labels = held_labels[head]
        metrics = calibrator.evaluate(logits, labels, n_bins=5, head=head)

        assert metrics.uncalibrated_ece > 0.12
        assert metrics.calibrated_ece < metrics.uncalibrated_ece
        assert metrics.calibrated_nll < metrics.uncalibrated_nll
        assert metrics.calibrated_ece <= 0.08


def test_calibrate_temperatures_from_dev_samples_no_dummy_default() -> None:
    """Verify calibrate_temperatures dynamically fits per-head temperatures from trajectory samples."""
    trajectories = generate_dataset(num_scenarios=12, seed=42)
    samples = extract_trajectory_multitask_samples(trajectories)
    dev_samples = [s for s in samples if s.get("split") == DatasetSplit.DEV.value] or samples

    temps = calibrate_temperatures(dev_samples=dev_samples, seed=42)

    dummy_defaults = {"intent": 1.05, "category": 1.12, "risk": 1.08, "completeness": 1.02}
    for head in ("intent", "category", "risk", "completeness"):
        assert head in temps
        assert isinstance(temps[head], float)
        assert temps[head] > 0.0
        assert temps[head] != dummy_defaults[head]


def test_multitask_dry_run_persists_fitted_temperatures_and_roundtrips() -> None:
    """Verify multitask training dry-run saves fitted temperatures.json and round-trips via calibrator."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir) / "multitask_dry_run_out"
        config = MultitaskTrainConfig(
            output_dir=str(out_dir),
            seed=42,
            temperature_scaling=True,
            version="v1.0.0",
        )

        manifest = run_train_multitask(config, dry_run=True)
        assert manifest is not None
        assert "multitask-temperatures" in manifest.artifacts

        temp_path = out_dir / "temperatures.json"
        assert temp_path.is_file()

        raw_temps = load_temperatures(temp_path)
        dummy_defaults = {"intent": 1.05, "category": 1.12, "risk": 1.08, "completeness": 1.02}
        for head in ("intent", "category", "risk", "completeness"):
            assert head in raw_temps
            assert raw_temps[head] != dummy_defaults[head]
            assert raw_temps[head] > 0.0

        calibrator = TemperatureCalibrator.from_json(temp_path)
        for head in ("intent", "category", "risk", "completeness"):
            assert calibrator.get_temperature(head) == raw_temps[head]


def test_ner_resolve_tag_weight() -> None:
    weights = {"O": 1.0, "B-LOC": 2.5, "I-LOC": 2.0, "B-OBJ": 2.5, "I-OBJ": 2.0, "B-TIME": 3.0, "I-TIME": 2.5}
    assert resolve_tag_weight(weights, "O") == 1.0
    assert resolve_tag_weight(weights, "B-LOC") == 2.5
    assert resolve_tag_weight(weights, "I-LOC") == 2.0
    assert resolve_tag_weight(weights, "B-TIME") == 3.0

    base_weights = {"O": 1.0, "LOC": 3.0, "OBJ": 2.0}
    assert resolve_tag_weight(base_weights, "B-LOC") == 3.0
    assert resolve_tag_weight(base_weights, "I-LOC") == 2.4
    assert resolve_tag_weight(base_weights, "B-OBJ") == 2.0
    assert resolve_tag_weight(base_weights, "B-TIME") == 1.0
    assert resolve_tag_weight(base_weights, "O") == 1.0


def test_weighted_ner_trainer_compute_loss(monkeypatch: pytest.MonkeyPatch) -> None:
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        torch = None  # type: ignore[assignment]
        nn = None  # type: ignore[assignment]

    if torch is None:
        mock_torch = mock.MagicMock()
        mock_nn = mock.MagicMock()

        class MockCrossEntropyLoss:
            def __init__(self, weight: Any = None, ignore_index: int = -100) -> None:
                self.weight = weight
                self.ignore_index = ignore_index

            def __call__(self, logits: Any, labels: Any) -> Any:
                loss = mock.MagicMock()
                loss.item.return_value = 2.5 if self.weight is not None else 1.0
                return loss

        mock_nn.CrossEntropyLoss = MockCrossEntropyLoss
        mock_torch.nn = mock_nn
        monkeypatch.setitem(sys.modules, "torch", mock_torch)
        monkeypatch.setitem(sys.modules, "torch.nn", mock_nn)

        TrainerCls = get_weighted_ner_trainer_class(object)

        class MockModel:
            def __call__(self, **kwargs: Any) -> Any:
                mock_logits = mock.MagicMock()
                mock_logits.device = "cpu"
                mock_logits.size.return_value = 3
                mock_loss = mock.MagicMock()
                mock_loss.item.return_value = 1.0
                return {"logits": mock_logits, "loss": mock_loss}

        model = MockModel()
        trainer_unweighted = TrainerCls(class_weights=None)
        loss_unweighted = trainer_unweighted.compute_loss(model, {"input_ids": [1], "labels": mock.MagicMock()})

        mock_weights = mock.MagicMock()
        mock_weights.to.return_value = mock_weights
        trainer_weighted = TrainerCls(class_weights=mock_weights)
        loss_weighted, outputs = trainer_weighted.compute_loss(model, {"input_ids": [1], "labels": mock.MagicMock()}, return_outputs=True)

        assert loss_weighted.item() > loss_unweighted.item()
        assert "loss" in outputs
        assert outputs["loss"] == loss_weighted
    else:
        TrainerCls = get_weighted_ner_trainer_class(object)

        class MockRealModel(nn.Module):
            def forward(self, input_ids: Any = None, **kwargs: Any) -> Any:
                logits = torch.tensor([[[2.0, -1.0, -1.0], [2.0, -1.0, -1.0], [0.0, 0.0, 0.0]]], dtype=torch.float)
                return {"logits": logits}

        model = MockRealModel()
        trainer_unweighted = TrainerCls(class_weights=torch.tensor([1.0, 1.0, 1.0], dtype=torch.float))
        inputs = {
            "input_ids": torch.tensor([[101, 102, 103]]),
            "labels": torch.tensor([[0, 1, -100]]),
        }
        loss_unweighted = trainer_unweighted.compute_loss(model, inputs)

        trainer_weighted = TrainerCls(class_weights=torch.tensor([1.0, 3.0, 2.0], dtype=torch.float))
        loss_weighted, outputs = trainer_weighted.compute_loss(model, inputs, return_outputs=True)

        assert loss_weighted.item() > loss_unweighted.item()
        assert "loss" in outputs
        assert outputs["loss"] == loss_weighted


def test_compute_ner_metrics_exact_calculation() -> None:
    id2label = {0: "O", 1: "B-LOC", 2: "I-LOC", 3: "B-OBJ", 4: "I-OBJ", 5: "B-TIME", 6: "I-TIME"}

    labels = [
        [0, 1, 2, 0, 5, -100],
        [3, 4, 0, 0, 0, -100],
    ]
    preds = [
        [0, 1, 2, 0, 5, 0],
        [3, 4, 0, 0, 0, 0],
    ]

    class MockEvalPred:
        predictions = [[
            [1.0 if idx == p else 0.0 for idx in range(7)]
            for p in seq
        ] for seq in preds]
        label_ids = labels

    metrics = compute_ner_metrics(MockEvalPred(), id2label)
    assert metrics["entity_f1"] == 1.0
    assert metrics["ner_entity_f1"] == 1.0
    assert metrics["entity_precision"] == 1.0
    assert metrics["entity_recall"] == 1.0
    assert metrics["accuracy"] == 1.0
    assert metrics["loc_f1"] == 1.0
    assert metrics["obj_f1"] == 1.0
    assert metrics["time_f1"] == 1.0

    labels_err = [[0, 1, 2, 0, 5]]
    preds_err = [[0, 1, 2, 3, 0]]
    class MockEvalPredErr:
        predictions = [[
            [1.0 if idx == p else 0.0 for idx in range(7)]
            for p in seq
        ] for seq in preds_err]
        label_ids = labels_err

    metrics_err = compute_ner_metrics(MockEvalPredErr(), id2label)
    assert metrics_err["entity_precision"] == 0.5
    assert metrics_err["entity_recall"] == 0.5
    assert metrics_err["entity_f1"] == 0.5
    assert metrics_err["loc_f1"] == 1.0
    assert metrics_err["obj_f1"] == 0.0
    assert metrics_err["time_f1"] == 0.0


def test_ner_training_early_stopping_and_best_model_selection(monkeypatch: pytest.MonkeyPatch) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir) / "ner_best_model_out"

        mock_torch = mock.MagicMock()
        mock_torch.cuda.is_available.return_value = False
        mock_torch.utils.data.Dataset = object
        mock_torch.tensor = lambda x, **kw: x
        mock_torch.float = "float"
        mock_torch.long = "long"

        mock_tokenizer_instance = mock.MagicMock()
        mock_tokenizer_instance.return_value = {
            "input_ids": [101, 1000, 102],
            "attention_mask": [1, 1, 1],
            "token_type_ids": [0, 0, 0],
            "offset_mapping": [(0, 0), (0, 4), (0, 0)],
        }

        def mock_save_pretrained(save_directory: str) -> None:
            p = Path(save_directory)
            p.mkdir(parents=True, exist_ok=True)
            (p / "tokenizer_config.json").write_text("{}", encoding="utf-8")

        mock_tokenizer_instance.save_pretrained.side_effect = mock_save_pretrained

        mock_model_instance = mock.MagicMock()
        mock_trainer_instance = mock.MagicMock()

        def mock_save_model(output_dir: str) -> None:
            p = Path(output_dir)
            p.mkdir(parents=True, exist_ok=True)
            (p / "model.safetensors").write_bytes(b"TRAINED_MOCK_NER_SAFETENSORS" + b"\x00" * 64)
            (p / "config.json").write_text("{}", encoding="utf-8")

        mock_trainer_instance.save_model.side_effect = mock_save_model

        mock_state = mock.MagicMock()
        mock_state.best_model_checkpoint = str(out_dir / "checkpoint-100")
        mock_state.best_metric = 0.875
        mock_trainer_instance.state = mock_state

        captured_kwargs: dict[str, Any] = {}

        class MockTrainer:
            def __init__(self, **kwargs: Any) -> None:
                captured_kwargs.update(kwargs)

            def train(self) -> None:
                pass

            def save_model(self, out_d: str) -> None:
                mock_save_model(out_d)

            @property
            def state(self) -> Any:
                return mock_state

        mock_transformers = mock.MagicMock()
        mock_transformers.AutoTokenizer.from_pretrained.return_value = mock_tokenizer_instance
        mock_transformers.AutoModelForTokenClassification.from_pretrained.return_value = mock_model_instance
        mock_transformers.Trainer = MockTrainer
        mock_transformers.TrainingArguments = mock.MagicMock()
        mock_transformers.EarlyStoppingCallback = mock.MagicMock()

        monkeypatch.setitem(sys.modules, "torch", mock_torch)
        monkeypatch.setitem(sys.modules, "torch.utils", mock_torch.utils)
        monkeypatch.setitem(sys.modules, "torch.utils.data", mock_torch.utils.data)
        monkeypatch.setitem(sys.modules, "transformers", mock_transformers)

        config = NERTrainConfig(
            model_name_or_path="indobenchmark/indobert-base-p1",
            output_dir=str(out_dir),
            seed=42,
            max_seq_length=448,
            learning_rate=3e-5,
            batch_size=16,
            gradient_accumulation_steps=2,
            num_epochs=5,
            fp16=False,
            local_files_only=True,
            dataloader_num_workers=2,
            class_weights={"O": 1.0, "B-LOC": 2.5, "B-OBJ": 2.5, "B-TIME": 3.0},
            metric_for_best_model="entity_f1",
            greater_is_better=True,
            early_stopping_patience=2,
            version="v1.0.0",
        )

        manifest = run_train_ner(config, dry_run=False, validate_only=False)

        assert manifest is not None
        assert "indobert-ner-bio" in manifest.artifacts
        assert (out_dir / "config.json").is_file()
        assert (out_dir / "kawal_ner_metadata.json").is_file()

        assert "callbacks" in captured_kwargs
        assert len(captured_kwargs["callbacks"]) > 0

        with (out_dir / "config.json").open("r", encoding="utf-8") as f:
            cfg = json.load(f)
        assert cfg.get("model_type") != "IndoBERT-TokenClassification-BIO"

        with (out_dir / "kawal_ner_metadata.json").open("r", encoding="utf-8") as f:
            meta = json.load(f)
        assert meta["metric_for_best_model"] == "entity_f1"
        assert meta["best_model_checkpoint"] == str(out_dir / "checkpoint-100")
        assert meta["best_metric"] == 0.875
        assert meta["class_weights"] == config.class_weights


def test_ner_compact_target_extraction_multi_turn_and_drainage() -> None:
    from contracts.models import DecisionMode, TrajectoryBubble, TrajectoryTurn, TurnExpectedAction
    from scripts.train_ner import extract_grounded_object_candidates

    traj = ComplaintTrajectory(
        scenario_id="sc-drainage-test",
        family_id="drainage-overflow-turn",
        category=Category.DRAINAGE_FLOOD,
        split=DatasetSplit.TRAIN,
        observable_facts=("pintu air dan saluran drainase mampet parah", "kali ciliwung"),
        world_truth={"category": "DRAINAGE_FLOOD", "flood_depth_cm": 45},
        turns=(
            TrajectoryTurn(
                turn=1,
                bubbles=(TrajectoryBubble(source_message_id="m1", text="Lapor pintu air tersumbat dan saluran drainase mampet"),),
                observable_facts=("pintu air tersumbat",),
                expected_action=TurnExpectedAction(turn=1, allowed_actions=(DecisionMode.REQUEST_CLARIFICATION,)),
            ),
            TrajectoryTurn(
                turn=2,
                bubbles=(TrajectoryBubble(source_message_id="m2", text="Pompa drainase juga mati total sejak kemarin"),),
                observable_facts=("mesin pompa drainase otomatis mati",),
                expected_action=TurnExpectedAction(turn=2, allowed_actions=(DecisionMode.EXECUTE,)),
            ),
        ),
    )

    cands_t1 = extract_grounded_object_candidates(traj, "Lapor pintu air tersumbat dan saluran drainase mampet")
    assert "pintu air" in cands_t1
    assert "saluran drainase" in cands_t1
    assert "mampet" not in cands_t1
    assert "tersumbat" not in cands_t1

    cands_t2 = extract_grounded_object_candidates(traj, "Pompa drainase juga mati total sejak kemarin")
    assert any("pompa drainase" in c for c in cands_t2)
    assert "mati" not in cands_t2
    assert "total" not in cands_t2


def test_train_ner_dry_run_preserves_standard_config_and_writes_kawal_metadata() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir) / "ner_dry_out"
        config = NERTrainConfig(
            model_name_or_path="indobenchmark/indobert-base-p1",
            output_dir=str(out_dir),
            seed=42,
            max_seq_length=448,
            num_epochs=1,
            batch_size=16,
            local_files_only=True,
            version="v1.0.0",
        )
        manifest = run_train_ner(config, dry_run=True, validate_only=False)
        assert manifest is not None

        cfg_file = out_dir / "config.json"
        assert cfg_file.is_file()
        with cfg_file.open("r", encoding="utf-8") as f:
            cfg = json.load(f)
        assert cfg["model_type"] == "bert"
        assert cfg["architectures"] == ["BertForTokenClassification"]
        assert cfg["model_type"] != "IndoBERT-TokenClassification-BIO"

        meta_file = out_dir / "kawal_ner_metadata.json"
        assert meta_file.is_file()
        with meta_file.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        assert meta["model_type"] == "IndoBERT-TokenClassification-BIO"
        assert "tagset" in meta
        assert meta["num_labels"] == len(config.tagset)


def test_train_transformers_ner_preserves_standard_hf_config(monkeypatch: pytest.MonkeyPatch) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir) / "ner_hf_out"

        mock_torch = mock.MagicMock()
        mock_torch.cuda.is_available.return_value = False
        mock_torch.utils.data.Dataset = object
        mock_torch.tensor = lambda x, **kw: x
        mock_torch.long = "long"

        mock_tokenizer_instance = mock.MagicMock()
        mock_tokenizer_instance.return_value = {
            "input_ids": [101, 1000, 102],
            "attention_mask": [1, 1, 1],
            "token_type_ids": [0, 0, 0],
            "offset_mapping": [(0, 0), (0, 4), (0, 0)],
        }

        def mock_save_pretrained(save_directory: str) -> None:
            p = Path(save_directory)
            p.mkdir(parents=True, exist_ok=True)
            (p / "tokenizer_config.json").write_text("{}", encoding="utf-8")

        mock_tokenizer_instance.save_pretrained.side_effect = mock_save_pretrained

        mock_model_instance = mock.MagicMock()
        mock_model_instance.config.to_dict.return_value = {
            "architectures": ["BertForTokenClassification"],
            "model_type": "bert",
            "vocab_size": 50000,
        }

        mock_trainer_instance = mock.MagicMock()

        def mock_save_model(output_dir: str) -> None:
            p = Path(output_dir)
            p.mkdir(parents=True, exist_ok=True)
            (p / "model.safetensors").write_bytes(b"MOCK_WEIGHTS" + b"\x00" * 64)
            (p / "config.json").write_text(
                json.dumps({"architectures": ["BertForTokenClassification"], "model_type": "bert", "vocab_size": 50000}),
                encoding="utf-8",
            )

        mock_trainer_instance.save_model.side_effect = mock_save_model

        mock_transformers = mock.MagicMock()
        mock_transformers.AutoTokenizer.from_pretrained.return_value = mock_tokenizer_instance
        mock_transformers.AutoModelForTokenClassification.from_pretrained.return_value = mock_model_instance
        mock_transformers.Trainer.return_value = mock_trainer_instance

        monkeypatch.setitem(sys.modules, "torch", mock_torch)
        monkeypatch.setitem(sys.modules, "torch.utils", mock_torch.utils)
        monkeypatch.setitem(sys.modules, "torch.utils.data", mock_torch.utils.data)
        monkeypatch.setitem(sys.modules, "transformers", mock_transformers)

        config = NERTrainConfig(
            model_name_or_path="indobenchmark/indobert-base-p1",
            output_dir=str(out_dir),
            seed=42,
            max_seq_length=448,
            learning_rate=3e-5,
            batch_size=16,
            gradient_accumulation_steps=2,
            num_epochs=1,
            fp16=False,
            local_files_only=True,
            version="v1.0.0",
        )

        manifest = run_train_ner(config, dry_run=False, validate_only=False)
        assert manifest is not None

        cfg_file = out_dir / "config.json"
        assert cfg_file.is_file()
        with cfg_file.open("r", encoding="utf-8") as f:
            cfg = json.load(f)
        assert cfg["model_type"] == "bert"
        assert cfg["architectures"] == ["BertForTokenClassification"]

        meta_file = out_dir / "kawal_ner_metadata.json"
        assert meta_file.is_file()
        with meta_file.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        assert meta["model_type"] == "IndoBERT-TokenClassification-BIO"


def test_reconstruct_ner_bert_config_from_affected_directory() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        overwritten_cfg = {
            "model_type": "IndoBERT-TokenClassification-BIO",
            "tagset": ["O", "B-LOC", "I-LOC", "B-OBJ", "I-OBJ", "B-TIME", "I-TIME"],
            "num_labels": 7,
            "id2label": {"0": "O", "1": "B-LOC", "2": "I-LOC", "3": "B-OBJ", "4": "I-OBJ", "5": "B-TIME", "6": "I-TIME"},
            "label2id": {"O": 0, "B-LOC": 1, "I-LOC": 2, "B-OBJ": 3, "I-OBJ": 4, "B-TIME": 5, "I-TIME": 6},
            "max_seq_length": 448,
        }
        (tmp_path / "config.json").write_text(json.dumps(overwritten_cfg), encoding="utf-8")
        (tmp_path / "tagset.json").write_text(json.dumps(overwritten_cfg["tagset"]), encoding="utf-8")

        base_dapt_dir = tmp_path.parent / "dapt"
        base_dapt_dir.mkdir(parents=True, exist_ok=True)
        (base_dapt_dir / "config.json").write_text(
            json.dumps({"model_type": "bert", "hidden_size": 768, "num_hidden_layers": 12, "vocab_size": 50000}),
            encoding="utf-8",
        )

        mock_transformers = mock.MagicMock()

        class MockBertConfig:
            def __init__(self, **kwargs: Any) -> None:
                self.__dict__.update(kwargs)

            @classmethod
            def from_dict(cls, data: dict[str, Any]) -> Any:
                return cls(**data)

            @classmethod
            def from_pretrained(cls, name: str, **kwargs: Any) -> Any:
                return cls(model_type="bert", hidden_size=768, vocab_size=50000, **kwargs)

        mock_transformers.BertConfig = MockBertConfig

        with mock.patch.dict(sys.modules, {"transformers": mock_transformers}):
            cfg = reconstruct_ner_bert_config(tmp_path)
            assert cfg.model_type == "bert"
            assert cfg.architectures == ["BertForTokenClassification"]
            assert cfg.num_labels == 7
            assert cfg.id2label[1] == "B-LOC"
            assert cfg.label2id["B-LOC"] == 1


def test_load_model_for_export_ner_triggers_reconstruction_only_when_needed(monkeypatch: pytest.MonkeyPatch) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        affected_dir = tmp_path / "affected_ner"
        affected_dir.mkdir()
        (affected_dir / "config.json").write_text(
            json.dumps({"model_type": "IndoBERT-TokenClassification-BIO", "tagset": ["O", "B-LOC"]}),
            encoding="utf-8",
        )
        (affected_dir / "tagset.json").write_text(json.dumps(["O", "B-LOC"]), encoding="utf-8")

        standard_dir = tmp_path / "standard_ner"
        standard_dir.mkdir()
        (standard_dir / "config.json").write_text(
            json.dumps({"model_type": "bert", "architectures": ["BertForTokenClassification"]}),
            encoding="utf-8",
        )

        mock_transformers = mock.MagicMock()

        class MockBertConfig:
            def __init__(self, **kwargs: Any) -> None:
                self.__dict__.update(kwargs)

            @classmethod
            def from_dict(cls, data: dict[str, Any]) -> Any:
                return cls(**data)

            @classmethod
            def from_pretrained(cls, name: str, **kwargs: Any) -> Any:
                return cls(model_type="bert", hidden_size=768, vocab_size=50000, **kwargs)

        mock_transformers.BertConfig = MockBertConfig
        mock_transformers.AutoConfig.for_model.side_effect = lambda m: MockBertConfig if m == "bert" else (_ for _ in ()).throw(KeyError(m))

        mock_model = mock.MagicMock()
        mock_transformers.AutoModelForTokenClassification.from_pretrained.return_value = mock_model

        monkeypatch.setitem(sys.modules, "transformers", mock_transformers)

        model1 = load_model_for_export(affected_dir, task="ner", local_files_only=True)
        assert model1 == mock_model
        assert "config" in mock_transformers.AutoModelForTokenClassification.from_pretrained.call_args.kwargs

        mock_transformers.AutoModelForTokenClassification.from_pretrained.reset_mock()
        model2 = load_model_for_export(standard_dir, task="ner", local_files_only=True)
        assert model2 == mock_model
        assert "config" not in mock_transformers.AutoModelForTokenClassification.from_pretrained.call_args.kwargs


def test_export_onnx_real_artifacts_ner_final_parity() -> None:
    ml_deps = check_ml_dependencies("torch", "transformers", "onnx", "onnxruntime")
    if not all(ml_deps.values()):
        pytest.skip("ML dependencies not available in environment")
    ner_path = Path("artifacts/ner-final")
    if not (ner_path / "model.safetensors").is_file():
        pytest.skip("artifacts/ner-final model.safetensors not found")

    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir) / "ner_onnx_out"
        config = ONNXExportConfig(
            model_path=str(ner_path),
            output_dir=str(out_dir),
            output_filename="model.onnx",
            task="ner",
            opset=17,
            precision="fp32",
            max_seq_length=448,
            validate_output=True,
            local_files_only=True,
            seed=42,
            version="v1.0.0",
        )
        manifest = run_export_onnx(config, dry_run=False, validate_only=False)
        assert manifest is not None
        assert "indobert-ner-onnx-fp32" in manifest.artifacts
        item = manifest.artifacts["indobert-ner-onnx-fp32"]

        onnx_file = out_dir / "model.onnx"
        assert onnx_file.is_file()
        expected_sha256 = compute_file_sha256(onnx_file)
        expected_size = onnx_file.stat().st_size

        assert item.sha256 == expected_sha256
        assert len(item.sha256) == 64
        assert item.size_bytes == expected_size
        assert item.size_bytes > 0

        manifest_file = out_dir / "manifest.json"
        assert manifest_file.is_file()
        persisted_manifest = ArtifactManifest.from_file(manifest_file)
        assert "indobert-ner-onnx-fp32" in persisted_manifest.artifacts
        assert persisted_manifest.artifacts["indobert-ner-onnx-fp32"].sha256 == expected_sha256

        validation_result = persisted_manifest.validate_artifact_file(
            "indobert-ner-onnx-fp32",
            out_dir,
            raise_on_error=True,
        )
        assert validation_result.is_valid is True
        assert validation_result.expected_sha256 == expected_sha256
        assert validation_result.actual_sha256 == expected_sha256

        all_validations = persisted_manifest.validate_all(out_dir, raise_on_error=True)
        assert all(v.is_valid for v in all_validations.values())

        run_export_onnx(config, validate_only=True)


# ---------------------------------------------------------------------------
# Regression: canonical OBJ span — one deterministic hit per phenomenon
# ---------------------------------------------------------------------------

def test_pick_canonical_returns_longest_match() -> None:
    aliases = ["kartu tanda penduduk", "ktp", "status pernikahan"]
    assert _pick_canonical(aliases, "blanko kartu tanda penduduk hilang") == "kartu tanda penduduk"
    assert _pick_canonical(aliases, "minta ktp baru") == "ktp"
    assert _pick_canonical(["x y z", "x y", "x"], "ada x y disini") == "x y"


def test_pick_canonical_returns_none_when_no_match() -> None:
    assert _pick_canonical(["jalan berlubang", "aspal ambles"], "tidak ada keluhan") is None


def test_pick_canonical_requires_token_boundary() -> None:
    assert _pick_canonical(["tps"], "tempat penampungan sementara tps liar") == "tps"
    assert _pick_canonical(["nik"], "teknik pengukuran") is None


def test_extract_grounded_yields_one_obj_per_phenomenon_kartu_keluarga() -> None:
    from contracts.models import Category, DatasetSplit
    traj = ComplaintTrajectory(
        scenario_id="sc-kk",
        family_id="kk",
        category=Category.CIVIL_ADMIN,
        split=DatasetSplit.TRAIN,
        world_truth={"document": "KARTU_KELUARGA"},
        observable_facts=("kartu keluarga warga tidak bisa diproses",),
        turns=(),
    )
    text = "Lapor kartu keluarga saya tidak bisa diproses di kelurahan"
    cands = extract_grounded_object_candidates(traj, text)
    kartu_keluarga_hits = [c for c in cands if c.lower() == "kartu keluarga"]
    nik_hits = [c for c in cands if c.lower() == "nik"]
    assert len(kartu_keluarga_hits) == 1, f"Expected exactly 1 'kartu keluarga', got {kartu_keluarga_hits}"
    assert nik_hits == [], f"Bare 'nik' must not appear as OBJ candidate: {nik_hits}"


def test_extract_grounded_nik_not_spurious_obj_when_kk_not_in_text() -> None:
    from contracts.models import Category, DatasetSplit
    traj = ComplaintTrajectory(
        scenario_id="sc-kk2",
        family_id="kk2",
        category=Category.CIVIL_ADMIN,
        split=DatasetSplit.TRAIN,
        world_truth={"document": "KARTU_KELUARGA"},
        observable_facts=("berkas kependudukan ditolak petugas kelurahan",),
        turns=(),
    )
    text = "Berkas kependudukan saya ditolak petugas kelurahan tanpa alasan"
    cands = extract_grounded_object_candidates(traj, text)
    nik_hits = [c for c in cands if c.lower() == "nik"]
    assert nik_hits == [], f"Bare 'nik' must never appear as OBJ: {nik_hits}"


def test_extract_grounded_tps_overflow_no_bare_sampah() -> None:
    from contracts.models import Category, DatasetSplit
    traj = ComplaintTrajectory(
        scenario_id="sc-tps",
        family_id="tps",
        category=Category.WASTE,
        split=DatasetSplit.TRAIN,
        world_truth={"waste_type": "TPS_OVERFLOW"},
        observable_facts=("tempat penampungan sementara sudah penuh dan meluber",),
        turns=(),
    )
    text = "Tempat penampungan sementara di sini penuh meluber ke jalan"
    cands = extract_grounded_object_candidates(traj, text)
    bare_sampah = [c for c in cands if c.lower() == "sampah"]
    assert bare_sampah == [], f"Bare 'sampah' cue word must not be OBJ: {bare_sampah}"
    tps_hits = [c for c in cands if "tps" in c.lower() or "penampungan sementara" in c.lower()]
    assert len(tps_hits) >= 1, f"Expected TPS canonical hit, got: {cands}"


def test_extract_grounded_phenomena_deduplication_no_overlapping_aliases() -> None:
    from contracts.models import Category, DatasetSplit
    traj = ComplaintTrajectory(
        scenario_id="sc-road",
        family_id="road",
        category=Category.ROAD,
        split=DatasetSplit.TRAIN,
        world_truth={"infra_type": "ROAD"},
        observable_facts=("jalan berlubang parah di depan kantor",),
        turns=(),
    )
    text = "Jalan berlubang parah di depan kantor sudah lama tidak diperbaiki"
    cands = extract_grounded_object_candidates(traj, text)
    lower_cands = [c.lower() for c in cands]
    assert lower_cands.count("jalan berlubang") <= 1, f"Duplicate canonical spans: {cands}"


def test_extract_grounded_open_burning_no_asap_pekat() -> None:
    from contracts.models import Category, DatasetSplit
    traj = ComplaintTrajectory(
        scenario_id="sc-burn",
        family_id="burn",
        category=Category.WASTE,
        split=DatasetSplit.TRAIN,
        world_truth={"waste_type": "OPEN_BURNING"},
        observable_facts=("pembakaran sampah mengeluarkan asap pekat",),
        turns=(),
    )
    text = "Warga membakar sampah sehingga asap pekat mengganggu"
    cands = extract_grounded_object_candidates(traj, text)
    asap_hits = [c for c in cands if c.lower() == "asap pekat"]
    assert asap_hits == [], f"'asap pekat' is a condition, not an OBJ entity: {asap_hits}"


def test_extract_grounded_non_entity_category_cues_not_obj() -> None:
    from contracts.models import Category, DatasetSplit
    traj = ComplaintTrajectory(
        scenario_id="sc-cue",
        family_id="cue",
        category=Category.CIVIL_ADMIN,
        split=DatasetSplit.TRAIN,
        world_truth={"document": "LEGALISIR"},
        observable_facts=("pengesahan berkas membutuhkan biaya tidak resmi",),
        turns=(),
    )
    text = "Biaya fotokopi tidak resmi diminta petugas untuk pengesahan berkas"
    cands = extract_grounded_object_candidates(traj, text)
    pengesahan_hits = [c for c in cands if c.lower() == "pengesahan berkas"]
    assert pengesahan_hits == [], f"'pengesahan berkas' is a process phrase, not an OBJ entity: {pengesahan_hits}"
    biaya_hits = [c for c in cands if "biaya fotokopi" in c.lower()]
    assert biaya_hits, f"Expected 'biaya fotokopi tidak resmi' as OBJ: {cands}"


# ---------------------------------------------------------------------------
# 17. Hifi Trajectories, RTX 3060 FP16 Presets, Canonical Spans & DAPT Train Provenance (No Network)
# ---------------------------------------------------------------------------

def test_multitask_hifi_trajectory_jsonl_without_schema_label_assumptions(tmp_path: Path) -> None:
    from scripts.train_multitask import (
        extract_trajectory_multitask_samples,
        load_or_generate_dataset,
        run_train_multitask,
    )

    jsonl_path = tmp_path / "hifi_multitask.jsonl"
    records = [
        {
            "scenario_id": "hifi-multitask-001",
            "family_id": "fam-hifi-wildlife",
            "split": "train",
            "category": "WILDLIFE_CONFLICT",
            "world_truth": {
                "intent": "URGENT_EVACUATION",
                "risk": "EXTREME",
                "completeness": "DETAILED",
                "unmodeled_confidence": 0.98,
            },
            "extra_provenance_meta": {"generator_version": "hifi-v2", "cluster": "8k-synth"},
            "turns": [
                {
                    "turn": 1,
                    "bubbles": [
                        {"text": "Kawanan monyet liar menyerang pemukiman warga dan merusak atap rumah."}
                    ],
                }
            ],
        },
        {
            "scenario_id": "hifi-multitask-002",
            "family_id": "fam-hifi-drain",
            "split": "dev",
            "category": "DRAINAGE_FLOOD",
            "world_truth": {
                "intent": "COMPLAINT",
                "risk": "MEDIUM",
                "completeness": "SUFFICIENT",
            },
            "turns": [
                {
                    "turn": 1,
                    "bubbles": [
                        {"text": "Saluran gorong-gorong air di kampung manggis mampet tersumbat sampah."}
                    ],
                }
            ],
        },
        {
            "scenario_id": "hifi-multitask-003",
            "family_id": "fam-hifi-admin",
            "split": "test",
            "category": "CIVIL_ADMIN",
            "world_truth": {
                "intent": "INQUIRY",
                "risk": "LOW",
                "completeness": "SUFFICIENT",
            },
            "turns": [
                {
                    "turn": 1,
                    "bubbles": [
                        {"text": "Bagaimana prosedur perpanjangan kartu tanda penduduk elektronik keliling?"}
                    ],
                }
            ],
        },
    ]

    with jsonl_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    trajectories = load_or_generate_dataset(str(jsonl_path), seed=42, audit_splits=True)
    assert len(trajectories) == 3

    samples = extract_trajectory_multitask_samples(trajectories)
    assert len(samples) >= 3

    wildlife_sample = next(s for s in samples if s["scenario_id"] == "hifi-multitask-001")
    assert wildlife_sample["category"] == "WILDLIFE_CONFLICT"
    assert wildlife_sample["intent"] == "URGENT_EVACUATION"
    assert wildlife_sample["risk"] == "EXTREME"
    assert wildlife_sample["completeness"] == "DETAILED"
    assert wildlife_sample["split"] == "train"

    out_dir = tmp_path / "hifi_multitask_out"
    config = MultitaskTrainConfig(
        dataset_path=str(jsonl_path),
        output_dir=str(out_dir),
        seed=42,
    )
    manifest = run_train_multitask(config, dry_run=True)
    assert manifest is not None
    assert (out_dir / "config.json").is_file()

    with open(out_dir / "config.json", "r", encoding="utf-8") as f:
        cfg_out = json.load(f)
    assert "WILDLIFE_CONFLICT" in cfg_out["heads"]["category"]["classes"]
    assert "URGENT_EVACUATION" in cfg_out["heads"]["intent"]["classes"]
    assert "EXTREME" in cfg_out["heads"]["risk"]["classes"]


def test_ner_hifi_trajectory_preserves_canonical_spans(tmp_path: Path) -> None:
    from scripts.train_ner import (
        extract_trajectory_ner_samples,
        load_or_generate_dataset,
        run_train_ner,
    )

    jsonl_path = tmp_path / "hifi_ner.jsonl"
    canonical_spans = [(0, 15, "OBJ"), (25, 43, "LOC"), (44, 56, "TIME")]
    records = [
        {
            "scenario_id": "hifi-ner-001",
            "family_id": "fam-hifi-water-leak",
            "split": "train",
            "category": "CLEAN_WATER",
            "unmodeled_extra_metadata": {"annotator": "gold-standard-m3", "batch": 8000},
            "canonical_spans": canonical_spans,
            "turns": [
                {
                    "turn": 1,
                    "bubbles": [
                        {
                            "text": "Pipa air bersih bocor di Jalan Veteran Selatan kemarin sore.",
                            "canonical_spans": canonical_spans,
                            "bubble_meta": {"device": "mobile"},
                        }
                    ],
                }
            ],
        },
        {
            "scenario_id": "hifi-ner-002",
            "family_id": "fam-hifi-road-hole",
            "split": "dev",
            "category": "ROAD",
            "turns": [
                {
                    "turn": 1,
                    "bubbles": [
                        {
                            "text": "Jalan berlubang parah di Jalan Kaliurang Km 5.",
                            "canonical_spans": [
                                {"start": 0, "end": 20, "label": "B-OBJ"},
                                {"start": 24, "end": 45, "label": "LOC"},
                            ],
                        }
                    ],
                }
            ],
        },
    ]

    with jsonl_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    trajectories = load_or_generate_dataset(str(jsonl_path), seed=42, audit_splits=True)
    assert len(trajectories) == 2

    samples = extract_trajectory_ner_samples(trajectories)
    assert len(samples) >= 2

    sample_1 = next(s for s in samples if s["scenario_id"] == "hifi-ner-001" and s["granularity"] == "bubble")
    spans_1 = sample_1["spans"]
    assert (0, 15, "OBJ") in spans_1
    assert (25, 43, "LOC") in spans_1
    assert (44, 56, "TIME") in spans_1

    sample_2 = next(s for s in samples if s["scenario_id"] == "hifi-ner-002" and s["granularity"] == "bubble")
    spans_2 = sample_2["spans"]
    assert (0, 20, "OBJ") in spans_2
    assert (25, 45, "LOC") in spans_2

    out_dir = tmp_path / "hifi_ner_out"
    ner_cfg = NERTrainConfig(
        dataset_path=str(jsonl_path),
        output_dir=str(out_dir),
        seed=42,
    )
    manifest = run_train_ner(ner_cfg, dry_run=True)
    assert manifest is not None
    assert (out_dir / "config.json").is_file()
    assert (out_dir / "kawal_ner_metadata.json").is_file()


def test_family_split_audit_on_hifi_trajectories(tmp_path: Path) -> None:
    from scripts.train_multitask import load_or_generate_dataset

    clean_jsonl = tmp_path / "clean_hifi.jsonl"
    clean_records = [
        {
            "scenario_id": "clean-1",
            "family_id": "fam-clean-alpha",
            "split": "train",
            "category": "ROAD",
            "turns": [{"turn": 1, "bubbles": [{"text": "Penerangan jalan umum padam di flyover"}]}],
        },
        {
            "scenario_id": "clean-2",
            "family_id": "fam-clean-beta",
            "split": "dev",
            "category": "ROAD",
            "turns": [{"turn": 1, "bubbles": [{"text": "Aspal jalan amblas di samping jembatan"}]}],
        },
    ]
    with clean_jsonl.open("w", encoding="utf-8") as f:
        for r in clean_records:
            f.write(json.dumps(r) + "\n")

    clean_loaded = load_or_generate_dataset(str(clean_jsonl), seed=42, audit_splits=True)
    assert len(clean_loaded) == 2

    leaking_jsonl = tmp_path / "leaking_hifi.jsonl"
    leaking_records = [
        {
            "scenario_id": "leak-1",
            "family_id": "fam-shared-leak",
            "split": "train",
            "category": "ROAD",
            "turns": [{"turn": 1, "bubbles": [{"text": "Penerangan jalan umum padam di flyover"}]}],
        },
        {
            "scenario_id": "leak-2",
            "family_id": "fam-shared-leak",
            "split": "dev",
            "category": "ROAD",
            "turns": [{"turn": 1, "bubbles": [{"text": "Aspal jalan amblas di samping jembatan"}]}],
        },
    ]
    with leaking_jsonl.open("w", encoding="utf-8") as f:
        for r in leaking_records:
            f.write(json.dumps(r) + "\n")

    with pytest.raises(DatasetAuditError) as exc_info:
        load_or_generate_dataset(str(leaking_jsonl), seed=42, audit_splits=True)
    assert "fam-shared-leak" in str(exc_info.value) or "Family overlap" in str(exc_info.value)


def test_rtx3060_safe_fp16_configurations() -> None:
    from scripts.train_multitask import (
        MultitaskTrainConfig,
        get_rtx3060_multitask_config,
    )
    from scripts.train_ner import (
        NERTrainConfig,
        get_rtx3060_ner_config,
    )
    from scripts.train_dapt import (
        DAPTConfig,
        get_rtx3060_dapt_config,
    )

    mt_cfg = get_rtx3060_multitask_config("data/trajectories.jsonl")
    assert mt_cfg.batch_size == 16
    assert mt_cfg.gradient_accumulation_steps == 2
    assert mt_cfg.fp16 is True
    assert mt_cfg.max_seq_length == 448
    assert mt_cfg.learning_rate == 2e-5
    assert mt_cfg.dataloader_num_workers == 2
    assert mt_cfg.save_total_limit == 2

    ner_cfg = get_rtx3060_ner_config("data/trajectories.jsonl")
    assert ner_cfg.batch_size == 16
    assert ner_cfg.gradient_accumulation_steps == 2
    assert ner_cfg.fp16 is True
    assert ner_cfg.max_seq_length == 448
    assert ner_cfg.learning_rate == 3e-5
    assert ner_cfg.dataloader_num_workers == 2
    assert ner_cfg.save_total_limit == 2

    dapt_cfg = get_rtx3060_dapt_config("data/corpus.jsonl")
    assert dapt_cfg.batch_size == 8
    assert dapt_cfg.gradient_accumulation_steps == 4
    assert dapt_cfg.fp16 is True
    assert dapt_cfg.max_seq_length == 448
    assert dapt_cfg.learning_rate == 2e-5
    assert dapt_cfg.dataloader_num_workers == 2
    assert dapt_cfg.save_total_limit == 2


def test_rtx3060_cli_flags_and_dry_run(tmp_path: Path) -> None:
    mt_dir = tmp_path / "mt_cli_3060"
    res_mt = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.train_multitask",
            "--rtx3060",
            "--dry-run",
            "--output-dir",
            str(mt_dir),
        ],
        capture_output=True,
        text=True,
    )
    assert res_mt.returncode == 0, res_mt.stderr
    with open(mt_dir / "config.json", "r", encoding="utf-8") as f:
        mt_json = json.load(f)
    assert mt_json["batch_size"] == 16
    assert mt_json["gradient_accumulation_steps"] == 2
    assert mt_json["fp16"] is True

    ner_dir = tmp_path / "ner_cli_3060"
    res_ner = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.train_ner",
            "--rtx3060",
            "--dry-run",
            "--output-dir",
            str(ner_dir),
        ],
        capture_output=True,
        text=True,
    )
    assert res_ner.returncode == 0, res_ner.stderr
    with open(ner_dir / "config.json", "r", encoding="utf-8") as f:
        ner_json = json.load(f)
    assert ner_json["batch_size"] == 16
    assert ner_json["gradient_accumulation_steps"] == 2
    assert ner_json["fp16"] is True

    dapt_dir = tmp_path / "dapt_cli_3060"
    res_dapt = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.train_dapt",
            "--rtx3060",
            "--dry-run",
            "--output-dir",
            str(dapt_dir),
        ],
        capture_output=True,
        text=True,
    )
    assert res_dapt.returncode == 0, res_dapt.stderr
    with open(dapt_dir / "manifest.json", "r", encoding="utf-8") as f:
        dapt_manifest = json.load(f)
    assert "indobert-dapt" in dapt_manifest["artifacts"]


def test_dapt_corpus_strictly_enforces_train_only_provenance(tmp_path: Path) -> None:
    mixed_jsonl = tmp_path / "mixed_corpus.jsonl"
    records = [
        {"split": "train", "text": "Kalimat resmi train masyarakat tentang drainase mampet."},
        {"split": "dev", "text": "Kalimat dev evaluasi yang tidak boleh masuk DAPT pretraining."},
        {"split": "test", "text": "Kalimat test heldout rahasia yang terlarang bocor ke DAPT."},
        {"world_truth": {"split": "train"}, "text": "Laporan jalan amblas dari pelapor warga train."},
        {"world_truth": {"split": "test"}, "text": "Laporan test rahasia independen tidak boleh bocor."},
    ]
    with mixed_jsonl.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    corpus = load_corpus(str(mixed_jsonl), pii_scrubbing=True)
    assert len(corpus) == 2
    assert any("drainase mampet" in c for c in corpus)
    assert any("pelapor warga train" in c for c in corpus)
    assert not any("dev evaluasi" in c for c in corpus)
    assert not any("test heldout" in c for c in corpus)
    assert not any("independen" in c for c in corpus)

    fallback_corpus = load_corpus(None, seed=42)
    assert len(fallback_corpus) > 0


def test_training_scripts_strict_no_network_guarantee(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import socket

    def _block_connect(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("Network access attempted during offline training execution!")

    monkeypatch.setattr(socket.socket, "connect", _block_connect)

    mt_out = tmp_path / "no_net_mt"
    ner_out = tmp_path / "no_net_ner"
    dapt_out = tmp_path / "no_net_dapt"

    mt_cfg = MultitaskTrainConfig(
        output_dir=str(mt_out),
        local_files_only=True,
    )
    mt_manifest = run_train_multitask(mt_cfg, dry_run=True)
    assert mt_manifest is not None

    ner_cfg = NERTrainConfig(
        output_dir=str(ner_out),
        local_files_only=True,
    )
    ner_manifest = run_train_ner(ner_cfg, dry_run=True)
    assert ner_manifest is not None

    dapt_cfg = DAPTConfig(
        output_dir=str(dapt_out),
        local_files_only=True,
    )
    dapt_manifest = run_train_dapt(dapt_cfg, dry_run=True)
    assert dapt_manifest is not None


# ---------------------------------------------------------------------------
# 18. Hifi Loading Contract, Canonical Spans Ground Truth & OOD Contamination Rejection
# ---------------------------------------------------------------------------

def test_hifi_ner_exact_canonical_spans_without_heuristic_fallback(tmp_path: Path) -> None:
    from scripts.train_ner import extract_trajectory_ner_samples, load_or_generate_dataset

    jsonl_path = tmp_path / "exact_canonical_spans.jsonl"
    # Text contains clear heuristic patterns: "lampu jalan padam" (OBJ), "Jalan Magelang Km 4" (LOC), "kemarin malam" (TIME).
    # Record 1 has canonical_spans ONLY covering the OBJ (0, 17, "OBJ").
    # It must NOT include LOC or TIME from heuristic fallback.
    text_1 = "Lampu jalan padam di Jalan Magelang Km 4 kemarin malam."
    records = [
        {
            "scenario_id": "hifi-exact-001",
            "family_id": "fam-hifi-exact-spans",
            "split": "train",
            "category": "ROAD",
            "unmodeled_annotator_notes": "Annotated exclusively for primary infrastructure defect",
            "turns": [
                {
                    "turn": 1,
                    "bubbles": [
                        {
                            "text": text_1,
                            "canonical_spans": [(0, 17, "OBJ")],
                            "bubble_meta": {"annotator_id": "auditor_01"},
                        }
                    ],
                }
            ],
        },
        {
            # Record 2 without canonical_spans should fallback to heuristic extraction
            "scenario_id": "hifi-heuristic-fallback-002",
            "family_id": "fam-hifi-fallback",
            "split": "dev",
            "category": "DRAINAGE_FLOOD",
            "turns": [
                {
                    "turn": 1,
                    "bubbles": [
                        {
                            "text": "Saluran gorong-gorong ambrol parah di Jalan Kaliurang Km 9 kemarin siang.",
                        }
                    ],
                }
            ],
        },
    ]

    with jsonl_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    trajectories = load_or_generate_dataset(str(jsonl_path), seed=42, audit_splits=True)
    assert len(trajectories) == 2

    samples = extract_trajectory_ner_samples(trajectories)
    assert len(samples) >= 2

    # Sample 1 must have EXACTLY the single canonical span and no heuristic additions
    sample_exact = next(s for s in samples if s["scenario_id"] == "hifi-exact-001" and s["granularity"] == "bubble")
    assert sample_exact["spans"] == [(0, 17, "OBJ")]

    # Sample 2 (no canonical spans) must have extracted multiple spans via heuristic fallback (OBJ, LOC, TIME)
    sample_fallback = next(s for s in samples if s["scenario_id"] == "hifi-heuristic-fallback-002" and s["granularity"] == "bubble")
    assert len(sample_fallback["spans"]) > 1
    assert any(lbl == "OBJ" for _, _, lbl in sample_fallback["spans"])
    assert any(lbl == "LOC" for _, _, lbl in sample_fallback["spans"])
    assert any(lbl == "TIME" for _, _, lbl in sample_fallback["spans"])


def test_hifi_ner_canonical_spans_preserved_through_pydantic_conversion() -> None:
    from contracts.models import Category, ComplaintTrajectory, DatasetSplit, TrajectoryBubble, TrajectoryTurn, TurnExpectedAction, DecisionMode
    from scripts.train_ner import (
        _preserve_canonical_spans_in_record,
        _parse_trajectory_entry,
        extract_trajectory_ner_samples,
    )

    # 1. Raw record with extra forbidden fields and canonical_spans
    raw_record = {
        "scenario_id": "hifi-pydantic-001",
        "family_id": "fam-hifi-pydantic",
        "split": "train",
        "category": "CLEAN_WATER",
        "unmodeled_extra_metadata": {"qc": "passed", "version": "hifi-v3"},
        "canonical_spans": [(0, 15, "OBJ")],
        "turns": [
            {
                "turn": 1,
                "bubbles": [
                    {
                        "source_message_id": "msg_t1_b1",
                        "text": "Pipa bocor parah di Jalan Malioboro tadi pagi.",
                        "canonical_spans": [(0, 16, "OBJ"), (20, 35, "LOC")],
                        "bubble_extra": "forbidden_by_pydantic_extra_forbid",
                    }
                ],
            }
        ],
    }

    # Verify _preserve_canonical_spans_in_record copies spans into world_truth
    _preserve_canonical_spans_in_record(raw_record)
    assert "canonical_spans" in raw_record["world_truth"]
    assert "bubble_canonical_spans" in raw_record["world_truth"]
    assert "msg_t1_b1" in raw_record["world_truth"]["bubble_canonical_spans"]

    # 2. Extract from raw dict
    samples_from_dict = extract_trajectory_ner_samples([raw_record])
    sample_dict = next(s for s in samples_from_dict if s["granularity"] == "bubble")
    assert (0, 16, "OBJ") in sample_dict["spans"]
    assert (20, 35, "LOC") in sample_dict["spans"]

    # 3. Create explicit Pydantic ComplaintTrajectory with canonical_spans stored in world_truth
    clean_bubble = TrajectoryBubble(
        source_message_id="msg_t1_b1",
        text="Pipa bocor parah di Jalan Malioboro tadi pagi.",
        offset_seconds=0,
    )
    clean_turn = TrajectoryTurn(
        turn=1,
        bubbles=(clean_bubble,),
        expected_action=TurnExpectedAction(turn=1, allowed_actions=(DecisionMode.EXECUTE,)),
    )
    pydantic_traj = ComplaintTrajectory(
        scenario_id="hifi-pydantic-002",
        family_id="fam-hifi-pydantic-2",
        split=DatasetSplit.TRAIN,
        category=Category.CLEAN_WATER,
        world_truth={
            "canonical_spans": [(0, 16, "OBJ"), (20, 35, "LOC")],
            "bubble_canonical_spans": {"msg_t1_b1": [(0, 16, "OBJ"), (20, 35, "LOC")]},
        },
        turns=(clean_turn,),
    )

    # Extract from ComplaintTrajectory instance
    samples_from_model = extract_trajectory_ner_samples([pydantic_traj])
    sample_model = next(s for s in samples_from_model if s["granularity"] == "bubble")
    assert sample_model["spans"] == [(0, 16, "OBJ"), (20, 35, "LOC")]


def test_empty_canonical_spans_treated_same_as_present_empty() -> None:
    """Issue 1 & 2: Empty canonical_spans=() must not trigger regex fallback, and typed objects must be handled."""
    from contracts.models import (
        Category,
        ComplaintTrajectory,
        DatasetSplit,
        DecisionMode,
        TrajectoryBubble,
        TrajectoryTurn,
        TurnExpectedAction,
    )
    # Bubble with empty canonical_spans tuple
    clean_bubble = TrajectoryBubble(
        source_message_id="msg_empty_1",
        text="Aspal ambles parah di Jalan Kaliurang KM 12.",
        offset_seconds=0,
        canonical_spans=(),
    )
    clean_turn = TrajectoryTurn(
        turn=1,
        bubbles=(clean_bubble,),
        expected_action=TurnExpectedAction(turn=1, allowed_actions=(DecisionMode.EXECUTE,)),
    )
    pydantic_traj = ComplaintTrajectory(
        scenario_id="hifi-empty-canonical-001",
        family_id="fam-empty-1",
        split=DatasetSplit.TRAIN,
        category=Category.ROAD,
        turns=(clean_turn,),
    )

    samples = extract_trajectory_ner_samples([pydantic_traj])
    bubble_sample = next(s for s in samples if s["granularity"] == "bubble")
    # Empty canonical spans must yield empty spans, NOT regex fallback matching Aspal/Jalan
    assert bubble_sample["spans"] == []

    # Also test dict with empty canonical_spans list/tuple
    dict_record = {
        "scenario_id": "hifi-empty-dict-002",
        "family_id": "fam-empty-2",
        "split": "train",
        "category": "ROAD",
        "turns": [
            {
                "turn": 1,
                "bubbles": [
                    {
                        "source_message_id": "msg_empty_2",
                        "text": "Aspal ambles parah di Jalan Kaliurang KM 12.",
                        "canonical_spans": [],
                    }
                ],
            }
        ],
    }
    dict_samples = extract_trajectory_ner_samples([dict_record])
    dict_bubble_sample = next(s for s in dict_samples if s["granularity"] == "bubble")
    assert dict_bubble_sample["spans"] == []


def test_world_truth_projection_to_full_text_uses_projected_offsets() -> None:
    """Issue 3: world_truth canonical spans projection to full_text must use projected offsets."""
    from scripts.train_ner import _preserve_canonical_spans_in_record
    b1_text = "Pipa bocor parah."
    b2_text = "Lokasi di Jalan Malioboro."
    # b1 has OBJ at (0, 10): "Pipa bocor"
    # b2 has LOC at (10, 25): "Jalan Malioboro"
    # In full text: "Pipa bocor parah.\nLokasi di Jalan Malioboro."
    # b1 offset = 0 -> span is (0, 10, OBJ)
    # b2 offset = len(b1_text) + 1 = 18 -> span is (18 + 10, 18 + 25, LOC) = (28, 43, LOC)

    raw_record = {
        "scenario_id": "hifi-proj-001",
        "family_id": "fam-proj-1",
        "split": "train",
        "category": "CLEAN_WATER",
        "turns": [
            {
                "turn": 1,
                "bubbles": [
                    {
                        "source_message_id": "b1",
                        "text": b1_text,
                        "canonical_spans": [(0, 10, "OBJ")],
                    },
                    {
                        "source_message_id": "b2",
                        "text": b2_text,
                        "canonical_spans": [(10, 25, "LOC")],
                    },
                ],
            }
        ],
    }

    _preserve_canonical_spans_in_record(raw_record)
    wt_spans = raw_record["world_truth"]["canonical_spans"]
    assert (0, 10, "OBJ") in wt_spans
    assert (28, 43, "LOC") in wt_spans
    # Verify it does NOT contain duplicated unprojected (10, 25, "LOC")
    assert (10, 25, "LOC") not in wt_spans

    samples = extract_trajectory_ner_samples([raw_record])
    full_sample = next(s for s in samples if s["granularity"] == "full")
    assert (0, 10, "OBJ") in full_sample["spans"]
    assert (28, 43, "LOC") in full_sample["spans"]
    full_text = full_sample["text"]
    assert full_text[0:10] == "Pipa bocor"
    assert full_text[28:43] == "Jalan Malioboro"


def test_hifi_multitask_world_truth_reliable_extraction_and_provenance() -> None:
    from contracts.models import Category, ComplaintTrajectory, DatasetSplit, TrajectoryBubble, TrajectoryTurn, TurnExpectedAction, DecisionMode
    from scripts.train_multitask import extract_trajectory_multitask_samples

    # 1. Dict case: world_truth values must override conflicting top-level values
    dict_record = {
        "scenario_id": "hifi-wt-override-001",
        "family_id": "fam-hifi-wt-1",
        "split": "train",
        "category": "ROAD",
        "intent": "FEEDBACK",
        "risk": "LOW",
        "completeness": "INCOMPLETE",
        "world_truth": {
            "category": "CLEAN_WATER",
            "intent": "INQUIRY",
            "risk": "URGENT",
            "completeness": "SUFFICIENT",
            "provenance": "hifi_synthetic",
        },
        "turns": [
            {
                "turn": 1,
                "bubbles": [{"text": "Pipa air bersih PDAM bocor meluber ke perumahan."}],
            }
        ],
    }

    samples = extract_trajectory_multitask_samples([dict_record])
    assert len(samples) >= 1
    sample = samples[0]
    assert sample["category"] == "CLEAN_WATER"
    assert sample["intent"] == "INQUIRY"
    assert sample["risk"] == "URGENT"
    assert sample["completeness"] == "SUFFICIENT"
    assert sample["provenance"] == "hifi_synthetic"

    # 2. Pydantic ComplaintTrajectory case: world_truth must be used reliably
    bubble = TrajectoryBubble(source_message_id="msg_01", text="Laporan tumpukan sampah liar di pasar.")
    turn = TrajectoryTurn(turn=1, bubbles=(bubble,), expected_action=TurnExpectedAction(turn=1, allowed_actions=(DecisionMode.EXECUTE,)))
    model_traj = ComplaintTrajectory(
        scenario_id="hifi-wt-override-002",
        family_id="fam-hifi-wt-2",
        split=DatasetSplit.TRAIN,
        category=Category.ROAD,  # Model has ROAD, but world_truth specifies WASTE
        world_truth={
            "category": "WASTE",
            "intent": "COMPLAINT",
            "risk": "HIGH",
            "completeness": "SUFFICIENT",
            "provenance": "hifi_synthetic",
        },
        turns=(turn,),
    )

    model_samples = extract_trajectory_multitask_samples([model_traj])
    assert len(model_samples) >= 1
    m_sample = model_samples[0]
    assert m_sample["category"] == "WASTE"
    assert m_sample["intent"] == "COMPLAINT"
    assert m_sample["risk"] == "HIGH"
    assert m_sample["completeness"] == "SUFFICIENT"
    assert m_sample["provenance"] == "hifi_synthetic"


def test_multitask_rejects_ood_training_contamination(tmp_path: Path) -> None:
    from scripts import DatasetAuditError
    from scripts.train_multitask import extract_trajectory_multitask_samples, load_or_generate_dataset

    # Case A: OOD canary scenario ID placed in train split -> must raise DatasetAuditError
    ood_record_scen = {
        "scenario_id": "ood_canary_0001",
        "family_id": "fam_clean_001",
        "split": "train",
        "category": "ROAD",
        "turns": [{"turn": 1, "bubbles": [{"text": "Canary test text"}]}],
    }
    with pytest.raises(DatasetAuditError, match="OOD training contamination detected"):
        extract_trajectory_multitask_samples([ood_record_scen])

    # Case B: OOD canary family ID placed in train split -> must raise DatasetAuditError
    ood_record_fam = {
        "scenario_id": "scenario_normal_001",
        "family_id": "fam_ood_canary_001",
        "split": "train",
        "category": "ROAD",
        "turns": [{"turn": 1, "bubbles": [{"text": "Canary test text"}]}],
    }
    with pytest.raises(DatasetAuditError, match="OOD training contamination detected"):
        extract_trajectory_multitask_samples([ood_record_fam])

    # Case C: is_ood flag or synthetic_ood provenance placed in train split -> must raise DatasetAuditError
    ood_record_prov = {
        "scenario_id": "scenario_normal_002",
        "family_id": "fam_clean_002",
        "split": "train",
        "category": "ROAD",
        "provenance": "synthetic_ood",
        "turns": [{"turn": 1, "bubbles": [{"text": "Canary test text"}]}],
    }
    with pytest.raises(DatasetAuditError, match="OOD training contamination detected"):
        extract_trajectory_multitask_samples([ood_record_prov])

    # Case D: OOD canary file with train split loaded via load_or_generate_dataset -> must raise DatasetAuditError
    ood_jsonl = tmp_path / "ood_contaminated_train.jsonl"
    with ood_jsonl.open("w", encoding="utf-8") as f:
        f.write(json.dumps(ood_record_scen) + "\n")

    with pytest.raises(DatasetAuditError, match="OOD training contamination detected"):
        load_or_generate_dataset(str(ood_jsonl), seed=42, audit_splits=True)

    # Case E: Legitimate OOD record with split="ood" or split="test" -> allowed, no error
    legit_ood = {
        "scenario_id": "ood_canary_0002",
        "family_id": "fam_ood_0002",
        "split": "ood",
        "category": "ROAD",
        "turns": [{"turn": 1, "bubbles": [{"text": "Legitimate canary evaluation holdout text"}]}],
    }
    samples_ood = extract_trajectory_multitask_samples([legit_ood])
    assert len(samples_ood) >= 1
    assert samples_ood[0]["split"] == "ood"


def test_ner_rejects_ood_training_contamination(tmp_path: Path) -> None:
    from scripts import DatasetAuditError
    from scripts.train_ner import extract_trajectory_ner_samples, load_or_generate_dataset

    ood_ner_record = {
        "scenario_id": "ood_canary_ner_001",
        "family_id": "fam_ood_ner_001",
        "split": "train",
        "category": "ROAD",
        "canonical_spans": [(0, 10, "OBJ")],
        "turns": [{"turn": 1, "bubbles": [{"text": "Pipa bocor di jalan tol"}]}],
    }

    with pytest.raises(DatasetAuditError, match="OOD training contamination detected"):
        extract_trajectory_ner_samples([ood_ner_record])

    ood_ner_jsonl = tmp_path / "ood_ner_train.jsonl"
    with ood_ner_jsonl.open("w", encoding="utf-8") as f:
        f.write(json.dumps(ood_ner_record) + "\n")

    with pytest.raises(DatasetAuditError, match="OOD training contamination detected"):
        load_or_generate_dataset(str(ood_ner_jsonl), seed=42, audit_splits=True)


def test_hifi_generated_dataset_ner_and_multitask_training_dry_runs(tmp_path: Path) -> None:
    from scripts.generate_m3_hifi import generate_hifi_dataset
    from scripts import NERTrainConfig, MultitaskTrainConfig, run_train_ner, run_train_multitask

    hifi_dir = tmp_path / "hifi_gen"
    generate_hifi_dataset(hifi_dir, families_per_category=1, variants_per_family=1)

    ner_dir = tmp_path / "ner_hifi_dry_run"
    ner_cfg = NERTrainConfig(
        dataset_path=str(hifi_dir / "train.jsonl"),
        output_dir=str(ner_dir),
        seed=42,
    )
    ner_manifest = run_train_ner(ner_cfg, dry_run=True)
    assert ner_manifest is not None
    assert (ner_dir / "config.json").is_file()

    mt_dir = tmp_path / "mt_hifi_dry_run"
    mt_cfg = MultitaskTrainConfig(
        dataset_path=str(hifi_dir / "train.jsonl"),
        output_dir=str(mt_dir),
        seed=42,
    )
    mt_manifest = run_train_multitask(mt_cfg, dry_run=True)
    assert mt_manifest is not None
    assert (mt_dir / "config.json").is_file()


def test_compute_dataset_lineage_file_and_synthetic(tmp_path: Path) -> None:
    text_corpus = tmp_path / "raw_text_corpus.txt"
    lines = [
        "Laporan kerusakan lampu jalan di jalan merdeka timur.\n",
        "Saluran air mampet mengakibatkan genangan setinggi dua puluh sentimeter.\n",
        "Jalan berlubang cukup parah di depan balai kota membahayakan pengendara motor.\n",
    ]
    text_corpus.write_text("".join(lines), encoding="utf-8")

    expected_sha = compute_file_sha256(text_corpus)
    lineage_text = compute_dataset_lineage(text_corpus, corpus=lines)
    assert lineage_text["is_synthetic"] is False
    assert lineage_text["source"] == "file"
    assert lineage_text["corpus_basename"] == "raw_text_corpus.txt"
    assert lineage_text["corpus_sha256"] == expected_sha
    assert lineage_text["sha256"] == expected_sha
    assert lineage_text["record_count"] == 3
    assert lineage_text["line_count"] == 3
    assert lineage_text["raw_record_count"] == 3

    json_corpus = tmp_path / "array_corpus.json"
    json_data = [
        {"split": "train", "text": "Aduan warga terkait tumpukan sampah liar di trotoar."},
        {"split": "train", "text": "Pohon tumbang menutup sebagian jalur busway koridor satu."},
    ]
    json_corpus.write_text(json.dumps(json_data), encoding="utf-8")

    lineage_json = compute_dataset_lineage(json_corpus, corpus=["Aduan warga", "Pohon tumbang"])
    assert lineage_json["raw_record_count"] == 2
    assert lineage_json["record_count"] == 2
    assert lineage_json["corpus_sha256"] == compute_file_sha256(json_corpus)

    lineage_synth = compute_dataset_lineage(None, corpus=["Baris satu", "Baris dua"])
    assert lineage_synth["is_synthetic"] is True
    assert lineage_synth["source"] == "synthetic_generator"
    assert lineage_synth["corpus_path"] is None
    assert lineage_synth["corpus_basename"] is None
    assert lineage_synth["record_count"] == 2
    assert lineage_synth["line_count"] == 2
    assert lineage_synth["sha256"] is not None

    with pytest.raises(FileNotFoundError):
        compute_dataset_lineage(tmp_path / "missing_file.txt")


def test_dapt_dry_run_generates_auditable_dataset_lineage(tmp_path: Path) -> None:
    corpus_file = tmp_path / "sample_corpus.txt"
    corpus_lines = [
        "Jalan amblas di dekat jembatan layang segera ditangani petugas dinas terkait.\n",
        "Penerangan jalan umum padam sejak tiga hari lalu di kawasan perumahan griya asri.\n",
    ]
    corpus_file.write_text("".join(corpus_lines), encoding="utf-8")
    expected_sha = compute_file_sha256(corpus_file)

    out_dir = tmp_path / "dapt_lineage_out"
    config = DAPTConfig(
        corpus_path=str(corpus_file),
        output_dir=str(out_dir),
        seed=42,
        version="v1.2.0",
    )

    manifest = run_train_dapt(config, dry_run=True, validate_only=False)
    assert manifest is not None
    assert "indobert-dapt" in manifest.artifacts
    assert manifest.artifacts["indobert-dapt"].version == "v1.2.0"

    manifest_file = out_dir / "manifest.json"
    assert manifest_file.is_file()
    loaded_manifest = ArtifactManifest.from_file(manifest_file)
    assert loaded_manifest.manifest_version == "1.0.0"
    val_results = loaded_manifest.validate_all(out_dir, raise_on_error=True)
    assert val_results["indobert-dapt"].is_valid is True

    dapt_meta_file = out_dir / "dapt_metadata.json"
    assert dapt_meta_file.is_file()
    with dapt_meta_file.open("r", encoding="utf-8") as f:
        meta = json.load(f)

    assert meta["model_type"] == "IndoBERT-DAPT"
    assert meta["version"] == "v1.2.0"
    assert "dataset_lineage" in meta
    lineage = meta["dataset_lineage"]
    assert lineage["corpus_basename"] == "sample_corpus.txt"
    assert lineage["corpus_sha256"] == expected_sha
    assert lineage["sha256"] == expected_sha
    assert lineage["record_count"] >= 1
    assert lineage["line_count"] >= 1
    assert meta["corpus_sha256"] == expected_sha
    assert meta["record_count"] == lineage["record_count"]

    kawal_meta_file = out_dir / "kawal_dapt_metadata.json"
    assert kawal_meta_file.is_file()
    with kawal_meta_file.open("r", encoding="utf-8") as f:
        kawal_meta = json.load(f)
    assert kawal_meta["corpus_sha256"] == expected_sha

    dataset_manifest_file = out_dir / "dataset_manifest.json"
    assert dataset_manifest_file.is_file()
    with dataset_manifest_file.open("r", encoding="utf-8") as f:
        ds_manifest = json.load(f)
    assert ds_manifest["corpus_sha256"] == expected_sha
    assert ds_manifest["record_count"] == lineage["record_count"]

    config_file = out_dir / "config.json"
    assert config_file.is_file()
    with config_file.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    assert "dataset_lineage" in cfg
    assert cfg["dataset_lineage"]["corpus_sha256"] == expected_sha
    assert cfg["task_specific_params"]["dapt"]["dataset_lineage"]["corpus_sha256"] == expected_sha


def test_dapt_huggingface_execution_outputs_dataset_lineage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    corpus_file = tmp_path / "hf_corpus.txt"
    corpus_file.write_text("Trotoar rusak di depan halte busway menyebabkan pejalan kaki terganggu.\n", encoding="utf-8")
    expected_sha = compute_file_sha256(corpus_file)
    out_dir = tmp_path / "hf_lineage_out"

    mock_torch = mock.MagicMock()
    mock_torch.cuda.is_available.return_value = False
    mock_torch.utils.data.Dataset = object
    mock_torch.tensor = lambda x, **kw: x
    mock_torch.long = "long"

    mock_tokenizer_instance = mock.MagicMock()
    mock_tokenizer_instance.return_value = {
        "input_ids": [[101, 1000, 102]],
        "attention_mask": [[1, 1, 1]],
        "special_tokens_mask": [[1, 0, 1]],
    }

    def mock_save_pretrained(save_directory: str) -> None:
        p = Path(save_directory)
        p.mkdir(parents=True, exist_ok=True)
        (p / "tokenizer_config.json").write_text("{}", encoding="utf-8")
        (p / "vocab.txt").write_text("[PAD]\n", encoding="utf-8")

    mock_tokenizer_instance.save_pretrained.side_effect = mock_save_pretrained
    mock_model_instance = mock.MagicMock()
    mock_trainer_instance = mock.MagicMock()

    def mock_save_model(output_dir: str) -> None:
        p = Path(output_dir)
        p.mkdir(parents=True, exist_ok=True)
        (p / "model.safetensors").write_bytes(b"TRAINED_SAFETENSORS_LINEAGE" + b"\x00" * 32)
        (p / "config.json").write_text("{}", encoding="utf-8")

    mock_trainer_instance.save_model.side_effect = mock_save_model
    mock_transformers = mock.MagicMock()
    mock_transformers.AutoTokenizer.from_pretrained.return_value = mock_tokenizer_instance
    mock_transformers.AutoModelForMaskedLM.from_pretrained.return_value = mock_model_instance
    mock_transformers.Trainer.return_value = mock_trainer_instance

    monkeypatch.setitem(sys.modules, "torch", mock_torch)
    monkeypatch.setitem(sys.modules, "torch.utils", mock_torch.utils)
    monkeypatch.setitem(sys.modules, "torch.utils.data", mock_torch.utils.data)
    monkeypatch.setitem(sys.modules, "transformers", mock_transformers)

    config = DAPTConfig(
        corpus_path=str(corpus_file),
        output_dir=str(out_dir),
        seed=42,
        max_seq_length=448,
        learning_rate=2e-5,
        batch_size=8,
        gradient_accumulation_steps=4,
        num_epochs=1,
        fp16=False,
        local_files_only=True,
        version="v1.0.0",
    )

    manifest = run_train_dapt(config, dry_run=False, validate_only=False)
    assert manifest is not None
    assert "indobert-dapt" in manifest.artifacts
    assert (out_dir / "manifest.json").is_file()

    dapt_meta = out_dir / "dapt_metadata.json"
    assert dapt_meta.is_file()
    with dapt_meta.open("r", encoding="utf-8") as f:
        meta_data = json.load(f)
    assert meta_data["dataset_lineage"]["corpus_sha256"] == expected_sha
    assert meta_data["dataset_lineage"]["record_count"] == 1

    ds_manifest = out_dir / "dataset_manifest.json"
    assert ds_manifest.is_file()
    with ds_manifest.open("r", encoding="utf-8") as f:
        ds_data = json.load(f)
    assert ds_data["corpus_sha256"] == expected_sha


def test_dapt_cli_outputs_auditable_dataset_lineage(tmp_path: Path) -> None:
    corpus_file = tmp_path / "cli_corpus.txt"
    corpus_file.write_text("Genangan air di jalan fatmawati akibat gorong-gorong tersumbat sampah plastik.\n", encoding="utf-8")
    expected_sha = compute_file_sha256(corpus_file)
    out_dir = tmp_path / "cli_lineage_out"

    res = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.train_dapt",
            "--corpus-path",
            str(corpus_file),
            "--output-dir",
            str(out_dir),
            "--dry-run",
        ],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0, res.stderr
    assert (out_dir / "manifest.json").is_file()
    assert (out_dir / "dapt_metadata.json").is_file()
    assert (out_dir / "dataset_manifest.json").is_file()

    with open(out_dir / "dapt_metadata.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["corpus_sha256"] == expected_sha
    assert meta["dataset_lineage"]["corpus_basename"] == "cli_corpus.txt"
    assert meta["record_count"] == 1


def test_dapt_lineage_backward_compatibility_with_synthetic_default(tmp_path: Path) -> None:
    out_dir = tmp_path / "compat_dapt_out"
    config = DAPTConfig(
        output_dir=str(out_dir),
        seed=42,
        version="v1.0.0",
    )
    manifest = run_train_dapt(config, dry_run=True, validate_only=False)
    assert manifest is not None
    assert manifest.get_checkpoint_version() == "v1.0.0"

    manifest_file = out_dir / "manifest.json"
    assert manifest_file.is_file()

    loaded = ArtifactManifest.from_file(manifest_file)
    assert loaded.environment == "local-cpu"
    assert "indobert-dapt" in loaded.artifacts
    assert loaded.artifacts["indobert-dapt"].precision == "fp16"

    val_results = loaded.validate_all(out_dir, raise_on_error=True)
    assert val_results["indobert-dapt"].is_valid is True

    dapt_meta_file = out_dir / "dapt_metadata.json"
    assert dapt_meta_file.is_file()
    with dapt_meta_file.open("r", encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["dataset_lineage"]["is_synthetic"] is True
    assert meta["dataset_lineage"]["record_count"] > 0


def test_multitask_sample_extraction_one_turn_dedup() -> None:
    """Ensure 1-turn/1-bubble trajectory emits only one deterministic sample instead of duplicate full and turn samples."""
    from contracts.models import (
        Category,
        ComplaintTrajectory,
        DatasetSplit,
        DecisionMode,
        TrajectoryBubble,
        TrajectoryTurn,
        TurnExpectedAction,
    )

    bubble = TrajectoryBubble(source_message_id="msg_001", text="Lampu jalan padam total di Sudirman.")
    turn = TrajectoryTurn(
        turn=1,
        bubbles=(bubble,),
        expected_action=TurnExpectedAction(turn=1, allowed_actions=(DecisionMode.EXECUTE,)),
    )
    traj = ComplaintTrajectory(
        scenario_id="sc-dedup-1t-001",
        family_id="fam-dedup-1t",
        split=DatasetSplit.TRAIN,
        category=Category.ROAD,
        world_truth={
            "category": "ROAD",
            "intent": "COMPLAINT",
            "risk": "HIGH",
            "completeness": "SUFFICIENT",
            "provenance": "synthetic_proxy",
        },
        turns=(turn,),
    )

    samples = extract_trajectory_multitask_samples([traj])
    assert len(samples) == 1
    sample = samples[0]
    assert sample["text"] == "Lampu jalan padam total di Sudirman."
    assert sample["granularity"] == "full"
    assert sample["intent"] == "COMPLAINT"
    assert sample["category"] == "ROAD"
    assert sample["risk"] == "HIGH"
    assert sample["completeness"] == "SUFFICIENT"
    assert sample["scenario_id"] == "sc-dedup-1t-001"


def test_multitask_sample_extraction_multi_turn_preservation() -> None:
    """Ensure multi-turn trajectories preserve full sample and all distinct partial-context turn samples."""
    from contracts.models import (
        Category,
        ComplaintTrajectory,
        DatasetSplit,
        DecisionMode,
        TrajectoryBubble,
        TrajectoryTurn,
        TurnExpectedAction,
    )

    t1_bubble = TrajectoryBubble(source_message_id="msg_101", text="Pipa PDAM pecah di trotoar.")
    t2_bubble = TrajectoryBubble(source_message_id="msg_102", text="Lokasi persis depan kantor pos.")

    turn1 = TrajectoryTurn(
        turn=1,
        bubbles=(t1_bubble,),
        expected_action=TurnExpectedAction(turn=1, allowed_actions=(DecisionMode.REQUEST_CLARIFICATION,)),
    )
    turn2 = TrajectoryTurn(
        turn=2,
        bubbles=(t2_bubble,),
        expected_action=TurnExpectedAction(turn=2, allowed_actions=(DecisionMode.EXECUTE,)),
    )
    traj = ComplaintTrajectory(
        scenario_id="sc-preserve-mt-001",
        family_id="fam-preserve-mt",
        split=DatasetSplit.DEV,
        category=Category.CLEAN_WATER,
        world_truth={
            "category": "CLEAN_WATER",
            "intent": "COMPLAINT",
            "risk": "HIGH",
            "completeness": "SUFFICIENT",
            "provenance": "synthetic_proxy",
        },
        turns=(turn1, turn2),
    )

    samples = extract_trajectory_multitask_samples([traj])
    assert len(samples) == 3

    granularities = [s["granularity"] for s in samples]
    assert granularities == ["full", "turn_1", "turn_2"]

    full_s = samples[0]
    turn1_s = samples[1]
    turn2_s = samples[2]

    assert full_s["text"] == "Pipa PDAM pecah di trotoar.\nLokasi persis depan kantor pos."
    assert turn1_s["text"] == "Pipa PDAM pecah di trotoar."
    assert turn2_s["text"] == "Pipa PDAM pecah di trotoar. Lokasi persis depan kantor pos."

    for s in samples:
        assert s["category"] == "CLEAN_WATER"
        assert s["intent"] == "COMPLAINT"
        assert s["risk"] == "HIGH"
        assert s["completeness"] == "SUFFICIENT"
        assert s["split"] == "dev"


def test_multitask_sample_extraction_does_not_collapse_separate_trajectories() -> None:
    """Ensure identical user bubbles/text across separate trajectories are preserved and NOT collapsed."""
    from contracts.models import (
        Category,
        ComplaintTrajectory,
        DatasetSplit,
        DecisionMode,
        TrajectoryBubble,
        TrajectoryTurn,
        TurnExpectedAction,
    )

    def _make_traj(scenario_id: str) -> ComplaintTrajectory:
        bubble = TrajectoryBubble(source_message_id=f"msg_{scenario_id}", text="Tumpukan sampah liar menumpuk.")
        turn = TrajectoryTurn(
            turn=1,
            bubbles=(bubble,),
            expected_action=TurnExpectedAction(turn=1, allowed_actions=(DecisionMode.EXECUTE,)),
        )
        return ComplaintTrajectory(
            scenario_id=scenario_id,
            family_id="fam-waste-repeat",
            split=DatasetSplit.TRAIN,
            category=Category.WASTE,
            world_truth={
                "category": "WASTE",
                "intent": "COMPLAINT",
                "risk": "MEDIUM",
                "completeness": "SUFFICIENT",
                "provenance": "synthetic_proxy",
            },
            turns=(turn,),
        )

    traj_a = _make_traj("sc-sep-001")
    traj_b = _make_traj("sc-sep-002")

    samples = extract_trajectory_multitask_samples([traj_a, traj_b])
    assert len(samples) == 2
    scenario_ids = [s["scenario_id"] for s in samples]
    assert scenario_ids == ["sc-sep-001", "sc-sep-002"]
    assert samples[0]["text"] == samples[1]["text"] == "Tumpukan sampah liar menumpuk."


def test_deduplicate_trajectory_multitask_samples_helper_direct() -> None:
    """Directly test deduplicate_trajectory_multitask_samples helper preserves first-seen and distinct samples."""
    raw_samples = [
        {"text": "Halo", "intent": "A", "category": "B", "risk": "C", "completeness": "D", "granularity": "full"},
        {"text": "Halo", "intent": "A", "category": "B", "risk": "C", "completeness": "D", "granularity": "turn_1"},
        {"text": "Halo", "intent": "DIFF", "category": "B", "risk": "C", "completeness": "D", "granularity": "turn_2"},
        {"text": "Halo lagi", "intent": "A", "category": "B", "risk": "C", "completeness": "D", "granularity": "turn_3"},
    ]
    deduped = deduplicate_trajectory_multitask_samples(raw_samples)
    assert len(deduped) == 3
    assert deduped[0]["granularity"] == "full"
    assert deduped[1]["intent"] == "DIFF"
    assert deduped[2]["text"] == "Halo lagi"


def test_ner_sample_extraction_one_turn_dedup() -> None:
    """Ensure 1-turn/1-bubble trajectory emits only one deterministic sample instead of duplicate full and bubble samples."""
    from contracts.models import (
        Category,
        ComplaintTrajectory,
        DatasetSplit,
        DecisionMode,
        TrajectoryBubble,
        TrajectoryTurn,
        TurnExpectedAction,
    )

    bubble = TrajectoryBubble(source_message_id="msg_ner_001", text="Lampu jalan padam total di Jalan Sudirman.")
    turn = TrajectoryTurn(
        turn=1,
        bubbles=(bubble,),
        expected_action=TurnExpectedAction(turn=1, allowed_actions=(DecisionMode.EXECUTE,)),
    )
    traj = ComplaintTrajectory(
        scenario_id="sc-ner-dedup-1t-001",
        family_id="fam-ner-dedup-1t",
        split=DatasetSplit.TRAIN,
        category=Category.ROAD,
        turns=(turn,),
    )

    samples = extract_trajectory_ner_samples([traj])
    assert len(samples) == 1
    sample = samples[0]
    assert sample["text"] == "Lampu jalan padam total di Jalan Sudirman."
    assert sample["granularity"] == "bubble"
    assert sample["scenario_id"] == "sc-ner-dedup-1t-001"
    assert sample["split"] == "train"


def test_ner_sample_extraction_one_turn_dedup_with_canonical_spans() -> None:
    """Ensure 1-turn/1-bubble trajectory with explicit canonical spans emits only one sample without heuristic contamination."""
    raw_record = {
        "scenario_id": "sc-ner-canon-1t-001",
        "family_id": "fam-ner-canon-1t",
        "split": "train",
        "category": "ROAD",
        "turns": [
            {
                "turn": 1,
                "bubbles": [
                    {
                        "source_message_id": "msg_c_01",
                        "text": "Pipa air bocor parah di Jalan Malioboro tadi pagi.",
                        "canonical_spans": [(0, 14, "OBJ")],
                    }
                ],
            }
        ],
    }

    samples = extract_trajectory_ner_samples([raw_record])
    assert len(samples) == 1
    sample = samples[0]
    assert sample["text"] == "Pipa air bocor parah di Jalan Malioboro tadi pagi."
    assert sample["granularity"] == "bubble"
    assert sample["spans"] == [(0, 14, "OBJ")]
    assert sample["canonical_spans"] == [(0, 14, "OBJ")]


def test_ner_sample_extraction_preserves_multiturn_bubbles_and_full() -> None:
    """Ensure multi-turn/multi-bubble trajectories preserve both individual bubble samples and full combined sample."""
    from contracts.models import (
        Category,
        ComplaintTrajectory,
        DatasetSplit,
        DecisionMode,
        TrajectoryBubble,
        TrajectoryTurn,
        TurnExpectedAction,
    )

    t1_bubble = TrajectoryBubble(source_message_id="msg_m_1", text="Pipa PDAM pecah di trotoar.")
    t2_bubble = TrajectoryBubble(source_message_id="msg_m_2", text="Lokasi persis depan kantor pos.")

    turn1 = TrajectoryTurn(
        turn=1,
        bubbles=(t1_bubble,),
        expected_action=TurnExpectedAction(turn=1, allowed_actions=(DecisionMode.REQUEST_CLARIFICATION,)),
    )
    turn2 = TrajectoryTurn(
        turn=2,
        bubbles=(t2_bubble,),
        expected_action=TurnExpectedAction(turn=2, allowed_actions=(DecisionMode.EXECUTE,)),
    )
    traj = ComplaintTrajectory(
        scenario_id="sc-ner-preserve-mt-001",
        family_id="fam-ner-preserve-mt",
        split=DatasetSplit.DEV,
        category=Category.CLEAN_WATER,
        turns=(turn1, turn2),
    )

    samples = extract_trajectory_ner_samples([traj])
    assert len(samples) == 3

    granularities = [s["granularity"] for s in samples]
    assert granularities == ["bubble", "bubble", "full"]

    bubble1_s = samples[0]
    bubble2_s = samples[1]
    full_s = samples[2]

    assert bubble1_s["text"] == "Pipa PDAM pecah di trotoar."
    assert bubble2_s["text"] == "Lokasi persis depan kantor pos."
    assert full_s["text"] == "Pipa PDAM pecah di trotoar.\nLokasi persis depan kantor pos."


def test_ner_sample_extraction_does_not_collapse_separate_trajectories() -> None:
    """Ensure identical text/spans across separate trajectories are preserved and NOT collapsed across trajectories."""
    from contracts.models import (
        Category,
        ComplaintTrajectory,
        DatasetSplit,
        DecisionMode,
        TrajectoryBubble,
        TrajectoryTurn,
        TurnExpectedAction,
    )

    def _make_traj(scenario_id: str) -> ComplaintTrajectory:
        bubble = TrajectoryBubble(source_message_id=f"msg_{scenario_id}", text="Tumpukan sampah liar menumpuk di jalan.")
        turn = TrajectoryTurn(
            turn=1,
            bubbles=(bubble,),
            expected_action=TurnExpectedAction(turn=1, allowed_actions=(DecisionMode.EXECUTE,)),
        )
        return ComplaintTrajectory(
            scenario_id=scenario_id,
            family_id="fam-waste-repeat",
            split=DatasetSplit.TRAIN,
            category=Category.WASTE,
            turns=(turn,),
        )

    traj_a = _make_traj("sc-ner-sep-001")
    traj_b = _make_traj("sc-ner-sep-002")

    samples = extract_trajectory_ner_samples([traj_a, traj_b])
    assert len(samples) == 2
    scenario_ids = [s["scenario_id"] for s in samples]
    assert scenario_ids == ["sc-ner-sep-001", "sc-ner-sep-002"]
    assert samples[0]["text"] == samples[1]["text"] == "Tumpukan sampah liar menumpuk di jalan."


def test_ner_sample_extraction_heuristic_fallback_no_contamination_when_bubble_lacks_canonical() -> None:
    """Ensure that when bubble lacks canonical spans, its heuristic spans do not contaminate traj_canonical."""
    raw_record = {
        "scenario_id": "sc-ner-contam-check-001",
        "family_id": "fam-ner-contam",
        "split": "train",
        "category": "ROAD",
        "world_truth": {
            "canonical_spans": [(0, 11, "OBJ")],
        },
        "turns": [
            {
                "turn": 1,
                "bubbles": [
                    {
                        "source_message_id": "msg_contam_1",
                        "text": "Lampu padam di Jalan Sudirman tadi malam.",
                    }
                ],
            }
        ],
    }
    samples = extract_trajectory_ner_samples([raw_record])
    assert len(samples) == 1
    sample = samples[0]
    # Because total_bubbles_count == 1, b_canonical falls back to traj canonical_spans [(0, 11, "OBJ")].
    # It must NOT include heuristic LOC ("Jalan Sudirman") or TIME ("tadi malam").
    assert sample["spans"] == [(0, 11, "OBJ")]


def test_ner_sample_extraction_ood_regression_protection() -> None:
    """Ensure OOD canary scenario in train/dev split still raises DatasetAuditError."""
    ood_record = {
        "scenario_id": "ood_canary_ner_001",
        "family_id": "fam-canary-ner",
        "split": "train",
        "turns": [
            {
                "turn": 1,
                "bubbles": [{"text": "Pohon tumbang menutup jalan."}],
            }
        ],
    }
    with pytest.raises(DatasetAuditError) as exc_info:
        extract_trajectory_ner_samples([ood_record])
    assert "OOD training contamination detected" in str(exc_info.value)


def test_deduplicate_trajectory_ner_samples_helper_direct() -> None:
    """Directly test deduplicate_trajectory_ner_samples helper preserves first-seen and distinct samples."""
    raw_samples = [
        {"text": "Halo Bandung", "canonical_spans": [(5, 12, "LOC")], "spans": [(5, 12, "LOC")], "granularity": "bubble"},
        {"text": "Halo Bandung", "canonical_spans": [(5, 12, "LOC")], "spans": [(5, 12, "LOC")], "granularity": "full"},
        {"text": "Halo Bandung", "canonical_spans": [(0, 4, "OBJ")], "spans": [(0, 4, "OBJ")], "granularity": "bubble_2"},
        {"text": "Lapor jalan rusak", "canonical_spans": [(6, 17, "OBJ")], "spans": [(6, 17, "OBJ")], "granularity": "bubble_3"},
    ]
    deduped = deduplicate_trajectory_ner_samples(raw_samples)
    assert len(deduped) == 3
    assert deduped[0]["granularity"] == "bubble"
    assert deduped[1]["canonical_spans"] == [(0, 4, "OBJ")]
    assert deduped[2]["text"] == "Lapor jalan rusak"


