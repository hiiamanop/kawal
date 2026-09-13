from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import tempfile

import pytest
from contracts.models import (
    AgentResultV1,
    AnalysisResult,
    Category,
    EvidenceItem,
    EvidenceKind,
    ResultStatus,
    RiskLevel,
    Sensitivity,
    TelemetryInfo,
    UncertaintyInfo,
    UncertaintyMethod,
    VersionInfo,
)
from pydantic import ValidationError
from services.intelligence.chunking import Chunk
from services.intelligence.ner import EntitySpan
from services.ml import (
    ArtifactIntegrityError,
    ArtifactManifest,
    ArtifactManifestItem,
    ArtifactNotFoundError,
    ClassificationPrediction,
    CpuFp32BenchmarkConfig,
    CpuFp32BenchmarkResult,
    LocalMLRuntime,
    ManifestValidationError,
    MLRuntimeResult,
    RuntimeStatus,
    SpanNER,
    chunk_text,
    compute_file_sha256,
    run_cpu_fp32_benchmark,
    runtime_prediction_to_agent_result,
    to_agent_result_v1,
)


def _create_dummy_file(directory: Path, filename: str, content: bytes) -> tuple[Path, str]:
    file_path = directory / filename
    file_path.write_bytes(content)
    expected_hash = sha256(content).hexdigest()
    return file_path, expected_hash


# --- 1. Manifest Validation Tests ---

def test_manifest_item_valid() -> None:
    item = ArtifactManifestItem(
        name="indobert-classifier",
        version="v1.0.0",
        sha256="a" * 64,
        path="models/classifier.onnx",
        size_bytes=1024,
        task="classification",
        precision="fp32",
    )
    assert item.name == "indobert-classifier"
    assert item.sha256 == "a" * 64
    assert item.precision == "fp32"


def test_manifest_item_rejects_invalid_hash() -> None:
    with pytest.raises(ValidationError):
        ArtifactManifestItem(
            name="indobert-classifier",
            version="v1.0.0",
            sha256="invalid_short_hash",
            path="models/classifier.onnx",
            size_bytes=1024,
        )


def test_manifest_item_rejects_path_traversal() -> None:
    with pytest.raises(ValidationError):
        ArtifactManifestItem(
            name="indobert-classifier",
            version="v1.0.0",
            sha256="b" * 64,
            path="../models/classifier.onnx",
            size_bytes=1024,
        )

    with pytest.raises(ValidationError):
        ArtifactManifestItem(
            name="indobert-classifier",
            version="v1.0.0",
            sha256="b" * 64,
            path="/etc/passwd",
            size_bytes=1024,
        )


def test_manifest_file_validation_success() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        base_dir = Path(tmpdir)
        content = b"indobert model weight mock content for cpu fp32"
        _, file_hash = _create_dummy_file(base_dir, "model.bin", content)

        manifest = ArtifactManifest(
            manifest_version="1.0.0",
            environment="local-cpu",
            artifacts={
                "indobert": ArtifactManifestItem(
                    name="indobert",
                    version="1.0.0",
                    sha256=file_hash,
                    path="model.bin",
                    size_bytes=len(content),
                    precision="fp32",
                )
            },
        )

        res = manifest.validate_artifact_file("indobert", base_dir)
        assert res.is_valid is True
        assert res.file_exists is True
        assert res.actual_sha256 == file_hash
        assert res.error_message is None


def test_manifest_file_validation_missing_file() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        base_dir = Path(tmpdir)
        manifest = ArtifactManifest(
            manifest_version="1.0.0",
            artifacts={
                "missing": ArtifactManifestItem(
                    name="missing",
                    version="1.0.0",
                    sha256="c" * 64,
                    path="does_not_exist.bin",
                    size_bytes=100,
                )
            },
        )

        res = manifest.validate_artifact_file("missing", base_dir, raise_on_error=False)
        assert res.is_valid is False
        assert res.file_exists is False
        assert "not found" in (res.error_message or "")

        with pytest.raises(ArtifactNotFoundError):
            manifest.validate_artifact_file("missing", base_dir, raise_on_error=True)


def test_manifest_file_validation_hash_mismatch() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        base_dir = Path(tmpdir)
        content = b"corrupted weight data"
        _create_dummy_file(base_dir, "corrupt.bin", content)

        manifest = ArtifactManifest(
            manifest_version="1.0.0",
            artifacts={
                "corrupt": ArtifactManifestItem(
                    name="corrupt",
                    version="1.0.0",
                    sha256="d" * 64,
                    path="corrupt.bin",
                    size_bytes=len(content),
                )
            },
        )

        res = manifest.validate_artifact_file("corrupt", base_dir, raise_on_error=False)
        assert res.is_valid is False
        assert res.file_exists is True
        assert res.actual_sha256 != "d" * 64
        assert "mismatch" in (res.error_message or "").lower()

        with pytest.raises(ArtifactIntegrityError):
            manifest.validate_artifact_file("corrupt", base_dir, raise_on_error=True)


def test_manifest_serialization_roundtrip() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        json_path = Path(tmpdir) / "manifest.json"
        data = {
            "manifest_version": "1.2.3",
            "environment": "local-cpu-fp32",
            "artifacts": {
                "multitask": {
                    "name": "multitask",
                    "version": "1.2.3",
                    "sha256": "e" * 64,
                    "path": "weights/multitask.onnx",
                    "size_bytes": 4096,
                    "task": "classification",
                    "precision": "fp32",
                }
            },
        }
        json_path.write_text(json.dumps(data), encoding="utf-8")

        loaded = ArtifactManifest.from_file(json_path)
        assert loaded.manifest_version == "1.2.3"
        assert loaded.environment == "local-cpu-fp32"
        assert "multitask" in loaded.artifacts
        assert loaded.to_dict()["manifest_version"] == "1.2.3"


# --- 2. CPU FP32 Benchmark Tests ---

def test_cpu_fp32_benchmark_config_defaults() -> None:
    config = CpuFp32BenchmarkConfig()
    assert config.batch_size == 1
    assert config.sequence_length == 448
    assert config.overlap_tokens == 64
    assert config.precision == "fp32"
    assert config.device == "cpu"
    assert config.target_p95_latency_ms == 100.0


def test_cpu_fp32_benchmark_execution() -> None:
    config = CpuFp32BenchmarkConfig(
        warmup_runs=2,
        benchmark_runs=5,
        target_p95_latency_ms=500.0,
    )
    result = run_cpu_fp32_benchmark(config)

    assert isinstance(result, CpuFp32BenchmarkResult)
    assert result.samples_count == 5
    assert result.p50_latency_ms <= result.p95_latency_ms
    assert result.p95_latency_ms <= result.p99_latency_ms
    assert result.min_latency_ms <= result.max_latency_ms
    assert result.throughput_qps > 0.0
    assert result.sla_met is True


def test_cpu_fp32_benchmark_sla_violation_flag() -> None:
    config = CpuFp32BenchmarkConfig(
        warmup_runs=0,
        benchmark_runs=3,
        target_p95_latency_ms=0.00001,  # Impossibly tight SLA
    )
    result = run_cpu_fp32_benchmark(config)
    assert result.sla_met is False


# --- 3. Chunking & Overlap Tests ---

def test_chunk_text_short() -> None:
    short_text = "Laporan jalan berlubang di Jalan Sudirman"
    chunks = chunk_text(short_text, max_chunk_tokens=448, overlap_tokens=64)
    assert len(chunks) == 1
    assert chunks[0] == short_text
    assert isinstance(chunks[0], Chunk)
    assert chunks[0].text == short_text
    assert chunks[0].token_count == 6


def test_chunk_text_empty() -> None:
    chunks = chunk_text("")
    assert len(chunks) == 1
    assert isinstance(chunks[0], Chunk)
    assert chunks[0].text == ""
    assert chunks[0].token_count == 0

    ws_chunks = chunk_text("   \n\t ")
    assert len(ws_chunks) == 1
    assert ws_chunks[0].text == ""
    assert ws_chunks[0].token_count == 0

    assert chunk_text("", return_strings=True) == []
    assert chunk_text("   \n\t ", return_strings=True) == []


def test_chunk_text_long_with_overlap() -> None:
    words = [f"word{i}" for i in range(1000)]
    long_text = " ".join(words)
    chunks = chunk_text(long_text, max_chunk_tokens=448, overlap_tokens=64)

    assert len(chunks) >= 3
    first_chunk_words = chunks[0].split()
    second_chunk_words = chunks[1].split()

    assert len(first_chunk_words) == 448
    assert first_chunk_words[384:] == second_chunk_words[:64]
    assert chunks[1].start_token == chunks[0].end_token - 64


def test_chunk_text_reconciled_with_intelligence_chunking() -> None:
    text = "Aduan jalan rusak parah di pertigaan lampu merah."
    chunks = chunk_text(text, max_tokens=10, overlap=2)
    assert all(isinstance(c, Chunk) for c in chunks)
    assert chunks[0].chunk_index == 0
    assert chunks[0].total_chunks == len(chunks)
    assert chunks[0].token_count <= 10
    assert chunks[0].start_char == 0
    assert text.startswith(chunks[0].text)


# --- 4. Unavailable-Safe Local ML Runtime Tests ---

def test_local_runtime_rejects_network() -> None:
    with pytest.raises(ValueError, match="Network access is strictly forbidden"):
        LocalMLRuntime(allow_network=True)


def test_local_runtime_unavailable_safe_fallback() -> None:
    runtime = LocalMLRuntime(manifest=None, artifacts_dir=None)

    assert runtime.is_available() is False

    result = runtime.predict(
        "Jalan berlubang besar di depan pasar",
        context={"jurisdiction_id": "JUR-TEST-01", "authority_unit_id": "UNIT-BINA-MARGA-01"},
    )

    assert isinstance(result, MLRuntimeResult)
    assert result.status == RuntimeStatus.UNAVAILABLE
    assert result.available is False
    assert result.prediction is None
    assert result.entities == ()
    assert result.reason is not None

    fallback = result.analysis_fallback
    assert isinstance(fallback, AnalysisResult)
    assert fallback.category == Category.ROAD
    assert fallback.risk == RiskLevel.MEDIUM
    assert fallback.sensitivity == Sensitivity.NORMAL
    assert fallback.jurisdiction_id == "JUR-TEST-01"
    assert fallback.authority_unit_id == "UNIT-BINA-MARGA-01"
    assert "ml_model_unavailable" in fallback.missing_fields
    assert fallback.has_mandatory_evidence is False


def test_local_runtime_unavailable_when_artifacts_missing() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        manifest = ArtifactManifest(
            manifest_version="1.0.0",
            artifacts={
                "indobert": ArtifactManifestItem(
                    name="indobert",
                    version="1.0.0",
                    sha256="0" * 64,
                    path="non_existent_file.bin",
                    size_bytes=100,
                )
            },
        )
        runtime = LocalMLRuntime(manifest=manifest, artifacts_dir=tmpdir)
        assert runtime.is_available() is False

        res = runtime.predict("Aduan jalan rusak")
        assert res.status == RuntimeStatus.UNAVAILABLE
        assert res.available is False


def test_local_runtime_unavailable_when_valid_artifacts_exist_but_no_callable_backend() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        base_dir = Path(tmpdir)
        content = b"valid onnx weight mock content"
        _, file_hash = _create_dummy_file(base_dir, "classifier.onnx", content)

        manifest = ArtifactManifest(
            manifest_version="1.0.0",
            environment="local-cpu",
            artifacts={
                "classifier": ArtifactManifestItem(
                    name="classifier",
                    version="1.0.0",
                    sha256=file_hash,
                    path="classifier.onnx",
                    size_bytes=len(content),
                    precision="fp32",
                )
            },
        )

        runtime = LocalMLRuntime(manifest=manifest, artifacts_dir=str(base_dir), inference_backend=None)
        assert runtime.is_available() is False

        res = runtime.predict("Laporan jalan rusak parah")
        assert res.status == RuntimeStatus.UNAVAILABLE
        assert res.available is False
        assert res.prediction is None
        assert res.entities == ()
        assert res.reason is not None
        assert "no callable inference backend" in res.reason.lower()
        assert isinstance(res.analysis_fallback, AnalysisResult)
        assert "ml_model_unavailable" in res.analysis_fallback.missing_fields


def test_local_runtime_unavailable_when_backend_not_callable() -> None:
    runtime = LocalMLRuntime(inference_backend="not_a_callable")
    assert runtime.is_available() is False

    res = runtime.predict("Aduan warga")
    assert res.status == RuntimeStatus.UNAVAILABLE
    assert res.available is False
    assert res.prediction is None
    assert res.entities == ()


def test_local_runtime_unavailable_safe_on_backend_exception() -> None:
    def failing_backend(text: str, context: dict | None) -> tuple[ClassificationPrediction, list[SpanNER]]:
        raise RuntimeError("Inference engine memory fault")

    runtime = LocalMLRuntime(inference_backend=failing_backend)
    assert runtime.is_available() is True

    result = runtime.predict("Ada lubang di jalan raya")
    assert result.status == RuntimeStatus.UNAVAILABLE
    assert result.available is False
    assert result.prediction is None
    assert result.entities == ()
    assert result.reason is not None
    assert "Inference engine memory fault" in result.reason
    assert isinstance(result.analysis_fallback, AnalysisResult)
    assert "ml_model_unavailable" in result.analysis_fallback.missing_fields


def test_local_runtime_backend_with_predict_method() -> None:
    class ObjectBackend:
        def predict(self, text: str, context: dict | None) -> tuple[ClassificationPrediction, list[SpanNER]]:
            pred = ClassificationPrediction(
                intent="COMPLAINT",
                category="ROAD",
                risk="LOW",
                completeness="COMPLETE",
                confidence=0.91,
                probabilities={"ROAD": 0.91},
            )
            return pred, []

    runtime = LocalMLRuntime(inference_backend=ObjectBackend())
    assert runtime.is_available() is True

    result = runtime.predict("Jalan sedikit retak")
    assert result.status == RuntimeStatus.AVAILABLE
    assert result.available is True
    assert result.prediction is not None
    assert result.prediction.category == "ROAD"


def test_local_runtime_available_with_mock_backend() -> None:
    def mock_backend(text: str, context: dict | None) -> tuple[ClassificationPrediction, list[SpanNER]]:
        pred = ClassificationPrediction(
            intent="COMPLAINT",
            category="ROAD",
            risk="MEDIUM",
            completeness="COMPLETE",
            confidence=0.94,
            probabilities={"ROAD": 0.94, "DRAINAGE": 0.06},
        )
        spans = [
            SpanNER(text="Jalan Sudirman", label="B-LOC", start=10, end=24, confidence=0.98),
            SpanNER(text="lubang jalan", label="B-OBJ", start=28, end=40, confidence=0.95),
        ]
        return pred, spans

    runtime = LocalMLRuntime(inference_backend=mock_backend)
    assert runtime.is_available() is True

    result = runtime.predict("Ada lubang di Jalan Sudirman")
    assert result.status == RuntimeStatus.AVAILABLE
    assert result.available is True
    assert result.prediction is not None
    assert result.prediction.category == "ROAD"
    assert result.prediction.confidence == 0.94
    assert result.prediction.completeness == "INCOMPLETE"
    assert len(result.entities) == 2
    assert result.entities[0].label == "B-LOC"
    assert result.entities[0].text == "Jalan Sudirman"
    assert result.latency_ms >= 0.0
    assert result.reason is None


def test_span_ner_entity_span_adaptation() -> None:
    span = SpanNER(text="Jalan Malioboro", label="LOCATION", start=5, end=20, confidence=0.95)
    assert isinstance(span, EntitySpan)
    assert isinstance(span, SpanNER)
    assert span.start == 5
    assert span.end == 20
    assert span.start_char == 5
    assert span.end_char == 20
    assert span.text == "Jalan Malioboro"
    assert span.label == "LOCATION"

    pure_es = span.to_entity_span()
    assert isinstance(pure_es, EntitySpan)
    assert not isinstance(pure_es, SpanNER)
    assert pure_es.start_char == 5
    assert pure_es.end_char == 20
    assert pure_es.label == "LOCATION"

    source_es = EntitySpan(
        label="AGENCY",
        text="Dinas PU",
        start_token=3,
        end_token=5,
        start_char=25,
        end_char=33,
        confidence=0.88,
        chunk_id="chk-1",
        source_message_id="msg-123",
    )
    adapted_span = SpanNER.from_entity_span(source_es)
    assert isinstance(adapted_span, SpanNER)
    assert isinstance(adapted_span, EntitySpan)
    assert adapted_span.start == 25
    assert adapted_span.end == 33
    assert adapted_span.start_char == 25
    assert adapted_span.end_char == 33
    assert adapted_span.start_token == 3
    assert adapted_span.end_token == 5
    assert adapted_span.chunk_id == "chk-1"
    assert adapted_span.source_message_id == "msg-123"


def test_local_runtime_adapts_backend_entity_spans() -> None:
    def backend_with_entity_spans(text: str, context: dict | None) -> tuple[ClassificationPrediction, list[EntitySpan]]:
        pred = ClassificationPrediction(
            intent="COMPLAINT",
            category="ROAD",
            risk="HIGH",
            completeness="COMPLETE",
            confidence=0.97,
            probabilities={"ROAD": 0.97},
        )
        spans = [
            EntitySpan(
                label="LOCATION",
                text="Jalan Godean",
                start_token=0,
                end_token=2,
                start_char=0,
                end_char=12,
                confidence=0.99,
            ),
        ]
        return pred, spans

    runtime = LocalMLRuntime(inference_backend=backend_with_entity_spans)
    assert runtime.is_available() is True

    result = runtime.predict("Jalan Godean amblas")
    assert result.status == RuntimeStatus.AVAILABLE
    assert result.available is True
    assert len(result.entities) == 1
    entity = result.entities[0]
    assert isinstance(entity, SpanNER)
    assert isinstance(entity, EntitySpan)
    assert entity.label == "LOCATION"
    assert entity.text == "Jalan Godean"
    assert entity.start == 0
    assert entity.start_char == 0
    assert entity.end == 12
    assert entity.end_char == 12
    assert entity.confidence == 0.99


def test_ml_runtime_result_adapts_entity_spans() -> None:
    fallback = AnalysisResult(
        category=Category.ROAD,
        risk=RiskLevel.MEDIUM,
        sensitivity=Sensitivity.NORMAL,
        jurisdiction_id="JUR-TEST",
        authority_unit_id="UNIT-TEST",
        missing_fields=("ml_model_unavailable",),
        has_mandatory_evidence=False,
    )
    es = EntitySpan(
        label="LOCATION",
        text="Jalan Sudirman",
        start_token=1,
        end_token=3,
        start_char=10,
        end_char=24,
        confidence=0.98,
    )
    res = MLRuntimeResult(
        status=RuntimeStatus.AVAILABLE,
        available=True,
        model_name="test-model",
        latency_ms=1.5,
        entities=[es],
        analysis_fallback=fallback,
    )
    assert len(res.entities) == 1
    assert isinstance(res.entities[0], SpanNER)
    assert isinstance(res.entities[0], EntitySpan)
    assert res.entities[0].start == 10
    assert res.entities[0].start_char == 10
    assert res.entities[0].end == 24
    assert res.entities[0].end_char == 24


def test_artifact_sha_validation_cached_at_init_and_not_rehashed_on_predict(monkeypatch: pytest.MonkeyPatch) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        base_dir = Path(tmpdir)
        content = b"fake-indobert-weights-fp32"
        _, file_hash = _create_dummy_file(base_dir, "weights.bin", content)

        manifest = ArtifactManifest(
            manifest_version="1.0.0",
            artifacts={
                "indobert": ArtifactManifestItem(
                    name="indobert",
                    version="2.0.0",
                    sha256=file_hash,
                    path="weights.bin",
                    size_bytes=len(content),
                    task="classification",
                )
            },
        )

        hash_call_count = 0
        original_compute = compute_file_sha256

        def counting_sha256(path: Path | str) -> str:
            nonlocal hash_call_count
            hash_call_count += 1
            return original_compute(path)

        monkeypatch.setattr("services.ml.manifest.compute_file_sha256", counting_sha256)
        monkeypatch.setattr("services.ml.runtime.compute_file_sha256", counting_sha256)

        def mock_backend(text: str, context: dict | None) -> tuple[ClassificationPrediction, list[SpanNER]]:
            pred = ClassificationPrediction(
                intent="COMPLAINT",
                category="ROAD",
                risk="LOW",
                completeness="COMPLETE",
                confidence=0.9,
            )
            return pred, []

        runtime = LocalMLRuntime(
            manifest=manifest,
            artifacts_dir=str(base_dir),
            inference_backend=mock_backend,
        )

        assert hash_call_count == 1
        assert runtime.is_available() is True
        assert hash_call_count == 1

        for _ in range(10):
            res = runtime.predict("Jalan rusak parah")
            assert res.status == RuntimeStatus.AVAILABLE
            assert res.available is True

        assert hash_call_count == 1

        runtime.warmup(force=True)
        assert hash_call_count == 2

        for _ in range(5):
            res = runtime.predict("Jalan rusak parah lagi")
            assert res.status == RuntimeStatus.AVAILABLE

        assert hash_call_count == 2


def test_runtime_uses_checkpoint_artifact_version_not_manifest_schema_version() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        base_dir = Path(tmpdir)
        content = b"checkpoint-model-bytes"
        _, file_hash = _create_dummy_file(base_dir, "model.bin", content)

        manifest = ArtifactManifest(
            manifest_version="1.0.0",
            artifacts={
                "indobert-multitask": ArtifactManifestItem(
                    name="indobert-multitask",
                    version="v3.2.1-indobert-base-p1",
                    sha256=file_hash,
                    path="model.bin",
                    size_bytes=len(content),
                    task="classification",
                )
            },
        )

        def mock_backend(text: str, context: dict | None) -> tuple[ClassificationPrediction, list[SpanNER]]:
            return (
                ClassificationPrediction(
                    intent="COMPLAINT",
                    category="ROAD",
                    risk="MEDIUM",
                    completeness="COMPLETE",
                    confidence=0.88,
                ),
                [],
            )

        runtime = LocalMLRuntime(
            manifest=manifest,
            artifacts_dir=str(base_dir),
            inference_backend=mock_backend,
            model_name="indobert-multitask",
        )

        assert runtime.is_available() is True
        result = runtime.predict("Laporan jalan")
        assert result.model_version == "v3.2.1-indobert-base-p1"
        assert result.model_version != manifest.manifest_version

        agent_result = runtime_prediction_to_agent_result(result)
        assert agent_result.versions.model == "v3.2.1-indobert-base-p1"
        assert agent_result.versions.model != manifest.manifest_version


def test_successful_backend_does_not_retain_unavailable_fallback_semantics() -> None:
    def mock_backend(text: str, context: dict | None) -> tuple[ClassificationPrediction, list[SpanNER]]:
        pred = ClassificationPrediction(
            intent="COMPLAINT",
            category="DRAINAGE_FLOOD",
            risk="URGENT",
            completeness="COMPLETE",
            confidence=0.96,
            probabilities={"DRAINAGE_FLOOD": 0.96, "ROAD": 0.04},
        )
        spans = [
            SpanNER(label="LOCATION", text="Kecamatan Sleman", start=0, end=16, confidence=0.99),
        ]
        return pred, spans

    runtime = LocalMLRuntime(
        inference_backend=mock_backend,
        model_name="custom-ner-clf",
        model_version="v1.0.0-custom",
    )
    assert runtime.is_available() is True

    result = runtime.predict(
        "Kecamatan Sleman terendam banjir bandang",
        context={"jurisdiction_id": "JUR-DIY-01", "authority_unit_id": "UNIT-BPBD-01"},
    )
    assert result.status == RuntimeStatus.AVAILABLE
    assert result.available is True
    assert result.reason is None
    assert result.prediction is not None
    assert result.prediction.category == "DRAINAGE_FLOOD"
    assert result.prediction.risk == "URGENT"
    assert len(result.entities) == 1

    assert result.analysis_fallback is not None
    assert result.analysis_fallback.category == Category.DRAINAGE_FLOOD
    assert result.analysis_fallback.risk == RiskLevel.URGENT
    assert result.analysis_fallback.jurisdiction_id == "JUR-DIY-01"
    assert result.analysis_fallback.authority_unit_id == "UNIT-BPBD-01"
    assert result.analysis_fallback.has_mandatory_evidence is True
    assert "ml_model_unavailable" not in result.analysis_fallback.missing_fields
    assert result.analysis_fallback.missing_fields == ("completeness_incomplete",)


def test_adapter_agent_result_v1_succeeded_mapping() -> None:
    def mock_backend(text: str, context: dict | None) -> tuple[ClassificationPrediction, list[SpanNER]]:
        pred = ClassificationPrediction(
            intent="COMPLAINT",
            category="WASTE",
            risk="MEDIUM",
            completeness="SUFFICIENT",
            confidence=0.92,
            probabilities={"WASTE": 0.92, "ROAD": 0.08},
        )
        spans = [
            SpanNER(label="LOCATION", text="Pasar Minggu", start=5, end=17, confidence=0.97),
        ]
        return pred, spans

    runtime = LocalMLRuntime(
        inference_backend=mock_backend,
        model_name="indobert-waste",
        model_version="chk-v1.4",
    )

    result = runtime.predict(
        "Ada Pasar Minggu banyak sampah menumpuk",
        context={"case_id": "case-999", "task_id": "task-888", "input_revision": 2},
    )

    agent_result = runtime_prediction_to_agent_result(
        result,
        text="Ada Pasar Minggu banyak sampah menumpuk",
        context={"case_id": "case-999", "task_id": "task-888", "input_revision": 2},
    )

    assert isinstance(agent_result, AgentResultV1)
    assert agent_result.status == ResultStatus.SUCCEEDED
    assert agent_result.case_id == "case-999"
    assert agent_result.task_id == "task-888"
    assert agent_result.input_revision == 2
    assert len(agent_result.input_hash) == 64
    assert agent_result.agent == "indobert-waste"
    assert agent_result.confidence == 0.92
    assert agent_result.error is None
    assert agent_result.data is not None
    assert agent_result.data["category"] == "WASTE"
    assert agent_result.data["risk"] == "MEDIUM"
    assert len(agent_result.data["entities"]) == 1

    assert isinstance(agent_result.uncertainty, UncertaintyInfo)
    assert agent_result.uncertainty.method == UncertaintyMethod.MARGIN
    assert agent_result.uncertainty.value is not None
    assert 0.0 <= agent_result.uncertainty.value <= 1.0

    assert len(agent_result.evidence) == 1
    assert isinstance(agent_result.evidence[0], EvidenceItem)
    assert agent_result.evidence[0].kind == EvidenceKind.TEXT_SPAN
    assert agent_result.evidence[0].claim == "LOCATION:Pasar Minggu"
    assert agent_result.evidence[0].start == 5
    assert agent_result.evidence[0].end == 17

    assert isinstance(agent_result.versions, VersionInfo)
    assert agent_result.versions.model == "chk-v1.4"

    assert isinstance(agent_result.telemetry, TelemetryInfo)
    assert agent_result.telemetry.latency_ms >= 0.0

    method_agent_res = result.to_agent_result(
        text="Ada Pasar Minggu banyak sampah menumpuk",
        context={"case_id": "case-999"},
    )
    assert isinstance(method_agent_res, AgentResultV1)
    assert method_agent_res.status == ResultStatus.SUCCEEDED

    runtime_agent_res = runtime.predict_agent_result(
        "Ada Pasar Minggu banyak sampah menumpuk",
        context={"case_id": "case-999"},
    )
    assert isinstance(runtime_agent_res, AgentResultV1)
    assert runtime_agent_res.status == ResultStatus.SUCCEEDED


def test_adapter_agent_result_v1_unavailable_mapping() -> None:
    runtime = LocalMLRuntime(
        inference_backend=None,
        model_name="indobert-offline",
        model_version="chk-v1.0",
    )
    result = runtime.predict("Jalan rusak parah", context={"case_id": "case-fail-1"})

    agent_result = runtime_prediction_to_agent_result(
        result,
        text="Jalan rusak parah",
        context={"case_id": "case-fail-1"},
    )

    assert isinstance(agent_result, AgentResultV1)
    assert agent_result.status == ResultStatus.UNAVAILABLE
    assert agent_result.data is None
    assert agent_result.confidence is None
    assert agent_result.error is not None
    assert "reason" in agent_result.error
    assert "fallback" in agent_result.error
    assert agent_result.uncertainty.method == UncertaintyMethod.UNKNOWN
    assert agent_result.uncertainty.value is None
    assert agent_result.evidence == ()
    assert agent_result.case_id == "case-fail-1"


def test_adapter_agent_result_v1_custom_overrides() -> None:
    def mock_backend(text: str, context: dict | None) -> tuple[ClassificationPrediction, list[SpanNER]]:
        return (
            ClassificationPrediction(
                intent="COMPLAINT",
                category="CLEAN_WATER",
                risk="LOW",
                completeness="SUFFICIENT",
                confidence=0.85,
            ),
            [],
        )

    runtime = LocalMLRuntime(
        inference_backend=mock_backend,
        model_name="indobert-water",
        model_version="chk-water-01",
    )
    result = runtime.predict("Air PAM mati")

    custom_unc = UncertaintyInfo(method=UncertaintyMethod.ENTROPY, value=0.12, calibrated=True)
    custom_ver = VersionInfo(model="custom-model-01", preprocess="custom-prep-01", calibration="cal-v1", prompt="sys-v1")
    custom_tel = TelemetryInfo(latency_ms=12.5, input_tokens=20, output_tokens=5, cost_usd=0.001, attempt=2)

    agent_result = runtime_prediction_to_agent_result(
        result,
        result_id="res-custom-001",
        task_id="task-custom-001",
        case_id="case-custom-001",
        input_revision=5,
        input_hash="f" * 64,
        agent="agent-water-custom",
        task="water_category",
        uncertainty=custom_unc,
        versions=custom_ver,
        telemetry=custom_tel,
    )

    assert agent_result.result_id == "res-custom-001"
    assert agent_result.task_id == "task-custom-001"
    assert agent_result.case_id == "case-custom-001"
    assert agent_result.input_revision == 5
    assert agent_result.input_hash == "f" * 64
    assert agent_result.agent == "agent-water-custom"
    assert agent_result.task == "water_category"
    assert agent_result.uncertainty == custom_unc
    assert agent_result.versions == custom_ver
    assert agent_result.telemetry == custom_tel
