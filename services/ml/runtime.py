from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path
import time
from typing import Any, Callable, Iterable, Literal, Sequence
from uuid import uuid4

from contracts.models import (
    AgentResultV1,
    AnalysisResult,
    Category,
    CostKind,
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
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from services.intelligence.chunking import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_OVERLAP_TOKENS,
    Chunk,
    TokenSpan,
    chunk_messages,
    chunk_text as _intelligence_chunk_text,
    tokenize_with_offsets,
)
from services.intelligence.completeness import resolve_location_completeness
from services.intelligence.ner import EntitySpan, parse_bio_tags
from services.ml.manifest import (
    ArtifactManifest,
    ArtifactManifestItem,
    ValidationResult,
    compute_file_sha256,
)


class RuntimeStatus(str):
    AVAILABLE = "AVAILABLE"
    UNAVAILABLE = "UNAVAILABLE"
    DEGRADED = "DEGRADED"


class ClassificationPrediction(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    intent: str
    category: str
    risk: str
    completeness: str
    confidence: float = Field(ge=0.0, le=1.0)
    probabilities: dict[str, float] = Field(default_factory=dict)


class SpanNER(EntitySpan):
    model_config = ConfigDict(extra="forbid", frozen=True)

    start: int = Field(default=0, ge=0)
    end: int = Field(default=0, ge=0)
    start_token: int = Field(default=0, ge=0)
    end_token: int = Field(default=0, ge=0)
    start_char: int = Field(default=0, ge=0)
    end_char: int = Field(default=0, ge=0)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if args:
            field_names = ["text", "label", "start", "end", "confidence"]
            for name, val in zip(field_names, args):
                kwargs[name] = val
        super().__init__(**kwargs)

    @model_validator(mode="before")
    @classmethod
    def _normalize_offsets(cls, data: Any) -> Any:
        if isinstance(data, EntitySpan):
            d = data.model_dump()
            d["start"] = d.get("start_char", 0)
            d["end"] = d.get("end_char", 0)
            return d
        if isinstance(data, dict):
            d = dict(data)
            if "start" in d and "start_char" not in d:
                d["start_char"] = d["start"]
            elif "start_char" in d and "start" not in d:
                d["start"] = d["start_char"]
            if "end" in d and "end_char" not in d:
                d["end_char"] = d["end"]
            elif "end_char" in d and "end" not in d:
                d["end"] = d["end_char"]
            return d
        return data

    @classmethod
    def from_entity_span(cls, span: EntitySpan) -> SpanNER:
        if isinstance(span, SpanNER):
            return span
        return cls(
            label=span.label,
            text=span.text,
            start=span.start_char,
            end=span.end_char,
            start_char=span.start_char,
            end_char=span.end_char,
            start_token=span.start_token,
            end_token=span.end_token,
            confidence=span.confidence,
            chunk_id=span.chunk_id,
            source_message_id=span.source_message_id,
        )

    def to_entity_span(self) -> EntitySpan:
        return EntitySpan(
            label=self.label,
            text=self.text,
            start_token=self.start_token,
            end_token=self.end_token,
            start_char=self.start_char,
            end_char=self.end_char,
            confidence=self.confidence,
            chunk_id=self.chunk_id,
            source_message_id=self.source_message_id,
        )


class MLRuntimeResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    status: str
    available: bool
    model_name: str
    model_version: str | None = None
    latency_ms: float = Field(ge=0.0)
    prediction: ClassificationPrediction | None = None
    entities: tuple[SpanNER, ...] = Field(default_factory=tuple)
    analysis_fallback: AnalysisResult | None = None
    reason: str | None = None

    @field_validator("entities", mode="before")
    @classmethod
    def _validate_entities(cls, value: Any) -> tuple[SpanNER, ...]:
        if not value:
            return ()
        adapted: list[SpanNER] = []
        for item in value:
            if isinstance(item, SpanNER):
                adapted.append(item)
            elif isinstance(item, EntitySpan):
                adapted.append(SpanNER.from_entity_span(item))
            elif isinstance(item, dict):
                adapted.append(SpanNER(**item))
            else:
                adapted.append(SpanNER.model_validate(item))
        return tuple(adapted)

    @property
    def analysis_result(self) -> AnalysisResult | None:
        return self.analysis_fallback

    def to_agent_result(
        self,
        *,
        text: str | None = None,
        context: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> AgentResultV1:
        return runtime_prediction_to_agent_result(self, text=text, context=context, **kwargs)


class CpuFp32BenchmarkConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    batch_size: int = Field(default=1, ge=1)
    sequence_length: int = Field(default=448, ge=1)
    overlap_tokens: int = Field(default=64, ge=0)
    num_threads: int = Field(default=1, ge=1)
    warmup_runs: int = Field(default=3, ge=0)
    benchmark_runs: int = Field(default=10, ge=1)
    target_p95_latency_ms: float = Field(default=100.0, gt=0.0)
    precision: Literal["fp32"] = "fp32"
    device: Literal["cpu"] = "cpu"


class CpuFp32BenchmarkResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    config: CpuFp32BenchmarkConfig
    mean_latency_ms: float = Field(ge=0.0)
    p50_latency_ms: float = Field(ge=0.0)
    p95_latency_ms: float = Field(ge=0.0)
    p99_latency_ms: float = Field(ge=0.0)
    min_latency_ms: float = Field(ge=0.0)
    max_latency_ms: float = Field(ge=0.0)
    throughput_qps: float = Field(ge=0.0)
    samples_count: int = Field(ge=1)
    sla_met: bool
    measured_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


def _calculate_percentile(sorted_values: list[float], percentile: float) -> float:
    if not sorted_values:
        return 0.0
    index = (len(sorted_values) - 1) * (percentile / 100.0)
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return sorted_values[int(index)]
    return sorted_values[int(lower)] * (upper - index) + sorted_values[int(upper)] * (index - lower)


def _synthetic_fp32_workload(sequence_length: int) -> float:
    accumulator = 0.0
    bounded_length = min(sequence_length, 256)
    for idx in range(bounded_length):
        multiplier = float(idx) * 0.001
        accumulator += multiplier * multiplier
    return accumulator


def run_cpu_fp32_benchmark(
    config: CpuFp32BenchmarkConfig,
    runner: Callable[[], Any] | None = None,
) -> CpuFp32BenchmarkResult:
    target_runner = runner or (lambda: _synthetic_fp32_workload(config.sequence_length))

    for _ in range(config.warmup_runs):
        target_runner()

    latencies_ms: list[float] = []
    for _ in range(config.benchmark_runs):
        start = time.perf_counter()
        target_runner()
        duration_ms = (time.perf_counter() - start) * 1000.0
        latencies_ms.append(duration_ms)

    sorted_latencies = sorted(latencies_ms)
    total_ms = sum(latencies_ms)
    mean_ms = total_ms / len(latencies_ms)
    p50_ms = _calculate_percentile(sorted_latencies, 50.0)
    p95_ms = _calculate_percentile(sorted_latencies, 95.0)
    p99_ms = _calculate_percentile(sorted_latencies, 99.0)
    min_ms = sorted_latencies[0]
    max_ms = sorted_latencies[-1]

    total_seconds = total_ms / 1000.0
    throughput = (len(latencies_ms) * config.batch_size) / total_seconds if total_seconds > 0 else 0.0
    sla_met = p95_ms <= config.target_p95_latency_ms

    return CpuFp32BenchmarkResult(
        config=config,
        mean_latency_ms=round(mean_ms, 4),
        p50_latency_ms=round(p50_ms, 4),
        p95_latency_ms=round(p95_ms, 4),
        p99_latency_ms=round(p99_ms, 4),
        min_latency_ms=round(min_ms, 4),
        max_latency_ms=round(max_ms, 4),
        throughput_qps=round(throughput, 2),
        samples_count=len(latencies_ms),
        sla_met=sla_met,
    )


class ChunkCompat(Chunk):
    def __eq__(self, other: Any) -> bool:
        if isinstance(other, str):
            return self.text == other
        return super().__eq__(other)

    def split(self, *args: Any, **kwargs: Any) -> list[str]:
        return self.text.split(*args, **kwargs)


def chunk_text(
    text: str,
    max_tokens: int | None = None,
    overlap: int | None = None,
    source_message_id: str | None = None,
    chunk_id_prefix: str = "chk",
    metadata: dict[str, Any] | None = None,
    *,
    max_chunk_tokens: int | None = None,
    overlap_tokens: int | None = None,
    return_strings: bool = False,
    as_strings: bool = False,
) -> list[ChunkCompat] | list[str]:
    resolved_max_tokens = (
        max_tokens
        if max_tokens is not None
        else (max_chunk_tokens if max_chunk_tokens is not None else DEFAULT_MAX_TOKENS)
    )
    resolved_overlap = (
        overlap
        if overlap is not None
        else (overlap_tokens if overlap_tokens is not None else DEFAULT_OVERLAP_TOKENS)
    )

    chunks = _intelligence_chunk_text(
        text=text,
        max_tokens=resolved_max_tokens,
        overlap=resolved_overlap,
        source_message_id=source_message_id,
        chunk_id_prefix=chunk_id_prefix,
        metadata=metadata,
    )

    if return_strings or as_strings:
        if not text.strip():
            return []
        return [c.text for c in chunks]

    return [ChunkCompat.model_validate(c.model_dump()) for c in chunks]


def _map_category(val: str) -> Category:
    try:
        return Category(val)
    except Exception:
        pass
    norm = str(val).upper().strip()
    for c in Category:
        if c.value == norm or c.name == norm:
            return c
    return Category.ROAD


def _map_risk(val: str) -> RiskLevel:
    try:
        return RiskLevel(val)
    except Exception:
        pass
    norm = str(val).upper().strip()
    for r in RiskLevel:
        if r.value == norm or r.name == norm:
            return r
    return RiskLevel.MEDIUM


def _map_sensitivity(val: Any) -> Sensitivity:
    if isinstance(val, Sensitivity):
        return val
    if isinstance(val, str):
        try:
            return Sensitivity(val.upper().strip())
        except Exception:
            pass
    return Sensitivity.NORMAL


class OnnxInferenceBackend:
    """CPU-first ONNX Runtime inference backend for Multitask + NER models."""

    def __init__(
        self,
        multitask_path: str | Path,
        ner_path: str | Path | None = None,
        tokenizer_path: str | Path | None = None,
        temperatures_path: str | Path | None = None,
        tagset_path: str | Path | None = None,
        intra_op_num_threads: int = 2,
        inter_op_num_threads: int = 1,
    ) -> None:
        import onnxruntime as ort
        from transformers import AutoTokenizer

        self.multitask_path = Path(multitask_path).resolve()
        self.ner_path = Path(ner_path).resolve() if ner_path is not None else None

        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = max(1, int(intra_op_num_threads))
        sess_options.inter_op_num_threads = max(1, int(inter_op_num_threads))
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self._multitask_session = ort.InferenceSession(
            str(self.multitask_path), sess_options, providers=["CPUExecutionProvider"]
        )

        self._ner_session = None
        if self.ner_path is not None and self.ner_path.is_file():
            self._ner_session = ort.InferenceSession(
                str(self.ner_path), sess_options, providers=["CPUExecutionProvider"]
            )

        tok_dir = Path(tokenizer_path).resolve() if tokenizer_path else self.multitask_path.parent
        if not (tok_dir / "tokenizer_config.json").is_file():
            candidates = [
                Path("artifacts/city12-v7-multitask"),
                Path("artifacts/city12-multitask"),
                Path("artifacts/city12-v4-ner"),
            ]
            for cand in candidates:
                if (cand / "tokenizer_config.json").is_file():
                    tok_dir = cand
                    break
        self._tokenizer = AutoTokenizer.from_pretrained(str(tok_dir), local_files_only=True)

        self._temperatures: dict[str, float] = {
            "intent": 1.0,
            "category": 1.0,
            "risk": 1.0,
            "completeness": 1.0,
        }
        temp_file = Path(temperatures_path).resolve() if temperatures_path else None
        if temp_file is None or not temp_file.is_file():
            for cand in (
                Path("artifacts/city12-v7-multitask/temperatures.json"),
                Path("artifacts/city12-multitask/temperatures.json"),
            ):
                if cand.is_file():
                    temp_file = cand
                    break
        if temp_file is not None and temp_file.is_file():
            try:
                with open(temp_file, "r", encoding="utf-8") as f:
                    self._temperatures.update(json.load(f))
            except Exception:
                pass

        self._tagset = [
            "O",
            "B-LOC",
            "I-LOC",
            "B-OBJ",
            "I-OBJ",
            "B-TIME",
            "I-TIME",
        ]
        tag_file = Path(tagset_path).resolve() if tagset_path else None
        if tag_file is None or not tag_file.is_file():
            for cand in (
                Path("artifacts/city12-v4-ner/tagset.json"),
                Path("artifacts/city12-ner/tagset.json"),
            ):
                if cand.is_file():
                    tag_file = cand
                    break
        if tag_file is not None and tag_file.is_file():
            try:
                with open(tag_file, "r", encoding="utf-8") as f:
                    self._tagset = json.load(f)
            except Exception:
                pass

        self._head_names = ["intent", "category", "risk", "completeness"]
        self._head_configs = {
            "intent": ["COMPLAINT", "INQUIRY", "FEEDBACK"],
            "category": [c.value for c in Category],
            "risk": ["LOW", "MEDIUM", "HIGH", "URGENT"],
            "completeness": ["SUFFICIENT", "INCOMPLETE", "AMBIGUOUS"],
        }

    def predict(
        self, text: str, context: dict[str, Any] | None = None
    ) -> tuple[ClassificationPrediction, tuple[SpanNER, ...]]:
        import numpy as np

        inputs = self._tokenizer(
            text,
            max_length=448,
            padding=True,
            truncation=True,
            return_offsets_mapping=True,
            return_tensors="np",
        )

        feed_dict = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
        }
        m_outputs = self._multitask_session.run(None, feed_dict)

        head_preds: dict[str, str] = {}
        head_probs: dict[str, float] = {}

        for h_idx, head in enumerate(self._head_names):
            t = self._temperatures.get(head, 1.0)
            if t <= 0.0:
                t = 1.0
            logits = m_outputs[h_idx][0] / t
            exp_z = np.exp(logits - np.max(logits))
            probs = exp_z / np.sum(exp_z)
            pred_idx = int(np.argmax(probs))
            classes = self._head_configs[head]
            pred_label = classes[pred_idx] if pred_idx < len(classes) else classes[0]
            head_preds[head] = pred_label
            head_probs[head] = float(probs[pred_idx])

        confidence = float(head_probs.get("category", 1.0))

        prediction = ClassificationPrediction(
            intent=head_preds.get("intent", "COMPLAINT"),
            category=head_preds.get("category", "ROAD"),
            risk=head_preds.get("risk", "MEDIUM"),
            completeness=head_preds.get("completeness", "SUFFICIENT"),
            confidence=round(confidence, 4),
            probabilities={k: round(v, 4) for k, v in head_probs.items()},
        )

        entities: list[SpanNER] = []
        if self._ner_session is not None:
            n_outputs = self._ner_session.run(None, feed_dict)
            n_logits = n_outputs[0][0]
            n_preds = np.argmax(n_logits, axis=-1)
            tags = [self._tagset[p] if p < len(self._tagset) else "O" for p in n_preds]
            tokens = self._tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
            offsets = [tuple(o) for o in inputs["offset_mapping"][0]]

            valid_tokens = []
            valid_tags = []
            valid_offsets = []
            for tok_str, tag, off in zip(tokens, tags, offsets):
                if off == (0, 0) and tok_str in ("[CLS]", "[SEP]", "[PAD]"):
                    continue
                valid_tokens.append(tok_str)
                valid_tags.append(tag)
                valid_offsets.append(off)

            extracted_spans = parse_bio_tags(
                tokens=valid_tokens,
                tags=valid_tags,
                text=text,
                token_offsets=valid_offsets,
            )
            entities = [SpanNER.from_entity_span(s) for s in extracted_spans]

        return prediction, tuple(entities)


class LocalMLRuntime:
    def __init__(
        self,
        manifest: ArtifactManifest | None = None,
        artifacts_dir: str | None = None,
        allow_network: bool = False,
        inference_backend: Any = None,
        model_name: str = "m3-indobert-multitask",
        model_version: str | None = None,
        artifact_name: str | None = None,
    ) -> None:
        if allow_network:
            raise ValueError("Network access is strictly forbidden in local-ML runtime")

        self.manifest = manifest
        self.artifacts_dir = artifacts_dir
        self.inference_backend = inference_backend
        self.model_name = model_name
        self.artifact_name = artifact_name
        self._configured_model_version = model_version
        self._artifacts_validation_cache: dict[str, ValidationResult] | None = None
        self._is_artifacts_valid: bool = False
        self._unavailable_detail: str | None = None

        if self.manifest is not None and self.artifacts_dir is not None:
            self.warmup()

    @classmethod
    def live_from_artifacts(
        cls,
        base_dir: str | Path = "artifacts",
        model_name: str = "city12-indobert-multitask",
        model_version: str = "v7.0.0",
        allow_fallback: bool = True,
    ) -> LocalMLRuntime:
        root = Path(base_dir).resolve()
        multitask_candidates = [
            root / "city12-v7-onnx" / "multitask" / "model.onnx",
            root / "city12-onnx" / "multitask" / "model.onnx",
            root / "city12-v4-onnx" / "multitask" / "model.onnx",
        ]
        ner_candidates = [
            root / "city12-v4-onnx" / "ner" / "model.onnx",
            root / "city12-onnx" / "ner" / "model.onnx",
        ]
        tok_candidates = [
            root / "city12-v7-multitask",
            root / "city12-multitask",
        ]

        multitask_path = next((p for p in multitask_candidates if p.is_file()), None)
        ner_path = next((p for p in ner_candidates if p.is_file()), None)
        tok_path = next((p for p in tok_candidates if p.is_dir()), None)

        if multitask_path is None:
            if not allow_fallback:
                raise FileNotFoundError(f"No ONNX multitask model found under {root}")
            return cls(model_name=model_name, model_version=model_version)

        try:
            backend = OnnxInferenceBackend(
                multitask_path=multitask_path,
                ner_path=ner_path,
                tokenizer_path=tok_path,
            )
            return cls(
                inference_backend=backend,
                model_name=model_name,
                model_version=model_version,
            )
        except Exception as exc:
            if not allow_fallback:
                raise
            return cls(model_name=model_name, model_version=model_version)

    def warmup(self, force: bool = False) -> dict[str, ValidationResult]:
        if self._artifacts_validation_cache is not None and not force:
            return self._artifacts_validation_cache

        if self.manifest is None:
            self._artifacts_validation_cache = {}
            self._is_artifacts_valid = True
            self._unavailable_detail = None
            return self._artifacts_validation_cache

        if self.artifacts_dir is None:
            self._artifacts_validation_cache = {}
            self._is_artifacts_valid = False
            self._unavailable_detail = (
                "Manifest provided but artifacts directory is missing; "
                "running in safe local fallback mode without network access"
            )
            return self._artifacts_validation_cache

        if not self.manifest.artifacts:
            self._artifacts_validation_cache = {}
            self._is_artifacts_valid = False
            self._unavailable_detail = (
                "Manifest contains no artifacts; "
                "running in safe local fallback mode without network access"
            )
            return self._artifacts_validation_cache

        results = self.manifest.validate_all(self.artifacts_dir, raise_on_error=False, use_cache=not force)
        self._artifacts_validation_cache = results
        self._is_artifacts_valid = bool(results) and all(res.is_valid for res in results.values())
        if not self._is_artifacts_valid:
            failed = [name for name, res in results.items() if not res.is_valid]
            self._unavailable_detail = (
                f"Artifacts validation failed or missing files: {failed}; "
                "running in safe local fallback mode without network access"
            )
        else:
            self._unavailable_detail = None
        return self._artifacts_validation_cache

    def get_artifact_version(self) -> str | None:
        if self.manifest is None or not self.manifest.artifacts:
            return self._configured_model_version

        if self.artifact_name and self.artifact_name in self.manifest.artifacts:
            return self.manifest.artifacts[self.artifact_name].version

        if self.model_name in self.manifest.artifacts:
            return self.manifest.artifacts[self.model_name].version

        target = self.artifact_name or self.model_name
        for item in self.manifest.artifacts.values():
            if item.name == target:
                return item.version

        if len(self.manifest.artifacts) == 1:
            return next(iter(self.manifest.artifacts.values())).version

        for item in self.manifest.artifacts.values():
            if item.task in ("classification", "multitask", "general"):
                return item.version

        first = next(iter(self.manifest.artifacts.values()))
        return first.version

    def get_model_version(self) -> str | None:
        version = self.get_artifact_version()
        if version is not None:
            return version
        return self._configured_model_version

    def _is_backend_callable(self) -> bool:
        if self.inference_backend is None:
            return False
        if callable(self.inference_backend):
            return True
        return hasattr(self.inference_backend, "predict") and callable(getattr(self.inference_backend, "predict"))

    def is_available(self) -> bool:
        if not self._is_backend_callable():
            return False

        if self.manifest is not None:
            if self._artifacts_validation_cache is None:
                self.warmup()
            return self._is_artifacts_valid

        return True

    def _unavailable_reason(self) -> str:
        if not self._is_backend_callable():
            if self.manifest is not None:
                return (
                    "Valid artifacts exist but no callable inference backend is configured; "
                    "running in safe local fallback mode without network access"
                )
            return (
                "Inference backend is not callable or not configured; "
                "running in safe local fallback mode without network access"
            )
        if self.manifest is not None:
            if self.artifacts_dir is None:
                return (
                    "Manifest provided but artifacts directory is missing; "
                    "running in safe local fallback mode without network access"
                )
            if not self.manifest.artifacts:
                return (
                    "Manifest contains no artifacts; "
                    "running in safe local fallback mode without network access"
                )
            if self._unavailable_detail:
                return self._unavailable_detail
            return (
                "Artifacts validation failed or missing files; "
                "running in safe local fallback mode without network access"
            )
        return (
            "Artifacts not loaded or runtime unavailable; "
            "running in safe local fallback mode without network access"
        )

    def build_safe_fallback(
        self,
        text: str,
        context: dict[str, Any] | None = None,
    ) -> AnalysisResult:
        missing = ["ml_model_unavailable"]
        if not text.strip():
            missing.append("text_empty")

        jurisdiction_id = "JUR-UNASSIGNED"
        authority_unit_id = "UNIT-UNASSIGNED"
        if context:
            jurisdiction_id = str(context.get("jurisdiction_id", jurisdiction_id))
            authority_unit_id = str(context.get("authority_unit_id", authority_unit_id))

        return AnalysisResult(
            category=Category.ROAD,
            risk=RiskLevel.MEDIUM,
            sensitivity=Sensitivity.NORMAL,
            jurisdiction_id=jurisdiction_id,
            authority_unit_id=authority_unit_id,
            missing_fields=tuple(missing),
            has_mandatory_evidence=False,
        )

    def build_prediction_analysis(
        self,
        prediction: ClassificationPrediction,
        entities: Sequence[SpanNER] = (),
        text: str = "",
        context: dict[str, Any] | None = None,
    ) -> AnalysisResult:
        category = _map_category(prediction.category)
        risk = _map_risk(prediction.risk)
        sensitivity = _map_sensitivity(context.get("sensitivity") if context else None)

        jurisdiction_id = "JUR-UNASSIGNED"
        authority_unit_id = "UNIT-UNASSIGNED"
        if context:
            jurisdiction_id = str(context.get("jurisdiction_id", jurisdiction_id))
            authority_unit_id = str(context.get("authority_unit_id", authority_unit_id))

        missing: list[str] = []
        if not text.strip():
            missing.append("text_empty")
        completeness = resolve_location_completeness(text, entities)
        if completeness in ("INCOMPLETE", "AMBIGUOUS"):
            missing.append(f"completeness_{completeness.lower()}")

        has_evidence = len(entities) > 0

        return AnalysisResult(
            category=category,
            risk=risk,
            sensitivity=sensitivity,
            jurisdiction_id=jurisdiction_id,
            authority_unit_id=authority_unit_id,
            missing_fields=tuple(missing),
            has_mandatory_evidence=has_evidence,
        )

    @staticmethod
    def _adapt_entities(raw_entities: Iterable[Any]) -> tuple[SpanNER, ...]:
        adapted: list[SpanNER] = []
        for item in raw_entities:
            if isinstance(item, SpanNER):
                adapted.append(item)
            elif isinstance(item, EntitySpan):
                adapted.append(SpanNER.from_entity_span(item))
            elif isinstance(item, dict):
                adapted.append(SpanNER(**item))
            else:
                adapted.append(SpanNER.model_validate(item))
        return tuple(adapted)

    def predict(
        self,
        text: str,
        context: dict[str, Any] | None = None,
    ) -> MLRuntimeResult:
        model_ver = self.get_model_version()
        if not self.is_available():
            fallback = self.build_safe_fallback(text, context)
            return MLRuntimeResult(
                status=RuntimeStatus.UNAVAILABLE,
                available=False,
                model_name=self.model_name,
                model_version=model_ver,
                latency_ms=0.0,
                prediction=None,
                entities=(),
                analysis_fallback=fallback,
                reason=self._unavailable_reason(),
            )

        start = time.perf_counter()
        prediction: ClassificationPrediction | None = None
        entities: tuple[SpanNER, ...] = ()

        try:
            if callable(self.inference_backend):
                output = self.inference_backend(text, context)
            elif hasattr(self.inference_backend, "predict") and callable(getattr(self.inference_backend, "predict")):
                output = self.inference_backend.predict(text, context)
            else:
                output = None

            if isinstance(output, tuple) and len(output) == 2:
                raw_pred, raw_entities = output
                if isinstance(raw_pred, dict):
                    prediction = ClassificationPrediction(**raw_pred)
                elif isinstance(raw_pred, ClassificationPrediction) or raw_pred is None:
                    prediction = raw_pred
                if raw_entities:
                    entities = self._adapt_entities(raw_entities)
            elif isinstance(output, ClassificationPrediction):
                prediction = output
            elif isinstance(output, dict):
                prediction = ClassificationPrediction(**output)
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            fallback = self.build_safe_fallback(text, context)
            return MLRuntimeResult(
                status=RuntimeStatus.UNAVAILABLE,
                available=False,
                model_name=self.model_name,
                model_version=model_ver,
                latency_ms=round(elapsed_ms, 4),
                prediction=None,
                entities=(),
                analysis_fallback=fallback,
                reason=f"Inference execution failed: {exc}; running in safe local fallback mode without network access",
            )

        elapsed_ms = (time.perf_counter() - start) * 1000.0

        if prediction is None and not entities:
            fallback = self.build_safe_fallback(text, context)
            return MLRuntimeResult(
                status=RuntimeStatus.UNAVAILABLE,
                available=False,
                model_name=self.model_name,
                model_version=model_ver,
                latency_ms=round(elapsed_ms, 4),
                prediction=None,
                entities=(),
                analysis_fallback=fallback,
                reason="Inference backend produced no valid prediction or entities; running in safe local fallback mode without network access",
            )

        if prediction is not None:
            prediction = prediction.model_copy(
                update={"completeness": resolve_location_completeness(text, entities)}
            )
        analysis_result = (
            self.build_prediction_analysis(
                prediction=prediction,
                entities=entities,
                text=text,
                context=context,
            )
            if prediction is not None
            else None
        )

        return MLRuntimeResult(
            status=RuntimeStatus.AVAILABLE,
            available=True,
            model_name=self.model_name,
            model_version=model_ver,
            latency_ms=round(elapsed_ms, 4),
            prediction=prediction,
            entities=entities,
            analysis_fallback=analysis_result,
            reason=None,
        )

    def predict_agent_result(
        self,
        text: str,
        context: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> AgentResultV1:
        res = self.predict(text, context)
        return runtime_prediction_to_agent_result(res, text=text, context=context, **kwargs)


def runtime_prediction_to_agent_result(
    result: MLRuntimeResult,
    *,
    text: str | None = None,
    context: dict[str, Any] | None = None,
    result_id: str | None = None,
    task_id: str | None = None,
    case_id: str | None = None,
    input_revision: int | None = None,
    input_hash: str | None = None,
    agent: str | None = None,
    task: str | None = None,
    preprocess_version: str = "chunk-448",
    calibration_version: str | None = None,
    prompt_version: str | None = None,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    cost_usd: float | None = None,
    attempt: int = 1,
    uncertainty: UncertaintyInfo | None = None,
    evidence: Sequence[EvidenceItem] | None = None,
    versions: VersionInfo | None = None,
    telemetry: TelemetryInfo | None = None,
) -> AgentResultV1:
    if input_hash is not None:
        resolved_input_hash = input_hash
    else:
        text_content = text if text is not None else (context.get("text", "") if context else "")
        if context:
            ctx_filtered = {k: v for k, v in context.items() if k != "text"}
            ctx_str = json.dumps(ctx_filtered, sort_keys=True, separators=(",", ":"), default=str)
            resolved_input_hash = sha256(f"{text_content}\n{ctx_str}".encode("utf-8")).hexdigest()
        else:
            resolved_input_hash = sha256(text_content.encode("utf-8")).hexdigest()

    resolved_case_id = (
        case_id
        or (str(context.get("case_id")) if context and "case_id" in context else None)
        or "case-unassigned"
    )
    resolved_task_id = (
        task_id
        or (str(context.get("task_id")) if context and "task_id" in context else None)
        or f"task-{uuid4().hex[:12]}"
    )
    resolved_result_id = result_id or f"res-{uuid4().hex[:16]}"
    resolved_input_revision = (
        input_revision
        if input_revision is not None
        else (int(context.get("case_revision", context.get("input_revision", 1))) if context else 1)
    )
    resolved_agent = agent or result.model_name
    resolved_task = task or (str(context.get("task")) if context and "task" in context else "category")

    is_success = result.available and result.prediction is not None
    if is_success:
        mapped_status = ResultStatus.SUCCEEDED
        pred = result.prediction
        data: dict[str, Any] | None = {
            "intent": pred.intent,
            "category": pred.category,
            "risk": pred.risk,
            "completeness": pred.completeness,
            "confidence": pred.confidence,
            "probabilities": dict(pred.probabilities),
        }
        if result.entities:
            data["entities"] = [e.model_dump() for e in result.entities]
        error: dict[str, Any] | None = None
        confidence: float | None = pred.confidence
    else:
        if result.status == "TIMED_OUT":
            mapped_status = ResultStatus.TIMED_OUT
        elif result.status == "INVALID_OUTPUT":
            mapped_status = ResultStatus.INVALID_OUTPUT
        else:
            mapped_status = ResultStatus.UNAVAILABLE
        data = None
        confidence = None
        error = {
            "reason": result.reason or "ML runtime unavailable or failed",
            "runtime_status": result.status,
            "model_name": result.model_name,
        }
        if result.analysis_fallback is not None:
            error["fallback"] = result.analysis_fallback.model_dump()

    if uncertainty is not None:
        resolved_uncertainty = uncertainty
    elif is_success and result.prediction is not None:
        pred = result.prediction
        if pred.probabilities and len(pred.probabilities) >= 2:
            sorted_probs = sorted(pred.probabilities.values(), reverse=True)
            margin = sorted_probs[0] - sorted_probs[1]
            unc_val = max(0.0, min(1.0, round(1.0 - margin, 4)))
            resolved_uncertainty = UncertaintyInfo(
                method=UncertaintyMethod.MARGIN,
                value=unc_val,
                calibrated=False,
            )
        else:
            unc_val = max(0.0, min(1.0, round(1.0 - pred.confidence, 4)))
            resolved_uncertainty = UncertaintyInfo(
                method=UncertaintyMethod.MARGIN,
                value=unc_val,
                calibrated=False,
            )
    else:
        resolved_uncertainty = UncertaintyInfo(
            method=UncertaintyMethod.UNKNOWN,
            value=None,
            calibrated=False,
        )

    if evidence is not None:
        resolved_evidence = tuple(evidence)
    elif result.entities:
        ev_items: list[EvidenceItem] = []
        for idx, span in enumerate(result.entities):
            ref = span.source_message_id or span.chunk_id or f"msg-span-{idx}"
            claim = f"{span.label}:{span.text}"
            start = span.start_char if span.start_char is not None else span.start
            end = span.end_char if span.end_char is not None else span.end
            if start is not None and end is not None and end < start:
                end = start
            ev_items.append(
                EvidenceItem(
                    kind=EvidenceKind.TEXT_SPAN,
                    ref=ref,
                    claim=claim,
                    start=start,
                    end=end,
                )
            )
        resolved_evidence = tuple(ev_items)
    else:
        resolved_evidence = ()

    if versions is not None:
        resolved_versions = versions
    else:
        model_str = result.model_version or result.model_name
        resolved_versions = VersionInfo(
            model=model_str,
            preprocess=preprocess_version,
            calibration=calibration_version,
            prompt=prompt_version,
        )

    if telemetry is not None:
        resolved_telemetry = telemetry
    else:
        resolved_telemetry = TelemetryInfo(
            latency_ms=max(0.0, float(result.latency_ms)),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
            cost_kind=CostKind.ACTUAL if cost_usd is not None else CostKind.UNKNOWN,
            attempt=attempt,
        )

    return AgentResultV1(
        result_id=resolved_result_id,
        task_id=resolved_task_id,
        case_id=resolved_case_id,
        input_revision=resolved_input_revision,
        input_hash=resolved_input_hash,
        agent=resolved_agent,
        task=resolved_task,
        status=mapped_status,
        data=data,
        confidence=confidence,
        uncertainty=resolved_uncertainty,
        evidence=resolved_evidence,
        versions=resolved_versions,
        telemetry=resolved_telemetry,
        error=error,
    )


to_agent_result_v1 = runtime_prediction_to_agent_result
