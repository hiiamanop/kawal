from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
import time
from unittest import mock

import pytest

from contracts.models import (
    AnalysisResult,
    Category,
    ComplaintTrajectory,
    DatasetSplit,
    DecisionMode,
    FamilySplitAuditResult,
    TrajectoryBubble,
    TrajectoryTurn,
    TurnExpectedAction,
)
from services.dataset import (
    ScenarioTemplate,
    TrajectoryGenerator,
    assign_split_by_family,
    audit_family_splits,
    build_family_split_map,
    generate_dataset,
    generate_trajectory,
    partition_families_by_split,
)
from services.dataset.generator import DEFAULT_TEMPLATES
from services.dataset.splits import (
    build_stratified_family_split_map,
    partition_stratified_family_splits,
)
from services.intelligence.calibration import (
    CpuBenchmarkConfig,
    CpuBenchmarkResult,
    TemperatureCalibrator,
    load_temperatures,
    run_cpu_benchmark,
    save_temperatures,
)
from services.intelligence.chunking import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_OVERLAP_TOKENS,
    Chunk,
    chunk_messages,
    chunk_text,
    tokenize_with_offsets,
)
from services.intelligence.ner import (
    EntitySpan,
    extract_entities_from_text,
    merge_overlapping_spans,
    parse_bio_tags,
)
from services.ml import (
    ArtifactManifest,
    ArtifactManifestItem,
    ClassificationPrediction,
    CpuFp32BenchmarkConfig,
    CpuFp32BenchmarkResult,
    LocalMLRuntime,
    MLRuntimeResult,
    RuntimeStatus,
    SpanNER,
    compute_file_sha256,
    run_cpu_fp32_benchmark,
)


def test_evaluation_split_audit_zero_leakage_on_full_dataset() -> None:
    seeds = (42, 101, 777)
    for seed in seeds:
        dataset = generate_dataset(
            num_scenarios=36,
            seed=seed,
            train_ratio=0.7,
            dev_ratio=0.15,
            multi_turn_ratio=0.5,
        )
        assert len(dataset) == 36

        audit = audit_family_splits(dataset)
        assert isinstance(audit, FamilySplitAuditResult)
        assert audit.passed is True
        assert len(audit.violations) == 0
        assert len(audit.oracle_leakages) == 0
        assert len(audit.cross_split_text_duplicates) == 0
        assert len(audit.family_overlap) == 0
        assert audit.total_trajectories == 36

        assert DatasetSplit.TRAIN.value in audit.split_trajectories
        assert DatasetSplit.DEV.value in audit.split_trajectories
        assert DatasetSplit.TEST.value in audit.split_trajectories

        assert audit.split_families[DatasetSplit.TRAIN.value] > 0
        assert audit.split_families[DatasetSplit.DEV.value] > 0
        assert audit.split_families[DatasetSplit.TEST.value] > 0


def test_evaluation_split_audit_determinism() -> None:
    ds1 = generate_dataset(num_scenarios=24, seed=99)
    ds2 = generate_dataset(num_scenarios=24, seed=99)

    dump1 = [t.model_dump_json() for t in ds1]
    dump2 = [t.model_dump_json() for t in ds2]
    assert dump1 == dump2

    audit1 = audit_family_splits(ds1)
    audit2 = audit_family_splits(ds2)
    assert audit1.model_dump_json() == audit2.model_dump_json()


def test_evaluation_stratified_family_split_p0_category_coverage() -> None:
    f2c = {t.family_id: t.category for t in DEFAULT_TEMPLATES}
    for seed in (0, 13, 42, 128, 999):
        split_map = build_stratified_family_split_map(f2c, seed=seed)
        partition = partition_stratified_family_splits(f2c, seed=seed)

        assert len(split_map) == len(DEFAULT_TEMPLATES)
        p0_categories = {t.category for t in DEFAULT_TEMPLATES}
        for cat in p0_categories:
            cat_families = [f for f, c in f2c.items() if c == cat]
            splits_for_cat = {split_map[f] for f in cat_families}
            assert DatasetSplit.TRAIN in splits_for_cat
            assert DatasetSplit.DEV in splits_for_cat
            assert DatasetSplit.TEST in splits_for_cat

            train_fams = set(partition[DatasetSplit.TRAIN])
            dev_fams = set(partition[DatasetSplit.DEV])
            test_fams = set(partition[DatasetSplit.TEST])

            assert train_fams.isdisjoint(dev_fams)
            assert train_fams.isdisjoint(test_fams)
            assert dev_fams.isdisjoint(test_fams)

            assert any(f in train_fams for f in cat_families)
            assert any(f in dev_fams for f in cat_families)
            assert any(f in test_fams for f in cat_families)


def test_evaluation_bubble_isolation_against_metadata_leakage() -> None:
    gen = TrajectoryGenerator()
    personas = ("STANDARD", "FORMAL", "FRUSTRATED_RAMBLING", "PANICKED")
    noise_levels = ("LOW", "MEDIUM", "HIGH")

    for template in DEFAULT_TEMPLATES:
        for persona in personas:
            for noise in noise_levels:
                traj = gen.generate_trajectory(
                    scenario_id=f"sc-eval-{template.family_id}",
                    family_id=template.family_id,
                    persona=persona,
                    noise_level=noise,
                    multi_turn=True,
                    seed=42,
                )
                for turn in traj.turns:
                    for bubble in turn.bubbles:
                        lower_text = bubble.text.lower()
                        assert traj.scenario_id.lower() not in lower_text
                        assert template.family_id.lower() not in lower_text
                        for cat in Category:
                            assert f"category: {cat.value.lower()}" not in lower_text
                            assert f"kategori: {cat.value.lower()}" not in lower_text
                            if "_" in cat.value:
                                assert cat.value.lower() not in lower_text
                        for k, v in template.world_truth.items():
                            assert f"{k}: {v}".lower() not in lower_text
                            if "_" in str(k):
                                assert str(k).lower() not in lower_text
                            if "_" in str(v):
                                assert str(v).lower() not in lower_text
                        assert "world_truth" not in lower_text


def test_evaluation_split_audit_failure_sensitivity() -> None:
    t1 = ComplaintTrajectory(
        scenario_id="sc-ok-1",
        family_id="fam-shared-eval",
        split=DatasetSplit.TRAIN,
        category=Category.ROAD,
        turns=(
            TrajectoryTurn(
                turn=1,
                bubbles=(
                    TrajectoryBubble(source_message_id="m1", text="Jalan ambrol berlubang"),
                ),
                expected_action=TurnExpectedAction(
                    turn=1,
                    allowed_actions=(DecisionMode.EXECUTE,),
                    missing=(),
                ),
            ),
        ),
    )
    t2 = ComplaintTrajectory(
        scenario_id="sc-ok-2",
        family_id="fam-shared-eval",
        split=DatasetSplit.TEST,
        category=Category.ROAD,
        turns=(
            TrajectoryTurn(
                turn=1,
                bubbles=(
                    TrajectoryBubble(source_message_id="m2", text="Aspal patah dan terbelah"),
                ),
                expected_action=TurnExpectedAction(
                    turn=1,
                    allowed_actions=(DecisionMode.EXECUTE,),
                    missing=(),
                ),
            ),
        ),
    )
    audit_overlap = audit_family_splits([t1, t2])
    assert audit_overlap.passed is False
    assert len(audit_overlap.family_overlap) > 0
    assert any("Family overlap detected" in v for v in audit_overlap.violations)

    t_hidden = ComplaintTrajectory(
        scenario_id="sc-leak-hf",
        family_id="fam-hf-leak",
        split=DatasetSplit.TRAIN,
        category=Category.ROAD,
        turns=(
            TrajectoryTurn(
                turn=1,
                bubbles=(
                    TrajectoryBubble(
                        source_message_id="m1",
                        text="Ada jalan rusak di RT04 RW02 Kelurahan Cikini",
                    ),
                ),
                hidden_facts=("Kelurahan Cikini",),
                expected_action=TurnExpectedAction(
                    turn=1,
                    allowed_actions=(DecisionMode.REQUEST_CLARIFICATION,),
                    missing=("jurisdiction",),
                ),
            ),
        ),
    )
    audit_hf = audit_family_splits([t_hidden])
    assert audit_hf.passed is False
    assert len(audit_hf.oracle_leakages) > 0

    repeated_text = "aspal mengelupas tergenang air saluran bocor parah sekali"
    t_dup_train = ComplaintTrajectory(
        scenario_id="sc-dup-1",
        family_id="fam-dup-1",
        split=DatasetSplit.TRAIN,
        category=Category.ROAD,
        turns=(
            TrajectoryTurn(
                turn=1,
                bubbles=(TrajectoryBubble(source_message_id="m1", text=repeated_text),),
                expected_action=TurnExpectedAction(
                    turn=1,
                    allowed_actions=(DecisionMode.EXECUTE,),
                    missing=(),
                ),
            ),
        ),
    )
    t_dup_test = ComplaintTrajectory(
        scenario_id="sc-dup-2",
        family_id="fam-dup-2",
        split=DatasetSplit.TEST,
        category=Category.ROAD,
        turns=(
            TrajectoryTurn(
                turn=1,
                bubbles=(TrajectoryBubble(source_message_id="m2", text=repeated_text),),
                expected_action=TurnExpectedAction(
                    turn=1,
                    allowed_actions=(DecisionMode.EXECUTE,),
                    missing=(),
                ),
            ),
        ),
    )
    audit_dup = audit_family_splits([t_dup_train, t_dup_test])
    assert audit_dup.passed is False
    assert len(audit_dup.cross_split_text_duplicates) > 0


def test_evaluation_chunking_geometry_448_64() -> None:
    step = DEFAULT_MAX_TOKENS - DEFAULT_OVERLAP_TOKENS
    assert DEFAULT_MAX_TOKENS == 448
    assert DEFAULT_OVERLAP_TOKENS == 64
    assert step == 384

    lengths = (100, 448, 449, 832, 833, 1152, 1200)
    for length in lengths:
        tokens = [f"tok_{i}" for i in range(length)]
        text = " ".join(tokens)
        chunks = chunk_text(text, max_tokens=448, overlap=64)

        if length <= 448:
            assert len(chunks) == 1
            assert chunks[0].start_token == 0
            assert chunks[0].end_token == length
        else:
            expected_count = math.ceil((length - 448) / step) + 1
            assert len(chunks) == expected_count

            for i in range(len(chunks) - 1):
                overlap = chunks[i].end_token - chunks[i + 1].start_token
                assert overlap == 64
                assert chunks[i + 1].start_token == chunks[i].start_token + step

            assert chunks[-1].end_token == length


def test_evaluation_long_text_tail_evidence_preserved_in_final_chunk() -> None:
    intro_words = [f"latar_belakang_aduan_nomor_{i}" for i in range(950)]
    intro_text = " ".join(intro_words)
    tail_evidence_location = "Jalan Magelang KM 14 Dusun Murangan Sleman"
    tail_evidence_object = "gorong-gorong drainase ambrol sedalam 2 meter"
    tail_evidence_time = "tadi subuh jam 05.00"
    tail_evidence_urgency = "dua pengendara motor terperosok"
    tail_text = (
        f"Titik lokasi kerusakan persis di {tail_evidence_location} dengan kondisi "
        f"{tail_evidence_object} terjadi {tail_evidence_time} sehingga {tail_evidence_urgency}."
    )
    full_complaint = f"{intro_text} {tail_text}"

    tokens = tokenize_with_offsets(full_complaint)
    assert len(tokens) > 950

    chunks = chunk_text(full_complaint, max_tokens=448, overlap=64)
    assert len(chunks) >= 3

    for chk in chunks[:-1]:
        assert tail_evidence_location not in chk.text
        assert tail_evidence_object not in chk.text

    final_chunk = chunks[-1]
    assert tail_evidence_location in final_chunk.text
    assert tail_evidence_object in final_chunk.text
    assert tail_evidence_time in final_chunk.text
    assert tail_evidence_urgency in final_chunk.text

    loc_local_offset = final_chunk.text.find(tail_evidence_location)
    assert loc_local_offset != -1

    loc_global_offset = final_chunk.to_global_char_offset(loc_local_offset)
    assert full_complaint[loc_global_offset : loc_global_offset + len(tail_evidence_location)] == tail_evidence_location
    assert final_chunk.to_local_char_offset(loc_global_offset) == loc_local_offset
    assert final_chunk.contains_global_offset(loc_global_offset) is True

    obj_local_offset = final_chunk.text.find(tail_evidence_object)
    assert obj_local_offset != -1
    obj_global_offset = final_chunk.to_global_char_offset(obj_local_offset)
    assert full_complaint[obj_global_offset : obj_global_offset + len(tail_evidence_object)] == tail_evidence_object


def test_evaluation_tail_evidence_preserved_through_span_aggregation() -> None:
    intro_words = [f"narasi_panjang_{i}" for i in range(950)]
    intro_text = " ".join(intro_words)
    tail_loc = "Jalan Kaliurang KM 12"
    tail_obj = "lubang aspal amblas parah"
    tail_text = f"Posisi berada di {tail_loc} terdapat {tail_obj}."
    full_text = f"{intro_text} {tail_text}"

    chunks = chunk_text(full_text, max_tokens=448, overlap=64)
    assert len(chunks) >= 3

    all_extracted_spans: list[EntitySpan] = []
    for chunk in chunks:
        extracted = extract_entities_from_text(chunk.text)
        for span in extracted:
            global_start = chunk.to_global_char_offset(span.start_char)
            global_end = chunk.to_global_char_offset(span.end_char)
            all_extracted_spans.append(
                EntitySpan(
                    label=span.label,
                    text=span.text,
                    start_token=span.start_token + chunk.start_token,
                    end_token=span.end_token + chunk.start_token,
                    start_char=global_start,
                    end_char=global_end,
                    confidence=span.confidence,
                    chunk_id=chunk.chunk_id,
                )
            )

    aggregated_spans = merge_overlapping_spans(all_extracted_spans, text=full_text)
    assert len(aggregated_spans) > 0

    tail_loc_spans = [s for s in aggregated_spans if tail_loc in s.text or s.text in tail_loc]
    assert len(tail_loc_spans) >= 1
    found_loc = tail_loc_spans[0]
    assert full_text[found_loc.start_char : found_loc.end_char] == found_loc.text
    assert found_loc.chunk_id == chunks[-1].chunk_id

    for span in aggregated_spans:
        assert full_text[span.start_char : span.end_char] == span.text


def test_evaluation_multi_message_thread_tail_evidence_provenance() -> None:
    messages: list[dict[str, str]] = []
    for i in range(1, 10):
        messages.append(
            {
                "source_message_id": f"msg-{i:03d}",
                "text": f"Pesan pengantar nomor {i} berisi informasi umum dari warga mengenai situasi di lingkungan.",
            }
        )

    tail_msg_id = "msg-010"
    tail_evidence = "Lokasi persis lubang jalan di Jl. Affandi Gejayan depan apotek"
    messages.append({"source_message_id": tail_msg_id, "text": tail_evidence})

    chunks = chunk_messages(messages, max_tokens=448, overlap=64, case_id="CASE-THREAD-10")
    assert len(chunks) >= 1

    tail_chunk = chunks[-1]
    assert tail_msg_id in tail_chunk.source_message_ids
    assert tail_evidence in tail_chunk.text

    loc_offset = tail_chunk.text.find("Jl. Affandi Gejayan")
    assert loc_offset != -1
    global_loc = tail_chunk.to_global_char_offset(loc_offset)
    assert tail_chunk.contains_global_offset(global_loc) is True


def test_evaluation_cpu_benchmark_serving_latency_sla() -> None:
    config_intel = CpuBenchmarkConfig(
        batch_size=1,
        sequence_length=448,
        overlap_tokens=64,
        num_threads=1,
        warmup_runs=5,
        benchmark_runs=20,
        target_p95_latency_ms=5000.0,
        precision="fp32",
        device="cpu",
    )
    result_intel = run_cpu_benchmark(config_intel)

    assert isinstance(result_intel, CpuBenchmarkResult)
    assert result_intel.sla_met is True
    assert result_intel.p95_latency_ms <= 5000.0
    assert result_intel.samples_count == 20
    assert result_intel.min_latency_ms <= result_intel.p50_latency_ms <= result_intel.p95_latency_ms <= result_intel.p99_latency_ms <= result_intel.max_latency_ms
    assert result_intel.throughput_qps > 0.0

    config_ml = CpuFp32BenchmarkConfig(
        batch_size=1,
        sequence_length=448,
        overlap_tokens=64,
        num_threads=1,
        warmup_runs=5,
        benchmark_runs=20,
        target_p95_latency_ms=5000.0,
        precision="fp32",
        device="cpu",
    )
    result_ml = run_cpu_fp32_benchmark(config_ml)

    assert isinstance(result_ml, CpuFp32BenchmarkResult)
    assert result_ml.sla_met is True
    assert result_ml.p95_latency_ms <= 5000.0
    assert result_ml.throughput_qps > 0.0

    long_text = "Laporan jalan aspal rusak dan berlubang parah di perempatan lampu merah. " * 30

    def pipeline_execution() -> None:
        chunks = chunk_text(long_text, max_tokens=448, overlap=64)
        spans: list[EntitySpan] = []
        for chk in chunks:
            chunk_spans = extract_entities_from_text(chk.text)
            for s in chunk_spans:
                spans.append(
                    EntitySpan(
                        label=s.label,
                        text=s.text,
                        start_token=s.start_token + chk.start_token,
                        end_token=s.end_token + chk.start_token,
                        start_char=chk.to_global_char_offset(s.start_char),
                        end_char=chk.to_global_char_offset(s.end_char),
                        confidence=s.confidence,
                        chunk_id=chk.chunk_id,
                    )
                )
        _ = merge_overlapping_spans(spans, text=long_text)

    pipeline_bench = run_cpu_benchmark(config_intel, runner=pipeline_execution)
    assert pipeline_bench.sla_met is True
    assert pipeline_bench.p95_latency_ms <= 5000.0
    assert pipeline_bench.p95_latency_ms < 20.0


def test_evaluation_distinguish_implementation_from_untrained_artifacts() -> None:
    with pytest.raises(ValueError, match="Network access is strictly forbidden"):
        LocalMLRuntime(allow_network=True)

    runtime_no_artifacts = LocalMLRuntime(allow_network=False)
    assert runtime_no_artifacts.is_available() is False

    res_no_artifacts = runtime_no_artifacts.predict("Laporan jalan berlubang di jalan raya")
    assert isinstance(res_no_artifacts, MLRuntimeResult)
    assert res_no_artifacts.status == RuntimeStatus.UNAVAILABLE
    assert res_no_artifacts.available is False
    assert res_no_artifacts.prediction is None
    assert res_no_artifacts.entities == ()
    assert isinstance(res_no_artifacts.analysis_fallback, AnalysisResult)
    assert res_no_artifacts.analysis_fallback.category == Category.ROAD
    assert "ml_model_unavailable" in res_no_artifacts.analysis_fallback.missing_fields

    def mock_trained_backend(text: str, context: dict | None = None) -> tuple[dict, list[SpanNER]]:
        pred = {
            "intent": "COMPLAINT",
            "category": "ROAD",
            "risk": "HIGH",
            "completeness": "COMPLETE",
            "confidence": 0.94,
            "probabilities": {"ROAD": 0.94, "WASTE": 0.04},
        }
        spans = [
            SpanNER(
                label="LOCATION",
                text="Jalan Magelang KM 14",
                start=0,
                end=20,
                start_char=0,
                end_char=20,
                start_token=0,
                end_token=4,
                confidence=0.96,
            )
        ]
        return pred, spans

    runtime_with_mock = LocalMLRuntime(
        allow_network=False,
        inference_backend=mock_trained_backend,
    )
    assert runtime_with_mock.is_available() is True

    res_mock = runtime_with_mock.predict("Jalan Magelang KM 14 rusak berlubang")
    assert res_mock.status == RuntimeStatus.AVAILABLE
    assert res_mock.available is True
    assert res_mock.prediction is not None
    assert res_mock.prediction.intent == "COMPLAINT"
    assert res_mock.prediction.category == "ROAD"
    assert res_mock.prediction.confidence == 0.94
    assert len(res_mock.entities) == 1
    assert res_mock.entities[0].label == "LOCATION"
    assert res_mock.entities[0].text == "Jalan Magelang KM 14"
    assert res_mock.reason is None
    assert isinstance(res_mock.analysis_fallback, AnalysisResult)
    assert res_mock.latency_ms >= 0.0


# ---------------------------------------------------------------------------
# M3 Offline Evaluation Module Tests (scripts/evaluate_m3.py)
# ---------------------------------------------------------------------------

from scripts.evaluate_m3 import (
    PROVENANCE_HIFI_SYNTHETIC,
    PROVENANCE_SYNTHETIC_INDEPENDENT,
    VALID_SYNTHETIC_PROVENANCES,
    EvaluationConfig,
    M3EvaluationReport,
    SimpleOfflineTokenizer,
    _extract_head_logits,
    audit_independent_held_out,
    build_arg_parser,
    compute_classification_metrics,
    compute_head_class_offsets,
    evaluate_multitask_model,
    evaluate_ner_model,
    evaluate_synthetic_quality_gates,
    is_skeleton_onnx,
    load_and_validate_hifi_manifest,
    load_config_target_thresholds,
    load_independent_held_out_dataset,
    load_onnx_inference_session,
    main as evaluate_main,
    make_onnx_benchmark_runner,
    resolve_artifact_path,
    resolve_multitask_temperatures,
    resolve_quality_gate_thresholds,
    run_evaluate_m3,
)
from scripts.train_multitask import EXPECTED_HEADS, HEAD_CONFIGS
from scripts.train_ner import DEFAULT_TAGSET
from services.ml.training import HeldoutDatasetManifest, HeldoutManifestItem


def test_evaluation_config_parsing_and_defaults() -> None:
    cfg = EvaluationConfig()
    assert cfg.seed == 42
    assert cfg.num_scenarios == 36
    assert cfg.run_benchmark is True
    assert cfg.target_p95_ms == 5000.0
    assert cfg.benchmark_runs == 20
    assert cfg.num_threads == 1
    assert cfg.intra_op_num_threads is None
    assert cfg.inter_op_num_threads is None
    assert cfg.dry_run is False
    assert cfg.validate_only is False
    assert cfg.eval_split == "test"
    assert cfg.max_seq_length == 448
    assert cfg.batch_size == 16
    assert cfg.device == "cpu"
    assert cfg.local_files_only is True
    assert cfg.enforce_quality_gates is True
    assert cfg.multitask_config_path is None
    assert cfg.ner_config_path is None
    assert cfg.min_macro_f1 is None
    assert cfg.min_head_macro_f1 is None
    assert cfg.min_intent_f1 is None
    assert cfg.min_category_f1 is None
    assert cfg.min_risk_f1 is None
    assert cfg.min_completeness_f1 is None
    assert cfg.min_ner_entity_f1 is None
    assert cfg.max_ece is None

    parser = build_arg_parser()
    args = parser.parse_args([
        "--dry-run",
        "--validate-only",
        "--eval-split", "dev",
        "--max-seq-length", "256",
        "--batch-size", "8",
        "--target-p95-ms", "3500.0",
        "--num-threads", "4",
        "--intra-op-num-threads", "4",
        "--inter-op-num-threads", "1",
        "--multitask-model-path", "model.onnx",
        "--ner-model-path", "ner.onnx",
        "--held-out-dataset-path", "held_out.jsonl",
        "--no-quality-gates",
        "--min-macro-f1", "0.80",
        "--min-intent-f1", "0.88",
        "--min-category-f1", "0.82",
        "--min-risk-f1", "0.80",
        "--min-completeness-f1", "0.85",
        "--min-ner-entity-f1", "0.75",
        "--max-ece", "0.12",
        "--min-head-macro-f1", '{"risk": 0.81}',
    ])
    assert args.dry_run is True
    assert args.validate_only is True
    assert args.eval_split == "dev"
    assert args.max_seq_length == 256
    assert args.batch_size == 8
    assert args.target_p95_ms == 3500.0
    assert args.num_threads == 4
    assert args.intra_op_num_threads == 4
    assert args.inter_op_num_threads == 1
    assert args.multitask_model_path == "model.onnx"
    assert args.ner_model_path == "ner.onnx"
    assert args.held_out_dataset_path == "held_out.jsonl"
    assert args.enforce_quality_gates is False
    assert args.min_macro_f1 == 0.80
    assert args.min_intent_f1 == 0.88
    assert args.min_category_f1 == 0.82
    assert args.min_risk_f1 == 0.80
    assert args.min_completeness_f1 == 0.85
    assert args.min_ner_entity_f1 == 0.75
    assert args.max_ece == 0.12
    assert args.min_head_macro_f1 == '{"risk": 0.81}'

    with pytest.raises(Exception):
        EvaluationConfig(unsupported_extra_parameter="illegal")  # type: ignore[call-arg]


def test_evaluation_dry_run_and_no_ml_behavior(tmp_path: Path) -> None:
    eval_output = tmp_path / "eval_out.json"
    cfg = EvaluationConfig(
        output_path=str(eval_output),
        dry_run=True,
        num_scenarios=12,
        seed=101,
        benchmark_runs=10,
    )
    report = run_evaluate_m3(cfg)

    assert isinstance(report, M3EvaluationReport)
    assert report.status == "PASSED"
    assert report.all_checks_passed is True
    assert report.sla_conformance is True
    assert report.split_audit["passed"] is True
    assert report.benchmark is not None
    assert report.benchmark["runner_type"] == "synthetic"
    assert report.benchmark["sla_met"] is True
    assert report.benchmark["sequence_length"] == 448
    assert report.metrics["multitask"]["evaluated"] is False
    assert report.metrics["ner"]["evaluated"] is False
    assert report.metrics["macro_f1_multitask"] > 0.0

    assert eval_output.is_file()
    loaded = M3EvaluationReport.model_validate_json(eval_output.read_text(encoding="utf-8"))
    assert loaded.status == "PASSED"
    assert loaded.seed == 101


def test_evaluation_validate_only_mode(tmp_path: Path) -> None:
    eval_output = tmp_path / "eval_val.json"
    cfg = EvaluationConfig(
        output_path=str(eval_output),
        validate_only=True,
        num_scenarios=12,
    )
    report = run_evaluate_m3(cfg)

    assert report.status == "PASSED"
    assert report.all_checks_passed is True
    assert report.sla_conformance is True
    assert report.benchmark is None
    assert report.metrics == {"validate_only": True}
    assert report.split_audit["passed"] is True
    assert eval_output.is_file()


def test_evaluation_held_out_dataset_saving_and_loading(tmp_path: Path) -> None:
    held_out_file = tmp_path / "held_out.jsonl"
    eval_out = tmp_path / "eval_held_out.json"

    cfg = EvaluationConfig(
        num_scenarios=18,
        held_out_dataset_path=str(held_out_file),
        output_path=str(eval_out),
        dry_run=True,
    )
    report = run_evaluate_m3(cfg)

    assert report.status == "PASSED"
    assert held_out_file.is_file()
    lines = [line.strip() for line in held_out_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) > 0

    first_traj = ComplaintTrajectory.model_validate_json(lines[0])
    assert isinstance(first_traj, ComplaintTrajectory)

    # Re-evaluate on saved held-out dataset
    cfg_reload = EvaluationConfig(
        dataset_path=str(held_out_file),
        output_path=str(tmp_path / "eval_reload.json"),
        dry_run=True,
    )
    report_reload = run_evaluate_m3(cfg_reload)
    assert report_reload.status == "PASSED"
    assert report_reload.split_audit["total_trajectories"] == len(lines)


def test_compute_classification_metrics_pure_python() -> None:
    preds = [0, 1, 0, 1, 2]
    targets = [0, 1, 1, 1, 2]
    classes = ["LOW", "MEDIUM", "HIGH"]
    metrics = compute_classification_metrics(preds, targets, classes)

    assert metrics["support"] == 5
    assert metrics["accuracy"] == 0.8  # 4/5 correct
    assert 0.0 <= metrics["macro_f1"] <= 1.0
    assert "LOW" in metrics["classes"]
    assert "MEDIUM" in metrics["classes"]
    assert "HIGH" in metrics["classes"]

    # Empty targets test
    empty = compute_classification_metrics([], [], ["X"])
    assert empty["accuracy"] == 0.0
    assert empty["macro_f1"] == 0.0
    assert empty["support"] == 0


def test_evaluate_multitask_model_with_mock_onnx() -> None:
    samples = [
        {
            "text": "jalan rusak di kaliurang",
            "intent": "COMPLAINT",
            "category": "ROAD",
            "risk": "MEDIUM",
            "completeness": "SUFFICIENT",
        },
        {
            "text": "sampah liar menumpuk bau",
            "intent": "COMPLAINT",
            "category": "WASTE",
            "risk": "HIGH",
            "completeness": "SUFFICIENT",
        },
    ]

    class MockONNXSession:
        def get_inputs(self) -> list[Any]:
            m1 = mock.Mock()
            m1.name = "input_ids"
            m2 = mock.Mock()
            m2.name = "attention_mask"
            return [m1, m2]

        def get_outputs(self) -> list[Any]:
            outs = []
            for h in ("intent_logits", "category_logits", "risk_logits", "completeness_logits"):
                m = mock.Mock()
                m.name = h
                outs.append(m)
            return outs

        def run(self, output_names: Any, feed_dict: dict[str, Any]) -> list[Any]:
            n = len(feed_dict["input_ids"])
            # intent: 3 classes (0: COMPLAINT)
            intent_out = [[10.0, 0.0, 0.0] for _ in range(n)]
            # category: 6 classes (0: ROAD, 2: WASTE)
            category_out = [[10.0, 0.0, 0.0, 0.0, 0.0, 0.0] for _ in range(n)]
            # risk: 4 classes (1: MEDIUM)
            risk_out = [[0.0, 10.0, 0.0, 0.0] for _ in range(n)]
            # completeness: 3 classes (0: SUFFICIENT)
            comp_out = [[10.0, 0.0, 0.0] for _ in range(n)]
            return [intent_out, category_out, risk_out, comp_out]

    mock_sess = MockONNXSession()
    tokenizer = SimpleOfflineTokenizer()

    res = evaluate_multitask_model(
        model=mock_sess,
        samples=samples,
        tokenizer=tokenizer,
        batch_size=2,
    )

    assert res["evaluated"] is True
    assert res["sample_count"] == 2
    assert "heads" in res
    for head in EXPECTED_HEADS:
        assert head in res["heads"]
        assert "accuracy" in res["heads"][head]
        assert "macro_f1" in res["heads"][head]
        assert "classes" in res["heads"][head]

    assert res["heads"]["intent"]["accuracy"] == 1.0  # Both COMPLAINT
    assert res["heads"]["completeness"]["accuracy"] == 1.0  # Both SUFFICIENT
    assert res["macro_f1"] > 0.0
    assert res["accuracy"] > 0.0


def test_evaluate_multitask_model_with_callable_mock() -> None:
    samples = [
        {
            "text": "aduan jalan amblas",
            "intent": "COMPLAINT",
            "category": "ROAD",
            "risk": "HIGH",
            "completeness": "SUFFICIENT",
        }
    ]

    def mock_predict(text: str) -> dict[str, Any]:
        return {
            "intent": "COMPLAINT",
            "category": "ROAD",
            "risk": "HIGH",
            "completeness": "SUFFICIENT",
        }

    res = evaluate_multitask_model(
        model=mock_predict,
        samples=samples,
        tokenizer=SimpleOfflineTokenizer(),
    )
    assert res["evaluated"] is True
    assert res["heads"]["category"]["accuracy"] == 1.0
    assert res["heads"]["risk"]["accuracy"] == 1.0
    assert res["accuracy"] == 1.0


def test_evaluate_ner_model_with_mock_onnx() -> None:
    samples = [
        {
            "text": "jalan berlubang di Sleman",
            "spans": [(0, 15, "OBJ"), (19, 25, "LOC")],
        }
    ]

    class MockNERONNXSession:
        def get_inputs(self) -> list[Any]:
            m1 = mock.Mock()
            m1.name = "input_ids"
            m2 = mock.Mock()
            m2.name = "attention_mask"
            return [m1, m2]

        def run(self, output_names: Any, feed_dict: dict[str, Any]) -> list[Any]:
            seq_len = len(feed_dict["input_ids"][0]) if isinstance(feed_dict["input_ids"], list) else feed_dict["input_ids"].shape[1]
            num_tags = len(DEFAULT_TAGSET)
            logits_token = [0.0] * num_tags
            logits_token[0] = 10.0  # Default to 'O'
            return [[[list(logits_token) for _ in range(seq_len)]]]

    mock_sess = MockNERONNXSession()
    tokenizer = SimpleOfflineTokenizer()

    res = evaluate_ner_model(
        model=mock_sess,
        samples=samples,
        tokenizer=tokenizer,
        tagset=DEFAULT_TAGSET,
    )

    assert res["evaluated"] is True
    assert "token_accuracy" in res
    assert "token_macro_f1" in res
    assert "entity_precision" in res
    assert "entity_recall" in res
    assert "entity_f1" in res
    assert res["total_tokens_evaluated"] > 0


def test_cpu_benchmark_runner_on_onnx_at_seq_448() -> None:
    mock_session = mock.MagicMock()
    mock_in1 = mock.Mock()
    mock_in1.name = "input_ids"
    mock_in2 = mock.Mock()
    mock_in2.name = "attention_mask"
    mock_session.get_inputs.return_value = [mock_in1, mock_in2]
    mock_session.run.return_value = [[[0.0]]]

    runner = make_onnx_benchmark_runner(mock_session, seq_len=448)
    assert callable(runner)

    bench_cfg = CpuFp32BenchmarkConfig(
        warmup_runs=3,
        benchmark_runs=5,
        sequence_length=448,
        overlap_tokens=64,
        target_p95_latency_ms=5000.0,
    )
    result = run_cpu_fp32_benchmark(bench_cfg, runner=runner)

    assert result.sla_met is True
    assert mock_session.run.call_count == 8  # 3 warmup + 5 benchmark runs
    call_args = mock_session.run.call_args[0]
    feed_dict = call_args[1]
    assert "input_ids" in feed_dict
    assert "attention_mask" in feed_dict
    assert len(feed_dict["input_ids"][0]) == 448 if isinstance(feed_dict["input_ids"], list) else feed_dict["input_ids"].shape == (1, 448)


def test_skeleton_onnx_detection_and_resolution(tmp_path: Path) -> None:
    skel_file = tmp_path / "skeleton_model.onnx"
    skel_file.write_bytes(b"\x08\x07\x12\nKAWAL_ONNX" + b"\x00" * 32)
    assert is_skeleton_onnx(skel_file) is True
    assert load_onnx_inference_session(skel_file) is None

    non_skel = tmp_path / "valid_size_model.onnx"
    non_skel.write_bytes(b"PROTOBUF_SAMPLE_NON_SKELETON_DATA_LONG_ENOUGH_FOR_TESTING" * 3)
    assert is_skeleton_onnx(non_skel) is False

    resolved = resolve_artifact_path(
        explicit_path=str(skel_file),
        manifest=None,
        artifacts_dir=None,
        task="multitask",
    )
    assert resolved == skel_file.resolve()


def test_manifest_artifact_sha_tamper_detection(tmp_path: Path) -> None:
    weight_file = tmp_path / "model.bin"
    weight_file.write_bytes(b"MOCK_WEIGHTS_CONTENT")

    manifest_item = ArtifactManifestItem(
        name="indobert-multitask",
        version="v1.0.0",
        sha256="0" * 64,
        path="model.bin",
        size_bytes=len(b"MOCK_WEIGHTS_CONTENT"),
        task="multitask",
        precision="fp32",
    )
    manifest = ArtifactManifest(artifacts={"indobert-multitask": manifest_item})
    m_path = tmp_path / "manifest.json"
    m_path.write_text(manifest.model_dump_json(), encoding="utf-8")

    cfg = EvaluationConfig(
        manifest_path=str(m_path),
        output_path=str(tmp_path / "eval_corrupt.json"),
        dry_run=True,
        num_scenarios=12,
    )
    report = run_evaluate_m3(cfg)

    assert report.status == "FAILED"
    assert report.all_checks_passed is False
    assert report.manifest_audit is not None
    assert report.manifest_audit["all_valid"] is False
    assert report.manifest_audit["items_checked"]["indobert-multitask"] is False


def test_evaluate_m3_full_offline_pipeline_with_mock_onnx_artifacts(tmp_path: Path) -> None:
    multitask_onnx = tmp_path / "multitask.onnx"
    multitask_onnx.write_bytes(b"PROTOBUF_SAMPLE_NON_SKELETON_DATA_LONG_ENOUGH_FOR_TESTING" * 3)

    ner_onnx = tmp_path / "ner.onnx"
    ner_onnx.write_bytes(b"PROTOBUF_SAMPLE_NON_SKELETON_DATA_LONG_ENOUGH_FOR_TESTING" * 3)

    eval_out = tmp_path / "full_eval.json"

    class MockONNXRuntimeSession:
        def __init__(self, task: str) -> None:
            self.task = task

        def get_inputs(self) -> list[Any]:
            m1 = mock.Mock()
            m1.name = "input_ids"
            m2 = mock.Mock()
            m2.name = "attention_mask"
            return [m1, m2]

        def get_outputs(self) -> list[Any]:
            if self.task == "multitask":
                outs = []
                for h in ("intent_logits", "category_logits", "risk_logits", "completeness_logits"):
                    m = mock.Mock()
                    m.name = h
                    outs.append(m)
                return outs
            else:
                m = mock.Mock()
                m.name = "logits"
                return [m]

        def run(self, output_names: Any, feed_dict: dict[str, Any]) -> list[Any]:
            n = len(feed_dict["input_ids"])
            seq_len = len(feed_dict["input_ids"][0])
            if self.task == "multitask":
                intent_out = [[10.0, 0.0, 0.0] for _ in range(n)]
                cat_out = [[10.0, 0.0, 0.0, 0.0, 0.0, 0.0] for _ in range(n)]
                risk_out = [[0.0, 10.0, 0.0, 0.0] for _ in range(n)]
                comp_out = [[10.0, 0.0, 0.0] for _ in range(n)]
                return [intent_out, cat_out, risk_out, comp_out]
            else:
                tag_len = len(DEFAULT_TAGSET)
                step = [0.0] * tag_len
                step[0] = 10.0
                return [[[list(step) for _ in range(seq_len)] for _ in range(n)]]

    def mock_load_session(p: Path | str, *args: Any, **kwargs: Any) -> Any:
        p_str = str(p)
        if "multitask" in p_str:
            return MockONNXRuntimeSession("multitask")
        elif "ner" in p_str:
            return MockONNXRuntimeSession("ner")
        return None

    with mock.patch("scripts.evaluate_m3.load_onnx_inference_session", side_effect=mock_load_session):
        cfg = EvaluationConfig(
            multitask_model_path=str(multitask_onnx),
            ner_model_path=str(ner_onnx),
            output_path=str(eval_out),
            num_scenarios=12,
            benchmark_runs=5,
        )
        report = run_evaluate_m3(cfg)

    # Distinct status: pipeline integrity PASSED, but synthetic-quality gates FAILED due to poor mock risk/NER/ECE
    assert report.status == "INTEGRITY_PASSED_QUALITY_FAILED"
    assert report.pipeline_status == "PASSED"
    assert report.quality_status == "FAILED"
    assert report.all_checks_passed is False
    assert report.sla_conformance is True
    assert report.quality_gates["passed"] is False
    assert len(report.quality_gates["failed_gates"]) > 0
    failed_gate_str = " ".join(report.quality_gates["failed_gates"])
    assert "risk_macro_f1" in failed_gate_str
    assert "ner_entity_f1" in failed_gate_str
    assert "ece" in failed_gate_str

    assert report.benchmark is not None
    assert report.benchmark["runner_type"] == "onnx"
    assert report.benchmark["sequence_length"] == 448
    assert report.metrics["multitask"]["evaluated"] is True
    assert report.metrics["ner"]["evaluated"] is True
    assert "heads" in report.metrics["multitask"]
    for head in EXPECTED_HEADS:
        assert head in report.metrics["multitask"]["heads"]
        assert "accuracy" in report.metrics["multitask"]["heads"][head]
        assert "macro_f1" in report.metrics["multitask"]["heads"][head]
    assert "entity_f1" in report.metrics["ner"]
    assert "token_accuracy" in report.metrics["ner"]
    assert eval_out.is_file()

    # When quality gates are not enforced, pipeline status propagates to overall status
    with mock.patch("scripts.evaluate_m3.load_onnx_inference_session", side_effect=mock_load_session):
        cfg_no_enforce = EvaluationConfig(
            multitask_model_path=str(multitask_onnx),
            ner_model_path=str(ner_onnx),
            output_path=str(tmp_path / "eval_no_enforce.json"),
            num_scenarios=12,
            benchmark_runs=5,
            enforce_quality_gates=False,
        )
        report_no_enforce = run_evaluate_m3(cfg_no_enforce)

    assert report_no_enforce.status == "PASSED"
    assert report_no_enforce.all_checks_passed is True
    assert report_no_enforce.pipeline_status == "PASSED"
    assert report_no_enforce.quality_status == "FAILED"


def test_evaluate_m3_cli_main_execution(tmp_path: Path) -> None:
    out_path = tmp_path / "cli_report.json"
    exit_code = evaluate_main([
        "--dry-run",
        "--num-scenarios", "12",
        "--output-path", str(out_path),
    ])
    assert exit_code == 0
    assert out_path.is_file()

    val_out = tmp_path / "val_report.json"
    exit_val = evaluate_main([
        "--validate-only",
        "--num-scenarios", "12",
        "--output-path", str(val_out),
    ])
    assert exit_val == 0
    assert val_out.is_file()

    exit_err = evaluate_main(["--dataset-path", "/nonexistent/invalid/dataset.jsonl"])
    assert exit_err == 1


def test_onnx_cpu_session_threading_configuration(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    model_file = tmp_path / "model.onnx"
    model_file.write_bytes(b"PROTOBUF_SAMPLE_NON_SKELETON_DATA_LONG_ENOUGH_FOR_TESTING" * 3)

    mock_ort = mock.MagicMock()
    mock_opts = mock.MagicMock()
    mock_ort.SessionOptions.return_value = mock_opts
    mock_ort.ExecutionMode.ORT_SEQUENTIAL = "SEQUENTIAL"
    mock_ort.GraphOptimizationLevel.ORT_ENABLE_ALL = "ALL"
    mock_sess = mock.MagicMock()
    mock_ort.InferenceSession.return_value = mock_sess

    monkeypatch.setitem(sys.modules, "onnxruntime", mock_ort)

    session = load_onnx_inference_session(
        model_file,
        intra_op_num_threads=4,
        inter_op_num_threads=2,
    )
    assert session is mock_sess
    assert mock_opts.intra_op_num_threads == 4
    assert mock_opts.inter_op_num_threads == 2
    assert mock_opts.execution_mode == "SEQUENTIAL"
    assert mock_opts.graph_optimization_level == "ALL"
    mock_ort.InferenceSession.assert_called_once_with(
        str(model_file.resolve()),
        mock_opts,
        providers=["CPUExecutionProvider"],
    )


def test_onnx_cpu_threading_accuracy_preservation_and_reproducibility() -> None:
    samples = [
        {
            "text": "laporan jalan aspal rusak parah",
            "intent": "COMPLAINT",
            "category": "ROAD",
            "risk": "HIGH",
            "completeness": "SUFFICIENT",
        },
        {
            "text": "tumpukan sampah liar bau sekali",
            "intent": "COMPLAINT",
            "category": "WASTE",
            "risk": "HIGH",
            "completeness": "SUFFICIENT",
        },
    ]

    def mock_predict(text: str) -> dict[str, Any]:
        return {
            "intent": "COMPLAINT",
            "category": "ROAD" if "jalan" in text else "WASTE",
            "risk": "HIGH",
            "completeness": "SUFFICIENT",
        }

    tok = SimpleOfflineTokenizer()
    run1 = evaluate_multitask_model(model=mock_predict, samples=samples, tokenizer=tok)
    run2 = evaluate_multitask_model(model=mock_predict, samples=samples, tokenizer=tok)

    assert run1["evaluated"] is True
    assert run1["accuracy"] == run2["accuracy"]
    assert run1["macro_f1"] == run2["macro_f1"]
    assert run1["heads"]["category"] == run2["heads"]["category"]


def test_evaluation_benchmark_threading_and_target_latency_propagation(tmp_path: Path) -> None:
    out_file = tmp_path / "bench_report.json"
    cfg = EvaluationConfig(
        output_path=str(out_file),
        dry_run=True,
        num_scenarios=12,
        seed=42,
        benchmark_runs=5,
        num_threads=4,
        intra_op_num_threads=4,
        inter_op_num_threads=1,
    )
    report = run_evaluate_m3(cfg)

    assert report.status == "PASSED"
    assert report.sla_conformance is True
    assert report.benchmark is not None
    assert report.benchmark["target_p95_ms"] == 5000.0
    assert report.benchmark["num_threads"] == 4
    assert report.benchmark["sla_met"] is True


def test_simple_offline_tokenizer_offsets_and_word_ids() -> None:
    tok = SimpleOfflineTokenizer()
    text = "Lapor jalan berlubang di Sleman"
    enc = tok(text, max_length=16, return_offsets_mapping=True)

    assert "offset_mapping" in enc
    assert hasattr(enc, "word_ids")
    word_ids = enc.word_ids(0)
    offsets = enc["offset_mapping"][0]

    # [CLS], "Lapor", "jalan", "berlubang", "di", "Sleman", [SEP], then pads
    assert word_ids[0] is None
    assert offsets[0] == (0, 0)

    assert word_ids[1] == 0
    assert offsets[1] == (0, 5)  # "Lapor"

    assert word_ids[2] == 1
    assert offsets[2] == (6, 11)  # "jalan"

    assert word_ids[3] == 2
    assert offsets[3] == (12, 21)  # "berlubang"

    assert word_ids[4] == 3
    assert offsets[4] == (22, 24)  # "di"

    assert word_ids[5] == 4
    assert offsets[5] == (25, 31)  # "Sleman"

    assert word_ids[6] is None
    assert offsets[6] == (0, 0)  # [SEP]


def test_evaluate_ner_model_with_subwords_and_tokenizer_offsets() -> None:
    # Text with punctuation and a multi-subword token:
    # "Lapor! jalan berlubang di kelurahan Sukamaju barat"
    # Entities: (7, 22, "OBJ") -> "jalan berlubang"
    #           (26, 44, "LOC") -> "kelurahan Sukamaju"
    text = "Lapor! jalan berlubang di kelurahan Sukamaju barat"
    samples = [
        {
            "text": text,
            "spans": [(7, 22, "OBJ"), (26, 44, "LOC")],
        }
    ]

    # Subwords:
    # [CLS] (0, 0)
    # "lapor" (0, 5)
    # "!" (5, 6)
    # "jalan" (7, 12)
    # "berlubang" (13, 22)
    # "di" (23, 25)
    # "kelurahan" (26, 35)
    # "sukam" (36, 41)
    # "##aju" (41, 44)
    # "barat" (45, 50)
    # [SEP] (0, 0)
    subword_offsets = [
        (0, 0),
        (0, 5),
        (5, 6),
        (7, 12),
        (13, 22),
        (23, 25),
        (26, 35),
        (36, 41),
        (41, 44),
        (45, 50),
        (0, 0),
    ]
    subword_word_ids = [None, 0, 1, 2, 3, 4, 5, 6, 6, 7, None]
    subword_tags = [
        "O",       # [CLS]
        "O",       # "lapor"
        "O",       # "!"
        "B-OBJ",   # "jalan"
        "I-OBJ",   # "berlubang"
        "O",       # "di"
        "B-LOC",   # "kelurahan"
        "I-LOC",   # "sukam"
        "I-LOC",   # "##aju"
        "O",       # "barat"
        "O",       # [SEP]
    ]
    tag2id = {t: i for i, t in enumerate(DEFAULT_TAGSET)}
    target_tag_ids = [tag2id[t] for t in subword_tags]

    class MockSubwordTokenizer:
        def __call__(self, txt: str, **kwargs: Any) -> Any:
            seq_len = len(subword_offsets)
            return {
                "input_ids": [[101] * seq_len],
                "attention_mask": [[1] * seq_len],
                "offset_mapping": [subword_offsets],
            }

        def word_ids(self, batch_index: int = 0) -> list[int | None]:
            return subword_word_ids

    class MockAccurateNERONNXSession:
        def get_inputs(self) -> list[Any]:
            m1 = mock.Mock()
            m1.name = "input_ids"
            m2 = mock.Mock()
            m2.name = "attention_mask"
            return [m1, m2]

        def run(self, output_names: Any, feed_dict: dict[str, Any]) -> list[Any]:
            seq_len = len(subword_offsets)
            num_tags = len(DEFAULT_TAGSET)
            logits_seq = []
            for t_id in target_tag_ids:
                step = [0.0] * num_tags
                step[t_id] = 10.0
                logits_seq.append(step)
            return [[logits_seq]]

    res = evaluate_ner_model(
        model=MockAccurateNERONNXSession(),
        samples=samples,
        tokenizer=MockSubwordTokenizer(),
        tagset=DEFAULT_TAGSET,
    )

    assert res["evaluated"] is True
    # Honest perfect metrics when alignment is correct
    assert res["entity_precision"] == 1.0
    assert res["entity_recall"] == 1.0
    assert res["entity_f1"] == 1.0
    assert res["token_accuracy"] == 1.0
    assert res["total_entities_true"] == 2
    assert res["total_entities_pred"] == 2
    assert res["per_type_metrics"]["OBJ"]["f1"] == 1.0
    assert res["per_type_metrics"]["LOC"]["f1"] == 1.0


def test_evaluate_ner_model_honest_metrics_on_partial_predictions() -> None:
    text = "Lapor jalan berlubang di Sleman besok pagi"
    # True spans: OBJ ("jalan berlubang": 6..21), LOC ("Sleman": 25..31)
    samples = [
        {
            "text": text,
            "spans": [(6, 21, "OBJ"), (25, 31, "LOC")],
        }
    ]

    # Predict:
    # OBJ correctly: "jalan berlubang" -> TP = 1
    # Miss LOC: "Sleman" as 'O' -> FN = 1
    # False positive TIME on "besok": -> FP = 1
    # Words in text: ['Lapor', 'jalan', 'berlubang', 'di', 'Sleman', 'besok', 'pagi']
    # Offsets: [(0, 5), (6, 11), (12, 21), (22, 24), (25, 31), (32, 37), (38, 42)]
    tok_offsets = [
        (0, 0),        # [CLS]
        (0, 5),        # "Lapor" -> O
        (6, 11),       # "jalan" -> B-OBJ
        (12, 21),      # "berlubang" -> I-OBJ
        (22, 24),      # "di" -> O
        (25, 31),      # "Sleman" -> O (missed!)
        (32, 37),      # "besok" -> B-TIME (FP!)
        (38, 42),      # "pagi" -> O
        (0, 0),        # [SEP]
    ]
    pred_tags = ["O", "O", "B-OBJ", "I-OBJ", "O", "O", "B-TIME", "O", "O"]
    tag2id = {t: i for i, t in enumerate(DEFAULT_TAGSET)}
    target_tag_ids = [tag2id[t] for t in pred_tags]

    class MockTok:
        def __call__(self, txt: str, **kwargs: Any) -> Any:
            n = len(tok_offsets)
            return {
                "input_ids": [[101] * n],
                "attention_mask": [[1] * n],
                "offset_mapping": [tok_offsets],
            }

        def word_ids(self, batch_index: int = 0) -> list[int | None]:
            return [None, 0, 1, 2, 3, 4, 5, 6, None]

    class MockONNX:
        def get_inputs(self) -> list[Any]:
            return []

        def run(self, output_names: Any, feed_dict: dict[str, Any]) -> list[Any]:
            num_tags = len(DEFAULT_TAGSET)
            logits_seq = []
            for t_id in target_tag_ids:
                step = [0.0] * num_tags
                step[t_id] = 10.0
                logits_seq.append(step)
            return [[logits_seq]]

    res = evaluate_ner_model(
        model=MockONNX(),
        samples=samples,
        tokenizer=MockTok(),
        tagset=DEFAULT_TAGSET,
    )

    assert res["evaluated"] is True
    # TP = 1 (OBJ), FP = 1 (TIME), FN = 1 (LOC)
    assert res["total_entities_true"] == 2
    assert res["total_entities_pred"] == 2
    # Precision = 1 / 2 = 0.5, Recall = 1 / 2 = 0.5, F1 = 0.5
    assert res["entity_precision"] == 0.5
    assert res["entity_recall"] == 0.5
    assert res["entity_f1"] == 0.5
    assert res["per_type_metrics"]["OBJ"]["f1"] == 1.0
    assert res["per_type_metrics"]["LOC"]["f1"] == 0.0
    assert res["per_type_metrics"]["TIME"]["f1"] == 0.0


def test_evaluate_ner_model_real_tokenizer_if_available() -> None:
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("indobenchmark/indobert-base-p1", local_files_only=True)
    except Exception:
        pytest.skip("Transformers or local indobert-base-p1 not available")

    from scripts.train_ner import align_spans_with_tokenizer

    tag2id = {t: i for i, t in enumerate(DEFAULT_TAGSET)}
    sample = {
        "text": "Lapor! kebakaran di kelurahan Sukamaju RT05",
        "spans": [(7, 16, "OBJ"), (20, 38, "LOC")],
    }
    enc = align_spans_with_tokenizer(tok, sample["text"], sample["spans"], tag2id, max_seq_length=448)
    labels = enc["labels"]

    class RealOracleONNX:
        def get_inputs(self) -> list[Any]:
            m1 = mock.Mock()
            m1.name = "input_ids"
            m2 = mock.Mock()
            m2.name = "attention_mask"
            return [m1, m2]

        def run(self, output_names: Any, feed_dict: dict[str, Any]) -> list[Any]:
            seq_len = len(feed_dict["input_ids"][0])
            logits = [[0.0] * len(DEFAULT_TAGSET) for _ in range(seq_len)]
            for i, lbl in enumerate(labels[:seq_len]):
                if lbl != -100:
                    logits[i][lbl] = 10.0
                else:
                    logits[i][0] = 10.0
            return [[logits]]

    res = evaluate_ner_model(RealOracleONNX(), [sample], tok)
    assert res["evaluated"] is True
    assert res["token_accuracy"] == 1.0
    assert res["entity_precision"] == 1.0
    assert res["entity_recall"] == 1.0
    assert res["entity_f1"] == 1.0
    assert res["total_entities_true"] == 2
    assert res["total_entities_pred"] == 2


def test_load_config_target_thresholds_defaults() -> None:
    thresholds = load_config_target_thresholds()
    assert thresholds["intent_f1"] == 0.9
    assert thresholds["category_macro_f1"] == 0.85
    assert thresholds["risk_macro_f1"] == 0.85
    assert thresholds["completeness_macro_f1"] == 0.85
    assert thresholds["macro_f1"] == 0.85
    assert thresholds["ner_f1"] == 0.8
    assert thresholds["ner_entity_f1"] == 0.8
    assert thresholds["ece"] == 0.08
    assert thresholds["latency_p95_ms"] == 5000.0


def test_resolve_quality_gate_thresholds_hierarchy() -> None:
    # 1. Default hierarchy without overrides
    cfg_default = EvaluationConfig()
    resolved_default = resolve_quality_gate_thresholds(cfg_default)
    assert resolved_default["intent_macro_f1"] == 0.9
    assert resolved_default["category_macro_f1"] == 0.85
    assert resolved_default["risk_macro_f1"] == 0.85
    assert resolved_default["completeness_macro_f1"] == 0.85
    assert resolved_default["ner_entity_f1"] == 0.8
    assert resolved_default["ece"] == 0.08

    # 2. Per-field and dictionary overrides take precedence
    cfg_custom = EvaluationConfig(
        min_intent_f1=0.95,
        min_risk_f1=0.75,
        min_head_macro_f1={"category": 0.80, "completeness": 0.82},
        min_ner_entity_f1=0.70,
        max_ece=0.05,
    )
    resolved_custom = resolve_quality_gate_thresholds(cfg_custom)
    assert resolved_custom["intent_macro_f1"] == 0.95
    assert resolved_custom["risk_macro_f1"] == 0.75
    assert resolved_custom["category_macro_f1"] == 0.80
    assert resolved_custom["completeness_macro_f1"] == 0.82
    assert resolved_custom["ner_entity_f1"] == 0.70
    assert resolved_custom["ece"] == 0.05


def test_evaluate_synthetic_quality_gates_logic() -> None:
    thresholds = {
        "intent_macro_f1": 0.90,
        "category_macro_f1": 0.85,
        "risk_macro_f1": 0.85,
        "completeness_macro_f1": 0.85,
        "macro_f1": 0.85,
        "ner_entity_f1": 0.80,
        "ece": 0.08,
    }

    # Case A: All synthetic quality gates pass
    passing_metrics = {
        "macro_f1_multitask": 0.91,
        "ece": 0.04,
        "multitask": {
            "macro_f1": 0.91,
            "heads": {
                "intent": {"macro_f1": 0.94},
                "category": {"macro_f1": 0.89},
                "risk": {"macro_f1": 0.88},
                "completeness": {"macro_f1": 0.93},
            },
        },
        "ner": {
            "entity_f1": 0.86,
        },
    }
    res_pass = evaluate_synthetic_quality_gates(passing_metrics, thresholds)
    assert res_pass["passed"] is True
    assert res_pass["status"] == "PASSED"
    assert len(res_pass["failed_gates"]) == 0
    assert len(res_pass["passed_gates"]) == 7

    # Case B: Quality gates fail on risk, NER, and ECE
    failing_metrics = {
        "macro_f1_multitask": 0.60,
        "ece": 0.1976,
        "multitask": {
            "macro_f1": 0.60,
            "heads": {
                "intent": {"macro_f1": 0.92},
                "category": {"macro_f1": 0.86},
                "risk": {"macro_f1": 0.20},  # Fails!
                "completeness": {"macro_f1": 0.88},
            },
        },
        "ner": {
            "entity_f1": 0.35,  # Fails!
        },
    }
    res_fail = evaluate_synthetic_quality_gates(failing_metrics, thresholds)
    assert res_fail["passed"] is False
    assert res_fail["status"] == "FAILED"
    assert len(res_fail["failed_gates"]) >= 3
    failed_str = " ".join(res_fail["failed_gates"])
    assert "risk_macro_f1" in failed_str
    assert "ner_entity_f1" in failed_str
    assert "ece" in failed_str
    assert "macro_f1_multitask" in failed_str

    # Transparent gate details schema
    assert "gate_details" in res_fail
    assert res_fail["gate_details"]["risk_macro_f1"]["passed"] is False
    assert res_fail["gate_details"]["risk_macro_f1"]["value"] == 0.2
    assert res_fail["gate_details"]["risk_macro_f1"]["threshold"] == 0.85
    assert res_fail["gate_details"]["ece"]["passed"] is False
    assert res_fail["gate_details"]["ece"]["operator"] == "<="


def test_distinct_status_separation_four_quadrants(tmp_path: Path) -> None:
    # 1. Pipeline passed, Quality passed -> PASSED
    rep_both_pass = M3EvaluationReport(
        seed=42,
        status="PASSED",
        pipeline_status="PASSED",
        quality_status="PASSED",
        quality_gates={"passed": True, "failed_gates": []},
        split_audit={"passed": True},
        metrics={},
        sla_conformance=True,
        all_checks_passed=True,
    )
    assert rep_both_pass.status == "PASSED"
    assert rep_both_pass.pipeline_status == "PASSED"
    assert rep_both_pass.quality_status == "PASSED"
    assert rep_both_pass.all_checks_passed is True

    # 2. Pipeline passed, Quality failed -> INTEGRITY_PASSED_QUALITY_FAILED
    rep_int_pass_qual_fail = M3EvaluationReport(
        seed=42,
        status="INTEGRITY_PASSED_QUALITY_FAILED",
        pipeline_status="PASSED",
        quality_status="FAILED",
        quality_gates={"passed": False, "failed_gates": ["risk_macro_f1: 0.1 < 0.85"]},
        split_audit={"passed": True},
        metrics={},
        sla_conformance=True,
        all_checks_passed=False,
    )
    assert rep_int_pass_qual_fail.status == "INTEGRITY_PASSED_QUALITY_FAILED"
    assert rep_int_pass_qual_fail.pipeline_status == "PASSED"
    assert rep_int_pass_qual_fail.quality_status == "FAILED"
    assert rep_int_pass_qual_fail.all_checks_passed is False

    # 3. Pipeline failed, Quality passed -> FAILED
    rep_int_fail_qual_pass = M3EvaluationReport(
        seed=42,
        status="FAILED",
        pipeline_status="FAILED",
        quality_status="PASSED",
        quality_gates={"passed": True, "failed_gates": []},
        split_audit={"passed": False},
        metrics={},
        sla_conformance=False,
        all_checks_passed=False,
    )
    assert rep_int_fail_qual_pass.status == "FAILED"
    assert rep_int_fail_qual_pass.pipeline_status == "FAILED"
    assert rep_int_fail_qual_pass.quality_status == "PASSED"
    assert rep_int_fail_qual_pass.all_checks_passed is False

    # 4. Pipeline failed, Quality failed -> FAILED
    rep_both_fail = M3EvaluationReport(
        seed=42,
        status="FAILED",
        pipeline_status="FAILED",
        quality_status="FAILED",
        quality_gates={"passed": False, "failed_gates": ["ece > 0.08"]},
        split_audit={"passed": False},
        metrics={},
        sla_conformance=False,
        all_checks_passed=False,
    )
    assert rep_both_fail.status == "FAILED"
    assert rep_both_fail.pipeline_status == "FAILED"
    assert rep_both_fail.quality_status == "FAILED"
    assert rep_both_fail.all_checks_passed is False


def test_evaluate_m3_with_configurable_thresholds_passing(tmp_path: Path) -> None:
    multitask_onnx = tmp_path / "mt.onnx"
    multitask_onnx.write_bytes(b"PROTOBUF_DATA_FOR_TESTING_1234567890" * 3)
    ner_onnx = tmp_path / "ner.onnx"
    ner_onnx.write_bytes(b"PROTOBUF_DATA_FOR_TESTING_1234567890" * 3)

    class MockSession:
        def __init__(self, task: str) -> None:
            self.task = task
        def get_inputs(self) -> list[Any]:
            return [mock.Mock(name="input_ids"), mock.Mock(name="attention_mask")]
        def get_outputs(self) -> list[Any]:
            if self.task == "multitask":
                outs = []
                for h in ("intent_logits", "category_logits", "risk_logits", "completeness_logits"):
                    m = mock.Mock()
                    m.name = h
                    outs.append(m)
                return outs
            m = mock.Mock()
            m.name = "logits"
            return [m]
        def run(self, output_names: Any, feed_dict: dict[str, Any]) -> list[Any]:
            n = len(feed_dict["input_ids"])
            seq_len = len(feed_dict["input_ids"][0])
            if self.task == "multitask":
                return [
                    [[10.0, 0.0, 0.0] for _ in range(n)],
                    [[10.0, 0.0, 0.0, 0.0, 0.0, 0.0] for _ in range(n)],
                    [[0.0, 10.0, 0.0, 0.0] for _ in range(n)],
                    [[10.0, 0.0, 0.0] for _ in range(n)],
                ]
            return [[[ [10.0] + [0.0] * (len(DEFAULT_TAGSET) - 1) for _ in range(seq_len)] for _ in range(n)]]

    def mock_load(p: Path | str, *args: Any, **kwargs: Any) -> Any:
        return MockSession("multitask") if "mt.onnx" in str(p) else MockSession("ner")

    out_file = tmp_path / "configured_eval.json"
    with mock.patch("scripts.evaluate_m3.load_onnx_inference_session", side_effect=mock_load):
        # Configure relaxed thresholds matching this synthetic baseline
        cfg = EvaluationConfig(
            multitask_model_path=str(multitask_onnx),
            ner_model_path=str(ner_onnx),
            output_path=str(out_file),
            num_scenarios=12,
            benchmark_runs=3,
            min_macro_f1=0.05,
            min_intent_f1=0.20,
            min_category_f1=0.0,
            min_risk_f1=0.0,
            min_completeness_f1=0.10,
            min_ner_entity_f1=0.0,
            max_ece=1.0,
        )
        report = run_evaluate_m3(cfg)

    assert report.status == "PASSED"
    assert report.pipeline_status == "PASSED"
    assert report.quality_status == "PASSED"
    assert report.all_checks_passed is True
    assert report.quality_gates["passed"] is True
    assert len(report.quality_gates["failed_gates"]) == 0
    assert out_file.is_file()


def test_evaluate_m3_cli_with_quality_gate_options(tmp_path: Path) -> None:
    # 1. CLI with dry-run passes by default
    cli_out1 = tmp_path / "cli_out1.json"
    code1 = evaluate_main([
        "--dry-run",
        "--num-scenarios", "12",
        "--output-path", str(cli_out1),
    ])
    assert code1 == 0
    assert cli_out1.is_file()

    # 2. CLI with impossible threshold fails with exit code 1
    cli_out2 = tmp_path / "cli_out2.json"
    code2 = evaluate_main([
        "--dry-run",
        "--num-scenarios", "12",
        "--output-path", str(cli_out2),
        "--min-risk-f1", "0.999",
    ])
    assert code2 == 1
    assert cli_out2.is_file()
    rep2 = M3EvaluationReport.model_validate_json(cli_out2.read_text(encoding="utf-8"))
    assert rep2.status == "INTEGRITY_PASSED_QUALITY_FAILED"
    assert rep2.pipeline_status == "PASSED"
    assert rep2.quality_status == "FAILED"
    assert rep2.all_checks_passed is False
    assert any("risk_macro_f1" in g for g in rep2.quality_gates["failed_gates"])

    # 3. CLI with impossible threshold but --no-quality-gates exits 0
    cli_out3 = tmp_path / "cli_out3.json"
    code3 = evaluate_main([
        "--dry-run",
        "--num-scenarios", "12",
        "--output-path", str(cli_out3),
        "--min-risk-f1", "0.999",
        "--no-quality-gates",
    ])
    assert code3 == 0
    assert cli_out3.is_file()
    rep3 = M3EvaluationReport.model_validate_json(cli_out3.read_text(encoding="utf-8"))
    assert rep3.status == "PASSED"
    assert rep3.all_checks_passed is True
    assert rep3.pipeline_status == "PASSED"
    assert rep3.quality_status == "FAILED"


def test_resolve_multitask_temperatures_discovery_hierarchy(tmp_path: Path) -> None:
    """Verify artifact-specific temperature discovery across config, paths, and manifest."""
    # 1. Explicit temperatures dictionary
    temps1, src1 = resolve_multitask_temperatures(explicit_temperatures={"intent": 2.1, "category": 1.4})
    assert temps1 == {"intent": 2.1, "category": 1.4}
    assert src1 is None

    # 2. Explicit temperature file path
    multitask_dir = tmp_path / "artifacts" / "multitask"
    multitask_dir.mkdir(parents=True)
    temp_file = multitask_dir / "temperatures.json"
    save_temperatures({"intent": 2.2, "category": 1.5, "risk": 1.8, "completeness": 2.3}, temp_file)

    temps2, src2 = resolve_multitask_temperatures(explicit_path=temp_file)
    assert temps2["intent"] == 2.2
    assert src2 == temp_file

    # 3. Explicit directory path containing temperatures.json
    temps3, src3 = resolve_multitask_temperatures(explicit_path=multitask_dir)
    assert temps3["category"] == 1.5
    assert src3 == temp_file

    # 4. Multitask model parent directory discovery
    model_file = multitask_dir / "model.onnx"
    model_file.write_bytes(b"dummy_onnx")
    temps4, src4 = resolve_multitask_temperatures(multitask_model_path=model_file)
    assert temps4["risk"] == 1.8
    assert src4 == temp_file

    # 5. Tokenizer parent directory discovery
    tok_dir = tmp_path / "tokenizer"
    tok_dir.mkdir(parents=True)
    tok_temp = tok_dir / "temperatures.json"
    save_temperatures({"intent": 1.9, "category": 1.3, "risk": 1.7, "completeness": 2.1}, tok_temp)
    temps5, src5 = resolve_multitask_temperatures(tokenizer_path=tok_dir)
    assert temps5["intent"] == 1.9
    assert src5 == tok_temp

    # 6. Manifest calibration artifact item discovery
    from services.ml.manifest import ArtifactManifestItem
    manifest_item = ArtifactManifestItem(
        name="multitask-temperatures",
        version="v1.0.0",
        task="calibration",
        path="multitask/temperatures.json",
        sha256="0" * 64,
        size_bytes=100,
        precision="fp32",
    )
    manifest = ArtifactManifest(
        artifacts={"multitask-temperatures": manifest_item},
    )
    temps6, src6 = resolve_multitask_temperatures(
        manifest=manifest,
        artifacts_dir=tmp_path / "artifacts",
    )
    assert temps6["category"] == 1.5
    assert src6 == temp_file

    # 7. No artifact available returns None, None without failing or fitting test labels
    temps7, src7 = resolve_multitask_temperatures(
        multitask_model_path=tmp_path / "empty_dir" / "model.onnx"
    )
    assert temps7 is None
    assert src7 is None


def test_evaluate_multitask_model_applies_per_head_temperatures_and_reports_per_head_ece() -> None:
    """Verify evaluator applies per-head temperature before accuracy/ECE and reports per-head ECE."""
    samples = []
    for i in range(100):
        intent = "COMPLAINT" if i % 2 == 0 else "INQUIRY"
        category = HEAD_CONFIGS["category"][i % len(HEAD_CONFIGS["category"])]
        risk = HEAD_CONFIGS["risk"][i % len(HEAD_CONFIGS["risk"])]
        completeness = "SUFFICIENT" if i % 3 != 0 else "INCOMPLETE"
        samples.append({
            "text": f"laporan keluhan warga nomor {i}",
            "intent": intent,
            "category": category,
            "risk": risk,
            "completeness": completeness,
        })

    class OverconfidentONNXSession:
        def get_inputs(self) -> list[Any]:
            return [mock.Mock(name="input_ids"), mock.Mock(name="attention_mask")]
        def get_outputs(self) -> list[Any]:
            outs = []
            for h in ("intent_logits", "category_logits", "risk_logits", "completeness_logits"):
                m = mock.Mock()
                m.name = h
                outs.append(m)
            return outs
        def run(self, output_names: Any, feed_dict: dict[str, Any]) -> list[Any]:
            n = len(feed_dict["input_ids"])
            # Severe overconfidence: logit margin of 4.5 -> uncalibrated ECE > 0.15
            intent_out = [[4.5, -4.5, -4.5] for _ in range(n)]
            category_out = [[4.5 if c == 0 else -4.5 for c in range(6)] for _ in range(n)]
            risk_out = [[4.5 if r == 0 else -4.5 for r in range(4)] for _ in range(n)]
            comp_out = [[4.5 if k == 0 else -4.5 for k in range(3)] for _ in range(n)]
            return [intent_out, category_out, risk_out, comp_out]

    model = OverconfidentONNXSession()
    tokenizer = SimpleOfflineTokenizer()

    # Evaluation with uncalibrated default (T=1.0) -> high uncalibrated ECE
    res_uncal = evaluate_multitask_model(
        model=model,
        samples=samples,
        tokenizer=tokenizer,
        temperatures={"intent": 1.0, "category": 1.0, "risk": 1.0, "completeness": 1.0},
    )
    assert res_uncal["evaluated"] is True
    assert res_uncal["head_ece"]["intent"] > 0.15
    assert res_uncal["head_ece"]["category"] > 0.15

    # Evaluation with dev-fitted artifact temperatures (T=4.0 for all heads)
    dev_fitted_temps = {"intent": 4.0, "category": 4.0, "risk": 4.0, "completeness": 4.0}
    res_cal = evaluate_multitask_model(
        model=model,
        samples=samples,
        tokenizer=tokenizer,
        temperatures=dev_fitted_temps,
    )
    # Per-head temperature application strictly reduces ECE across every head
    for head in EXPECTED_HEADS:
        assert res_cal["head_ece"][head] < res_uncal["head_ece"][head]
        assert "ece" in res_cal["heads"][head]
        assert "calibrated_ece" in res_cal["heads"][head]
        assert res_cal["heads"][head]["ece"] == res_cal["head_ece"][head]
        assert res_cal["heads"][head]["temperature"] == 4.0

    assert res_cal["calibrated_ece"] < res_uncal["calibrated_ece"]
    assert res_cal["temperatures_applied"] == dev_fitted_temps


def test_evaluate_multitask_model_onnx_logit_formats() -> None:
    """Verify ONNX logit extraction handles named outputs, positional, 3D, and 2D concatenated layouts."""
    samples = [
        {"text": "jalan rusak parah", "intent": "COMPLAINT", "category": "ROAD", "risk": "MEDIUM", "completeness": "SUFFICIENT"},
        {"text": "tumpukan sampah liar", "intent": "COMPLAINT", "category": "WASTE", "risk": "HIGH", "completeness": "SUFFICIENT"},
    ]
    tokenizer = SimpleOfflineTokenizer()

    # Format 1: Named outputs
    class SessionNamed:
        def get_inputs(self) -> list[Any]: return [mock.Mock(name="input_ids"), mock.Mock(name="attention_mask")]
        def get_outputs(self) -> list[Any]:
            outs = []
            for h in ("intent_logits", "category_logits", "risk_logits", "completeness_logits"):
                m = mock.Mock()
                m.name = h
                outs.append(m)
            return outs
        def run(self, names: Any, feed_dict: dict[str, Any]) -> list[Any]:
            n = len(feed_dict["input_ids"])
            return [
                [[2.0, 0.0, 0.0] for _ in range(n)],
                [[2.0 if c == 0 else 0.0 for c in range(6)] for _ in range(n)],
                [[2.0 if r == 1 else 0.0 for r in range(4)] for _ in range(n)],
                [[2.0, 0.0, 0.0] for _ in range(n)],
            ]
    res1 = evaluate_multitask_model(SessionNamed(), samples, tokenizer)
    assert res1["evaluated"] is True
    assert set(res1["head_ece"].keys()) == set(EXPECTED_HEADS)

    # Format 2: Positional / generic output names
    class SessionPositional:
        def get_inputs(self) -> list[Any]: return [mock.Mock(name="input_ids"), mock.Mock(name="attention_mask")]
        def get_outputs(self) -> list[Any]:
            outs = []
            for h in ("out_0", "out_1", "out_2", "out_3"):
                m = mock.Mock()
                m.name = h
                outs.append(m)
            return outs
        def run(self, names: Any, feed_dict: dict[str, Any]) -> list[Any]:
            n = len(feed_dict["input_ids"])
            return [
                [[2.0, 0.0, 0.0] for _ in range(n)],
                [[2.0 if c == 0 else 0.0 for c in range(6)] for _ in range(n)],
                [[2.0 if r == 1 else 0.0 for r in range(4)] for _ in range(n)],
                [[2.0, 0.0, 0.0] for _ in range(n)],
            ]
    res2 = evaluate_multitask_model(SessionPositional(), samples, tokenizer)
    assert res2["evaluated"] is True
    assert set(res2["head_ece"].keys()) == set(EXPECTED_HEADS)

    # Format 3: 3D multi-head tensor output (batch, 4, max_classes)
    class Session3D:
        def get_inputs(self) -> list[Any]: return [mock.Mock(name="input_ids"), mock.Mock(name="attention_mask")]
        def get_outputs(self) -> list[Any]:
            m = mock.Mock()
            m.name = "logits"
            return [m]
        def run(self, names: Any, feed_dict: dict[str, Any]) -> list[Any]:
            n = len(feed_dict["input_ids"])
            return [
                [
                    [
                        [2.0 if c == 0 else 0.0 for c in range(6)],
                        [2.0 if c == 0 else 0.0 for c in range(6)],
                        [2.0 if c == 1 else 0.0 for c in range(6)],
                        [2.0 if c == 0 else 0.0 for c in range(6)],
                    ]
                    for _ in range(n)
                ]
            ]
    res3 = evaluate_multitask_model(Session3D(), samples, tokenizer)
    assert res3["evaluated"] is True
    assert set(res3["head_ece"].keys()) == set(EXPECTED_HEADS)

    # Format 4: 2D concatenated logits tensor (batch, 16 - legacy backward compatibility)
    class SessionConcatenated16:
        def get_inputs(self) -> list[Any]: return [mock.Mock(name="input_ids"), mock.Mock(name="attention_mask")]
        def get_outputs(self) -> list[Any]:
            m = mock.Mock()
            m.name = "logits"
            return [m]
        def run(self, names: Any, feed_dict: dict[str, Any]) -> list[Any]:
            n = len(feed_dict["input_ids"])
            row = [
                2.0, 0.0, 0.0,
                2.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 2.0, 0.0, 0.0,
                2.0, 0.0, 0.0,
            ]
            return [[list(row) for _ in range(n)]]
    res4 = evaluate_multitask_model(SessionConcatenated16(), samples, tokenizer)
    assert res4["evaluated"] is True
    assert set(res4["head_ece"].keys()) == set(EXPECTED_HEADS)

    # Format 5: 2D concatenated logits tensor (batch, 22 - dynamic 12-category HEAD_CONFIGS)
    class SessionConcatenated22:
        def get_inputs(self) -> list[Any]: return [mock.Mock(name="input_ids"), mock.Mock(name="attention_mask")]
        def get_outputs(self) -> list[Any]:
            m = mock.Mock()
            m.name = "logits"
            return [m]
        def run(self, names: Any, feed_dict: dict[str, Any]) -> list[Any]:
            n = len(feed_dict["input_ids"])
            row = (
                [2.0, 0.0, 0.0]
                + [2.0 if c == 0 else 0.0 for c in range(12)]
                + [0.0, 2.0, 0.0, 0.0]
                + [2.0, 0.0, 0.0]
            )
            return [[list(row) for _ in range(n)]]
    res5 = evaluate_multitask_model(SessionConcatenated22(), samples, tokenizer)
    assert res5["evaluated"] is True
    assert set(res5["head_ece"].keys()) == set(EXPECTED_HEADS)


def test_dynamic_head_class_offsets_derivation_from_head_configs() -> None:
    """Verify concatenated ONNX class offsets are dynamically derived from HEAD_CONFIGS."""
    # With active HEAD_CONFIGS (12 categories -> 22 logits total)
    offsets, total = compute_head_class_offsets()
    assert total == 22
    assert offsets["intent"] == (0, 3)
    assert offsets["category"] == (3, 15)
    assert offsets["risk"] == (15, 19)
    assert offsets["completeness"] == (19, 22)
    assert len(HEAD_CONFIGS["category"]) == 12

    # With custom/legacy 6-category configuration (16 logits total)
    custom_6cat_configs = {
        "intent": ["COMPLAINT", "INQUIRY", "FEEDBACK"],
        "category": ["ROAD", "DRAINAGE_FLOOD", "WASTE", "CLEAN_WATER", "CIVIL_ADMIN", "HEALTH_SERVICE"],
        "risk": ["LOW", "MEDIUM", "HIGH", "URGENT"],
        "completeness": ["SUFFICIENT", "INCOMPLETE", "AMBIGUOUS"],
    }
    custom_offsets, custom_total = compute_head_class_offsets(custom_6cat_configs)
    assert custom_total == 16
    assert custom_offsets["intent"] == (0, 3)
    assert custom_offsets["category"] == (3, 9)
    assert custom_offsets["risk"] == (9, 13)
    assert custom_offsets["completeness"] == (13, 16)


class _MockNdarray:
    """Mock 2D numpy.ndarray for tests in environments without numpy installed."""

    def __init__(self, data: list[list[float]]):
        self.data = [list(r) for r in data]
        self.ndim = 2
        self.shape = (len(data), len(data[0]) if data else 0)

    def __getitem__(self, item: Any) -> _MockNdarray:
        if isinstance(item, tuple) and len(item) == 2:
            _, col_slice = item
            if isinstance(col_slice, slice):
                return _MockNdarray([r[col_slice] for r in self.data])
        raise NotImplementedError

    def tolist(self) -> list[list[float]]:
        return [list(r) for r in self.data]


def test_extract_head_logits_concatenated_22_logits_numpy_and_list() -> None:
    """Verify _extract_head_logits correctly slices 12-category 22-logit concatenated ONNX outputs."""
    row_values = (
        [100.0 + i for i in range(3)]
        + [200.0 + i for i in range(12)]
        + [300.0 + i for i in range(4)]
        + [400.0 + i for i in range(3)]
    )
    assert len(row_values) == 22

    batch_size = 2
    out_list = [[list(row_values) for _ in range(batch_size)]]

    # 1. Verify list layout
    intent_list = _extract_head_logits(out_list, ["logits"], "intent", 0)
    category_list = _extract_head_logits(out_list, ["logits"], "category", 1)
    risk_list = _extract_head_logits(out_list, ["logits"], "risk", 2)
    completeness_list = _extract_head_logits(out_list, ["logits"], "completeness", 3)

    assert len(intent_list[0]) == 3
    assert intent_list[0] == [100.0, 101.0, 102.0]
    assert len(category_list[0]) == 12
    assert category_list[0] == [200.0 + i for i in range(12)]
    assert len(risk_list[0]) == 4
    assert risk_list[0] == [300.0, 301.0, 302.0, 303.0]
    assert len(completeness_list[0]) == 3
    assert completeness_list[0] == [400.0, 401.0, 402.0]

    # 2. Verify 2D array / ndarray layout
    mock_np_mod = mock.MagicMock()
    mock_np_mod.ndarray = _MockNdarray
    with mock.patch.dict(sys.modules, {"numpy": mock_np_mod}):
        out_arr = [_MockNdarray([list(row_values) for _ in range(batch_size)])]
        intent_arr = _extract_head_logits(out_arr, ["logits"], "intent", 0)
        category_arr = _extract_head_logits(out_arr, ["logits"], "category", 1)
        risk_arr = _extract_head_logits(out_arr, ["logits"], "risk", 2)
        completeness_arr = _extract_head_logits(out_arr, ["logits"], "completeness", 3)

        assert intent_arr.shape == (batch_size, 3)
        assert intent_arr.tolist()[0] == [100.0, 101.0, 102.0]
        assert category_arr.shape == (batch_size, 12)
        assert category_arr.tolist()[0] == [200.0 + i for i in range(12)]
        assert risk_arr.shape == (batch_size, 4)
        assert risk_arr.tolist()[0] == [300.0, 301.0, 302.0, 303.0]
        assert completeness_arr.shape == (batch_size, 3)
        assert completeness_arr.tolist()[0] == [400.0, 401.0, 402.0]


def test_extract_head_logits_preserves_separate_output_onnx_behavior() -> None:
    """Verify separate-output ONNX behavior (named, positional, dict) is strictly preserved."""
    intent_out = [[1.0, 0.0, 0.0]]
    category_out = [[3.5 if c == 5 else 0.0 for c in range(12)]]
    risk_out = [[0.0, 2.0, 0.0, 0.0]]
    comp_out = [[0.0, 0.0, 1.5]]

    # 1. Dict output (Case 1)
    dict_outputs = {
        "intent_logits": intent_out,
        "category_logits": category_out,
        "risk_logits": risk_out,
        "completeness_logits": comp_out,
    }
    assert _extract_head_logits(dict_outputs, [], "intent", 0) == intent_out
    assert _extract_head_logits(dict_outputs, [], "category", 1) == category_out
    assert _extract_head_logits(dict_outputs, [], "risk", 2) == risk_out
    assert _extract_head_logits(dict_outputs, [], "completeness", 3) == comp_out

    # 2. Named outputs matching head names (Case 2)
    named_session_outputs = [intent_out, category_out, risk_out, comp_out]
    output_meta = []
    for h in ("intent_logits", "category_logits", "risk_logits", "completeness_logits"):
        m = mock.Mock()
        m.name = h
        output_meta.append(m)
    for h_idx, head in enumerate(EXPECTED_HEADS):
        extracted = _extract_head_logits(named_session_outputs, output_meta, head, h_idx)
        assert extracted == named_session_outputs[h_idx]

    # 3. Positional outputs with generic names (Case 3, len >= 4)
    generic_meta = ["out_0", "out_1", "out_2", "out_3"]
    for h_idx, head in enumerate(EXPECTED_HEADS):
        extracted = _extract_head_logits(named_session_outputs, generic_meta, head, h_idx)
        assert extracted == named_session_outputs[h_idx]


def test_extract_head_logits_legacy_16_concatenated_fallback() -> None:
    """Verify legacy 16-logit concatenated output is still supported as a backward-compatibility fallback."""
    row_16 = [1.0] * 3 + [2.0] * 6 + [3.0] * 4 + [4.0] * 3
    assert len(row_16) == 16
    out_list = [[list(row_16), list(row_16)]]

    intent = _extract_head_logits(out_list, ["logits"], "intent", 0)
    category = _extract_head_logits(out_list, ["logits"], "category", 1)
    risk = _extract_head_logits(out_list, ["logits"], "risk", 2)
    comp = _extract_head_logits(out_list, ["logits"], "completeness", 3)

    assert len(intent[0]) == 3
    assert len(category[0]) == 6
    assert len(risk[0]) == 4
    assert len(comp[0]) == 3

    # Also verify with 2D ndarray
    mock_np_mod = mock.MagicMock()
    mock_np_mod.ndarray = _MockNdarray
    with mock.patch.dict(sys.modules, {"numpy": mock_np_mod}):
        out_arr = [_MockNdarray([list(row_16), list(row_16)])]
        intent_arr = _extract_head_logits(out_arr, ["logits"], "intent", 0)
        category_arr = _extract_head_logits(out_arr, ["logits"], "category", 1)
        risk_arr = _extract_head_logits(out_arr, ["logits"], "risk", 2)
        comp_arr = _extract_head_logits(out_arr, ["logits"], "completeness", 3)

        assert intent_arr.shape == (2, 3)
        assert category_arr.shape == (2, 6)
        assert risk_arr.shape == (2, 4)
        assert comp_arr.shape == (2, 3)


def test_evaluate_multitask_model_with_12_categories_22_logits_concatenated() -> None:
    """End-to-end evaluation test with concatenated 22-logit ONNX model across all 12 categories."""
    samples = [
        {"text": f"Laporan masalah kategori {cat}", "intent": "COMPLAINT", "category": cat, "risk": "MEDIUM", "completeness": "SUFFICIENT"}
        for cat in HEAD_CONFIGS["category"]
    ]
    tokenizer = SimpleOfflineTokenizer()

    class SessionConcatenated22:
        def get_inputs(self) -> list[Any]:
            return [mock.Mock(name="input_ids"), mock.Mock(name="attention_mask")]

        def get_outputs(self) -> list[Any]:
            m = mock.Mock()
            m.name = "logits"
            return [m]

        def run(self, names: Any, feed_dict: dict[str, Any]) -> list[Any]:
            n = len(feed_dict["input_ids"])
            rows = []
            for _ in range(n):
                # intent: 3 logits (COMPLAINT highest)
                # category: 12 logits (index 6 highest -> PUBLIC_ORDER)
                # risk: 4 logits (MEDIUM highest)
                # completeness: 3 logits (SUFFICIENT highest)
                row = (
                    [3.0, 0.0, 0.0]
                    + [3.0 if c == 6 else 0.0 for c in range(12)]
                    + [0.0, 3.0, 0.0, 0.0]
                    + [3.0, 0.0, 0.0]
                )
                rows.append(row)
            return [rows]

    res = evaluate_multitask_model(SessionConcatenated22(), samples, tokenizer)
    assert res["evaluated"] is True
    assert set(res["head_ece"].keys()) == set(EXPECTED_HEADS)
    assert len(res["heads"]["category"]["classes"]) == 12
    assert res["heads"]["category"]["support"] == 12


def test_calibration_strictly_uses_dev_fitted_artifacts_never_test_labels(tmp_path: Path) -> None:
    """Verify evaluation never calls fit/fit_temperature on test samples (zero test leakage)."""
    model_dir = tmp_path / "artifacts" / "multitask"
    model_dir.mkdir(parents=True)
    model_path = model_dir / "multitask.onnx"
    model_path.write_bytes(b"PROTOBUF_DATA_FOR_TESTING_1234567890" * 3)

    temp_path = model_dir / "temperatures.json"
    dev_temps = {"intent": 2.15, "category": 1.48, "risk": 1.72, "completeness": 2.25}
    save_temperatures(dev_temps, temp_path)

    out_file = tmp_path / "leakage_eval.json"
    cfg = EvaluationConfig(
        multitask_model_path=str(model_path),
        temperature_path=str(temp_path),
        output_path=str(out_file),
        num_scenarios=12,
        eval_split="test",
        dry_run=True,
    )

    with mock.patch("services.intelligence.calibration.fit_temperature") as mock_fit_temp:
        with mock.patch.object(TemperatureCalibrator, "fit") as mock_cal_fit:
            report = run_evaluate_m3(cfg)
            mock_fit_temp.assert_not_called()
            mock_cal_fit.assert_not_called()

    assert report.metrics["temperatures_source"] == str(temp_path)
    assert report.metrics["temperatures_applied"] == dev_temps
    assert "head_ece" in report.metrics
    for head in EXPECTED_HEADS:
        assert head in report.metrics["head_ece"]


def test_run_evaluate_m3_cli_with_temperature_path(tmp_path: Path) -> None:
    """Verify CLI accepts --temperature-path and propagates per-head temperatures."""
    temp_file = tmp_path / "cli_temps.json"
    cli_temps = {"intent": 2.05, "category": 1.62, "risk": 1.88, "completeness": 2.12}
    save_temperatures(cli_temps, temp_file)

    out_file = tmp_path / "cli_temp_eval.json"
    code = evaluate_main([
        "--dry-run",
        "--num-scenarios", "12",
        "--temperature-path", str(temp_file),
        "--output-path", str(out_file),
    ])
    assert code == 0
    assert out_file.is_file()

    import json
    data = json.loads(out_file.read_text(encoding="utf-8"))
    assert data["metrics"]["temperatures_source"] == str(temp_file)
    assert data["metrics"]["temperatures_applied"] == cli_temps
    assert "head_ece" in data["metrics"]
    for head in EXPECTED_HEADS:
        assert head in data["metrics"]["head_ece"]


# ---------------------------------------------------------------------------
# Independent Held-Out Synthetic Evaluation & Audit Tests
# ---------------------------------------------------------------------------


def test_load_independent_held_out_trajectory_format(tmp_path: Path) -> None:
    indep_file = tmp_path / "indep_traj.jsonl"
    traj_record = {
        "scenario_id": "sep-gen-sc-01",
        "family_id": "sep-gen-fam-01",
        "category": "ROAD",
        "split": "test",
        "provenance": PROVENANCE_SYNTHETIC_INDEPENDENT,
        "generator": "external_synthetic_generator_v1",
        "world_truth": {
            "intent": "COMPLAINT",
            "category": "ROAD",
            "risk": "HIGH",
            "completeness": "SUFFICIENT",
        },
        "turns": [
            {
                "turn": 1,
                "bubbles": [
                    {
                        "source_message_id": "sep-msg-01",
                        "text": "Laporan jalan amblas parah di Jalan Kaliurang KM 12 depan apotek",
                    }
                ],
                "observable_facts": ["Jalan Kaliurang KM 12"],
                "expected_action": {
                    "turn": 1,
                    "allowed_actions": ["EXECUTE"],
                    "missing": [],
                },
            }
        ],
    }
    indep_file.write_text(json.dumps(traj_record) + "\n", encoding="utf-8")

    mt_samples, ner_samples, raw_records, meta = load_independent_held_out_dataset(indep_file)

    assert len(raw_records) == 1
    assert len(mt_samples) == 1
    assert mt_samples[0]["intent"] == "COMPLAINT"
    assert mt_samples[0]["category"] == "ROAD"
    assert mt_samples[0]["risk"] == "HIGH"
    assert mt_samples[0]["completeness"] == "SUFFICIENT"
    assert "Jalan Kaliurang KM 12" in mt_samples[0]["text"]

    assert len(ner_samples) >= 1
    assert meta["provenance"] == PROVENANCE_SYNTHETIC_INDEPENDENT
    assert meta["is_human_written"] is False
    assert meta["human_written"] is False
    assert len(meta["file_sha256"]) == 64


def test_load_independent_held_out_flat_sample_format(tmp_path: Path) -> None:
    indep_file = tmp_path / "indep_flat.jsonl"
    records = [
        {
            "text": "Sampah menggunung bau busuk di pasar kranji",
            "intent": "COMPLAINT",
            "category": "WASTE",
            "risk": "MEDIUM",
            "completeness": "SUFFICIENT",
            "spans": [[0, 6, "OBJECT"], [28, 40, "LOCATION"]],
            "provenance": PROVENANCE_SYNTHETIC_INDEPENDENT,
            "family_id": "sep-fam-waste-01",
        },
        {
            "text": "Saluran air tersumbat lumpur pekat di jalan sudirman",
            "intent": "COMPLAINT",
            "category": "DRAINAGE_FLOOD",
            "risk": "HIGH",
            "completeness": "SUFFICIENT",
            "entities": [{"start": 0, "end": 11, "label": "OBJECT"}, {"start": 35, "end": 49, "label": "LOCATION"}],
            "provenance": PROVENANCE_SYNTHETIC_INDEPENDENT,
            "family_id": "sep-fam-flood-02",
        },
    ]
    with indep_file.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    mt_samples, ner_samples, raw_records, meta = load_independent_held_out_dataset(indep_file)

    assert len(raw_records) == 2
    assert len(mt_samples) == 2
    assert mt_samples[0]["category"] == "WASTE"
    assert mt_samples[1]["category"] == "DRAINAGE_FLOOD"
    assert len(ner_samples) == 2
    assert ner_samples[0]["spans"] == [(0, 6, "OBJECT"), (28, 40, "LOCATION")]
    assert ner_samples[1]["spans"] == [(0, 11, "OBJECT"), (35, 49, "LOCATION")]
    assert meta["provenance"] == PROVENANCE_SYNTHETIC_INDEPENDENT


def test_load_independent_held_out_metadata_header_line(tmp_path: Path) -> None:
    indep_file = tmp_path / "indep_meta.jsonl"
    header = {"_type": "metadata", "provenance": PROVENANCE_SYNTHETIC_INDEPENDENT, "generator": "separate_gen_v2"}
    rec = {
        "text": "Pipa PDAM pecah air menyembur",
        "intent": "COMPLAINT",
        "category": "CLEAN_WATER",
        "risk": "HIGH",
        "completeness": "SUFFICIENT",
        "family_id": "sep-fam-pdam-01",
    }
    content = json.dumps(header) + "\n" + json.dumps(rec) + "\n"
    indep_file.write_text(content, encoding="utf-8")

    mt_samples, ner_samples, raw_records, meta = load_independent_held_out_dataset(indep_file)
    assert len(raw_records) == 1
    assert meta["generator"] == "separate_gen_v2"
    assert meta["provenance"] == PROVENANCE_SYNTHETIC_INDEPENDENT
    assert mt_samples[0]["category"] == "CLEAN_WATER"


def test_load_independent_held_out_missing_provenance_rejected(tmp_path: Path) -> None:
    indep_file = tmp_path / "indep_no_prov.jsonl"
    rec = {"text": "Jalan amblas", "category": "ROAD", "intent": "COMPLAINT"}
    indep_file.write_text(json.dumps(rec) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="missing required provenance"):
        load_independent_held_out_dataset(indep_file)


def test_load_independent_held_out_invalid_provenance_rejected(tmp_path: Path) -> None:
    indep_file = tmp_path / "indep_bad_prov.jsonl"
    rec = {"text": "Jalan amblas", "category": "ROAD", "intent": "COMPLAINT", "provenance": "internal_inhouse"}
    indep_file.write_text(json.dumps(rec) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="invalid provenance"):
        load_independent_held_out_dataset(indep_file)


def test_load_independent_held_out_human_written_claim_rejected(tmp_path: Path) -> None:
    indep_file1 = tmp_path / "indep_human_prov.jsonl"
    rec1 = {"text": "Jalan amblas", "category": "ROAD", "provenance": "human_written"}
    indep_file1.write_text(json.dumps(rec1) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="claims to be human-written"):
        load_independent_held_out_dataset(indep_file1)

    indep_file2 = tmp_path / "indep_human_flag.jsonl"
    rec2 = {
        "text": "Jalan amblas",
        "category": "ROAD",
        "provenance": PROVENANCE_SYNTHETIC_INDEPENDENT,
        "human_written": True,
    }
    indep_file2.write_text(json.dumps(rec2) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="claims to be human-written"):
        load_independent_held_out_dataset(indep_file2)


def test_audit_independent_dataset_training_text_leakage(tmp_path: Path) -> None:
    train_text = "jalan berlubang sangat parah di depan kantor pos cianjur"
    training_traj = ComplaintTrajectory(
        scenario_id="sc-train-01",
        family_id="fam-train-01",
        split=DatasetSplit.TRAIN,
        category=Category.ROAD,
        turns=(
            TrajectoryTurn(
                turn=1,
                bubbles=(TrajectoryBubble(source_message_id="b1", text=train_text),),
                expected_action=TurnExpectedAction(turn=1, allowed_actions=(DecisionMode.EXECUTE,), missing=()),
            ),
        ),
    )

    indep_file = tmp_path / "leaked_text.jsonl"
    indep_rec = {
        "text": train_text,
        "intent": "COMPLAINT",
        "category": "ROAD",
        "risk": "HIGH",
        "completeness": "SUFFICIENT",
        "provenance": PROVENANCE_SYNTHETIC_INDEPENDENT,
        "family_id": "sep-fam-unique-01",
    }
    indep_file.write_text(json.dumps(indep_rec) + "\n", encoding="utf-8")
    _, _, raw_recs, _ = load_independent_held_out_dataset(indep_file)

    audit = audit_independent_held_out(
        independent_records=raw_recs,
        independent_path=indep_file,
        training_trajectories=[training_traj],
    )

    assert audit["passed"] is False
    assert audit["not_used_for_training"] is False
    assert audit["text_overlap_count"] == 1
    assert any("Text overlap" in v for v in audit["violations"])


def test_audit_independent_dataset_calibration_dev_leakage(tmp_path: Path) -> None:
    dev_text = "gorong-gorong tersumbat sampah daun dan ranting di jalan kaliurang"
    dev_traj = ComplaintTrajectory(
        scenario_id="sc-dev-01",
        family_id="fam-dev-01",
        split=DatasetSplit.DEV,
        category=Category.DRAINAGE_FLOOD,
        turns=(
            TrajectoryTurn(
                turn=1,
                bubbles=(TrajectoryBubble(source_message_id="b1", text=dev_text),),
                expected_action=TurnExpectedAction(turn=1, allowed_actions=(DecisionMode.EXECUTE,), missing=()),
            ),
        ),
    )

    indep_file = tmp_path / "leaked_dev.jsonl"
    indep_rec = {
        "text": dev_text,
        "intent": "COMPLAINT",
        "category": "DRAINAGE_FLOOD",
        "risk": "MEDIUM",
        "completeness": "SUFFICIENT",
        "provenance": PROVENANCE_SYNTHETIC_INDEPENDENT,
        "family_id": "sep-fam-unique-02",
    }
    indep_file.write_text(json.dumps(indep_rec) + "\n", encoding="utf-8")
    _, _, raw_recs, _ = load_independent_held_out_dataset(indep_file)

    audit = audit_independent_held_out(
        independent_records=raw_recs,
        independent_path=indep_file,
        training_trajectories=[dev_traj],
    )

    assert audit["passed"] is False
    assert audit["not_used_for_calibration"] is False
    assert audit["text_overlap_count"] == 1
    assert any("Text overlap" in v for v in audit["violations"])


def test_audit_independent_dataset_family_overlap(tmp_path: Path) -> None:
    shared_fam = "fam-shared-collision"
    train_traj = ComplaintTrajectory(
        scenario_id="sc-train-02",
        family_id=shared_fam,
        split=DatasetSplit.TRAIN,
        category=Category.ROAD,
        turns=(
            TrajectoryTurn(
                turn=1,
                bubbles=(TrajectoryBubble(source_message_id="b1", text="Teks unik internal"),),
                expected_action=TurnExpectedAction(turn=1, allowed_actions=(DecisionMode.EXECUTE,), missing=()),
            ),
        ),
    )

    indep_file = tmp_path / "leaked_fam.jsonl"
    indep_rec = {
        "text": "Teks unik terpisah dari generator lain",
        "intent": "COMPLAINT",
        "category": "ROAD",
        "risk": "LOW",
        "completeness": "SUFFICIENT",
        "provenance": PROVENANCE_SYNTHETIC_INDEPENDENT,
        "family_id": shared_fam,
    }
    indep_file.write_text(json.dumps(indep_rec) + "\n", encoding="utf-8")
    _, _, raw_recs, _ = load_independent_held_out_dataset(indep_file)

    audit = audit_independent_held_out(
        independent_records=raw_recs,
        independent_path=indep_file,
        training_trajectories=[train_traj],
    )

    assert audit["passed"] is False
    assert audit["not_used_for_training"] is False
    assert audit["family_overlap_count"] == 1
    assert any("Family overlap" in v for v in audit["violations"])


def test_audit_independent_dataset_manifest_hash_collision(tmp_path: Path) -> None:
    indep_file = tmp_path / "colliding_file.jsonl"
    content = json.dumps({
        "text": "Pohon tumbang menutup jalan",
        "intent": "COMPLAINT",
        "category": "ROAD",
        "risk": "HIGH",
        "completeness": "SUFFICIENT",
        "provenance": PROVENANCE_SYNTHETIC_INDEPENDENT,
    }) + "\n"
    indep_file.write_text(content, encoding="utf-8")
    actual_sha = compute_file_sha256(indep_file)

    manifest_item = ArtifactManifestItem(
        name="calibrated-temperatures",
        version="v1.0.0",
        sha256=actual_sha,
        path="colliding_file.jsonl",
        size_bytes=len(content.encode("utf-8")),
        task="calibration",
        precision="fp32",
    )
    manifest = ArtifactManifest(artifacts={"calibrated-temperatures": manifest_item})

    _, _, raw_recs, _ = load_independent_held_out_dataset(indep_file)

    audit = audit_independent_held_out(
        independent_records=raw_recs,
        independent_path=indep_file,
        training_trajectories=[],
        manifest=manifest,
    )

    assert audit["passed"] is False
    assert audit["manifest_collision"] is True
    assert audit["not_used_for_calibration"] is False
    assert any("Manifest collision" in v for v in audit["violations"])


def test_evaluate_m3_with_clean_independent_held_out_dry_run(tmp_path: Path) -> None:
    indep_file = tmp_path / "clean_indep.jsonl"
    records = [
        {
            "text": f"Aduan infrastruktur nomor {i} dari sistem generator eksternal independen",
            "intent": "COMPLAINT",
            "category": "ROAD" if i % 2 == 0 else "WASTE",
            "risk": "HIGH" if i % 2 == 0 else "MEDIUM",
            "completeness": "SUFFICIENT",
            "spans": [[0, 19, "OBJECT"]],
            "provenance": PROVENANCE_SYNTHETIC_INDEPENDENT,
            "family_id": f"clean-external-fam-{i:03d}",
        }
        for i in range(1, 11)
    ]
    with indep_file.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    eval_out = tmp_path / "eval_clean_indep.json"
    cfg = EvaluationConfig(
        independent_held_out_path=str(indep_file),
        output_path=str(eval_out),
        dry_run=True,
        num_scenarios=12,
    )
    report = run_evaluate_m3(cfg)

    assert report.status == "PASSED"
    assert report.pipeline_status == "PASSED"
    assert report.quality_status == "PASSED"
    assert report.all_checks_passed is True

    # Check synthetic_independent_evaluation
    assert report.synthetic_independent_evaluation is not None
    indep_eval = report.synthetic_independent_evaluation
    assert indep_eval["provenance"] == PROVENANCE_SYNTHETIC_INDEPENDENT
    assert indep_eval["dataset_type"] == "synthetic_independent"
    assert indep_eval["is_human_written"] is False
    assert indep_eval["human_written"] is False
    assert indep_eval["status"] == "PASSED"
    assert indep_eval["quality_status"] == "PASSED"
    assert "NOT human-written" in indep_eval["disclaimer"]

    # Check audit section
    assert indep_eval["audit"]["passed"] is True
    assert indep_eval["audit"]["provenance_verified"] is True
    assert indep_eval["audit"]["not_used_for_training"] is True
    assert indep_eval["audit"]["not_used_for_calibration"] is True
    assert indep_eval["audit"]["text_overlap_count"] == 0
    assert indep_eval["audit"]["family_overlap_count"] == 0

    # Check metrics segregation
    assert "synthetic_independent" in report.metrics
    indep_metric = report.metrics["synthetic_independent"]
    assert indep_metric["provenance"] == PROVENANCE_SYNTHETIC_INDEPENDENT
    assert indep_metric["is_human_written"] is False
    assert indep_metric["status"] == "PASSED"
    assert indep_metric["macro_f1"] >= 0.85

    # File output verified
    assert eval_out.is_file()
    loaded_report = M3EvaluationReport.model_validate_json(eval_out.read_text(encoding="utf-8"))
    assert loaded_report.synthetic_independent_evaluation is not None
    assert loaded_report.synthetic_independent_evaluation["is_human_written"] is False


def test_evaluate_m3_with_contaminated_independent_held_out_blocks_pipeline(tmp_path: Path) -> None:
    # First generate standard dataset to discover its text
    ds = generate_dataset(num_scenarios=12, seed=42)
    train_text = ds[0].turns[0].bubbles[0].text

    indep_file = tmp_path / "contaminated_indep.jsonl"
    indep_rec = {
        "text": train_text,
        "intent": "COMPLAINT",
        "category": "ROAD",
        "risk": "HIGH",
        "completeness": "SUFFICIENT",
        "provenance": PROVENANCE_SYNTHETIC_INDEPENDENT,
        "family_id": "sep-fam-contam-01",
    }
    indep_file.write_text(json.dumps(indep_rec) + "\n", encoding="utf-8")

    cfg = EvaluationConfig(
        independent_held_out_path=str(indep_file),
        output_path=str(tmp_path / "eval_contam.json"),
        dry_run=True,
        num_scenarios=12,
        seed=42,
    )
    report = run_evaluate_m3(cfg)

    assert report.status == "FAILED"
    assert report.pipeline_status == "FAILED"
    assert report.all_checks_passed is False
    assert report.synthetic_independent_evaluation is not None
    assert report.synthetic_independent_evaluation["status"] == "FAILED"
    assert report.synthetic_independent_evaluation["audit"]["passed"] is False
    assert report.synthetic_independent_evaluation["audit"]["not_used_for_training"] is False


def test_evaluate_m3_with_mock_onnx_and_independent_held_out(tmp_path: Path) -> None:
    multitask_onnx = tmp_path / "multitask.onnx"
    multitask_onnx.write_bytes(b"PROTOBUF_SAMPLE_NON_SKELETON_DATA_LONG_ENOUGH_FOR_TESTING" * 3)
    ner_onnx = tmp_path / "ner.onnx"
    ner_onnx.write_bytes(b"PROTOBUF_SAMPLE_NON_SKELETON_DATA_LONG_ENOUGH_FOR_TESTING" * 3)

    indep_file = tmp_path / "indep_mock.jsonl"
    records = [
        {
            "text": f"Laporan masalah sampah nomor {i} di jalan magelang",
            "intent": "COMPLAINT",
            "category": "WASTE",
            "risk": "LOW",
            "completeness": "SUFFICIENT",
            "spans": [[0, 22, "OBJECT"]],
            "provenance": PROVENANCE_SYNTHETIC_INDEPENDENT,
            "family_id": f"external-mock-fam-{i}",
        }
        for i in range(1, 6)
    ]
    with indep_file.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    class MockMultitaskSession:
        def get_inputs(self) -> list[Any]:
            m1 = mock.Mock()
            m1.name = "input_ids"
            m2 = mock.Mock()
            m2.name = "attention_mask"
            return [m1, m2]

        def get_outputs(self) -> list[Any]:
            return [
                mock.Mock(name="intent_logits"),
                mock.Mock(name="category_logits"),
                mock.Mock(name="risk_logits"),
                mock.Mock(name="completeness_logits"),
            ]

        def run(self, output_names: Any, feed_dict: dict[str, Any]) -> list[Any]:
            bsz = len(feed_dict["input_ids"])
            # Predict high confidence for expected labels: intent=COMPLAINT(0), category=ROAD(0), risk=LOW(0), comp=SUFFICIENT(0)
            int_l = [[8.0, -8.0, -8.0] for _ in range(bsz)]
            cat_l = [[8.0, -8.0, -8.0, -8.0, -8.0, -8.0] for _ in range(bsz)]
            risk_l = [[8.0, -8.0, -8.0, -8.0] for _ in range(bsz)]
            comp_l = [[8.0, -8.0, -8.0] for _ in range(bsz)]
            return [int_l, cat_l, risk_l, comp_l]

    class MockNERSession:
        def get_inputs(self) -> list[Any]:
            m1 = mock.Mock()
            m1.name = "input_ids"
            m2 = mock.Mock()
            m2.name = "attention_mask"
            return [m1, m2]

        def get_outputs(self) -> list[Any]:
            return [mock.Mock(name="logits")]

        def run(self, output_names: Any, feed_dict: dict[str, Any]) -> list[Any]:
            bsz = len(feed_dict["input_ids"])
            seq_l = len(feed_dict["input_ids"][0])
            tag_len = len(DEFAULT_TAGSET)
            step = [0.0] * tag_len
            step[0] = 5.0
            return [[[list(step) for _ in range(seq_l)] for _ in range(bsz)]]

    def mock_load(path: Path | str, **kwargs: Any) -> Any:
        if "ner" in str(path):
            return MockNERSession()
        return MockMultitaskSession()

    with mock.patch("scripts.evaluate_m3.load_onnx_inference_session", side_effect=mock_load):
        eval_out = tmp_path / "eval_indep_mock_onnx.json"
        cfg = EvaluationConfig(
            multitask_model_path=str(multitask_onnx),
            ner_model_path=str(ner_onnx),
            independent_held_out_path=str(indep_file),
            output_path=str(eval_out),
            num_scenarios=12,
            benchmark_runs=5,
            enforce_quality_gates=False,
        )
        report = run_evaluate_m3(cfg)

    assert report.status == "PASSED"
    assert report.synthetic_independent_evaluation is not None
    assert report.synthetic_independent_evaluation["provenance"] == PROVENANCE_SYNTHETIC_INDEPENDENT
    assert report.synthetic_independent_evaluation["is_human_written"] is False
    assert report.synthetic_independent_evaluation["metrics"]["multitask"]["evaluated"] is True
    assert report.synthetic_independent_evaluation["metrics"]["ner"]["evaluated"] is True
    assert report.metrics["synthetic_independent"]["status"] == "PASSED"


def test_evaluate_m3_cli_with_independent_held_out(tmp_path: Path) -> None:
    indep_file = tmp_path / "cli_indep.jsonl"
    rec = {
        "text": "Lampu penerangan padam total di jalan affandi",
        "intent": "COMPLAINT",
        "category": "ROAD",
        "risk": "MEDIUM",
        "completeness": "SUFFICIENT",
        "spans": [[0, 22, "OBJECT"]],
        "provenance": PROVENANCE_SYNTHETIC_INDEPENDENT,
        "family_id": "sep-fam-cli-01",
    }
    indep_file.write_text(json.dumps(rec) + "\n", encoding="utf-8")

    out_file = tmp_path / "cli_indep_report.json"
    exit_code = evaluate_main([
        "--dry-run",
        "--num-scenarios", "12",
        "--independent-held-out-path", str(indep_file),
        "--output-path", str(out_file),
    ])
    assert exit_code == 0
    assert out_file.is_file()

    data = json.loads(out_file.read_text(encoding="utf-8"))
    assert data["status"] == "PASSED"
    assert "synthetic_independent_evaluation" in data
    assert data["synthetic_independent_evaluation"]["provenance"] == PROVENANCE_SYNTHETIC_INDEPENDENT
    assert data["synthetic_independent_evaluation"]["is_human_written"] is False
    assert data["synthetic_independent_evaluation"]["human_written"] is False
    assert data["synthetic_independent_evaluation"]["audit"]["passed"] is True
    assert data["synthetic_independent_evaluation"]["audit"]["not_used_for_training"] is True
    assert data["synthetic_independent_evaluation"]["audit"]["not_used_for_calibration"] is True
    assert data["metrics"]["synthetic_independent"]["is_human_written"] is False
    assert "internal_quality_gates" in data
    assert data["internal_quality_status"] == "PASSED"
    assert "independent_quality_gates" in data
    assert data["independent_quality_status"] == "PASSED"
    assert data["independent_quality_gates"]["is_canary"] is True
    assert data["independent_quality_gates"]["is_human_written"] is False


# ---------------------------------------------------------------------------
# Separation of Internal Synthetic Quality Gates vs Independent OOD Canary Tests
# ---------------------------------------------------------------------------


def test_separation_of_internal_quality_gates_and_independent_canary_root_report(tmp_path: Path) -> None:
    indep_file = tmp_path / "indep_sep.jsonl"
    records = [
        {
            "text": f"Laporan jalan amblas nomor {i} dari canary terpisah",
            "intent": "COMPLAINT",
            "category": "ROAD",
            "risk": "HIGH",
            "completeness": "SUFFICIENT",
            "spans": [[0, 20, "OBJECT"]],
            "provenance": PROVENANCE_SYNTHETIC_INDEPENDENT,
            "family_id": f"sep-canary-fam-{i:03d}",
        }
        for i in range(1, 11)
    ]
    with indep_file.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    eval_out = tmp_path / "eval_sep.json"
    cfg = EvaluationConfig(
        independent_held_out_path=str(indep_file),
        output_path=str(eval_out),
        dry_run=True,
        num_scenarios=12,
    )
    report = run_evaluate_m3(cfg)

    # 1. Root-level exposure of internal synthetic quality gates
    assert report.internal_quality_gates is not None
    assert isinstance(report.internal_quality_gates, dict)
    assert report.internal_quality_gates["status"] == "PASSED"
    assert report.internal_quality_gates["gate_type"] == "internal_synthetic"
    assert report.internal_quality_status == "PASSED"
    assert report.internal_status == "PASSED"

    # 2. Root-level exposure of independent synthetic OOD canary quality gates & status
    assert report.independent_quality_gates is not None
    assert isinstance(report.independent_quality_gates, dict)
    assert report.independent_quality_gates["status"] == "PASSED"
    assert report.independent_quality_gates["gate_type"] == "independent_synthetic_ood_canary"
    assert report.independent_quality_gates["is_canary"] is True
    assert report.independent_quality_gates["canary"] is True
    assert report.independent_quality_gates["provenance"] == PROVENANCE_SYNTHETIC_INDEPENDENT
    assert report.independent_quality_gates["is_human_written"] is False
    assert report.independent_quality_gates["human_written"] is False
    assert report.independent_quality_status == "PASSED"
    assert report.independent_status == "PASSED"
    assert report.independent_canary_status == "PASSED"

    # 3. Global status must clearly be PASSED when both internal and independent pass
    assert report.status == "PASSED"
    assert report.pipeline_status == "PASSED"
    assert report.all_checks_passed is True

    # 4. Backward compatibility preserved
    assert report.quality_gates == report.internal_quality_gates
    assert report.quality_status == report.internal_quality_status
    assert report.synthetic_independent_evaluation is not None
    assert report.synthetic_independent_evaluation["status"] == "PASSED"


def test_independent_quality_gates_omitted_when_no_independent_dataset(tmp_path: Path) -> None:
    eval_out = tmp_path / "eval_no_indep.json"
    cfg = EvaluationConfig(
        output_path=str(eval_out),
        dry_run=True,
        num_scenarios=12,
    )
    report = run_evaluate_m3(cfg)

    # Internal gates exist and pass
    assert report.internal_quality_gates is not None
    assert report.internal_quality_gates["status"] == "PASSED"
    assert report.internal_quality_status == "PASSED"

    # Independent canary fields are strictly None when independent dataset is not provided
    assert report.independent_quality_gates is None
    assert report.independent_quality_status is None
    assert report.independent_status is None
    assert report.independent_canary_status is None
    assert report.synthetic_independent_evaluation is None

    # Global status is PASSED
    assert report.status == "PASSED"
    assert report.all_checks_passed is True


def test_global_status_matrix_passed_only_when_appropriate(tmp_path: Path) -> None:
    indep_file = tmp_path / "indep_matrix.jsonl"
    records = [
        {
            "text": f"Laporan fasilitas nomor {i} dari generator ood canary",
            "intent": "COMPLAINT",
            "category": "DRAINAGE_FLOOD",
            "risk": "HIGH",
            "completeness": "SUFFICIENT",
            "spans": [[0, 17, "OBJECT"]],
            "provenance": PROVENANCE_SYNTHETIC_INDEPENDENT,
            "family_id": f"ood-canary-fam-{i:03d}",
        }
        for i in range(1, 11)
    ]
    with indep_file.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    # Case A: Both internal and independent pass -> global PASSED
    cfg_both_pass = EvaluationConfig(
        independent_held_out_path=str(indep_file),
        output_path=str(tmp_path / "rep_both_pass.json"),
        dry_run=True,
        num_scenarios=12,
    )
    rep_both_pass = run_evaluate_m3(cfg_both_pass)
    assert rep_both_pass.internal_quality_status == "PASSED"
    assert rep_both_pass.independent_quality_status == "PASSED"
    assert rep_both_pass.status == "PASSED"
    assert rep_both_pass.all_checks_passed is True

    # Case B1: Independent canary quality fails, internal passes -> global INTEGRITY_PASSED_QUALITY_FAILED (NOT PASSED)
    gate_pass = {
        "passed": True,
        "status": "PASSED",
        "synthetic_evaluation": True,
        "failed_gates": [],
        "passed_gates": ["intent_macro_f1: 0.95 >= 0.90"],
        "gate_details": {},
    }
    gate_fail = {
        "passed": False,
        "status": "FAILED",
        "synthetic_evaluation": True,
        "failed_gates": ["category_macro_f1: 0.70 < 0.85"],
        "passed_gates": [],
        "gate_details": {},
    }

    # 1st call is independent canary gates, 2nd call is internal gates
    with mock.patch("scripts.evaluate_m3.evaluate_synthetic_quality_gates", side_effect=[dict(gate_fail), dict(gate_pass)]):
        cfg_canary_fail = EvaluationConfig(
            independent_held_out_path=str(indep_file),
            output_path=str(tmp_path / "rep_canary_fail.json"),
            dry_run=True,
            num_scenarios=12,
        )
        rep_canary_fail = run_evaluate_m3(cfg_canary_fail)
        assert rep_canary_fail.internal_quality_status == "PASSED"
        assert rep_canary_fail.independent_quality_status == "FAILED"
        assert rep_canary_fail.status == "INTEGRITY_PASSED_QUALITY_FAILED"
        assert rep_canary_fail.all_checks_passed is False

    # Case B2: Internal quality fails, independent canary passes -> global INTEGRITY_PASSED_QUALITY_FAILED (NOT PASSED)
    with mock.patch("scripts.evaluate_m3.evaluate_synthetic_quality_gates", side_effect=[dict(gate_pass), dict(gate_fail)]):
        cfg_int_fail = EvaluationConfig(
            independent_held_out_path=str(indep_file),
            output_path=str(tmp_path / "rep_int_fail.json"),
            dry_run=True,
            num_scenarios=12,
        )
        rep_int_fail = run_evaluate_m3(cfg_int_fail)
        assert rep_int_fail.internal_quality_status == "FAILED"
        assert rep_int_fail.independent_quality_status == "PASSED"
        assert rep_int_fail.status == "INTEGRITY_PASSED_QUALITY_FAILED"
        assert rep_int_fail.all_checks_passed is False

    # Case C: Pipeline integrity fails (independent text leakage) -> global FAILED (NOT PASSED)
    ds = generate_dataset(num_scenarios=12, seed=42)
    leaked_text = ds[0].turns[0].bubbles[0].text
    leaked_indep_file = tmp_path / "indep_leaked.jsonl"
    leaked_rec = {
        "text": leaked_text,
        "intent": "COMPLAINT",
        "category": "ROAD",
        "risk": "HIGH",
        "completeness": "SUFFICIENT",
        "provenance": PROVENANCE_SYNTHETIC_INDEPENDENT,
        "family_id": "leaked-fam-001",
    }
    leaked_indep_file.write_text(json.dumps(leaked_rec) + "\n", encoding="utf-8")
    cfg_leak = EvaluationConfig(
        independent_held_out_path=str(leaked_indep_file),
        output_path=str(tmp_path / "rep_leak.json"),
        dry_run=True,
        num_scenarios=12,
        seed=42,
    )
    rep_leak = run_evaluate_m3(cfg_leak)
    assert rep_leak.pipeline_status == "FAILED"
    assert rep_leak.status == "FAILED"
    assert rep_leak.all_checks_passed is False


def test_metrics_are_not_silently_mixed(tmp_path: Path) -> None:
    indep_file = tmp_path / "indep_non_mix.jsonl"
    records = [
        {
            "text": f"Laporan pohon tumbang {i} ood canary unik",
            "intent": "COMPLAINT",
            "category": "ROAD",
            "risk": "MEDIUM",
            "completeness": "SUFFICIENT",
            "spans": [[0, 21, "OBJECT"]],
            "provenance": PROVENANCE_SYNTHETIC_INDEPENDENT,
            "family_id": f"non-mix-fam-{i:03d}",
        }
        for i in range(1, 6)
    ]
    with indep_file.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    cfg = EvaluationConfig(
        independent_held_out_path=str(indep_file),
        output_path=str(tmp_path / "rep_non_mix.json"),
        dry_run=True,
        num_scenarios=12,
    )
    report = run_evaluate_m3(cfg)

    # 1. Root metrics are strictly internal synthetic metrics
    assert report.metrics["metrics_source"] == "internal_synthetic"
    assert report.metrics["independent_metrics_segregated"] is True
    assert "macro_f1_multitask" in report.metrics
    assert report.metrics["held_out_trajectories_count"] > 0
    assert report.metrics["multitask_samples_count"] > 0

    # 2. Canary metrics are isolated in synthetic_independent_evaluation
    assert report.synthetic_independent_evaluation is not None
    canary_metrics = report.synthetic_independent_evaluation["metrics"]
    assert canary_metrics["metrics_isolated"] is True
    assert canary_metrics["multitask_samples_count"] == 5
    assert canary_metrics["provenance"] == PROVENANCE_SYNTHETIC_INDEPENDENT
    assert canary_metrics["is_human_written"] is False

    # 3. Segregation in metrics dictionary for backward compatibility
    assert "synthetic_independent" in report.metrics
    assert report.metrics["synthetic_independent"]["is_canary"] is True
    assert report.metrics["synthetic_independent"]["is_human_written"] is False


def test_independent_canary_never_influences_calibration(tmp_path: Path) -> None:
    # 1. Prepare dev-fitted temperature file
    cal_file = tmp_path / "temperatures.json"
    save_temperatures({"intent": 1.7, "category": 1.3, "risk": 1.5, "completeness": 1.4}, cal_file)

    indep_file = tmp_path / "indep_cal_test.jsonl"
    rec = {
        "text": "Lampu penerangan jalan padam di jalan godean km 5",
        "intent": "COMPLAINT",
        "category": "ROAD",
        "risk": "LOW",
        "completeness": "SUFFICIENT",
        "provenance": PROVENANCE_SYNTHETIC_INDEPENDENT,
        "family_id": "canary-cal-fam-01",
    }
    indep_file.write_text(json.dumps(rec) + "\n", encoding="utf-8")

    cfg = EvaluationConfig(
        temperature_path=str(cal_file),
        independent_held_out_path=str(indep_file),
        output_path=str(tmp_path / "rep_cal.json"),
        dry_run=True,
        num_scenarios=12,
    )
    report = run_evaluate_m3(cfg)

    # Calibration source is strictly the calibrated temperature file, unaffected by canary
    assert report.metrics["temperatures_source"] == str(cal_file.resolve())
    assert report.metrics["temperatures_applied"]["intent"] == 1.7

    # Audit confirms independent canary was NOT used for calibration
    indep_eval = report.synthetic_independent_evaluation
    assert indep_eval is not None
    assert indep_eval["audit"]["not_used_for_calibration"] is True

    # 2. Collision detection: passing independent dataset path as temperatures path fails audit
    audit_colliding = audit_independent_held_out(
        independent_records=[rec],
        independent_path=indep_file,
        training_trajectories=[],
        calibration_path=indep_file,
    )
    assert audit_colliding["passed"] is False
    assert audit_colliding["not_used_for_calibration"] is False
    assert any("Collision" in v for v in audit_colliding["violations"])


def test_independent_canary_never_described_human(tmp_path: Path) -> None:
    # 1. Dataset claiming human-written provenance is strictly rejected
    human_claimed_file = tmp_path / "claims_human.jsonl"
    rec = {
        "text": "Tumpukan sampah liar di pasar",
        "category": "WASTE",
        "provenance": PROVENANCE_SYNTHETIC_INDEPENDENT,
        "is_human_written": True,
    }
    human_claimed_file.write_text(json.dumps(rec) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="claims to be human-written"):
        load_independent_held_out_dataset(human_claimed_file)

    # 2. Valid canary report has is_human_written False everywhere
    valid_file = tmp_path / "valid_canary.jsonl"
    rec_valid = {
        "text": "Tumpukan sampah liar di pasar",
        "intent": "COMPLAINT",
        "category": "WASTE",
        "risk": "HIGH",
        "completeness": "SUFFICIENT",
        "provenance": PROVENANCE_SYNTHETIC_INDEPENDENT,
        "family_id": "valid-fam-01",
    }
    valid_file.write_text(json.dumps(rec_valid) + "\n", encoding="utf-8")
    cfg = EvaluationConfig(
        independent_held_out_path=str(valid_file),
        output_path=str(tmp_path / "rep_human_check.json"),
        dry_run=True,
        num_scenarios=12,
    )
    report = run_evaluate_m3(cfg)
    assert report.synthetic_independent_evaluation["is_human_written"] is False
    assert report.synthetic_independent_evaluation["human_written"] is False
    assert report.synthetic_independent_evaluation["audit"]["is_human_written"] is False
    assert report.independent_quality_gates["is_human_written"] is False
    assert report.independent_quality_gates["human_written"] is False
    assert "NOT human-written" in report.synthetic_independent_evaluation["disclaimer"]


def test_m3_evaluation_report_backward_compatibility_sync() -> None:
    # 1. Old-format dict with only quality_gates and quality_status
    legacy_data = {
        "evaluation_version": "1.0.0",
        "seed": 42,
        "status": "PASSED",
        "pipeline_status": "PASSED",
        "quality_status": "PASSED",
        "quality_gates": {
            "passed": True,
            "status": "PASSED",
            "synthetic_evaluation": True,
            "failed_gates": [],
            "passed_gates": ["intent_macro_f1: 0.95 >= 0.90"],
            "gate_details": {},
        },
        "split_audit": {"passed": True},
        "metrics": {"macro_f1_multitask": 0.91},
        "sla_conformance": True,
        "all_checks_passed": True,
    }

    rep = M3EvaluationReport.model_validate(legacy_data)
    assert rep.quality_status == "PASSED"
    assert rep.quality_gates["passed"] is True
    # Synchronized into new internal fields
    assert rep.internal_quality_status == "PASSED"
    assert rep.internal_status == "PASSED"
    assert rep.internal_quality_gates["passed"] is True
    # Independent canary is None
    assert rep.independent_quality_gates is None
    assert rep.independent_quality_status is None
    assert rep.independent_status is None
    assert rep.independent_canary_status is None

    # 2. Backward compatibility when synthetic_independent_evaluation is present
    legacy_with_indep = dict(legacy_data)
    legacy_with_indep["synthetic_independent_evaluation"] = {
        "status": "PASSED",
        "quality_status": "PASSED",
        "quality_gates": {"passed": True, "status": "PASSED"},
        "is_human_written": False,
    }
    rep_indep = M3EvaluationReport.model_validate(legacy_with_indep)
    assert rep_indep.independent_quality_gates == {"passed": True, "status": "PASSED"}
    assert rep_indep.independent_quality_status == "PASSED"
    assert rep_indep.independent_status == "PASSED"
    assert rep_indep.independent_canary_status == "PASSED"
    assert rep_indep.internal_proxy_quality["status"] == "PASSED"
    assert rep_indep.internal_proxy_quality["is_human_written"] is False
    assert rep_indep.internal_proxy_quality["is_real_world"] is False
    assert rep_indep.ood_proxy_robustness is not None
    assert rep_indep.ood_proxy_robustness["status"] == "PASSED"
    assert rep_indep.ood_proxy_robustness["is_human_written"] is False
    assert rep_indep.ood_proxy_robustness["is_real_world"] is False


# ---------------------------------------------------------------------------
# Tests for Hifi Dataset Manifests, OOD Canary Paths, Zero-Contamination,
# and Separate Reporting of Internal Proxy Quality & OOD Proxy Robustness
# ---------------------------------------------------------------------------

def test_load_and_validate_hifi_manifest_heldout_dataset_format(tmp_path: Path) -> None:
    canary_file = tmp_path / "canary_ood.jsonl"
    recs = [
        {
            "text": f"Laporan kerusakan infrastruktur nomor {i} dari canary terpisah",
            "intent": "COMPLAINT",
            "category": "ROAD",
            "risk": "HIGH",
            "completeness": "SUFFICIENT",
            "provenance": PROVENANCE_SYNTHETIC_INDEPENDENT,
            "family_id": f"fam-canary-{i:03d}",
        }
        for i in range(5)
    ]
    with canary_file.open("w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")

    file_sha = compute_file_sha256(canary_file)
    item = HeldoutManifestItem(
        name="canary_ood",
        split="test",
        path="canary_ood.jsonl",
        sha256=file_sha,
        expected_count=5,
    )
    manifest = HeldoutDatasetManifest(datasets={"canary_ood": item})

    audit, resolved_paths = load_and_validate_hifi_manifest(manifest, base_dir=tmp_path)
    assert audit["passed"] is True
    assert audit["is_human_written"] is False
    assert audit["human_written"] is False
    assert audit["is_real_world"] is False
    assert audit["real_world"] is False
    assert len(resolved_paths) == 1
    assert resolved_paths[0] == canary_file.resolve()
    assert audit["items_checked"]["canary_ood"]["sha256_matches"] is True
    assert audit["items_checked"]["canary_ood"]["count_matches"] is True


def test_load_and_validate_hifi_manifest_artifact_format_file(tmp_path: Path) -> None:
    canary_file = tmp_path / "test_canary.jsonl"
    recs = [
        {
            "text": "Saluran air meluap di perumahan warga",
            "intent": "COMPLAINT",
            "category": "DRAINAGE_FLOOD",
            "risk": "MEDIUM",
            "completeness": "SUFFICIENT",
            "provenance": PROVENANCE_SYNTHETIC_INDEPENDENT,
            "family_id": "fam-canary-01",
        }
    ]
    with canary_file.open("w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")

    file_sha = compute_file_sha256(canary_file)
    manifest_dict = {
        "manifest_version": "1.0.0",
        "artifacts": {
            "test_canary": {
                "name": "test_canary",
                "path": "test_canary.jsonl",
                "sha256": file_sha,
                "task": "holdout",
                "split": "test",
                "expected_count": 1,
            }
        },
    }
    manifest_file = tmp_path / "manifest.json"
    manifest_file.write_text(json.dumps(manifest_dict), encoding="utf-8")

    audit, resolved_paths = load_and_validate_hifi_manifest(manifest_file)
    assert audit["passed"] is True
    assert len(resolved_paths) == 1
    assert resolved_paths[0] == canary_file.resolve()
    assert audit["is_human_written"] is False
    assert audit["is_real_world"] is False


def test_hifi_manifest_rejects_human_written_or_real_world_claims(tmp_path: Path) -> None:
    # 1. Manifest level claim
    bad_manifest = tmp_path / "bad_manifest.json"
    bad_manifest.write_text(json.dumps({
        "manifest_version": "1.0.0",
        "real_world": True,
        "datasets": {"item": {"name": "item", "path": "item.jsonl", "sha256": "0" * 64}},
    }), encoding="utf-8")

    with pytest.raises(ValueError, match="claims to be human-written or real-world"):
        load_and_validate_hifi_manifest(bad_manifest)

    # 2. Item level claim
    dummy_file = tmp_path / "dummy.jsonl"
    dummy_file.write_text("{}\n", encoding="utf-8")
    bad_item_manifest = tmp_path / "bad_item_manifest.json"
    bad_item_manifest.write_text(json.dumps({
        "manifest_version": "1.0.0",
        "datasets": {
            "item": {
                "name": "item",
                "path": "dummy.jsonl",
                "sha256": compute_file_sha256(dummy_file),
                "is_human_written": True,
            }
        },
    }), encoding="utf-8")

    with pytest.raises(ValueError, match="claims to be human-written or real-world"):
        load_and_validate_hifi_manifest(bad_item_manifest)


def test_hifi_manifest_rejects_train_dev_calibration_splits(tmp_path: Path) -> None:
    forbidden_splits = ["train", "dev", "val", "calibration"]
    for split_val in forbidden_splits:
        manifest_dict = {
            "manifest_version": "1.0.0",
            "is_canary": True,
            "datasets": {
                "bad_split_item": {
                    "name": "bad_split_item",
                    "path": "test.jsonl",
                    "sha256": "0" * 64,
                    "split": split_val,
                }
            },
        }
        with pytest.raises(ValueError, match="forbidden training/calibration split"):
            load_and_validate_hifi_manifest(manifest_dict, base_dir=tmp_path, allow_internal_splits=False)


def test_hifi_internal_manifest_allows_train_dev_test_splits(tmp_path: Path) -> None:
    for split_val in ["train", "dev", "test"]:
        split_file = tmp_path / f"{split_val}.jsonl"
        split_file.write_text(json.dumps({
            "text": f"Laporan uji internal split {split_val}",
            "intent": "COMPLAINT",
            "category": "ROAD",
            "risk": "MEDIUM",
            "completeness": "SUFFICIENT",
            "provenance": PROVENANCE_HIFI_SYNTHETIC,
        }) + "\n", encoding="utf-8")
        manifest_dict = {
            "manifest_version": "1.0.0",
            "human_written": False,
            "real_world": False,
            "provenance": PROVENANCE_HIFI_SYNTHETIC,
            "datasets": {
                f"{split_val}_dataset": {
                    "name": f"{split_val}_dataset",
                    "path": f"{split_val}.jsonl",
                    "sha256": compute_file_sha256(split_file),
                    "expected_count": 1,
                    "split": split_val,
                }
            },
        }
        audit, resolved = load_and_validate_hifi_manifest(manifest_dict, base_dir=tmp_path, allow_internal_splits=True)
        assert audit["passed"] is True
        assert len(resolved) == 1


def test_strict_zero_contamination_detects_forbidden_split_in_canary(tmp_path: Path) -> None:
    canary_file = tmp_path / "forbidden_split.jsonl"
    recs = [
        {
            "text": "Laporan pengaduan warga dari sistem canary",
            "intent": "COMPLAINT",
            "category": "ROAD",
            "risk": "HIGH",
            "completeness": "SUFFICIENT",
            "provenance": PROVENANCE_SYNTHETIC_INDEPENDENT,
            "family_id": "fam-canary-clean-01",
            "split": "train",
        }
    ]
    canary_file.write_text(json.dumps(recs[0]) + "\n", encoding="utf-8")
    _, _, raw_recs, _ = load_independent_held_out_dataset(canary_file)

    audit = audit_independent_held_out(
        independent_records=raw_recs,
        independent_path=canary_file,
        training_trajectories=[],
    )
    assert audit["passed"] is False
    assert audit["zero_contamination"] is False
    assert audit["not_used_for_training"] is False
    assert audit["split_contamination_count"] == 1
    assert any("Split contamination" in v for v in audit["violations"])


def test_strict_zero_contamination_detects_scenario_id_collision(tmp_path: Path) -> None:
    colliding_sid = "sc-train-scenario-001"
    train_traj = ComplaintTrajectory(
        scenario_id=colliding_sid,
        family_id="fam-train-01",
        split=DatasetSplit.TRAIN,
        category=Category.ROAD,
        turns=(
            TrajectoryTurn(
                turn=1,
                bubbles=(TrajectoryBubble(source_message_id="b1", text="Jalan ambrol"),),
                expected_action=TurnExpectedAction(turn=1, allowed_actions=(DecisionMode.EXECUTE,), missing=()),
            ),
        ),
    )

    canary_file = tmp_path / "scenario_colliding.jsonl"
    rec = {
        "scenario_id": colliding_sid,
        "text": "Aduan jalan dari canary terpisah tanpa kesamaan teks",
        "intent": "COMPLAINT",
        "category": "ROAD",
        "risk": "LOW",
        "completeness": "SUFFICIENT",
        "provenance": PROVENANCE_SYNTHETIC_INDEPENDENT,
        "family_id": "fam-canary-distinct-01",
    }
    canary_file.write_text(json.dumps(rec) + "\n", encoding="utf-8")
    _, _, raw_recs, _ = load_independent_held_out_dataset(canary_file)

    audit = audit_independent_held_out(
        independent_records=raw_recs,
        independent_path=canary_file,
        training_trajectories=[train_traj],
    )
    assert audit["passed"] is False
    assert audit["zero_contamination"] is False
    assert audit["scenario_overlap_count"] == 1
    assert any("Scenario collision" in v for v in audit["violations"])


def test_strict_zero_contamination_detects_normalized_text_overlap(tmp_path: Path) -> None:
    train_traj = ComplaintTrajectory(
        scenario_id="sc-train-02",
        family_id="fam-train-02",
        split=DatasetSplit.TRAIN,
        category=Category.WASTE,
        turns=(
            TrajectoryTurn(
                turn=1,
                bubbles=(TrajectoryBubble(source_message_id="b1", text="Tumpukan sampah liar menumpuk di pasar induk"),),
                expected_action=TurnExpectedAction(turn=1, allowed_actions=(DecisionMode.EXECUTE,), missing=()),
            ),
        ),
    )

    canary_file = tmp_path / "normalized_overlap.jsonl"
    rec = {
        "text": "  Tumpukan   sampah  liar   menumpuk   di  pasar   induk  ",
        "intent": "COMPLAINT",
        "category": "WASTE",
        "risk": "MEDIUM",
        "completeness": "SUFFICIENT",
        "provenance": PROVENANCE_SYNTHETIC_INDEPENDENT,
        "family_id": "fam-canary-norm-01",
    }
    canary_file.write_text(json.dumps(rec) + "\n", encoding="utf-8")
    _, _, raw_recs, _ = load_independent_held_out_dataset(canary_file)

    audit = audit_independent_held_out(
        independent_records=raw_recs,
        independent_path=canary_file,
        training_trajectories=[train_traj],
    )
    assert audit["passed"] is False
    assert audit["zero_contamination"] is False
    assert any("text overlap" in v.lower() for v in audit["violations"])


def test_strict_zero_contamination_detects_scenario_id_leakage_in_bubble(tmp_path: Path) -> None:
    leaked_sid = "sc-dev-critical-099"
    dev_traj = ComplaintTrajectory(
        scenario_id=leaked_sid,
        family_id="fam-dev-09",
        split=DatasetSplit.DEV,
        category=Category.DRAINAGE_FLOOD,
        turns=(
            TrajectoryTurn(
                turn=1,
                bubbles=(TrajectoryBubble(source_message_id="b1", text="Drainase utama tersumbat"),),
                expected_action=TurnExpectedAction(turn=1, allowed_actions=(DecisionMode.EXECUTE,), missing=()),
            ),
        ),
    )

    canary_file = tmp_path / "sid_leakage.jsonl"
    rec = {
        "text": f"Laporan pengaduan mengacu pada referensi internal {leaked_sid} untuk penanganan segera",
        "intent": "COMPLAINT",
        "category": "DRAINAGE_FLOOD",
        "risk": "HIGH",
        "completeness": "SUFFICIENT",
        "provenance": PROVENANCE_SYNTHETIC_INDEPENDENT,
        "family_id": "fam-canary-leak-01",
    }
    canary_file.write_text(json.dumps(rec) + "\n", encoding="utf-8")
    _, _, raw_recs, _ = load_independent_held_out_dataset(canary_file)

    audit = audit_independent_held_out(
        independent_records=raw_recs,
        independent_path=canary_file,
        training_trajectories=[dev_traj],
    )
    assert audit["passed"] is False
    assert audit["zero_contamination"] is False
    assert any("Scenario ID leakage" in v for v in audit["violations"])


def test_strict_zero_contamination_detects_ngram_contamination(tmp_path: Path) -> None:
    train_traj = ComplaintTrajectory(
        scenario_id="sc-train-ngram",
        family_id="fam-train-ngram",
        split=DatasetSplit.TRAIN,
        category=Category.ROAD,
        turns=(
            TrajectoryTurn(
                turn=1,
                bubbles=(TrajectoryBubble(source_message_id="b1", text="gorong-gorong ambrol menyumbat saluran kaliurang kilometer tujuh"),),
                expected_action=TurnExpectedAction(turn=1, allowed_actions=(DecisionMode.EXECUTE,), missing=()),
            ),
        ),
    )

    canary_file = tmp_path / "ngram_leaked.jsonl"
    rec = {
        "text": "Warga melaporkan gorong-gorong ambrol menyumbat saluran kaliurang kilometer di lokasi lain",
        "intent": "COMPLAINT",
        "category": "ROAD",
        "risk": "HIGH",
        "completeness": "SUFFICIENT",
        "provenance": PROVENANCE_SYNTHETIC_INDEPENDENT,
        "family_id": "fam-canary-distinct-ngram",
    }
    canary_file.write_text(json.dumps(rec) + "\n", encoding="utf-8")
    _, _, raw_recs, _ = load_independent_held_out_dataset(canary_file)

    audit = audit_independent_held_out(
        independent_records=raw_recs,
        independent_path=canary_file,
        training_trajectories=[train_traj],
    )
    assert audit["passed"] is False
    assert audit["zero_contamination"] is False
    assert audit["ngram_overlap_count"] >= 1
    assert any("N-gram contamination" in v for v in audit["violations"])


def test_separate_reporting_of_internal_proxy_quality_and_ood_proxy_robustness(tmp_path: Path) -> None:
    canary_file = tmp_path / "canary_separate.jsonl"
    recs = [
        {
            "text": f"Laporan proxy canary independen nomor {i}",
            "intent": "COMPLAINT",
            "category": "ROAD",
            "risk": "HIGH",
            "completeness": "SUFFICIENT",
            "spans": [[0, 20, "OBJECT"]],
            "provenance": PROVENANCE_SYNTHETIC_INDEPENDENT,
            "family_id": f"fam-canary-sep-{i:03d}",
        }
        for i in range(1, 11)
    ]
    with canary_file.open("w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")

    manifest_dict = {
        "manifest_version": "1.0.0",
        "datasets": {
            "canary_dataset": {
                "name": "canary_dataset",
                "path": "canary_separate.jsonl",
                "sha256": compute_file_sha256(canary_file),
                "expected_count": 10,
                "split": "test",
            }
        },
    }
    manifest_file = tmp_path / "hifi_manifest.json"
    manifest_file.write_text(json.dumps(manifest_dict), encoding="utf-8")

    out_file = tmp_path / "eval_out_separate.json"
    cfg = EvaluationConfig(
        hifi_manifest_path=str(manifest_file),
        output_path=str(out_file),
        dry_run=True,
        num_scenarios=12,
    )
    report = run_evaluate_m3(cfg)

    # 1. Report contains separate internal_proxy_quality
    assert report.internal_proxy_quality is not None
    ipq = report.internal_proxy_quality
    assert ipq["status"] == "PASSED"
    assert ipq["quality_status"] == "PASSED"
    assert ipq["evaluation_type"] == "internal_proxy_quality"
    assert ipq["proxy_type"] == "internal_proxy_synthetic"
    assert ipq["is_human_written"] is False
    assert ipq["human_written"] is False
    assert ipq["is_real_world"] is False
    assert ipq["real_world"] is False
    assert "NOT human-written" in ipq["disclaimer"]
    assert "NOT real-world" in ipq["disclaimer"]

    # 2. Report contains separate ood_proxy_robustness
    assert report.ood_proxy_robustness is not None
    opr = report.ood_proxy_robustness
    assert opr["status"] == "PASSED"
    assert opr["quality_status"] == "PASSED"
    assert opr["canary_status"] == "PASSED"
    assert opr["evaluation_type"] == "ood_proxy_robustness"
    assert opr["proxy_type"] == "ood_canary_proxy_synthetic"
    assert opr["is_human_written"] is False
    assert opr["human_written"] is False
    assert opr["is_real_world"] is False
    assert opr["real_world"] is False
    assert "NOT human-written" in opr["disclaimer"]
    assert "NOT real-world" in opr["disclaimer"]
    assert opr["zero_contamination_verified"] is True
    assert opr["hifi_manifest"]["passed"] is True

    # 3. Overall status and compatibility
    assert report.status == "PASSED"
    assert report.all_checks_passed is True
    assert report.hifi_manifest_audit is not None
    assert report.hifi_manifest_audit["passed"] is True


def test_ood_canary_path_and_canary_path_config_aliases(tmp_path: Path) -> None:
    canary_file = tmp_path / "canary_alias.jsonl"
    recs = [
        {
            "text": f"Laporan alias canary nomor {i}",
            "intent": "COMPLAINT",
            "category": "WASTE",
            "risk": "MEDIUM",
            "completeness": "SUFFICIENT",
            "spans": [[0, 15, "OBJECT"]],
            "provenance": PROVENANCE_SYNTHETIC_INDEPENDENT,
            "family_id": f"fam-canary-alias-{i:03d}",
        }
        for i in range(1, 11)
    ]
    with canary_file.open("w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")

    out_file = tmp_path / "eval_alias.json"
    cfg = EvaluationConfig(
        ood_canary_path=str(canary_file),
        output_path=str(out_file),
        dry_run=True,
        num_scenarios=12,
    )
    report = run_evaluate_m3(cfg)
    assert report.status == "PASSED"
    assert report.ood_proxy_robustness is not None
    assert report.ood_proxy_robustness["status"] == "PASSED"


def test_cli_parser_supports_hifi_manifest_and_ood_canary_flags(tmp_path: Path) -> None:
    parser = build_arg_parser()
    args = parser.parse_args([
        "--hifi-manifest-path", "artifacts/hifi/manifest.json",
        "--ood-canary-path", "artifacts/hifi/canary.jsonl",
        "--dry-run",
    ])
    assert args.hifi_manifest_path == "artifacts/hifi/manifest.json"
    assert args.independent_held_out_path == "artifacts/hifi/canary.jsonl"
    assert args.dry_run is True


def test_validate_only_mode_with_hifi_manifest_and_canary(tmp_path: Path) -> None:
    canary_file = tmp_path / "val_canary.jsonl"
    recs = [
        {
            "text": "Laporan pengaduan validasi canary jalan berlubang",
            "intent": "COMPLAINT",
            "category": "ROAD",
            "risk": "HIGH",
            "completeness": "SUFFICIENT",
            "provenance": PROVENANCE_SYNTHETIC_INDEPENDENT,
            "family_id": "fam-val-canary-01",
        }
    ]
    canary_file.write_text(json.dumps(recs[0]) + "\n", encoding="utf-8")

    manifest_dict = {
        "manifest_version": "1.0.0",
        "datasets": {
            "canary": {
                "name": "canary",
                "path": "val_canary.jsonl",
                "sha256": compute_file_sha256(canary_file),
                "expected_count": 1,
            }
        },
    }
    manifest_file = tmp_path / "manifest.json"
    manifest_file.write_text(json.dumps(manifest_dict), encoding="utf-8")

    out_file = tmp_path / "eval_val_only.json"
    cfg = EvaluationConfig(
        hifi_manifest_path=str(manifest_file),
        output_path=str(out_file),
        validate_only=True,
        num_scenarios=12,
    )
    report = run_evaluate_m3(cfg)
    assert report.status == "PASSED"
    assert report.pipeline_status == "PASSED"
    assert report.internal_proxy_quality["status"] == "SKIPPED"
    assert report.internal_proxy_quality["is_human_written"] is False
    assert report.internal_proxy_quality["is_real_world"] is False
    assert report.ood_proxy_robustness is not None
    assert report.ood_proxy_robustness["status"] == "PASSED"
    assert report.ood_proxy_robustness["quality_status"] == "SKIPPED"
    assert report.ood_proxy_robustness["is_human_written"] is False
    assert report.ood_proxy_robustness["is_real_world"] is False


def test_hifi_internal_provenance_rejected_as_ood_canary(tmp_path: Path) -> None:
    canary_file = tmp_path / "bad_hifi_canary.jsonl"
    canary_file.write_text(
        json.dumps({
            "text": "Laporan jalan rusak internal",
            "category": "ROAD",
            "intent": "COMPLAINT",
            "provenance": PROVENANCE_HIFI_SYNTHETIC,
        }) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="HiFi internal train/dev/test records are allowed only as internal synthetic proxy data"):
        load_independent_held_out_dataset(canary_file)


def test_hifi_internal_provenance_in_audit_flags_violation(tmp_path: Path) -> None:
    canary_file = tmp_path / "hifi_audit_canary.jsonl"
    rec = {
        "text": "Laporan gorong-gorong hifi internal",
        "category": "DRAINAGE_FLOOD",
        "intent": "COMPLAINT",
        "provenance": PROVENANCE_HIFI_SYNTHETIC,
        "split": "test",
    }
    canary_file.write_text(json.dumps(rec) + "\n", encoding="utf-8")
    audit = audit_independent_held_out(
        independent_records=[rec],
        independent_path=canary_file,
        training_trajectories=[],
    )
    assert audit["passed"] is False
    assert any("HiFi internal train/dev/test records are allowed only as internal synthetic proxy data" in v for v in audit["violations"])


def test_ood_canary_strictly_requires_synthetic_independent(tmp_path: Path) -> None:
    canary_file = tmp_path / "non_independent_canary.jsonl"
    canary_file.write_text(
        json.dumps({
            "text": "Laporan pengaduan warga ood",
            "category": "ROAD",
            "intent": "COMPLAINT",
            "provenance": "external_vendor_synthetic",
        }) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Expected 'synthetic_independent'"):
        load_independent_held_out_dataset(canary_file)


def test_ood_canary_cannot_enter_calibration_or_training_via_provenance_leak(tmp_path: Path) -> None:
    canary_file = tmp_path / "canary_clean.jsonl"
    canary_file.write_text(
        json.dumps({
            "text": "Laporan ood canary bersih unik",
            "category": "ROAD",
            "intent": "COMPLAINT",
            "provenance": PROVENANCE_SYNTHETIC_INDEPENDENT,
            "family_id": "fam-canary-uniq-99",
            "split": "test",
        }) + "\n",
        encoding="utf-8",
    )
    _, _, raw_recs, _ = load_independent_held_out_dataset(canary_file)

    leaked_train_traj = {
        "scenario_id": "sc-leaked-train-01",
        "family_id": "fam-train-01",
        "split": "train",
        "category": "ROAD",
        "world_truth": {"provenance": PROVENANCE_SYNTHETIC_INDEPENDENT},
        "turns": [{"turn": 1, "bubbles": [{"text": "Pesan pelatihan"}]}],
    }

    audit = audit_independent_held_out(
        independent_records=raw_recs,
        independent_path=canary_file,
        training_trajectories=[leaked_train_traj],
    )
    assert audit["passed"] is False
    assert audit["not_used_for_training"] is False
    assert any("OOD canary cannot enter calibration/training" in v for v in audit["violations"])


def test_root_report_explicitly_separates_internal_and_ood(tmp_path: Path) -> None:
    canary_file = tmp_path / "canary_sep.jsonl"
    canary_file.write_text(
        json.dumps({
            "text": "Aduan jalan berlubang di seberang stasiun",
            "category": "ROAD",
            "intent": "COMPLAINT",
            "provenance": PROVENANCE_SYNTHETIC_INDEPENDENT,
            "family_id": "fam-canary-sep-01",
            "split": "test",
        }) + "\n",
        encoding="utf-8",
    )

    out_file = tmp_path / "eval_out_sep.json"
    cfg = EvaluationConfig(
        ood_canary_path=str(canary_file),
        output_path=str(out_file),
        dry_run=True,
        num_scenarios=12,
    )
    report = run_evaluate_m3(cfg)

    assert report.internal_proxy_quality is not None
    assert report.internal_evaluation is not None
    assert report.internal_evaluation["status"] == report.internal_quality_status
    assert report.internal_evaluation["proxy_type"] == "internal_proxy_synthetic"

    assert report.ood_proxy_robustness is not None
    assert report.ood_evaluation is not None
    assert report.ood_evaluation["status"] == report.independent_status
    assert report.ood_evaluation["proxy_type"] == "ood_canary_proxy_synthetic"

    assert "internal" in report.metrics
    assert "ood" in report.metrics
    assert report.metrics["internal"]["proxy_type"] == "internal_proxy_synthetic"
    assert report.metrics["ood"]["proxy_type"] == "ood_canary_proxy_synthetic"


def test_ner_eval_reads_and_prioritizes_canonical_spans(tmp_path: Path) -> None:
    canary_file = tmp_path / "canary_with_canonical_spans.jsonl"
    traj_record = {
        "scenario_id": "sc-canonical-01",
        "family_id": "fam-canonical-01",
        "category": "ROAD",
        "split": "test",
        "provenance": PROVENANCE_SYNTHETIC_INDEPENDENT,
        "turns": [
            {
                "turn": 1,
                "bubbles": [
                    {
                        "source_message_id": "msg-can-01",
                        "text": "Aspal ambles parah di Jalan Kaliurang KM 12",
                        "canonical_spans": [
                            {"start": 0, "end": 12, "label": "OBJ"},
                            {"start": 22, "end": 44, "label": "LOC"},
                        ],
                    }
                ],
            }
        ],
    }
    canary_file.write_text(json.dumps(traj_record) + "\n", encoding="utf-8")

    _, ner_samples, _, _ = load_independent_held_out_dataset(canary_file)
    assert len(ner_samples) >= 1
    bubble_sample = ner_samples[0]
    assert bubble_sample["text"] == "Aspal ambles parah di Jalan Kaliurang KM 12"
    assert (0, 12, "OBJ") in bubble_sample["spans"]
    assert (22, 44, "LOC") in bubble_sample["spans"]


def test_evaluate_ner_model_uses_canonical_spans() -> None:
    sample = {
        "text": "Lampu jalan mati total di Cibadak",
        "canonical_spans": [
            (0, 16, "OBJ"),
            (25, 32, "LOC"),
        ],
    }

    def mock_model(text: str) -> list[tuple[int, int, str]]:
        return [(0, 16, "OBJ"), (25, 32, "LOC")]

    res = evaluate_ner_model(
        model=mock_model,
        samples=[sample],
        tokenizer=None,
        max_seq_length=128,
    )
    assert res["evaluated"] is True
    assert res["entity_f1"] == 1.0
    assert res["entity_precision"] == 1.0
    assert res["entity_recall"] == 1.0


def test_evaluate_ner_model_empty_canonical_spans_no_fallback() -> None:
    """Issue 1 & 2: Present but empty canonical spans should evaluate accurately without regex fallback."""
    sample = {
        "text": "Jalan rusak parah di Kaliurang",
        "canonical_spans": [],
    }

    # Model predicts nothing, ground truth is empty -> precision/recall/f1 = 1.0
    res = evaluate_ner_model(
        model=lambda t: [],
        samples=[sample],
        tokenizer=None,
        max_seq_length=128,
    )
    assert res["evaluated"] is True
    assert res["total_entities_true"] == 0
    assert res["total_entities_pred"] == 0
    assert res["entity_f1"] == 1.0


def test_evaluate_ner_model_consistent_granularity_filter() -> None:
    """Issue 5: sample granularity (bubble vs full) must be consistent between trainer and evaluator."""
    b1_sample = {
        "text": "Pipa bocor",
        "spans": [(0, 10, "OBJ")],
        "canonical_spans": [(0, 10, "OBJ")],
        "granularity": "bubble",
    }
    full_sample = {
        "text": "Pipa bocor\nDi jalan Malioboro",
        "spans": [(0, 10, "OBJ"), (14, 29, "LOC")],
        "canonical_spans": [(0, 10, "OBJ"), (14, 29, "LOC")],
        "granularity": "full",
    }

    res_all = evaluate_ner_model(
        model=lambda t: [],
        samples=[b1_sample, full_sample],
        tokenizer=None,
    )
    assert res_all["sample_count"] == 2

    res_bubble = evaluate_ner_model(
        model=lambda t: [],
        samples=[b1_sample, full_sample],
        tokenizer=None,
        granularity="bubble",
    )
    assert res_bubble["sample_count"] == 1
    assert res_bubble.get("granularity") == "bubble"

    res_full = evaluate_ner_model(
        model=lambda t: [],
        samples=[b1_sample, full_sample],
        tokenizer=None,
        granularity="full",
    )
    assert res_full["sample_count"] == 1
    assert res_full.get("granularity") == "full"


def test_evaluate_ner_model_subword_exact_char_span_alignment() -> None:
    """Issue 4: NER evaluator must align subword predictions to exact char spans consistently."""
    # Text with punctuation boundary: "aspal ambles," -> exact char span should be (0, 12, "OBJ")
    text = "aspal ambles, segera perbaiki"
    samples = [
        {
            "text": text,
            "spans": [(0, 12, "OBJ")],
        }
    ]
    # Subwords:
    # "aspal" (0, 5)
    # "ambles" (6, 12)
    # "," (12, 13) - punctuation included by coarse subword chunking
    subword_offsets = [
        (0, 5),
        (6, 13),  # Includes comma
        (14, 20),
        (21, 29),
    ]
    tag2id = {t: i for i, t in enumerate(DEFAULT_TAGSET)}
    target_tag_ids = [tag2id["B-OBJ"], tag2id["I-OBJ"], tag2id["O"], tag2id["O"]]

    class MockPunctTokenizer:
        def __call__(self, txt: str, **kwargs: Any) -> Any:
            seq_len = len(subword_offsets)
            return {
                "input_ids": [[101] * seq_len],
                "attention_mask": [[1] * seq_len],
                "offset_mapping": [subword_offsets],
            }

    class MockONNX:
        def get_inputs(self) -> list[Any]:
            m1 = mock.Mock()
            m1.name = "input_ids"
            m2 = mock.Mock()
            m2.name = "attention_mask"
            return [m1, m2]

        def run(self, output_names: Any, feed_dict: dict[str, Any]) -> list[Any]:
            seq_len = len(subword_offsets)
            num_tags = len(DEFAULT_TAGSET)
            logits_seq = []
            for t_id in target_tag_ids:
                step = [0.0] * num_tags
                step[t_id] = 10.0
                logits_seq.append(step)
            return [[logits_seq]]

    res = evaluate_ner_model(
        model=MockONNX(),
        samples=samples,
        tokenizer=MockPunctTokenizer(),
        tagset=DEFAULT_TAGSET,
    )
    assert res["evaluated"] is True
    # The comma at index 12 is cleanly stripped so the predicted entity matches (0, 12, 'OBJ') exactly
    assert res["entity_precision"] == 1.0
    assert res["entity_recall"] == 1.0
    assert res["entity_f1"] == 1.0


def test_internal_dataset_with_canonical_spans_evaluated_in_pipeline(tmp_path: Path) -> None:
    dataset_file = tmp_path / "internal_with_canonical.jsonl"
    records = [
        {
            "scenario_id": f"sc-int-{i:03d}",
            "family_id": f"fam-int-{i:03d}",
            "category": "ROAD",
            "split": "test",
            "provenance": PROVENANCE_HIFI_SYNTHETIC,
            "turns": [
                {
                    "turn": 1,
                    "bubbles": [
                        {
                            "source_message_id": f"msg-int-{i}",
                            "text": "Saluran drainase tersumbat di Kelurahan Sukajadi",
                            "canonical_spans": [
                                [0, 16, "OBJ"],
                                [29, 48, "LOC"],
                            ],
                        }
                    ],
                }
            ],
        }
        for i in range(6)
    ]
    with dataset_file.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    out_file = tmp_path / "eval_out_internal_canonical.json"
    cfg = EvaluationConfig(
        dataset_path=str(dataset_file),
        output_path=str(out_file),
        dry_run=True,
        num_scenarios=6,
    )
    report = run_evaluate_m3(cfg)
    assert report.status == "PASSED"
    assert report.internal_proxy_quality["status"] == "PASSED"
    assert report.internal_evaluation["proxy_type"] == "internal_proxy_synthetic"


