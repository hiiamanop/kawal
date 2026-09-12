from __future__ import annotations

import math
from datetime import datetime, timezone

import pytest

from services.intelligence.chunking import (
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
from services.intelligence.retrieval import (
    BM25Retriever,
    IndexedDocument,
    compute_lexical_embedding,
    cosine_similarity,
    rerank_candidates,
)
from services.intelligence.calibration import (
    CpuBenchmarkConfig,
    CpuBenchmarkResult,
    MultiHeadTemperatureCalibrator,
    TemperatureCalibrator,
    apply_temperature,
    apply_temperature_batch,
    compute_ece,
    discover_temperatures_path,
    fit_multitask_temperatures,
    fit_temperature,
    load_temperatures,
    run_cpu_benchmark,
    save_temperatures,
)


def test_chunking_window_and_overlap() -> None:
    words = [f"kata_{i}" for i in range(1000)]
    full_text = " ".join(words)

    tokens = tokenize_with_offsets(full_text)
    assert len(tokens) == 1000

    chunks = chunk_text(full_text, max_tokens=448, overlap=64)

    assert len(chunks) == 3
    assert chunks[0].chunk_index == 0
    assert chunks[0].total_chunks == 3
    assert chunks[0].token_count == 448
    assert chunks[0].start_token == 0
    assert chunks[0].end_token == 448

    assert chunks[1].chunk_index == 1
    assert chunks[1].token_count == 448
    assert chunks[1].start_token == 384
    assert chunks[1].end_token == 832

    assert chunks[2].chunk_index == 2
    assert chunks[2].start_token == 768
    assert chunks[2].end_token == 1000
    assert chunks[2].token_count == 232

    assert chunks[1].start_token == chunks[0].end_token - 64
    assert chunks[2].start_token == chunks[1].end_token - 64


def test_tail_facts_preserved_in_long_complaints() -> None:
    intro_words = [f"pembukaan_{i}" for i in range(950)]
    intro_text = " ".join(intro_words)
    tail_fact = "Lokasi jalan berlubang di Jl. Magelang KM 14 rusak parah tadi sore."
    long_complaint = f"{intro_text} {tail_fact}"

    chunks = chunk_text(long_complaint, max_tokens=448, overlap=64)

    assert len(chunks) >= 3
    last_chunk = chunks[-1]

    assert "Jl. Magelang KM 14" in last_chunk.text
    assert "rusak parah" in last_chunk.text

    fact_local_offset = last_chunk.text.find("Jl. Magelang KM 14")
    assert fact_local_offset != -1

    global_offset = last_chunk.to_global_char_offset(fact_local_offset)
    assert long_complaint[global_offset : global_offset + len("Jl. Magelang KM 14")] == "Jl. Magelang KM 14"

    local_mapped_back = last_chunk.to_local_char_offset(global_offset)
    assert local_mapped_back == fact_local_offset
    assert last_chunk.contains_global_offset(global_offset)


def test_chunk_messages_preserves_provenance() -> None:
    messages = [
        {"source_message_id": "msg-001", "text": "Halo admin ini laporan pertama saya."},
        {"source_message_id": "msg-002", "text": "Jalan berlubang besar di depan SDN 1 Cibinong."},
        {"source_message_id": "msg-003", "text": "Kejadiannya baru kemarin sore mohon ditindaklanjuti."},
    ]

    chunks = chunk_messages(messages, max_tokens=448, overlap=64, case_id="case-100")
    assert len(chunks) == 1
    chunk = chunks[0]

    assert "msg-001" in chunk.source_message_ids
    assert "msg-002" in chunk.source_message_ids
    assert "msg-003" in chunk.source_message_ids
    assert chunk.source_message_id == "msg-001"


def test_bio_tag_to_span_parser_standard() -> None:
    tokens = [
        "Laporan", "jalan", "berlubang", "di", "Jl.", "Sudirman",
        "No.", "10", "terjadi", "kemarin", "sore",
    ]
    tags = [
        "O", "B-OBJECT", "I-OBJECT", "O", "B-LOCATION", "I-LOCATION",
        "I-LOCATION", "I-LOCATION", "O", "B-TIME", "I-TIME",
    ]
    confs = [0.99] * len(tokens)

    spans = parse_bio_tags(tokens, tags, confidences=confs)

    assert len(spans) == 3

    assert spans[0].label == "OBJECT"
    assert spans[0].text == "jalan berlubang"
    assert spans[0].start_token == 1
    assert spans[0].end_token == 3

    assert spans[1].label == "LOCATION"
    assert spans[1].text == "Jl. Sudirman No. 10"
    assert spans[1].start_token == 4
    assert spans[1].end_token == 8

    assert spans[2].label == "TIME"
    assert spans[2].text == "kemarin sore"
    assert spans[2].start_token == 9
    assert spans[2].end_token == 11


def test_bio_parser_malformed_and_edge_cases() -> None:
    tokens = ["Cipinang", "Jakarta", "aspal", "lubang", "pukul", "15.00"]
    tags = ["I-LOCATION", "I-LOCATION", "B-OBJECT", "B-OBJECT", "B-TIME", "I-TIME"]

    spans = parse_bio_tags(tokens, tags)

    assert len(spans) == 4
    assert spans[0].label == "LOCATION"
    assert spans[0].text == "Cipinang Jakarta"
    assert spans[1].label == "OBJECT"
    assert spans[1].text == "aspal"
    assert spans[2].label == "OBJECT"
    assert spans[2].text == "lubang"
    assert spans[3].label == "TIME"
    assert spans[3].text == "pukul 15.00"


def test_ner_overlapping_spans_merging() -> None:
    span1 = EntitySpan(
        label="LOCATION",
        text="Jl. Merdeka",
        start_token=5,
        end_token=7,
        start_char=20,
        end_char=31,
        confidence=0.90,
        chunk_id="chk-0",
    )
    span2 = EntitySpan(
        label="LOCATION",
        text="Jl. Merdeka No. 5",
        start_token=5,
        end_token=9,
        start_char=20,
        end_char=37,
        confidence=0.95,
        chunk_id="chk-1",
    )
    span3 = EntitySpan(
        label="OBJECT",
        text="lampu mati",
        start_token=10,
        end_token=12,
        start_char=40,
        end_char=50,
        confidence=0.88,
        chunk_id="chk-1",
    )

    merged = merge_overlapping_spans([span1, span2, span3])

    assert len(merged) == 2
    assert merged[0].label == "LOCATION"
    assert merged[0].text == "Jl. Merdeka No. 5"
    assert merged[0].start_char == 20
    assert merged[0].end_char == 37
    assert merged[0].confidence == 0.95

    assert merged[1].label == "OBJECT"
    assert merged[1].text == "lampu mati"


def test_merge_overlapping_spans_union_text_regression() -> None:
    span_a = EntitySpan(
        label="LOCATION",
        text="Jl. Kaliurang",
        start_token=0,
        end_token=2,
        start_char=0,
        end_char=13,
        confidence=0.85,
    )
    span_b = EntitySpan(
        label="LOCATION",
        text="Kaliurang KM 7",
        start_token=1,
        end_token=4,
        start_char=4,
        end_char=18,
        confidence=0.92,
    )

    merged = merge_overlapping_spans([span_a, span_b])
    assert len(merged) == 1
    assert merged[0].label == "LOCATION"
    assert merged[0].start_char == 0
    assert merged[0].end_char == 18
    assert merged[0].text == "Jl. Kaliurang KM 7"
    assert merged[0].confidence == 0.92

    span_c = EntitySpan(
        label="LOCATION",
        text="Kabupaten",
        start_token=5,
        end_token=6,
        start_char=20,
        end_char=29,
        confidence=0.80,
    )
    span_d = EntitySpan(
        label="LOCATION",
        text="Sleman",
        start_token=6,
        end_token=7,
        start_char=30,
        end_char=36,
        confidence=0.88,
    )
    merged_adj = merge_overlapping_spans([span_c, span_d])
    assert len(merged_adj) == 1
    assert merged_adj[0].text == "Kabupaten Sleman"
    assert merged_adj[0].start_char == 20
    assert merged_adj[0].end_char == 36

    span_e = EntitySpan(
        label="OBJECT",
        text="jalan",
        start_token=0,
        end_token=1,
        start_char=0,
        end_char=5,
        confidence=0.75,
    )
    span_f = EntitySpan(
        label="OBJECT",
        text="an",
        start_token=1,
        end_token=2,
        start_char=5,
        end_char=7,
        confidence=0.80,
    )
    merged_touching = merge_overlapping_spans([span_e, span_f])
    assert len(merged_touching) == 1
    assert merged_touching[0].text == "jalanan"
    assert merged_touching[0].start_char == 0
    assert merged_touching[0].end_char == 7

    span_g = EntitySpan(
        label="OBJECT",
        text="jalan raya bogor",
        start_token=0,
        end_token=3,
        start_char=10,
        end_char=26,
        confidence=0.85,
    )
    span_h = EntitySpan(
        label="OBJECT",
        text="bogor timur",
        start_token=2,
        end_token=4,
        start_char=21,
        end_char=32,
        confidence=0.90,
    )
    merged_longer_first = merge_overlapping_spans([span_g, span_h])
    assert len(merged_longer_first) == 1
    assert merged_longer_first[0].text == "jalan raya bogor timur"
    assert merged_longer_first[0].start_char == 10
    assert merged_longer_first[0].end_char == 32

    s1 = EntitySpan(
        label="LOCATION",
        text="Kecamatan",
        start_token=0,
        end_token=1,
        start_char=0,
        end_char=9,
        confidence=0.7,
    )
    s2 = EntitySpan(
        label="LOCATION",
        text="atan Coblong",
        start_token=1,
        end_token=3,
        start_char=5,
        end_char=17,
        confidence=0.8,
    )
    s3 = EntitySpan(
        label="LOCATION",
        text="long Kidul",
        start_token=2,
        end_token=4,
        start_char=13,
        end_char=23,
        confidence=0.9,
    )
    merged_chain = merge_overlapping_spans([s1, s2, s3])
    assert len(merged_chain) == 1
    assert merged_chain[0].text == "Kecamatan Coblong Kidul"
    assert merged_chain[0].start_char == 0
    assert merged_chain[0].end_char == 23

    d1 = EntitySpan(
        label="TIME",
        text="kemarin",
        start_token=0,
        end_token=1,
        start_char=0,
        end_char=7,
        confidence=0.9,
    )
    d2 = EntitySpan(
        label="TIME",
        text="kemarin",
        start_token=0,
        end_token=1,
        start_char=0,
        end_char=7,
        confidence=0.8,
    )
    merged_det = merge_overlapping_spans([d1, d2])
    assert len(merged_det) == 1
    assert merged_det[0].confidence == 0.9


def test_deterministic_text_extraction() -> None:
    text = "Ada jalan berlubang di Jalan Kaliurang KM 7 sejak kemarin sore."
    entities = extract_entities_from_text(text)

    labels = {e.label for e in entities}
    assert "OBJECT" in labels
    assert "LOCATION" in labels
    assert "TIME" in labels

    loc = next(e for e in entities if e.label == "LOCATION")
    assert "Jalan Kaliurang KM 7" in loc.text


def test_lexical_embedding_and_similarity() -> None:
    text1 = "jalan rusak dan berlubang parah"
    text2 = "jalan rusak serta aspal berlubang parah"
    text3 = "lampu penerangan jalan padam di perumahan"

    emb1 = compute_lexical_embedding(text1, dimension=128)
    emb2 = compute_lexical_embedding(text2, dimension=128)
    emb3 = compute_lexical_embedding(text3, dimension=128)

    assert len(emb1.dense_vector) == 128
    norm1 = math.sqrt(sum(v * v for v in emb1.dense_vector))
    assert math.isclose(norm1, 1.0, rel_tol=1e-3)

    sim_self = cosine_similarity(emb1.dense_vector, emb1.dense_vector)
    assert math.isclose(sim_self, 1.0, rel_tol=1e-5)

    sim_related = cosine_similarity(emb1.dense_vector, emb2.dense_vector)
    sim_unrelated = cosine_similarity(emb1.dense_vector, emb3.dense_vector)

    assert sim_related > sim_unrelated


def test_bm25_retrieval_and_reranking() -> None:
    docs = [
        IndexedDocument(
            doc_id="doc-road-1",
            text="Laporan aspal jalan berlubang sangat dalam dan berbahaya bagi pemotor.",
            metadata={"category": "ROAD"},
        ),
        IndexedDocument(
            doc_id="doc-road-2",
            text="Jalanan berdebu di pinggir jalan raya tidak ada perbaikan.",
            metadata={"category": "ROAD"},
        ),
        IndexedDocument(
            doc_id="doc-light-1",
            text="Lampu jalan mati total setiap malam gelap gulita.",
            metadata={"category": "LIGHTING"},
        ),
        IndexedDocument(
            doc_id="doc-waste-1",
            text="Sampah menumpuk berbau busuk di TPS liar.",
            metadata={"category": "WASTE"},
        ),
    ]

    retriever = BM25Retriever()
    retriever.index_documents(docs)

    query = "jalan berlubang"
    results = retriever.retrieve(query, top_k=3)

    assert len(results) >= 2
    assert results[0].doc_id == "doc-road-1"
    assert results[0].lexical_score > 0.0

    reranked = retriever.retrieve_and_rerank(query, top_k=3)
    assert len(reranked) >= 2
    assert reranked[0].doc_id == "doc-road-1"
    assert reranked[0].rank == 1
    assert reranked[0].combined_score >= reranked[1].combined_score


def test_rerank_candidates_non_default_embedding_dimension() -> None:
    non_default_dim = 128
    docs = [
        IndexedDocument(
            doc_id="doc-road-1",
            text="Laporan aspal jalan berlubang sangat dalam dan berbahaya bagi pemotor.",
            metadata={"category": "ROAD"},
        ),
        IndexedDocument(
            doc_id="doc-road-2",
            text="Jalanan berdebu di pinggir jalan raya tidak ada perbaikan.",
            metadata={"category": "ROAD"},
        ),
        IndexedDocument(
            doc_id="doc-waste-1",
            text="Sampah menumpuk berbau busuk di TPS liar.",
            metadata={"category": "WASTE"},
        ),
    ]

    retriever = BM25Retriever(embedding_dim=non_default_dim)
    retriever.index_documents(docs)

    assert retriever.embedding_dim == non_default_dim
    assert retriever._documents["doc-road-1"].embedding is not None
    assert len(retriever._documents["doc-road-1"].embedding.dense_vector) == non_default_dim

    query = "jalan berlubang"
    candidates = retriever.retrieve(query, top_k=2)

    reranked = rerank_candidates(
        query=query,
        candidates=candidates,
        retriever=retriever,
        top_k=2,
        lexical_weight=0.0,
        cosine_weight=1.0,
        coverage_weight=0.0,
        phrase_weight=0.0,
    )

    doc_emb = retriever._documents["doc-road-1"].embedding
    expected_cos = cosine_similarity(
        compute_lexical_embedding(query, dimension=non_default_dim).dense_vector,
        doc_emb.dense_vector,
    )
    assert expected_cos > 0.0
    assert math.isclose(reranked[0].rerank_score, round(expected_cos, 4), rel_tol=1e-4)

    reranked_all = retriever.retrieve_and_rerank(query, top_k=2)
    assert len(reranked_all) == 2
    assert reranked_all[0].doc_id == "doc-road-1"


def test_temperature_calibration_fit_and_apply() -> None:
    import random

    rng = random.Random(42)
    logits_batch: list[list[float]] = []
    labels: list[int] = []

    for i in range(40):
        true_cls = i % 2
        labels.append(true_cls)
        if rng.random() < 0.75:
            logits = [6.0, 0.0] if true_cls == 0 else [0.0, 6.0]
        else:
            logits = [0.0, 6.0] if true_cls == 0 else [6.0, 0.0]
        logits_batch.append(logits)

    uncal_probs = apply_temperature_batch(logits_batch, 1.0)
    for p_dist in uncal_probs:
        assert math.isclose(sum(p_dist), 1.0, rel_tol=1e-5)

    uncal_ece = compute_ece(uncal_probs, labels, n_bins=5)
    assert uncal_ece > 0.15

    calibrator = TemperatureCalibrator()
    optimal_t = calibrator.fit(logits_batch, labels)

    assert optimal_t > 1.0

    cal_probs = calibrator.calibrate_batch(logits_batch)
    for p_dist in cal_probs:
        assert math.isclose(sum(p_dist), 1.0, rel_tol=1e-5)

    cal_ece = compute_ece(cal_probs, labels, n_bins=5)
    assert cal_ece < uncal_ece

    eval_metrics = calibrator.evaluate(logits_batch, labels, n_bins=5)
    assert eval_metrics.calibrated_nll < eval_metrics.uncalibrated_nll
    assert eval_metrics.calibrated_ece < eval_metrics.uncalibrated_ece
    assert eval_metrics.optimal_temperature == optimal_t


def test_cpu_benchmark_measurement_contract() -> None:
    config = CpuBenchmarkConfig(
        batch_size=1,
        sequence_length=448,
        overlap_tokens=64,
        num_threads=1,
        warmup_runs=2,
        benchmark_runs=5,
        target_p95_latency_ms=100.0,
        precision="fp32",
        device="cpu",
    )

    result = run_cpu_benchmark(config)

    assert isinstance(result, CpuBenchmarkResult)
    assert result.config == config
    assert result.samples_count == 5
    assert result.mean_latency_ms >= 0.0
    assert result.p50_latency_ms >= 0.0
    assert result.p95_latency_ms >= result.p50_latency_ms
    assert result.p99_latency_ms >= result.p95_latency_ms
    assert result.min_latency_ms <= result.p50_latency_ms <= result.max_latency_ms
    assert result.throughput_qps >= 0.0
    assert isinstance(result.sla_met, bool)
    assert result.measured_at.tzinfo == timezone.utc


def test_temperature_calibrator_multihead_fit_and_apply_held_logits() -> None:
    """Deterministic verification that multi-head calibration strictly improves ECE and NLL on held logits."""
    import random

    heads_config = {
        "intent": 3,
        "category": 6,
        "risk": 4,
        "completeness": 3,
    }

    def generate_split(seed: int, n_samples: int) -> tuple[dict[str, list[list[float]]], dict[str, list[int]]]:
        rng = random.Random(seed)
        split_logits: dict[str, list[list[float]]] = {}
        split_labels: dict[str, list[int]] = {}

        for idx_h, (head, k) in enumerate(heads_config.items()):
            rng = random.Random(seed + idx_h * 101)
            h_logits: list[list[float]] = []
            h_labels: list[int] = []
            for _ in range(n_samples):
                y = rng.randrange(k)
                h_labels.append(y)
                # Model exhibits overconfidence: ~75% accuracy, large positive logit on predicted class
                is_correct = rng.random() < 0.75
                pred_y = y if is_correct else (y + 1 + rng.randrange(k - 1)) % k
                row = [round(rng.uniform(-0.25, 0.25), 4) for _ in range(k)]
                row[pred_y] += 3.85
                h_logits.append(row)
            split_logits[head] = h_logits
            split_labels[head] = h_labels
        return split_logits, split_labels

    dev_logits, dev_labels = generate_split(seed=42, n_samples=120)
    held_logits, held_labels = generate_split(seed=999, n_samples=120)

    calibrator = TemperatureCalibrator()
    fitted_temps = calibrator.fit_multitask(dev_logits, dev_labels)

    assert set(fitted_temps.keys()) == {"intent", "category", "risk", "completeness"}
    for head, temp in fitted_temps.items():
        assert temp > 1.0, f"Expected overconfident head {head} to require T > 1.0, got {temp}"

    for head in heads_config:
        logits = held_logits[head]
        labels = held_labels[head]
        metrics = calibrator.evaluate(logits, labels, n_bins=5, head=head)

        # Confirm uncalibrated ECE reflects inadequate calibration (> 0.12, approx real ECE .1824)
        assert metrics.uncalibrated_ece > 0.12, f"Expected high uncalibrated ECE on head {head}, got {metrics.uncalibrated_ece}"

        # Confirm strict improvement on held logits
        assert metrics.calibrated_ece < metrics.uncalibrated_ece, (
            f"Head {head} ECE did not improve: {metrics.calibrated_ece} >= {metrics.uncalibrated_ece}"
        )
        assert metrics.calibrated_nll < metrics.uncalibrated_nll, (
            f"Head {head} NLL did not improve: {metrics.calibrated_nll} >= {metrics.uncalibrated_nll}"
        )
        # Meets M3 target ECE threshold <= 0.08
        assert metrics.calibrated_ece <= 0.08, (
            f"Head {head} calibrated ECE {metrics.calibrated_ece} exceeds 0.08 threshold"
        )


def test_temperature_calibrator_serialization_round_trip(tmp_path: Path) -> None:
    """Deterministic verification that per-head temperatures round-trip through temperatures.json."""
    import random
    from pathlib import Path

    rng = random.Random(123)
    dev_logits = {
        "intent": [[3.5, 0.2, -0.5] if i % 2 == 0 else [-0.5, 3.2, 0.1] for i in range(50)],
        "category": [[4.0, 0.1, 0.0, 0.0, 0.0, 0.0] if i % 2 == 0 else [0.0, 4.0, 0.0, 0.0, 0.0, 0.0] for i in range(50)],
        "risk": [[3.8, 0.0, -0.2, 0.1] if i % 2 == 0 else [0.0, 3.6, 0.0, 0.0] for i in range(50)],
        "completeness": [[3.9, 0.0, 0.0] if i % 2 == 0 else [0.0, 3.7, 0.0] for i in range(50)],
    }
    dev_labels = {
        "intent": [0 if i % 2 == 0 else 1 for i in range(50)],
        "category": [0 if i % 2 == 0 else 1 for i in range(50)],
        "risk": [0 if i % 2 == 0 else 1 for i in range(50)],
        "completeness": [0 if i % 2 == 0 else 1 for i in range(50)],
    }

    calibrator = TemperatureCalibrator()
    calibrator.fit_multitask(dev_logits, dev_labels)

    temp_file = tmp_path / "temperatures.json"
    saved_path = calibrator.save_json(temp_file)
    assert saved_path.is_file()

    # Verify JSON structure matches canonical per-head temperatures.json format
    import json
    with temp_file.open("r", encoding="utf-8") as f:
        data = json.load(f)

    assert set(data.keys()) == {"intent", "category", "risk", "completeness"}
    for head, val in data.items():
        assert isinstance(val, (int, float))
        assert val > 0.0

    # Load back using classmethod and load_temperatures helper
    loaded_calibrator = TemperatureCalibrator.from_json(temp_file)
    helper_loaded = load_temperatures(temp_file)

    for head in ("intent", "category", "risk", "completeness"):
        expected_t = calibrator.get_temperature(head)
        assert loaded_calibrator.get_temperature(head) == expected_t
        assert helper_loaded[head] == expected_t

    # Test that loaded calibrator produces bit-for-bit identical outputs on test logits
    test_logits = [2.8, -0.5, 1.2]
    orig_probs = calibrator.calibrate(test_logits, head="intent")
    loaded_probs = loaded_calibrator.calibrate(test_logits, head="intent")
    assert orig_probs == loaded_probs


def test_temperature_calibrator_per_head_dispatch_and_no_dummy_default() -> None:
    """Verify distinct per-head temperatures are properly dispatched and applied without dummy defaults."""
    custom_temps = {
        "intent": 1.45,
        "category": 2.10,
        "risk": 1.85,
        "completeness": 2.30,
    }

    calibrator = MultiHeadTemperatureCalibrator(temperatures=custom_temps)

    assert calibrator.get_temperature("intent") == 1.45
    assert calibrator.get_temperature("category") == 2.10
    assert calibrator.get_temperature("risk") == 1.85
    assert calibrator.get_temperature("completeness") == 2.30

    logits = [4.0, 1.0, 0.0]
    # Calibrate intent (T=1.45) vs completeness (T=2.30)
    intent_probs = calibrator.calibrate(logits, head="intent")
    comp_probs = calibrator.calibrate(logits, head="completeness")

    # Higher temperature should make probabilities closer to uniform (lower max probability)
    assert max(intent_probs) > max(comp_probs)

    # Test error handling
    with pytest.raises(ValueError, match="temperature must be positive"):
        TemperatureCalibrator(temperature=-1.0)

    with pytest.raises(ValueError, match="temperature for intent must be positive"):
        TemperatureCalibrator(temperatures={"intent": 0.0})


def test_robust_nll_minimization_properties() -> None:
    """Verify that fit_temperature never degrades uncalibrated NLL and finds optimal T."""
    import random
    from services.intelligence.calibration import _compute_nll

    rng = random.Random(777)
    # Generate 5 diverse scenario distributions
    for test_idx in range(5):
        k = 4
        logits_batch = []
        labels = []
        for _ in range(80):
            y = rng.randrange(k)
            labels.append(y)
            row = [rng.gauss(0, 1) for _ in range(k)]
            row[y] += rng.uniform(0.5, 4.0)
            logits_batch.append(row)

        uncal_nll = _compute_nll(logits_batch, labels, 1.0)
        opt_t = fit_temperature(logits_batch, labels)
        cal_nll = _compute_nll(logits_batch, labels, opt_t)

        assert opt_t > 0.0
        assert cal_nll <= uncal_nll + 1e-6, f"NLL degraded on test {test_idx}: {cal_nll} > {uncal_nll}"


def test_discover_temperatures_path_and_from_artifact(tmp_path: Path) -> None:
    """Verify robust discovery of temperatures artifact across candidate paths."""
    # 1. Explicit file path
    artifact_dir = tmp_path / "artifacts" / "multitask"
    artifact_dir.mkdir(parents=True)
    temp_file = artifact_dir / "temperatures.json"
    save_temperatures({"intent": 2.1, "category": 1.5, "risk": 1.8, "completeness": 2.3}, temp_file)

    p1 = discover_temperatures_path(explicit_path=temp_file)
    assert p1 == temp_file

    # 2. Explicit directory path
    p2 = discover_temperatures_path(explicit_path=artifact_dir)
    assert p2 == temp_file

    # 3. Discovery from candidate model path in directory
    model_file = artifact_dir / "model.onnx"
    model_file.write_bytes(b"dummy")
    p3 = discover_temperatures_path(candidate_paths=[model_file])
    assert p3 == temp_file

    # 4. Discovery via TemperatureCalibrator.from_artifact
    calibrator = TemperatureCalibrator.from_artifact(candidate_paths=[model_file])
    assert calibrator.get_temperature("intent") == 2.1
    assert calibrator.get_temperature("category") == 1.5
    assert calibrator.get_temperature("risk") == 1.8
    assert calibrator.get_temperature("completeness") == 2.3
    assert calibrator.source_path == temp_file

    # 5. Default fallback when no artifact found
    cal_default = TemperatureCalibrator.from_artifact(candidate_paths=[tmp_path / "non_existent"])
    assert cal_default.get_temperature("intent") == 1.0
    assert cal_default.source_path is None


def test_load_temperatures_nested_and_dict_formats(tmp_path: Path) -> None:
    """Verify loading per-head temperatures from canonical and nested JSON/dict formats."""
    # Format A: Canonical
    t_canon = load_temperatures({"intent": 2.2, "category": 1.4})
    assert t_canon["intent"] == 2.2
    assert t_canon["category"] == 1.4

    # Format B: Wrapped under 'heads'
    t_heads = load_temperatures({"heads": {"intent": 2.2, "category": 1.4}})
    assert t_heads["intent"] == 2.2
    assert t_heads["category"] == 1.4

    # Format C: Wrapped under 'temperatures'
    t_temps = load_temperatures({"temperatures": {"intent": 2.2, "category": 1.4}})
    assert t_temps["intent"] == 2.2
    assert t_temps["category"] == 1.4

    # Format D: File containing nested 'heads'
    nested_file = tmp_path / "nested_temps.json"
    import json
    nested_file.write_text(json.dumps({"heads": {"risk": 1.9, "completeness": 2.4}}), encoding="utf-8")
    t_file = load_temperatures(nested_file)
    assert t_file["risk"] == 1.9
    assert t_file["completeness"] == 2.4
