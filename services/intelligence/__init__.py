from __future__ import annotations

from services.intelligence.chunking import (
    Chunk,
    TokenSpan,
    chunk_messages,
    chunk_text,
    tokenize_with_offsets,
)
from services.intelligence.ner import (
    SUPPORTED_ENTITY_TYPES,
    EntitySpan,
    extract_entities_from_text,
    merge_overlapping_spans,
    parse_bio_tags,
)
from services.intelligence.retrieval import (
    BM25Retriever,
    IndexedDocument,
    LexicalEmbedding,
    LexicalRetriever,
    RetrievalResult,
    compute_lexical_embedding,
    cosine_similarity,
    rerank_candidates,
)
from services.intelligence.calibration import (
    CalibrationMetrics,
    CpuBenchmarkConfig,
    CpuBenchmarkResult,
    TemperatureCalibrator,
    apply_temperature,
    apply_temperature_batch,
    compute_ece,
    fit_temperature,
    run_cpu_benchmark,
)

__all__ = [
    "Chunk",
    "TokenSpan",
    "chunk_text",
    "chunk_messages",
    "tokenize_with_offsets",
    "SUPPORTED_ENTITY_TYPES",
    "EntitySpan",
    "parse_bio_tags",
    "extract_entities_from_text",
    "merge_overlapping_spans",
    "LexicalEmbedding",
    "compute_lexical_embedding",
    "cosine_similarity",
    "IndexedDocument",
    "RetrievalResult",
    "BM25Retriever",
    "LexicalRetriever",
    "rerank_candidates",
    "CalibrationMetrics",
    "CpuBenchmarkConfig",
    "CpuBenchmarkResult",
    "TemperatureCalibrator",
    "apply_temperature",
    "apply_temperature_batch",
    "compute_ece",
    "fit_temperature",
    "run_cpu_benchmark",
]
