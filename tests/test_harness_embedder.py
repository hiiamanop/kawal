from __future__ import annotations

from pathlib import Path

from services.intelligence.embedder import (
    DeterministicFallbackEmbedder,
    IndoBertEmbedder,
)


def test_deterministic_embedder_identity() -> None:
    embedder = DeterministicFallbackEmbedder(dim=768)
    assert embedder.model_id == "deterministic-hash-768-v1"
    assert embedder.is_fallback is True


def test_indobert_missing_artifact_reports_fallback(tmp_path: Path) -> None:
    embedder = IndoBertEmbedder(model_path=tmp_path / "missing")
    assert embedder.is_fallback is True
    assert embedder.model_id == "indobert-fallback-768"
