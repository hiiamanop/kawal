from __future__ import annotations

from pathlib import Path

import pytest

from services.harness.cache import HarnessAnalysisCache
from services.harness.cache_store import InMemoryHarnessCacheStore
from services.intelligence.embedder import (
    DeterministicFallbackEmbedder,
    IndoBertEmbedder,
    TextEmbedder,
    _artifact_fingerprint,
)


def test_deterministic_embedder_identity() -> None:
    embedder = DeterministicFallbackEmbedder(dim=768)
    assert embedder.model_id == "deterministic-hash-768-v1"
    assert embedder.is_fallback is True


def test_indobert_missing_artifact_reports_fallback(tmp_path: Path) -> None:
    embedder = IndoBertEmbedder(model_path=tmp_path / "missing")
    assert embedder.is_fallback is True
    assert embedder.model_id == "indobert-fallback-768-cls"


def test_indobert_pooling_changes_identity(tmp_path: Path) -> None:
    embedder = IndoBertEmbedder(model_path=tmp_path / "missing", pooling="mean")
    assert embedder.model_id == "indobert-fallback-768-mean"


def test_artifact_fingerprint_changes_with_weights(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    for path, payload in ((first, b"a" * 2048), (second, b"b" * 2048)):
        path.mkdir()
        (path / "config.json").write_text('{"model": "indobert"}', encoding="utf-8")
        (path / "model.safetensors").write_bytes(payload)
    assert _artifact_fingerprint(first) == _artifact_fingerprint(first)
    assert _artifact_fingerprint(first) != _artifact_fingerprint(second)


def test_cache_rejects_embedder_without_identity() -> None:
    class UnknownEmbedder(TextEmbedder):
        pass

    with pytest.raises(ValueError):
        HarnessAnalysisCache(
            store=InMemoryHarnessCacheStore(),
            embedder=UnknownEmbedder(),
            namespace="harness-x",
            analysis_version="v1",
        )
