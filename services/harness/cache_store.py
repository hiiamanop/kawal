from __future__ import annotations

from datetime import datetime, timezone

import numpy as np

from services.harness.cache import (
    CacheCandidate,
    CacheEntry,
    CacheQuery,
    _lookup_key,
)


class InMemoryHarnessCacheStore:
    """In-memory vector store used by tests and offline evaluation."""

    def __init__(self) -> None:
        self._entries: dict[str, CacheEntry] = {}
        self._hash_index: dict[tuple[str, str, str, str], str] = {}

    @property
    def count(self) -> int:
        return len(self._entries)

    def find_exact(
        self,
        namespace: str,
        analysis_version: str,
        embedding_model: str,
        text_hash: str,
    ) -> CacheEntry | None:
        entry_id = self._hash_index.get(
            _lookup_key(namespace, analysis_version, embedding_model, text_hash)
        )
        return self._entries.get(entry_id) if entry_id else None

    def find_similar(self, query: CacheQuery) -> list[CacheCandidate]:
        candidates = [
            entry
            for entry in self._entries.values()
            if entry.namespace == query.namespace
            and entry.analysis_version == query.analysis_version
            and entry.embedding_model == query.embedding_model
            and entry.category == query.category
            and entry.source_intent == query.source_intent
        ]
        if not candidates:
            return []

        query_vector = np.asarray(query.embedding, dtype=np.float32)
        query_norm = np.linalg.norm(query_vector)
        if query_norm <= 1e-9:
            return []

        scored: list[CacheCandidate] = []
        for entry in candidates:
            entry_vector = np.asarray(entry.embedding, dtype=np.float32)
            denominator = query_norm * np.linalg.norm(entry_vector)
            if denominator <= 1e-9:
                continue
            similarity = float(np.dot(query_vector, entry_vector) / denominator)
            if similarity >= query.min_similarity:
                scored.append(
                    CacheCandidate(
                        entry=entry,
                        similarity=min(1.0, max(0.0, similarity)),
                    )
                )
        scored.sort(key=lambda candidate: candidate.similarity, reverse=True)
        return scored[: query.limit]

    def save(self, entry: CacheEntry) -> None:
        self._entries[entry.cache_key] = entry
        self._hash_index[
            _lookup_key(
                entry.namespace,
                entry.analysis_version,
                entry.embedding_model,
                entry.text_hash,
            )
        ] = entry.cache_key

    def increment_hit(self, cache_key: str) -> None:
        entry = self._entries.get(cache_key)
        if entry is None:
            return
        self._entries[cache_key] = entry.model_copy(
            update={
                "hit_count": entry.hit_count + 1,
                "last_hit_at": datetime.now(timezone.utc),
            }
        )

    def reset(self, namespace: str, analysis_version: str) -> int:
        keys = [
            key
            for key, entry in self._entries.items()
            if entry.namespace == namespace
            and entry.analysis_version == analysis_version
        ]
        for key in keys:
            del self._entries[key]
        self._hash_index = {
            _lookup_key(
                entry.namespace,
                entry.analysis_version,
                entry.embedding_model,
                entry.text_hash,
            ): key
            for key, entry in self._entries.items()
        }
        return len(keys)
