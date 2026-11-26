from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
from typing import Any, Callable, Sequence

import numpy as np

from services.harness.cache import (
    CachedAnalysis,
    CacheCandidate,
    CacheEntry,
    CacheQuery,
    _lookup_key,
)

logger = logging.getLogger("harness-cache-store")


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


_CACHE_COLUMNS = """
    cache_key, namespace, analysis_version, embedding_model, text_hash,
    source_post_id, source_run_id, embedding, category, rule_category,
    source_intent, location_quality, location_entities, evidence_spans,
    analysis_payload, record_guards, trust_decision, status, hit_count,
    created_at, last_hit_at
"""


def _vector_literal(embedding: Sequence[float]) -> str:
    return "[" + ",".join(f"{float(value):.9f}" for value in embedding) + "]"


def _parse_vector(value: Any) -> list[float]:
    if isinstance(value, str):
        return [float(part) for part in value.strip().strip("[]").split(",") if part]
    return [float(part) for part in value]


def _parse_json(value: Any, default: Any) -> Any:
    if value is None:
        return default
    if isinstance(value, str):
        return json.loads(value)
    return value


class PgVectorHarnessCacheStore:
    """PostgreSQL + pgvector store for versioned harness analyses."""

    def __init__(self, connection_factory: Callable[[], Any]) -> None:
        self._connection_factory = connection_factory

    def find_exact(
        self,
        namespace: str,
        analysis_version: str,
        embedding_model: str,
        text_hash: str,
    ) -> CacheEntry | None:
        try:
            with self._connection_factory() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        f"""
                        SELECT {_CACHE_COLUMNS}
                        FROM harness_case_cache
                        WHERE namespace = %s
                          AND analysis_version = %s
                          AND embedding_model = %s
                          AND text_hash = %s
                        LIMIT 1
                        """,
                        (namespace, analysis_version, embedding_model, text_hash),
                    )
                    row = cur.fetchone()
                    return self._row_to_entry(row) if row else None
        except Exception as exc:
            logger.warning("PgVectorHarnessCacheStore find_exact error: %s", exc)
            return None

    def find_similar(self, query: CacheQuery) -> list[CacheCandidate]:
        try:
            vector = _vector_literal(query.embedding)
            with self._connection_factory() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        f"""
                        SELECT {_CACHE_COLUMNS},
                               1 - (embedding <=> %s::vector) AS similarity
                        FROM harness_case_cache
                        WHERE namespace = %s
                          AND analysis_version = %s
                          AND embedding_model = %s
                          AND category = %s
                          AND source_intent = %s
                          AND 1 - (embedding <=> %s::vector) >= %s
                        ORDER BY embedding <=> %s::vector
                        LIMIT %s
                        """,
                        (
                            vector,
                            query.namespace,
                            query.analysis_version,
                            query.embedding_model,
                            query.category.value,
                            query.source_intent,
                            vector,
                            query.min_similarity,
                            vector,
                            query.limit,
                        ),
                    )
                    rows = cur.fetchall()
                    return [
                        CacheCandidate(
                            entry=self._row_to_entry(row[:-1]),
                            similarity=min(1.0, max(0.0, float(row[-1]))),
                        )
                        for row in rows
                    ]
        except Exception as exc:
            logger.warning("PgVectorHarnessCacheStore find_similar error: %s", exc)
            return []

    def save(self, entry: CacheEntry) -> None:
        try:
            with self._connection_factory() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO harness_case_cache (
                            cache_key, namespace, analysis_version, embedding_model, text_hash,
                            source_post_id, source_run_id, embedding, category, rule_category,
                            source_intent, location_quality, location_entities, evidence_spans,
                            analysis_payload, record_guards, trust_decision, status, hit_count,
                            created_at, last_hit_at
                        ) VALUES (
                            %s, %s, %s, %s, %s,
                            %s, %s, %s::vector, %s, %s,
                            %s, %s, %s::jsonb, %s::jsonb,
                            %s::jsonb, %s::jsonb, %s, %s, %s,
                            %s, %s
                        )
                        ON CONFLICT (cache_key) DO UPDATE SET
                            source_post_id = EXCLUDED.source_post_id,
                            source_run_id = EXCLUDED.source_run_id,
                            embedding = EXCLUDED.embedding,
                            category = EXCLUDED.category,
                            rule_category = EXCLUDED.rule_category,
                            source_intent = EXCLUDED.source_intent,
                            location_quality = EXCLUDED.location_quality,
                            location_entities = EXCLUDED.location_entities,
                            evidence_spans = EXCLUDED.evidence_spans,
                            analysis_payload = EXCLUDED.analysis_payload,
                            record_guards = EXCLUDED.record_guards,
                            trust_decision = EXCLUDED.trust_decision,
                            status = EXCLUDED.status
                        """,
                        (
                            entry.cache_key,
                            entry.namespace,
                            entry.analysis_version,
                            entry.embedding_model,
                            entry.text_hash,
                            entry.source_post_id,
                            entry.source_run_id,
                            _vector_literal(entry.embedding),
                            entry.category.value,
                            entry.rule_category.value,
                            entry.source_intent,
                            entry.location_quality,
                            json.dumps(entry.location_entities),
                            json.dumps(entry.evidence_spans),
                            json.dumps(entry.analysis.model_dump(mode="json")),
                            json.dumps(entry.record_guards),
                            entry.trust_decision,
                            entry.status,
                            entry.hit_count,
                            entry.created_at,
                            entry.last_hit_at,
                        ),
                    )
                    conn.commit()
        except Exception as exc:
            logger.warning("PgVectorHarnessCacheStore save error: %s", exc)

    def increment_hit(self, cache_key: str) -> None:
        try:
            with self._connection_factory() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        UPDATE harness_case_cache
                        SET hit_count = hit_count + 1,
                            last_hit_at = clock_timestamp()
                        WHERE cache_key = %s
                        """,
                        (cache_key,),
                    )
                    conn.commit()
        except Exception as exc:
            logger.warning("PgVectorHarnessCacheStore increment_hit error: %s", exc)

    def reset(self, namespace: str, analysis_version: str) -> int:
        try:
            with self._connection_factory() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        DELETE FROM harness_case_cache
                        WHERE namespace = %s AND analysis_version = %s
                        """,
                        (namespace, analysis_version),
                    )
                    deleted = int(cur.rowcount or 0)
                    conn.commit()
                    return deleted
        except Exception as exc:
            logger.warning("PgVectorHarnessCacheStore reset error: %s", exc)
            return 0

    @staticmethod
    def _row_to_entry(row: Sequence[Any]) -> CacheEntry:
        location_entities = tuple(_parse_json(row[12], default=[]))
        evidence_spans = tuple(_parse_json(row[13], default=[]))
        analysis = CachedAnalysis.model_validate(_parse_json(row[14], default={}))
        record_guards = _parse_json(row[15], default={})
        return CacheEntry(
            cache_key=str(row[0]),
            namespace=str(row[1]),
            analysis_version=str(row[2]),
            embedding_model=str(row[3]),
            text_hash=str(row[4]),
            source_post_id=str(row[5]),
            source_run_id=str(row[6]),
            embedding=tuple(_parse_vector(row[7])),
            category=row[8],
            rule_category=row[9],
            source_intent=row[10],
            location_quality=row[11],
            location_entities=location_entities,
            evidence_spans=evidence_spans,
            analysis=analysis,
            record_guards=record_guards,
            trust_decision=row[16],
            status=row[17],
            hit_count=int(row[18]),
            created_at=row[19],
            last_hit_at=row[20],
        )
