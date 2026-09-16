from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import logging
from typing import Any, Literal, Protocol, Sequence
from uuid import uuid4

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from contracts.models import Category, RiskLevel
from services.intelligence.embedder import IndoBertEmbedder, TextEmbedder, cosine_similarity

logger = logging.getLogger("semantic-cache")


class KnowledgeBankEntry(BaseModel):
    model_config = ConfigDict(extra="ignore")

    entry_id: str = Field(default_factory=lambda: str(uuid4()))
    tenant_id: str
    case_id: str
    text_hash: str
    raw_text: str
    embedding: list[float]
    category: Category
    risk: RiskLevel
    completeness: str
    location_entities: list[str] = Field(default_factory=list)
    root_cause_category: Category | None = None
    root_cause_summary: str | None = None
    ticket_id: str | None = None
    is_active_incident: bool = True
    incident_cluster_id: str | None = None
    hit_count: int = 0
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class CacheLookupResult(BaseModel):
    model_config = ConfigDict(extra="ignore")

    hit: bool
    match_type: Literal["EXACT", "SEMANTIC", "NONE"] = "NONE"
    similarity: float = 0.0
    matched_entry: KnowledgeBankEntry | None = None
    is_duplicate_incident: bool = False
    suggested_action: Literal["REUSE_ANALYSIS", "LINK_DUPLICATE_TICKET", "RUN_FULL_PIPELINE"] = "RUN_FULL_PIPELINE"
    explanation: str = ""


class CacheStore(Protocol):
    """Protocol for semantic cache storage engines."""

    def find_by_hash(self, tenant_id: str, text_hash: str) -> KnowledgeBankEntry | None:
        ...

    def find_similar(
        self,
        tenant_id: str,
        embedding: Sequence[float],
        min_similarity: float,
        limit: int = 5,
        category: Category | None = None,
    ) -> list[tuple[KnowledgeBankEntry, float]]:
        ...

    def save_entry(self, entry: KnowledgeBankEntry) -> None:
        ...

    def increment_hit(self, entry_id: str) -> None:
        ...

    def link_to_cluster(self, entry_id: str, cluster_id: str) -> None:
        ...


class InMemoryCacheStore:
    """In-memory cache store with NumPy cosine similarity indexing."""

    def __init__(self) -> None:
        self._entries: dict[str, KnowledgeBankEntry] = {}
        self._hash_index: dict[tuple[str, str], str] = {}

    def find_by_hash(self, tenant_id: str, text_hash: str) -> KnowledgeBankEntry | None:
        entry_id = self._hash_index.get((tenant_id, text_hash))
        return self._entries.get(entry_id) if entry_id else None

    def find_similar(
        self,
        tenant_id: str,
        embedding: Sequence[float],
        min_similarity: float,
        limit: int = 5,
        category: Category | None = None,
    ) -> list[tuple[KnowledgeBankEntry, float]]:
        candidates = [
            e for e in self._entries.values()
            if e.tenant_id == tenant_id and (category is None or e.category == category)
        ]
        if not candidates:
            return []

        query_vec = np.asarray(embedding, dtype=np.float32)
        norm_q = np.linalg.norm(query_vec)
        if norm_q <= 1e-9:
            return []

        scored: list[tuple[KnowledgeBankEntry, float]] = []
        for cand in candidates:
            cand_vec = np.asarray(cand.embedding, dtype=np.float32)
            sim = float(np.dot(query_vec, cand_vec) / (norm_q * np.linalg.norm(cand_vec) + 1e-9))
            if sim >= min_similarity:
                scored.append((cand, sim))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:limit]

    def save_entry(self, entry: KnowledgeBankEntry) -> None:
        self._entries[entry.entry_id] = entry
        self._hash_index[(entry.tenant_id, entry.text_hash)] = entry.entry_id

    def increment_hit(self, entry_id: str) -> None:
        entry = self._entries.get(entry_id)
        if entry:
            entry.hit_count += 1
            entry.updated_at = datetime.now(timezone.utc)

    def link_to_cluster(self, entry_id: str, cluster_id: str) -> None:
        entry = self._entries.get(entry_id)
        if entry:
            entry.incident_cluster_id = cluster_id
            entry.updated_at = datetime.now(timezone.utc)

    @property
    def count(self) -> int:
        return len(self._entries)


class PostgresCacheStore:
    """PostgreSQL storage backend querying case_knowledge_bank with dot product cosine similarity."""

    def __init__(self, connection_factory: Any) -> None:
        self._connection_factory = connection_factory

    def find_by_hash(self, tenant_id: str, text_hash: str) -> KnowledgeBankEntry | None:
        try:
            with self._connection_factory() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT entry_id, tenant_id, case_id, text_hash, raw_text, embedding,
                               category, risk, completeness, location_entities,
                               root_cause_category, root_cause_summary, ticket_id,
                               is_active_incident, incident_cluster_id, hit_count, created_at, updated_at
                        FROM case_knowledge_bank
                        WHERE tenant_id = %s AND text_hash = %s
                        LIMIT 1
                        """,
                        (tenant_id, text_hash),
                    )
                    row = cur.fetchone()
                    if row:
                        return self._row_to_entry(row)
        except Exception as exc:
            logger.warning("PostgresCacheStore find_by_hash error: %s", exc)
        return None

    def find_similar(
        self,
        tenant_id: str,
        embedding: Sequence[float],
        min_similarity: float,
        limit: int = 5,
        category: Category | None = None,
    ) -> list[tuple[KnowledgeBankEntry, float]]:
        try:
            emb_list = [float(x) for x in embedding]
            query = """
                SELECT entry_id, tenant_id, case_id, text_hash, raw_text, embedding,
                       category, risk, completeness, location_entities,
                       root_cause_category, root_cause_summary, ticket_id,
                       is_active_incident, incident_cluster_id, hit_count, created_at, updated_at,
                       cosine_similarity_float8(embedding, %s) AS sim
                FROM case_knowledge_bank
                WHERE tenant_id = %s
            """
            params: list[Any] = [emb_list, tenant_id]
            if category is not None:
                query += " AND category = %s"
                params.append(category.value)

            query += " AND cosine_similarity_float8(embedding, %s) >= %s"
            params.extend([emb_list, min_similarity])
            query += " ORDER BY sim DESC LIMIT %s"
            params.append(limit)

            with self._connection_factory() as conn:
                with conn.cursor() as cur:
                    cur.execute(query, tuple(params))
                    rows = cur.fetchall()
                    results: list[tuple[KnowledgeBankEntry, float]] = []
                    for row in rows:
                        entry = self._row_to_entry(row[:-1])
                        sim = float(row[-1])
                        results.append((entry, sim))
                    return results
        except Exception as exc:
            logger.warning("PostgresCacheStore find_similar error: %s", exc)
            return []

    def save_entry(self, entry: KnowledgeBankEntry) -> None:
        try:
            import json
            with self._connection_factory() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO case_knowledge_bank (
                            entry_id, tenant_id, case_id, text_hash, raw_text, embedding,
                            category, risk, completeness, location_entities,
                            root_cause_category, root_cause_summary, ticket_id,
                            is_active_incident, incident_cluster_id, hit_count
                        ) VALUES (
                            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                        )
                        ON CONFLICT (case_id) DO UPDATE SET
                            raw_text = EXCLUDED.raw_text,
                            embedding = EXCLUDED.embedding,
                            category = EXCLUDED.category,
                            risk = EXCLUDED.risk,
                            completeness = EXCLUDED.completeness,
                            location_entities = EXCLUDED.location_entities,
                            root_cause_category = EXCLUDED.root_cause_category,
                            root_cause_summary = EXCLUDED.root_cause_summary,
                            ticket_id = COALESCE(EXCLUDED.ticket_id, case_knowledge_bank.ticket_id),
                            updated_at = clock_timestamp()
                        """,
                        (
                            entry.entry_id,
                            entry.tenant_id,
                            entry.case_id,
                            entry.text_hash,
                            entry.raw_text,
                            entry.embedding,
                            entry.category.value,
                            entry.risk.value,
                            entry.completeness,
                            json.dumps(entry.location_entities),
                            entry.root_cause_category.value if entry.root_cause_category else None,
                            entry.root_cause_summary,
                            entry.ticket_id,
                            entry.is_active_incident,
                            entry.incident_cluster_id,
                            entry.hit_count,
                        ),
                    )
                    conn.commit()
        except Exception as exc:
            logger.warning("PostgresCacheStore save_entry error: %s", exc)

    def increment_hit(self, entry_id: str) -> None:
        try:
            with self._connection_factory() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        UPDATE case_knowledge_bank
                        SET hit_count = hit_count + 1, updated_at = clock_timestamp()
                        WHERE entry_id = %s
                        """,
                        (entry_id,),
                    )
                    conn.commit()
        except Exception as exc:
            logger.warning("PostgresCacheStore increment_hit error: %s", exc)

    def link_to_cluster(self, entry_id: str, cluster_id: str) -> None:
        try:
            with self._connection_factory() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        UPDATE case_knowledge_bank
                        SET incident_cluster_id = %s, updated_at = clock_timestamp()
                        WHERE entry_id = %s
                        """,
                        (cluster_id, entry_id),
                    )
                    conn.commit()
        except Exception as exc:
            logger.warning("PostgresCacheStore link_to_cluster error: %s", exc)

    def _row_to_entry(self, row: Sequence[Any]) -> KnowledgeBankEntry:
        import json
        locs = row[9]
        if isinstance(locs, str):
            locs = json.loads(locs)
        elif locs is None:
            locs = []

        root_cat = None
        if row[10]:
            try:
                root_cat = Category(row[10])
            except ValueError:
                pass

        return KnowledgeBankEntry(
            entry_id=str(row[0]),
            tenant_id=str(row[1]),
            case_id=str(row[2]),
            text_hash=str(row[3]),
            raw_text=str(row[4]),
            embedding=[float(x) for x in row[5]],
            category=Category(row[6]),
            risk=RiskLevel(row[7]),
            completeness=str(row[8]),
            location_entities=list(locs),
            root_cause_category=root_cat,
            root_cause_summary=row[11],
            ticket_id=row[12],
            is_active_incident=bool(row[13]),
            incident_cluster_id=row[14],
            hit_count=int(row[15]),
            created_at=row[16],
            updated_at=row[17],
        )


class SemanticCacheEngine:
    """High-performance Semantic Cache and Incident Deduplicator for citizen complaints.

    Intercepts redundant complaints to save LLM reasoning tokens and prevent duplicate tickets.
    """

    def __init__(
        self,
        store: CacheStore | None = None,
        embedder: TextEmbedder | None = None,
        exact_match_enabled: bool = True,
        semantic_threshold: float = 0.88,
        duplicate_threshold: float = 0.85,
    ) -> None:
        self._store = store if store is not None else InMemoryCacheStore()
        self._embedder = embedder if embedder is not None else IndoBertEmbedder.get_instance()
        self._exact_match_enabled = exact_match_enabled
        self._semantic_threshold = semantic_threshold
        self._duplicate_threshold = duplicate_threshold

    @staticmethod
    def compute_hash(text: str) -> str:
        norm = " ".join(text.strip().lower().split())
        return hashlib.sha256(norm.encode("utf-8")).hexdigest()

    def lookup(
        self,
        tenant_id: str,
        text: str,
        location_hints: Sequence[str] = (),
        category_hint: Category | None = None,
    ) -> CacheLookupResult:
        """Lookup complaint in semantic cache for exact match, causal reuse, or duplicate incident."""
        text_hash = self.compute_hash(text)

        # 1. Exact Match Lookup (0ms hash lookup)
        if self._exact_match_enabled:
            exact = self._store.find_by_hash(tenant_id, text_hash)
            if exact is not None:
                self._store.increment_hit(exact.entry_id)
                is_duplicate = bool(exact.is_active_incident and exact.ticket_id)
                action: Literal["REUSE_ANALYSIS", "LINK_DUPLICATE_TICKET", "RUN_FULL_PIPELINE"] = (
                    "LINK_DUPLICATE_TICKET" if is_duplicate else "REUSE_ANALYSIS"
                )
                return CacheLookupResult(
                    hit=True,
                    match_type="EXACT",
                    similarity=1.0,
                    matched_entry=exact,
                    is_duplicate_incident=is_duplicate,
                    suggested_action=action,
                    explanation="Exact match hit: complaint previously processed identically",
                )

        # 2. Semantic Embedding Generation (~15ms)
        emb = self._embedder.embed_text(text)

        # 3. Vector Similarity Search
        similars = self._store.find_similar(
            tenant_id=tenant_id,
            embedding=emb,
            min_similarity=self._duplicate_threshold,
            limit=5,
            category=category_hint,
        )
        if not similars:
            return CacheLookupResult(
                hit=False,
                match_type="NONE",
                similarity=0.0,
                suggested_action="RUN_FULL_PIPELINE",
                explanation="No similar complaints found in knowledge bank",
            )

        best_entry, best_sim = similars[0]
        has_location_overlap = self._check_location_overlap(
            location_hints, best_entry.location_entities, text, best_entry.raw_text
        )

        # 4. Duplicate Incident Evaluation
        if (
            best_sim >= self._duplicate_threshold
            and has_location_overlap
            and best_entry.is_active_incident
            and best_entry.ticket_id is not None
        ):
            self._store.increment_hit(best_entry.entry_id)
            return CacheLookupResult(
                hit=True,
                match_type="SEMANTIC",
                similarity=best_sim,
                matched_entry=best_entry,
                is_duplicate_incident=True,
                suggested_action="LINK_DUPLICATE_TICKET",
                explanation=f"Semantic cluster duplicate: overlaps with active ticket {best_entry.ticket_id} (sim={best_sim:.3f})",
            )

        # 5. Semantic Causal / Prediction Reuse
        if best_sim >= self._semantic_threshold:
            self._store.increment_hit(best_entry.entry_id)
            return CacheLookupResult(
                hit=True,
                match_type="SEMANTIC",
                similarity=best_sim,
                matched_entry=best_entry,
                is_duplicate_incident=False,
                suggested_action="REUSE_ANALYSIS",
                explanation=f"High semantic similarity hit (sim={best_sim:.3f}): reusable category and root cause",
            )

        return CacheLookupResult(
            hit=False,
            match_type="NONE",
            similarity=best_sim,
            matched_entry=best_entry,
            suggested_action="RUN_FULL_PIPELINE",
            explanation=f"Partial similarity (sim={best_sim:.3f}) below reuse threshold ({self._semantic_threshold})",
        )

    def record_case(
        self,
        tenant_id: str,
        case_id: str,
        text: str,
        category: Category,
        risk: RiskLevel,
        completeness: str,
        location_entities: Sequence[str] = (),
        root_cause_category: Category | None = None,
        root_cause_summary: str | None = None,
        ticket_id: str | None = None,
        embedding: Sequence[float] | None = None,
    ) -> KnowledgeBankEntry:
        """Register a verified complaint into the knowledge bank."""
        emb = list(embedding) if embedding is not None else self._embedder.embed_text(text)
        entry = KnowledgeBankEntry(
            tenant_id=tenant_id,
            case_id=case_id,
            text_hash=self.compute_hash(text),
            raw_text=text,
            embedding=emb,
            category=category,
            risk=risk,
            completeness=completeness,
            location_entities=[loc.strip().lower() for loc in location_entities if loc.strip()],
            root_cause_category=root_cause_category,
            root_cause_summary=root_cause_summary,
            ticket_id=ticket_id,
            is_active_incident=ticket_id is not None,
            incident_cluster_id=f"cluster-{case_id}" if ticket_id else None,
        )
        self._store.save_entry(entry)
        logger.info("Recorded case %s into knowledge bank (category=%s)", case_id, category.value)
        return entry

    @staticmethod
    def _check_location_overlap(
        locs1: Sequence[str],
        locs2: Sequence[str],
        text1: str = "",
        text2: str = "",
    ) -> bool:
        stopwords = {
            "di", "jl", "jalan", "gg", "gang", "kota", "kabupaten", "daerah", "dekat", "no", "nomor", "rt", "rw",
            "bandung", "jakarta", "surabaya", "semarang", "medan", "bekasi", "depok", "tangerang", "bogor",
            "jawa", "barat", "timur", "tengah", "indonesia",
            # Descriptive words that may appear after 'jalan' or 'daerah'
            "rusak", "berlubang", "lubang", "bolong", "parah", "hancur", "amblas", "ambruk",
            "sering", "bikin", "banyak", "banget", "jatuh", "motor", "mobil", "rawan", "bahaya",
            "mampet", "banjir", "bau", "busuk", "sampah", "tergenang", "meluap"
        }

        if locs1 and locs2:
            def tokenize_loc(loc: str) -> set[str]:
                return {w for w in loc.lower().replace(",", " ").replace(".", " ").split() if len(w) > 2 and w not in stopwords}

            tokens1 = set().union(*(tokenize_loc(l) for l in locs1))
            tokens2 = set().union(*(tokenize_loc(l) for l in locs2))
            if len(tokens1.intersection(tokens2)) > 0:
                return True

        t1_lower = text1.lower()
        for loc in locs2:
            cleaned = loc.lower().strip()
            if len(cleaned) > 3 and cleaned not in stopwords and cleaned in t1_lower:
                return True

        t2_lower = text2.lower()
        for loc in locs1:
            cleaned = loc.lower().strip()
            if len(cleaned) > 3 and cleaned not in stopwords and cleaned in t2_lower:
                return True

        return False
