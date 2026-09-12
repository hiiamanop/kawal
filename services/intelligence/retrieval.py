from __future__ import annotations

import hashlib
import math
import re
from typing import Any, Sequence
from pydantic import BaseModel, ConfigDict, Field


_WORD_REGEX = re.compile(r"\b\w+\b", re.UNICODE)


def _tokenize(text: str) -> list[str]:
    return [w.lower() for w in _WORD_REGEX.findall(text)]


class LexicalEmbedding(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    dense_vector: tuple[float, ...] = Field(default_factory=tuple)
    sparse_weights: dict[str, float] = Field(default_factory=dict)
    dimension: int = Field(default=256, ge=8)


def compute_lexical_embedding(
    text: str,
    dimension: int = 256,
) -> LexicalEmbedding:
    tokens = _tokenize(text)
    if not tokens:
        return LexicalEmbedding(
            dense_vector=tuple([0.0] * dimension),
            sparse_weights={},
            dimension=dimension,
        )

    sparse: dict[str, float] = {}
    dense = [0.0] * dimension

    for token in tokens:
        sparse[token] = sparse.get(token, 0.0) + 1.0

        digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
        slot = int(digest[:8], 16) % dimension
        sign = 1.0 if int(digest[8:10], 16) % 2 == 0 else -1.0
        dense[slot] += sign * 1.0

    total_tokens = float(len(tokens))
    for k in sparse:
        sparse[k] /= total_tokens

    norm = math.sqrt(sum(v * v for v in dense))
    if norm > 0.0:
        dense_norm = tuple(round(v / norm, 6) for v in dense)
    else:
        dense_norm = tuple(dense)

    return LexicalEmbedding(
        dense_vector=dense_norm,
        sparse_weights=sparse,
        dimension=dimension,
    )


def cosine_similarity(
    vec1: tuple[float, ...],
    vec2: tuple[float, ...],
) -> float:
    if not vec1 or not vec2 or len(vec1) != len(vec2):
        return 0.0
    dot = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))
    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0
    val = dot / (norm1 * norm2)
    return max(0.0, min(1.0, float(val)))


class IndexedDocument(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    doc_id: str
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    embedding: LexicalEmbedding | None = None


class RetrievalResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    doc_id: str
    text: str
    lexical_score: float = Field(ge=0.0)
    rerank_score: float = Field(ge=0.0)
    combined_score: float = Field(ge=0.0)
    rank: int = Field(ge=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class BM25Retriever:
    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        embedding_dim: int = 256,
    ) -> None:
        self.k1 = k1
        self.b = b
        self.embedding_dim = embedding_dim
        self._documents: dict[str, IndexedDocument] = {}
        self._doc_tokens: dict[str, list[str]] = {}
        self._doc_lens: dict[str, int] = {}
        self._doc_freq: dict[str, int] = {}
        self._avg_dl: float = 0.0

    @property
    def document_count(self) -> int:
        return len(self._documents)

    def add_document(
        self,
        doc_id: str,
        text: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        meta = metadata or {}
        embedding = compute_lexical_embedding(text, dimension=self.embedding_dim)
        doc = IndexedDocument(
            doc_id=doc_id,
            text=text,
            metadata=meta,
            embedding=embedding,
        )
        self._documents[doc_id] = doc
        tokens = _tokenize(text)
        self._doc_tokens[doc_id] = tokens
        self._doc_lens[doc_id] = len(tokens)

        unique_tokens = set(tokens)
        for t in unique_tokens:
            self._doc_freq[t] = self._doc_freq.get(t, 0) + 1

        self._avg_dl = sum(self._doc_lens.values()) / max(1, len(self._documents))

    def index_documents(self, documents: Sequence[IndexedDocument]) -> None:
        for doc in documents:
            self.add_document(doc.doc_id, doc.text, doc.metadata)

    def retrieve(self, query: str, top_k: int = 10) -> list[RetrievalResult]:
        if not self._documents:
            return []

        q_tokens = _tokenize(query)
        if not q_tokens:
            return []

        n_docs = len(self._documents)
        scores: dict[str, float] = {}

        for t in q_tokens:
            df = self._doc_freq.get(t, 0)
            if df == 0:
                continue
            idf = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))

            for doc_id, d_tokens in self._doc_tokens.items():
                tf = d_tokens.count(t)
                if tf == 0:
                    continue
                d_len = self._doc_lens[doc_id]
                denom = tf + self.k1 * (1.0 - self.b + self.b * (d_len / max(1.0, self._avg_dl)))
                score = idf * (tf * (self.k1 + 1.0)) / denom
                scores[doc_id] = scores.get(doc_id, 0.0) + score

        sorted_ids = sorted(scores.keys(), key=lambda did: scores[did], reverse=True)[:top_k]
        results: list[RetrievalResult] = []
        for rank, did in enumerate(sorted_ids, start=1):
            doc = self._documents[did]
            raw_score = scores[did]
            results.append(
                RetrievalResult(
                    doc_id=did,
                    text=doc.text,
                    lexical_score=round(raw_score, 4),
                    rerank_score=round(raw_score, 4),
                    combined_score=round(raw_score, 4),
                    rank=rank,
                    metadata=doc.metadata,
                )
            )
        return results

    def retrieve_and_rerank(
        self,
        query: str,
        top_k: int = 5,
        candidate_pool: int = 20,
        lexical_weight: float = 0.5,
        cosine_weight: float = 0.2,
        coverage_weight: float = 0.2,
        phrase_weight: float = 0.1,
    ) -> list[RetrievalResult]:
        candidates = self.retrieve(query, top_k=candidate_pool)
        return rerank_candidates(
            query=query,
            candidates=candidates,
            retriever=self,
            top_k=top_k,
            lexical_weight=lexical_weight,
            cosine_weight=cosine_weight,
            coverage_weight=coverage_weight,
            phrase_weight=phrase_weight,
        )


def rerank_candidates(
    query: str,
    candidates: Sequence[RetrievalResult],
    retriever: BM25Retriever | None = None,
    top_k: int = 5,
    lexical_weight: float = 0.5,
    cosine_weight: float = 0.2,
    coverage_weight: float = 0.2,
    phrase_weight: float = 0.1,
) -> list[RetrievalResult]:
    if not candidates:
        return []

    q_tokens = _tokenize(query)
    q_lower = query.lower().strip()
    embedding_dim = retriever.embedding_dim if retriever is not None else 256
    q_emb = compute_lexical_embedding(query, dimension=embedding_dim)

    max_lexical = max((c.lexical_score for c in candidates), default=1.0)
    if max_lexical <= 0.0:
        max_lexical = 1.0

    scored_items: list[tuple[float, float, float, RetrievalResult]] = []

    for item in candidates:
        norm_lex = item.lexical_score / max_lexical

        d_lower = item.text.lower()
        d_tokens = set(_tokenize(item.text))

        if q_tokens:
            matches = sum(1 for qt in q_tokens if qt in d_tokens)
            coverage = matches / len(q_tokens)
        else:
            coverage = 0.0

        phrase_match = 1.0 if q_lower in d_lower else 0.0
        if not phrase_match and len(q_tokens) >= 2:
            bigrams = [" ".join(q_tokens[i : i + 2]) for i in range(len(q_tokens) - 1)]
            matched_bigrams = sum(1 for bg in bigrams if bg in d_lower)
            phrase_match = matched_bigrams / len(bigrams) if bigrams else 0.0

        if retriever and item.doc_id in retriever._documents:
            doc_emb = retriever._documents[item.doc_id].embedding
            cos = (
                cosine_similarity(q_emb.dense_vector, doc_emb.dense_vector)
                if doc_emb
                else 0.0
            )
        else:
            doc_emb = compute_lexical_embedding(item.text, dimension=embedding_dim)
            cos = cosine_similarity(q_emb.dense_vector, doc_emb.dense_vector)

        rerank_score = (
            cosine_weight * cos
            + coverage_weight * coverage
            + phrase_weight * phrase_match
        )
        combined_score = (
            lexical_weight * norm_lex
            + rerank_score
        )

        scored_items.append((combined_score, norm_lex, rerank_score, item))

    scored_items.sort(key=lambda x: x[0], reverse=True)

    reranked: list[RetrievalResult] = []
    for rank, (comb, norm_lex, r_score, item) in enumerate(scored_items[:top_k], start=1):
        reranked.append(
            RetrievalResult(
                doc_id=item.doc_id,
                text=item.text,
                lexical_score=item.lexical_score,
                rerank_score=round(r_score, 4),
                combined_score=round(comb, 4),
                rank=rank,
                metadata=item.metadata,
            )
        )
    return reranked


LexicalRetriever = BM25Retriever
