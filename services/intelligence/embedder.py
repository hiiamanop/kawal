from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Sequence

import numpy as np

logger = logging.getLogger("embedder")


class TextEmbedder:
    """Interface for complaint embedding generators."""

    def embed_text(self, text: str) -> list[float]:
        raise NotImplementedError

    def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        raise NotImplementedError


class DeterministicFallbackEmbedder(TextEmbedder):
    """Deterministic hash-based bag-of-words pseudo-embedder for test or low-resource modes.

    Produces a fixed 768-dimensional normalized vector using salted hashing.
    """

    def __init__(self, dim: int = 768) -> None:
        self.dim = dim

    def embed_text(self, text: str) -> list[float]:
        tokens = [t.strip().lower() for t in text.split() if t.strip()]
        if not tokens:
            vec = np.zeros(self.dim, dtype=np.float32)
            vec[0] = 1.0
            return vec.tolist()

        vec = np.zeros(self.dim, dtype=np.float32)
        for token in tokens:
            h = int(hashlib.sha256(token.encode("utf-8")).hexdigest()[:8], 16)
            idx = h % self.dim
            vec[idx] += 1.0

        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        else:
            vec[0] = 1.0
        return vec.tolist()

    def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        return [self.embed_text(t) for t in texts]


class IndoBertEmbedder(TextEmbedder):
    """CPU-optimized IndoBERT DAPT embedder with attention-weighted mean pooling and L2 normalization."""

    _instance: IndoBertEmbedder | None = None

    def __init__(self, model_path: str | Path | None = None) -> None:
        cand_paths = [
            Path(model_path).resolve() if model_path else None,
            Path("artifacts/hifi-v3-dapt").resolve(),
            Path("artifacts/city12-v4-multitask").resolve(),
        ]
        chosen_path: Path | None = None
        for p in cand_paths:
            if p is not None and (p / "config.json").is_file():
                chosen_path = p
                break

        if chosen_path is None:
            logger.warning("No IndoBERT artifact directory found; falling back to deterministic embedder")
            self._fallback = DeterministicFallbackEmbedder(dim=768)
            self._model = None
            self._tokenizer = None
            return

        try:
            import torch
            from transformers import AutoModel, AutoTokenizer

            self._torch = torch
            self._tokenizer = AutoTokenizer.from_pretrained(str(chosen_path))
            self._model = AutoModel.from_pretrained(str(chosen_path))
            self._model.eval()
            self._fallback = None
            logger.info("IndoBertEmbedder initialized from %s", chosen_path)
        except Exception as exc:
            logger.warning("Failed to load PyTorch/Transformers embedder (%s); using deterministic fallback", exc)
            self._fallback = DeterministicFallbackEmbedder(dim=768)
            self._model = None
            self._tokenizer = None

    @classmethod
    def get_instance(cls, model_path: str | Path | None = None) -> IndoBertEmbedder:
        if cls._instance is None:
            cls._instance = cls(model_path=model_path)
        return cls._instance

    def embed_text(self, text: str) -> list[float]:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        if self._fallback is not None or self._model is None or self._tokenizer is None:
            return self._fallback.embed_batch(texts) if self._fallback else []

        if not texts:
            return []

        cleaned = [t.strip() if t.strip() else "<empty>" for t in texts]
        with self._torch.no_grad():
            inputs = self._tokenizer(
                cleaned,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            )
            out = self._model(**inputs)
            # Attention-weighted mean pooling
            mask = inputs["attention_mask"].unsqueeze(-1).expand(out.last_hidden_state.size()).float()
            sum_embeddings = self._torch.sum(out.last_hidden_state * mask, 1)
            sum_mask = self._torch.clamp(mask.sum(1), min=1e-9)
            mean_pooled = sum_embeddings / sum_mask

            vecs = mean_pooled.cpu().numpy()
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            normed = vecs / np.maximum(norms, 1e-9)
            return normed.astype(float).tolist()


def cosine_similarity(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
    """Compute cosine similarity between two float vectors.

    Assumes vectors are already L2-normalized; otherwise computes normalized dot product.
    """
    a = np.asarray(vec_a, dtype=np.float32)
    b = np.asarray(vec_b, dtype=np.float32)
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-9:
        return 0.0
    return float(np.dot(a, b) / denom)
