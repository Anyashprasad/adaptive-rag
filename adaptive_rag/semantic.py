from __future__ import annotations

from abc import ABC, abstractmethod
import hashlib
import numpy as np
from .models import DocumentChunk, SearchHit


class SemanticBackend(ABC):
    name = "abstract"

    @abstractmethod
    def build(self, chunks: list[DocumentChunk]) -> None: ...

    @abstractmethod
    def search(self, query: str, k: int = 8) -> list[SearchHit]: ...


class SentenceTransformerBackend(SemanticBackend):
    name = "sentence-transformer"

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self.chunks: list[DocumentChunk] = []
        self.embeddings: np.ndarray | None = None

    def build(self, chunks: list[DocumentChunk]) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "SentenceTransformer backend is not installed. Install with "
                "`pip install -e '.[semantic]'` or use the hashing backend for smoke tests."
            ) from exc
        self.model = SentenceTransformer(self.model_name)
        self.chunks = chunks
        texts = [c.text for c in chunks]
        if hasattr(self.model, "encode_document"):
            self.embeddings = self.model.encode_document(texts, normalize_embeddings=True, show_progress_bar=False)
        else:
            self.embeddings = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        self.embeddings = np.asarray(self.embeddings, dtype=np.float32)

    def search(self, query: str, k: int = 8) -> list[SearchHit]:
        if self.model is None or self.embeddings is None or not self.chunks:
            return []
        if hasattr(self.model, "encode_query"):
            q = self.model.encode_query(query, normalize_embeddings=True)
        else:
            q = self.model.encode(query, normalize_embeddings=True)
        q = np.asarray(q, dtype=np.float32)
        scores = self.embeddings @ q
        idx = np.argsort(scores)[::-1][:k]
        return [
            SearchHit(
                self.chunks[int(i)], float(scores[int(i)]), "semantic",
                {"cosine": round(float(scores[int(i)]), 4), "backend": self.name},
            ) for i in idx
        ]


class HashingSemanticBackend(SemanticBackend):
    """Deterministic offline embedding-like backend for tests/smoke demos.

    It validates orchestration without network/model downloads; do not use its quality numbers
    as claims about dense embedding retrieval.
    """

    name = "hashing-smoke"

    def __init__(self, dims: int = 512):
        self.dims = dims
        self.chunks: list[DocumentChunk] = []
        self.embeddings: np.ndarray | None = None

    def _embed(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dims, dtype=np.float32)
        words = text.lower().split()
        grams = words + [" ".join(words[i:i+2]) for i in range(max(0, len(words)-1))]
        for g in grams:
            h = int(hashlib.blake2b(g.encode(), digest_size=8).hexdigest(), 16)
            vec[h % self.dims] += 1.0
        norm = np.linalg.norm(vec)
        return vec / norm if norm else vec

    def build(self, chunks: list[DocumentChunk]) -> None:
        self.chunks = chunks
        self.embeddings = np.stack([self._embed(c.text) for c in chunks]) if chunks else np.zeros((0, self.dims))

    def search(self, query: str, k: int = 8) -> list[SearchHit]:
        if self.embeddings is None or not len(self.chunks):
            return []
        q = self._embed(query)
        scores = self.embeddings @ q
        idx = np.argsort(scores)[::-1][:k]
        return [
            SearchHit(
                self.chunks[int(i)], float(scores[int(i)]), "semantic",
                {"cosine": round(float(scores[int(i)]), 4), "backend": self.name},
            ) for i in idx
        ]


class LSASemanticBackend(SemanticBackend):
    """Offline Latent Semantic Analysis backend using only NumPy.

    This is a genuine latent semantic baseline for validation without model downloads.
    For production-quality semantic retrieval, use SentenceTransformerBackend.
    """

    name = "lsa-local"

    def __init__(self, dims: int = 64):
        self.dims = dims
        self.chunks: list[DocumentChunk] = []
        self.vocab: dict[str, int] = {}
        self.idf: np.ndarray | None = None
        self.projection: np.ndarray | None = None
        self.embeddings: np.ndarray | None = None

    @staticmethod
    def _terms(text: str) -> list[str]:
        import re
        return re.findall(r"[a-z0-9][a-z0-9_-]+", text.lower())

    def build(self, chunks: list[DocumentChunk]) -> None:
        self.chunks = chunks
        docs = [self._terms(c.text) for c in chunks]
        vocab_terms = sorted({t for d in docs for t in d})
        self.vocab = {t: i for i, t in enumerate(vocab_terms)}
        if not chunks or not self.vocab:
            self.idf = np.zeros(0, dtype=np.float32)
            self.projection = np.zeros((0, 0), dtype=np.float32)
            self.embeddings = np.zeros((0, 0), dtype=np.float32)
            return
        x = np.zeros((len(chunks), len(self.vocab)), dtype=np.float64)
        df = np.zeros(len(self.vocab), dtype=np.float64)
        for r, terms in enumerate(docs):
            counts: dict[int, int] = {}
            for term in terms:
                idx = self.vocab[term]
                counts[idx] = counts.get(idx, 0) + 1
            for idx, count in counts.items():
                x[r, idx] = 1.0 + np.log(float(count))
                df[idx] += 1.0
        self.idf = (np.log((1.0 + len(chunks)) / (1.0 + df)) + 1.0).astype(np.float32)
        x *= self.idf
        rank = max(1, min(self.dims, max(1, len(chunks) - 1), len(self.vocab)))
        _, _, vt = np.linalg.svd(x, full_matrices=False)
        self.projection = vt[:rank].T.astype(np.float32)
        latent = x @ self.projection
        norms = np.linalg.norm(latent, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.embeddings = (latent / norms).astype(np.float32)

    def search(self, query: str, k: int = 8) -> list[SearchHit]:
        if self.embeddings is None or self.projection is None or self.idf is None or not self.chunks:
            return []
        q = np.zeros(len(self.vocab), dtype=np.float32)
        counts: dict[int, int] = {}
        for term in self._terms(query):
            idx = self.vocab.get(term)
            if idx is not None:
                counts[idx] = counts.get(idx, 0) + 1
        for idx, count in counts.items():
            q[idx] = (1.0 + np.log(float(count))) * self.idf[idx]
        latent = q @ self.projection
        norm = np.linalg.norm(latent)
        if norm:
            latent = latent / norm
        scores = self.embeddings @ latent
        idxs = np.argsort(scores)[::-1][:k]
        return [
            SearchHit(
                self.chunks[int(i)], float(scores[int(i)]), "semantic",
                {"cosine": round(float(scores[int(i)]), 4), "backend": self.name},
            ) for i in idxs
        ]
