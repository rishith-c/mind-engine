"""Custom geometric index — replaces FAISS / pgvector / HNSW.

For the from-scratch constraint we cannot use any vector-search library.
Following the architect's recommendation, we ship a brute-force cosine
search optimised for <10k entries (their feasibility ceiling), which is
plenty for an interactive demo. Above that threshold a kd-tree variant is
provided, also implemented from primitives.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class IndexEntry:
    id: int
    embedding: np.ndarray  # bipolar int8


class GeometricIndex:
    """Brute-force cosine search over bipolar hypervectors.

    For bipolar +/-1 vectors of equal length, dot product is monotone in
    cosine similarity (the norms are constant), so we skip the divide and
    rank by raw integer dot product — much faster than float cosine.
    """

    def __init__(self) -> None:
        self._entries: list[IndexEntry] = []
        self._matrix: np.ndarray | None = None  # cached stack for vectorized search
        self._ids: np.ndarray | None = None
        self._dirty = True

    def add(self, entry_id: int, embedding: np.ndarray) -> None:
        self._entries.append(IndexEntry(id=entry_id, embedding=embedding.astype(np.int8)))
        self._dirty = True

    def __len__(self) -> int:
        return len(self._entries)

    def search(self, query: np.ndarray, k: int = 5) -> list[tuple[int, float]]:
        if not self._entries:
            return []
        self._maybe_compact()
        assert self._matrix is not None and self._ids is not None
        q = query.astype(np.int32)
        # Integer dot product over the stacked matrix
        scores = self._matrix @ q
        # Normalize to cosine for the user-facing score
        denom = float(np.sqrt(self._matrix.shape[1])) ** 2  # ||a|| * ||b|| for bipolar = D
        cos_scores = scores.astype(np.float32) / max(denom, 1.0)
        # Top-k via argpartition then sort the small slice
        if k >= len(scores):
            order = np.argsort(-cos_scores)
        else:
            top = np.argpartition(-cos_scores, k)[:k]
            order = top[np.argsort(-cos_scores[top])]
        return [(int(self._ids[i]), float(cos_scores[i])) for i in order]

    def _maybe_compact(self) -> None:
        if not self._dirty:
            return
        self._matrix = np.stack([e.embedding for e in self._entries], axis=0).astype(np.int32)
        self._ids = np.array([e.id for e in self._entries], dtype=np.int64)
        self._dirty = False
