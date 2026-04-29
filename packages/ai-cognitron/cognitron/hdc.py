"""Hyperdimensional Computing (HDC) encoder — built from scratch.

A Vector Symbolic Architecture using 10,000-dimensional binary hypervectors.
This replaces pretrained text embeddings entirely. Concepts are composed
through three primitive operations:

    bind(a, b)    — XOR (associates two concepts; reversible)
    bundle(*v)    — majority vote (superposes; commutative; lossy)
    permute(v, n) — cyclic shift (encodes order/role)

Similarity is Hamming distance (or normalized cosine on the bipolar form).

Why bipolar (-1/+1) internally: addition then sign() reproduces majority vote
exactly with no tie-breaking ambiguity, and cosine similarity is well-defined.
We expose binary {0,1} vectors at the API boundary for compactness when stored.

This is a complete reimplementation; we use numpy only as a tensor primitive.
No external embedding model is consulted.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

DEFAULT_DIM = 10_000


# ---------------------------------------------------------------------------
# Primitive operations on bipolar hypervectors (values are -1 / +1)
# ---------------------------------------------------------------------------


def random_hypervector(dim: int = DEFAULT_DIM, rng: np.random.Generator | None = None) -> np.ndarray:
    """Sample a fresh i.i.d. bipolar hypervector. Two independent samples are
    near-orthogonal by Johnson-Lindenstrauss; this is the foundation of HDC."""
    rng = rng or np.random.default_rng()
    bits = rng.integers(0, 2, size=dim, dtype=np.int8)
    return (bits * 2 - 1).astype(np.int8)  # {0,1} -> {-1,+1}


def bind(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Binding: element-wise multiply (== XOR for bipolar). Self-inverse:
    bind(bind(a,b), b) == a. Used to associate role with filler ('color' bound
    to 'red')."""
    return (a * b).astype(np.int8)


def bundle(*vectors: np.ndarray) -> np.ndarray:
    """Bundling: element-wise sum then sign(). Superposes multiple concepts
    into one vector that is similar to all of them. Lossy but graceful."""
    if not vectors:
        raise ValueError("bundle requires at least one vector")
    stacked = np.stack(vectors, axis=0).astype(np.int32)
    summed = stacked.sum(axis=0)
    # Break ties deterministically toward +1 — important so that bundling the
    # same set in any order produces the same vector.
    summed = np.where(summed == 0, 1, summed)
    return np.sign(summed).astype(np.int8)


def permute(v: np.ndarray, n: int = 1) -> np.ndarray:
    """Cyclic shift by n positions. Used to encode order/position/role."""
    return np.roll(v, n).astype(np.int8)


# ---------------------------------------------------------------------------
# Similarity functions
# ---------------------------------------------------------------------------


def hamming(a: np.ndarray, b: np.ndarray) -> float:
    """Normalized Hamming distance for bipolar vectors. 0 = identical, 1 = anti."""
    return float(np.mean(a != b))


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity. For bipolar vectors of the same dim this equals
    1 - 2 * hamming(a, b)."""
    a_f = a.astype(np.float32)
    b_f = b.astype(np.float32)
    denom = np.linalg.norm(a_f) * np.linalg.norm(b_f)
    if denom == 0.0:
        return 0.0
    return float(np.dot(a_f, b_f) / denom)


# ---------------------------------------------------------------------------
# Encoder: text -> hypervector
# ---------------------------------------------------------------------------


@dataclass
class HDCEncoder:
    """Encodes text into a hypervector without any pretrained model.

    Strategy:
      1. Each unique token (after lowercasing + simple split) deterministically
         maps to a fresh random hypervector. The mapping is seeded by a hash of
         the token itself, so the same token always produces the same vector,
         across processes and machines, with no learned weights.
      2. Token order is encoded by permuting each token's vector by its
         position index, then bundling. This makes "dog bites man" different
         from "man bites dog" while keeping similar word sets close.
      3. Optional n-gram binding adds local-order sensitivity.

    Notes for the from-scratch constraint: this is *not* a learned embedding.
    Vectors are derived purely from token identity via hashing. Quality is
    validated by the bundled tests; if poor on the actual workload, the
    architect-recommended fallback is to LSH random projection (see
    `random_projection_encode`).
    """

    dim: int = DEFAULT_DIM
    # Default OFF for retrieval workloads — n-gram bind bundles add noise
    # that overwhelms the bag-of-tokens signal on short texts. Empirically
    # turning ngrams on dropped same-topic similarity from +0.12 to +0.04.
    use_ngrams: bool = False
    ngram_size: int = 2
    use_position: bool = False  # disabled by default — see encode() docstring
    _cache: dict[str, np.ndarray] | None = None

    def __post_init__(self) -> None:
        self._cache = {}

    def _token_vector(self, token: str) -> np.ndarray:
        """Deterministic per-token hypervector via SHA-256 seeded RNG."""
        cache = self._cache
        assert cache is not None
        if token in cache:
            return cache[token]
        seed_bytes = hashlib.sha256(token.encode("utf-8")).digest()[:8]
        seed = int.from_bytes(seed_bytes, "big")
        rng = np.random.default_rng(seed)
        v = random_hypervector(self.dim, rng=rng)
        cache[token] = v
        return v

    # A tiny stoplist. The from-scratch constraint forbids learned IDF, but
    # filtering surface noise is plain text processing, not ML. Without this
    # the bundle is dominated by 'the', 'a', 'and', etc. and topic
    # separation collapses (validate_hdc test_text_separation made this
    # failure visible).
    _STOPWORDS = frozenset(
        {
            "a", "an", "and", "or", "but", "the", "is", "are", "was", "were",
            "be", "been", "being", "have", "has", "had", "do", "does", "did",
            "i", "you", "he", "she", "it", "we", "they", "me", "him", "her",
            "us", "them", "my", "your", "his", "its", "our", "their",
            "this", "that", "these", "those", "to", "of", "in", "on", "at",
            "by", "for", "with", "from", "as", "into", "about",
        }
    )

    @classmethod
    def _tokenize(cls, text: str) -> list[str]:
        """Lowercase, split on non-alphanumerics, drop stopwords + 1-char tokens."""
        out: list[str] = []
        cur: list[str] = []
        for ch in text.lower():
            if ch.isalnum():
                cur.append(ch)
            elif cur:
                out.append("".join(cur))
                cur = []
        if cur:
            out.append("".join(cur))
        return [t for t in out if len(t) > 1 and t not in cls._STOPWORDS]

    def encode(self, text: str) -> np.ndarray:
        """Encode a string into a single hypervector.

        Strategy:
          - Bag-of-tokens bundle by default — best for retrieval (HDC has
            no learned semantics, so similarity reduces to lexical overlap;
            order-sensitive encoding makes shared tokens position-dependent
            and *destroys* that overlap signal).
          - Optional n-gram binding adds local-order awareness without
            destroying the bag similarity.
          - Optional positional permutation is available behind
            `use_position=True` for downstream tasks that need order
            sensitivity (e.g. classifying "dog bites man" vs "man bites
            dog"), but is OFF by default for retrieval compatibility.
        """
        tokens = self._tokenize(text)
        if not tokens:
            return np.zeros(self.dim, dtype=np.int8)

        token_vecs = [self._token_vector(t) for t in tokens]

        if self.use_position:
            token_vecs = [permute(v, i) for i, v in enumerate(token_vecs)]

        v = bundle(*token_vecs)

        if self.use_ngrams and len(tokens) >= self.ngram_size:
            ngram_vecs: list[np.ndarray] = []
            for i in range(len(tokens) - self.ngram_size + 1):
                window = [self._token_vector(tokens[i + k]) for k in range(self.ngram_size)]
                acc = window[0]
                for w in window[1:]:
                    acc = bind(acc, w)
                ngram_vecs.append(acc)
            ngram_bundle = bundle(*ngram_vecs)
            v = bundle(v, ngram_bundle)

        return v

    def encode_many(self, texts: list[str]) -> np.ndarray:
        """Vectorized encoding. Returns shape (len(texts), dim)."""
        return np.stack([self.encode(t) for t in texts], axis=0)


# ---------------------------------------------------------------------------
# Random-projection fallback encoder (architect-recommended risk mitigation)
# ---------------------------------------------------------------------------


def random_projection_encode(
    text: str, dim: int = DEFAULT_DIM, projection_seed: int = 1337
) -> np.ndarray:
    """LSH-style fallback: hash character n-grams into a sparse +/-1 vector,
    then sign() the projection. Faster, weaker than full HDC, but uniform in
    quality on tiny datasets."""
    rng = np.random.default_rng(projection_seed)
    proj = rng.choice([-1, 1], size=dim).astype(np.int8)
    acc = np.zeros(dim, dtype=np.int32)
    for n in (3, 4, 5):
        for i in range(len(text) - n + 1):
            gram = text[i : i + n].lower()
            h = int.from_bytes(
                hashlib.sha256(gram.encode("utf-8")).digest()[:8], "big"
            ) % dim
            acc[h] += 1
    signal = acc.astype(np.int32) * proj
    signal = np.where(signal == 0, 1, signal)
    return np.sign(signal).astype(np.int8)
