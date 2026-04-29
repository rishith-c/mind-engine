"""Validate HDC encoder quality in isolation.

Runs deterministic tests showing:
  1. Random hypervectors are near-orthogonal (mean cosine ~ 0).
  2. bind/unbind is exact: unbind(bind(a,b), b) == a.
  3. bundle preserves similarity to its members.
  4. Encoded similar sentences cluster (intra-class > inter-class cosine).

Intended to be run as: python -m cognitron.validate_hdc
"""

from __future__ import annotations

import numpy as np

from cognitron.hdc import (
    DEFAULT_DIM,
    HDCEncoder,
    bind,
    bundle,
    cosine,
    random_hypervector,
)


def test_orthogonality() -> None:
    rng = np.random.default_rng(0)
    pairs = [(random_hypervector(rng=rng), random_hypervector(rng=rng)) for _ in range(200)]
    sims = [cosine(a, b) for a, b in pairs]
    mean = float(np.mean(sims))
    std = float(np.std(sims))
    print(f"[orthogonality] mean cosine of random pairs = {mean:+.4f}  (should be ~0)")
    print(f"[orthogonality] std                          = {std:+.4f}  (~ 1/sqrt(D))")
    assert abs(mean) < 0.05, f"random hypervectors not orthogonal: mean={mean}"
    expected_std = 1.0 / np.sqrt(DEFAULT_DIM)
    assert std < 5 * expected_std, f"std too high: {std} vs expected {expected_std}"


def test_bind_inverse() -> None:
    rng = np.random.default_rng(1)
    a = random_hypervector(rng=rng)
    b = random_hypervector(rng=rng)
    recovered = bind(bind(a, b), b)
    assert np.array_equal(a, recovered), "bind is not its own inverse"
    print("[bind-inverse] bind(bind(a,b), b) == a   PASS")


def test_bundle_preserves_similarity() -> None:
    rng = np.random.default_rng(2)
    members = [random_hypervector(rng=rng) for _ in range(5)]
    superposed = bundle(*members)
    for i, m in enumerate(members):
        sim = cosine(superposed, m)
        print(f"[bundle] cosine(superposed, member_{i}) = {sim:+.3f}")
        assert sim > 0.2, f"bundle lost member {i} (sim={sim})"


def test_text_separation() -> None:
    """HDC has no learned semantics; it detects lexical overlap. The test
    therefore uses realistic same-topic prose where vocabulary genuinely
    repeats (as it does in any real corpus)."""
    enc = HDCEncoder()
    cooking = [
        "the bread oven preheats while the dough rises on the counter",
        "knead the bread dough then let it rise in a warm oven",
        "the dough was kneaded and rose nicely once placed in the oven",
    ]
    space = [
        "the rocket launches astronauts into orbit aboard the space station",
        "astronauts on the space station study microgravity from orbit",
        "the orbiting space station carries astronauts and instruments",
    ]

    cook_vecs = enc.encode_many(cooking)
    space_vecs = enc.encode_many(space)

    intra_cook = [
        cosine(cook_vecs[i], cook_vecs[j])
        for i in range(len(cooking))
        for j in range(i + 1, len(cooking))
    ]
    intra_space = [
        cosine(space_vecs[i], space_vecs[j])
        for i in range(len(space))
        for j in range(i + 1, len(space))
    ]
    inter = [cosine(c, s) for c in cook_vecs for s in space_vecs]

    intra = float(np.mean(intra_cook + intra_space))
    inter_mean = float(np.mean(inter))
    margin = intra - inter_mean
    print(f"[text] intra-class cosine = {intra:+.3f}")
    print(f"[text] inter-class cosine = {inter_mean:+.3f}")
    print(f"[text] margin (intra - inter) = {margin:+.3f}  (positive = encoder works)")
    # On 6-word sentences with no learned IDF, margins of 0.005-0.05 are
    # realistic for HDC. We require positive separation only — the demo
    # network uses longer thoughts where margin is sharper.
    assert margin > 0.005, f"encoder fails to separate topics: margin={margin}"


def main() -> None:
    print("=" * 60)
    print("HDC Encoder Validation — from-scratch hyperdimensional computing")
    print("=" * 60)
    test_orthogonality()
    print()
    test_bind_inverse()
    print()
    test_bundle_preserves_similarity()
    print()
    test_text_separation()
    print()
    print("All HDC validation checks PASSED.")


if __name__ == "__main__":
    main()
