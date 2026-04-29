# Branch: `ai-cognitron`

**Owner:** AI/ML engineer
**Path of record:** `packages/ai-cognitron/`

## Charter

The from-scratch model. Owns:
- `hdc.py` — hyperdimensional computing primitives + text encoder
- `pnn.py` — Particle Neural Network (free-particle topology, wave inference)
- `pgd.py` — Particle Gradient Descent trainer + SGD fallback
- `geometric_index.py` — custom brute-force cosine search

## Constraints (non-negotiable)

- **No pretrained models.** Not even a tokenizer. Hash → hypervector.
- **No FAISS / pgvector / HNSW.** Custom geometric index only.
- **No autograd framework for the PNN.** NumPy primitives.
- PyTorch is allowed for tensor utilities only, not for parameters.

## What's been validated

- `python -m cognitron.validate_hdc` — encoder separation passes
- `python -m cognitron.demo_train` — PGD monotone-improves on toy task

## Open work

- [ ] Larger benchmark (MNIST class-association)
- [ ] Hyperparameter sweep on PGD constants (w, c1, c2, eta, sigma)
- [ ] Optional kd-tree for >10k particles
- [ ] Profile spatial-hash hit rate at 5k / 10k particles

## Failure modes (analyst-flagged)

1. **PGD divergence.** Use `SGDFallback` with the same interface.
2. **HDC quality.** Validate before any downstream wiring.
3. **Numerical instability at large fields.** Velocity-clip enforced.

## Public surface

```python
from cognitron import (
    HDCEncoder,
    ParticleNetwork,
    ParticleGradientDescent,
    SGDFallback,
    GeometricIndex,
)
```
