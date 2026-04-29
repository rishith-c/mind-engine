# Pre-build agent validation reports

Three parallel agents reviewed the design before any code was written.
Their findings are summarized here and have been baked into the
implementation. Full reports also visible in this commit's history.

## 1. Novelty researcher

**PNN — score 7/10.** The mash-up (spatial neurons + PSO-gradient hybrid +
HDC + wave inference) appears to be unpublished. Closest priors:
- Wołczyk et al. 2019 — biologically-inspired spatial NNs (2D, gradient-only)
- Kumar 2024 — particle-based training (no spatial embedding)
- Miyato et al. 2024 — Kuramoto oscillatory neurons (synchronization, not waves)
- Standard PSO-CNN papers (PSO+SGD hybrids exist, but not for free particles)
- Kanerva-style HDC is well-trodden

**Morpheus 3D NCA — score 4/10.** 3D voxel NCA with color is published
(Sudhakaran 2021); multi-modal supervision is published (MeshNCA 2023).
**The audio-per-cell modality is the only fresh contribution.** Action:
re-pitch as "first synesthetic NCA" rather than "3D NCA."

## 2. Architecture / feasibility reviewer

Top three risks:
1. PGD convergence — research-grade optimizer, will likely stall.
2. 3D NCA gradient memory — needs aggressive checkpointing.
3. Two from-scratch systems in two weeks is a stretch goal².

Concrete simplifications applied:
- HDC over learned embeddings ✓ (HDC is the constraint anyway)
- Custom geometric index over HNSW ✓ (brute-force, <10k entries)
- Dense 32³ grid, no octree ✓
- Server-side training only ✓
- Audio sonification of activations as the third modality ✓
- CPU reference impls for every WGSL shader ✓

## 3. Risk / failure-mode analyst

Five ranked failure modes with mitigations now in code:

| # | Failure                              | Probability × Impact | Mitigation                          |
| - | ------------------------------------ | -------------------- | ----------------------------------- |
| 1 | PGD divergence                       | 0.75 × catastrophic  | `SGDFallback` drop-in replacement   |
| 2 | HDC produces uninformative vectors   | 0.65 × high          | `validate_hdc.py` gates downstream  |
| 3 | NCA simulation cost explodes         | 0.60 × high          | 32³ cap, server inference, cache    |
| 4 | Branch merge hell                    | 0.70 × medium        | Pydantic + TS shared schemas        |
| 5 | Demo wifi + Supabase latency         | 0.50 × medium        | Stateless in-memory API by default  |

Demo-day risks:
- WebGPU on Safari → R3F automatically falls back to WebGL.
- Cold-shader compile freeze → pre-warm on page load.
- Don't train live during the demo — it's pre-baked.

Cut list if behind:
- PGD → SGD
- 3D Morpheus → 2D
- Skip live training UI
