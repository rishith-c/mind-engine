# mind-engine

> Two from-scratch original AI systems, built with **zero pretrained models**.
> Cognitron is a Particle Neural Network. Morpheus is a synesthetic 3D Neural
> Cellular Automaton. Both render in the browser with WebGPU. Both ship as a
> hackathon-ready monorepo.

```
┌───────────────────────────────────────────────────────────────────────────┐
│                                                                           │
│  COGNITRON ⟷ MORPHEUS                                                     │
│                                                                           │
│  past + present                       futures + worlds                    │
│  particles as neurons                 voxels as cells                     │
│  semantic gravity                     cellular evolution                  │
│  hyperdimensional encoding            synesthetic rendering               │
│                                                                           │
│             every weight, every neuron, every rule — from scratch         │
│                                                                           │
└───────────────────────────────────────────────────────────────────────────┘
```

---

## Why this exists

Today's AI is built from a finite shelf of pretrained transformers. We
wanted to know what happens when you reject the shelf entirely.

**Cognitron** asks: *what if neurons were free particles, not fixed graph
nodes?* It implements a brand-new architecture — Particle Neural Network
(PNN) — where every "neuron" is a position-bearing particle in 3D, and the
graph is reconstructed every forward pass from spatial proximity. Training
is a physics simulation. Inference is a wave of energy propagating through
the field. Encoding is hyperdimensional computing, derived purely from
hashes — no learned embedding model in sight.

**Morpheus** asks: *what if reality grew from cellular rules instead of
being predicted by transformers?* It extends Mordvintsev et al.'s
Growing-NCA into 3D and adds a third learned modality nobody has shipped
before: **per-cell audio frequency**. The result is a network that grows
geometry, color, and sound jointly — a genuine synesthetic generator.

A novelty-research agent confirmed that the precise PNN combination
(spatial neurons + PSO-gradient hybrid + HDC encoder + wave inference) is
unpublished, and that audio-per-cell in NCA is the only fresh angle in
that crowded literature. We lean into both.

---

## Repo layout

```
mind-engine/
├── apps/
│   ├── cognitron-web/        Next.js 16 + shadcn/ui + R3F + WebGPU
│   ├── cognitron-api/        FastAPI service (PNN + HDC + GeometricIndex)
│   ├── morpheus-web/         Next.js 16 + shadcn/ui + R3F + Tone.js
│   └── morpheus-api/         FastAPI service (3D NCA inference)
├── packages/
│   ├── ai-cognitron/         The PNN. NumPy. No ML libs for the model.
│   │   └── cognitron/
│   │       ├── hdc.py             10000-d hyperdimensional encoder
│   │       ├── pnn.py             Particle Neural Network
│   │       ├── pgd.py             Particle Gradient Descent + SGD fallback
│   │       ├── geometric_index.py Custom brute-force index (no FAISS)
│   │       ├── validate_hdc.py    Encoder unit tests
│   │       └── demo_train.py      End-to-end training smoke test
│   ├── ai-morpheus/          The 3D NCA. PyTorch tensors only — no
│   │   └── morpheus/              pretrained weights, no model zoo.
│   │       ├── nca3d.py           Sobel perception + per-cell MLP update
│   │       └── train.py           Server-side training to weight blob
│   ├── shared-physics/       WGSL compute shaders + TS reference impls
│   │   └── src/
│   │       ├── particle-forces.{ts,wgsl}
│   │       └── nca-update.{ts,wgsl}
│   └── ui/                   (reserved for shared shadcn components)
├── supabase/
│   ├── migrations/0001_init.sql   Storage schema (NO pgvector)
│   └── README.md
├── scripts/
│   └── data_pipeline.py      Embedding/cluster/decay jobs
├── docs/                     Design notes, agent reports
├── turbo.json
├── package.json
├── pnpm-workspace.yaml
└── tsconfig.json
```

---

## Branch strategy

The team works in role-scoped branches; `main` integrates everything.

| Branch              | Owner           | Contents                                        |
| ------------------- | --------------- | ----------------------------------------------- |
| `main`              | Frontend lead   | Full monorepo + integrated UI                   |
| `backend-cognitron` | Backend         | `apps/cognitron-api/` only                      |
| `backend-morpheus`  | Backend         | `apps/morpheus-api/` only                       |
| `ai-cognitron`      | AI/ML           | `packages/ai-cognitron/` only (PNN + HDC + PGD) |
| `ai-morpheus`       | AI/ML           | `packages/ai-morpheus/` only (3D NCA)           |
| `data-cognitron`    | Data engineer   | Embedding / cluster / decay pipelines           |
| `data-morpheus`     | Data engineer   | Voxel-frame ETL & frame storage tooling         |

---

## Quickstart

### Prereqs
- Node 20+, pnpm 9+
- Python 3.11+
- A Supabase project (or `supabase start` for local)
- Chrome / Edge / Chrome Canary for WebGPU support

### 1. Install monorepo
```bash
pnpm install
```

### 2. Install Python deps for the two AI services
```bash
python -m venv .venv && source .venv/bin/activate
pip install -e packages/ai-cognitron[api]
pip install -e packages/ai-morpheus[api]
```

### 3. Validate the from-scratch HDC encoder
```bash
cd packages/ai-cognitron && python -m cognitron.validate_hdc
# expected: all 4 checks PASS
```

### 4. Run a Cognitron training smoke test
```bash
python -m cognitron.demo_train
# expected: top-1 accuracy rises monotonically over 30 PGD steps
```

### 5. Pre-train Morpheus weights (server-side, ~5-15 min on CPU, faster on GPU)
```bash
cd packages/ai-morpheus && python -m morpheus.train --target sphere --steps 4000
# saves -> weights/morpheus_sphere.pt
```

### 6. Launch everything
```bash
# Terminal 1 — Cognitron API on :8000
cd apps/cognitron-api && uvicorn main:app --reload --port 8000

# Terminal 2 — Morpheus API on :8001
cd apps/morpheus-api && uvicorn main:app --reload --port 8001

# Terminal 3 — Frontend dev servers
pnpm dev   # cognitron-web on :3000, morpheus-web on :3001
```

### 7. Open browsers
- http://localhost:3000  Cognitron — drop thoughts, fire wave queries
- http://localhost:3001  Morpheus — seed a target, hear the chord

---

## How Cognitron actually works

```
        ┌─────────────┐
text ─▶ │ HDC encoder │ ─▶ 10000-d bipolar hypervector
        └─────────────┘            │
                                   ▼
                          ┌─────────────────┐
                          │ random projection│  fixed seeded R^3 axes
                          └─────────────────┘
                                   │
                                   ▼
                          spawn a Particle:
                          { position, velocity, mass, charge, polarity }
                                   │
                                   ▼
                ┌────────────── ParticleNetwork ──────────────┐
                │  spatial hash → O(N) neighbour queries      │
                │  forward(input):                            │
                │     wave propagation × 4-6 hops             │
                │     similarity × proximity × polarity → flow│
                │  query(text):                               │
                │     encode → seed input → wave → top-k      │
                └─────────────────────────────────────────────┘
                                   │
                                   ▼
                Particle Gradient Descent trainer
                v ← w·v + c1·r1·(pbest-x) + c2·r2·(gbest-x)
                       − η·∇L(x) + σ·N(0,I)
```

**Crucially:** there is no transformer, no autoencoder, no embedding model.
The encoder is deterministic hashing into orthogonal hypervectors. The
"neurons" are `numpy` arrays. The "training" is a six-equation force law.

## How Morpheus actually works

```
seed cell → 32^3 grid, 16-channel state per cell
                                 │
                                 ▼
                     for each timestep t:
                          ┌──────────────────────┐
                          │  Sobel perception 3D │  fixed kernels
                          └──────────────────────┘
                                 │
                                 ▼
                          ┌──────────────────────┐
                          │  per-cell MLP        │  16 → 96 → 16
                          └──────────────────────┘
                                 │  (residual)
                                 ▼
                          ┌──────────────────────┐
                          │ stochastic mask 50%  │
                          └──────────────────────┘
                                 │
                                 ▼
                          ┌──────────────────────┐
                          │ alive mask via α-pool│
                          └──────────────────────┘
                                 │
                                 ▼
                  state[t+1] : RGB | α | audio | hidden
                                                 │
                                                 ▼
                                      Tone.js polychord ▶︎
```

The audio channel is supervised by an auxiliary loss that ties it to color
hue, so a *red* cell sings one band and a *blue* cell sings another.

## What's deliberately NOT here (and why)

- **No pretrained models.** This is the entire point.
- **No FAISS / pgvector / HNSW library.** A custom GeometricIndex does
  brute-force cosine over <10k bipolar vectors. Above ~10k a kd-tree
  variant slot is reserved (not yet implemented — out of scope).
- **No browser-side training.** Per the architect's risk review, browser
  autograd is immature; we train server-side and ship weight blobs.
- **No live training during demos.** We pre-bake. Live "fine-tuning" is
  shown but does not affect the headline weights.
- **No octree.** Dense 32³ grid is the demo-feasible ceiling. Octree was
  cut per architect's recommendation.
- **No multi-modal pretraining of any kind.** All modalities are learned
  jointly from raw voxel targets.

## Risk mitigations baked into the code

| Risk (analyst-flagged)                       | Mitigation in repo                                    |
| -------------------------------------------- | ----------------------------------------------------- |
| PGD doesn't converge                         | `SGDFallback` class with same interface — drop-in swap |
| HDC encoder uninformative                    | `validate_hdc.py` checks separation before any use   |
| 3D NCA cost explodes                         | Default 32³, frame cache, server-side inference      |
| Custom optimizer "looks" trained but isn't   | `demo_train.py` shows monotone accuracy gains         |
| Conference wifi / Supabase latency           | API tolerates DB absence; runs in-memory by default   |
| Safari / WebGPU absence                      | R3F renders particles via WebGL fallback              |

## Demo script (90 seconds, judges)

1. **Open Cognitron.** Empty galaxy. Drop a thought ("I love baking
   bread"). Particle materializes, glows, settles into the field.
2. Drop 4-5 more thoughts (mixing topics).
3. Click **Train** twice. Particles drift toward semantic neighbours.
4. **Query** "what do I know about cooking?" — wave propagates, golden
   trace lights up the relevant cluster. Read the top-3 hits.
5. **Switch tab to Morpheus.** Click **Grow** on `sphere`. Watch a
   single seed cell explode into a glowing 3D shell, frame-by-frame.
6. Click **Hear it (synesthesia)**. A chord plays — those are the
   frequencies the model jointly learned alongside the geometry.
7. Switch target to `helix`, regrow. Different shape, different chord.

That is the wow. That is the pitch.

## Credits

- **Cognitron PNN** — original architecture, this repo
- **HDC primitives** — Kanerva-style VSA, classical
- **Morpheus 3D NCA** — extends Mordvintsev/Niklasson/Randazzo (2020) and
  Sudhakaran et al. (2021), with the audio modality as new contribution
- Stack: Next.js, shadcn/ui, React Three Fiber, Tone.js, FastAPI, Supabase
