# mind-engine

Two original AI systems built entirely from scratch, with zero pretrained models. Cognitron is a Particle Neural Network. Morpheus is a synesthetic 3D Neural Cellular Automaton. Both render live in the browser with WebGPU.

## What this is

This is a monorepo containing two AI architectures that were designed and implemented without using any pretrained models, embedding layers, or model zoos. Every weight, every neuron, every update rule is built from raw primitives.

**Cognitron** treats neurons as free particles in 3D space instead of fixed nodes in a graph. Each particle has a position, velocity, mass, charge, and a 10,000-dimensional hypervector identity derived purely from hashing (no learned embeddings). The network topology is not stored. It is reconstructed every forward pass from spatial proximity. Training is a physics simulation where particles drift toward their semantic neighbors. Inference is a wave of energy propagating through the field, with conductance determined by embedding similarity, distance falloff, and polarity. You drop thoughts into the particle field, train the layout, and then fire queries as waves that light up relevant clusters.

**Morpheus** extends the Growing Neural Cellular Automaton (Mordvintsev et al., 2020) from 2D to 3D and adds a modality that has not been shipped before: per-cell audio frequency. Each cell in a 32x32x32 grid holds 16 channels. The first five are observable (RGB, alpha for geometry, and audio frequency). The rest are hidden state. At each timestep, cells perceive their 26-neighborhood through fixed 3D Sobel kernels, run a small MLP, apply a stochastic update mask, and add the residual back. The result is a network that grows geometry, color, and sound jointly from a single seed cell. When you play the audio, a red cell sings one frequency band and a blue cell sings another, creating a genuine synesthetic experience.

## Why it is built this way

The question driving this project was: what happens when you reject the standard shelf of pretrained transformers entirely?

Cognitron's Particle Neural Network is a genuine new architecture. A novelty-research agent confirmed that the specific combination of spatial neurons, PSO-gradient hybrid training, HDC encoder, and wave inference is unpublished. The hyperdimensional encoder uses deterministic hashing to map tokens to near-orthogonal 10,000-d bipolar vectors. This means it can detect lexical overlap ("what do I know about cooking?" retrieves the "baking bread" thought because they share tokens) but not semantic similarity ("tell me about pets" will not retrieve "cats" because those are independent random hypervectors). That is an honest trade-off of the from-scratch constraint, and the demo is designed around it: the live training loop physically pulls particles together, so the visualization tells the cluster story even when a specific query misses.

Morpheus's audio-per-cell NCA is the other fresh contribution. Existing NCA work handles geometry and color. Adding a learned audio channel tied to color hue through an auxiliary loss means the network generates a polychord that changes as the organism grows and mutates, which is both novel and immediately visceral in a demo.

The monorepo is organized as a Turborepo workspace with pnpm. AI packages are pure Python (NumPy for Cognitron, PyTorch for Morpheus). Web frontends are Next.js with React Three Fiber for 3D rendering and WebGPU compute shaders. APIs are FastAPI. The shared-physics package contains both TypeScript reference implementations and WGSL compute shaders for the same force and update equations, so the browser can run particle physics and NCA updates on the GPU at interactive frame rates.

## How Cognitron works

```
text --> HDC encoder --> 10,000-d bipolar hypervector
                               |
                               v
                      random projection (fixed seeded R^3 axes)
                               |
                               v
                      spawn a Particle:
                      { position, velocity, mass, charge, polarity }
                               |
                               v
                    ParticleNetwork (spatial hash, O(N) neighbor queries)
                      forward: wave propagation x 4-6 hops
                      query:   encode -> seed -> wave -> top-k results
                               |
                               v
                    Particle Gradient Descent trainer
                      v = w*v + c1*r1*(pbest-x) + c2*r2*(gbest-x)
                          - lr*grad(L) + noise
```

There is no transformer, no autoencoder, no embedding model. The encoder is deterministic hashing into orthogonal hypervectors. The neurons are numpy arrays. The training is a six-equation force law with an SGD fallback if PGD does not converge.

## How Morpheus works

```
seed cell --> 32^3 grid, 16-channel state per cell
                          |
              for each timestep:
                          |
                  Sobel perception 3D (fixed kernels)
                          |
                  per-cell MLP (16 -> 96 -> 16, residual)
                          |
                  stochastic mask (50% of cells update)
                          |
                  alive mask (alpha pooling)
                          |
              state[t+1]: RGB | alpha | audio | hidden
                                         |
                                   Tone.js polychord
```

The audio channel is supervised by an auxiliary loss that ties frequency to color hue, so the sound the organism produces is a direct expression of its visual structure.

## Quickstart

### Prerequisites

- Node 20 or newer, pnpm 9 or newer
- Python 3.11 or newer
- Chrome, Edge, or Chrome Canary (for WebGPU support)
- Optional: a Supabase project for persistence (or `supabase start` for local)

### 1. Install the monorepo

```bash
git clone https://github.com/yourusername/mind-engine.git
cd mind-engine
pnpm install
```

### 2. Install Python dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e "packages/ai-cognitron[api]"
pip install -e "packages/ai-morpheus[api]"
```

### 3. Validate the HDC encoder

```bash
cd packages/ai-cognitron && python -m cognitron.validate_hdc
# All 4 checks should PASS
```

### 4. Run a Cognitron training smoke test

```bash
python -m cognitron.demo_train
# Top-1 accuracy should rise monotonically over 30 PGD steps
```

### 5. Pre-train Morpheus weights

This takes about 5 to 15 minutes on CPU, faster on GPU.

```bash
cd packages/ai-morpheus && python -m morpheus.train --target sphere --steps 4000
# Saves to weights/morpheus_sphere.pt
```

### 6. Launch everything

```bash
# Terminal 1: Cognitron API on port 8000
cd apps/cognitron-api && uvicorn main:app --reload --port 8000

# Terminal 2: Morpheus API on port 8001
cd apps/morpheus-api && uvicorn main:app --reload --port 8001

# Terminal 3: Both web frontends
pnpm dev
# cognitron-web on localhost:3000, morpheus-web on localhost:3001
```

## Project structure

```
mind-engine/
  apps/
    cognitron-web/         Next.js + shadcn/ui + React Three Fiber + WebGPU
    cognitron-api/         FastAPI service (PNN + HDC + GeometricIndex)
    morpheus-web/          Next.js + shadcn/ui + React Three Fiber + Tone.js
    morpheus-api/          FastAPI service (3D NCA inference)
  packages/
    ai-cognitron/
      cognitron/
        hdc.py             10,000-d hyperdimensional encoder
        pnn.py             Particle Neural Network
        pgd.py             Particle Gradient Descent + SGD fallback
        geometric_index.py Custom brute-force index (no FAISS)
        validate_hdc.py    Encoder unit tests
        demo_train.py      End-to-end training smoke test
    ai-morpheus/
      morpheus/
        nca3d.py           Sobel perception + per-cell MLP update
        train.py           Server-side training to weight blob
    shared-physics/
      src/
        particle-forces.ts + .wgsl    Force equations in TS and WGSL
        nca-update.ts + .wgsl         NCA step in TS and WGSL
  supabase/
    migrations/            Storage schema (no pgvector)
  scripts/
    data_pipeline.py       Embedding, clustering, and decay jobs
  turbo.json
  package.json
  pnpm-workspace.yaml
```

## Screenshots / Demo

<!-- Add screenshot: Cognitron particle field with several thoughts dropped in, showing colored particles floating in 3D space -->

<!-- Add screenshot: Cognitron after training, particles clustered by topic, with a wave query propagating through the field and lighting up a cluster -->

<!-- Add screenshot: Morpheus 3D voxel organism growing frame-by-frame from a single seed cell into a sphere shape -->

<!-- Add screenshot: Morpheus with the synesthesia audio panel open, showing per-cell frequencies mapped to color -->

## Honest limits

HDC has no learned semantics. It encodes text through deterministic hashing, so similarity is lexical overlap, not meaning. This is the price of the from-scratch constraint, and the design leans into it rather than hiding it.

The Cognitron particle network is capped at roughly 5,000 to 20,000 particles for interactive performance. The Morpheus grid is capped at 32x32x32 cells (about 33,000 cells) for the same reason. Neither cap is architectural. They are demo-feasibility constraints identified during architect review.

Browser-side training is deliberately excluded. Browser autograd libraries are not mature enough to be reliable in a live demo. Training happens server-side, and the browser receives weight blobs.

## Credits

- **Cognitron PNN**: original architecture, this repo
- **HDC primitives**: Kanerva-style Vector Symbolic Architecture, classical
- **Morpheus 3D NCA**: extends Mordvintsev, Niklasson, and Randazzo (2020) and Sudhakaran et al. (2021), with the audio modality as new contribution
- Stack: Next.js, shadcn/ui, React Three Fiber, Tone.js, FastAPI, Supabase
