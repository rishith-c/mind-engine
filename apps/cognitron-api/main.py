"""Cognitron HTTP + WebSocket API.

Endpoints
---------
POST   /thought           add a particle from text
GET    /query?q=...       wave-propagation retrieval
GET    /state             full snapshot for the WebGPU renderer
POST   /train/step        run a single PGD step (used by the live demo)
WS     /stream            push particle position updates to the browser
DELETE /reset             clear the field

The API is intentionally thin. All AI logic lives in `cognitron`. The DB
(Supabase) is only used for persistence — embeddings are never stored as
pgvector since the from-scratch constraint forbids it; raw bipolar bytes
are stored in a `bytea` column and similarity is computed in Python.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from typing import Any

# Make `cognitron` importable when running uvicorn from the apps directory
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "packages" / "ai-cognitron"))

import numpy as np  # noqa: E402
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from cognitron import (  # noqa: E402
    HDCEncoder,
    ParticleNetwork,
    ParticleGradientDescent,
    PGDConfig,
)


# ---------------------------------------------------------------------------
# Schemas (Pydantic)
# ---------------------------------------------------------------------------


class ThoughtIn(BaseModel):
    text: str = Field(..., min_length=1, max_length=4000)
    mass: float = Field(default=1.0, ge=0.01, le=10.0)
    polarity: int = Field(default=1)


class ParticleOut(BaseModel):
    id: int
    text: str
    position: list[float]
    mass: float
    polarity: int


class QueryHit(BaseModel):
    id: int
    text: str
    score: float


class QueryOut(BaseModel):
    query: str
    hits: list[QueryHit]


class StateOut(BaseModel):
    n: int
    positions: list[list[float]]
    masses: list[float]
    polarities: list[int]
    ids: list[int]
    texts: list[str]


# ---------------------------------------------------------------------------
# App + global state
# ---------------------------------------------------------------------------


app = FastAPI(
    title="Cognitron API",
    description="Particle Neural Network as a service",
    version="0.1.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)

EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "2048"))
encoder = HDCEncoder(dim=EMBEDDING_DIM)
network = ParticleNetwork(embedding_dim=EMBEDDING_DIM, radius=0.45, propagation_steps=5)


def _to_particle_out(p: Any) -> ParticleOut:
    return ParticleOut(
        id=p.id, text=p.text, position=p.position.tolist(), mass=p.mass, polarity=int(p.polarity)
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/health")
async def health() -> dict[str, Any]:
    return {"ok": True, "particles": len(network)}


@app.post("/thought", response_model=ParticleOut)
async def add_thought(body: ThoughtIn) -> ParticleOut:
    embedding = encoder.encode(body.text)
    p = network.add_particle(embedding, text=body.text, mass=body.mass, polarity=body.polarity)
    return _to_particle_out(p)


@app.get("/query", response_model=QueryOut)
async def query(q: str, k: int = 5) -> QueryOut:
    if not q.strip():
        raise HTTPException(status_code=400, detail="empty query")
    embedding = encoder.encode(q)
    hits_raw = network.query(embedding, k=k)
    hits: list[QueryHit] = []
    for pid, score in hits_raw:
        p = network._particles.get(pid)
        if p is None:
            continue
        hits.append(QueryHit(id=pid, text=p.text, score=float(score)))
    network.reinforce([h.id for h in hits], amount=0.05)
    return QueryOut(query=q, hits=hits)


@app.get("/state", response_model=StateOut)
async def state() -> StateOut:
    snap = network.snapshot()
    return StateOut(
        n=int(len(network)),
        positions=snap["positions"].tolist(),
        masses=snap["masses"].tolist(),
        polarities=[int(x) for x in snap["polarities"].tolist()],
        ids=snap["ids"].tolist(),
        texts=snap["texts"],
    )


@app.post("/train/step")
async def train_step(epochs: int = 1) -> dict[str, Any]:
    if len(network) < 2:
        raise HTTPException(status_code=400, detail="need at least 2 particles to train")

    # Self-supervised loss: place each particle near its k-NN in embedding
    # space. Concretely, minimise the average squared distance between each
    # particle and its 3 most-similar neighbours.
    def loss(_net: ParticleNetwork) -> float:
        particles = _net.particles
        positions = np.stack([p.position for p in particles], axis=0)
        embeddings = np.stack([p.embedding for p in particles], axis=0).astype(np.float32)
        # Cosine similarity over embeddings
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
        sim = (embeddings @ embeddings.T) / (norms @ norms.T)
        np.fill_diagonal(sim, -1.0)
        # Top-3 neighbours per row
        top3 = np.argsort(-sim, axis=1)[:, :3]
        loss_val = 0.0
        for i, neighbour_idx in enumerate(top3):
            for j in neighbour_idx:
                loss_val += float(np.sum((positions[i] - positions[j]) ** 2))
        return loss_val / max(len(particles), 1)

    opt = ParticleGradientDescent(network, loss, PGDConfig(diffusion=0.005))
    losses: list[float] = []
    for _ in range(max(1, epochs)):
        losses.append(opt.step())
    return {"losses": losses}


@app.delete("/reset")
async def reset() -> dict[str, Any]:
    global network
    network = ParticleNetwork(embedding_dim=EMBEDDING_DIM, radius=0.45, propagation_steps=5)
    return {"ok": True}


# ---------------------------------------------------------------------------
# WebSocket — push position updates ~10 times/second while the field changes
# ---------------------------------------------------------------------------


class StreamHub:
    def __init__(self) -> None:
        self._clients: set[WebSocket] = set()
        self._lock = asyncio.Lock()

    async def join(self, ws: WebSocket) -> None:
        await ws.accept()
        async with self._lock:
            self._clients.add(ws)

    async def leave(self, ws: WebSocket) -> None:
        async with self._lock:
            self._clients.discard(ws)

    async def broadcast(self, payload: dict[str, Any]) -> None:
        async with self._lock:
            stale: list[WebSocket] = []
            for ws in self._clients:
                try:
                    await ws.send_json(payload)
                except Exception:
                    stale.append(ws)
            for ws in stale:
                self._clients.discard(ws)


hub = StreamHub()


@app.websocket("/stream")
async def stream(ws: WebSocket) -> None:
    await hub.join(ws)
    try:
        while True:
            snap = network.snapshot()
            await ws.send_json(
                {
                    "type": "snapshot",
                    "n": int(len(network)),
                    "positions": snap["positions"].tolist(),
                    "masses": snap["masses"].tolist(),
                    "ids": snap["ids"].tolist(),
                }
            )
            await asyncio.sleep(0.1)
    except WebSocketDisconnect:
        pass
    finally:
        await hub.leave(ws)
