"""Morpheus HTTP + WebSocket API.

Endpoints
---------
POST   /seed         start a new growth from a seed cell with a target
GET    /frame/{t}    get voxel snapshot at growth step t (cached if cold)
WS     /stream       push frames live during growth
GET    /audio/{t}    per-cell frequency snapshot at step t

Training is offline (`python -m morpheus.train ...`). This service only runs
inference, loads pre-trained weights, and streams frames to the browser.
"""

from __future__ import annotations

import asyncio
import io
import os
import sys
from pathlib import Path
from typing import Any

# Make `morpheus` importable when running uvicorn from the apps directory
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "packages" / "ai-morpheus"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from morpheus import NCA3D, NCAConfig, seed_grid, extract_voxels, extract_audio_field  # noqa: E402


WEIGHT_DIR = Path(os.getenv("MORPHEUS_WEIGHTS", str(ROOT / "weights")))


# ---------------------------------------------------------------------------
# Inference engine — keeps a single rolling state in memory
# ---------------------------------------------------------------------------


class GrowthSession:
    """Encapsulates one ongoing growth: the model, the rolling state, and a
    cache of past frames so the renderer can scrub backwards."""

    def __init__(self, model: NCA3D, cfg: NCAConfig, max_cache: int = 200):
        self.model = model
        self.cfg = cfg
        self.max_cache = max_cache
        self.frame_cache: dict[int, torch.Tensor] = {}
        self.audio_cache: dict[int, torch.Tensor] = {}
        self.state = seed_grid(cfg, batch=1)
        self.t = 0
        self._record(0)

    def step(self) -> None:
        with torch.no_grad():
            self.state = self.model(self.state)
        self.t += 1
        self._record(self.t)

    def _record(self, t: int) -> None:
        # Detach + CPU + half precision to keep cache footprint small
        self.frame_cache[t] = extract_voxels(self.state).cpu().half()
        self.audio_cache[t] = extract_audio_field(self.state, self.cfg).cpu().half()
        # Trim oldest if cache too big
        while len(self.frame_cache) > self.max_cache:
            oldest = min(self.frame_cache.keys())
            del self.frame_cache[oldest]
            del self.audio_cache[oldest]


SESSION: GrowthSession | None = None


def load_session(target: str = "sphere") -> GrowthSession:
    weights_path = WEIGHT_DIR / f"morpheus_{target}.pt"
    cfg = NCAConfig()
    model = NCA3D(cfg)
    if weights_path.exists():
        ckpt = torch.load(weights_path, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        print(f"loaded weights: {weights_path}")
    else:
        print(
            f"WARNING: weights {weights_path} not found — running with random "
            f"init. Train first: python -m morpheus.train --target {target}"
        )
    model.eval()
    return GrowthSession(model, cfg)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class SeedIn(BaseModel):
    target: str = Field(default="sphere")
    steps: int = Field(default=64, ge=1, le=200)


class FrameOut(BaseModel):
    t: int
    grid_size: int
    rgba: list  # nested D x H x W x 4


class AudioOut(BaseModel):
    t: int
    grid_size: int
    frequencies: list  # nested D x H x W


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------


app = FastAPI(title="Morpheus API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health() -> dict[str, Any]:
    return {"ok": True, "session_alive": SESSION is not None, "t": SESSION.t if SESSION else 0}


@app.post("/seed", response_model=FrameOut)
async def seed(body: SeedIn) -> FrameOut:
    global SESSION
    SESSION = load_session(body.target)
    for _ in range(body.steps):
        SESSION.step()
    return await frame(SESSION.t)


@app.get("/frame/{t}", response_model=FrameOut)
async def frame(t: int) -> FrameOut:
    if SESSION is None:
        raise HTTPException(status_code=404, detail="no active session — call /seed first")
    if t not in SESSION.frame_cache:
        raise HTTPException(status_code=404, detail=f"frame {t} not in cache")
    rgba = SESSION.frame_cache[t][0].float().tolist()  # (D, H, W, 4)
    return FrameOut(t=t, grid_size=SESSION.cfg.grid_size, rgba=rgba)


@app.get("/audio/{t}", response_model=AudioOut)
async def audio(t: int) -> AudioOut:
    if SESSION is None:
        raise HTTPException(status_code=404, detail="no active session — call /seed first")
    if t not in SESSION.audio_cache:
        raise HTTPException(status_code=404, detail=f"audio frame {t} not in cache")
    freqs = SESSION.audio_cache[t][0].float().tolist()
    return AudioOut(t=t, grid_size=SESSION.cfg.grid_size, frequencies=freqs)


@app.websocket("/stream")
async def stream(ws: WebSocket) -> None:
    await ws.accept()
    try:
        while True:
            if SESSION is None:
                await asyncio.sleep(0.2)
                continue
            SESSION.step()
            rgba = SESSION.frame_cache[SESSION.t][0].float().tolist()
            await ws.send_json({"type": "frame", "t": SESSION.t, "rgba": rgba})
            await asyncio.sleep(1 / 12)  # ~12 fps for stream
    except WebSocketDisconnect:
        return


@app.get("/binary/frame/{t}")
async def binary_frame(t: int) -> bytes:
    """Same as /frame but raw float16 bytes for fast browser decode."""
    if SESSION is None or t not in SESSION.frame_cache:
        raise HTTPException(status_code=404, detail="frame missing")
    arr = SESSION.frame_cache[t][0].numpy().astype(np.float16)
    buf = io.BytesIO()
    np.save(buf, arr)
    return buf.getvalue()
