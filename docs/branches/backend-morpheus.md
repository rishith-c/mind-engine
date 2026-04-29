# Branch: `backend-morpheus`

**Owner:** Backend developer
**Path of record:** `apps/morpheus-api/`

## Charter

Owns the Morpheus inference service. Loads pre-trained 3D NCA weights and
streams growth frames to the browser.

## Boundaries

- **In:** weight loading, frame caching, growth scheduling, WS streaming,
  binary frame export for fast browser decode.
- **Out:** training (lives on `ai-morpheus`), data ETL (lives on
  `data-morpheus`).

## API contract

| Method | Path                | Purpose                                  |
| ------ | ------------------- | ---------------------------------------- |
| GET    | `/health`           | liveness                                 |
| POST   | `/seed`             | start a new growth from scratch          |
| GET    | `/frame/{t}`        | RGBA voxel snapshot at step t            |
| GET    | `/audio/{t}`        | per-cell frequency snapshot at step t    |
| GET    | `/binary/frame/{t}` | float16 raw bytes (fast decode)          |
| WS     | `/stream`           | live frames @ 12 Hz during growth        |

## Memory budget (per architect)

- 32³ × 16 channels × float32 ≈ 2 MB per frame
- Cache 200 frames → ~400 MB process memory ceiling
- After 200 frames the oldest is dropped; the renderer can rewind that far.

## Local dev

```bash
cd apps/morpheus-api
uvicorn main:app --reload --port 8001
```

Requires that you've trained at least one target:

```bash
cd packages/ai-morpheus && python -m morpheus.train --target sphere --steps 4000
```
