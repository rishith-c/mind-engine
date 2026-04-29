# Branch: `backend-cognitron`

**Owner:** Backend developer
**Path of record:** `apps/cognitron-api/`

## Charter

Owns the Cognitron HTTP + WebSocket service. Wraps the from-scratch
`cognitron` Python package as a stateless API.

## Boundaries

- **In:** request validation, persistence, WebSocket fan-out, error mapping,
  rate limiting, auth (when added), Supabase client wiring.
- **Out:** the AI itself (lives in `packages/ai-cognitron/` on `ai-cognitron`).

## API contract

| Method | Path           | Purpose                              |
| ------ | -------------- | ------------------------------------ |
| GET    | `/health`      | liveness                             |
| POST   | `/thought`     | add a particle from text             |
| GET    | `/query?q=...` | wave-propagation retrieval           |
| GET    | `/state`       | snapshot for WebGPU renderer         |
| POST   | `/train/step`  | run N PGD steps; returns loss curve  |
| DELETE | `/reset`       | clear field                          |
| WS     | `/stream`      | push field snapshots @ 10 Hz         |

## Local dev

```bash
cd apps/cognitron-api
uvicorn main:app --reload --port 8000
```

## Pydantic schemas

All in `apps/cognitron-api/main.py`. When changing them, regenerate the TS
types in `apps/cognitron-web/src/lib/api.ts` to keep the contract in sync —
see analyst's risk #4 (branch merge hell).
