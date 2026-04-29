# Branch: `data-cognitron`

**Owner:** Data engineer
**Path of record:** `scripts/data_pipeline.py`, `supabase/migrations/`

## Charter

Owns the data lifecycle for Cognitron particles:
1. **Embed pending thoughts** — fill in HDC vectors for rows that arrived
   without one (e.g. dropped from API in degraded mode).
2. **Cluster report** — periodic k-means on the field, written to JSON
   for the frontend overlay.
3. **Decay step** — apply `cognitron_decay_step()` SQL function and log.

## Schedule

Suggested cron / Supabase scheduled functions:

| Job              | Cadence       |
| ---------------- | ------------- |
| `embed`          | every 5 min   |
| `cluster`        | every 15 min  |
| `decay`          | every hour    |

## Run locally

```bash
python scripts/data_pipeline.py all
```

## Schema (Supabase)

See `supabase/migrations/0001_init.sql`. Key tables:
- `cognitron_particles` — id, text, embedding (bytea, no pgvector),
  position_xyz, mass, charge, polarity, timestamps
- `cognitron_query_log` — query text, hit ids, hit scores

The decay function is in SQL for latency; the Python script mirrors it
for offline runs.
