# Supabase

Storage layer for mind-engine. **No pgvector.** Per the from-scratch
constraint, all similarity search runs in the application layer.

## Setup

```bash
# Run a local supabase instance (or connect to your project)
supabase start
supabase db reset    # applies migrations/0001_init.sql

# Or push to a hosted Supabase project
supabase db push
```

## Layout

- `migrations/0001_init.sql` — initial schema (particles, frames, decay job)
- Two namespaces: `cognitron_*` and `morpheus_*`

## Data lifecycle

- Cognitron particles **decay** at -1.5%/hour and are GC'd at mass < 0.001.
- Morpheus frames are pinned to a `seed_id`; deleting the seed cascades.
- Embeddings are stored as `bytea` (raw bipolar int8) and reloaded into
  the custom `GeometricIndex` at API startup.
