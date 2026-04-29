# Branch: `data-morpheus`

**Owner:** Data engineer
**Path of record:** Voxel ETL helpers (planned)

## Charter

Owns the data lifecycle for Morpheus growths:
1. **Frame archival** — flush in-memory frame cache to Supabase
   `morpheus_frames` after each session
2. **Voxel compaction** — convert dense float32 frames to float16 + RLE for
   long-term storage
3. **Replay export** — generate WebM / GIF from a stored growth for sharing

## Schema

See `supabase/migrations/0001_init.sql`:
- `morpheus_seeds(id, target, grid_size, n_channels, weights_url, …)`
- `morpheus_frames(id, seed_id, t, voxels bytea, audio_field bytea, …)`

Storage estimate: 32³ × 16 × float16 ≈ 1 MB / frame; 200-frame growth ≈ 200 MB.
Run RLE compaction after archive to drop empty space.

## Open work

- [ ] Implement `voxels_to_rle.py`
- [ ] WebM exporter using ffmpeg + extracted RGBA
- [ ] Long-tail frame caching — cold storage to Supabase Storage bucket
- [ ] Audio rendering: per-frame chord → wav

## Local dev

(stub) — scripts go under `scripts/morpheus/` when implemented.
