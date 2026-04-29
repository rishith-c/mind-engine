# Branch: `ai-morpheus`

**Owner:** AI/ML engineer
**Path of record:** `packages/ai-morpheus/`

## Charter

The 3D synesthetic NCA. Owns:
- `nca3d.py` — model definition (Sobel perception, per-cell MLP, alive mask)
- `train.py` — server-side training loop (Mordvintsev-style sample pool)

## Novelty contribution

3D voxel NCA + color is published. **The audio modality (per-cell
frequency, jointly trained with hue) is what makes Morpheus new.**
This is the only fresh angle in the literature per the novelty review.

## Architecture

```python
NCAConfig(
    grid_size = 32,        # per architect's recommended ceiling
    n_channels = 16,       # 5 visible (R,G,B,α,audio) + 11 hidden
    hidden_dim = 96,
    fire_rate = 0.5,       # stochastic update mask
    clip_alpha = 0.1,
    audio_min_hz = 80,
    audio_max_hz = 2000,
)
```

Visible channels:
- `[0:3]` — RGB
- `[3]`   — alpha (geometry)
- `[4]`   — audio (frequency mapped via sigmoid into [audio_min, audio_max])

## Training

```bash
python -m morpheus.train --target sphere --steps 4000
python -m morpheus.train --target helix  --steps 4000
```

Saves to `weights/morpheus_<target>.pt`.

## Open work

- [ ] More targets (cube, spiral, organic shapes)
- [ ] User-prompt → seed conditioning (text encoder for prompt)
- [ ] Damage / regrowth experiments
- [ ] Auxiliary loss tuning — audio-color correlation weight

## Failure modes (analyst-flagged)

- Gradient explosion across long unrolls → `clip_grad_norm_(1.0)`
- 3D simulation cost → grid capped at 32³, no octree
- Audio channel drift → auxiliary correlation loss anchors it to hue
