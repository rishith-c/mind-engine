"""Train Morpheus to grow a target voxel pattern from a single seed cell.

Following Mordvintsev's 2D recipe in 3D:
  - Random number of update steps per training sample (in [64, 96])
  - L2 loss against a target RGBA voxel pattern
  - Sample pool: persist intermediate states and resample for stability
  - Audio channel is supervised by an auxiliary loss that encourages it to
    correlate with the cell's color hue (so red cells emit one frequency
    band, blue cells another) — this is what makes the result *synesthetic*

We train server-side and ship the weights as a binary blob, per the
architect's recommendation that browser-side training is not yet viable.

Run: python -m morpheus.train --steps 4000 --target sphere
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from morpheus.nca3d import NCA3D, NCAConfig, seed_grid


# ---------------------------------------------------------------------------
# Built-in voxel targets (no external data dependencies)
# ---------------------------------------------------------------------------


def make_sphere_target(cfg: NCAConfig) -> torch.Tensor:
    """RGBA voxel sphere: orange surface, hollow inside."""
    g = cfg.grid_size
    coords = torch.stack(torch.meshgrid(
        torch.linspace(-1, 1, g),
        torch.linspace(-1, 1, g),
        torch.linspace(-1, 1, g),
        indexing="ij",
    ), dim=-1)
    r = coords.norm(dim=-1)
    surface = ((r > 0.55) & (r < 0.7)).float()
    rgba = torch.zeros(4, g, g, g)
    rgba[0] = 1.0 * surface  # red
    rgba[1] = 0.5 * surface  # green
    rgba[2] = 0.1 * surface  # blue
    rgba[3] = surface
    return rgba


def make_helix_target(cfg: NCAConfig) -> torch.Tensor:
    """Double-helix-style target — visually striking + tests directional growth."""
    g = cfg.grid_size
    rgba = torch.zeros(4, g, g, g)
    for t_idx in range(g):
        t = (t_idx / g) * 4 * torch.pi
        cx = int((g / 2) + (g / 4) * torch.cos(t))
        cy = int((g / 2) + (g / 4) * torch.sin(t))
        cz = t_idx
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                x = cx + dx
                y = cy + dy
                if 0 <= x < g and 0 <= y < g:
                    rgba[0, x, y, cz] = 0.3
                    rgba[1, x, y, cz] = 0.8
                    rgba[2, x, y, cz] = 1.0
                    rgba[3, x, y, cz] = 1.0
    return rgba


TARGETS = {
    "sphere": make_sphere_target,
    "helix": make_helix_target,
}


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------


def synesthetic_loss(state: torch.Tensor, target_rgba: torch.Tensor) -> torch.Tensor:
    """Visual L2 + audio-color correlation auxiliary."""
    # Visual loss: L2 over RGBA channels
    pred_rgba = state[:, :4]
    visual = F.mse_loss(pred_rgba, target_rgba.unsqueeze(0).expand_as(pred_rgba))
    # Audio aux: encourage audio channel to track hue (here, R - B as a
    # simple hue proxy). Without this, the audio channel drifts to noise.
    audio_target = (state[:, 0:1].detach() - state[:, 2:3].detach())
    audio_pred = state[:, 4:5]
    audio_aux = F.mse_loss(audio_pred, audio_target)
    return visual + 0.1 * audio_aux


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train(target_name: str = "sphere", n_steps: int = 4000, out_path: Path | None = None) -> Path:
    cfg = NCAConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"== Morpheus 3D NCA training :: target={target_name}  device={device} ==")

    target = TARGETS[target_name](cfg).to(device)
    model = NCA3D(cfg).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=2e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=1500, gamma=0.3)

    # Sample pool — persists intermediate states so the network learns
    # to *maintain* the pattern, not just regrow from seed every time.
    pool_size = 32
    pool = seed_grid(cfg, batch=pool_size, device=device)

    for step in range(n_steps):
        # Sample 8 grids from the pool (with seed reset on the worst one)
        idx = torch.randint(0, pool_size, (8,))
        batch = pool[idx].clone()
        # Reset the worst-loss element to seed (Mordvintsev trick)
        with torch.no_grad():
            losses_per = ((batch[:, :4] - target.unsqueeze(0)) ** 2).mean(dim=(1, 2, 3, 4))
            worst = int(losses_per.argmax())
        seed = seed_grid(cfg, batch=1, device=device)
        batch[worst : worst + 1] = seed

        # Random number of growth steps
        n_steps_grow = int(torch.randint(64, 97, (1,)).item())
        x = batch
        for _ in range(n_steps_grow):
            x = model(x)

        loss = synesthetic_loss(x, target)
        optim.zero_grad()
        loss.backward()
        # Clip gradients — NCAs are notorious for exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optim.step()
        scheduler.step()

        # Write back into pool
        pool[idx] = x.detach()

        if step % 100 == 0 or step == n_steps - 1:
            print(f"   step {step:5d}  loss={loss.item():.6f}")

    out_path = out_path or Path("./weights") / f"morpheus_{target_name}.pt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "cfg": cfg.__dict__, "target": target_name}, out_path)
    print(f"saved -> {out_path}")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=list(TARGETS.keys()), default="sphere")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    train(args.target, args.steps, args.out)


if __name__ == "__main__":
    main()
