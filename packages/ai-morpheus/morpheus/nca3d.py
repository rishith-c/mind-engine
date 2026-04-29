"""3D Synesthetic Neural Cellular Automaton.

Each cell is a vector of `n_channels` values. The first 5 channels are
"observable":
    [0:3] = RGB
    [3]   = alpha (geometry occupancy)
    [4]   = audio (per-cell frequency mapping for synesthetic playback)

The remaining channels are hidden state. Updates are local: each cell sees a
3D Sobel-style perception of itself + 26-neighbourhood, runs a small per-cell
MLP, and adds the residual back. Updates are stochastic-masked (1/2 of cells
update per step, randomly selected) for robustness, exactly as in the 2D
Growing-NCA paper.

This is built from primitives. The only library dependency is PyTorch — used
for tensor ops + autograd, not for any pretrained weights or models.

Per architect feedback we keep the grid at 32^3 by default (about 33k cells),
which is the demo-feasible ceiling for live WebGPU playback.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Sobel-style 3D perception — fixed (non-learned) gradient + identity filters
# ---------------------------------------------------------------------------


def _sobel_kernels_3d() -> torch.Tensor:
    """Build 4 fixed 3x3x3 perception kernels:
        identity, gradient-x, gradient-y, gradient-z

    These are stacked into a single conv weight of shape (4, 1, 3, 3, 3)
    that we apply per-channel via grouped convolution. This is the textbook
    way to give each cell awareness of its neighbourhood without learning
    anything in the perception step.
    """
    g = torch.tensor([1.0, 2.0, 1.0])
    s = torch.tensor([1.0, 0.0, -1.0])

    # 1D outer products
    sobel_x = torch.einsum("i,j,k->ijk", s, g, g) / 16.0
    sobel_y = torch.einsum("i,j,k->ijk", g, s, g) / 16.0
    sobel_z = torch.einsum("i,j,k->ijk", g, g, s) / 16.0
    identity = torch.zeros(3, 3, 3)
    identity[1, 1, 1] = 1.0

    return torch.stack([identity, sobel_x, sobel_y, sobel_z], dim=0).unsqueeze(1)


# ---------------------------------------------------------------------------
# Config + Network
# ---------------------------------------------------------------------------


@dataclass
class NCAConfig:
    grid_size: int = 32
    n_channels: int = 16  # 5 visible + 11 hidden
    hidden_dim: int = 96
    fire_rate: float = 0.5  # stochastic update mask probability
    clip_alpha: float = 0.1  # cell is "alive" if alpha > this
    audio_min_hz: float = 80.0  # per-cell audio mapped into this band
    audio_max_hz: float = 2000.0


class NCA3D(nn.Module):
    """Single-step update operator. Iterating it yields growth.

    Forward signature:  state -> next_state, with the same shape (N, C, D, H, W).
    """

    def __init__(self, cfg: NCAConfig | None = None):
        super().__init__()
        self.cfg = cfg or NCAConfig()
        self.register_buffer("perception_kernel", _sobel_kernels_3d(), persistent=False)
        c = self.cfg.n_channels
        # Per-cell update MLP (1x1x1 convolutions over the perception output)
        self.update = nn.Sequential(
            nn.Conv3d(4 * c, self.cfg.hidden_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(self.cfg.hidden_dim, c, kernel_size=1, bias=False),
        )
        # Init the final layer to zero so the initial residual = 0 (the
        # standard Growing-NCA trick that prevents wild early dynamics).
        nn.init.zeros_(self.update[-1].weight)

    def perceive(self, state: torch.Tensor) -> torch.Tensor:
        """Apply the fixed Sobel kernels per channel via grouped conv."""
        n, c, d, h, w = state.shape
        kernel = self.perception_kernel.to(state.dtype)  # (4, 1, 3, 3, 3)
        # Replicate kernel to (4*c, 1, 3, 3, 3) for grouped conv
        kernel = kernel.repeat_interleave(c, dim=0)  # (4c, 1, 3, 3, 3)
        # Need state shape (N, c*4, ...) where channels are interleaved [c, c, c, c]
        # Use grouped conv with groups=c by splitting kernel correctly.
        # Easier: stack 4 separate convs by reshaping.
        # We pad with replicate to keep boundary stable.
        padded = F.pad(state, (1, 1, 1, 1, 1, 1), mode="replicate")
        outputs = []
        for k in range(4):
            sub_kernel = kernel[k * c : (k + 1) * c]  # (c, 1, 3, 3, 3)
            outputs.append(F.conv3d(padded, sub_kernel, groups=c))
        # outputs is list of (n, c, d, h, w); concatenate along channel
        return torch.cat(outputs, dim=1)  # (n, 4c, d, h, w)

    def life_mask(self, state: torch.Tensor) -> torch.Tensor:
        """A cell is 'alive' if its 3D max-pooled alpha > clip_alpha. Dead
        cells are zeroed after the update so empty space stays empty."""
        alpha = state[:, 3:4]
        pooled = F.max_pool3d(alpha, kernel_size=3, stride=1, padding=1)
        return (pooled > self.cfg.clip_alpha).to(state.dtype)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        pre_alive = self.life_mask(state)
        perception = self.perceive(state)
        delta = self.update(perception)
        # Stochastic update mask
        mask_shape = (state.shape[0], 1, *state.shape[2:])
        update_mask = (torch.rand(mask_shape, device=state.device) < self.cfg.fire_rate).to(state.dtype)
        new_state = state + delta * update_mask
        post_alive = self.life_mask(new_state)
        alive = (pre_alive * post_alive)
        return new_state * alive


# ---------------------------------------------------------------------------
# Helpers — seeding, observable extraction, audio extraction
# ---------------------------------------------------------------------------


def seed_grid(cfg: NCAConfig, batch: int = 1, device: torch.device | str = "cpu") -> torch.Tensor:
    """A single live cell at the centre, all channels = 1."""
    state = torch.zeros(batch, cfg.n_channels, cfg.grid_size, cfg.grid_size, cfg.grid_size, device=device)
    c = cfg.grid_size // 2
    state[:, 3:, c, c, c] = 1.0  # alpha + hidden channels lit
    return state


def extract_voxels(state: torch.Tensor) -> torch.Tensor:
    """Return (N, D, H, W, 4) tensor of (R, G, B, A) suitable for the WebGPU
    voxel renderer. Values are clipped to [0, 1]."""
    rgba = state[:, :4].clamp(0.0, 1.0)  # (N, 4, D, H, W)
    return rgba.permute(0, 2, 3, 4, 1).contiguous()  # (N, D, H, W, 4)


def extract_audio_field(state: torch.Tensor, cfg: NCAConfig) -> torch.Tensor:
    """Return (N, D, H, W) tensor of per-cell frequency in Hz. The audio
    channel is sigmoid-mapped into [audio_min_hz, audio_max_hz] so the
    front-end can play a chord based on the most-active alive cells."""
    raw = torch.sigmoid(state[:, 4])  # (N, D, H, W)
    return cfg.audio_min_hz + raw * (cfg.audio_max_hz - cfg.audio_min_hz)
