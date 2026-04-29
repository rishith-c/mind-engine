"""Particle Gradient Descent — original training algorithm.

Combines three forces on each particle's *position* (positions are the only
trainable parameter; embeddings are fixed identity, masses follow usage):

  v_{t+1} = w * v_t                               # inertia
          + c1 * r1 * (pbest - x_t)               # personal best (PSO)
          + c2 * r2 * (gbest - x_t)               # global best  (PSO)
          - eta * grad_L(x_t)                     # gradient term
          + sigma * N(0, I)                       # diffusion noise

  x_{t+1} = x_t + v_{t+1}

The gradient is computed numerically (finite differences) on the loss
function provided by the user — the architecture deliberately does NOT
require an autograd framework, since the project's "from scratch" constraint
forbids ML libraries for the model itself.

A SGDFallback class with the same interface is provided per the analyst's
risk-mitigation recommendation (top failure mode #1 — custom optimizer
divergence). If the demo task fails to converge under PGD, swap to SGDFallback
without changing any caller code.

Typical demo task: place query particles near semantically similar memory
particles. The loss is the negative absorbed energy at output particles —
i.e., the network is trained to make true associations *physically close*.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from cognitron.pnn import Particle, ParticleNetwork


LossFn = Callable[[ParticleNetwork], float]


@dataclass
class PGDConfig:
    inertia: float = 0.7
    cognitive: float = 1.4  # c1, pull toward personal best
    social: float = 1.4  # c2, pull toward global best
    learning_rate: float = 0.05
    diffusion: float = 0.005
    grad_eps: float = 1e-2  # finite-difference probe size
    velocity_clip: float = 0.2
    seed: int = 0


class ParticleGradientDescent:
    """Trainer. Single canonical loop in step()."""

    def __init__(self, network: ParticleNetwork, loss_fn: LossFn, config: PGDConfig | None = None):
        self.network = network
        self.loss_fn = loss_fn
        self.cfg = config or PGDConfig()
        self._rng = np.random.default_rng(self.cfg.seed)

        # Per-particle personal best position and loss
        self._pbest_pos: dict[int, np.ndarray] = {}
        self._pbest_loss: dict[int, float] = {}
        self._gbest_pos: np.ndarray | None = None
        self._gbest_loss = float("inf")
        self._loss_history: list[float] = []

    def step(self) -> float:
        """One PGD step over all particles. Returns current global loss."""
        cfg = self.cfg
        loss = float(self.loss_fn(self.network))
        self._loss_history.append(loss)

        # Update global best
        if loss < self._gbest_loss:
            self._gbest_loss = loss
            self._gbest_pos = np.mean(
                np.stack([p.position for p in self.network.particles], axis=0), axis=0
            )

        # Per-particle update
        for particle in self.network.particles:
            grad = self._numeric_grad(particle)

            # Personal best
            prev_best = self._pbest_loss.get(particle.id, float("inf"))
            if loss < prev_best:
                self._pbest_loss[particle.id] = loss
                self._pbest_pos[particle.id] = particle.position.copy()
            pbest = self._pbest_pos.get(particle.id, particle.position)
            gbest = self._gbest_pos if self._gbest_pos is not None else particle.position

            r1 = self._rng.random()
            r2 = self._rng.random()
            new_v = (
                cfg.inertia * particle.velocity
                + cfg.cognitive * r1 * (pbest - particle.position)
                + cfg.social * r2 * (gbest - particle.position)
                - cfg.learning_rate * grad
                + cfg.diffusion * self._rng.standard_normal(3).astype(np.float32)
            )
            # Clip velocity for stability — common PSO practice
            mag = float(np.linalg.norm(new_v))
            if mag > cfg.velocity_clip:
                new_v = new_v * (cfg.velocity_clip / mag)

            new_pos = particle.position + new_v
            new_pos = np.clip(new_pos, -1.5, 1.5)  # keep field bounded for viz

            updated = particle.with_velocity(new_v).with_position(new_pos)
            self.network.replace_particle(updated)

        return loss

    @property
    def loss_history(self) -> list[float]:
        return list(self._loss_history)

    # -- internals ----------------------------------------------------------

    def _numeric_grad(self, particle: Particle) -> np.ndarray:
        """Three-axis central-difference gradient of the loss w.r.t. this
        particle's position. O(6) loss evaluations per particle per step;
        viable up to ~1k particles, which is fine for the demo task. For
        larger fields, swap in stochastic coordinate descent or use the
        SGDFallback below.
        """
        eps = self.cfg.grad_eps
        original = particle.position.copy()
        grad = np.zeros(3, dtype=np.float32)
        for axis in range(3):
            for sign, slot in ((+1, 0), (-1, 1)):
                offset = np.zeros(3, dtype=np.float32)
                offset[axis] = sign * eps
                self.network.replace_particle(particle.with_position(original + offset))
                if slot == 0:
                    f_plus = float(self.loss_fn(self.network))
                else:
                    f_minus = float(self.loss_fn(self.network))
            grad[axis] = (f_plus - f_minus) / (2 * eps)
        # Restore
        self.network.replace_particle(particle.with_position(original))
        return grad


# ---------------------------------------------------------------------------
# Risk-mitigation fallback: plain SGD with momentum on positions
# ---------------------------------------------------------------------------


@dataclass
class SGDConfig:
    learning_rate: float = 0.05
    momentum: float = 0.9
    grad_eps: float = 1e-2
    velocity_clip: float = 0.2


class SGDFallback:
    """Same step() interface as ParticleGradientDescent but uses plain
    momentum SGD on positions. If PGD fails to converge (analyst-flagged
    risk #1), the caller can swap optimizers without other code changes.
    """

    def __init__(self, network: ParticleNetwork, loss_fn: LossFn, config: SGDConfig | None = None):
        self.network = network
        self.loss_fn = loss_fn
        self.cfg = config or SGDConfig()
        self._loss_history: list[float] = []

    def step(self) -> float:
        cfg = self.cfg
        loss = float(self.loss_fn(self.network))
        self._loss_history.append(loss)
        for particle in self.network.particles:
            grad = self._numeric_grad(particle)
            new_v = cfg.momentum * particle.velocity - cfg.learning_rate * grad
            mag = float(np.linalg.norm(new_v))
            if mag > cfg.velocity_clip:
                new_v = new_v * (cfg.velocity_clip / mag)
            new_pos = np.clip(particle.position + new_v, -1.5, 1.5)
            self.network.replace_particle(
                particle.with_velocity(new_v).with_position(new_pos)
            )
        return loss

    @property
    def loss_history(self) -> list[float]:
        return list(self._loss_history)

    def _numeric_grad(self, particle: Particle) -> np.ndarray:
        eps = self.cfg.grad_eps
        original = particle.position.copy()
        grad = np.zeros(3, dtype=np.float32)
        for axis in range(3):
            for sign in (+1, -1):
                offset = np.zeros(3, dtype=np.float32)
                offset[axis] = sign * eps
                self.network.replace_particle(particle.with_position(original + offset))
                if sign == +1:
                    f_plus = float(self.loss_fn(self.network))
                else:
                    f_minus = float(self.loss_fn(self.network))
            grad[axis] = (f_plus - f_minus) / (2 * eps)
        self.network.replace_particle(particle.with_position(original))
        return grad
