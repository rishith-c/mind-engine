"""Particle Neural Network (PNN) — original architecture.

Novel idea: instead of a fixed-topology weighted graph, neurons are *free
particles* in continuous 3D space. Connections are not stored; they are
reconstructed each forward pass from spatial proximity and embedding
similarity. This means the network's topology is an emergent property of
particle positions — and it physically reorganizes during training.

Each particle carries:
  - position:    R^3   (where it lives in latent space)
  - velocity:    R^3   (used by the Particle Gradient Descent trainer)
  - embedding:   {-1, +1}^D   (its semantic identity, an HDC hypervector)
  - mass:        R+    (importance / persistence; decays over time)
  - charge:      R     (firing threshold)
  - polarity:    ±1    (excitatory or inhibitory)

Forward pass = wave propagation. Input particles inject energy; energy
diffuses through the field with strength proportional to embedding similarity
× polarity × proximity falloff. Output particles measure their absorbed
energy, which becomes the network's response.

Spatial hashing keeps neighbour queries O(N) per step instead of O(N^2),
which is required to stay within the architect's recommended ~5-20k particle
ceiling at interactive frame rates.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from cognitron.hdc import DEFAULT_DIM, cosine

# Hyperparameters (kept small and named so they can be swept later)
DEFAULT_RADIUS = 0.35  # spatial neighborhood radius for force calculations
DEFAULT_PROPAGATION_STEPS = 6  # how many wave-propagation steps per forward pass
DEFAULT_DAMPING = 0.85  # energy retained between propagation steps
DEFAULT_SIMILARITY_THRESHOLD = 0.05  # embeddings below this don't conduct


@dataclass
class Particle:
    """A single neuron-as-particle. Immutable update pattern: methods return
    new Particle copies rather than mutating in place. This matches the
    project's coding-style rule and makes the trainer's state easier to
    reason about.
    """

    id: int
    position: np.ndarray  # shape (3,), float32
    embedding: np.ndarray  # shape (D,), int8 in {-1,+1}
    mass: float = 1.0
    charge: float = 0.5
    polarity: int = 1  # +1 excitatory, -1 inhibitory
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    text: str = ""  # optional human-readable provenance

    def with_position(self, p: np.ndarray) -> "Particle":
        return Particle(
            id=self.id,
            position=np.asarray(p, dtype=np.float32),
            embedding=self.embedding,
            mass=self.mass,
            charge=self.charge,
            polarity=self.polarity,
            velocity=self.velocity,
            text=self.text,
        )

    def with_velocity(self, v: np.ndarray) -> "Particle":
        return Particle(
            id=self.id,
            position=self.position,
            embedding=self.embedding,
            mass=self.mass,
            charge=self.charge,
            polarity=self.polarity,
            velocity=np.asarray(v, dtype=np.float32),
            text=self.text,
        )

    def with_mass(self, m: float) -> "Particle":
        return Particle(
            id=self.id,
            position=self.position,
            embedding=self.embedding,
            mass=float(m),
            charge=self.charge,
            polarity=self.polarity,
            velocity=self.velocity,
            text=self.text,
        )


# ---------------------------------------------------------------------------
# Spatial hash grid
# ---------------------------------------------------------------------------


class SpatialHash:
    """Uniform-grid spatial hash for radius queries on particle positions.

    Cell size is exactly the query radius, so a radius query touches at most
    27 cells (3x3x3 cube around the query cell). This is the standard
    Stanford / GDC technique and is what makes >5k-particle networks tractable
    on a CPU.
    """

    def __init__(self, cell_size: float):
        self.cell_size = float(cell_size)
        self._cells: dict[tuple[int, int, int], list[int]] = defaultdict(list)

    def _key(self, p: np.ndarray) -> tuple[int, int, int]:
        s = self.cell_size
        return (int(p[0] // s), int(p[1] // s), int(p[2] // s))

    def rebuild(self, particles: list[Particle]) -> None:
        self._cells.clear()
        for particle in particles:
            self._cells[self._key(particle.position)].append(particle.id)

    def neighbors(self, position: np.ndarray) -> list[int]:
        cx, cy, cz = self._key(position)
        out: list[int] = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    out.extend(self._cells.get((cx + dx, cy + dy, cz + dz), ()))
        return out


# ---------------------------------------------------------------------------
# Particle Network
# ---------------------------------------------------------------------------


class ParticleNetwork:
    """A field of particles that performs wave-propagation inference.

    Stable interface:
      add_particle(embedding, text=, ...) -> Particle
      forward(input_ids, energy=1.0) -> dict[id -> absorbed_energy]
      query(embedding, k=5) -> list[(particle_id, score)]
      decay(half_life_steps=N) -> None    (simulated forgetting)
      snapshot() -> SnapshotDict          (for serialization / WebGPU upload)
    """

    def __init__(
        self,
        embedding_dim: int = DEFAULT_DIM,
        radius: float = DEFAULT_RADIUS,
        propagation_steps: int = DEFAULT_PROPAGATION_STEPS,
        damping: float = DEFAULT_DAMPING,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        seed: int = 0,
    ) -> None:
        self.embedding_dim = embedding_dim
        self.radius = float(radius)
        self.propagation_steps = int(propagation_steps)
        self.damping = float(damping)
        self.similarity_threshold = float(similarity_threshold)
        self._rng = np.random.default_rng(seed)
        self._particles: dict[int, Particle] = {}
        self._next_id = 0
        self._hash = SpatialHash(cell_size=self.radius)
        self._hash_dirty = True

    # -- public API ---------------------------------------------------------

    @property
    def particles(self) -> list[Particle]:
        return list(self._particles.values())

    def __len__(self) -> int:
        return len(self._particles)

    def add_particle(
        self,
        embedding: np.ndarray,
        text: str = "",
        mass: float = 1.0,
        polarity: int = 1,
        position: np.ndarray | None = None,
    ) -> Particle:
        if embedding.shape[0] != self.embedding_dim:
            raise ValueError(
                f"embedding dim {embedding.shape[0]} != network dim {self.embedding_dim}"
            )
        pid = self._next_id
        self._next_id += 1
        if position is None:
            position = self._initial_position_for(embedding)
        particle = Particle(
            id=pid,
            position=np.asarray(position, dtype=np.float32),
            embedding=embedding.astype(np.int8),
            mass=float(mass),
            polarity=int(polarity),
            charge=0.5,
            text=text,
        )
        self._particles[pid] = particle
        self._hash_dirty = True
        return particle

    def replace_particle(self, particle: Particle) -> None:
        self._particles[particle.id] = particle
        self._hash_dirty = True

    def forward(
        self, input_ids: Iterable[int], energy: float = 1.0
    ) -> dict[int, float]:
        """Wave-propagation inference. Inject `energy` at each input particle,
        propagate `propagation_steps` rounds, return absorbed-energy per
        particle. Output ranking is taken by the caller."""
        self._maybe_rebuild_hash()
        absorbed: dict[int, float] = defaultdict(float)
        # Energy currently traveling on each particle this step.
        traveling: dict[int, float] = defaultdict(float)
        for pid in input_ids:
            if pid not in self._particles:
                continue
            traveling[pid] = energy
            absorbed[pid] += energy

        for _step in range(self.propagation_steps):
            next_traveling: dict[int, float] = defaultdict(float)
            for pid, e in traveling.items():
                if e < 1e-6:
                    continue
                src = self._particles[pid]
                neighbors = self._hash.neighbors(src.position)
                # Compute conductances to each neighbor
                pairs: list[tuple[int, float]] = []
                total_weight = 0.0
                for nid in neighbors:
                    if nid == pid:
                        continue
                    dst = self._particles[nid]
                    d = float(np.linalg.norm(dst.position - src.position))
                    if d > self.radius:
                        continue
                    sim = cosine(src.embedding, dst.embedding)
                    if sim < self.similarity_threshold:
                        continue
                    falloff = max(0.0, 1.0 - d / self.radius)
                    w = sim * falloff * dst.mass * dst.polarity
                    if w <= 0:
                        continue
                    pairs.append((nid, w))
                    total_weight += w
                if total_weight <= 0:
                    continue
                # Distribute energy proportionally and apply damping
                outflow = e * self.damping
                for nid, w in pairs:
                    share = outflow * (w / total_weight)
                    next_traveling[nid] += share
                    absorbed[nid] += share
            traveling = next_traveling

        return dict(absorbed)

    def query(
        self, embedding: np.ndarray, k: int = 5, energy: float = 1.0
    ) -> list[tuple[int, float]]:
        """Treat the query embedding as a transient input particle. Find the
        k particles that absorb the most energy from a wave seeded at the
        embedding's natural position."""
        self._maybe_rebuild_hash()
        seed_pos = self._initial_position_for(embedding)
        # Pick the closest existing particle and use it as the input. If the
        # field is empty, return [].
        if not self._particles:
            return []
        best_id, best_d = None, float("inf")
        # Brute-force over all particles for the seed search — this is
        # O(N) and dominated by the wave step's neighbour search.
        for p in self._particles.values():
            d = float(np.linalg.norm(p.position - seed_pos))
            if d < best_d:
                best_d, best_id = d, p.id
        assert best_id is not None
        absorbed = self.forward([best_id], energy=energy)
        ranked = sorted(absorbed.items(), key=lambda kv: kv[1], reverse=True)
        return ranked[:k]

    def decay(self, factor: float = 0.99) -> None:
        """Multiplicative mass decay — a simple forgetting curve."""
        for pid, p in list(self._particles.items()):
            new_mass = p.mass * factor
            if new_mass < 1e-3:
                del self._particles[pid]
                self._hash_dirty = True
            else:
                self._particles[pid] = p.with_mass(new_mass)
                self._hash_dirty = True

    def reinforce(self, particle_ids: Iterable[int], amount: float = 0.1) -> None:
        """Reinforce mass on accessed particles — the inverse of decay."""
        for pid in particle_ids:
            p = self._particles.get(pid)
            if p is None:
                continue
            self._particles[pid] = p.with_mass(p.mass + amount)

    def snapshot(self) -> dict:
        """Serialize the field for the WebGPU renderer or persistence layer.

        Returns a dict with parallel arrays — exactly what the WGSL shader
        consumes, no per-particle objects to walk on the JS side.
        """
        n = len(self._particles)
        positions = np.zeros((n, 3), dtype=np.float32)
        masses = np.zeros(n, dtype=np.float32)
        polarities = np.zeros(n, dtype=np.int8)
        ids = np.zeros(n, dtype=np.int32)
        texts: list[str] = []
        for i, p in enumerate(self._particles.values()):
            positions[i] = p.position
            masses[i] = p.mass
            polarities[i] = p.polarity
            ids[i] = p.id
            texts.append(p.text)
        return {
            "positions": positions,
            "masses": masses,
            "polarities": polarities,
            "ids": ids,
            "texts": texts,
        }

    # -- internals ----------------------------------------------------------

    def _maybe_rebuild_hash(self) -> None:
        if self._hash_dirty:
            self._hash.rebuild(list(self._particles.values()))
            self._hash_dirty = False

    def _initial_position_for(self, embedding: np.ndarray) -> np.ndarray:
        """Project a hypervector into 3D using fixed random projection.

        Three orthogonal random directions in the embedding space define x/y/z
        in the visualisation. Because the projection is fixed (seeded), the
        same embedding always lands at the same coordinate — which is
        important so the visualisation is deterministic and so similar
        thoughts naturally appear near each other before any training.
        """
        # Use a fixed seed independent of self._rng so the projection is
        # stable across instantiations.
        proj = np.random.default_rng(2024).standard_normal((3, self.embedding_dim)).astype(
            np.float32
        )
        # Normalize each axis projection
        proj /= np.linalg.norm(proj, axis=1, keepdims=True) + 1e-8
        coord = proj @ embedding.astype(np.float32)
        # Scale into [-1, 1]^3 cube (rough — exact bounds depend on dim)
        coord = coord / (np.sqrt(self.embedding_dim) * 0.5)
        return np.clip(coord, -1.0, 1.0).astype(np.float32)
