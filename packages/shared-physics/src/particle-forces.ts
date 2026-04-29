/**
 * CPU reference implementation of the particle-force compute pass.
 *
 * Inputs (parallel arrays — same memory layout the WGSL shader binds):
 *   positions   : Float32Array  length 3N    (xyz)
 *   masses      : Float32Array  length  N
 *   polarities  : Int32Array    length  N    (+1 or -1)
 *   embeddings  : Int8Array     length D*N  (bipolar -1/+1)
 *
 * Output:
 *   newPositions : Float32Array length 3N    — positions after one step
 *   absorbedEnergy : Float32Array length N   — read by query rendering
 *
 * The WGSL shader in shaders/particle-forces.wgsl mirrors this code
 * function-for-function. Differences are caught by `diffStep()`.
 */

export interface ParticleField {
  n: number;
  d: number; // embedding dimension
  positions: Float32Array; // 3N
  velocities: Float32Array; // 3N
  masses: Float32Array; // N
  polarities: Int32Array; // N
  embeddings: Int8Array; // D*N
}

export interface ForceParams {
  radius: number; // neighbor cutoff in world units
  similarityThreshold: number; // embeddings below this don't conduct
  damping: number; // velocity damping each step
  dt: number; // integration timestep
}

export const DEFAULT_FORCE_PARAMS: ForceParams = {
  radius: 0.45,
  similarityThreshold: 0.05,
  damping: 0.92,
  dt: 1 / 60,
};

/** Cosine similarity for two slices of the bipolar embeddings buffer. */
export function bipolarCosine(
  embeddings: Int8Array,
  d: number,
  i: number,
  j: number,
): number {
  let dot = 0;
  const offI = i * d;
  const offJ = j * d;
  for (let k = 0; k < d; k++) {
    dot += embeddings[offI + k] * embeddings[offJ + k];
  }
  // For bipolar +/-1 vectors, ||a|| = ||b|| = sqrt(D), so denom = D.
  return dot / d;
}

/**
 * One Euler step of the particle field. Mutates field's velocity/position
 * arrays in place; returns the per-particle absorbed energy from the wave
 * propagation pass (used to colour the field by activity).
 */
export function stepParticles(
  field: ParticleField,
  params: ForceParams = DEFAULT_FORCE_PARAMS,
  inputs: Map<number, number> = new Map(),
): Float32Array {
  const { n, d, positions, velocities, masses, polarities } = field;
  const { radius, similarityThreshold, damping, dt } = params;
  const r2 = radius * radius;

  // Wave propagation: travel one hop, accumulate absorbed energy
  const absorbed = new Float32Array(n);
  const traveling = new Float32Array(n);
  inputs.forEach((energy, idx) => {
    if (idx >= 0 && idx < n) {
      traveling[idx] = energy;
      absorbed[idx] = energy;
    }
  });
  // 4 propagation hops — kept small so the GPU shader can unroll
  for (let step = 0; step < 4; step++) {
    const next = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      const e = traveling[i];
      if (e < 1e-6) continue;
      // Find neighbors (brute-force for the reference impl; the shader uses
      // a spatial hash, but for verification on small N this is identical).
      let totalW = 0;
      const weights: { j: number; w: number }[] = [];
      for (let j = 0; j < n; j++) {
        if (j === i) continue;
        const dx = positions[3 * j] - positions[3 * i];
        const dy = positions[3 * j + 1] - positions[3 * i + 1];
        const dz = positions[3 * j + 2] - positions[3 * i + 2];
        const distSq = dx * dx + dy * dy + dz * dz;
        if (distSq > r2) continue;
        const sim = bipolarCosine(field.embeddings, d, i, j);
        if (sim < similarityThreshold) continue;
        const dist = Math.sqrt(distSq);
        const falloff = Math.max(0, 1 - dist / radius);
        const w = sim * falloff * masses[j] * polarities[j];
        if (w <= 0) continue;
        weights.push({ j, w });
        totalW += w;
      }
      if (totalW <= 0) continue;
      const outflow = e * damping;
      for (const { j, w } of weights) {
        const share = (outflow * w) / totalW;
        next[j] += share;
        absorbed[j] += share;
      }
    }
    traveling.set(next);
  }

  // Integrate velocities under attractive force from high-similarity
  // neighbors. The trainer in the Python backend handles the *learning*
  // step; this in-browser pass only handles ambient drift toward
  // semantic neighbours so the live visualisation looks alive even
  // without a training loop running.
  for (let i = 0; i < n; i++) {
    let fx = 0;
    let fy = 0;
    let fz = 0;
    for (let j = 0; j < n; j++) {
      if (j === i) continue;
      const dx = positions[3 * j] - positions[3 * i];
      const dy = positions[3 * j + 1] - positions[3 * i + 1];
      const dz = positions[3 * j + 2] - positions[3 * i + 2];
      const distSq = dx * dx + dy * dy + dz * dz;
      if (distSq > r2 || distSq < 1e-6) continue;
      const sim = bipolarCosine(field.embeddings, d, i, j);
      if (sim < similarityThreshold) continue;
      const dist = Math.sqrt(distSq);
      const k = (sim - 0.5) * 0.02; // attractive if sim>0.5, repulsive otherwise
      fx += (k * dx) / dist;
      fy += (k * dy) / dist;
      fz += (k * dz) / dist;
    }
    velocities[3 * i] = velocities[3 * i] * damping + fx * dt;
    velocities[3 * i + 1] = velocities[3 * i + 1] * damping + fy * dt;
    velocities[3 * i + 2] = velocities[3 * i + 2] * damping + fz * dt;
    positions[3 * i] += velocities[3 * i] * dt;
    positions[3 * i + 1] += velocities[3 * i + 1] * dt;
    positions[3 * i + 2] += velocities[3 * i + 2] * dt;
    // Soft bound the field
    for (let axis = 0; axis < 3; axis++) {
      const idx = 3 * i + axis;
      if (positions[idx] > 1.5) {
        positions[idx] = 1.5;
        velocities[idx] *= -0.5;
      } else if (positions[idx] < -1.5) {
        positions[idx] = -1.5;
        velocities[idx] *= -0.5;
      }
    }
  }
  return absorbed;
}
