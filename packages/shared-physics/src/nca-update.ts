/**
 * CPU reference for the NCA cell-update shader. The full network is run on
 * the Python backend; this client-side step exists only to interpolate
 * between server frames smoothly without round-tripping every microstep.
 *
 * Implementation note: this is *not* the full Sobel-perception + MLP update
 * (that lives on the server). It is a low-cost local-blur micro-update used
 * purely for visual smoothness during streaming playback.
 */

export interface VoxelField {
  size: number; // grid edge
  rgba: Float32Array; // size^3 * 4
}

export function smoothStep(field: VoxelField, alpha: number = 0.15): VoxelField {
  const { size } = field;
  const out = new Float32Array(field.rgba.length);
  const idx = (x: number, y: number, z: number, c: number) =>
    ((x * size + y) * size + z) * 4 + c;

  for (let x = 0; x < size; x++) {
    for (let y = 0; y < size; y++) {
      for (let z = 0; z < size; z++) {
        for (let c = 0; c < 4; c++) {
          let sum = 0;
          let count = 0;
          for (let dx = -1; dx <= 1; dx++) {
            for (let dy = -1; dy <= 1; dy++) {
              for (let dz = -1; dz <= 1; dz++) {
                const nx = x + dx;
                const ny = y + dy;
                const nz = z + dz;
                if (nx < 0 || ny < 0 || nz < 0 || nx >= size || ny >= size || nz >= size)
                  continue;
                sum += field.rgba[idx(nx, ny, nz, c)];
                count++;
              }
            }
          }
          const mean = sum / count;
          const cur = field.rgba[idx(x, y, z, c)];
          out[idx(x, y, z, c)] = cur * (1 - alpha) + mean * alpha;
        }
      }
    }
  }
  return { size, rgba: out };
}
