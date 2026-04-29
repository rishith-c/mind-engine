// Particle-force compute shader (Cognitron live view).
//
// Mirrors the TS reference in src/particle-forces.ts. To debug, render both,
// download GPU buffers, and diff against the CPU output.
//
// Bind group layout (group 0):
//   binding 0 : storage rw    Positions  (vec3<f32> N, padded to vec4)
//   binding 1 : storage rw    Velocities (vec3<f32> N, padded to vec4)
//   binding 2 : storage  r    Masses     (f32 N)
//   binding 3 : storage  r    Polarities (i32 N)
//   binding 4 : storage  r    Embeddings (i32 D*N packed: 4 int8 per word)
//   binding 5 : uniform       Params     (radius, simThresh, damping, dt, n, d)

struct Params {
  radius : f32,
  similarity_threshold : f32,
  damping : f32,
  dt : f32,
  n : u32,
  d : u32,
  _pad0 : u32,
  _pad1 : u32,
};

@group(0) @binding(0) var<storage, read_write> positions  : array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> velocities : array<vec4<f32>>;
@group(0) @binding(2) var<storage, read>       masses     : array<f32>;
@group(0) @binding(3) var<storage, read>       polarities : array<i32>;
@group(0) @binding(4) var<storage, read>       embeddings : array<i32>;
@group(0) @binding(5) var<uniform>             params     : Params;

// Bipolar cosine: ints packed 4-per-word as +/-1. For demo we keep things
// simple and store one int8 per i32 word — wastes memory but trivial to
// read on the GPU.
fn cosine(i : u32, j : u32) -> f32 {
  var dot : f32 = 0.0;
  let off_i = i * params.d;
  let off_j = j * params.d;
  for (var k : u32 = 0u; k < params.d; k = k + 1u) {
    let a = f32(embeddings[off_i + k]);
    let b = f32(embeddings[off_j + k]);
    dot = dot + a * b;
  }
  return dot / f32(params.d);
}

@compute @workgroup_size(64)
fn integrate(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  if (i >= params.n) {
    return;
  }
  let pos_i = positions[i].xyz;
  var force = vec3<f32>(0.0);
  let r = params.radius;
  let r2 = r * r;

  for (var j : u32 = 0u; j < params.n; j = j + 1u) {
    if (j == i) { continue; }
    let pos_j = positions[j].xyz;
    let delta = pos_j - pos_i;
    let dist_sq = dot(delta, delta);
    if (dist_sq > r2 || dist_sq < 1e-6) { continue; }
    let sim = cosine(i, j);
    if (sim < params.similarity_threshold) { continue; }
    let dist = sqrt(dist_sq);
    let k = (sim - 0.5) * 0.02;
    force = force + (k / dist) * delta;
  }

  var v = velocities[i].xyz * params.damping + force * params.dt;
  var p = pos_i + v * params.dt;

  // Soft bound +/-1.5 cube
  for (var axis : u32 = 0u; axis < 3u; axis = axis + 1u) {
    if (p[axis] > 1.5) { p[axis] = 1.5; v[axis] = v[axis] * -0.5; }
    if (p[axis] < -1.5) { p[axis] = -1.5; v[axis] = v[axis] * -0.5; }
  }

  positions[i] = vec4<f32>(p, 1.0);
  velocities[i] = vec4<f32>(v, 0.0);
}
