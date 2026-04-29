// NCA voxel smoothing pass (frontend only — full NCA inference is on the
// Python service). Used to interpolate between server-streamed frames so
// the visual feels continuous at 60fps even when the server pushes 12fps.

struct Params {
  size : u32,
  alpha : f32,
};

@group(0) @binding(0) var<storage, read> rgba_in  : array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> rgba_out : array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params : Params;

fn idx(x : i32, y : i32, z : i32) -> i32 {
  let s = i32(params.size);
  return (x * s + y) * s + z;
}

@compute @workgroup_size(4, 4, 4)
fn smooth(@builtin(global_invocation_id) gid : vec3<u32>) {
  let s = i32(params.size);
  if (gid.x >= u32(s) || gid.y >= u32(s) || gid.z >= u32(s)) { return; }
  let x = i32(gid.x);
  let y = i32(gid.y);
  let z = i32(gid.z);
  var sum = vec4<f32>(0.0);
  var count = 0.0;
  for (var dx = -1; dx <= 1; dx = dx + 1) {
    for (var dy = -1; dy <= 1; dy = dy + 1) {
      for (var dz = -1; dz <= 1; dz = dz + 1) {
        let nx = x + dx;
        let ny = y + dy;
        let nz = z + dz;
        if (nx < 0 || ny < 0 || nz < 0 || nx >= s || ny >= s || nz >= s) { continue; }
        sum = sum + rgba_in[idx(nx, ny, nz)];
        count = count + 1.0;
      }
    }
  }
  let mean = sum / count;
  let cur = rgba_in[idx(x, y, z)];
  rgba_out[idx(x, y, z)] = mix(cur, mean, params.alpha);
}
