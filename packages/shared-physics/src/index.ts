/**
 * @mind/physics — shared force-and-update math used by both Cognitron and
 * Morpheus, in both their WGSL shader form (for the GPU) and their TS
 * reference form (for testing + WebGPU-unsupported fallback).
 *
 * Architect's directive: "WebGPU compute shader debugging has no real
 * debugger ... write CPU reference implementations for every shader, and
 * diff outputs." That is exactly what this package does.
 */

export * from "./particle-forces";
export * from "./nca-update";
export { default as PARTICLE_SHADER_WGSL } from "./shaders/particle-forces.wgsl?raw";
export { default as NCA_SHADER_WGSL } from "./shaders/nca-update.wgsl?raw";
