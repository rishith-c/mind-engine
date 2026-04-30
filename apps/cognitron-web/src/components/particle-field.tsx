"use client";

import { useMemo, useRef } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { OrbitControls, Stars } from "@react-three/drei";
import { EffectComposer, Bloom } from "@react-three/postprocessing";
import * as THREE from "three";
import { useCognitronStore } from "@/lib/store";

/**
 * Cognitron particle field — proper "neural substrate" visualisation.
 *
 * Architecture (per code-review feedback):
 *   1. THREE.Points with custom shader for procedural radial alpha + size
 *      attenuation + depth fade. No sphere geometry, no light sources.
 *   2. Dual-pass: a small bright core layer + a large soft halo layer with
 *      additive blending. This is what makes points look like cosmic objects
 *      instead of styrofoam balls.
 *   3. Idle Brownian drift in useFrame so the field always looks alive.
 *   4. <Bloom/> postprocessing — does most of the cosmic-glow work for free.
 *   5. Highlighted particles (query hits) flash accent cyan.
 */

const VERT_SHADER = /* glsl */ `
  attribute float aSize;
  attribute float aHighlight;
  varying float vHighlight;
  varying float vDepth;

  void main() {
    vHighlight = aHighlight;
    vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
    vDepth = -mvPosition.z;
    gl_PointSize = aSize * (300.0 / vDepth);
    gl_Position = projectionMatrix * mvPosition;
  }
`;

const FRAG_SHADER = /* glsl */ `
  uniform vec3 uColorBase;
  uniform vec3 uColorHighlight;
  uniform float uOpacity;
  varying float vHighlight;
  varying float vDepth;

  void main() {
    // Procedural radial alpha — soft circle without a texture
    vec2 uv = gl_PointCoord - 0.5;
    float r = length(uv) * 2.0;
    if (r > 1.0) discard;
    // Smooth gaussian-ish falloff
    float alpha = pow(1.0 - r, 2.4);

    // Depth fade — closer = brighter (bias toward camera)
    float depthFade = smoothstep(8.0, 1.5, vDepth);

    vec3 col = mix(uColorBase, uColorHighlight, vHighlight);
    gl_FragColor = vec4(col, alpha * uOpacity * depthFade);
  }
`;

function buildShader(
  base: THREE.Color,
  highlight: THREE.Color,
  opacity: number,
  blending: THREE.Blending,
): THREE.ShaderMaterial {
  return new THREE.ShaderMaterial({
    uniforms: {
      uColorBase: { value: base },
      uColorHighlight: { value: highlight },
      uOpacity: { value: opacity },
    },
    vertexShader: VERT_SHADER,
    fragmentShader: FRAG_SHADER,
    transparent: true,
    depthWrite: false,
    blending,
  });
}

interface FieldGeometryArgs {
  positions: Float32Array;
  sizes: Float32Array;
  highlights: Float32Array;
}

function makeGeometry({
  positions,
  sizes,
  highlights,
}: FieldGeometryArgs): THREE.BufferGeometry {
  const geom = new THREE.BufferGeometry();
  geom.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  geom.setAttribute("aSize", new THREE.BufferAttribute(sizes, 1));
  geom.setAttribute("aHighlight", new THREE.BufferAttribute(highlights, 1));
  return geom;
}

function ParticlePoints() {
  const corePoints = useRef<THREE.Points>(null);
  const haloPoints = useRef<THREE.Points>(null);

  const field = useCognitronStore((s) => s.field);
  const highlighted = useCognitronStore((s) => s.highlighted);

  // Buffers — re-allocated when field size changes; otherwise updated in
  // place via useFrame for idle drift.
  const buffers = useMemo(() => {
    if (!field || field.n === 0) {
      return {
        positions: new Float32Array(0),
        sizes: new Float32Array(0),
        highlights: new Float32Array(0),
        rest: new Float32Array(0),
        idHashes: new Float32Array(0),
      };
    }
    const n = field.n;
    const positions = new Float32Array(n * 3);
    const rest = new Float32Array(n * 3);
    const sizes = new Float32Array(n);
    const highlights = new Float32Array(n);
    const idHashes = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      const [x, y, z] = field.positions[i];
      positions[3 * i] = x;
      positions[3 * i + 1] = y;
      positions[3 * i + 2] = z;
      rest[3 * i] = x;
      rest[3 * i + 1] = y;
      rest[3 * i + 2] = z;
      sizes[i] = 0.5 + Math.min(1.5, field.masses[i] * 1.2);
      highlights[i] = highlighted.has(field.ids[i]) ? 1.0 : 0.0;
      // Cheap deterministic hash off the integer id for drift offsets
      idHashes[i] = (Math.sin(field.ids[i] * 73.31) * 43758.5453) % 1;
    }
    return { positions, rest, sizes, highlights, idHashes };
  }, [field, highlighted]);

  const coreGeom = useMemo(
    () => makeGeometry({
      positions: buffers.positions,
      sizes: new Float32Array(buffers.sizes.map((s) => s * 6)),
      highlights: buffers.highlights,
    }),
    [buffers],
  );

  const haloGeom = useMemo(
    () => makeGeometry({
      positions: buffers.positions,
      sizes: new Float32Array(buffers.sizes.map((s) => s * 28)),
      highlights: buffers.highlights,
    }),
    [buffers],
  );

  const coreMat = useMemo(
    () =>
      buildShader(
        new THREE.Color("#f0f4f7"),
        new THREE.Color("#80e8f7"),
        1.0,
        THREE.AdditiveBlending,
      ),
    [],
  );
  const haloMat = useMemo(
    () =>
      buildShader(
        new THREE.Color("#80c8e0"),
        new THREE.Color("#80e8f7"),
        0.22,
        THREE.AdditiveBlending,
      ),
    [],
  );

  useFrame((state) => {
    if (!field || field.n === 0) return;
    const t = state.clock.elapsedTime;
    const corePos = corePoints.current?.geometry.attributes.position;
    const haloPos = haloPoints.current?.geometry.attributes.position;
    const coreHi = corePoints.current?.geometry.attributes.aHighlight;
    const haloHi = haloPoints.current?.geometry.attributes.aHighlight;
    if (!corePos || !haloPos || !coreHi || !haloHi) return;

    const corePosArr = corePos.array as Float32Array;
    const haloPosArr = haloPos.array as Float32Array;
    const coreHiArr = coreHi.array as Float32Array;
    const haloHiArr = haloHi.array as Float32Array;

    for (let i = 0; i < field.n; i++) {
      const phaseX = buffers.idHashes[i] * 6.28;
      const phaseY = phaseX + 1.4;
      const phaseZ = phaseX + 2.8;

      // Brownian-ish drift around the rest position
      const dx = Math.sin(t * 0.42 + phaseX) * 0.018;
      const dy = Math.cos(t * 0.36 + phaseY) * 0.018;
      const dz = Math.sin(t * 0.50 + phaseZ) * 0.014;

      corePosArr[3 * i] = buffers.rest[3 * i] + dx;
      corePosArr[3 * i + 1] = buffers.rest[3 * i + 1] + dy;
      corePosArr[3 * i + 2] = buffers.rest[3 * i + 2] + dz;
      haloPosArr[3 * i] = corePosArr[3 * i];
      haloPosArr[3 * i + 1] = corePosArr[3 * i + 1];
      haloPosArr[3 * i + 2] = corePosArr[3 * i + 2];

      // Highlight pulse for query hits
      const isHi = highlighted.has(field.ids[i]);
      const target = isHi ? 0.7 + 0.3 * Math.sin(t * 6.0) : 0.0;
      coreHiArr[i] = target;
      haloHiArr[i] = target;
    }
    corePos.needsUpdate = true;
    haloPos.needsUpdate = true;
    coreHi.needsUpdate = true;
    haloHi.needsUpdate = true;
  });

  if (!field || field.n === 0) return null;

  return (
    <group>
      <points ref={haloPoints} geometry={haloGeom} material={haloMat} />
      <points ref={corePoints} geometry={coreGeom} material={coreMat} />
    </group>
  );
}

function ConnectionLines() {
  const field = useCognitronStore((s) => s.field);
  const highlighted = useCognitronStore((s) => s.highlighted);

  const segments = useMemo(() => {
    if (!field || highlighted.size === 0) return null;
    const positions = field.positions;
    const ids = field.ids;
    const idToIdx = new Map<number, number>();
    ids.forEach((id, idx) => idToIdx.set(id, idx));

    const out: number[] = [];
    highlighted.forEach((id) => {
      const i = idToIdx.get(id);
      if (i === undefined) return;
      // Connect to k nearest particles
      const dists = positions
        .map(
          (p, j) =>
            [
              j,
              Math.hypot(
                p[0] - positions[i][0],
                p[1] - positions[i][1],
                p[2] - positions[i][2],
              ),
            ] as [number, number],
        )
        .sort((a, b) => a[1] - b[1])
        .slice(1, 4);
      dists.forEach(([j]) => {
        out.push(
          positions[i][0],
          positions[i][1],
          positions[i][2],
          positions[j][0],
          positions[j][1],
          positions[j][2],
        );
      });
    });

    if (out.length === 0) return null;
    const geom = new THREE.BufferGeometry();
    geom.setAttribute("position", new THREE.BufferAttribute(new Float32Array(out), 3));
    return geom;
  }, [field, highlighted]);

  if (!segments) return null;
  return (
    <lineSegments geometry={segments}>
      <lineBasicMaterial
        color="#80e8f7"
        transparent
        opacity={0.35}
        blending={THREE.AdditiveBlending}
        depthWrite={false}
      />
    </lineSegments>
  );
}

export function ParticleField() {
  return (
    <Canvas
      camera={{ position: [0, 0.4, 3.4], fov: 48 }}
      style={{ background: "transparent" }}
      gl={{ antialias: true, alpha: true }}
    >
      <Stars
        radius={40}
        depth={80}
        count={1800}
        factor={1.6}
        saturation={0}
        fade
        speed={0.3}
      />
      <ParticlePoints />
      <ConnectionLines />
      <OrbitControls
        enablePan={false}
        autoRotate
        autoRotateSpeed={0.18}
        minDistance={2.0}
        maxDistance={8}
        target={[0, 0, 0]}
      />
      <EffectComposer>
        <Bloom
          intensity={0.9}
          luminanceThreshold={0.08}
          luminanceSmoothing={0.4}
          mipmapBlur
        />
      </EffectComposer>
    </Canvas>
  );
}
