"use client";

import { useMemo, useRef } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { OrbitControls, Stars } from "@react-three/drei";
import * as THREE from "three";
import { useCognitronStore } from "@/lib/store";

/**
 * 3D visualisation of the particle field. We use instanced meshes so a
 * 5,000-particle render stays at 60fps on integrated GPUs (the architect's
 * recommended ceiling). Each particle is colored by polarity and pulsed by
 * its highlight state during a query.
 */

const tmpObj = new THREE.Object3D();
const tmpColor = new THREE.Color();

function ParticleInstances() {
  const meshRef = useRef<THREE.InstancedMesh>(null);
  const field = useCognitronStore((s) => s.field);
  const highlighted = useCognitronStore((s) => s.highlighted);

  const baseColors = useMemo(() => {
    if (!field) return new Float32Array(0);
    const cols = new Float32Array(field.n * 3);
    for (let i = 0; i < field.n; i++) {
      // Editorial palette: paper white for excitatory, oxide cyan for
      // inhibitory. Restrained on purpose — the field reads as a
      // scientific instrument, not a discotheque.
      const pol = field.polarities[i];
      if (pol > 0) {
        cols[3 * i] = 0.92;
        cols[3 * i + 1] = 0.93;
        cols[3 * i + 2] = 0.96;
      } else {
        cols[3 * i] = 0.45;
        cols[3 * i + 1] = 0.85;
        cols[3 * i + 2] = 0.92;
      }
    }
    return cols;
  }, [field]);

  useFrame((state) => {
    if (!meshRef.current || !field) return;
    const t = state.clock.elapsedTime;
    for (let i = 0; i < field.n; i++) {
      const [x, y, z] = field.positions[i];
      tmpObj.position.set(x, y, z);
      const isHi = highlighted.has(field.ids[i]);
      const baseScale = 0.02 + Math.min(0.05, field.masses[i] * 0.02);
      const pulseScale = isHi ? 1.6 + Math.sin(t * 8) * 0.4 : 1.0;
      tmpObj.scale.setScalar(baseScale * pulseScale);
      tmpObj.updateMatrix();
      meshRef.current.setMatrixAt(i, tmpObj.matrix);

      const r = baseColors[3 * i];
      const g = baseColors[3 * i + 1];
      const b = baseColors[3 * i + 2];
      if (isHi) {
        // Highlighted hits glow accent cyan — the only accent color used
        tmpColor.setRGB(0.55, 0.95, 1.0);
      } else {
        tmpColor.setRGB(r, g, b);
      }
      meshRef.current.setColorAt(i, tmpColor);
    }
    meshRef.current.instanceMatrix.needsUpdate = true;
    if (meshRef.current.instanceColor) meshRef.current.instanceColor.needsUpdate = true;
  });

  if (!field || field.n === 0) return null;

  return (
    <instancedMesh
      ref={meshRef}
      args={[undefined, undefined, Math.max(field.n, 1)]}
      castShadow={false}
      receiveShadow={false}
    >
      <sphereGeometry args={[1, 12, 12]} />
      <meshBasicMaterial toneMapped={false} />
    </instancedMesh>
  );
}

function ConnectionLines() {
  // Light "wave traces" between highlighted particles and their neighbours
  const field = useCognitronStore((s) => s.field);
  const highlighted = useCognitronStore((s) => s.highlighted);

  const segments = useMemo(() => {
    if (!field || highlighted.size === 0) return [] as [THREE.Vector3, THREE.Vector3][];
    const out: [THREE.Vector3, THREE.Vector3][] = [];
    const positions = field.positions;
    const ids = field.ids;
    const idToIdx = new Map<number, number>();
    ids.forEach((id, idx) => idToIdx.set(id, idx));
    highlighted.forEach((id) => {
      const i = idToIdx.get(id);
      if (i === undefined) return;
      const a = new THREE.Vector3(...positions[i]);
      // Connect to k nearest particles
      const dists = positions
        .map(
          (p, j) =>
            [
              j,
              Math.hypot(p[0] - positions[i][0], p[1] - positions[i][1], p[2] - positions[i][2]),
            ] as [number, number],
        )
        .sort((x, y) => x[1] - y[1])
        .slice(1, 5);
      dists.forEach(([j]) => {
        const b = new THREE.Vector3(...positions[j]);
        out.push([a, b]);
      });
    });
    return out;
  }, [field, highlighted]);

  if (segments.length === 0) return null;

  return (
    <group>
      {segments.map((seg, i) => (
        <line key={i}>
          <bufferGeometry attach="geometry">
            <bufferAttribute
              attach="attributes-position"
              args={[
                new Float32Array([
                  seg[0].x,
                  seg[0].y,
                  seg[0].z,
                  seg[1].x,
                  seg[1].y,
                  seg[1].z,
                ]),
                3,
              ]}
            />
          </bufferGeometry>
          <lineBasicMaterial color="#80e8f7" transparent opacity={0.4} />
        </line>
      ))}
    </group>
  );
}

export function ParticleField() {
  return (
    <Canvas
      camera={{ position: [2.6, 1.6, 2.6], fov: 50 }}
      style={{ background: "transparent" }}
    >
      <ambientLight intensity={0.5} />
      <Stars
        radius={28}
        depth={60}
        count={1200}
        factor={2.2}
        saturation={0}
        fade
        speed={0.4}
      />
      <ParticleInstances />
      <ConnectionLines />
      <OrbitControls
        enablePan={false}
        autoRotate
        autoRotateSpeed={0.25}
        minDistance={1.8}
        maxDistance={7}
      />
    </Canvas>
  );
}
