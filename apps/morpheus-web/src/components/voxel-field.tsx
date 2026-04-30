"use client";

import { useMemo, useRef } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import { EffectComposer, Bloom } from "@react-three/postprocessing";
import * as THREE from "three";

interface VoxelFieldProps {
  rgba: number[][][][] | null; // D x H x W x 4
  gridSize: number;
}

interface LiveVoxel {
  x: number;
  y: number;
  z: number;
  r: number;
  g: number;
  b: number;
  a: number;
}

const tmpObj = new THREE.Object3D();
const tmpColor = new THREE.Color();

function VoxelInstances({ rgba, gridSize }: VoxelFieldProps) {
  const meshRef = useRef<THREE.InstancedMesh>(null);
  const haloRef = useRef<THREE.InstancedMesh>(null);

  const live: LiveVoxel[] = useMemo(() => {
    if (!rgba) return [];
    const out: LiveVoxel[] = [];
    for (let x = 0; x < gridSize; x++) {
      for (let y = 0; y < gridSize; y++) {
        for (let z = 0; z < gridSize; z++) {
          const cell = rgba[x][y][z];
          if (cell[3] < 0.08) continue;
          out.push({ x, y, z, r: cell[0], g: cell[1], b: cell[2], a: cell[3] });
        }
      }
    }
    return out;
  }, [rgba, gridSize]);

  useFrame((state) => {
    if (!meshRef.current || !haloRef.current || live.length === 0) return;
    const t = state.clock.elapsedTime;
    const half = gridSize / 2;
    const cellScale = 1 / gridSize;
    for (let i = 0; i < live.length; i++) {
      const c = live[i];
      const px = (c.x - half) / half;
      const py = (c.y - half) / half;
      const pz = (c.z - half) / half;
      const breathe = 1 + 0.05 * Math.sin(t * 1.6 + i * 0.13);

      // Solid core
      tmpObj.position.set(px, py, pz);
      tmpObj.scale.setScalar(cellScale * 1.05 * breathe);
      tmpObj.rotation.set(0, 0, 0);
      tmpObj.updateMatrix();
      meshRef.current.setMatrixAt(i, tmpObj.matrix);
      tmpColor.setRGB(c.r * c.a, c.g * c.a, c.b * c.a);
      meshRef.current.setColorAt(i, tmpColor);

      // Glow halo (additive)
      tmpObj.scale.setScalar(cellScale * 1.85 * breathe);
      tmpObj.updateMatrix();
      haloRef.current.setMatrixAt(i, tmpObj.matrix);
      tmpColor.setRGB(c.r * 0.6, c.g * 0.6, c.b * 0.6);
      haloRef.current.setColorAt(i, tmpColor);
    }
    meshRef.current.count = live.length;
    haloRef.current.count = live.length;
    meshRef.current.instanceMatrix.needsUpdate = true;
    haloRef.current.instanceMatrix.needsUpdate = true;
    if (meshRef.current.instanceColor) {
      meshRef.current.instanceColor.needsUpdate = true;
    }
    if (haloRef.current.instanceColor) {
      haloRef.current.instanceColor.needsUpdate = true;
    }
  });

  if (live.length === 0) return null;

  const max = Math.max(gridSize ** 3, 1);

  return (
    <group>
      <instancedMesh ref={haloRef} args={[undefined, undefined, max]}>
        <boxGeometry args={[1, 1, 1]} />
        <meshBasicMaterial
          transparent
          opacity={0.32}
          blending={THREE.AdditiveBlending}
          depthWrite={false}
          toneMapped={false}
        />
      </instancedMesh>
      <instancedMesh ref={meshRef} args={[undefined, undefined, max]}>
        <boxGeometry args={[1, 1, 1]} />
        <meshStandardMaterial
          roughness={0.35}
          metalness={0.05}
          toneMapped={false}
          emissiveIntensity={0.6}
        />
      </instancedMesh>
    </group>
  );
}

export function VoxelField({ rgba, gridSize }: VoxelFieldProps) {
  return (
    <Canvas
      camera={{ position: [2.4, 1.6, 2.4], fov: 48 }}
      style={{ background: "transparent" }}
      gl={{ antialias: true, alpha: true }}
    >
      <ambientLight intensity={0.35} />
      <directionalLight position={[3, 4, 2]} intensity={1.0} />
      <directionalLight position={[-3, -2, -2]} intensity={0.4} color="#ffd9a8" />
      <VoxelInstances rgba={rgba} gridSize={gridSize} />
      <OrbitControls
        enablePan={false}
        autoRotate
        autoRotateSpeed={0.3}
        minDistance={1.6}
        maxDistance={6}
      />
      <EffectComposer>
        <Bloom
          intensity={0.7}
          luminanceThreshold={0.15}
          luminanceSmoothing={0.4}
          mipmapBlur
        />
      </EffectComposer>
    </Canvas>
  );
}
