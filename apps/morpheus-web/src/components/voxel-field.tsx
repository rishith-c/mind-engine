"use client";

import { useMemo, useRef } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import * as THREE from "three";

interface VoxelFieldProps {
  rgba: number[][][][] | null; // D x H x W x 4
  gridSize: number;
}

const tmpObj = new THREE.Object3D();
const tmpColor = new THREE.Color();

/**
 * Renders the (D x H x W x 4) voxel volume as instanced cubes. We skip cells
 * with alpha < 0.05 (architect's clip threshold) — at 32^3 dense that keeps
 * us well below the GPU instance budget.
 */
export function VoxelField({ rgba, gridSize }: VoxelFieldProps) {
  const meshRef = useRef<THREE.InstancedMesh>(null);

  const flat = useMemo(() => {
    if (!rgba) return null;
    const out: { x: number; y: number; z: number; r: number; g: number; b: number; a: number }[] =
      [];
    for (let x = 0; x < gridSize; x++) {
      for (let y = 0; y < gridSize; y++) {
        for (let z = 0; z < gridSize; z++) {
          const cell = rgba[x][y][z];
          if (cell[3] < 0.05) continue;
          out.push({ x, y, z, r: cell[0], g: cell[1], b: cell[2], a: cell[3] });
        }
      }
    }
    return out;
  }, [rgba, gridSize]);

  useFrame((state) => {
    if (!meshRef.current || !flat) return;
    const t = state.clock.elapsedTime;
    const half = gridSize / 2;
    for (let i = 0; i < flat.length; i++) {
      const c = flat[i];
      tmpObj.position.set(
        (c.x - half) / half,
        (c.y - half) / half,
        (c.z - half) / half,
      );
      const breathe = 1 + 0.06 * Math.sin(t * 2 + i * 0.01);
      tmpObj.scale.setScalar((1 / gridSize) * 1.1 * breathe);
      tmpObj.updateMatrix();
      meshRef.current.setMatrixAt(i, tmpObj.matrix);
      tmpColor.setRGB(c.r, c.g, c.b);
      meshRef.current.setColorAt(i, tmpColor);
    }
    meshRef.current.count = flat.length;
    meshRef.current.instanceMatrix.needsUpdate = true;
    if (meshRef.current.instanceColor) meshRef.current.instanceColor.needsUpdate = true;
  });

  return (
    <Canvas
      camera={{ position: [2.4, 1.6, 2.4], fov: 50 }}
      style={{ background: "radial-gradient(circle at 50% 30%, #082030 0%, #02070d 80%)" }}
    >
      <ambientLight intensity={0.4} />
      <directionalLight position={[3, 4, 2]} intensity={1.2} />
      <directionalLight position={[-3, -2, -2]} intensity={0.4} color="#80c8ff" />
      <instancedMesh ref={meshRef} args={[undefined, undefined, gridSize ** 3]}>
        <boxGeometry args={[1, 1, 1]} />
        <meshStandardMaterial roughness={0.45} metalness={0.1} toneMapped={false} />
      </instancedMesh>
      <OrbitControls
        enablePan={false}
        autoRotate
        autoRotateSpeed={0.6}
        minDistance={1.2}
        maxDistance={5}
      />
    </Canvas>
  );
}
