import type { NextConfig } from "next";

// Cognitron-web doesn't directly import .wgsl (those live in shared-physics
// and are only used by morpheus-web). Keeping config lean fixes the
// Turbopack/Webpack mismatch warning.
const config: NextConfig = {
  reactStrictMode: true,
  transpilePackages: ["@mind/physics", "three"],
  experimental: {
    optimizePackageImports: ["lucide-react"],
  },
};

export default config;
