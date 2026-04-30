import type { NextConfig } from "next";

const config: NextConfig = {
  reactStrictMode: true,
  transpilePackages: ["@mind/physics", "three"],
  experimental: {
    optimizePackageImports: ["lucide-react"],
  },
};

export default config;
