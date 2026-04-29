import type { NextConfig } from "next";

const config: NextConfig = {
  reactStrictMode: true,
  transpilePackages: ["@mind/physics", "three"],
  webpack: (config) => {
    config.module.rules.push({ test: /\.wgsl$/, type: "asset/source" });
    return config;
  },
};

export default config;
