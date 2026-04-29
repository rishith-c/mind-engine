import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Morpheus — Synesthetic Cellular Reality",
  description:
    "From-scratch 3D Neural Cellular Automaton with per-cell audio synesthesia. No pretrained models.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className="dark">
      <body>{children}</body>
    </html>
  );
}
