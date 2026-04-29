import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Cognitron — A Living Particle Brain",
  description:
    "From-scratch Particle Neural Network with Hyperdimensional Computing. No pretrained models.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className="dark">
      <body>{children}</body>
    </html>
  );
}
