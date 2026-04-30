import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Cognitron — A Living Particle Brain",
  description:
    "From-scratch Particle Neural Network with Hyperdimensional Computing. No pretrained models.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    // suppressHydrationWarning on <html> + <body> tolerates attributes
    // injected by browser extensions (Grammarly, Kapture, dark-mode hacks,
    // etc.) which otherwise cause a hydration mismatch on first paint.
    <html lang="en" className="dark" suppressHydrationWarning>
      <body suppressHydrationWarning>{children}</body>
    </html>
  );
}
