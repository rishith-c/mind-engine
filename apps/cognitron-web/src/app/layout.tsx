import type { Metadata } from "next";
import { Instrument_Serif, JetBrains_Mono, Geist } from "next/font/google";
import "./globals.css";

// Editorial scientific instrument typography:
//  - Instrument Serif italic for display (the hero wordmark)
//  - JetBrains Mono for all stats, labels, technical text
//  - Geist Sans for body — a clean, characterful neo-grotesque
const display = Instrument_Serif({
  weight: ["400"],
  style: ["italic"],
  subsets: ["latin"],
  variable: "--font-display",
  display: "swap",
});

const mono = JetBrains_Mono({
  subsets: ["latin"],
  variable: "--font-mono",
  display: "swap",
});

const sans = Geist({
  subsets: ["latin"],
  variable: "--font-sans",
  display: "swap",
});

export const metadata: Metadata = {
  title: "Cognitron — Particle Neural Network",
  description:
    "A neural substrate built from particles. No transformers. No pretraining. From-scratch hyperdimensional computing + wave-propagation inference.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html
      lang="en"
      className={`${display.variable} ${mono.variable} ${sans.variable} dark`}
      suppressHydrationWarning
    >
      <body
        className="font-sans"
        style={{ fontFamily: "var(--font-sans)" }}
        suppressHydrationWarning
      >
        {children}
      </body>
    </html>
  );
}
