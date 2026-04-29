/**
 * Thin typed client for the Cognitron Python API.
 */

import { API_BASE } from "./utils";

export interface Particle {
  id: number;
  text: string;
  position: [number, number, number];
  mass: number;
  polarity: number;
}

export interface QueryHit {
  id: number;
  text: string;
  score: number;
}

export interface FieldSnapshot {
  n: number;
  positions: number[][];
  masses: number[];
  polarities: number[];
  ids: number[];
  texts: string[];
}

async function jfetch<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(API_BASE + path, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!res.ok) {
    throw new Error(`${path} failed: ${res.status} ${await res.text()}`);
  }
  return res.json() as Promise<T>;
}

export const api = {
  health: () => jfetch<{ ok: boolean; particles: number }>("/health"),
  addThought: (text: string, mass = 1.0, polarity = 1) =>
    jfetch<Particle>("/thought", {
      method: "POST",
      body: JSON.stringify({ text, mass, polarity }),
    }),
  query: (q: string, k = 5) =>
    jfetch<{ query: string; hits: QueryHit[] }>(
      `/query?q=${encodeURIComponent(q)}&k=${k}`,
    ),
  state: () => jfetch<FieldSnapshot>("/state"),
  trainStep: (epochs = 1) =>
    jfetch<{ losses: number[] }>(`/train/step?epochs=${epochs}`, { method: "POST" }),
  reset: () => jfetch<{ ok: boolean }>("/reset", { method: "DELETE" }),
};
