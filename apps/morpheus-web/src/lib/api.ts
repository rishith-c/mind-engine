import { API_BASE } from "./utils";

export interface Frame {
  t: number;
  grid_size: number;
  rgba: number[][][][]; // D x H x W x 4
}

export interface AudioFrame {
  t: number;
  grid_size: number;
  frequencies: number[][][]; // D x H x W
}

async function jfetch<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(API_BASE + path, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!res.ok) throw new Error(`${path} -> ${res.status} ${await res.text()}`);
  return res.json() as Promise<T>;
}

export const morpheusApi = {
  health: () => jfetch<{ ok: boolean; t: number }>("/health"),
  seed: (target: string, steps: number) =>
    jfetch<Frame>("/seed", { method: "POST", body: JSON.stringify({ target, steps }) }),
  frame: (t: number) => jfetch<Frame>(`/frame/${t}`),
  audio: (t: number) => jfetch<AudioFrame>(`/audio/${t}`),
};
