"use client";

import { create } from "zustand";
import { api, FieldSnapshot, Particle, QueryHit } from "./api";

interface CognitronState {
  field: FieldSnapshot | null;
  highlighted: Set<number>;
  hits: QueryHit[];
  busy: boolean;
  loss: number | null;
  refresh(): Promise<void>;
  addThought(text: string): Promise<Particle | null>;
  query(q: string): Promise<void>;
  trainBurst(epochs: number): Promise<void>;
  reset(): Promise<void>;
  setHighlighted(ids: number[]): void;
}

export const useCognitronStore = create<CognitronState>((set, get) => ({
  field: null,
  highlighted: new Set(),
  hits: [],
  busy: false,
  loss: null,

  async refresh() {
    try {
      const f = await api.state();
      set({ field: f });
    } catch (err) {
      console.error("refresh failed", err);
    }
  },

  async addThought(text: string) {
    if (!text.trim()) return null;
    set({ busy: true });
    try {
      const p = await api.addThought(text);
      await get().refresh();
      // Briefly highlight the newly-added particle
      set({ highlighted: new Set([p.id]) });
      setTimeout(() => set({ highlighted: new Set() }), 1500);
      return p;
    } finally {
      set({ busy: false });
    }
  },

  async query(q: string) {
    if (!q.trim()) return;
    set({ busy: true });
    try {
      const { hits } = await api.query(q, 5);
      set({ hits, highlighted: new Set(hits.map((h) => h.id)) });
    } finally {
      set({ busy: false });
    }
  },

  async trainBurst(epochs: number) {
    set({ busy: true });
    try {
      const { losses } = await api.trainStep(epochs);
      set({ loss: losses[losses.length - 1] ?? null });
      await get().refresh();
    } finally {
      set({ busy: false });
    }
  },

  async reset() {
    await api.reset();
    set({ hits: [], highlighted: new Set(), loss: null });
    await get().refresh();
  },

  setHighlighted(ids: number[]) {
    set({ highlighted: new Set(ids) });
  },
}));
