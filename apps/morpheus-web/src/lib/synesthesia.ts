"use client";

import * as Tone from "tone";

let synth: Tone.PolySynth | null = null;

function ensureSynth(): Tone.PolySynth {
  if (!synth) {
    synth = new Tone.PolySynth(Tone.Synth, {
      oscillator: { type: "sine" },
      envelope: { attack: 0.06, decay: 0.3, sustain: 0.4, release: 1.6 },
    }).toDestination();
    synth.volume.value = -10;
  }
  return synth;
}

export async function playSynesthesia(
  frequencies: number[][][],
  rgba: number[][][][],
): Promise<void> {
  await Tone.start();
  const s = ensureSynth();

  const candidates: { f: number; alpha: number }[] = [];
  const D = frequencies.length;
  for (let x = 0; x < D; x++) {
    for (let y = 0; y < D; y++) {
      for (let z = 0; z < D; z++) {
        const a = rgba[x]?.[y]?.[z]?.[3] ?? 0;
        if (a < 0.3) continue;
        candidates.push({ f: frequencies[x][y][z], alpha: a });
      }
    }
  }
  candidates.sort((a, b) => b.alpha - a.alpha);
  const top = candidates.slice(0, 4).map((c) => c.f);
  if (top.length === 0) return;

  s.triggerAttackRelease(top, "1n");
}
