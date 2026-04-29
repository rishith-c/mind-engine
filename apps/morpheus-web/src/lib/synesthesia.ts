"use client";

import * as Tone from "tone";

/**
 * Synesthetic audio: take the per-cell frequency field from the model and
 * emit a chord built from the most-active cells. The "audio modality" is
 * the only fresh angle in 3D NCA literature per the novelty researcher;
 * this is where we showcase it.
 */

let synth: Tone.PolySynth | null = null;

function ensureSynth(): Tone.PolySynth {
  if (!synth) {
    synth = new Tone.PolySynth(Tone.Synth, {
      oscillator: { type: "sine" },
      envelope: { attack: 0.05, decay: 0.2, sustain: 0.4, release: 1.5 },
    }).toDestination();
    synth.volume.value = -10;
  }
  return synth;
}

export async function playSynesthesia(frequencies: number[][][], rgba: number[][][][]) {
  await Tone.start();
  const s = ensureSynth();

  // Find the top-N most-alive cells, take their frequencies as a chord.
  const candidates: { f: number; alpha: number }[] = [];
  const D = frequencies.length;
  for (let x = 0; x < D; x++) {
    for (let y = 0; y < D; y++) {
      for (let z = 0; z < D; z++) {
        const a = rgba[x][y][z][3];
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
