"use client";

import { useState } from "react";
import { Sparkles, Music, Loader2, Boxes } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { VoxelField } from "@/components/voxel-field";
import { morpheusApi, Frame } from "@/lib/api";
import { playSynesthesia } from "@/lib/synesthesia";

const TARGETS = ["sphere", "helix"] as const;
type Target = (typeof TARGETS)[number];

export default function MorpheusPage() {
  const [target, setTarget] = useState<Target>("sphere");
  const [steps, setSteps] = useState(64);
  const [frame, setFrame] = useState<Frame | null>(null);
  const [loading, setLoading] = useState(false);
  const [audioFrame, setAudioFrame] = useState<number[][][] | null>(null);

  async function grow() {
    setLoading(true);
    try {
      const f = await morpheusApi.seed(target, steps);
      setFrame(f);
      const audio = await morpheusApi.audio(f.t);
      setAudioFrame(audio.frequencies);
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  }

  async function playChord() {
    if (!audioFrame || !frame) return;
    await playSynesthesia(audioFrame, frame.rgba);
  }

  return (
    <main className="relative h-screen w-screen overflow-hidden">
      <div className="absolute inset-0">
        {frame ? (
          <VoxelField rgba={frame.rgba} gridSize={frame.grid_size} />
        ) : (
          <div className="flex h-full w-full items-center justify-center text-muted-foreground">
            <div className="text-center">
              <Boxes className="mx-auto h-16 w-16 opacity-30" />
              <p className="mt-4 text-sm">No reality grown yet — seed one to begin.</p>
            </div>
          </div>
        )}
      </div>

      {/* Header */}
      <div className="pointer-events-none absolute left-1/2 top-6 z-20 -translate-x-1/2 text-center">
        <h1 className="text-3xl font-bold tracking-tight glow-text flex items-center gap-3 justify-center">
          <Sparkles className="h-8 w-8 text-primary" />
          Morpheus
        </h1>
        <p className="text-xs text-muted-foreground mt-1">
          Synesthetic 3D neural cellular automaton · 100% from-scratch · grown {frame?.t ?? 0} steps
        </p>
      </div>

      {/* Control panel */}
      <Card className="pointer-events-auto absolute left-6 top-24 z-10 w-80">
        <CardHeader>
          <CardTitle className="text-sm">Seed a reality</CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          <div className="flex gap-2">
            {TARGETS.map((t) => (
              <Button
                key={t}
                size="sm"
                variant={t === target ? "default" : "outline"}
                onClick={() => setTarget(t)}
              >
                {t}
              </Button>
            ))}
          </div>
          <div>
            <label className="text-xs text-muted-foreground">growth steps</label>
            <input
              type="range"
              min={16}
              max={128}
              value={steps}
              onChange={(e) => setSteps(parseInt(e.target.value))}
              className="w-full"
            />
            <div className="text-xs font-mono">{steps}</div>
          </div>
          <Button className="w-full" onClick={grow} disabled={loading}>
            {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : null}
            Grow
          </Button>
          <Button
            variant="outline"
            className="w-full"
            onClick={playChord}
            disabled={!audioFrame || loading}
          >
            <Music className="h-4 w-4" /> Hear it (synesthesia)
          </Button>
        </CardContent>
      </Card>

      {/* Footer caption */}
      <div className="pointer-events-none absolute bottom-4 left-1/2 -translate-x-1/2 text-center text-[10px] text-muted-foreground">
        Cells grow geometry, color, and audio jointly · per-cell frequency emits a chord based on
        cell activity
      </div>
    </main>
  );
}
