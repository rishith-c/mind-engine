"use client";

import { useState } from "react";
import dynamic from "next/dynamic";
import { motion } from "framer-motion";
import { Boxes, Music, Loader2, Play } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import { Slider } from "@/components/ui/slider";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";

import { morpheusApi, Frame } from "@/lib/api";
import { playSynesthesia } from "@/lib/synesthesia";

const VoxelField = dynamic(
  () => import("@/components/voxel-field").then((m) => m.VoxelField),
  { ssr: false, loading: () => <div className="h-full w-full" /> },
);

const fadeUp = {
  initial: { opacity: 0, y: 8 },
  animate: { opacity: 1, y: 0 },
  transition: { duration: 0.7, ease: [0.22, 1, 0.36, 1] as const },
};

const TARGETS = ["sphere", "helix"] as const;
type Target = (typeof TARGETS)[number];

export default function MorpheusPage() {
  const [target, setTarget] = useState<Target>("sphere");
  const [steps, setSteps] = useState<number>(64);
  const [frame, setFrame] = useState<Frame | null>(null);
  const [audioFrame, setAudioFrame] = useState<number[][][] | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const grow = async (): Promise<void> => {
    setLoading(true);
    setError(null);
    try {
      const f = await morpheusApi.seed(target, steps);
      setFrame(f);
      const a = await morpheusApi.audio(f.t);
      setAudioFrame(a.frequencies);
    } catch (err) {
      setError(err instanceof Error ? err.message : "growth failed");
    } finally {
      setLoading(false);
    }
  };

  const playChord = async (): Promise<void> => {
    if (!audioFrame || !frame) return;
    await playSynesthesia(audioFrame, frame.rgba);
  };

  const aliveCells = frame
    ? frame.rgba.flat(2).filter((c: number[]) => c[3] >= 0.08).length
    : 0;

  return (
    <TooltipProvider delayDuration={250}>
      <main className="relative h-screen w-screen overflow-hidden bg-background">
        {/* Background */}
        <div className="graph-grid pointer-events-none absolute inset-0 z-0" />
        <div className="accent-glow pointer-events-none absolute inset-0 z-0" />

        {/* 3D voxel field */}
        <div className="absolute inset-0 z-0">
          {frame ? (
            <VoxelField rgba={frame.rgba} gridSize={frame.grid_size} />
          ) : (
            <div className="flex h-full w-full items-center justify-center">
              <div className="text-center">
                <Boxes className="mx-auto h-12 w-12 text-foreground/15" />
                <p className="mono micro mt-4 text-foreground/40">
                  no reality grown — seed one
                </p>
              </div>
            </div>
          )}
        </div>

        {/* Vignettes */}
        <div
          aria-hidden
          className="pointer-events-none absolute inset-x-0 top-0 z-10 h-44 bg-gradient-to-b from-background via-background/40 to-transparent"
        />
        <div
          aria-hidden
          className="pointer-events-none absolute inset-x-0 bottom-0 z-10 h-32 bg-gradient-to-t from-background to-transparent"
        />

        {/* ─────────────────── Top instrument bar ─────────────────── */}
        <motion.header
          {...fadeUp}
          className="pointer-events-none absolute inset-x-0 top-0 z-20 flex items-center justify-between px-8 py-6"
        >
          <div className="flex items-center gap-3">
            <span className="inline-flex h-1.5 w-1.5 rounded-full bg-accent shadow-[0_0_8px_hsl(var(--accent))]" />
            <span className="mono micro tabular text-foreground/70">
              morpheus / nca-001
            </span>
          </div>
          <div className="flex items-center gap-6 mono text-[11px] text-foreground/60 tabular">
            <span>
              <span className="micro mr-2">step</span>
              {String(frame?.t ?? 0).padStart(3, "0")}
            </span>
            <span className="hidden md:inline">
              <span className="micro mr-2">grid</span>
              {frame?.grid_size ?? 32}³
            </span>
            <span className="hidden md:inline">
              <span className="micro mr-2">alive</span>
              {String(aliveCells).padStart(5, "0")}
            </span>
          </div>
        </motion.header>

        {/* ─────────────────── Centered editorial hero ─────────────────── */}
        <motion.section
          {...fadeUp}
          transition={{ ...fadeUp.transition, delay: 0.1 }}
          className="pointer-events-none absolute left-1/2 top-[14%] z-20 -translate-x-1/2 text-center"
        >
          <div className="mono micro mb-4 flex items-center justify-center gap-2 text-foreground/55">
            <span className="h-px w-8 bg-foreground/20" />
            synesthetic 3d cellular automaton
            <span className="h-px w-8 bg-foreground/20" />
          </div>

          <h1 className="display text-7xl leading-none tracking-tight md:text-8xl">
            Morpheus
          </h1>

          <p className="mx-auto mt-6 max-w-md text-balance text-[13px] leading-relaxed text-foreground/65">
            One seed cell. Local rules. Geometry, color, and sound grow
            jointly into a living three-dimensional reality.
          </p>
        </motion.section>

        {/* ─────────────────── Lower-left: control instrument ─────────────────── */}
        <motion.aside
          {...fadeUp}
          transition={{ ...fadeUp.transition, delay: 0.2 }}
          className="pointer-events-auto absolute bottom-10 left-8 z-20 w-[20rem]"
        >
          <div className="surface rounded-md">
            <div className="flex items-center justify-between border-b border-border/60 px-4 py-3">
              <div className="flex items-center gap-2">
                <Boxes className="h-3 w-3 text-foreground/60" />
                <span className="mono micro text-foreground/75">
                  seed · target
                </span>
              </div>
              <span className="mono micro text-foreground/40">3d nca</span>
            </div>

            <div className="space-y-4 p-4">
              <div className="grid grid-cols-2 gap-2">
                {TARGETS.map((t) => (
                  <Button
                    key={t}
                    size="sm"
                    onClick={() => setTarget(t)}
                    className={`mono h-8 rounded-sm text-[11px] tracking-wider ${
                      target === t
                        ? "bg-foreground text-background hover:bg-foreground/90"
                        : "border border-border/60 bg-transparent text-foreground/70 hover:bg-foreground/5"
                    }`}
                  >
                    {t}
                  </Button>
                ))}
              </div>

              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <span className="mono micro text-foreground/60">
                    growth steps
                  </span>
                  <span className="mono text-[11px] tabular text-foreground/85">
                    {steps}
                  </span>
                </div>
                <Slider
                  value={[steps]}
                  min={16}
                  max={128}
                  step={1}
                  onValueChange={(v) => setSteps(v[0] ?? 64)}
                  className="[&_[role=slider]]:h-3 [&_[role=slider]]:w-3 [&_[role=slider]]:border-foreground"
                />
              </div>

              <Button
                onClick={grow}
                disabled={loading}
                className="mono h-9 w-full rounded-sm bg-accent text-[11px] tracking-wider text-accent-foreground hover:bg-accent/90"
              >
                {loading ? (
                  <Loader2 className="h-3 w-3 animate-spin" />
                ) : (
                  <Play className="h-3 w-3" />
                )}
                grow reality
              </Button>

              <Tooltip>
                <TooltipTrigger asChild>
                  <Button
                    variant="outline"
                    onClick={playChord}
                    disabled={!audioFrame || loading}
                    className="mono h-9 w-full rounded-sm border-border/60 bg-transparent text-[11px] tracking-wider text-foreground/80 hover:bg-foreground/5"
                  >
                    <Music className="h-3 w-3" /> hear it · synesthesia
                  </Button>
                </TooltipTrigger>
                <TooltipContent side="top" className="mono text-[10px]">
                  Plays a chord built from the most-active cells&apos;
                  per-cell frequencies
                </TooltipContent>
              </Tooltip>
            </div>

            <Separator className="bg-border/40" />

            <div className="grid grid-cols-3 divide-x divide-border/40">
              <Stat label="alive" value={aliveCells} loading={loading} />
              <Stat label="step" value={frame?.t ?? 0} loading={loading} />
              <Stat
                label="grid"
                value={frame ? `${frame.grid_size}³` : "—"}
                loading={false}
              />
            </div>
          </div>

          {error && (
            <div className="mono mt-3 rounded-sm border border-destructive/40 bg-destructive/10 px-3 py-2 text-[11px] text-destructive-foreground">
              {error}
            </div>
          )}
        </motion.aside>

        {/* ─────────────────── Lower-right: legend ─────────────────── */}
        <motion.aside
          {...fadeUp}
          transition={{ ...fadeUp.transition, delay: 0.3 }}
          className="pointer-events-auto absolute bottom-10 right-8 z-20 w-[22rem]"
        >
          <div className="surface rounded-md">
            <div className="flex items-center justify-between border-b border-border/60 px-4 py-3">
              <span className="mono micro text-foreground/75">
                how it grows
              </span>
              <span className="mono micro text-foreground/40">
                3 modalities
              </span>
            </div>
            <div className="space-y-3 p-4">
              <LegendRow
                index="01"
                label="geometry"
                desc="Per-cell alpha learned via sobel-3d perception + a tiny per-cell MLP. Stochastic update mask."
              />
              <LegendRow
                index="02"
                label="color"
                desc="RGB grown from the same shared latent. Hue is differentiable; gradients flow through neighbours."
              />
              <LegendRow
                index="03"
                label="audio"
                desc="A jointly-trained per-cell frequency channel. The chord is the field's voice — the only synesthetic NCA."
              />
            </div>
          </div>
        </motion.aside>

        <motion.footer
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.6, duration: 0.8 }}
          className="pointer-events-none absolute inset-x-0 bottom-3 z-20 text-center"
        >
          <p className="mono micro text-foreground/35">
            no pretrained models · cells learn rules from one seed · 100%
            from-scratch
          </p>
        </motion.footer>
      </main>
    </TooltipProvider>
  );
}

interface StatProps {
  label: string;
  value: number | string;
  loading: boolean;
}

function Stat({ label, value, loading }: StatProps) {
  return (
    <div className="px-4 py-3">
      <div className="mono micro mb-1 text-foreground/45">{label}</div>
      {loading ? (
        <Skeleton className="h-4 w-12 rounded-none bg-foreground/10" />
      ) : (
        <div className="display text-[18px] leading-none text-foreground/90 tabular">
          {typeof value === "number" ? value.toLocaleString() : value}
        </div>
      )}
    </div>
  );
}

interface LegendRowProps {
  index: string;
  label: string;
  desc: string;
}

function LegendRow({ index, label, desc }: LegendRowProps) {
  return (
    <div className="flex items-baseline gap-3 border-b border-border/30 pb-3 last:border-b-0 last:pb-0">
      <span className="mono w-6 text-[10px] text-foreground/40 tabular">
        {index}
      </span>
      <div className="flex-1">
        <div className="mono micro text-foreground/85">{label}</div>
        <div className="mt-1 text-[11px] leading-snug text-foreground/55">
          {desc}
        </div>
      </div>
    </div>
  );
}
