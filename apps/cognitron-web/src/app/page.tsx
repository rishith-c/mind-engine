"use client";

import { useEffect, useState } from "react";
import dynamic from "next/dynamic";
import { motion, AnimatePresence } from "framer-motion";
import { ArrowRight, Plus, Sparkles, RotateCcw } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Separator } from "@/components/ui/separator";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";

import { useCognitronStore } from "@/lib/store";

const ParticleField = dynamic(
  () => import("@/components/particle-field").then((m) => m.ParticleField),
  { ssr: false, loading: () => <div className="h-full w-full" /> },
);

const fadeUp = {
  initial: { opacity: 0, y: 8 },
  animate: { opacity: 1, y: 0 },
  transition: { duration: 0.7, ease: [0.22, 1, 0.36, 1] as const },
};

const list = {
  hidden: {},
  show: { transition: { staggerChildren: 0.04, delayChildren: 0.06 } },
};

const listRow = {
  hidden: { opacity: 0, x: -4 },
  show: {
    opacity: 1,
    x: 0,
    transition: { duration: 0.35, ease: [0.22, 1, 0.36, 1] as const },
  },
};

export default function CognitronPage() {
  const [thought, setThought] = useState<string>("");
  const [queryText, setQueryText] = useState<string>("");

  const field = useCognitronStore((s) => s.field);
  const hits = useCognitronStore((s) => s.hits);
  const busy = useCognitronStore((s) => s.busy);
  const loss = useCognitronStore((s) => s.loss);
  const refresh = useCognitronStore((s) => s.refresh);
  const addThought = useCognitronStore((s) => s.addThought);
  const ask = useCognitronStore((s) => s.query);
  const trainBurst = useCognitronStore((s) => s.trainBurst);
  const reset = useCognitronStore((s) => s.reset);

  useEffect(() => {
    refresh();
    const t = setInterval(refresh, 2500);
    return () => clearInterval(t);
  }, [refresh]);

  const handleAdd = async (): Promise<void> => {
    if (!thought.trim() || busy) return;
    await addThought(thought);
    setThought("");
  };

  const handleAsk = async (): Promise<void> => {
    if (!queryText.trim() || busy) return;
    await ask(queryText);
  };

  const particleCount = field?.n ?? 0;
  const lossDisplay = loss !== null ? loss.toFixed(4) : "—";

  return (
    <TooltipProvider delayDuration={250}>
      <main className="relative h-screen w-screen overflow-hidden bg-background">
        {/* Background layers — graph paper grid + the particle field */}
        <div className="graph-grid pointer-events-none absolute inset-0 z-0" />
        <div className="accent-glow pointer-events-none absolute inset-0 z-0" />
        <div className="absolute inset-0 z-0">
          <ParticleField />
        </div>

        {/* Top + bottom vignettes for legibility */}
        <div
          aria-hidden
          className="pointer-events-none absolute inset-x-0 top-0 z-10 h-44 bg-gradient-to-b from-background via-background/40 to-transparent"
        />
        <div
          aria-hidden
          className="pointer-events-none absolute inset-x-0 bottom-0 z-10 h-32 bg-gradient-to-t from-background to-transparent"
        />

        {/* ─────────────────── Top bar — instrument header ─────────────────── */}
        <motion.header
          {...fadeUp}
          className="pointer-events-none absolute inset-x-0 top-0 z-20 flex items-center justify-between px-8 py-6"
        >
          <div className="flex items-center gap-3">
            <span className="inline-flex h-1.5 w-1.5 rounded-full bg-accent shadow-[0_0_8px_hsl(var(--accent))]" />
            <span className="mono micro tabular text-foreground/70">
              cognitron / pnn-001
            </span>
          </div>
          <div className="flex items-center gap-6 mono text-[11px] text-foreground/60 tabular">
            <span>
              <span className="micro mr-2">particles</span>
              {String(particleCount).padStart(4, "0")}
            </span>
            <span className="hidden md:inline">
              <span className="micro mr-2">loss</span>
              {lossDisplay}
            </span>
            <span className="hidden md:inline">
              <span className="micro mr-2">freq</span>
              2.5 hz
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
            particle neural network · live
            <span className="h-px w-8 bg-foreground/20" />
          </div>

          <h1 className="display text-7xl leading-none tracking-tight md:text-8xl">
            Cognitron
          </h1>

          <p className="mx-auto mt-6 max-w-md text-balance text-[13px] leading-relaxed text-foreground/65">
            A neural substrate of free particles in continuous space. Topology
            emerges from semantic gravity. Inference is a wave.
          </p>
        </motion.section>

        {/* ─────────────────── Lower-left: capture (instrument panel) ─────────────────── */}
        <motion.aside
          {...fadeUp}
          transition={{ ...fadeUp.transition, delay: 0.2 }}
          className="pointer-events-auto absolute bottom-10 left-8 z-20 w-[20rem]"
        >
          <div className="surface rounded-md">
            <div className="flex items-center justify-between border-b border-border/60 px-4 py-3">
              <div className="flex items-center gap-2">
                <Sparkles className="h-3 w-3 text-foreground/60" />
                <span className="mono micro text-foreground/75">
                  capture · 01
                </span>
              </div>
              <span className="mono micro text-foreground/40">input</span>
            </div>

            <div className="space-y-3 p-4">
              <Input
                placeholder="A thought..."
                value={thought}
                onChange={(e) => setThought(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter") void handleAdd();
                }}
                disabled={busy}
                className="mono h-9 rounded-sm border-border/60 bg-transparent text-[13px] placeholder:text-foreground/30 focus-visible:ring-1 focus-visible:ring-accent/60"
              />

              <div className="grid grid-cols-3 gap-2">
                <Button
                  size="sm"
                  onClick={handleAdd}
                  disabled={busy || !thought.trim()}
                  className="mono col-span-1 h-8 rounded-sm bg-foreground text-background text-[11px] tracking-wider hover:bg-foreground/90"
                >
                  <Plus className="h-3 w-3" /> add
                </Button>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <Button
                      size="sm"
                      variant="outline"
                      onClick={() => trainBurst(3)}
                      disabled={busy}
                      className="mono h-8 rounded-sm border-border/60 bg-transparent text-[11px] tracking-wider text-foreground/70 hover:bg-foreground/5"
                    >
                      train ×3
                    </Button>
                  </TooltipTrigger>
                  <TooltipContent
                    side="top"
                    className="mono text-[10px] tracking-wider"
                  >
                    Run 3 PGD epochs
                  </TooltipContent>
                </Tooltip>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <Button
                      size="sm"
                      variant="outline"
                      onClick={reset}
                      disabled={busy}
                      className="mono h-8 rounded-sm border-border/60 bg-transparent text-[11px] tracking-wider text-foreground/70 hover:bg-foreground/5"
                    >
                      <RotateCcw className="h-3 w-3" />
                    </Button>
                  </TooltipTrigger>
                  <TooltipContent side="top" className="mono text-[10px]">
                    Clear field
                  </TooltipContent>
                </Tooltip>
              </div>
            </div>

            <Separator className="bg-border/40" />

            <div className="grid grid-cols-2 divide-x divide-border/40">
              <Stat label="particles" value={particleCount} loading={!field} />
              <Stat label="last loss" value={lossDisplay} loading={false} mono />
            </div>
          </div>
        </motion.aside>

        {/* ─────────────────── Lower-right: query (instrument panel) ─────────────────── */}
        <motion.aside
          {...fadeUp}
          transition={{ ...fadeUp.transition, delay: 0.3 }}
          className="pointer-events-auto absolute bottom-10 right-8 z-20 w-[24rem]"
        >
          <div className="surface rounded-md">
            <div className="flex items-center justify-between border-b border-border/60 px-4 py-3">
              <div className="flex items-center gap-2">
                <ArrowRight className="h-3 w-3 text-foreground/60" />
                <span className="mono micro text-foreground/75">
                  query · wave-propagation
                </span>
              </div>
              <span className="mono micro text-foreground/40">k=5</span>
            </div>

            <div className="space-y-3 p-4">
              <div className="flex gap-2">
                <Input
                  placeholder="ask the field..."
                  value={queryText}
                  onChange={(e) => setQueryText(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter") void handleAsk();
                  }}
                  disabled={busy}
                  className="mono h-9 rounded-sm border-border/60 bg-transparent text-[13px] placeholder:text-foreground/30 focus-visible:ring-1 focus-visible:ring-accent/60"
                />
                <Button
                  size="sm"
                  onClick={handleAsk}
                  disabled={busy || !queryText.trim()}
                  className="mono h-9 rounded-sm bg-accent px-4 text-[11px] tracking-wider text-accent-foreground hover:bg-accent/90"
                >
                  fire
                </Button>
              </div>

              <ScrollArea className="h-[14rem]">
                <AnimatePresence mode="popLayout">
                  {hits.length === 0 ? (
                    <motion.div
                      key="empty"
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      exit={{ opacity: 0 }}
                      className="flex h-[12rem] flex-col items-center justify-center gap-2 text-center"
                    >
                      <span className="mono micro text-foreground/40">
                        — awaiting query —
                      </span>
                      <span className="text-[11px] text-foreground/40">
                        a wave seeded at the most-similar particle propagates
                        for 5 hops; resonance ranks by absorbed energy.
                      </span>
                    </motion.div>
                  ) : (
                    <motion.ol
                      key="hits"
                      variants={list}
                      initial="hidden"
                      animate="show"
                      className="space-y-1"
                    >
                      {hits.map((h, i) => (
                        <motion.li
                          key={h.id}
                          variants={listRow}
                          layout
                          className="group flex items-baseline gap-3 border-b border-border/30 py-2 last:border-b-0"
                        >
                          <span className="mono w-5 text-[10px] text-foreground/40 tabular">
                            {String(i + 1).padStart(2, "0")}
                          </span>
                          <span className="flex-1 text-[12px] leading-snug text-foreground/85">
                            {h.text}
                          </span>
                          <span className="mono w-12 shrink-0 text-right text-[11px] text-accent tabular">
                            {h.score.toFixed(3)}
                          </span>
                        </motion.li>
                      ))}
                    </motion.ol>
                  )}
                </AnimatePresence>
              </ScrollArea>
            </div>
          </div>
        </motion.aside>

        {/* ─────────────────── Footer caption ─────────────────── */}
        <motion.footer
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.6, duration: 0.8 }}
          className="pointer-events-none absolute inset-x-0 bottom-3 z-20 text-center"
        >
          <p className="mono micro text-foreground/35">
            no transformers · no pretraining · 100% from-scratch hyperdimensional computing
          </p>
        </motion.footer>
      </main>
    </TooltipProvider>
  );
}

// ─────────────────────────────────────────────────────────────────────────────

interface StatProps {
  label: string;
  value: number | string;
  loading: boolean;
  mono?: boolean;
}

function Stat({ label, value, loading, mono = false }: StatProps) {
  return (
    <div className="px-4 py-3">
      <div className="mono micro mb-1 text-foreground/45">{label}</div>
      {loading ? (
        <Skeleton className="h-4 w-12 rounded-none bg-foreground/10" />
      ) : (
        <div
          className={`tabular text-[15px] text-foreground/90 ${
            mono ? "mono" : "display text-[20px] leading-none"
          }`}
        >
          {typeof value === "number" ? value.toLocaleString() : value}
        </div>
      )}
    </div>
  );
}
