"use client";

import { useEffect, useState } from "react";
import dynamic from "next/dynamic";
import { motion, AnimatePresence } from "framer-motion";
import {
  Brain,
  Zap,
  Sparkles,
  Play,
  RotateCcw,
  Activity,
} from "lucide-react";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";

import { useCognitronStore } from "@/lib/store";

// React Three Fiber + drei must be client-only. SSR will hit
// "ReactCurrentOwner of undefined" because R3F reads React internals.
const ParticleField = dynamic(
  () => import("@/components/particle-field").then((m) => m.ParticleField),
  { ssr: false, loading: () => <div className="h-full w-full" /> },
);

// Motion presets — keep everything physical, never bouncy.
const fadeUp = {
  initial: { opacity: 0, y: 16, filter: "blur(8px)" },
  animate: { opacity: 1, y: 0, filter: "blur(0px)" },
  transition: { duration: 0.7, ease: [0.22, 1, 0.36, 1] as const },
};

const listContainer = {
  hidden: {},
  show: {
    transition: { staggerChildren: 0.06, delayChildren: 0.04 },
  },
};

const listItem = {
  hidden: { opacity: 0, y: 8, filter: "blur(4px)" },
  show: {
    opacity: 1,
    y: 0,
    filter: "blur(0px)",
    transition: { duration: 0.4, ease: [0.22, 1, 0.36, 1] as const },
  },
};

const buttonPress = {
  whileHover: { scale: 1.02 },
  whileTap: { scale: 0.97 },
  transition: { type: "spring" as const, stiffness: 400, damping: 28 },
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
        {/* Aurora ambient background — sits beneath the particle galaxy */}
        <div className="aurora pointer-events-none absolute inset-0 z-0 opacity-60" />

        {/* 3D particle galaxy */}
        <div className="absolute inset-0 z-0">
          <ParticleField />
        </div>

        {/* Top vignette so the hero text reads cleanly over the galaxy */}
        <div
          aria-hidden
          className="pointer-events-none absolute inset-x-0 top-0 z-10 h-64 bg-gradient-to-b from-background/70 via-background/20 to-transparent"
        />

        {/* Bottom vignette for the caption */}
        <div
          aria-hidden
          className="pointer-events-none absolute inset-x-0 bottom-0 z-10 h-40 bg-gradient-to-t from-background/80 via-background/20 to-transparent"
        />

        {/* ───────────────────────── HERO ───────────────────────── */}
        <motion.header
          {...fadeUp}
          className="pointer-events-none absolute left-1/2 top-10 z-20 -translate-x-1/2 text-center"
        >
          <div className="mb-3 flex items-center justify-center gap-2">
            <Badge
              variant="outline"
              className="glass border-white/10 px-3 py-1 text-[10px] font-medium uppercase tracking-[0.18em] text-muted-foreground"
            >
              <Sparkles className="mr-1.5 h-3 w-3 text-accent" />
              Particle Neural Network · live
            </Badge>
          </div>

          <h1 className="glow-text flex items-center justify-center gap-3 text-5xl font-semibold tracking-tight md:text-6xl">
            <Brain className="h-10 w-10 text-primary drop-shadow-[0_0_20px_hsl(var(--primary)/0.6)]" />
            <span className="bg-gradient-to-br from-white via-white to-white/60 bg-clip-text text-transparent">
              Cognitron
            </span>
          </h1>

          <motion.p
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.25, duration: 0.6, ease: [0.22, 1, 0.36, 1] }}
            className="mt-3 text-sm text-muted-foreground/90"
          >
            A neural substrate built from particles. No transformers. No pretraining.
            Just attraction and resonance.
          </motion.p>
        </motion.header>

        {/* ─────────────────── LEFT: DROP A THOUGHT ─────────────────── */}
        <motion.div
          {...fadeUp}
          transition={{ ...fadeUp.transition, delay: 0.15 }}
          className="pointer-events-auto absolute left-6 top-44 z-20 w-[22rem]"
        >
          <Card className="glass gradient-border overflow-hidden border-white/5 bg-transparent shadow-2xl">
            <CardHeader className="pb-3">
              <div className="flex items-center justify-between">
                <CardTitle className="flex items-center gap-2 text-sm font-medium text-foreground/90">
                  <Sparkles className="h-4 w-4 text-accent" />
                  Drop a Thought
                </CardTitle>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <Badge
                      variant="secondary"
                      className="cursor-default border-white/5 bg-white/5 text-[10px] font-mono tabular-nums text-muted-foreground"
                    >
                      <Activity className="mr-1 h-2.5 w-2.5 text-accent" />
                      live
                    </Badge>
                  </TooltipTrigger>
                  <TooltipContent side="left" className="text-xs">
                    Field auto-refreshes every 2.5s
                  </TooltipContent>
                </Tooltip>
              </div>
            </CardHeader>

            <CardContent className="space-y-4">
              <div className="space-y-2">
                <Input
                  placeholder="A thought, fact, or idea..."
                  value={thought}
                  onChange={(e) => setThought(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter") void handleAdd();
                  }}
                  disabled={busy}
                  className="border-white/10 bg-white/[0.04] placeholder:text-muted-foreground/60 focus-visible:ring-primary/40"
                />

                <div className="flex gap-2">
                  <motion.div className="flex-1" {...buttonPress}>
                    <Button
                      onClick={() => void handleAdd()}
                      disabled={busy || !thought.trim()}
                      className="w-full bg-primary text-primary-foreground shadow-[0_0_24px_-8px_hsl(var(--primary)/0.8)] hover:bg-primary/90"
                      size="sm"
                    >
                      <Sparkles className="mr-1.5 h-3.5 w-3.5" />
                      Add particle
                    </Button>
                  </motion.div>

                  <Tooltip>
                    <TooltipTrigger asChild>
                      <motion.div {...buttonPress}>
                        <Button
                          onClick={() => void trainBurst(3)}
                          disabled={busy}
                          variant="ghost"
                          size="sm"
                          className="border border-white/10 bg-white/[0.03] hover:bg-white/[0.08]"
                        >
                          <Play className="h-3.5 w-3.5 text-accent" />
                        </Button>
                      </motion.div>
                    </TooltipTrigger>
                    <TooltipContent side="bottom" className="text-xs">
                      Run 3 training epochs
                    </TooltipContent>
                  </Tooltip>

                  <Tooltip>
                    <TooltipTrigger asChild>
                      <motion.div {...buttonPress}>
                        <Button
                          onClick={() => void reset()}
                          disabled={busy}
                          variant="ghost"
                          size="sm"
                          className="border border-white/10 bg-white/[0.03] text-muted-foreground hover:bg-destructive/20 hover:text-destructive-foreground"
                        >
                          <RotateCcw className="h-3.5 w-3.5" />
                        </Button>
                      </motion.div>
                    </TooltipTrigger>
                    <TooltipContent side="bottom" className="text-xs">
                      Reset the field
                    </TooltipContent>
                  </Tooltip>
                </div>
              </div>

              <Separator className="bg-white/5" />

              {/* Stat row — tabular numerics */}
              <div className="grid grid-cols-2 gap-3">
                <Stat
                  label="Particles"
                  value={
                    field === null ? (
                      <Skeleton className="h-5 w-10 bg-white/10" />
                    ) : (
                      <span className="text-base font-semibold tabular-nums text-foreground">
                        {particleCount.toLocaleString()}
                      </span>
                    )
                  }
                />
                <Stat
                  label="Last loss"
                  value={
                    <span
                      className={`text-base font-semibold tabular-nums ${
                        loss === null ? "text-muted-foreground/50" : "text-accent"
                      }`}
                    >
                      {lossDisplay}
                    </span>
                  }
                />
              </div>
            </CardContent>
          </Card>
        </motion.div>

        {/* ─────────────────── RIGHT: WAVE QUERY ─────────────────── */}
        <motion.div
          {...fadeUp}
          transition={{ ...fadeUp.transition, delay: 0.25 }}
          className="pointer-events-auto absolute right-6 top-44 z-20 w-[24rem]"
        >
          <Card className="glass gradient-border overflow-hidden border-white/5 bg-transparent shadow-2xl">
            <CardHeader className="pb-3">
              <CardTitle className="flex items-center gap-2 text-sm font-medium text-foreground/90">
                <Zap className="h-4 w-4 text-primary" />
                Wave-Propagation Query
              </CardTitle>
            </CardHeader>

            <CardContent className="space-y-3">
              <div className="flex gap-2">
                <Input
                  placeholder="Ask the field something..."
                  value={queryText}
                  onChange={(e) => setQueryText(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter") void handleAsk();
                  }}
                  disabled={busy}
                  className="border-white/10 bg-white/[0.04] placeholder:text-muted-foreground/60 focus-visible:ring-accent/40"
                />
                <motion.div {...buttonPress}>
                  <Button
                    onClick={() => void handleAsk()}
                    disabled={busy || !queryText.trim()}
                    size="sm"
                    className="bg-accent text-accent-foreground shadow-[0_0_24px_-8px_hsl(var(--accent)/0.8)] hover:bg-accent/90"
                  >
                    <Zap className="mr-1.5 h-3.5 w-3.5" />
                    Fire
                  </Button>
                </motion.div>
              </div>

              <Separator className="bg-white/5" />

              {/* Hit list */}
              <div className="min-h-[12rem]">
                {busy && hits.length === 0 ? (
                  <div className="space-y-2">
                    {[0, 1, 2].map((i) => (
                      <Skeleton key={i} className="h-12 w-full bg-white/[0.04]" />
                    ))}
                  </div>
                ) : hits.length === 0 ? (
                  <EmptyHits />
                ) : (
                  <ScrollArea className="h-[18rem] pr-3">
                    <motion.ol
                      key={hits.map((h) => h.id).join(",")}
                      variants={listContainer}
                      initial="hidden"
                      animate="show"
                      className="space-y-2"
                    >
                      <AnimatePresence initial={false}>
                        {hits.map((h, idx) => (
                          <motion.li
                            key={`${h.id}-${idx}`}
                            variants={listItem}
                            layout
                            whileHover={{
                              y: -1,
                              transition: { duration: 0.15 },
                            }}
                            className="group relative flex items-start gap-3 rounded-lg border border-white/5 bg-white/[0.03] p-3 transition-colors hover:border-white/10 hover:bg-white/[0.06]"
                          >
                            <div className="flex h-6 w-6 shrink-0 items-center justify-center rounded-md bg-white/[0.04] text-[10px] font-mono tabular-nums text-muted-foreground">
                              {idx + 1}
                            </div>

                            <div className="min-w-0 flex-1">
                              <p className="truncate text-xs leading-relaxed text-foreground/90">
                                {h.text}
                              </p>
                              <p className="mt-1 text-[10px] text-muted-foreground/70">
                                particle #{h.id}
                              </p>
                            </div>

                            <Badge
                              variant="outline"
                              className="shrink-0 border-primary/30 bg-primary/10 font-mono text-[10px] tabular-nums text-primary"
                            >
                              {h.score.toFixed(3)}
                            </Badge>
                          </motion.li>
                        ))}
                      </AnimatePresence>
                    </motion.ol>
                  </ScrollArea>
                )}
              </div>
            </CardContent>
          </Card>
        </motion.div>

        {/* ───────────────────── FOOTER CAPTION ───────────────────── */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.6, duration: 0.8 }}
          className="pointer-events-none absolute bottom-6 left-1/2 z-20 -translate-x-1/2 text-center"
        >
          <p className="mx-auto max-w-2xl text-[11px] leading-relaxed text-muted-foreground/70">
            Particles attract by semantic similarity. Wave queries propagate through
            the field, lighting up resonant memory.
            <span className="mx-2 text-muted-foreground/30">·</span>
            <span className="font-mono tabular-nums">no pretrained model in sight</span>
          </p>
        </motion.div>
      </main>
    </TooltipProvider>
  );
}

/* ─────────────────────── helpers ─────────────────────── */

interface StatProps {
  label: string;
  value: React.ReactNode;
}

function Stat({ label, value }: StatProps) {
  return (
    <div className="rounded-lg border border-white/5 bg-white/[0.02] px-3 py-2 transition-colors hover:bg-white/[0.04]">
      <div className="text-[10px] font-medium uppercase tracking-wider text-muted-foreground/70">
        {label}
      </div>
      <div className="mt-0.5">{value}</div>
    </div>
  );
}

function EmptyHits() {
  return (
    <div className="flex h-[12rem] flex-col items-center justify-center gap-2 rounded-lg border border-dashed border-white/5 bg-white/[0.01] text-center">
      <Zap className="h-5 w-5 text-muted-foreground/40" />
      <p className="text-xs text-muted-foreground/60">
        Fire a query to ripple through the field
      </p>
      <p className="text-[10px] text-muted-foreground/40">
        Top-5 resonant particles will surface here
      </p>
    </div>
  );
}
