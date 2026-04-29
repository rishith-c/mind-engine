"use client";

import { useEffect, useState } from "react";
import { Brain, Zap, RotateCcw, Sparkles, Play } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { ParticleField } from "@/components/particle-field";
import { useCognitronStore } from "@/lib/store";

export default function CognitronPage() {
  const [thought, setThought] = useState("");
  const [query, setQuery] = useState("");
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

  return (
    <main className="relative h-screen w-screen overflow-hidden">
      <div className="absolute inset-0">
        <ParticleField />
      </div>

      {/* Header */}
      <div className="pointer-events-none absolute left-1/2 top-6 z-20 -translate-x-1/2 text-center">
        <h1 className="text-3xl font-bold tracking-tight glow-text flex items-center gap-3 justify-center">
          <Brain className="h-8 w-8 text-primary" />
          Cognitron
        </h1>
        <p className="text-xs text-muted-foreground mt-1">
          A particle neural network · 100% from-scratch · {field?.n ?? 0} thoughts alive
        </p>
      </div>

      {/* Left panel: capture thoughts */}
      <Card className="pointer-events-auto absolute left-6 top-24 z-10 w-80">
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-sm">
            <Sparkles className="h-4 w-4" /> Drop a thought
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-2">
          <Input
            placeholder="A thought, fact, idea..."
            value={thought}
            onChange={(e) => setThought(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && thought.trim() && !busy) {
                addThought(thought).then(() => setThought(""));
              }
            }}
          />
          <div className="flex gap-2">
            <Button
              className="flex-1"
              size="sm"
              onClick={() => addThought(thought).then(() => setThought(""))}
              disabled={busy || !thought.trim()}
            >
              Add particle
            </Button>
            <Button size="sm" variant="ghost" onClick={() => trainBurst(3)} disabled={busy}>
              <Play className="h-3 w-3" /> Train
            </Button>
            <Button size="sm" variant="ghost" onClick={reset} disabled={busy}>
              <RotateCcw className="h-3 w-3" />
            </Button>
          </div>
          {loss !== null && (
            <p className="text-[10px] font-mono text-muted-foreground">
              loss = {loss.toFixed(4)}
            </p>
          )}
        </CardContent>
      </Card>

      {/* Right panel: query */}
      <Card className="pointer-events-auto absolute right-6 top-24 z-10 w-96">
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-sm">
            <Zap className="h-4 w-4" /> Wave-propagation query
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-2">
          <Input
            placeholder="Ask something..."
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && query.trim() && !busy) ask(query);
            }}
          />
          <Button
            className="w-full"
            size="sm"
            onClick={() => ask(query)}
            disabled={busy || !query.trim()}
          >
            Fire neurons
          </Button>
          {hits.length > 0 && (
            <ul className="mt-2 space-y-1 text-xs">
              {hits.map((h) => (
                <li
                  key={h.id}
                  className="flex items-start gap-2 rounded border border-border/50 p-2"
                >
                  <span className="font-mono text-primary tabular-nums">
                    {h.score.toFixed(2)}
                  </span>
                  <span className="text-muted-foreground">{h.text}</span>
                </li>
              ))}
            </ul>
          )}
        </CardContent>
      </Card>

      {/* Footer credit */}
      <div className="pointer-events-none absolute bottom-4 left-1/2 -translate-x-1/2 text-center text-[10px] text-muted-foreground">
        Particles attract by semantic similarity · Wave queries propagate through the field · No
        pretrained model in sight
      </div>
    </main>
  );
}
