"use client";

import { useCallback, useRef, useState } from "react";
import {
    Card,
    CardContent,
    CardDescription,
    CardHeader,
    CardTitle,
} from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { api } from "@/lib/api";

interface ProductionPlan {
    title: string;
    genre: string;
    bpm: number;
    key: string;
    scale: string;
    energy: string;
    mood: string;
    structure: string[];
    synth_presets: Record<string, string>;
    effect_chains: Record<string, string>;
    description: string;
}

interface RenderResult {
    title: string;
    genre: string;
    bpm: number;
    key: string;
    duration_s: number;
    sections: number;
    tracks: string[];
    output: string;
}

type Stage = "idle" | "planning" | "planned" | "rendering" | "complete";

export default function CreatorPage() {
    const [description, setDescription] = useState("");
    const [stage, setStage] = useState<Stage>("idle");
    const [plan, setPlan] = useState<ProductionPlan | null>(null);
    const [result, setResult] = useState<RenderResult | null>(null);
    const [error, setError] = useState<string | null>(null);
    const [renderMode, setRenderMode] = useState<"ai" | "offline">("offline");

    const handlePlan = useCallback(async () => {
        if (!description.trim()) return;
        setStage("planning");
        setError(null);

        try {
            const res = await api<ProductionPlan>("/api/brain/plan", {
                method: "POST",
                body: JSON.stringify({ description, use_llm: true }),
            });
            setPlan(res);
            setStage("planned");
        } catch (err: unknown) {
            const message = err instanceof Error ? err.message : "Failed to generate plan";
            setError(message);
            setStage("idle");
        }
    }, [description]);

    const handleRender = useCallback(async () => {
        setStage("rendering");
        setError(null);

        try {
            const body = plan
                ? { genre: plan.genre, key: plan.key, scale: plan.scale, bpm: plan.bpm }
                : { genre: "house", key: "C", scale: "minor", bpm: 120 };

            const res = await api<RenderResult>("/api/brain/render", {
                method: "POST",
                body: JSON.stringify(body),
            });
            setResult(res);
            setStage("complete");
        } catch (err: unknown) {
            const message = err instanceof Error ? err.message : "Failed to render";
            setError(message);
            setStage("planned");
        }
    }, [plan]);

    const handleQuickRender = useCallback(async () => {
        setStage("rendering");
        setError(null);

        try {
            const res = await api<RenderResult>("/api/brain/render", {
                method: "POST",
                body: JSON.stringify({ genre: "house", key: "C", scale: "minor", bpm: 120 }),
            });
            setResult(res);
            setStage("complete");
        } catch (err: unknown) {
            const message = err instanceof Error ? err.message : "Failed to render";
            setError(message);
            setStage("idle");
        }
    }, []);

    const reset = () => {
        setStage("idle");
        setPlan(null);
        setResult(null);
        setError(null);
        setDescription("");
    };

    const stageColors: Record<Stage, string> = {
        idle: "bg-blue-500/20 text-blue-400",
        planning: "bg-yellow-500/20 text-yellow-400",
        planned: "bg-emerald-500/20 text-emerald-400",
        rendering: "bg-purple-500/20 text-purple-400",
        complete: "bg-green-500/20 text-green-400",
    };

    return (
        <div className="flex flex-col gap-6">
            <div>
                <h1 className="text-3xl font-bold tracking-tight">
                    🌟 Creator
                </h1>
                <p className="text-muted-foreground mt-1">
                    Describe your track → AURALIS produces everything
                </p>
            </div>

            {/* Stage Indicator */}
            <div className="flex items-center gap-4">
                {(["idle", "planning", "planned", "rendering", "complete"] as Stage[]).map((s, i) => (
                    <div key={s} className="flex items-center gap-2">
                        <div className={`w-3 h-3 rounded-full ${stage === s ? "bg-emerald-500 animate-pulse" :
                            (["idle", "planning", "planned", "rendering", "complete"].indexOf(stage) > i)
                                ? "bg-emerald-500" : "bg-zinc-700"
                            }`} />
                        <span className={`text-xs capitalize ${stage === s ? "text-white font-bold" : "text-zinc-500"}`}>
                            {s === "idle" ? "Describe" : s}
                        </span>
                    </div>
                ))}
            </div>

            {error && (
                <Card className="border-red-500/30">
                    <CardContent className="p-4">
                        <p className="text-red-400 text-sm">❌ {error}</p>
                    </CardContent>
                </Card>
            )}

            {/* Step 1: Describe */}
            {(stage === "idle" || stage === "planning") && (
                <Card>
                    <CardHeader>
                        <CardTitle>1. Describe Your Track</CardTitle>
                        <CardDescription>
                            Tell AURALIS what you want to create — genre, mood, energy, references
                        </CardDescription>
                    </CardHeader>
                    <CardContent className="space-y-4">
                        <textarea
                            className="w-full h-32 bg-zinc-900 border border-zinc-700 rounded-lg p-4 text-white placeholder:text-zinc-500 resize-none focus:border-emerald-500 focus:outline-none"
                            placeholder="e.g. A deep house track with warm pads, syncopated bassline, tribal percussion, and emotional breakdown. 122 BPM, F# minor. Inspired by Mono Aullador's Sierra Nevada vibes..."
                            value={description}
                            onChange={(e) => setDescription(e.target.value)}
                            disabled={stage === "planning"}
                        />
                        <div className="flex gap-3">
                            <Button
                                onClick={handlePlan}
                                disabled={!description.trim() || stage === "planning"}
                                className="bg-emerald-600 hover:bg-emerald-700"
                            >
                                {stage === "planning" ? "🧠 Thinking..." : "🧠 AI Plan"}
                            </Button>
                            <Button
                                onClick={handleQuickRender}
                                variant="outline"
                                disabled={stage === "planning"}
                            >
                                ⚡ Quick Render (offline)
                            </Button>
                        </div>
                    </CardContent>
                </Card>
            )}

            {/* Step 2: Plan Preview */}
            {plan && (stage === "planned" || stage === "rendering" || stage === "complete") && (
                <Card>
                    <CardHeader>
                        <CardTitle>2. Production Plan</CardTitle>
                        <CardDescription>AI-generated plan ready for rendering</CardDescription>
                    </CardHeader>
                    <CardContent className="space-y-4">
                        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                            <div className="bg-zinc-900 rounded-lg p-3">
                                <p className="text-xs text-zinc-500">Title</p>
                                <p className="font-bold text-white">{plan.title}</p>
                            </div>
                            <div className="bg-zinc-900 rounded-lg p-3">
                                <p className="text-xs text-zinc-500">Genre</p>
                                <p className="font-bold text-emerald-400">{plan.genre}</p>
                            </div>
                            <div className="bg-zinc-900 rounded-lg p-3">
                                <p className="text-xs text-zinc-500">BPM / Key</p>
                                <p className="font-bold text-white">{plan.bpm} / {plan.key} {plan.scale}</p>
                            </div>
                            <div className="bg-zinc-900 rounded-lg p-3">
                                <p className="text-xs text-zinc-500">Energy</p>
                                <p className="font-bold text-amber-400">{plan.energy}</p>
                            </div>
                        </div>

                        <div>
                            <p className="text-xs text-zinc-500 mb-2">Structure</p>
                            <div className="flex flex-wrap gap-1">
                                {plan.structure.map((s, i) => (
                                    <Badge key={i} variant="secondary" className="text-xs">{s}</Badge>
                                ))}
                            </div>
                        </div>

                        <div>
                            <p className="text-xs text-zinc-500 mb-1">Mood</p>
                            <p className="text-sm text-zinc-300">{plan.mood}</p>
                        </div>

                        {stage === "planned" && (
                            <Button onClick={handleRender} className="bg-purple-600 hover:bg-purple-700">
                                🎹 Render Track
                            </Button>
                        )}
                        {stage === "rendering" && (
                            <div className="flex items-center gap-3">
                                <div className="w-4 h-4 border-2 border-purple-500 border-t-transparent rounded-full animate-spin" />
                                <span className="text-purple-400">Rendering... Grid → Hands → Mix</span>
                            </div>
                        )}
                    </CardContent>
                </Card>
            )}

            {/* Step 3: Result */}
            {result && stage === "complete" && (
                <Card className="border-emerald-500/30">
                    <CardHeader>
                        <CardTitle>3. ✅ Track Complete</CardTitle>
                        <CardDescription>Your track has been rendered</CardDescription>
                    </CardHeader>
                    <CardContent className="space-y-4">
                        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                            <div className="bg-zinc-900 rounded-lg p-3">
                                <p className="text-xs text-zinc-500">Title</p>
                                <p className="font-bold text-white">{result.title}</p>
                            </div>
                            <div className="bg-zinc-900 rounded-lg p-3">
                                <p className="text-xs text-zinc-500">Genre</p>
                                <p className="font-bold text-emerald-400">{result.genre}</p>
                            </div>
                            <div className="bg-zinc-900 rounded-lg p-3">
                                <p className="text-xs text-zinc-500">Duration</p>
                                <p className="font-bold text-white">{Math.floor(result.duration_s / 60)}:{String(Math.floor(result.duration_s % 60)).padStart(2, "0")}</p>
                            </div>
                            <div className="bg-zinc-900 rounded-lg p-3">
                                <p className="text-xs text-zinc-500">Tracks</p>
                                <p className="font-bold text-white">{result.tracks.length}</p>
                            </div>
                        </div>

                        <div>
                            <p className="text-xs text-zinc-500 mb-2">Stems</p>
                            <div className="flex flex-wrap gap-1">
                                {result.tracks.map((t) => (
                                    <Badge key={t} className="bg-emerald-500/20 text-emerald-400">{t}</Badge>
                                ))}
                            </div>
                        </div>

                        <div className="flex gap-3">
                            <Button onClick={reset} variant="outline">
                                🔄 Create Another
                            </Button>
                        </div>
                    </CardContent>
                </Card>
            )}
        </div>
    );
}
