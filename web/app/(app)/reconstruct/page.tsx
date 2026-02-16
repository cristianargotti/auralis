"use client";

import { useState, useEffect, useCallback } from "react";
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

/* ── Types ──────────────────────────────────────────── */

interface Section {
    name: string;
    start_bar: number;
    end_bar: number;
    bars: number;
    rms_db: number;
    stereo_sm: number;
    elements: string[];
    description: string;
}

interface StageStatus {
    status: "pending" | "running" | "completed" | "error";
    message: string;
}

interface ReconstructJob {
    job_id: string;
    status: "running" | "completed" | "error";
    stage: string;
    progress: number;
    stages: Record<string, StageStatus>;
    result: Record<string, unknown> | null;
}

interface Blueprint {
    profile: {
        title: string;
        artist: string;
        bpm: number;
        key: string;
        scale: string;
        total_bars: number;
        duration_s: number;
    };
    sections: Section[];
    energy_map: { bar: number; rms_db: number; phase: string }[];
    quality_targets: Record<string, unknown>;
}

/* ── Stage Definitions ──────────────────────────────── */

const STAGES = [
    { key: "load", label: "EAR", icon: "👂", desc: "Load reference analysis" },
    { key: "plan", label: "BRAIN", icon: "🧠", desc: "Generate arrangement plan" },
    { key: "grid", label: "GRID", icon: "📐", desc: "Compose MIDI patterns" },
    { key: "hands", label: "HANDS", icon: "🎹", desc: "Synthesize audio" },
    { key: "console", label: "CONSOLE", icon: "🎚️", desc: "Mix & master" },
    { key: "qc", label: "QC", icon: "🔍", desc: "A/B comparison" },
];

/* ── Component ──────────────────────────────────────── */

export default function ReconstructPage() {
    const [blueprint, setBlueprint] = useState<Blueprint | null>(null);
    const [sections, setSections] = useState<Section[]>([]);
    const [job, setJob] = useState<ReconstructJob | null>(null);
    const [loading, setLoading] = useState(true);
    const [reconstructing, setReconstructing] = useState(false);
    const [selectedSection, setSelectedSection] = useState<string | null>(null);

    // Load blueprint data
    useEffect(() => {
        const loadData = async () => {
            try {
                const [bp, secs] = await Promise.all([
                    api<Blueprint>("/api/reconstruct/blueprint"),
                    api<Section[]>("/api/reconstruct/sections"),
                ]);
                setBlueprint(bp);
                setSections(secs);
            } catch {
                // Offline — use static data
            } finally {
                setLoading(false);
            }
        };
        loadData();
    }, []);

    // Poll job status
    useEffect(() => {
        if (!job || job.status !== "running") return;
        const interval = setInterval(async () => {
            try {
                const updated = await api<ReconstructJob>(
                    `/api/reconstruct/status/${job.job_id}`
                );
                setJob(updated);
                if (updated.status !== "running") {
                    setReconstructing(false);
                }
            } catch {
                /* ignore */
            }
        }, 1000);
        return () => clearInterval(interval);
    }, [job]);

    const startReconstruction = useCallback(async () => {
        setReconstructing(true);
        try {
            const result = await api<ReconstructJob>("/api/reconstruct/start", {
                method: "POST",
                body: JSON.stringify({
                    project_id: "million-pieces-benchmark",
                    mode: "full",
                }),
            });
            setJob(result as unknown as ReconstructJob);
        } catch {
            setReconstructing(false);
        }
    }, []);

    const getStageColor = (status: string) => {
        switch (status) {
            case "completed": return "bg-emerald-500/20 text-emerald-400 border-emerald-500/30";
            case "running": return "bg-amber-500/20 text-amber-400 border-amber-500/30 animate-pulse";
            case "error": return "bg-red-500/20 text-red-400 border-red-500/30";
            default: return "bg-zinc-800/50 text-zinc-500 border-zinc-700/30";
        }
    };

    const getEnergyColor = (rms: number) => {
        if (rms > -9) return "bg-red-500";
        if (rms > -12) return "bg-orange-500";
        if (rms > -15) return "bg-amber-500";
        if (rms > -18) return "bg-emerald-500";
        return "bg-cyan-500";
    };

    const getEnergyHeight = (rms: number) => {
        const normalized = Math.max(0, Math.min(1, (rms + 25) / 20));
        return `${Math.max(8, normalized * 100)}%`;
    };

    if (loading) {
        return (
            <div className="flex items-center justify-center min-h-screen">
                <div className="text-center space-y-4">
                    <div className="text-4xl animate-pulse">🔬</div>
                    <p className="text-zinc-400">Loading reconstruction blueprint...</p>
                </div>
            </div>
        );
    }

    return (
        <div className="space-y-6 p-6">
            {/* Header */}
            <div className="flex items-center justify-between">
                <div>
                    <h1 className="text-3xl font-bold bg-gradient-to-r from-amber-400 via-orange-500 to-red-500 bg-clip-text text-transparent">
                        🔬 Reconstruct
                    </h1>
                    <p className="text-zinc-400 mt-1">
                        Bar-by-bar track reconstruction — the ultimate production benchmark
                    </p>
                </div>
                <Button
                    onClick={startReconstruction}
                    disabled={reconstructing}
                    className="bg-gradient-to-r from-amber-600 to-red-600 hover:from-amber-500 hover:to-red-500 text-white font-semibold px-6 py-3"
                    size="lg"
                >
                    {reconstructing ? "⚡ Reconstructing..." : "🚀 Start Reconstruction"}
                </Button>
            </div>

            {/* Blueprint Info */}
            {blueprint && (
                <Card className="bg-zinc-900/50 border-zinc-800">
                    <CardHeader>
                        <CardTitle className="text-lg">
                            🎵 {blueprint.profile.title}
                        </CardTitle>
                        <CardDescription>{blueprint.profile.artist}</CardDescription>
                    </CardHeader>
                    <CardContent>
                        <div className="flex gap-3 flex-wrap">
                            <Badge variant="outline" className="border-amber-500/30 text-amber-400">
                                {blueprint.profile.bpm} BPM
                            </Badge>
                            <Badge variant="outline" className="border-emerald-500/30 text-emerald-400">
                                Key: {blueprint.profile.key} {blueprint.profile.scale}
                            </Badge>
                            <Badge variant="outline" className="border-cyan-500/30 text-cyan-400">
                                {blueprint.profile.total_bars} bars
                            </Badge>
                            <Badge variant="outline" className="border-purple-500/30 text-purple-400">
                                {Math.floor(blueprint.profile.duration_s / 60)}:{String(Math.floor(blueprint.profile.duration_s % 60)).padStart(2, "0")}
                            </Badge>
                            <Badge variant="outline" className="border-zinc-500/30 text-zinc-400">
                                {sections.length} sections
                            </Badge>
                        </div>
                    </CardContent>
                </Card>
            )}

            {/* Pipeline Stages */}
            {job && (
                <Card className="bg-zinc-900/50 border-zinc-800">
                    <CardHeader>
                        <div className="flex items-center justify-between">
                            <CardTitle className="text-lg">⚡ Pipeline</CardTitle>
                            <Badge
                                variant="outline"
                                className={job.status === "completed"
                                    ? "border-emerald-500/30 text-emerald-400"
                                    : "border-amber-500/30 text-amber-400"
                                }
                            >
                                {job.progress}%
                            </Badge>
                        </div>
                    </CardHeader>
                    <CardContent>
                        {/* Progress bar */}
                        <div className="w-full h-2 bg-zinc-800 rounded-full mb-6 overflow-hidden">
                            <div
                                className="h-full bg-gradient-to-r from-amber-500 to-red-500 rounded-full transition-all duration-500"
                                style={{ width: `${job.progress}%` }}
                            />
                        </div>

                        {/* Stage cards */}
                        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3">
                            {STAGES.map((stage) => {
                                const status = job.stages[stage.key];
                                return (
                                    <div
                                        key={stage.key}
                                        className={`rounded-lg border p-3 text-center transition-all ${getStageColor(status?.status || "pending")}`}
                                    >
                                        <div className="text-2xl mb-1">{stage.icon}</div>
                                        <div className="text-xs font-bold">{stage.label}</div>
                                        <div className="text-[10px] mt-1 opacity-70">
                                            {status?.status === "completed" ? "✓" : status?.status === "running" ? "●" : "○"}
                                        </div>
                                    </div>
                                );
                            })}
                        </div>

                        {/* Current stage message */}
                        {job.status === "running" && (
                            <div className="mt-4 text-sm text-zinc-400 text-center animate-pulse">
                                {job.stages[job.stage]?.message || "Processing..."}
                            </div>
                        )}

                        {/* Result summary */}
                        {job.status === "completed" && job.result && (
                            <div className="mt-4 p-4 rounded-lg bg-emerald-500/10 border border-emerald-500/20">
                                <p className="text-emerald-400 font-semibold text-center">
                                    ✅ Reconstruction complete — {(job.result as Record<string, unknown>).rendered_sections as number} sections rendered
                                </p>
                            </div>
                        )}
                    </CardContent>
                </Card>
            )}

            {/* Energy Map Visualization */}
            {blueprint && (
                <Card className="bg-zinc-900/50 border-zinc-800">
                    <CardHeader>
                        <CardTitle className="text-lg">📊 Energy Map</CardTitle>
                        <CardDescription>
                            RMS energy per 8-bar block — click to inspect
                        </CardDescription>
                    </CardHeader>
                    <CardContent>
                        <div className="flex items-end gap-1 h-32">
                            {blueprint.energy_map.map((point, i) => (
                                <div
                                    key={i}
                                    className="flex-1 flex flex-col items-center gap-1 cursor-pointer group"
                                    title={`Bar ${point.bar}: ${point.rms_db.toFixed(1)} dB — ${point.phase}`}
                                >
                                    <div
                                        className={`w-full rounded-t transition-all group-hover:opacity-80 ${getEnergyColor(point.rms_db)}`}
                                        style={{ height: getEnergyHeight(point.rms_db) }}
                                    />
                                    <span className="text-[8px] text-zinc-600 group-hover:text-zinc-400">
                                        {point.bar}
                                    </span>
                                </div>
                            ))}
                        </div>
                        <div className="flex justify-between mt-2 text-[10px] text-zinc-600">
                            <span>0:00</span>
                            <span>Intro</span>
                            <span>Groove</span>
                            <span>Breakdown</span>
                            <span>Drop</span>
                            <span>Groove</span>
                            <span>Outro</span>
                            <span>5:32</span>
                        </div>
                    </CardContent>
                </Card>
            )}

            {/* Section Detail Grid */}
            <Card className="bg-zinc-900/50 border-zinc-800">
                <CardHeader>
                    <CardTitle className="text-lg">🎼 Sections</CardTitle>
                    <CardDescription>
                        {sections.length} sections — bar-by-bar arrangement blueprint
                    </CardDescription>
                </CardHeader>
                <CardContent>
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
                        {sections.map((section) => (
                            <div
                                key={section.name}
                                className={`rounded-lg border p-4 cursor-pointer transition-all ${selectedSection === section.name
                                        ? "border-amber-500/50 bg-amber-500/10"
                                        : "border-zinc-800 bg-zinc-900/30 hover:border-zinc-700"
                                    }`}
                                onClick={() =>
                                    setSelectedSection(
                                        selectedSection === section.name ? null : section.name
                                    )
                                }
                            >
                                <div className="flex items-center justify-between mb-2">
                                    <span className="font-semibold text-sm capitalize">
                                        {section.name.replace(/_/g, " ")}
                                    </span>
                                    <Badge variant="outline" className="text-[10px] border-zinc-700">
                                        {section.start_bar}–{section.end_bar}
                                    </Badge>
                                </div>
                                <div className="flex gap-2 mb-2">
                                    <span className="text-[10px] text-amber-400">
                                        {section.rms_db.toFixed(1)} dB
                                    </span>
                                    <span className="text-[10px] text-cyan-400">
                                        S/M: {section.stereo_sm.toFixed(3)}
                                    </span>
                                </div>
                                <div className="flex flex-wrap gap-1">
                                    {section.elements.map((el) => (
                                        <Badge
                                            key={el}
                                            variant="secondary"
                                            className="text-[9px] bg-zinc-800 text-zinc-400"
                                        >
                                            {el}
                                        </Badge>
                                    ))}
                                </div>
                                {selectedSection === section.name && (
                                    <p className="text-xs text-zinc-500 mt-3 leading-relaxed">
                                        {section.description}
                                    </p>
                                )}
                            </div>
                        ))}
                    </div>
                </CardContent>
            </Card>

            {/* 6-Layer Architecture Footer */}
            <div className="grid grid-cols-6 gap-2 text-center">
                {[
                    { icon: "👂", label: "EAR", color: "text-cyan-400" },
                    { icon: "🎹", label: "HANDS", color: "text-purple-400" },
                    { icon: "🎚️", label: "CONSOLE", color: "text-amber-400" },
                    { icon: "📐", label: "GRID", color: "text-emerald-400" },
                    { icon: "🧠", label: "BRAIN", color: "text-pink-400" },
                    { icon: "🔍", label: "QC", color: "text-orange-400" },
                ].map((layer) => (
                    <div
                        key={layer.label}
                        className="rounded-lg bg-zinc-900/30 border border-zinc-800/50 p-2"
                    >
                        <div className="text-lg">{layer.icon}</div>
                        <div className={`text-[10px] font-bold ${layer.color}`}>
                            {layer.label}
                        </div>
                    </div>
                ))}
            </div>
        </div>
    );
}
