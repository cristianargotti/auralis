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

/* ── Types (track-agnostic) ──────────────────────────── */

interface StageStatus {
    status: "pending" | "running" | "completed" | "error";
    message: string;
}

interface ReconstructJob {
    job_id: string;
    project_id: string;
    status: "running" | "completed" | "error";
    stage: string;
    progress: number;
    stages: Record<string, StageStatus>;
    result: {
        analysis?: {
            bpm: number;
            key: string;
            scale: string;
            duration: number;
            sections_detected: number;
        };
        midi_tracks?: Record<string, { notes: number; pitch_range: number[]; confidence: number }>;
        rendered_sections?: number;
        master?: { target_lufs: number };
        qc?: {
            dimensions: Record<string, number>;
            overall_score: number;
            target_score: number;
        };
    } | null;
}

interface AnalysisData {
    tempo: number;
    key: string;
    scale: string;
    duration: number;
    integrated_lufs: number;
    true_peak_dbfs: number;
    dynamic_range_db: number;
    band_energy_profile: Record<string, number>;
    sections: {
        name: string;
        start_time: number;
        end_time: number;
        start_bar: number;
        end_bar: number;
        avg_rms_db: number;
        element_count: number;
        characteristics: string[];
    }[];
    energy_curve: { bar: number; rms_db: number }[];
}

/* ── Pipeline stages ─────────────────────────────────── */

const STAGES = [
    { key: "ear", label: "EAR", icon: "👂", desc: "Separate + profile" },
    { key: "plan", label: "BRAIN", icon: "🧠", desc: "Auto-detect structure" },
    { key: "grid", label: "GRID", icon: "📐", desc: "Map MIDI patterns" },
    { key: "hands", label: "HANDS", icon: "🎹", desc: "Synthesize audio" },
    { key: "console", label: "CONSOLE", icon: "🎚️", desc: "Mix & master" },
    { key: "qc", label: "QC", icon: "🔍", desc: "12D comparison" },
];

const QC_DIMENSIONS = [
    "spectral_similarity", "rms_match", "stereo_width_match",
    "bass_pattern_match", "kick_pattern_match", "harmonic_progression",
    "energy_curve", "reverb_match", "dynamic_range",
    "bpm_accuracy", "arrangement_match", "timbre_similarity",
];

/* ── Component ─────────────────────────────────────── */

export default function ReconstructPage() {
    const [projectId, setProjectId] = useState("");
    const [analysis, setAnalysis] = useState<AnalysisData | null>(null);
    const [job, setJob] = useState<ReconstructJob | null>(null);
    const [loading, setLoading] = useState(false);
    const [reconstructing, setReconstructing] = useState(false);
    const [selectedSection, setSelectedSection] = useState<string | null>(null);

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
                    // Load analysis after completion
                    if (updated.result?.analysis) {
                        loadAnalysis(updated.project_id);
                    }
                }
            } catch { /* ignore */ }
        }, 800);
        return () => clearInterval(interval);
    }, [job]);

    const loadAnalysis = async (pid: string) => {
        try {
            const data = await api<AnalysisData>(`/api/reconstruct/analysis/${pid}`);
            setAnalysis(data);
        } catch { /* no analysis yet */ }
    };

    const startReconstruction = useCallback(async () => {
        if (!projectId.trim()) return;
        setReconstructing(true);
        try {
            const result = await api<ReconstructJob>("/api/reconstruct/start", {
                method: "POST",
                body: JSON.stringify({
                    project_id: projectId.trim(),
                    mode: "full",
                    separator: "auto",
                }),
            });
            setJob(result);
        } catch {
            setReconstructing(false);
        }
    }, [projectId]);

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
        const normalized = Math.max(0, Math.min(1, (rms + 40) / 35));
        return `${Math.max(8, normalized * 100)}%`;
    };

    const formatTime = (seconds: number) => {
        const m = Math.floor(seconds / 60);
        const s = Math.floor(seconds % 60);
        return `${m}:${String(s).padStart(2, "0")}`;
    };

    return (
        <div className="space-y-6 p-6">
            {/* Header */}
            <div className="flex items-center justify-between">
                <div>
                    <h1 className="text-3xl font-bold bg-gradient-to-r from-amber-400 via-orange-500 to-red-500 bg-clip-text text-transparent">
                        🔬 Reconstruct
                    </h1>
                    <p className="text-zinc-400 mt-1">
                        Upload any track → auto-detect everything → bar-by-bar reconstruction
                    </p>
                </div>
            </div>

            {/* Project Input + Start */}
            <Card className="bg-zinc-900/50 border-zinc-800">
                <CardContent className="pt-6">
                    <div className="flex gap-3">
                        <input
                            type="text"
                            value={projectId}
                            onChange={(e) => setProjectId(e.target.value)}
                            placeholder="Enter project ID (from uploaded track)..."
                            className="flex-1 rounded-lg bg-zinc-800 border border-zinc-700 px-4 py-3 text-sm text-zinc-200 placeholder:text-zinc-500 focus:border-amber-500/50 focus:outline-none transition-colors"
                        />
                        <Button
                            onClick={startReconstruction}
                            disabled={reconstructing || !projectId.trim()}
                            className="bg-gradient-to-r from-amber-600 to-red-600 hover:from-amber-500 hover:to-red-500 text-white font-semibold px-6"
                            size="lg"
                        >
                            {reconstructing ? "⚡ Running..." : "🚀 Reconstruct"}
                        </Button>
                    </div>
                    <p className="text-[11px] text-zinc-600 mt-2">
                        Upload a track via Deconstructor → use the project ID here → full pipeline runs automatically
                    </p>
                </CardContent>
            </Card>

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
                                    : job.status === "error"
                                        ? "border-red-500/30 text-red-400"
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
                                className={`h-full rounded-full transition-all duration-500 ${job.status === "error"
                                        ? "bg-red-500"
                                        : "bg-gradient-to-r from-amber-500 to-red-500"
                                    }`}
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
                                            {status?.status === "completed" ? "✓" : status?.status === "running" ? "●" : status?.status === "error" ? "✗" : "○"}
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

                        {/* Error message */}
                        {job.status === "error" && (
                            <div className="mt-4 p-3 rounded-lg bg-red-500/10 border border-red-500/20 text-center">
                                <p className="text-red-400 text-sm">
                                    ❌ {job.stages[job.stage]?.message || "Pipeline error"}
                                </p>
                            </div>
                        )}

                        {/* Completion */}
                        {job.status === "completed" && job.result && (
                            <div className="mt-4 p-4 rounded-lg bg-emerald-500/10 border border-emerald-500/20">
                                <p className="text-emerald-400 font-semibold text-center">
                                    ✅ Reconstruction complete — {job.result.rendered_sections} sections rendered
                                </p>
                                {job.result.analysis && (
                                    <div className="flex gap-3 justify-center mt-3">
                                        <Badge variant="outline" className="border-amber-500/30 text-amber-400">
                                            {job.result.analysis.bpm.toFixed(1)} BPM
                                        </Badge>
                                        <Badge variant="outline" className="border-emerald-500/30 text-emerald-400">
                                            {job.result.analysis.key} {job.result.analysis.scale}
                                        </Badge>
                                        <Badge variant="outline" className="border-cyan-500/30 text-cyan-400">
                                            {job.result.analysis.sections_detected} sections
                                        </Badge>
                                        <Badge variant="outline" className="border-purple-500/30 text-purple-400">
                                            {formatTime(job.result.analysis.duration)}
                                        </Badge>
                                    </div>
                                )}
                            </div>
                        )}
                    </CardContent>
                </Card>
            )}

            {/* Analysis Results — auto-detected data */}
            {analysis && (
                <>
                    {/* Track DNA */}
                    <Card className="bg-zinc-900/50 border-zinc-800">
                        <CardHeader>
                            <CardTitle className="text-lg">🧬 Track DNA</CardTitle>
                            <CardDescription>Auto-detected — nothing hardcoded</CardDescription>
                        </CardHeader>
                        <CardContent>
                            <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-7 gap-3">
                                {[
                                    { label: "BPM", value: analysis.tempo.toFixed(1), color: "text-amber-400" },
                                    { label: "Key", value: `${analysis.key} ${analysis.scale}`, color: "text-emerald-400" },
                                    { label: "Duration", value: formatTime(analysis.duration), color: "text-purple-400" },
                                    { label: "LUFS", value: `${analysis.integrated_lufs.toFixed(1)}`, color: "text-cyan-400" },
                                    { label: "Peak", value: `${analysis.true_peak_dbfs.toFixed(1)} dBFS`, color: "text-orange-400" },
                                    { label: "Dyn Range", value: `${analysis.dynamic_range_db.toFixed(1)} dB`, color: "text-pink-400" },
                                    { label: "Sections", value: `${analysis.sections.length}`, color: "text-zinc-300" },
                                ].map((item) => (
                                    <div key={item.label} className="text-center p-3 rounded-lg bg-zinc-800/50">
                                        <div className={`text-lg font-bold ${item.color}`}>{item.value}</div>
                                        <div className="text-[10px] text-zinc-500 mt-1">{item.label}</div>
                                    </div>
                                ))}
                            </div>
                        </CardContent>
                    </Card>

                    {/* Energy Curve */}
                    {analysis.energy_curve.length > 0 && (
                        <Card className="bg-zinc-900/50 border-zinc-800">
                            <CardHeader>
                                <CardTitle className="text-lg">📊 Energy Map</CardTitle>
                                <CardDescription>
                                    Per-bar RMS energy — auto-detected from audio ({analysis.energy_curve.length} bars)
                                </CardDescription>
                            </CardHeader>
                            <CardContent>
                                <div className="flex items-end gap-[1px] h-32">
                                    {analysis.energy_curve.map((point, i) => (
                                        <div
                                            key={i}
                                            className="flex-1 cursor-pointer group"
                                            title={`Bar ${point.bar}: ${point.rms_db.toFixed(1)} dB`}
                                        >
                                            <div
                                                className={`w-full rounded-t transition-all group-hover:opacity-70 ${getEnergyColor(point.rms_db)}`}
                                                style={{ height: getEnergyHeight(point.rms_db) }}
                                            />
                                        </div>
                                    ))}
                                </div>
                                <div className="flex justify-between mt-2 text-[10px] text-zinc-600">
                                    <span>Bar 0</span>
                                    <span>Bar {Math.floor(analysis.energy_curve.length / 2)}</span>
                                    <span>Bar {analysis.energy_curve.length}</span>
                                </div>
                            </CardContent>
                        </Card>
                    )}

                    {/* Auto-Detected Sections */}
                    <Card className="bg-zinc-900/50 border-zinc-800">
                        <CardHeader>
                            <CardTitle className="text-lg">🎼 Sections</CardTitle>
                            <CardDescription>
                                {analysis.sections.length} sections — auto-detected from energy analysis
                            </CardDescription>
                        </CardHeader>
                        <CardContent>
                            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
                                {analysis.sections.map((section, i) => (
                                    <div
                                        key={i}
                                        className={`rounded-lg border p-4 cursor-pointer transition-all ${selectedSection === section.name + i
                                                ? "border-amber-500/50 bg-amber-500/10"
                                                : "border-zinc-800 bg-zinc-900/30 hover:border-zinc-700"
                                            }`}
                                        onClick={() =>
                                            setSelectedSection(
                                                selectedSection === section.name + i ? null : section.name + i
                                            )
                                        }
                                    >
                                        <div className="flex items-center justify-between mb-2">
                                            <span className="font-semibold text-sm">
                                                {section.name}
                                            </span>
                                            <Badge variant="outline" className="text-[10px] border-zinc-700">
                                                Bar {section.start_bar}–{section.end_bar}
                                            </Badge>
                                        </div>
                                        <div className="flex gap-3 mb-2">
                                            <span className="text-[10px] text-amber-400">
                                                {section.avg_rms_db.toFixed(1)} dB
                                            </span>
                                            <span className="text-[10px] text-cyan-400">
                                                {formatTime(section.start_time)} — {formatTime(section.end_time)}
                                            </span>
                                            <span className="text-[10px] text-purple-400">
                                                {section.element_count} elements
                                            </span>
                                        </div>
                                        <div className="flex flex-wrap gap-1">
                                            {section.characteristics.map((c) => (
                                                <Badge
                                                    key={c}
                                                    variant="secondary"
                                                    className="text-[9px] bg-zinc-800 text-zinc-400"
                                                >
                                                    {c}
                                                </Badge>
                                            ))}
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </CardContent>
                    </Card>

                    {/* QC Dimensions Preview */}
                    {job?.result?.qc && (
                        <Card className="bg-zinc-900/50 border-zinc-800">
                            <CardHeader>
                                <CardTitle className="text-lg">🔍 QC — 12 Dimensions</CardTitle>
                                <CardDescription>
                                    Target: ≥{job.result.qc.target_score}% average score
                                </CardDescription>
                            </CardHeader>
                            <CardContent>
                                <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-2">
                                    {QC_DIMENSIONS.map((dim) => {
                                        const score = job.result?.qc?.dimensions[dim] || 0;
                                        return (
                                            <div key={dim} className="rounded-lg bg-zinc-800/50 p-3">
                                                <div className="text-[10px] text-zinc-500 capitalize">
                                                    {dim.replace(/_/g, " ")}
                                                </div>
                                                <div className="flex items-center gap-2 mt-1">
                                                    <div className="flex-1 h-1.5 bg-zinc-700 rounded-full overflow-hidden">
                                                        <div
                                                            className={`h-full rounded-full ${score >= 90 ? "bg-emerald-500" :
                                                                    score >= 70 ? "bg-amber-500" : "bg-zinc-600"
                                                                }`}
                                                            style={{ width: `${score}%` }}
                                                        />
                                                    </div>
                                                    <span className="text-[10px] text-zinc-400 w-6 text-right">
                                                        {score}
                                                    </span>
                                                </div>
                                            </div>
                                        );
                                    })}
                                </div>
                            </CardContent>
                        </Card>
                    )}
                </>
            )}

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

            {/* Footer */}
            <div className="text-center text-zinc-600 text-xs py-2 border-t border-zinc-800/50">
                100% Track-Agnostic — Upload any track, auto-detect everything, reconstruct bar-by-bar
            </div>
        </div>
    );
}
