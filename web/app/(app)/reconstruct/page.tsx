"use client";

import { useState, useEffect, useCallback, useRef, useMemo } from "react";
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
import dynamic from "next/dynamic";

const WaveformXRay = dynamic(
    () => import("@/components/visualizer/WaveformXRay"),
    { ssr: false, loading: () => <div className="h-32 bg-zinc-900/50 rounded-xl animate-pulse" /> }
);

/* ── Types (track-agnostic) ──────────────────────────── */

interface StageStatus {
    status: "pending" | "running" | "completed" | "error";
    message: string;
}

interface QCDimensionResult {
    score: number;
    detail: string;
}

interface LogEntry {
    ts: string;
    level: "info" | "success" | "warn" | "error" | "stage";
    msg: string;
}

interface StemAnalysis {
    rms_db: number;
    peak_db: number;
    energy_pct: number;
    duration: number;
    file_size_mb: number;
    freq_bands: {
        low: number;
        mid: number;
        high: number;
    };
    error?: string;
}

interface ReconstructJob {
    job_id: string;
    project_id: string;
    status: "running" | "completed" | "error";
    stage: string;
    progress: number;
    stages: Record<string, StageStatus>;
    logs?: LogEntry[];
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
        rendered_stems: number;
        stem_analysis?: Record<string, StemAnalysis>;
        master?: {
            output?: string;
            peak_dbtp?: number;
            rms_db?: number;
            est_lufs?: number;
            stages_applied?: string[];
            error?: string;
        };
        qc?: {
            dimensions: Record<string, QCDimensionResult | number>;
            overall_score: number;
            target_score: number;
            passed?: boolean;
            weakest?: string;
            strongest?: string;
        };
        files?: {
            original?: string;
            mix?: string;
            master?: string;
            stems?: string[];
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
    const [uploading, setUploading] = useState(false);
    const [uploadProgress, setUploadProgress] = useState(0);
    const [uploadFile_, setUploadFile_] = useState<{ name: string; size: number } | null>(null);
    const [dragOver, setDragOver] = useState(false);
    const [elapsed, setElapsed] = useState(0);
    const [stageTimings, setStageTimings] = useState<Record<string, number>>({});
    const [pipelineStartTime, setPipelineStartTime] = useState<number | null>(null);
    const [lastStageChange, setLastStageChange] = useState<number | null>(null);
    const fileInputRef = useRef<HTMLInputElement>(null);
    const prevStageRef = useRef<string | null>(null);
    const [logsOpen, setLogsOpen] = useState(true);
    const logsEndRef = useRef<HTMLDivElement>(null);
    const [xrayAnalysis, setXrayAnalysis] = useState<any>(null);

    // Elapsed timer
    useEffect(() => {
        if (!pipelineStartTime || !reconstructing) return;
        const timer = setInterval(() => {
            setElapsed(Math.floor((Date.now() - pipelineStartTime) / 1000));
        }, 1000);
        return () => clearInterval(timer);
    }, [pipelineStartTime, reconstructing]);

    // Track stage timings
    useEffect(() => {
        if (!job || !lastStageChange) return;
        const currentStage = job.stage;
        if (prevStageRef.current && prevStageRef.current !== currentStage) {
            const now = Date.now();
            setStageTimings((prev) => ({
                ...prev,
                [prevStageRef.current!]: Math.floor((now - lastStageChange) / 1000),
            }));
            setLastStageChange(now);
        }
        prevStageRef.current = currentStage;
    }, [job?.stage]);

    // Auto-scroll logs
    useEffect(() => {
        if (logsOpen && logsEndRef.current) {
            logsEndRef.current.scrollIntoView({ behavior: "smooth" });
        }
    }, [job?.logs?.length, logsOpen]);

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
                    // Capture final stage timing
                    if (lastStageChange && prevStageRef.current) {
                        setStageTimings((prev) => ({
                            ...prev,
                            [prevStageRef.current!]: Math.floor((Date.now() - lastStageChange) / 1000),
                        }));
                    }
                    if (updated.result?.analysis) {
                        loadAnalysis(updated.project_id);
                    }
                    // Load X-Ray deep analysis
                    loadXrayAnalysis(updated.job_id);
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

    const loadXrayAnalysis = async (jobId: string) => {
        try {
            const data = await api<any>(`/api/reconstruct/waveform/${jobId}`);
            if (data?.analysis) {
                setXrayAnalysis(data.analysis);
            }
        } catch { /* waveform analysis not ready yet */ }
    };

    const formatBytes = (bytes: number) => {
        if (bytes < 1024) return `${bytes} B`;
        if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
        return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
    };

    const formatElapsed = (secs: number) => {
        const m = Math.floor(secs / 60);
        const s = secs % 60;
        return m > 0 ? `${m}m ${String(s).padStart(2, "0")}s` : `${s}s`;
    };

    const uploadFile = useCallback(async (file: File) => {
        setUploading(true);
        setUploadProgress(0);
        setUploadFile_({ name: file.name, size: file.size });
        try {
            const formData = new FormData();
            formData.append("file", file);
            const token = typeof window !== "undefined" ? localStorage.getItem("auralis_token") : null;

            // Use XMLHttpRequest for progress tracking
            const data = await new Promise<{ project_id: string }>((resolve, reject) => {
                const xhr = new XMLHttpRequest();
                xhr.open("POST", `${process.env.NEXT_PUBLIC_API_URL || ""}/api/ear/upload`);
                if (token) xhr.setRequestHeader("Authorization", `Bearer ${token}`);

                xhr.upload.onprogress = (e) => {
                    if (e.lengthComputable) {
                        setUploadProgress(Math.round((e.loaded / e.total) * 100));
                    }
                };

                xhr.onload = () => {
                    if (xhr.status >= 200 && xhr.status < 300) {
                        resolve(JSON.parse(xhr.responseText));
                    } else {
                        reject(new Error(`Upload failed: ${xhr.status}`));
                    }
                };
                xhr.onerror = () => reject(new Error("Upload failed"));
                xhr.send(formData);
            });

            setProjectId(data.project_id);
            return data.project_id;
        } finally {
            setUploading(false);
            setUploadProgress(0);
        }
    }, []);

    const startReconstruction = useCallback(async (pid?: string) => {
        const id = pid || projectId.trim();
        if (!id) return;
        setReconstructing(true);
        setElapsed(0);
        setStageTimings({});
        const now = Date.now();
        setPipelineStartTime(now);
        setLastStageChange(now);
        prevStageRef.current = null;
        try {
            const result = await api<ReconstructJob>("/api/reconstruct/start", {
                method: "POST",
                body: JSON.stringify({
                    project_id: id,
                    mode: "full",
                    separator: "auto",
                }),
            });
            setJob(result);
        } catch {
            setReconstructing(false);
            setPipelineStartTime(null);
        }
    }, [projectId]);

    const handleDrop = useCallback(async (e: React.DragEvent) => {
        e.preventDefault();
        setDragOver(false);
        const file = e.dataTransfer.files[0];
        if (!file) return;
        const pid = await uploadFile(file);
        if (pid) {
            await startReconstruction(pid);
        }
    }, [uploadFile, startReconstruction]);

    const handleFileSelect = useCallback(async (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0];
        if (!file) return;
        const pid = await uploadFile(file);
        if (pid) {
            await startReconstruction(pid);
        }
    }, [uploadFile, startReconstruction]);

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

    const getStemConfig = (name: string) => {
        const configs: Record<string, { icon: string; gradient: string; glow: string; textColor: string; barGradient: string }> = {
            vocals: { icon: "🎤", gradient: "from-violet-900/40 to-purple-900/40", glow: "rgba(139,92,246,0.15)", textColor: "text-violet-400", barGradient: "from-violet-600 to-purple-500" },
            drums: { icon: "🥁", gradient: "from-red-900/40 to-orange-900/40", glow: "rgba(239,68,68,0.15)", textColor: "text-red-400", barGradient: "from-red-600 to-orange-500" },
            bass: { icon: "🎸", gradient: "from-blue-900/40 to-cyan-900/40", glow: "rgba(59,130,246,0.15)", textColor: "text-blue-400", barGradient: "from-blue-600 to-cyan-500" },
            other: { icon: "🎹", gradient: "from-emerald-900/40 to-teal-900/40", glow: "rgba(16,185,129,0.15)", textColor: "text-emerald-400", barGradient: "from-emerald-600 to-teal-500" },
            piano: { icon: "🎹", gradient: "from-pink-900/40 to-rose-900/40", glow: "rgba(236,72,153,0.15)", textColor: "text-pink-400", barGradient: "from-pink-600 to-rose-500" },
            guitar: { icon: "🎸", gradient: "from-amber-900/40 to-yellow-900/40", glow: "rgba(245,158,11,0.15)", textColor: "text-amber-400", barGradient: "from-amber-600 to-yellow-500" },
            instrumental: { icon: "🎵", gradient: "from-teal-900/40 to-emerald-900/40", glow: "rgba(20,184,166,0.15)", textColor: "text-teal-400", barGradient: "from-teal-600 to-emerald-500" },
        };
        return configs[name.toLowerCase()] || { icon: "🎵", gradient: "from-zinc-800/40 to-zinc-900/40", glow: "rgba(161,161,170,0.1)", textColor: "text-zinc-400", barGradient: "from-zinc-600 to-zinc-500" };
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

            {/* Upload + Start */}
            <Card className="bg-zinc-900/50 border-zinc-800">
                <CardContent className="pt-6">
                    {/* Drag & Drop Zone */}
                    <div
                        onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
                        onDragLeave={() => setDragOver(false)}
                        onDrop={handleDrop}
                        onClick={() => !uploading && fileInputRef.current?.click()}
                        className={`relative rounded-xl border-2 border-dashed p-8 text-center transition-all mb-4 overflow-hidden ${uploading
                            ? "border-cyan-500/50 bg-cyan-500/5 cursor-wait"
                            : dragOver
                                ? "border-amber-500 bg-amber-500/10 cursor-copy"
                                : "border-zinc-700 hover:border-zinc-600 bg-zinc-800/30 cursor-pointer"
                            }`}
                    >
                        <input
                            ref={fileInputRef}
                            type="file"
                            accept=".wav,.mp3,.flac,.ogg,.m4a,.aac"
                            onChange={handleFileSelect}
                            className="hidden"
                        />

                        {/* Upload progress bar — fills from left */}
                        {uploading && (
                            <div className="absolute bottom-0 left-0 right-0 h-1 bg-zinc-800">
                                <div
                                    className="h-full bg-gradient-to-r from-cyan-500 to-teal-400 transition-all duration-300 ease-out"
                                    style={{ width: `${uploadProgress}%` }}
                                />
                            </div>
                        )}

                        {uploading ? (
                            <div className="space-y-3">
                                {/* Animated spinner */}
                                <div className="flex justify-center">
                                    <div className="w-10 h-10 rounded-full border-2 border-cyan-500/30 border-t-cyan-400 animate-spin" />
                                </div>
                                {/* File info */}
                                <div>
                                    <p className="text-sm text-cyan-300 font-medium">
                                        Uploading — {uploadProgress}%
                                    </p>
                                    {uploadFile_ && (
                                        <p className="text-[11px] text-zinc-500 mt-1">
                                            📄 {uploadFile_.name}
                                            <span className="text-zinc-600 ml-2">{formatBytes(uploadFile_.size)}</span>
                                        </p>
                                    )}
                                </div>
                                {/* Progress bar (detailed) */}
                                <div className="max-w-xs mx-auto">
                                    <div className="w-full h-2 bg-zinc-800 rounded-full overflow-hidden">
                                        <div
                                            className="h-full bg-gradient-to-r from-cyan-500 to-teal-400 rounded-full transition-all duration-300"
                                            style={{ width: `${uploadProgress}%` }}
                                        />
                                    </div>
                                    <div className="flex justify-between mt-1 text-[10px] text-zinc-600">
                                        <span>{formatBytes(Math.round(uploadFile_!.size * uploadProgress / 100))}</span>
                                        <span>{formatBytes(uploadFile_!.size)}</span>
                                    </div>
                                </div>
                            </div>
                        ) : (
                            <>
                                <div className="text-4xl mb-2">{dragOver ? "📥" : "🎵"}</div>
                                <p className="text-sm text-zinc-300 font-medium">
                                    Drop audio file here or click to browse
                                </p>
                                <p className="text-[11px] text-zinc-600 mt-1">
                                    WAV, MP3, FLAC, OGG, AAC — auto-runs full reconstruction pipeline
                                </p>
                            </>
                        )}
                    </div>

                    {/* Or use existing project ID */}
                    <div className="flex gap-3">
                        <input
                            type="text"
                            value={projectId}
                            onChange={(e) => setProjectId(e.target.value)}
                            placeholder="Or enter existing project ID..."
                            className="flex-1 rounded-lg bg-zinc-800 border border-zinc-700 px-4 py-3 text-sm text-zinc-200 placeholder:text-zinc-500 focus:border-amber-500/50 focus:outline-none transition-colors"
                        />
                        <Button
                            onClick={() => startReconstruction()}
                            disabled={reconstructing || uploading || !projectId.trim()}
                            className="bg-gradient-to-r from-amber-600 to-red-600 hover:from-amber-500 hover:to-red-500 text-white font-semibold px-6"
                            size="lg"
                        >
                            {reconstructing ? "⚡ Running..." : "🚀 Reconstruct"}
                        </Button>
                    </div>
                </CardContent>
            </Card>

            {/* Pipeline Stages */}
            {job && (
                <Card className="bg-zinc-900/50 border-zinc-800">
                    <CardHeader>
                        <div className="flex items-center justify-between">
                            <div className="flex items-center gap-3">
                                <CardTitle className="text-lg">⚡ Pipeline</CardTitle>
                                {job.status === "running" && (
                                    <span className="text-xs font-mono text-zinc-500 bg-zinc-800/50 px-2 py-1 rounded">
                                        ⏱ {formatElapsed(elapsed)}
                                    </span>
                                )}
                                {job.status === "completed" && (
                                    <span className="text-xs font-mono text-emerald-500 bg-emerald-500/10 px-2 py-1 rounded">
                                        ✓ {formatElapsed(elapsed)}
                                    </span>
                                )}
                            </div>
                            <div className="flex items-center gap-2">
                                <Badge
                                    variant="outline"
                                    className={job.status === "completed"
                                        ? "border-emerald-500/30 text-emerald-400"
                                        : job.status === "error"
                                            ? "border-red-500/30 text-red-400"
                                            : "border-amber-500/30 text-amber-400"
                                    }
                                >
                                    {job.status === "completed" ? "Complete" : job.status === "error" ? "Error" : `${job.progress}%`}
                                </Badge>
                            </div>
                        </div>
                    </CardHeader>
                    <CardContent>
                        {/* Progress bar */}
                        <div className="w-full h-2 bg-zinc-800 rounded-full mb-6 overflow-hidden">
                            <div
                                className={`h-full rounded-full transition-all duration-700 ease-out ${job.status === "error"
                                    ? "bg-red-500"
                                    : job.status === "completed"
                                        ? "bg-gradient-to-r from-emerald-500 to-teal-400"
                                        : "bg-gradient-to-r from-amber-500 via-orange-500 to-red-500"
                                    }`}
                                style={{ width: `${job.progress}%` }}
                            />
                        </div>

                        {/* Stage cards — upgraded with timing & connectors */}
                        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3">
                            {STAGES.map((stage, idx) => {
                                const status = job.stages[stage.key];
                                const stageStatus = status?.status || "pending";
                                const timing = stageTimings[stage.key];
                                const isActive = stageStatus === "running";
                                return (
                                    <div
                                        key={stage.key}
                                        className={`relative rounded-lg border p-3 text-center transition-all duration-500 ${getStageColor(stageStatus)} ${isActive ? "ring-1 ring-amber-500/30 shadow-lg shadow-amber-500/10" : ""
                                            }`}
                                    >
                                        {/* Stage number */}
                                        <div className="absolute -top-2 -left-1 text-[9px] font-bold bg-zinc-900 text-zinc-600 px-1.5 rounded-full border border-zinc-800">
                                            {idx + 1}
                                        </div>

                                        {/* Icon with status indicator */}
                                        <div className="text-2xl mb-1 relative inline-block">
                                            {stage.icon}
                                            {stageStatus === "completed" && (
                                                <span className="absolute -top-1 -right-3 text-xs">✅</span>
                                            )}
                                            {isActive && (
                                                <span className="absolute -top-1 -right-3 w-3 h-3 rounded-full bg-amber-400 animate-ping" />
                                            )}
                                        </div>

                                        <div className="text-xs font-bold">{stage.label}</div>
                                        <div className="text-[10px] mt-0.5 opacity-60">{stage.desc}</div>

                                        {/* Status + timing */}
                                        <div className="mt-1.5 text-[10px]">
                                            {stageStatus === "completed" && timing !== undefined ? (
                                                <span className="text-emerald-400 font-mono">{timing}s</span>
                                            ) : isActive ? (
                                                <span className="text-amber-400 animate-pulse">Processing…</span>
                                            ) : stageStatus === "error" ? (
                                                <span className="text-red-400">Failed</span>
                                            ) : (
                                                <span className="text-zinc-600">Queued</span>
                                            )}
                                        </div>
                                    </div>
                                );
                            })}
                        </div>

                        {/* Current stage message */}
                        {job.status === "running" && (
                            <div className="mt-4 flex items-center justify-center gap-2 text-sm text-zinc-400">
                                <div className="w-4 h-4 rounded-full border-2 border-amber-500/30 border-t-amber-400 animate-spin" />
                                <span>{job.stages[job.stage]?.message || "Processing..."}</span>
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
                                    ✅ Reconstruction complete — {job.result.rendered_stems ?? 0} stems rendered
                                    {elapsed > 0 && (
                                        <span className="text-emerald-500/60 font-normal ml-2">in {formatElapsed(elapsed)}</span>
                                    )}
                                </p>
                                {job.result.analysis && (
                                    <div className="flex flex-wrap gap-3 justify-center mt-3">
                                        <Badge variant="outline" className="border-amber-500/30 text-amber-400">
                                            {(job.result.analysis.bpm ?? 0).toFixed(1)} BPM
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
                                        {job.result.master?.est_lufs && (
                                            <Badge variant="outline" className="border-orange-500/30 text-orange-400">
                                                Master: {(job.result.master.est_lufs ?? 0).toFixed(1)} LUFS
                                            </Badge>
                                        )}
                                        {job.result.qc && (
                                            <Badge variant="outline" className={`${job.result.qc.passed
                                                ? "border-emerald-500/30 text-emerald-400"
                                                : "border-amber-500/30 text-amber-400"
                                                }`}>
                                                QC: {(job.result.qc.overall_score ?? 0).toFixed(1)}%
                                            </Badge>
                                        )}
                                    </div>
                                )}

                                {/* Stage timing summary */}
                                {Object.keys(stageTimings).length > 0 && (
                                    <div className="flex flex-wrap gap-2 justify-center mt-3 pt-3 border-t border-emerald-500/10">
                                        {STAGES.map((s) => {
                                            const t = stageTimings[s.key];
                                            if (t === undefined) return null;
                                            return (
                                                <span key={s.key} className="text-[10px] text-zinc-500 font-mono">
                                                    {s.label}: {t}s
                                                </span>
                                            );
                                        })}
                                    </div>
                                )}
                            </div>
                        )}
                    </CardContent>
                </Card>
            )}

            {/* Live Logs Terminal */}
            {job && (job.logs?.length ?? 0) > 0 && (
                <Card className="bg-zinc-950/80 border-zinc-800">
                    <CardHeader className="pb-2 cursor-pointer" onClick={() => setLogsOpen(!logsOpen)}>
                        <div className="flex items-center justify-between">
                            <div className="flex items-center gap-2">
                                <span className="text-green-500 font-mono text-sm">▸</span>
                                <CardTitle className="text-sm font-mono text-zinc-400">Engine Logs</CardTitle>
                                <Badge variant="outline" className="border-zinc-700 text-zinc-500 text-[10px] font-mono">
                                    {job.logs?.length ?? 0} lines
                                </Badge>
                                {job.status === "running" && (
                                    <span className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />
                                )}
                            </div>
                            <span className="text-zinc-600 text-xs">{logsOpen ? "▾" : "▸"}</span>
                        </div>
                    </CardHeader>
                    {logsOpen && (
                        <CardContent className="pt-0">
                            <div className="bg-black/60 rounded-lg border border-zinc-800/50 p-3 max-h-[280px] overflow-y-auto font-mono text-[11px] leading-relaxed scrollbar-thin scrollbar-thumb-zinc-700 scrollbar-track-transparent">
                                {job.logs?.map((log, i) => {
                                    const color =
                                        log.level === "success" ? "text-emerald-400"
                                            : log.level === "error" ? "text-red-400"
                                                : log.level === "warn" ? "text-amber-400"
                                                    : log.level === "stage" ? "text-cyan-400 font-bold"
                                                        : "text-zinc-400";
                                    return (
                                        <div key={i} className={`flex gap-2 ${color} ${log.level === "stage" ? "mt-2 mb-0.5" : ""}`}>
                                            <span className="text-zinc-600 select-none shrink-0">{log.ts}</span>
                                            <span className={`select-none shrink-0 w-4 text-center ${log.level === "error" ? "text-red-500" : log.level === "warn" ? "text-amber-500" : log.level === "success" ? "text-emerald-500" : log.level === "stage" ? "text-cyan-500" : "text-zinc-600"}`}>
                                                {log.level === "error" ? "✗" : log.level === "warn" ? "!" : log.level === "success" ? "✓" : log.level === "stage" ? "▸" : "·"}
                                            </span>
                                            <span className="break-all">{log.msg}</span>
                                        </div>
                                    );
                                })}
                                <div ref={logsEndRef} />
                            </div>
                        </CardContent>
                    )}
                </Card>
            )}

            {/* 🎛️ Separated Stems — Instrument Detection Visual */}
            {job?.result?.stem_analysis && Object.keys(job.result.stem_analysis).length > 0 && (
                <Card className="bg-zinc-900/50 border-zinc-800 overflow-hidden">
                    <CardHeader>
                        <div className="flex items-center justify-between">
                            <div className="flex items-center gap-2">
                                <CardTitle className="text-lg">🎛️ Separated Stems</CardTitle>
                                <Badge variant="outline" className="border-cyan-500/30 text-cyan-400">
                                    {Object.keys(job.result.stem_analysis).length} instruments
                                </Badge>
                            </div>
                            <CardDescription>AI-detected — per-instrument analysis</CardDescription>
                        </div>
                    </CardHeader>
                    <CardContent className="space-y-3">
                        {Object.entries(job.result.stem_analysis).map(([name, data], idx) => {
                            if (data.error) return null;
                            const config = getStemConfig(name);
                            const energyClamped = Math.min(data.energy_pct ?? 0, 100);
                            return (
                                <div
                                    key={name}
                                    className="relative group rounded-xl border border-zinc-800/80 bg-gradient-to-r from-zinc-900/90 to-zinc-950/90 backdrop-blur-sm p-4 hover:border-zinc-700/60 transition-all duration-500"
                                    style={{
                                        animationDelay: `${idx * 120}ms`,
                                        animation: "fadeSlideIn 0.5s ease-out both",
                                    }}
                                >
                                    {/* Glow effect */}
                                    <div
                                        className="absolute inset-0 rounded-xl opacity-0 group-hover:opacity-100 transition-opacity duration-500"
                                        style={{ background: `radial-gradient(ellipse at 0% 50%, ${config.glow}, transparent 60%)` }}
                                    />

                                    <div className="relative flex items-center gap-4">
                                        {/* Icon + Name */}
                                        <div className="flex items-center gap-3 min-w-[140px]">
                                            <div className={`text-3xl p-2 rounded-lg bg-gradient-to-br ${config.gradient} bg-opacity-20 shadow-lg`} style={{ boxShadow: `0 0 20px ${config.glow}` }}>
                                                {config.icon}
                                            </div>
                                            <div>
                                                <div className="font-bold text-sm capitalize text-zinc-100">{name}</div>
                                                <div className="text-[10px] text-zinc-500 font-mono">{data.file_size_mb ?? '?'} MB</div>
                                            </div>
                                        </div>

                                        {/* Energy bar */}
                                        <div className="flex-1">
                                            <div className="flex items-center justify-between mb-1">
                                                <span className="text-[10px] text-zinc-500">Energy contribution</span>
                                                <span className={`text-xs font-bold font-mono ${config.textColor}`}>{data.energy_pct ?? 0}%</span>
                                            </div>
                                            <div className="w-full h-3 bg-zinc-800/80 rounded-full overflow-hidden">
                                                <div
                                                    className={`h-full rounded-full bg-gradient-to-r ${config.barGradient} transition-all duration-1000 ease-out`}
                                                    style={{ width: `${energyClamped}%` }}
                                                />
                                            </div>
                                        </div>

                                        {/* dB Meters */}
                                        <div className="flex gap-4 min-w-[120px]">
                                            <div className="text-center">
                                                <div className="text-[9px] text-zinc-600 uppercase tracking-wider">RMS</div>
                                                <div className={`text-sm font-mono font-bold ${config.textColor}`}>{data.rms_db ?? '–'} dB</div>
                                            </div>
                                            <div className="text-center">
                                                <div className="text-[9px] text-zinc-600 uppercase tracking-wider">Peak</div>
                                                <div className="text-sm font-mono font-bold text-zinc-300">{data.peak_db ?? '–'} dBFS</div>
                                            </div>
                                        </div>

                                        {/* Frequency Spectrum */}
                                        <div className="min-w-[80px]">
                                            <div className="text-[9px] text-zinc-600 uppercase tracking-wider text-center mb-1">Spectrum</div>
                                            <div className="flex gap-0.5 items-end h-6 justify-center">
                                                <div
                                                    className="w-5 rounded-t bg-gradient-to-t from-blue-600 to-blue-400 transition-all duration-700"
                                                    style={{ height: `${Math.max((data.freq_bands?.low ?? 0) * 0.24, 2)}px` }}
                                                    title={`Low: ${data.freq_bands?.low ?? 0}%`}
                                                />
                                                <div
                                                    className="w-5 rounded-t bg-gradient-to-t from-emerald-600 to-emerald-400 transition-all duration-700"
                                                    style={{ height: `${Math.max((data.freq_bands?.mid ?? 0) * 0.24, 2)}px` }}
                                                    title={`Mid: ${data.freq_bands?.mid ?? 0}%`}
                                                />
                                                <div
                                                    className="w-5 rounded-t bg-gradient-to-t from-amber-600 to-amber-400 transition-all duration-700"
                                                    style={{ height: `${Math.max((data.freq_bands?.high ?? 0) * 0.24, 2)}px` }}
                                                    title={`High: ${data.freq_bands?.high ?? 0}%`}
                                                />
                                            </div>
                                            <div className="flex gap-0.5 justify-center mt-0.5">
                                                <span className="text-[7px] text-blue-400 w-5 text-center">L</span>
                                                <span className="text-[7px] text-emerald-400 w-5 text-center">M</span>
                                                <span className="text-[7px] text-amber-400 w-5 text-center">H</span>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            );
                        })}
                    </CardContent>
                </Card>
            )}

            {/* 🧬 Sonic Intelligence — Track DNA Deep Analysis */}
            {xrayAnalysis && Object.keys(xrayAnalysis).length > 0 && (
                <Card className="bg-zinc-900/50 border-zinc-800 overflow-hidden">
                    <CardHeader>
                        <div className="flex items-center justify-between">
                            <div className="flex items-center gap-2">
                                <CardTitle className="text-lg">🧬 Sonic Intelligence</CardTitle>
                                <Badge variant="outline" className="border-purple-500/30 text-purple-400">
                                    Track DNA
                                </Badge>
                            </div>
                            <CardDescription>Deep spectral analysis — every element identified</CardDescription>
                        </div>
                    </CardHeader>
                    <CardContent className="space-y-6">

                        {/* ── Arrangement Row ── */}
                        {xrayAnalysis.arrangement && (
                            <div>
                                <div className="flex items-center gap-2 mb-3">
                                    <span className="text-sm font-semibold text-zinc-300">🔗 Arrangement</span>
                                    <div className="flex gap-2 ml-auto">
                                        {xrayAnalysis.arrangement.key && (
                                            <Badge className="bg-emerald-500/20 text-emerald-400 border-emerald-500/30">
                                                {xrayAnalysis.arrangement.key} {xrayAnalysis.arrangement.scale}
                                            </Badge>
                                        )}
                                        {xrayAnalysis.arrangement.tempo_bpm > 0 && (
                                            <Badge className="bg-amber-500/20 text-amber-400 border-amber-500/30">
                                                {xrayAnalysis.arrangement.tempo_bpm} BPM
                                            </Badge>
                                        )}
                                        {xrayAnalysis.arrangement.sidechain_detected && (
                                            <Badge className="bg-pink-500/20 text-pink-400 border-pink-500/30">
                                                ⛓️ Sidechain
                                            </Badge>
                                        )}
                                    </div>
                                </div>
                                {/* Section timeline */}
                                {xrayAnalysis.arrangement.sections?.length > 0 && (
                                    <div className="flex gap-1 h-8 rounded-lg overflow-hidden">
                                        {xrayAnalysis.arrangement.sections.map((sec: any, i: number) => {
                                            const sectionColors: Record<string, string> = {
                                                Intro: 'bg-blue-600/60',
                                                Verse: 'bg-emerald-600/60',
                                                Chorus: 'bg-amber-600/60',
                                                Drop: 'bg-red-600/60',
                                                Bridge: 'bg-purple-600/60',
                                                Build: 'bg-orange-600/60',
                                                Outro: 'bg-zinc-600/60',
                                            };
                                            const totalDur = xrayAnalysis.arrangement.sections.reduce(
                                                (acc: number, s: any) => acc + (s.end - s.start), 0
                                            );
                                            const width = ((sec.end - sec.start) / totalDur) * 100;
                                            return (
                                                <div
                                                    key={i}
                                                    className={`${sectionColors[sec.label] || 'bg-zinc-700/60'} relative group flex items-center justify-center rounded transition-all hover:brightness-125 cursor-pointer`}
                                                    style={{ width: `${Math.max(width, 2)}%` }}
                                                    title={`${sec.label}: ${sec.start}s – ${sec.end}s`}
                                                >
                                                    <span className="text-[9px] font-bold text-white/80 truncate px-1">{sec.label}</span>
                                                    <div className="absolute -bottom-5 opacity-0 group-hover:opacity-100 text-[8px] text-zinc-400 whitespace-nowrap transition-opacity">
                                                        {Math.round(sec.start)}s–{Math.round(sec.end)}s
                                                    </div>
                                                </div>
                                            );
                                        })}
                                    </div>
                                )}
                            </div>
                        )}

                        {/* ── Instruments Grid ── */}
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">

                            {/* Drums */}
                            {xrayAnalysis.drums && (
                                <div className="rounded-xl border border-red-500/20 bg-gradient-to-br from-red-950/30 to-zinc-950 p-4">
                                    <div className="flex items-center justify-between mb-3">
                                        <span className="text-sm font-bold text-red-400">🥁 Percussion</span>
                                        <Badge variant="outline" className="text-[10px] border-red-500/30 text-red-300">
                                            {xrayAnalysis.drums.total_hits} hits
                                        </Badge>
                                    </div>
                                    <div className="flex flex-wrap gap-1.5">
                                        {xrayAnalysis.drums.instruments && Object.entries(xrayAnalysis.drums.instruments).sort((a: any, b: any) => b[1] - a[1]).map(([name, count]: [string, any]) => (
                                            <div key={name} className="flex items-center gap-1 px-2 py-1 rounded-md bg-red-500/10 border border-red-500/20">
                                                <span className="text-[11px] text-red-300 font-medium">{name}</span>
                                                <span className="text-[9px] text-red-400/70 font-mono">{count}</span>
                                            </div>
                                        ))}
                                    </div>
                                </div>
                            )}

                            {/* Bass */}
                            {xrayAnalysis.bass && (
                                <div className="rounded-xl border border-blue-500/20 bg-gradient-to-br from-blue-950/30 to-zinc-950 p-4">
                                    <div className="flex items-center justify-between mb-3">
                                        <span className="text-sm font-bold text-blue-400">🎸 Bass</span>
                                        <Badge variant="outline" className="text-[10px] border-blue-500/30 text-blue-300">
                                            {xrayAnalysis.bass.type}
                                        </Badge>
                                    </div>
                                    <div className="grid grid-cols-3 gap-2 text-center">
                                        <div className="bg-blue-500/10 rounded-lg p-2">
                                            <div className="text-[10px] text-zinc-500">Confidence</div>
                                            <div className="text-sm font-bold text-blue-300 font-mono">
                                                {Math.round((xrayAnalysis.bass.type_confidence ?? 0) * 100)}%
                                            </div>
                                        </div>
                                        <div className="bg-blue-500/10 rounded-lg p-2">
                                            <div className="text-[10px] text-zinc-500">Notes</div>
                                            <div className="text-sm font-bold text-blue-300 font-mono">
                                                {xrayAnalysis.bass.summary?.total_notes ?? 0}
                                            </div>
                                        </div>
                                        <div className="bg-blue-500/10 rounded-lg p-2">
                                            <div className="text-[10px] text-zinc-500">Style</div>
                                            <div className="text-[11px] font-medium text-blue-300">
                                                {xrayAnalysis.bass.style?.has_slides ? 'Slides' : xrayAnalysis.bass.style?.staccato_ratio > 0.5 ? 'Staccato' : 'Sustain'}
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            )}

                            {/* Vocals */}
                            {xrayAnalysis.vocals && xrayAnalysis.vocals.total_regions > 0 && (
                                <div className="rounded-xl border border-violet-500/20 bg-gradient-to-br from-violet-950/30 to-zinc-950 p-4">
                                    <div className="flex items-center justify-between mb-3">
                                        <span className="text-sm font-bold text-violet-400">🎤 Vocals</span>
                                        <Badge variant="outline" className="text-[10px] border-violet-500/30 text-violet-300">
                                            {xrayAnalysis.vocals.total_regions} regions · {xrayAnalysis.vocals.total_duration}s
                                        </Badge>
                                    </div>
                                    <div className="flex flex-wrap gap-1.5 mb-2">
                                        {xrayAnalysis.vocals.types && Object.entries(xrayAnalysis.vocals.types).map(([type, count]: [string, any]) => (
                                            <div key={type} className="flex items-center gap-1 px-2 py-1 rounded-md bg-violet-500/10 border border-violet-500/20">
                                                <span className="text-[11px] text-violet-300 font-medium">{type}</span>
                                                <span className="text-[9px] text-violet-400/70 font-mono">{count}</span>
                                            </div>
                                        ))}
                                    </div>
                                    {xrayAnalysis.vocals.effects_detected?.length > 0 && (
                                        <div className="flex gap-1.5 mt-2">
                                            {xrayAnalysis.vocals.effects_detected.map((fx: string, i: number) => (
                                                <Badge key={i} className="bg-violet-500/15 text-violet-300 text-[9px] border-violet-500/20">
                                                    ✨ {fx}
                                                </Badge>
                                            ))}
                                        </div>
                                    )}
                                </div>
                            )}

                            {/* Other — Instruments + FX */}
                            {xrayAnalysis.other && xrayAnalysis.other.total_elements > 0 && (
                                <div className="rounded-xl border border-emerald-500/20 bg-gradient-to-br from-emerald-950/30 to-zinc-950 p-4">
                                    <div className="flex items-center justify-between mb-3">
                                        <span className="text-sm font-bold text-emerald-400">🎹 Instruments & FX</span>
                                        <Badge variant="outline" className="text-[10px] border-emerald-500/30 text-emerald-300">
                                            {xrayAnalysis.other.total_elements} elements
                                        </Badge>
                                    </div>
                                    {xrayAnalysis.other.instruments_detected?.length > 0 && (
                                        <div className="mb-2">
                                            <div className="text-[9px] text-zinc-500 uppercase tracking-wider mb-1">Instruments</div>
                                            <div className="flex flex-wrap gap-1.5">
                                                {xrayAnalysis.other.instruments_detected.map((inst: string) => {
                                                    const instIcons: Record<string, string> = {
                                                        Pad: '🎛️', Lead: '🎹', Pluck: '🪕', Piano: '🎹',
                                                        Guitar: '🎸', Strings: '🎻', Arp: '🔄', Synth: '🎛️',
                                                    };
                                                    const instCount = xrayAnalysis.other.types?.[inst] ?? 0;
                                                    return (
                                                        <div key={inst} className="flex items-center gap-1 px-2 py-1 rounded-md bg-emerald-500/10 border border-emerald-500/20">
                                                            <span className="text-xs">{instIcons[inst] || '🎵'}</span>
                                                            <span className="text-[11px] text-emerald-300 font-medium">{inst}</span>
                                                            <span className="text-[9px] text-emerald-400/60 font-mono">{instCount}</span>
                                                        </div>
                                                    );
                                                })}
                                            </div>
                                        </div>
                                    )}
                                    {xrayAnalysis.other.fx_detected?.length > 0 && (
                                        <div>
                                            <div className="text-[9px] text-zinc-500 uppercase tracking-wider mb-1">Effects</div>
                                            <div className="flex flex-wrap gap-1.5">
                                                {xrayAnalysis.other.fx_detected.map((fx: string) => {
                                                    const fxIcons: Record<string, string> = {
                                                        Riser: '📈', Sweep: '🌊', Impact: '💥',
                                                        'White Noise': '🌫️', Reverse: '⏪',
                                                    };
                                                    const fxCount = xrayAnalysis.other.types?.[fx] ?? 0;
                                                    return (
                                                        <div key={fx} className="flex items-center gap-1 px-2 py-1 rounded-md bg-orange-500/10 border border-orange-500/20">
                                                            <span className="text-xs">{fxIcons[fx] || '✨'}</span>
                                                            <span className="text-[11px] text-orange-300 font-medium">{fx}</span>
                                                            <span className="text-[9px] text-orange-400/60 font-mono">{fxCount}</span>
                                                        </div>
                                                    );
                                                })}
                                            </div>
                                        </div>
                                    )}
                                </div>
                            )}
                        </div>

                        {/* Vocal Effects Detail */}
                        {xrayAnalysis.vocal_effects?.length > 0 && (
                            <div className="rounded-lg border border-violet-500/10 bg-violet-500/5 p-3">
                                <div className="text-[9px] text-zinc-500 uppercase tracking-wider mb-2">🎙️ Vocal Effects Detected</div>
                                <div className="flex flex-wrap gap-2">
                                    {xrayAnalysis.vocal_effects.map((eff: any, i: number) => (
                                        <div key={i} className="text-[11px] text-violet-300 bg-violet-500/10 rounded-md px-2 py-1">
                                            {eff.type === 'Reverb Tail' && `🔊 Reverb at ${eff.time}s (${eff.duration}s tail)`}
                                            {eff.type === 'Delay/Echo' && `🔁 Delay: ${eff.delay_ms}ms (${Math.round(eff.strength * 100)}% strength)`}
                                        </div>
                                    ))}
                                </div>
                            </div>
                        )}
                    </CardContent>
                </Card>
            )}

            {/* 🩻 Interactive Waveform X-Ray */}
            {job?.result?.stem_analysis && (
                <WaveformXRay jobId={job.job_id} />
            )}

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

                    {/* QC Dimensions */}
                    {job?.result?.qc && (
                        <Card className="bg-zinc-900/50 border-zinc-800">
                            <CardHeader>
                                <div className="flex items-center justify-between">
                                    <div>
                                        <CardTitle className="text-lg">🔍 QC — 12 Dimensions</CardTitle>
                                        <CardDescription>
                                            Target: ≥{job.result.qc.target_score}% average score
                                        </CardDescription>
                                    </div>
                                    <div className="text-right">
                                        <div className={`text-2xl font-bold ${job.result.qc.passed ? "text-emerald-400" :
                                            job.result.qc.overall_score >= 70 ? "text-amber-400" : "text-red-400"
                                            }`}>
                                            {(job.result.qc.overall_score ?? 0).toFixed(1)}%
                                        </div>
                                        <Badge variant="outline" className={`text-[10px] ${job.result.qc.passed
                                            ? "border-emerald-500/30 text-emerald-400"
                                            : "border-red-500/30 text-red-400"
                                            }`}>
                                            {job.result.qc.passed ? "✅ PASSED" : "⚠️ Below target"}
                                        </Badge>
                                    </div>
                                </div>
                                {job.result.qc.weakest && (
                                    <div className="flex gap-4 mt-2 text-[11px]">
                                        <span className="text-red-400">Weakest: {job.result.qc.weakest.replace(/_/g, " ")}</span>
                                        <span className="text-emerald-400">Strongest: {job.result.qc.strongest?.replace(/_/g, " ")}</span>
                                    </div>
                                )}
                            </CardHeader>
                            <CardContent>
                                <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-2">
                                    {QC_DIMENSIONS.map((dim) => {
                                        const dimData = job.result?.qc?.dimensions[dim];
                                        const score = typeof dimData === "object" ? dimData.score : (dimData || 0);
                                        const detail = typeof dimData === "object" ? dimData.detail : "";
                                        const isWeakest = job.result?.qc?.weakest === dim;
                                        const isStrongest = job.result?.qc?.strongest === dim;
                                        return (
                                            <div
                                                key={dim}
                                                className={`rounded-lg p-3 transition-all ${isWeakest ? "bg-red-500/10 border border-red-500/20" :
                                                    isStrongest ? "bg-emerald-500/10 border border-emerald-500/20" :
                                                        "bg-zinc-800/50"
                                                    }`}
                                                title={detail}
                                            >
                                                <div className="flex items-center justify-between">
                                                    <div className="text-[10px] text-zinc-500 capitalize">
                                                        {dim.replace(/_/g, " ")}
                                                    </div>
                                                    <span className={`text-[11px] font-bold ${score >= 90 ? "text-emerald-400" :
                                                        score >= 70 ? "text-amber-400" :
                                                            score > 0 ? "text-red-400" : "text-zinc-600"
                                                        }`}>
                                                        {(score ?? 0).toFixed(1)}%
                                                    </span>
                                                </div>
                                                <div className="flex items-center gap-2 mt-1">
                                                    <div className="flex-1 h-1.5 bg-zinc-700 rounded-full overflow-hidden">
                                                        <div
                                                            className={`h-full rounded-full transition-all ${score >= 90 ? "bg-emerald-500" :
                                                                score >= 70 ? "bg-amber-500" :
                                                                    score > 0 ? "bg-red-500" : "bg-zinc-600"
                                                                }`}
                                                            style={{ width: `${score}%` }}
                                                        />
                                                    </div>
                                                </div>
                                                {detail && (
                                                    <div className="text-[9px] text-zinc-600 mt-1 truncate">
                                                        {detail}
                                                    </div>
                                                )}
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
