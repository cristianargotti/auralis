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
                                    ✅ Reconstruction complete — {job.result.rendered_stems} stems rendered
                                    {elapsed > 0 && (
                                        <span className="text-emerald-500/60 font-normal ml-2">in {formatElapsed(elapsed)}</span>
                                    )}
                                </p>
                                {job.result.analysis && (
                                    <div className="flex flex-wrap gap-3 justify-center mt-3">
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
                                        {job.result.master?.est_lufs && (
                                            <Badge variant="outline" className="border-orange-500/30 text-orange-400">
                                                Master: {job.result.master.est_lufs.toFixed(1)} LUFS
                                            </Badge>
                                        )}
                                        {job.result.qc && (
                                            <Badge variant="outline" className={`${job.result.qc.passed
                                                ? "border-emerald-500/30 text-emerald-400"
                                                : "border-amber-500/30 text-amber-400"
                                                }`}>
                                                QC: {job.result.qc.overall_score.toFixed(1)}%
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
                                            {job.result.qc.overall_score.toFixed(1)}%
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
                                                        {score.toFixed(1)}%
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
