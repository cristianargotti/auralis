"use client";

import { useState, useEffect, useCallback, useRef } from "react";
import {
    Card,
    CardContent,
    CardDescription,
    CardHeader,
    CardTitle,
} from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
    listReferences,
    addReference,
    removeReference,
    getGapAnalysis,
    getReferenceAverages,
    uploadTrack,
    startAnalysis,
    getJobStatus,
    api,
    type ReferenceEntry,
    type GapReport,
} from "@/lib/api";

/* ── Types ──────────────────────────────────────────── */

interface ReconstructJob {
    job_id: string;
    status: string;
    original_name: string;
    created_at: string;
}

type UploadStage = "idle" | "uploading" | "analyzing" | "adding" | "done" | "error";

const UPLOAD_STEPS: { key: UploadStage; label: string; icon: string }[] = [
    { key: "uploading", label: "Uploading", icon: "📤" },
    { key: "analyzing", label: "Analyzing DNA", icon: "🧬" },
    { key: "adding", label: "Adding to Bank", icon: "⭐" },
    { key: "done", label: "Complete", icon: "✅" },
];

/* ── Page ───────────────────────────────────────────── */

export default function ReferenceBankPage() {
    const [references, setReferences] = useState<ReferenceEntry[]>([]);
    const [averages, setAverages] = useState<Record<string, unknown> | null>(null);
    const [jobs, setJobs] = useState<ReconstructJob[]>([]);
    const [loading, setLoading] = useState(true);
    const [adding, setAdding] = useState<string | null>(null);
    const [gapReport, setGapReport] = useState<GapReport | null>(null);
    const [gapJobId, setGapJobId] = useState<string>("");
    const [gapLoading, setGapLoading] = useState(false);
    const [message, setMessage] = useState<string | null>(null);

    // Upload state
    const fileInputRef = useRef<HTMLInputElement>(null);
    const [uploadStage, setUploadStage] = useState<UploadStage>("idle");
    const [uploadFileName, setUploadFileName] = useState("");
    const [dragOver, setDragOver] = useState(false);

    const loadData = useCallback(async () => {
        setLoading(true);
        try {
            const [refsData, jobsData] = await Promise.all([
                listReferences(),
                api<{ jobs: ReconstructJob[] }>("/api/reconstruct/jobs"),
            ]);
            setReferences(refsData.references);
            setJobs(jobsData.jobs?.filter((j) => j.status === "complete") || []);

            if (refsData.count > 0) {
                const avgs = await getReferenceAverages();
                setAverages(avgs);
            }
        } catch {
            /* ignore initial load errors */
        } finally {
            setLoading(false);
        }
    }, []);

    useEffect(() => {
        loadData();
    }, [loadData]);

    const handleAddReference = async (jobId: string, name: string) => {
        setAdding(jobId);
        try {
            const result = await addReference(jobId, name);
            setMessage(`✅ ${result.message}`);
            await loadData();
        } catch (err: unknown) {
            const msg = err instanceof Error ? err.message : "Unknown error";
            setMessage(`❌ Error: ${msg}`);
        } finally {
            setAdding(null);
            setTimeout(() => setMessage(null), 5000);
        }
    };

    const handleRemove = async (trackId: string) => {
        try {
            await removeReference(trackId);
            setMessage("🗑️ Reference removed");
            await loadData();
        } catch (err: unknown) {
            const msg = err instanceof Error ? err.message : "Unknown error";
            setMessage(`❌ ${msg}`);
        }
        setTimeout(() => setMessage(null), 3000);
    };

    const handleGapAnalysis = async () => {
        if (!gapJobId) return;
        setGapLoading(true);
        try {
            const report = await getGapAnalysis(gapJobId);
            setGapReport(report);
        } catch (err: unknown) {
            const msg = err instanceof Error ? err.message : "Unknown error";
            setMessage(`❌ Gap analysis failed: ${msg}`);
        } finally {
            setGapLoading(false);
        }
    };

    const processFile = async (file: File) => {
        setUploadFileName(file.name);
        setUploadStage("uploading");

        try {
            const up = await uploadTrack(file);
            setUploadStage("analyzing");

            const job = await startAnalysis(up.project_id);

            const poll = setInterval(async () => {
                try {
                    const status = await getJobStatus(job.job_id);
                    if (status.status === "complete") {
                        clearInterval(poll);
                        setUploadStage("adding");
                        await addReference(status.job_id, file.name);
                        setUploadStage("done");
                        loadData();
                        setTimeout(() => {
                            setUploadStage("idle");
                            setUploadFileName("");
                        }, 3000);
                    } else if (status.status === "error") {
                        clearInterval(poll);
                        throw new Error(status.message);
                    }
                } catch (err) {
                    clearInterval(poll);
                    setUploadStage("error");
                    const msg = err instanceof Error ? err.message : "Error";
                    setMessage(`❌ ${msg}`);
                    setTimeout(() => setUploadStage("idle"), 4000);
                }
            }, 2000);
        } catch (err) {
            setUploadStage("error");
            const msg = err instanceof Error ? err.message : "Upload failed";
            setMessage(`❌ ${msg}`);
            setTimeout(() => setUploadStage("idle"), 4000);
        }
    };

    const handleDrop = useCallback(async (e: React.DragEvent) => {
        e.preventDefault();
        setDragOver(false);
        const file = e.dataTransfer.files[0];
        if (file) processFile(file);
    }, []);

    const handleFileSelect = useCallback(async (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0];
        if (file) processFile(file);
    }, []);

    const alreadyInBank = new Set(references.map((r) => r.track_id));
    const availableJobs = jobs.filter((j) => !alreadyInBank.has(j.job_id));

    const getStageIndex = () => UPLOAD_STEPS.findIndex(s => s.key === uploadStage);
    const isUploading = uploadStage !== "idle";

    const getStemIcon = (name: string) => {
        const icons: Record<string, string> = { drums: "🥁", bass: "🎸", vocals: "🎤", other: "🎹" };
        return icons[name] || "🎵";
    };

    return (
        <div className="space-y-8 p-6">
            {/* Header */}
            <div className="flex items-start justify-between">
                <div>
                    <h1 className="text-3xl font-bold bg-gradient-to-r from-cyan-400 via-violet-400 to-fuchsia-400 bg-clip-text text-transparent">
                        🧬 Reference DNA Bank
                    </h1>
                    <p className="text-sm text-zinc-500 mt-2 max-w-xl">
                        Build your library of professional tracks. Auralis extracts their DNA
                        fingerprint and auto-corrects your music toward reference quality.
                    </p>
                </div>
            </div>

            {/* Status Message */}
            {message && (
                <div className="rounded-xl bg-zinc-900/50 border border-zinc-800 p-3 text-sm animate-in fade-in slide-in-from-top-2 duration-300">
                    {message}
                </div>
            )}

            {/* ══════ UPLOAD ZONE ══════ */}
            <Card className="bg-zinc-900/30 border-zinc-800/50 overflow-hidden">
                <CardContent className="p-0">
                    <input
                        type="file"
                        ref={fileInputRef}
                        className="hidden"
                        accept=".wav,.mp3,.flac,.aiff,.ogg,.m4a"
                        onChange={handleFileSelect}
                    />

                    {!isUploading ? (
                        /* ── Idle: Drag & Drop Zone ── */
                        <div
                            onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
                            onDragLeave={() => setDragOver(false)}
                            onDrop={handleDrop}
                            onClick={() => fileInputRef.current?.click()}
                            className={`relative p-10 text-center cursor-pointer transition-all duration-300 group ${dragOver
                                    ? "bg-cyan-500/10 border-b-2 border-cyan-500"
                                    : "hover:bg-zinc-800/30"
                                }`}
                        >
                            {/* Glow effect */}
                            <div className={`absolute inset-0 transition-opacity duration-500 pointer-events-none ${dragOver ? "opacity-100" : "opacity-0 group-hover:opacity-50"
                                }`} style={{
                                    background: "radial-gradient(ellipse at center, oklch(0.75 0.15 195 / 8%) 0%, transparent 70%)"
                                }} />

                            <div className="relative z-10">
                                <div className={`text-5xl mb-4 transition-transform duration-300 ${dragOver ? "scale-125 -translate-y-1" : "group-hover:scale-110"
                                    }`}>
                                    {dragOver ? "📥" : "🧬"}
                                </div>
                                <p className="text-base font-medium text-zinc-300 mb-1">
                                    Drop reference track here
                                </p>
                                <p className="text-xs text-zinc-600">
                                    WAV, MP3, FLAC, AIFF — Auralis will extract the DNA automatically
                                </p>
                                <Button
                                    variant="outline"
                                    size="sm"
                                    className="mt-4 border-zinc-700 text-zinc-400 hover:text-zinc-200 hover:border-zinc-500"
                                    onClick={(e) => { e.stopPropagation(); fileInputRef.current?.click(); }}
                                >
                                    Or browse files
                                </Button>
                            </div>
                        </div>
                    ) : (
                        /* ── Active: Pipeline Progress ── */
                        <div className="p-8">
                            {/* File name */}
                            <div className="text-center mb-6">
                                <span className="text-xs text-zinc-500 font-mono bg-zinc-800/50 px-3 py-1 rounded-full">
                                    📄 {uploadFileName}
                                </span>
                            </div>

                            {/* Step Indicators */}
                            <div className="flex items-center justify-center gap-0 max-w-md mx-auto">
                                {UPLOAD_STEPS.map((step, idx) => {
                                    const currentIdx = getStageIndex();
                                    const isActive = idx === currentIdx;
                                    const isComplete = idx < currentIdx || uploadStage === "done";
                                    const isPending = idx > currentIdx;

                                    return (
                                        <div key={step.key} className="flex items-center">
                                            {/* Step circle */}
                                            <div className={`flex flex-col items-center transition-all duration-500 ${isActive ? "scale-110" : ""
                                                }`}>
                                                <div className={`w-12 h-12 rounded-full flex items-center justify-center text-xl transition-all duration-500 ${isComplete
                                                        ? "bg-emerald-500/20 border-2 border-emerald-500/50 shadow-lg shadow-emerald-500/10"
                                                        : isActive
                                                            ? "bg-cyan-500/20 border-2 border-cyan-500/50 shadow-lg shadow-cyan-500/20 animate-pulse"
                                                            : "bg-zinc-800/50 border-2 border-zinc-700/30"
                                                    }`}>
                                                    {isComplete ? "✅" : isActive ? (
                                                        <span className="animate-bounce">{step.icon}</span>
                                                    ) : (
                                                        <span className="opacity-30">{step.icon}</span>
                                                    )}
                                                </div>
                                                <span className={`text-[10px] mt-2 font-medium transition-colors ${isComplete
                                                        ? "text-emerald-400"
                                                        : isActive
                                                            ? "text-cyan-400"
                                                            : "text-zinc-600"
                                                    }`}>
                                                    {step.label}
                                                </span>
                                            </div>

                                            {/* Connector line */}
                                            {idx < UPLOAD_STEPS.length - 1 && (
                                                <div className={`w-12 h-0.5 mx-1 mt-[-18px] rounded-full transition-all duration-700 ${isComplete
                                                        ? "bg-emerald-500/40"
                                                        : "bg-zinc-800"
                                                    }`} />
                                            )}
                                        </div>
                                    );
                                })}
                            </div>

                            {/* Stage message */}
                            <div className="text-center mt-6">
                                {uploadStage === "done" ? (
                                    <p className="text-sm text-emerald-400 font-medium animate-in fade-in">
                                        ✅ {uploadFileName} added to your DNA bank
                                    </p>
                                ) : uploadStage === "error" ? (
                                    <p className="text-sm text-red-400 font-medium animate-in fade-in">
                                        ❌ Failed — check the file and try again
                                    </p>
                                ) : (
                                    <div className="flex items-center justify-center gap-2 text-sm text-zinc-400">
                                        <div className="w-4 h-4 rounded-full border-2 border-cyan-500/30 border-t-cyan-400 animate-spin" />
                                        Processing...
                                    </div>
                                )}
                            </div>
                        </div>
                    )}
                </CardContent>
            </Card>

            {/* ══════ STATS BAR ══════ */}
            <div className="grid grid-cols-3 gap-4">
                <Card className="bg-zinc-900/30 border-zinc-800/50 hover:border-cyan-500/20 transition-colors group">
                    <CardContent className="p-5 text-center">
                        <div className="text-4xl font-bold bg-gradient-to-b from-zinc-100 to-zinc-400 bg-clip-text text-transparent group-hover:from-cyan-300 group-hover:to-cyan-500 transition-all">
                            {references.length}
                        </div>
                        <div className="text-xs text-zinc-500 mt-2 uppercase tracking-wider font-medium">
                            References
                        </div>
                        {references.length >= 3 && (
                            <div className="mt-2">
                                <Badge variant="outline" className="text-[9px] border-emerald-500/20 text-emerald-500 bg-emerald-500/5">
                                    Optimal
                                </Badge>
                            </div>
                        )}
                    </CardContent>
                </Card>

                <Card className="bg-zinc-900/30 border-zinc-800/50 hover:border-violet-500/20 transition-colors group">
                    <CardContent className="p-5 text-center">
                        <div className="text-4xl font-bold bg-gradient-to-b from-zinc-100 to-zinc-400 bg-clip-text text-transparent group-hover:from-violet-300 group-hover:to-violet-500 transition-all font-mono">
                            {averages && typeof averages === "object" && "master" in averages
                                ? `${(averages.master as { lufs?: number })?.lufs?.toFixed(1) ?? "—"}`
                                : "—"}
                        </div>
                        <div className="text-xs text-zinc-500 mt-2 uppercase tracking-wider font-medium">
                            Avg LUFS
                        </div>
                    </CardContent>
                </Card>

                <Card className="bg-zinc-900/30 border-zinc-800/50 hover:border-emerald-500/20 transition-colors group">
                    <CardContent className="p-5 text-center">
                        <div className="text-4xl font-bold bg-gradient-to-b from-zinc-100 to-zinc-400 bg-clip-text text-transparent group-hover:from-emerald-300 group-hover:to-emerald-500 transition-all font-mono">
                            {averages && typeof averages === "object" && "master" in averages
                                ? `${(averages.master as { bpm?: number })?.bpm?.toFixed(0) ?? "—"}`
                                : "—"}
                        </div>
                        <div className="text-xs text-zinc-500 mt-2 uppercase tracking-wider font-medium">
                            Avg BPM
                        </div>
                    </CardContent>
                </Card>
            </div>

            {/* ══════ ADD FROM COMPLETED JOBS ══════ */}
            {availableJobs.length > 0 && (
                <Card className="bg-zinc-900/30 border-zinc-800/50">
                    <CardHeader className="pb-3">
                        <CardTitle className="text-sm flex items-center gap-2">
                            <span className="w-2 h-2 rounded-full bg-amber-500 animate-pulse" />
                            Ready to Add
                        </CardTitle>
                        <CardDescription>
                            These processed tracks can be added to your reference bank
                        </CardDescription>
                    </CardHeader>
                    <CardContent className="space-y-2">
                        {availableJobs.map((job) => (
                            <div
                                key={job.job_id}
                                className="flex items-center justify-between rounded-xl border border-zinc-800/50 bg-zinc-900/30 p-3 hover:border-amber-500/20 transition-colors group"
                            >
                                <div className="flex items-center gap-3">
                                    <div className="w-9 h-9 rounded-lg bg-amber-500/10 flex items-center justify-center text-lg">
                                        🎵
                                    </div>
                                    <div>
                                        <div className="text-sm font-medium text-zinc-300">
                                            {job.original_name || job.job_id.slice(0, 8)}
                                        </div>
                                        <div className="text-[10px] text-zinc-600 font-mono">
                                            {job.job_id.slice(0, 16)}…
                                        </div>
                                    </div>
                                </div>
                                <Button
                                    size="sm"
                                    onClick={() => handleAddReference(job.job_id, job.original_name || "")}
                                    disabled={adding === job.job_id}
                                    className="bg-amber-500/10 text-amber-400 hover:bg-amber-500/20 border border-amber-500/20 hover:border-amber-500/40 transition-all"
                                >
                                    {adding === job.job_id ? (
                                        <>
                                            <span className="w-3 h-3 mr-1 rounded-full border-2 border-amber-500/30 border-t-amber-400 animate-spin" />
                                            Adding…
                                        </>
                                    ) : (
                                        "⭐ Add to Bank"
                                    )}
                                </Button>
                            </div>
                        ))}
                    </CardContent>
                </Card>
            )}

            {/* ══════ REFERENCE LIBRARY ══════ */}
            <Card className="bg-zinc-900/30 border-zinc-800/50">
                <CardHeader className="pb-3">
                    <div className="flex items-center justify-between">
                        <div>
                            <CardTitle className="text-base flex items-center gap-2">
                                📚 Reference Library
                                {references.length > 0 && (
                                    <Badge variant="outline" className="text-[10px] border-zinc-700 text-zinc-500 font-mono">
                                        {references.length}
                                    </Badge>
                                )}
                            </CardTitle>
                            <CardDescription className="mt-1">
                                Professional tracks used as quality benchmarks
                            </CardDescription>
                        </div>
                    </div>
                </CardHeader>
                <CardContent>
                    {loading ? (
                        <div className="space-y-3">
                            {[1, 2, 3].map((i) => (
                                <div key={i} className="h-16 bg-zinc-800/30 rounded-xl animate-pulse" style={{ animationDelay: `${i * 150}ms` }} />
                            ))}
                        </div>
                    ) : references.length === 0 ? (
                        /* ── Empty State ── */
                        <div className="text-center py-12">
                            <div className="relative inline-block mb-4">
                                <div className="text-6xl animate-pulse-slow">🧬</div>
                                <div className="absolute -inset-4 rounded-full" style={{
                                    background: "radial-gradient(circle, oklch(0.75 0.15 195 / 10%) 0%, transparent 70%)"
                                }} />
                            </div>
                            <p className="text-zinc-400 text-sm font-medium mb-1">
                                No references yet
                            </p>
                            <p className="text-zinc-600 text-xs max-w-xs mx-auto">
                                Drop a professional track above to start building your DNA bank.
                                We recommend 3–5 references for best results.
                            </p>
                            <div className="flex items-center justify-center gap-1 mt-4">
                                {[1, 2, 3, 4, 5].map((i) => (
                                    <div key={i} className={`w-2 h-2 rounded-full ${i <= references.length ? "bg-cyan-500" : "bg-zinc-800 border border-zinc-700"
                                        }`} />
                                ))}
                                <span className="text-[10px] text-zinc-600 ml-2">0 / 5 recommended</span>
                            </div>
                        </div>
                    ) : (
                        /* ── Reference Cards ── */
                        <div className="space-y-2">
                            {references.map((ref, idx) => (
                                <div
                                    key={ref.track_id}
                                    className="flex items-center justify-between rounded-xl border border-zinc-800/50 bg-zinc-900/20 p-4 group hover:border-cyan-500/15 transition-all duration-300"
                                    style={{ animationDelay: `${idx * 80}ms` }}
                                >
                                    <div className="flex items-center gap-4 min-w-0">
                                        {/* Track icon with glow */}
                                        <div className="relative">
                                            <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-cyan-500/20 to-violet-500/20 border border-cyan-500/10 flex items-center justify-center text-lg group-hover:shadow-lg group-hover:shadow-cyan-500/10 transition-shadow">
                                                🎵
                                            </div>
                                            <div className="absolute -bottom-0.5 -right-0.5 w-3 h-3 rounded-full bg-emerald-500 border-2 border-zinc-900 flex items-center justify-center">
                                                <span className="text-[6px]">✓</span>
                                            </div>
                                        </div>

                                        <div className="min-w-0">
                                            <div className="text-sm font-medium text-zinc-200 truncate group-hover:text-zinc-100 transition-colors">
                                                {ref.name}
                                            </div>
                                            <div className="flex gap-2 mt-1.5 flex-wrap">
                                                <Badge
                                                    variant="outline"
                                                    className="text-[10px] border-cyan-500/20 text-cyan-500 bg-cyan-500/5 font-mono"
                                                >
                                                    {ref.bpm} BPM
                                                </Badge>
                                                <Badge
                                                    variant="outline"
                                                    className="text-[10px] border-violet-500/20 text-violet-400 bg-violet-500/5"
                                                >
                                                    {ref.key}
                                                </Badge>
                                                <Badge
                                                    variant="outline"
                                                    className="text-[10px] border-emerald-500/20 text-emerald-400 bg-emerald-500/5 font-mono"
                                                >
                                                    {ref.lufs.toFixed(1)} LUFS
                                                </Badge>
                                            </div>
                                        </div>
                                    </div>

                                    <Button
                                        size="sm"
                                        variant="ghost"
                                        onClick={() => handleRemove(ref.track_id)}
                                        className="opacity-0 group-hover:opacity-100 transition-all text-zinc-600 hover:text-red-400 hover:bg-red-500/10"
                                    >
                                        ✕
                                    </Button>
                                </div>
                            ))}

                            {/* Progress indicator */}
                            <div className="flex items-center gap-2 pt-3 px-1">
                                <div className="flex gap-1">
                                    {[1, 2, 3, 4, 5].map((i) => (
                                        <div key={i} className={`w-2 h-2 rounded-full transition-colors ${i <= references.length ? "bg-cyan-500" : "bg-zinc-800 border border-zinc-700"
                                            }`} />
                                    ))}
                                </div>
                                <span className="text-[10px] text-zinc-600">
                                    {references.length} / 5 recommended
                                    {references.length >= 3 && references.length < 5 && " • Good coverage"}
                                    {references.length >= 5 && " • Excellent coverage"}
                                </span>
                            </div>
                        </div>
                    )}
                </CardContent>
            </Card>

            {/* ══════ GAP ANALYSIS ══════ */}
            <Card className="bg-zinc-900/30 border-zinc-800/50">
                <CardHeader className="pb-3">
                    <CardTitle className="text-base flex items-center gap-2">
                        🔍 Gap Analysis
                    </CardTitle>
                    <CardDescription>
                        Compare any processed track against your reference bank
                    </CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                    <div className="flex gap-2">
                        <select
                            value={gapJobId}
                            onChange={(e) => setGapJobId(e.target.value)}
                            className="flex-1 rounded-xl border border-zinc-800/50 bg-zinc-900/50 px-4 py-2.5 text-sm text-zinc-300 focus:outline-none focus:border-cyan-500/30 transition-colors appearance-none"
                        >
                            <option value="">Select a track…</option>
                            {jobs.map((j) => (
                                <option key={j.job_id} value={j.job_id}>
                                    {j.original_name || j.job_id.slice(0, 12)}
                                </option>
                            ))}
                        </select>
                        <Button
                            onClick={handleGapAnalysis}
                            disabled={!gapJobId || gapLoading || references.length === 0}
                            className="bg-gradient-to-r from-cyan-600/20 to-violet-600/20 border border-cyan-500/20 text-cyan-400 hover:text-cyan-300 hover:border-cyan-500/40 transition-all disabled:opacity-30"
                        >
                            {gapLoading ? (
                                <>
                                    <span className="w-3 h-3 mr-2 rounded-full border-2 border-cyan-500/30 border-t-cyan-400 animate-spin" />
                                    Analyzing…
                                </>
                            ) : (
                                "Analyze Gap"
                            )}
                        </Button>
                    </div>

                    {references.length === 0 && (
                        <p className="text-xs text-zinc-600 bg-zinc-800/30 rounded-lg px-3 py-2">
                            💡 Add at least one reference track to enable gap analysis.
                        </p>
                    )}

                    {/* Gap Report */}
                    {gapReport && (
                        <div className="space-y-4 pt-4 border-t border-zinc-800/30">
                            {/* Score */}
                            <div className="text-center py-4">
                                <div className={`text-6xl font-bold font-mono ${gapReport.overall_score >= 80
                                        ? "bg-gradient-to-b from-emerald-300 to-emerald-500"
                                        : gapReport.overall_score >= 60
                                            ? "bg-gradient-to-b from-cyan-300 to-cyan-500"
                                            : "bg-gradient-to-b from-red-300 to-red-500"
                                    } bg-clip-text text-transparent`}>
                                    {gapReport.overall_score.toFixed(0)}
                                </div>
                                <div className="text-xs text-zinc-500 mt-2">
                                    Overall Score vs {gapReport.reference_count} references
                                </div>
                                <div className="w-full max-w-sm mx-auto bg-zinc-800/50 rounded-full h-2 mt-4 overflow-hidden">
                                    <div
                                        className="h-full rounded-full transition-all duration-1000 ease-out"
                                        style={{
                                            width: `${gapReport.overall_score}%`,
                                            background: gapReport.overall_score >= 80
                                                ? "linear-gradient(to right, oklch(0.7 0.17 160), oklch(0.8 0.15 160))"
                                                : gapReport.overall_score >= 60
                                                    ? "linear-gradient(to right, oklch(0.65 0.15 195), oklch(0.75 0.15 195))"
                                                    : "linear-gradient(to right, #dc2626, #ef4444)",
                                        }}
                                    />
                                </div>
                            </div>

                            {/* LUFS Gap */}
                            <div className="flex justify-between items-center rounded-xl border border-zinc-800/30 bg-zinc-900/20 p-4">
                                <div className="flex items-center gap-3">
                                    <div className="w-8 h-8 rounded-lg bg-violet-500/10 flex items-center justify-center">💎</div>
                                    <span className="text-sm text-zinc-300">Master LUFS</span>
                                </div>
                                <div className="flex items-center gap-2">
                                    <span className="text-sm font-mono text-zinc-400">
                                        {gapReport.your_lufs.toFixed(1)}
                                    </span>
                                    <span className="text-zinc-600">→</span>
                                    <span className="text-sm font-mono text-zinc-300">
                                        {gapReport.ref_lufs.toFixed(1)}
                                    </span>
                                    <Badge
                                        variant="outline"
                                        className={`text-[10px] font-mono ${Math.abs(gapReport.lufs_gap) < 2
                                                ? "border-emerald-500/20 text-emerald-400 bg-emerald-500/5"
                                                : "border-red-500/20 text-red-400 bg-red-500/5"
                                            }`}
                                    >
                                        {gapReport.lufs_gap > 0 ? "+" : ""}
                                        {gapReport.lufs_gap.toFixed(1)} dB
                                    </Badge>
                                </div>
                            </div>

                            {/* Per-Stem Gaps */}
                            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                                {Object.entries(gapReport.stem_gaps).map(([name, gap]) => (
                                    <div
                                        key={name}
                                        className="rounded-xl border border-zinc-800/30 bg-zinc-900/20 p-4"
                                    >
                                        <div className="flex justify-between items-center mb-3">
                                            <span className="text-sm font-medium flex items-center gap-2">
                                                {getStemIcon(name)} <span className="capitalize">{name}</span>
                                            </span>
                                            <Badge
                                                variant="outline"
                                                className={`text-[10px] font-mono ${gap.quality_score >= 80
                                                        ? "border-emerald-500/20 text-emerald-400 bg-emerald-500/5"
                                                        : gap.quality_score >= 60
                                                            ? "border-cyan-500/20 text-cyan-400 bg-cyan-500/5"
                                                            : "border-red-500/20 text-red-400 bg-red-500/5"
                                                    }`}
                                            >
                                                {gap.quality_score.toFixed(0)}/100
                                            </Badge>
                                        </div>
                                        <div className="w-full bg-zinc-800/50 rounded-full h-1.5 mb-3 overflow-hidden">
                                            <div
                                                className="h-full rounded-full transition-all duration-700"
                                                style={{
                                                    width: `${gap.quality_score}%`,
                                                    background: gap.quality_score >= 80
                                                        ? "oklch(0.7 0.17 160)"
                                                        : gap.quality_score >= 60
                                                            ? "oklch(0.75 0.15 195)"
                                                            : "#ef4444",
                                                }}
                                            />
                                        </div>
                                        {gap.suggestions.length > 0 && (
                                            <div className="space-y-1">
                                                {gap.suggestions.map((s, i) => (
                                                    <div key={i} className="text-[11px] text-zinc-500 pl-2 border-l-2 border-zinc-800">
                                                        {s}
                                                    </div>
                                                ))}
                                            </div>
                                        )}
                                    </div>
                                ))}
                            </div>

                            {/* Top Improvements */}
                            {gapReport.top_improvements.length > 0 && (
                                <div className="rounded-xl border border-cyan-500/10 bg-cyan-500/5 p-4">
                                    <div className="text-sm font-medium mb-3 flex items-center gap-2">
                                        🎯 Priority Improvements
                                    </div>
                                    <div className="space-y-2">
                                        {gapReport.top_improvements.map((imp, i) => (
                                            <div key={i} className="flex items-start gap-2 text-xs text-zinc-400">
                                                <span className="text-cyan-500 font-bold min-w-[16px]">{i + 1}.</span>
                                                <span>{imp}</span>
                                            </div>
                                        ))}
                                    </div>
                                </div>
                            )}
                        </div>
                    )}
                </CardContent>
            </Card>
        </div>
    );
}
