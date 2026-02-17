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
    const [uploadingRef, setUploadingRef] = useState(false);
    const [uploadProgress, setUploadProgress] = useState(0);

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

    const handleUploadReference = async (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0];
        if (!file) return;

        setUploadingRef(true);
        setUploadProgress(0);
        setMessage("⏳ Uploading reference...");

        try {
            // 1. Upload
            const up = await uploadTrack(file);
            setUploadProgress(25);
            setMessage("⏳ Analyzing structure...");

            // 2. Analyze
            const job = await startAnalysis(up.project_id);
            setUploadProgress(50);

            // 3. Poll for completion
            const poll = setInterval(async () => {
                try {
                    const status = await getJobStatus(job.job_id);
                    if (status.status === "complete") {
                        clearInterval(poll);
                        setUploadProgress(90);
                        setMessage("⏳ Adding to bank...");

                        // 4. Add to bank
                        await addReference(status.job_id, file.name);
                        setMessage(`✅ Added ${file.name} to references`);
                        setUploadingRef(false);
                        loadData();
                    } else if (status.status === "error") {
                        clearInterval(poll);
                        throw new Error(status.message);
                    }
                } catch (err) {
                    clearInterval(poll);
                    setUploadingRef(false);
                    const msg = err instanceof Error ? err.message : "Polling error";
                    setMessage(`❌ ${msg}`);
                }
            }, 1000);
        } catch (err) {
            setUploadingRef(false);
            const msg = err instanceof Error ? err.message : "Upload failed";
            setMessage(`❌ ${msg}`);
        }
    };

    const alreadyInBank = new Set(references.map((r) => r.track_id));
    const availableJobs = jobs.filter((j) => !alreadyInBank.has(j.job_id));

    return (
        <div className="space-y-6">
            {/* Header */}
            <div>
                <h1 className="text-2xl font-bold text-gradient">🧬 Reference DNA Bank</h1>
                <p className="text-sm text-muted-foreground mt-1">
                    Build your library of professional tracks. Auralis compares
                    your music against these DNA fingerprints to auto-correct
                    toward reference quality.
                </p>
            </div>

            {/* Direct Upload Action */}
            <div className="flex justify-end">
                <Button
                    onClick={() => fileInputRef.current?.click()}
                    disabled={uploadingRef}
                    className="bg-gradient-auralis shadow-lg shadow-cyan-500/20"
                >
                    {uploadingRef ? (
                        <>
                            <span className="w-4 h-4 mr-2 rounded-full border-2 border-white/30 border-t-white animate-spin"></span>
                            Processing... {uploadProgress}%
                        </>
                    ) : (
                        <>
                            <span className="mr-2">📤</span> Upload Reference
                        </>
                    )}
                </Button>
            </div>

            {/* Status Message */}
            {message && (
                <div className="glass rounded-lg p-3 text-sm animate-in fade-in">
                    {message}
                </div>
            )}

            {/* Stats Bar */}
            <div className="grid grid-cols-3 gap-4">
                <Card className="glass-strong border-border/30">
                    <CardContent className="p-4 text-center">
                        <div className="text-3xl font-bold text-gradient">
                            {references.length}
                        </div>
                        <div className="text-xs text-muted-foreground mt-1">
                            References
                        </div>
                    </CardContent>
                </Card>

                {/* Upload Button overlay */}
                <input
                    type="file"
                    ref={fileInputRef}
                    className="hidden"
                    accept=".wav,.mp3,.flac,.aiff"
                    onChange={handleUploadReference}
                />

                <Card className="glass-strong border-border/30">
                    <CardContent className="p-4 text-center">
                        <div className="text-3xl font-bold text-auralis-cyan">
                            {averages && typeof averages === "object" && "master" in averages
                                ? `${(averages.master as { lufs?: number })?.lufs?.toFixed(1) ?? "—"}`
                                : "—"}
                        </div>
                        <div className="text-xs text-muted-foreground mt-1">
                            Avg LUFS
                        </div>
                    </CardContent>
                </Card>
                <Card className="glass-strong border-border/30">
                    <CardContent className="p-4 text-center">
                        <div className="text-3xl font-bold text-auralis-emerald">
                            {averages && typeof averages === "object" && "master" in averages
                                ? `${(averages.master as { bpm?: number })?.bpm?.toFixed(0) ?? "—"}`
                                : "—"}
                        </div>
                        <div className="text-xs text-muted-foreground mt-1">
                            Avg BPM
                        </div>
                    </CardContent>
                </Card>
            </div>

            {/* Add to Bank */}
            {availableJobs.length > 0 && (
                <Card className="glass-strong border-border/30">
                    <CardHeader className="pb-3">
                        <CardTitle className="text-base">
                            ⭐ Add to Reference Bank
                        </CardTitle>
                        <CardDescription>
                            Completed tracks ready to add as references
                        </CardDescription>
                    </CardHeader>
                    <CardContent className="space-y-2">
                        {availableJobs.map((job) => (
                            <div
                                key={job.job_id}
                                className="flex items-center justify-between rounded-lg border border-border/20 bg-background/30 p-3"
                            >
                                <div className="flex items-center gap-3">
                                    <span className="text-lg">🎵</span>
                                    <div>
                                        <div className="text-sm font-medium">
                                            {job.original_name || job.job_id.slice(0, 8)}
                                        </div>
                                        <div className="text-xs text-muted-foreground">
                                            {job.job_id.slice(0, 12)}...
                                        </div>
                                    </div>
                                </div>
                                <Button
                                    size="sm"
                                    onClick={() =>
                                        handleAddReference(
                                            job.job_id,
                                            job.original_name || ""
                                        )
                                    }
                                    disabled={adding === job.job_id}
                                    className="bg-auralis-cyan/20 text-auralis-cyan hover:bg-auralis-cyan/30 border border-auralis-cyan/30"
                                >
                                    {adding === job.job_id
                                        ? "Adding..."
                                        : "⭐ Add to Bank"}
                                </Button>
                            </div>
                        ))}
                    </CardContent>
                </Card>
            )}

            {/* Reference Library */}
            <Card className="glass-strong border-border/30">
                <CardHeader className="pb-3">
                    <CardTitle className="text-base">
                        📚 Reference Library ({references.length})
                    </CardTitle>
                    <CardDescription>
                        Professional tracks used as quality benchmarks
                    </CardDescription>
                </CardHeader>
                <CardContent>
                    {loading ? (
                        <div className="text-center py-8 text-muted-foreground">
                            Loading...
                        </div>
                    ) : references.length === 0 ? (
                        <div className="text-center py-8">
                            <div className="text-4xl mb-3">🧬</div>
                            <p className="text-muted-foreground text-sm">
                                No references yet. Process a track through
                                Reconstruct, then add it here.
                            </p>
                        </div>
                    ) : (
                        <div className="space-y-2">
                            {references.map((ref) => (
                                <div
                                    key={ref.track_id}
                                    className="flex items-center justify-between rounded-lg border border-border/20 bg-background/30 p-3 group"
                                >
                                    <div className="flex items-center gap-3 min-w-0">
                                        <span className="text-lg">🎵</span>
                                        <div className="min-w-0">
                                            <div className="text-sm font-medium truncate">
                                                {ref.name}
                                            </div>
                                            <div className="flex gap-2 mt-1">
                                                <Badge
                                                    variant="outline"
                                                    className="text-[10px] border-auralis-cyan/30 text-auralis-cyan"
                                                >
                                                    {ref.bpm} BPM
                                                </Badge>
                                                <Badge
                                                    variant="outline"
                                                    className="text-[10px] border-auralis-purple/30 text-auralis-purple"
                                                >
                                                    {ref.key}
                                                </Badge>
                                                <Badge
                                                    variant="outline"
                                                    className="text-[10px] border-auralis-emerald/30 text-auralis-emerald"
                                                >
                                                    {ref.lufs.toFixed(1)} LUFS
                                                </Badge>
                                            </div>
                                        </div>
                                    </div>
                                    <Button
                                        size="sm"
                                        variant="ghost"
                                        onClick={() =>
                                            handleRemove(ref.track_id)
                                        }
                                        className="opacity-0 group-hover:opacity-100 transition-opacity text-destructive hover:text-destructive"
                                    >
                                        ✕
                                    </Button>
                                </div>
                            ))}
                        </div>
                    )}
                </CardContent>
            </Card>

            {/* Gap Analysis */}
            <Card className="glass-strong border-border/30">
                <CardHeader className="pb-3">
                    <CardTitle className="text-base">
                        🔍 Gap Analysis
                    </CardTitle>
                    <CardDescription>
                        Compare any processed track against your reference
                        bank
                    </CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                    <div className="flex gap-2">
                        <select
                            value={gapJobId}
                            onChange={(e) => setGapJobId(e.target.value)}
                            className="flex-1 rounded-lg border border-border/30 bg-background/50 px-3 py-2 text-sm"
                        >
                            <option value="">Select a track...</option>
                            {jobs.map((j) => (
                                <option key={j.job_id} value={j.job_id}>
                                    {j.original_name || j.job_id.slice(0, 12)}
                                </option>
                            ))}
                        </select>
                        <Button
                            onClick={handleGapAnalysis}
                            disabled={
                                !gapJobId ||
                                gapLoading ||
                                references.length === 0
                            }
                            className="bg-gradient-auralis"
                        >
                            {gapLoading ? "Analyzing..." : "Analyze Gap"}
                        </Button>
                    </div>

                    {references.length === 0 && (
                        <p className="text-xs text-muted-foreground">
                            Add at least one reference track to enable gap
                            analysis.
                        </p>
                    )}

                    {/* Gap Report Display */}
                    {gapReport && (
                        <div className="space-y-4 pt-2">
                            {/* Score */}
                            <div className="text-center">
                                <div className="text-5xl font-bold text-gradient">
                                    {gapReport.overall_score.toFixed(0)}
                                </div>
                                <div className="text-sm text-muted-foreground mt-1">
                                    Overall Score vs {gapReport.reference_count}{" "}
                                    references
                                </div>
                                <div className="w-full bg-background/50 rounded-full h-3 mt-3">
                                    <div
                                        className="h-3 rounded-full transition-all duration-700"
                                        style={{
                                            width: `${gapReport.overall_score}%`,
                                            background:
                                                gapReport.overall_score >= 80
                                                    ? "var(--auralis-emerald)"
                                                    : gapReport.overall_score >=
                                                        60
                                                        ? "var(--auralis-cyan)"
                                                        : "#ef4444",
                                        }}
                                    />
                                </div>
                            </div>

                            {/* LUFS Gap */}
                            <div className="flex justify-between items-center rounded-lg border border-border/20 bg-background/30 p-3">
                                <span className="text-sm">💎 Master LUFS</span>
                                <div className="text-right">
                                    <span className="text-sm font-mono">
                                        {gapReport.your_lufs.toFixed(1)} →{" "}
                                        {gapReport.ref_lufs.toFixed(1)}
                                    </span>
                                    <Badge
                                        variant="outline"
                                        className={`ml-2 text-[10px] ${Math.abs(gapReport.lufs_gap) < 2
                                            ? "border-auralis-emerald/30 text-auralis-emerald"
                                            : "border-destructive/30 text-destructive"
                                            }`}
                                    >
                                        {gapReport.lufs_gap > 0 ? "+" : ""}
                                        {gapReport.lufs_gap.toFixed(1)} LUFS
                                    </Badge>
                                </div>
                            </div>

                            {/* Per-Stem Gaps */}
                            {Object.entries(gapReport.stem_gaps).map(
                                ([name, gap]) => {
                                    const emoji =
                                        name === "drums"
                                            ? "🥁"
                                            : name === "bass"
                                                ? "🎸"
                                                : name === "vocals"
                                                    ? "🎤"
                                                    : "🎹";
                                    return (
                                        <div
                                            key={name}
                                            className="rounded-lg border border-border/20 bg-background/30 p-3"
                                        >
                                            <div className="flex justify-between items-center mb-2">
                                                <span className="text-sm font-medium">
                                                    {emoji} {name}
                                                </span>
                                                <Badge
                                                    variant="outline"
                                                    className={`text-[10px] ${gap.quality_score >= 80
                                                        ? "border-auralis-emerald/30 text-auralis-emerald"
                                                        : gap.quality_score >=
                                                            60
                                                            ? "border-auralis-cyan/30 text-auralis-cyan"
                                                            : "border-destructive/30 text-destructive"
                                                        }`}
                                                >
                                                    {gap.quality_score.toFixed(0)}
                                                    /100
                                                </Badge>
                                            </div>
                                            <div className="w-full bg-background/50 rounded-full h-2 mb-2">
                                                <div
                                                    className="h-2 rounded-full transition-all duration-500"
                                                    style={{
                                                        width: `${gap.quality_score}%`,
                                                        background:
                                                            gap.quality_score >=
                                                                80
                                                                ? "var(--auralis-emerald)"
                                                                : gap.quality_score >=
                                                                    60
                                                                    ? "var(--auralis-cyan)"
                                                                    : "#ef4444",
                                                    }}
                                                />
                                            </div>
                                            {gap.suggestions.length > 0 && (
                                                <div className="space-y-1">
                                                    {gap.suggestions.map(
                                                        (s, i) => (
                                                            <div
                                                                key={i}
                                                                className="text-xs text-muted-foreground pl-2 border-l border-border/30"
                                                            >
                                                                → {s}
                                                            </div>
                                                        )
                                                    )}
                                                </div>
                                            )}
                                        </div>
                                    );
                                }
                            )}

                            {/* Top Improvements */}
                            {gapReport.top_improvements.length > 0 && (
                                <div className="rounded-lg border border-auralis-cyan/20 bg-auralis-cyan/5 p-3">
                                    <div className="text-sm font-medium mb-2">
                                        🎯 Top Improvements
                                    </div>
                                    {gapReport.top_improvements.map((imp, i) => (
                                        <div
                                            key={i}
                                            className="text-xs text-muted-foreground py-1"
                                        >
                                            {i + 1}. {imp}
                                        </div>
                                    ))}
                                </div>
                            )}
                        </div>
                    )}
                </CardContent>
            </Card>
        </div>
    );
}
