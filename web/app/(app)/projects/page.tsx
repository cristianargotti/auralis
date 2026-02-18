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
import { apiFetch } from "@/lib/api";
import Link from "next/link";

/* ── Types ──────────────────────────────────────────── */

interface ProjectJob {
    job_id: string;
    project_id: string;
    status: "running" | "completed" | "error";
    stage: string;
    progress: number;
    cleaned: boolean;
    disk_size_mb: number;
    file_count: number;
    created_at: number | null;
    has_master: boolean;
    has_mix: boolean;
    has_original: boolean;
    has_stems: boolean;
    has_brain: boolean;
    has_stem_analysis: boolean;
    bpm: number | null;
    key: string | null;
    scale: string | null;
    duration: number | null;
    qc_score: number | null;
    qc_passed: boolean | null;
}

interface ProjectStats {
    total_projects: number;
    total_disk_gb: number;
    total_disk_mb: number;
    total_jobs: number;
    completed: number;
    running: number;
    errored: number;
    with_intelligence: number;
}

type SortKey = "created_at" | "disk_size_mb" | "status" | "qc_score" | "project_id";

/* ── Component ──────────────────────────────────────── */

export default function ProjectsPage() {
    const [jobs, setJobs] = useState<ProjectJob[]>([]);
    const [stats, setStats] = useState<ProjectStats | null>(null);
    const [loading, setLoading] = useState(true);
    const [sortKey, setSortKey] = useState<SortKey>("created_at");
    const [sortAsc, setSortAsc] = useState(false);
    const [selected, setSelected] = useState<Set<string>>(new Set());
    const [deleting, setDeleting] = useState<Set<string>>(new Set());
    const [downloading, setDownloading] = useState<Set<string>>(new Set());
    const [confirmDelete, setConfirmDelete] = useState<string | null>(null);
    const [confirmBulk, setConfirmBulk] = useState(false);

    const fetchData = useCallback(async () => {
        try {
            const [jobsRes, statsRes] = await Promise.all([
                apiFetch("/api/reconstruct/jobs"),
                apiFetch("/api/reconstruct/projects/stats"),
            ]);
            const jobsData = await jobsRes.json();
            const statsData = await statsRes.json();
            setJobs(jobsData);
            setStats(statsData);
        } catch (e) {
            console.error("Failed to load projects", e);
        } finally {
            setLoading(false);
        }
    }, []);

    useEffect(() => {
        fetchData();
    }, [fetchData]);

    /* ── Sort ──────────────────────────────────────────── */

    const sortedJobs = [...jobs].sort((a, b) => {
        let cmp = 0;
        switch (sortKey) {
            case "created_at":
                cmp = (a.created_at ?? 0) - (b.created_at ?? 0);
                break;
            case "disk_size_mb":
                cmp = a.disk_size_mb - b.disk_size_mb;
                break;
            case "status":
                cmp = a.status.localeCompare(b.status);
                break;
            case "qc_score":
                cmp = (a.qc_score ?? 0) - (b.qc_score ?? 0);
                break;
            case "project_id":
                cmp = a.project_id.localeCompare(b.project_id);
                break;
        }
        return sortAsc ? cmp : -cmp;
    });

    const handleSort = (key: SortKey) => {
        if (sortKey === key) {
            setSortAsc(!sortAsc);
        } else {
            setSortKey(key);
            setSortAsc(false);
        }
    };

    const sortIcon = (key: SortKey) =>
        sortKey === key ? (sortAsc ? " ↑" : " ↓") : "";

    /* ── Actions ───────────────────────────────────────── */

    const handleDelete = async (jobId: string) => {
        setDeleting((prev) => new Set(prev).add(jobId));
        setConfirmDelete(null);
        try {
            await apiFetch(`/api/reconstruct/cleanup/${jobId}`, {
                method: "DELETE",
            });
            await fetchData();
        } catch (e) {
            console.error("Delete failed", e);
        } finally {
            setDeleting((prev) => {
                const next = new Set(prev);
                next.delete(jobId);
                return next;
            });
        }
    };

    const handleBulkDelete = async () => {
        setConfirmBulk(false);
        for (const jobId of selected) {
            await handleDelete(jobId);
        }
        setSelected(new Set());
    };

    const handleDownload = async (jobId: string, fileKey: string, filename: string) => {
        setDownloading((prev) => new Set(prev).add(jobId));
        try {
            const res = await apiFetch(
                `/api/reconstruct/audio/${jobId}/${fileKey}`
            );
            const blob = await res.blob();
            const url = URL.createObjectURL(blob);
            const a = document.createElement("a");
            a.href = url;
            a.download = filename;
            a.click();
            URL.revokeObjectURL(url);
        } catch (e) {
            console.error("Download failed", e);
        } finally {
            setDownloading((prev) => {
                const next = new Set(prev);
                next.delete(jobId);
                return next;
            });
        }
    };

    const toggleSelect = (jobId: string) => {
        setSelected((prev) => {
            const next = new Set(prev);
            if (next.has(jobId)) next.delete(jobId);
            else next.add(jobId);
            return next;
        });
    };

    const toggleSelectAll = () => {
        if (selected.size === jobs.length) {
            setSelected(new Set());
        } else {
            setSelected(new Set(jobs.map((j) => j.job_id)));
        }
    };

    /* ── Helpers ────────────────────────────────────────── */

    const formatDate = (ts: number | null) => {
        if (!ts) return "—";
        const d = new Date(ts * 1000);
        return d.toLocaleDateString("en-US", {
            month: "short",
            day: "numeric",
            hour: "2-digit",
            minute: "2-digit",
        });
    };

    const formatDuration = (secs: number | null) => {
        if (!secs) return "—";
        const m = Math.floor(secs / 60);
        const s = Math.round(secs % 60);
        return `${m}:${s.toString().padStart(2, "0")}`;
    };

    const statusBadge = (status: string, cleaned: boolean) => {
        if (cleaned)
            return (
                <Badge
                    variant="outline"
                    className="border-zinc-600/30 text-zinc-500 text-[9px]"
                >
                    🧹 Cleaned
                </Badge>
            );
        switch (status) {
            case "completed":
                return (
                    <Badge
                        variant="outline"
                        className="border-emerald-500/30 text-emerald-400 text-[9px]"
                    >
                        ✅ Complete
                    </Badge>
                );
            case "running":
                return (
                    <Badge
                        variant="outline"
                        className="border-amber-500/30 text-amber-400 text-[9px]"
                    >
                        ⚡ Running
                    </Badge>
                );
            case "error":
                return (
                    <Badge
                        variant="outline"
                        className="border-red-500/30 text-red-400 text-[9px]"
                    >
                        ❌ Error
                    </Badge>
                );
            default:
                return null;
        }
    };

    /* ── Render ─────────────────────────────────────────── */

    if (loading) {
        return (
            <div className="space-y-4">
                <div className="h-8 w-48 bg-zinc-800/50 rounded-lg animate-pulse" />
                <div className="h-24 bg-zinc-800/30 rounded-xl animate-pulse" />
                <div className="h-96 bg-zinc-800/20 rounded-xl animate-pulse" />
            </div>
        );
    }

    return (
        <div className="space-y-6">
            {/* Header */}
            <div className="flex items-center justify-between">
                <div>
                    <h1 className="text-2xl font-bold text-zinc-100 flex items-center gap-3">
                        <span className="text-3xl">📁</span> Projects
                    </h1>
                    <p className="text-sm text-zinc-500 mt-1">
                        Manage all reconstructed tracks, stems, and masters
                    </p>
                </div>
                <div className="flex items-center gap-3">
                    <Button
                        onClick={fetchData}
                        variant="outline"
                        size="sm"
                        className="border-zinc-700 text-zinc-400 hover:text-zinc-200"
                    >
                        🔄 Refresh
                    </Button>
                    <Link href="/reconstruct">
                        <Button
                            size="sm"
                            className="bg-gradient-to-r from-amber-600 to-red-600 hover:from-amber-500 hover:to-red-500 text-white font-semibold"
                        >
                            🚀 New Reconstruction
                        </Button>
                    </Link>
                </div>
            </div>

            {/* Stats Cards */}
            {stats && (
                <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                    <Card className="bg-zinc-900/50 border-zinc-800">
                        <CardContent className="pt-4 pb-3 px-4">
                            <div className="text-[10px] text-zinc-500 uppercase tracking-wider font-semibold">
                                Total Projects
                            </div>
                            <div className="text-2xl font-bold text-zinc-100 mt-1">
                                {stats.total_projects}
                            </div>
                            <div className="text-[10px] text-zinc-600 mt-1">
                                {stats.completed} completed • {stats.running}{" "}
                                running
                            </div>
                        </CardContent>
                    </Card>
                    <Card className="bg-zinc-900/50 border-zinc-800">
                        <CardContent className="pt-4 pb-3 px-4">
                            <div className="text-[10px] text-zinc-500 uppercase tracking-wider font-semibold">
                                Disk Usage
                            </div>
                            <div className="text-2xl font-bold text-zinc-100 mt-1">
                                {stats.total_disk_gb} GB
                            </div>
                            <div className="text-[10px] text-zinc-600 mt-1">
                                {stats.total_disk_mb} MB total
                            </div>
                        </CardContent>
                    </Card>
                    <Card className="bg-zinc-900/50 border-zinc-800">
                        <CardContent className="pt-4 pb-3 px-4">
                            <div className="text-[10px] text-zinc-500 uppercase tracking-wider font-semibold">
                                Intelligence
                            </div>
                            <div className="text-2xl font-bold text-pink-400 mt-1">
                                {stats.with_intelligence}
                            </div>
                            <div className="text-[10px] text-zinc-600 mt-1">
                                🧠 Brain-guided reconstructions
                            </div>
                        </CardContent>
                    </Card>
                    <Card className="bg-zinc-900/50 border-zinc-800">
                        <CardContent className="pt-4 pb-3 px-4">
                            <div className="text-[10px] text-zinc-500 uppercase tracking-wider font-semibold">
                                Errors
                            </div>
                            <div
                                className={`text-2xl font-bold mt-1 ${stats.errored > 0
                                        ? "text-red-400"
                                        : "text-emerald-400"
                                    }`}
                            >
                                {stats.errored}
                            </div>
                            <div className="text-[10px] text-zinc-600 mt-1">
                                {stats.errored > 0
                                    ? "Need attention"
                                    : "All clear"}
                            </div>
                        </CardContent>
                    </Card>
                </div>
            )}

            {/* Bulk Actions */}
            {selected.size > 0 && (
                <div className="flex items-center gap-3 px-4 py-3 rounded-xl bg-red-950/20 border border-red-800/30">
                    <span className="text-sm text-red-400 font-medium">
                        {selected.size} selected
                    </span>
                    <Button
                        size="sm"
                        variant="destructive"
                        onClick={() => setConfirmBulk(true)}
                        className="h-7 text-xs"
                    >
                        🗑️ Delete Selected
                    </Button>
                    <Button
                        size="sm"
                        variant="ghost"
                        onClick={() => setSelected(new Set())}
                        className="h-7 text-xs text-zinc-500"
                    >
                        Cancel
                    </Button>
                </div>
            )}

            {/* Projects Table */}
            <Card className="bg-zinc-900/50 border-zinc-800 overflow-hidden">
                <CardHeader className="pb-2">
                    <div className="flex items-center justify-between">
                        <div className="flex items-center gap-2">
                            <CardTitle className="text-sm">
                                All Projects
                            </CardTitle>
                            <Badge
                                variant="outline"
                                className="border-zinc-700 text-zinc-500 text-[10px]"
                            >
                                {jobs.length} jobs
                            </Badge>
                        </div>
                        <CardDescription className="text-[10px]">
                            Click column headers to sort
                        </CardDescription>
                    </div>
                </CardHeader>
                <CardContent className="p-0">
                    <div className="overflow-x-auto">
                        <table className="w-full text-sm">
                            <thead>
                                <tr className="border-b border-zinc-800/80 text-[10px] uppercase tracking-wider text-zinc-500">
                                    <th className="px-4 py-3 text-left w-8">
                                        <input
                                            type="checkbox"
                                            checked={
                                                selected.size === jobs.length &&
                                                jobs.length > 0
                                            }
                                            onChange={toggleSelectAll}
                                            className="rounded border-zinc-600 bg-zinc-800"
                                        />
                                    </th>
                                    <th
                                        className="px-4 py-3 text-left cursor-pointer hover:text-zinc-300 transition-colors"
                                        onClick={() =>
                                            handleSort("project_id")
                                        }
                                    >
                                        Project{sortIcon("project_id")}
                                    </th>
                                    <th
                                        className="px-4 py-3 text-left cursor-pointer hover:text-zinc-300 transition-colors"
                                        onClick={() => handleSort("status")}
                                    >
                                        Status{sortIcon("status")}
                                    </th>
                                    <th className="px-4 py-3 text-left">
                                        Track
                                    </th>
                                    <th
                                        className="px-4 py-3 text-left cursor-pointer hover:text-zinc-300 transition-colors"
                                        onClick={() => handleSort("qc_score")}
                                    >
                                        QC{sortIcon("qc_score")}
                                    </th>
                                    <th className="px-4 py-3 text-left">
                                        Files
                                    </th>
                                    <th
                                        className="px-4 py-3 text-left cursor-pointer hover:text-zinc-300 transition-colors"
                                        onClick={() =>
                                            handleSort("disk_size_mb")
                                        }
                                    >
                                        Size{sortIcon("disk_size_mb")}
                                    </th>
                                    <th
                                        className="px-4 py-3 text-left cursor-pointer hover:text-zinc-300 transition-colors"
                                        onClick={() =>
                                            handleSort("created_at")
                                        }
                                    >
                                        Date{sortIcon("created_at")}
                                    </th>
                                    <th className="px-4 py-3 text-right">
                                        Actions
                                    </th>
                                </tr>
                            </thead>
                            <tbody>
                                {sortedJobs.map((job) => (
                                    <tr
                                        key={job.job_id}
                                        className={`border-b border-zinc-800/30 hover:bg-zinc-800/20 transition-colors ${selected.has(job.job_id)
                                                ? "bg-cyan-500/5"
                                                : ""
                                            }`}
                                    >
                                        <td className="px-4 py-3">
                                            <input
                                                type="checkbox"
                                                checked={selected.has(
                                                    job.job_id
                                                )}
                                                onChange={() =>
                                                    toggleSelect(job.job_id)
                                                }
                                                className="rounded border-zinc-600 bg-zinc-800"
                                            />
                                        </td>
                                        <td className="px-4 py-3">
                                            <span className="font-mono text-xs text-zinc-300">
                                                {job.project_id}
                                            </span>
                                        </td>
                                        <td className="px-4 py-3">
                                            {statusBadge(
                                                job.status,
                                                job.cleaned
                                            )}
                                        </td>
                                        <td className="px-4 py-3">
                                            <div className="flex items-center gap-2">
                                                {job.bpm && (
                                                    <span className="text-[10px] text-amber-400 font-mono">
                                                        {job.bpm.toFixed(0)}{" "}
                                                        BPM
                                                    </span>
                                                )}
                                                {job.key && (
                                                    <span className="text-[10px] text-emerald-400 font-mono">
                                                        {job.key}{" "}
                                                        {job.scale ?? ""}
                                                    </span>
                                                )}
                                                {job.duration && (
                                                    <span className="text-[10px] text-zinc-500 font-mono">
                                                        {formatDuration(
                                                            job.duration
                                                        )}
                                                    </span>
                                                )}
                                            </div>
                                        </td>
                                        <td className="px-4 py-3">
                                            {job.qc_score !== null ? (
                                                <span
                                                    className={`text-xs font-mono font-bold ${job.qc_score >= 90
                                                            ? "text-emerald-400"
                                                            : job.qc_score >= 70
                                                                ? "text-amber-400"
                                                                : "text-red-400"
                                                        }`}
                                                >
                                                    {job.qc_score.toFixed(1)}%
                                                </span>
                                            ) : (
                                                <span className="text-zinc-600 text-xs">
                                                    —
                                                </span>
                                            )}
                                        </td>
                                        <td className="px-4 py-3">
                                            <div className="flex items-center gap-1">
                                                {job.has_master && (
                                                    <span
                                                        className="text-[10px]"
                                                        title="Master"
                                                    >
                                                        💎
                                                    </span>
                                                )}
                                                {job.has_mix && (
                                                    <span
                                                        className="text-[10px]"
                                                        title="Mix"
                                                    >
                                                        🎚️
                                                    </span>
                                                )}
                                                {job.has_stems && (
                                                    <span
                                                        className="text-[10px]"
                                                        title="Stems"
                                                    >
                                                        🎵
                                                    </span>
                                                )}
                                                {job.has_brain && (
                                                    <span
                                                        className="text-[10px]"
                                                        title="Brain Intelligence"
                                                    >
                                                        🧠
                                                    </span>
                                                )}
                                            </div>
                                        </td>
                                        <td className="px-4 py-3">
                                            <span className="text-xs text-zinc-400 font-mono">
                                                {job.disk_size_mb > 1000
                                                    ? `${(job.disk_size_mb / 1024).toFixed(1)} GB`
                                                    : `${job.disk_size_mb} MB`}
                                            </span>
                                        </td>
                                        <td className="px-4 py-3">
                                            <span className="text-xs text-zinc-500">
                                                {formatDate(job.created_at)}
                                            </span>
                                        </td>
                                        <td className="px-4 py-3">
                                            <div className="flex items-center justify-end gap-1">
                                                {/* Open in Reconstruct */}
                                                <Link
                                                    href={`/reconstruct?project=${job.project_id}`}
                                                >
                                                    <button
                                                        className="text-[10px] px-2 py-1 rounded-md bg-zinc-800/50 border border-zinc-700/50 text-zinc-400 hover:text-zinc-200 hover:border-zinc-600 transition-all"
                                                        title="Open in Reconstruct"
                                                    >
                                                        🔬
                                                    </button>
                                                </Link>

                                                {/* Download Master */}
                                                {job.has_master &&
                                                    !job.cleaned && (
                                                        <button
                                                            onClick={() =>
                                                                handleDownload(
                                                                    job.job_id,
                                                                    "master",
                                                                    `${job.project_id}_master.wav`
                                                                )
                                                            }
                                                            disabled={downloading.has(
                                                                job.job_id
                                                            )}
                                                            className="text-[10px] px-2 py-1 rounded-md bg-purple-900/20 border border-purple-500/30 text-purple-400 hover:text-purple-300 hover:border-purple-500/50 transition-all disabled:opacity-50"
                                                            title="Download Master"
                                                        >
                                                            {downloading.has(
                                                                job.job_id
                                                            )
                                                                ? "⏳"
                                                                : "💎"}
                                                        </button>
                                                    )}

                                                {/* Delete */}
                                                {!job.cleaned && (
                                                    <button
                                                        onClick={() =>
                                                            setConfirmDelete(
                                                                job.job_id
                                                            )
                                                        }
                                                        disabled={deleting.has(
                                                            job.job_id
                                                        )}
                                                        className="text-[10px] px-2 py-1 rounded-md bg-red-950/20 border border-red-800/30 text-red-400 hover:text-red-300 hover:border-red-700/50 transition-all disabled:opacity-50"
                                                        title="Delete project files"
                                                    >
                                                        {deleting.has(
                                                            job.job_id
                                                        )
                                                            ? "⏳"
                                                            : "🗑️"}
                                                    </button>
                                                )}
                                            </div>
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>

                    {jobs.length === 0 && (
                        <div className="text-center py-12 text-zinc-500">
                            <div className="text-4xl mb-3">📭</div>
                            <p className="text-sm">No projects yet</p>
                            <p className="text-xs text-zinc-600 mt-1">
                                Upload a track in Reconstruct to get started
                            </p>
                        </div>
                    )}
                </CardContent>
            </Card>

            {/* Delete Confirmation Modal */}
            {(confirmDelete || confirmBulk) && (
                <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm">
                    <div className="bg-zinc-900 border border-zinc-700 rounded-xl p-6 max-w-md w-full mx-4 shadow-2xl">
                        <div className="text-center">
                            <div className="text-4xl mb-3">⚠️</div>
                            <h3 className="text-lg font-bold text-zinc-100 mb-2">
                                {confirmBulk
                                    ? `Delete ${selected.size} projects?`
                                    : "Delete this project?"}
                            </h3>
                            <p className="text-sm text-zinc-400 mb-6">
                                This will permanently delete all audio files
                                (original, stems, mix, master). Job metadata
                                will be kept for reference. This cannot be
                                undone.
                            </p>
                            <div className="flex items-center gap-3 justify-center">
                                <Button
                                    variant="outline"
                                    onClick={() => {
                                        setConfirmDelete(null);
                                        setConfirmBulk(false);
                                    }}
                                    className="border-zinc-700 text-zinc-400"
                                >
                                    Cancel
                                </Button>
                                <Button
                                    variant="destructive"
                                    onClick={() =>
                                        confirmBulk
                                            ? handleBulkDelete()
                                            : handleDelete(confirmDelete!)
                                    }
                                >
                                    🗑️ Delete Files
                                </Button>
                            </div>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
}
