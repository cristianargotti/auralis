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
import { Progress } from "@/components/ui/progress";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { uploadTrack, startAnalysis, getJobStatus } from "@/lib/api";

type JobStatusType = "idle" | "uploading" | "analyzing" | "complete" | "error";

interface AnalysisJob {
    status: JobStatusType;
    progress: number;
    message: string;
    projectId: string | null;
    result: Record<string, unknown> | null;
}

const VALID_EXTENSIONS = [".wav", ".mp3", ".flac", ".aiff", ".aif"];

export default function DeconstructPage() {
    const fileInputRef = useRef<HTMLInputElement>(null);
    const [dragOver, setDragOver] = useState(false);
    const [job, setJob] = useState<AnalysisJob>({
        status: "idle",
        progress: 0,
        message: "",
        projectId: null,
        result: null,
    });

    const processFile = useCallback(async (file: File) => {
        const ext = "." + file.name.split(".").pop()?.toLowerCase();
        if (!VALID_EXTENSIONS.includes(ext)) {
            setJob((prev) => ({
                ...prev,
                status: "error",
                message: `Unsupported format: ${ext}. Use WAV, MP3, FLAC, or AIFF.`,
            }));
            return;
        }

        // Upload
        setJob({
            status: "uploading",
            progress: 10,
            message: `Uploading ${file.name}...`,
            projectId: null,
            result: null,
        });

        try {
            const uploadData = await uploadTrack(file);
            const projectId = uploadData.project_id;

            setJob((prev) => ({
                ...prev,
                status: "analyzing",
                progress: 25,
                message: "Starting analysis pipeline...",
                projectId,
            }));

            // Start analysis
            const analyzeData = await startAnalysis(projectId);
            const jobId = analyzeData.job_id;

            // Poll status
            const poll = setInterval(async () => {
                try {
                    const statusData = await getJobStatus(jobId);

                    const progress = Math.min(
                        25 + (statusData.progress / statusData.total_steps) * 75,
                        100
                    );

                    setJob((prev) => ({
                        ...prev,
                        progress,
                        message: statusData.message,
                    }));

                    if (statusData.status === "complete") {
                        clearInterval(poll);
                        setJob((prev) => ({
                            ...prev,
                            status: "complete",
                            progress: 100,
                            message: "Analysis complete!",
                            result: statusData.result,
                        }));
                    } else if (statusData.status === "error") {
                        clearInterval(poll);
                        setJob((prev) => ({
                            ...prev,
                            status: "error",
                            message: statusData.message,
                        }));
                    }
                } catch {
                    /* keep polling */
                }
            }, 1500);
        } catch (err) {
            setJob((prev) => ({
                ...prev,
                status: "error",
                message: err instanceof Error ? err.message : "Unknown error",
            }));
        }
    }, []);

    const handleDrop = useCallback(
        (e: React.DragEvent) => {
            e.preventDefault();
            setDragOver(false);
            const file = e.dataTransfer.files[0];
            if (file) processFile(file);
        },
        [processFile]
    );

    const handleFileSelect = useCallback(
        (e: React.ChangeEvent<HTMLInputElement>) => {
            const file = e.target.files?.[0];
            if (file) processFile(file);
        },
        [processFile]
    );

    return (
        <div className="space-y-6">
            <div>
                <h1 className="text-2xl font-bold tracking-tight">
                    <span className="text-gradient">Deconstructor</span>
                </h1>
                <p className="text-sm text-muted-foreground mt-1">
                    Upload any track — AURALIS deconstructs it into atoms.
                </p>
            </div>

            {/* Upload zone */}
            <Card
                className={`glass border-2 border-dashed transition-all duration-300 cursor-pointer ${dragOver
                    ? "border-primary bg-primary/5 glow-cyan"
                    : job.status === "idle"
                        ? "border-border/40 hover:border-primary/40 hover:bg-primary/3"
                        : "border-border/20"
                    }`}
                onDragOver={(e) => {
                    e.preventDefault();
                    setDragOver(true);
                }}
                onDragLeave={() => setDragOver(false)}
                onDrop={handleDrop}
                onClick={() => job.status === "idle" && fileInputRef.current?.click()}
            >
                <CardContent className="flex flex-col items-center justify-center py-16">
                    <input
                        ref={fileInputRef}
                        type="file"
                        accept={VALID_EXTENSIONS.join(",")}
                        className="hidden"
                        onChange={handleFileSelect}
                    />

                    {job.status === "idle" && (
                        <>
                            <div className="text-5xl mb-4 opacity-60">🎵</div>
                            <p className="text-lg font-medium text-foreground/80">
                                Drop your track here
                            </p>
                            <p className="text-sm text-muted-foreground mt-1">
                                or click to browse • WAV, MP3, FLAC, AIFF
                            </p>
                        </>
                    )}

                    {(job.status === "uploading" || job.status === "analyzing") && (
                        <>
                            <div className="text-4xl mb-4 animate-pulse">👂</div>
                            <p className="text-sm font-medium text-primary mb-3">
                                {job.message}
                            </p>
                            <div className="w-64 space-y-2">
                                <Progress value={job.progress} className="h-2" />
                                <p className="text-xs text-muted-foreground text-center">
                                    {Math.round(job.progress)}%
                                </p>
                            </div>
                        </>
                    )}

                    {job.status === "complete" && (
                        <>
                            <div className="text-4xl mb-4">✅</div>
                            <p className="text-sm font-medium text-auralis-emerald">
                                {job.message}
                            </p>
                            <Button
                                size="sm"
                                variant="outline"
                                className="mt-4"
                                onClick={(e) => {
                                    e.stopPropagation();
                                    setJob({
                                        status: "idle",
                                        progress: 0,
                                        message: "",
                                        projectId: null,
                                        result: null,
                                    });
                                }}
                            >
                                Upload Another Track
                            </Button>
                        </>
                    )}

                    {job.status === "error" && (
                        <>
                            <div className="text-4xl mb-4">❌</div>
                            <p className="text-sm text-destructive">{job.message}</p>
                            <Button
                                size="sm"
                                variant="outline"
                                className="mt-4"
                                onClick={(e) => {
                                    e.stopPropagation();
                                    setJob({
                                        status: "idle",
                                        progress: 0,
                                        message: "",
                                        projectId: null,
                                        result: null,
                                    });
                                }}
                            >
                                Try Again
                            </Button>
                        </>
                    )}
                </CardContent>
            </Card>

            {/* Analysis results */}
            {job.status === "complete" && job.result && (
                <Tabs defaultValue="dna" className="space-y-4">
                    <TabsList className="bg-secondary/50">
                        <TabsTrigger value="dna">Track DNA</TabsTrigger>
                        <TabsTrigger value="stems">Stems</TabsTrigger>
                        <TabsTrigger value="midi">MIDI</TabsTrigger>
                    </TabsList>

                    <TabsContent value="dna">
                        <Card className="glass border-border/30">
                            <CardHeader>
                                <CardTitle className="text-sm flex items-center gap-2">
                                    <span>🧬</span> Track DNA Map
                                </CardTitle>
                                <CardDescription>
                                    Complete fingerprint of your track
                                </CardDescription>
                            </CardHeader>
                            <CardContent>
                                <pre className="text-xs text-muted-foreground font-mono bg-secondary/30 rounded-lg p-4 overflow-auto max-h-96">
                                    {JSON.stringify(job.result.track_dna, null, 2)}
                                </pre>
                            </CardContent>
                        </Card>
                    </TabsContent>

                    <TabsContent value="stems">
                        <Card className="glass border-border/30">
                            <CardHeader>
                                <CardTitle className="text-sm flex items-center gap-2">
                                    <span>🎼</span> Separated Stems
                                </CardTitle>
                            </CardHeader>
                            <CardContent>
                                {typeof job.result.stems === "object" && job.result.stems !== null && "stems" in (job.result.stems as Record<string, unknown>) ? (
                                    <div className="grid grid-cols-4 gap-3">
                                        {Object.keys(
                                            (job.result.stems as Record<string, unknown>).stems as Record<string, string>
                                        ).map((stem) => (
                                            <div
                                                key={stem}
                                                className="rounded-xl bg-secondary/30 p-4 text-center"
                                            >
                                                <p className="text-2xl mb-2">
                                                    {stem === "vocals" ? "🎤" : stem === "drums" ? "🥁" : stem === "bass" ? "🎸" : "🎹"}
                                                </p>
                                                <p className="text-xs font-medium capitalize">{stem}</p>
                                                <Badge variant="outline" className="mt-2 text-[10px]">
                                                    Ready
                                                </Badge>
                                            </div>
                                        ))}
                                    </div>
                                ) : (
                                    <pre className="text-xs text-muted-foreground font-mono bg-secondary/30 rounded-lg p-4 overflow-auto">
                                        {JSON.stringify(job.result.stems, null, 2)}
                                    </pre>
                                )}
                            </CardContent>
                        </Card>
                    </TabsContent>

                    <TabsContent value="midi">
                        <Card className="glass border-border/30">
                            <CardHeader>
                                <CardTitle className="text-sm flex items-center gap-2">
                                    <span>🎹</span> Extracted MIDI
                                </CardTitle>
                            </CardHeader>
                            <CardContent>
                                <pre className="text-xs text-muted-foreground font-mono bg-secondary/30 rounded-lg p-4 overflow-auto max-h-96">
                                    {JSON.stringify(job.result.midi, null, 2)}
                                </pre>
                            </CardContent>
                        </Card>
                    </TabsContent>
                </Tabs>
            )}

            {/* Pipeline info */}
            <Card className="glass border-border/30">
                <CardHeader className="pb-3">
                    <CardTitle className="text-xs text-muted-foreground uppercase tracking-wider">
                        Analysis Pipeline
                    </CardTitle>
                </CardHeader>
                <CardContent>
                    <div className="grid grid-cols-4 gap-4">
                        {[
                            { step: 1, label: "Spectral Analysis", icon: "📊", desc: "10-band frequency fingerprint" },
                            { step: 2, label: "Track DNA", icon: "🧬", desc: "Key, tempo, sections, loudness" },
                            { step: 3, label: "Stem Separation", icon: "🎼", desc: "HTDemucs v4 (GPU)" },
                            { step: 4, label: "MIDI Extraction", icon: "🎹", desc: "basic-pitch transcription" },
                        ].map((item) => (
                            <div
                                key={item.step}
                                className={`rounded-lg p-3 text-center transition-all ${job.status === "analyzing" && job.progress >= item.step * 25
                                    ? "bg-primary/8 border border-primary/20"
                                    : "bg-secondary/30"
                                    }`}
                            >
                                <span className="text-xl">{item.icon}</span>
                                <p className="text-xs font-medium mt-1">{item.label}</p>
                                <p className="text-[10px] text-muted-foreground">{item.desc}</p>
                            </div>
                        ))}
                    </div>
                </CardContent>
            </Card>
        </div>
    );
}
