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
import {
    uploadTrack,
    runQC,
    fetchVizImage,
    type QCResult,
} from "@/lib/api";

type Stage = "idle" | "uploading" | "analyzing" | "visualizing" | "complete" | "error";

const BAND_LABELS = ["Sub", "Bass", "Low Mid", "Mid", "Upper Mid", "Presence", "Air"];

function MetricBar({ label, value, max, unit, good }: {
    label: string; value: number; max: number; unit: string; good?: boolean;
}) {
    const pct = Math.min(100, Math.max(0, ((value - (max * -1)) / (max * 2)) * 100));
    return (
        <div className="space-y-1">
            <div className="flex justify-between text-xs">
                <span className="text-muted-foreground">{label}</span>
                <span className={`font-mono font-medium ${good === undefined ? "" : good ? "text-auralis-emerald" : "text-destructive"}`}>
                    {value.toFixed(1)} {unit}
                </span>
            </div>
            <div className="h-1.5 rounded-full bg-secondary/40 overflow-hidden">
                <div
                    className={`h-full rounded-full transition-all duration-500 ${good === undefined ? "bg-primary" : good ? "bg-auralis-emerald" : "bg-destructive"}`}
                    style={{ width: `${Math.min(100, Math.abs(pct))}%` }}
                />
            </div>
        </div>
    );
}

export default function QCPage() {
    const fileInputRef = useRef<HTMLInputElement>(null);
    const [stage, setStage] = useState<Stage>("idle");
    const [projectId, setProjectId] = useState<string | null>(null);
    const [qcResult, setQcResult] = useState<QCResult | null>(null);
    const [vizImages, setVizImages] = useState<Record<string, string>>({});
    const [message, setMessage] = useState("");
    const [progress, setProgress] = useState(0);

    const reset = () => {
        setStage("idle");
        setProjectId(null);
        setQcResult(null);
        setVizImages({});
        setMessage("");
        setProgress(0);
    };

    const processFile = useCallback(async (file: File) => {
        try {
            setStage("uploading");
            setMessage(`Uploading ${file.name}...`);
            setProgress(15);
            const upload = await uploadTrack(file);
            setProjectId(upload.project_id);

            setStage("analyzing");
            setMessage("Running 7-dimension quality analysis...");
            setProgress(40);
            const qc = await runQC(upload.project_id);
            setQcResult(qc);

            setStage("visualizing");
            setMessage("Generating visual reports...");
            setProgress(70);
            const vizTypes = ["spectrum", "waveform", "radar", "loudness"];
            const images: Record<string, string> = {};
            for (const type of vizTypes) {
                try {
                    images[type] = await fetchVizImage(type, upload.project_id);
                } catch { /* skip */ }
            }
            setVizImages(images);

            setStage("complete");
            setMessage("QC analysis complete!");
            setProgress(100);
        } catch (err) {
            setStage("error");
            setMessage(err instanceof Error ? err.message : "Unknown error");
        }
    }, []);

    return (
        <div className="space-y-6">
            <div>
                <h1 className="text-2xl font-bold tracking-tight">
                    <span className="text-gradient">QC Report</span>
                </h1>
                <p className="text-sm text-muted-foreground mt-1">
                    7-dimension quality scoring with high-resolution visual analysis.
                </p>
            </div>

            {/* Upload zone */}
            {stage === "idle" && (
                <Card
                    className="glass border-2 border-dashed border-border/40 hover:border-primary/40 hover:bg-primary/3 transition-all duration-300 cursor-pointer"
                    onClick={() => fileInputRef.current?.click()}
                >
                    <CardContent className="flex flex-col items-center justify-center py-16">
                        <input
                            ref={fileInputRef}
                            type="file"
                            accept=".wav,.flac,.mp3,.aiff,.aif"
                            className="hidden"
                            onChange={(e) => {
                                const file = e.target.files?.[0];
                                if (file) processFile(file);
                            }}
                        />
                        <div className="text-5xl mb-4 opacity-60">🔍</div>
                        <p className="text-lg font-medium text-foreground/80">
                            Drop a track or master to analyze
                        </p>
                        <p className="text-sm text-muted-foreground mt-1">
                            WAV, FLAC, MP3, AIFF — get a detailed quality report
                        </p>
                    </CardContent>
                </Card>
            )}

            {/* Processing */}
            {(stage === "uploading" || stage === "analyzing" || stage === "visualizing") && (
                <Card className="glass border-border/30 glow-cyan">
                    <CardContent className="flex flex-col items-center justify-center py-12">
                        <div className="text-4xl mb-4 animate-pulse">
                            {stage === "uploading" ? "📤" : stage === "analyzing" ? "🔬" : "📊"}
                        </div>
                        <p className="text-sm font-medium text-primary mb-3">{message}</p>
                        <div className="w-80 space-y-2">
                            <Progress value={progress} className="h-2" />
                            <p className="text-xs text-muted-foreground text-center">{Math.round(progress)}%</p>
                        </div>
                    </CardContent>
                </Card>
            )}

            {/* Error */}
            {stage === "error" && (
                <Card className="glass border-destructive/30">
                    <CardContent className="flex flex-col items-center justify-center py-12">
                        <div className="text-4xl mb-4">❌</div>
                        <p className="text-sm text-destructive">{message}</p>
                        <Button size="sm" variant="outline" className="mt-4" onClick={reset}>Try Again</Button>
                    </CardContent>
                </Card>
            )}

            {/* Results */}
            {stage === "complete" && qcResult && (
                <>
                    {/* Pass/Fail banner */}
                    <Card className={`glass ${qcResult.pass_fail === "PASS" ? "border-auralis-emerald/30 glow-cyan" : "border-destructive/30"}`}>
                        <CardContent className="flex items-center gap-6 py-6">
                            <div className={`text-5xl ${qcResult.pass_fail === "PASS" ? "" : "grayscale"}`}>
                                {qcResult.pass_fail === "PASS" ? "✅" : "⚠️"}
                            </div>
                            <div className="flex-1">
                                <p className={`text-2xl font-bold ${qcResult.pass_fail === "PASS" ? "text-auralis-emerald" : "text-destructive"}`}>
                                    {qcResult.pass_fail}
                                </p>
                                <p className="text-sm text-muted-foreground">
                                    {qcResult.issues.length === 0
                                        ? "All quality dimensions within acceptable range"
                                        : `${qcResult.issues.length} issue(s) detected`}
                                </p>
                            </div>
                            {qcResult.loudness && (
                                <div className="text-right">
                                    <p className="text-3xl font-bold font-mono text-primary">
                                        {qcResult.loudness.integrated_lufs.toFixed(1)}
                                    </p>
                                    <p className="text-[10px] text-muted-foreground uppercase">LUFS</p>
                                </div>
                            )}
                        </CardContent>
                    </Card>

                    {/* Issues */}
                    {qcResult.issues.length > 0 && (
                        <Card className="glass border-destructive/20">
                            <CardContent className="py-4 space-y-1">
                                {qcResult.issues.map((issue, i) => (
                                    <p key={i} className="text-sm text-destructive flex items-center gap-2">
                                        <span>⚠️</span> {issue}
                                    </p>
                                ))}
                            </CardContent>
                        </Card>
                    )}

                    {/* Metrics grid */}
                    <div className="grid grid-cols-3 gap-4">
                        <Card className="glass border-border/30">
                            <CardHeader className="pb-2">
                                <CardTitle className="text-xs text-muted-foreground uppercase tracking-wider">
                                    Dynamics
                                </CardTitle>
                            </CardHeader>
                            <CardContent className="space-y-3">
                                <MetricBar label="Peak" value={qcResult.dynamics.peak_db} max={3} unit="dB"
                                    good={qcResult.dynamics.peak_db < -0.1} />
                                <MetricBar label="RMS" value={qcResult.dynamics.rms_db} max={20} unit="dB" />
                                <MetricBar label="Crest Factor" value={qcResult.dynamics.crest_factor_db} max={20} unit="dB"
                                    good={qcResult.dynamics.crest_factor_db > 4} />
                                <MetricBar label="Dynamic Range" value={qcResult.dynamics.dynamic_range_db} max={15} unit="dB" />
                            </CardContent>
                        </Card>

                        {qcResult.stereo && (
                            <Card className="glass border-border/30">
                                <CardHeader className="pb-2">
                                    <CardTitle className="text-xs text-muted-foreground uppercase tracking-wider">
                                        Stereo Image
                                    </CardTitle>
                                </CardHeader>
                                <CardContent className="space-y-3">
                                    <MetricBar label="Correlation" value={qcResult.stereo.correlation} max={1} unit=""
                                        good={qcResult.stereo.correlation > 0.3} />
                                    <MetricBar label="Width" value={qcResult.stereo.width} max={1} unit="" />
                                    <div className="flex items-center justify-between mt-2">
                                        <span className="text-xs text-muted-foreground">Mono Compatible</span>
                                        <Badge variant={qcResult.stereo.mono_compatible ? "default" : "destructive"} className="text-[10px]">
                                            {qcResult.stereo.mono_compatible ? "✓ Yes" : "✗ No"}
                                        </Badge>
                                    </div>
                                </CardContent>
                            </Card>
                        )}

                        <Card className="glass border-border/30">
                            <CardHeader className="pb-2">
                                <CardTitle className="text-xs text-muted-foreground uppercase tracking-wider">
                                    Spectrum (dB)
                                </CardTitle>
                            </CardHeader>
                            <CardContent>
                                <div className="space-y-1.5">
                                    {BAND_LABELS.map((label, i) => {
                                        const key = ["sub", "bass", "low_mid", "mid", "upper_mid", "presence", "brilliance"][i];
                                        const val = (qcResult.spectrum as Record<string, number>)[key] || 0;
                                        const maxVal = 120;
                                        return (
                                            <div key={label} className="flex items-center gap-2 text-xs">
                                                <span className="w-16 text-muted-foreground text-[10px]">{label}</span>
                                                <div className="flex-1 h-1.5 rounded-full bg-secondary/30 overflow-hidden">
                                                    <div
                                                        className="h-full rounded-full bg-primary/70"
                                                        style={{ width: `${(val / maxVal) * 100}%` }}
                                                    />
                                                </div>
                                                <span className="w-10 text-right font-mono text-[10px]">{val.toFixed(0)}</span>
                                            </div>
                                        );
                                    })}
                                </div>
                            </CardContent>
                        </Card>
                    </div>

                    {/* Visual reports */}
                    {Object.keys(vizImages).length > 0 && (
                        <Card className="glass border-border/30">
                            <CardHeader className="pb-3">
                                <CardTitle className="text-sm flex items-center gap-2">
                                    <span>📊</span> Visual Analysis
                                    <Badge variant="outline" className="text-[10px]">300dpi</Badge>
                                </CardTitle>
                                <CardDescription className="text-xs">
                                    High-resolution spectral and waveform analysis
                                </CardDescription>
                            </CardHeader>
                            <CardContent className="space-y-6">
                                {vizImages.radar && (
                                    <div className="flex justify-center">
                                        <img src={vizImages.radar} alt="QC Radar" className="max-w-md rounded-lg" />
                                    </div>
                                )}
                                {vizImages.spectrum && (
                                    <div>
                                        <p className="text-xs text-muted-foreground mb-2 uppercase tracking-wider">Spectral Distribution</p>
                                        <img src={vizImages.spectrum} alt="Spectrum" className="w-full rounded-lg" />
                                    </div>
                                )}
                                {vizImages.loudness && (
                                    <div>
                                        <p className="text-xs text-muted-foreground mb-2 uppercase tracking-wider">Loudness Over Time</p>
                                        <img src={vizImages.loudness} alt="Loudness" className="w-full rounded-lg" />
                                    </div>
                                )}
                                {vizImages.waveform && (
                                    <div>
                                        <p className="text-xs text-muted-foreground mb-2 uppercase tracking-wider">Waveform</p>
                                        <img src={vizImages.waveform} alt="Waveform" className="w-full rounded-lg" />
                                    </div>
                                )}
                            </CardContent>
                        </Card>
                    )}

                    <Button variant="outline" onClick={reset}>
                        Analyze Another Track
                    </Button>
                </>
            )}

            {/* Dimensions info */}
            <Card className="glass border-border/30">
                <CardHeader className="pb-3">
                    <CardTitle className="text-xs text-muted-foreground uppercase tracking-wider">
                        Analysis Dimensions
                    </CardTitle>
                </CardHeader>
                <CardContent>
                    <div className="grid grid-cols-4 gap-2">
                        {[
                            { label: "Peak & RMS", icon: "📏" },
                            { label: "Clipping", icon: "⚡" },
                            { label: "Stereo Image", icon: "🔈" },
                            { label: "Loudness", icon: "📊" },
                            { label: "Spectrum", icon: "🌈" },
                            { label: "Crest Factor", icon: "📐" },
                            { label: "Dynamic Range", icon: "🎚️" },
                            { label: "Mono Compat.", icon: "🔇" },
                        ].map((dim) => (
                            <div key={dim.label} className="rounded-lg bg-secondary/30 px-3 py-2 text-center">
                                <span className="text-sm">{dim.icon}</span>
                                <p className="text-[10px] font-medium text-muted-foreground mt-0.5">{dim.label}</p>
                            </div>
                        ))}
                    </div>
                </CardContent>
            </Card>
        </div>
    );
}
