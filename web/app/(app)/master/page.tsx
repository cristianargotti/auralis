"use client";

import { useCallback, useEffect, useRef, useState } from "react";
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
    masterTrack,
    runQC,
    fetchVizImage,
    type MasterResult,
    type QCResult,
} from "@/lib/api";

type Stage = "idle" | "uploading" | "mastering" | "qc" | "visualizing" | "complete" | "error";

const PRESETS = [
    { key: "mood_check", label: "Mood Check", lufs: "-14", desc: "Quick preview, gentle processing", icon: "🎧" },
    { key: "streaming", label: "Streaming", lufs: "-14", desc: "Spotify/Apple Music optimized", icon: "📡" },
    { key: "club", label: "Club", lufs: "-8", desc: "Maximum impact, full loudness", icon: "🔊" },
];

const STAGES_INFO = [
    { key: "ms_eq", label: "M/S EQ", icon: "🎛️" },
    { key: "soft_clip_1", label: "Soft Clip", icon: "📐" },
    { key: "saturation", label: "Saturation", icon: "🔥" },
    { key: "exciter", label: "Exciter", icon: "✨" },
    { key: "compression", label: "Compression", icon: "💪" },
    { key: "makeup_gain", label: "Makeup", icon: "📈" },
    { key: "stereo_width", label: "Stereo", icon: "🔈" },
    { key: "soft_clip_2", label: "Pre-Limit", icon: "📐" },
    { key: "limiter", label: "Limiter", icon: "🧱" },
    { key: "dither", label: "Dither", icon: "🎲" },
];

export default function MasterPage() {
    const fileInputRef = useRef<HTMLInputElement>(null);
    const [stage, setStage] = useState<Stage>("idle");
    const [preset, setPreset] = useState("streaming");
    const [projectId, setProjectId] = useState<string | null>(null);
    const [masterResult, setMasterResult] = useState<MasterResult | null>(null);
    const [qcResult, setQcResult] = useState<QCResult | null>(null);
    const [vizImages, setVizImages] = useState<Record<string, string>>({});
    const [message, setMessage] = useState("");
    const [progress, setProgress] = useState(0);

    const reset = () => {
        setStage("idle");
        setProjectId(null);
        setMasterResult(null);
        setQcResult(null);
        setVizImages({});
        setMessage("");
        setProgress(0);
    };

    const processFile = useCallback(async (file: File) => {
        try {
            // Upload
            setStage("uploading");
            setMessage(`Uploading ${file.name}...`);
            setProgress(10);
            const upload = await uploadTrack(file);
            setProjectId(upload.project_id);

            // Master
            setStage("mastering");
            setMessage("Running 10-stage mastering chain...");
            setProgress(30);
            const result = await masterTrack(upload.project_id, preset);
            setMasterResult(result);

            // QC
            setStage("qc");
            setMessage("Running quality analysis...");
            setProgress(60);
            const qc = await runQC(upload.project_id);
            setQcResult(qc);

            // Visualizations
            setStage("visualizing");
            setMessage("Generating comparison charts...");
            setProgress(80);

            const vizTypes = ["spectrum", "waveform", "radar", "loudness"];
            const images: Record<string, string> = {};
            for (const type of vizTypes) {
                try {
                    images[type] = await fetchVizImage(type, upload.project_id);
                } catch { /* skip failed viz */ }
            }
            setVizImages(images);

            setStage("complete");
            setMessage("Mastering complete!");
            setProgress(100);
        } catch (err) {
            setStage("error");
            setMessage(err instanceof Error ? err.message : "Unknown error");
        }
    }, [preset]);

    return (
        <div className="space-y-6">
            <div>
                <h1 className="text-2xl font-bold tracking-tight">
                    <span className="text-gradient">Master Suite</span>
                </h1>
                <p className="text-sm text-muted-foreground mt-1">
                    10-stage studio mastering — upload, process, validate.
                </p>
            </div>

            {/* Preset selector */}
            {stage === "idle" && (
                <Card className="glass border-border/30">
                    <CardHeader className="pb-3">
                        <CardTitle className="text-sm text-muted-foreground uppercase tracking-wider">
                            Select Preset
                        </CardTitle>
                    </CardHeader>
                    <CardContent>
                        <div className="grid grid-cols-3 gap-3">
                            {PRESETS.map((p) => (
                                <button
                                    key={p.key}
                                    onClick={() => setPreset(p.key)}
                                    className={`rounded-xl p-4 text-left transition-all duration-300 border ${preset === p.key
                                            ? "bg-primary/8 border-primary/40 glow-cyan"
                                            : "bg-secondary/30 border-border/20 hover:border-primary/20"
                                        }`}
                                >
                                    <div className="flex items-center gap-2 mb-2">
                                        <span className="text-xl">{p.icon}</span>
                                        <span className="text-sm font-bold">{p.label}</span>
                                        <Badge variant="outline" className="ml-auto text-[10px]">
                                            {p.lufs} LUFS
                                        </Badge>
                                    </div>
                                    <p className="text-xs text-muted-foreground">{p.desc}</p>
                                </button>
                            ))}
                        </div>
                    </CardContent>
                </Card>
            )}

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
                        <div className="text-5xl mb-4 opacity-60">💎</div>
                        <p className="text-lg font-medium text-foreground/80">
                            Drop your mix here to master
                        </p>
                        <p className="text-sm text-muted-foreground mt-1">
                            WAV, FLAC, MP3, AIFF • Preset: <span className="text-primary font-medium">{PRESETS.find(p => p.key === preset)?.label}</span>
                        </p>
                    </CardContent>
                </Card>
            )}

            {/* Processing state */}
            {(stage === "uploading" || stage === "mastering" || stage === "qc" || stage === "visualizing") && (
                <Card className="glass border-border/30 glow-cyan">
                    <CardContent className="flex flex-col items-center justify-center py-12">
                        <div className="text-4xl mb-4 animate-pulse">
                            {stage === "uploading" ? "📤" : stage === "mastering" ? "🎛️" : stage === "qc" ? "🔍" : "📊"}
                        </div>
                        <p className="text-sm font-medium text-primary mb-3">{message}</p>
                        <div className="w-80 space-y-2">
                            <Progress value={progress} className="h-2" />
                            <p className="text-xs text-muted-foreground text-center">{Math.round(progress)}%</p>
                        </div>
                    </CardContent>
                </Card>
            )}

            {/* Error state */}
            {stage === "error" && (
                <Card className="glass border-destructive/30">
                    <CardContent className="flex flex-col items-center justify-center py-12">
                        <div className="text-4xl mb-4">❌</div>
                        <p className="text-sm text-destructive">{message}</p>
                        <Button size="sm" variant="outline" className="mt-4" onClick={reset}>
                            Try Again
                        </Button>
                    </CardContent>
                </Card>
            )}

            {/* Results */}
            {stage === "complete" && masterResult && (
                <>
                    {/* Summary cards */}
                    <div className="grid grid-cols-4 gap-3">
                        <Card className="glass border-border/30">
                            <CardContent className="py-4 text-center">
                                <p className="text-2xl font-bold text-primary">
                                    {masterResult.est_lufs.toFixed(1)}
                                </p>
                                <p className="text-[10px] text-muted-foreground uppercase tracking-wider mt-1">LUFS</p>
                            </CardContent>
                        </Card>
                        <Card className="glass border-border/30">
                            <CardContent className="py-4 text-center">
                                <p className="text-2xl font-bold text-foreground">
                                    {masterResult.peak_dbtp.toFixed(1)}
                                </p>
                                <p className="text-[10px] text-muted-foreground uppercase tracking-wider mt-1">Peak dBTP</p>
                            </CardContent>
                        </Card>
                        <Card className="glass border-border/30">
                            <CardContent className="py-4 text-center">
                                <p className={`text-2xl font-bold ${masterResult.clipping_samples === 0 ? "text-auralis-emerald" : "text-destructive"}`}>
                                    {masterResult.clipping_samples === 0 ? "✓" : masterResult.clipping_samples}
                                </p>
                                <p className="text-[10px] text-muted-foreground uppercase tracking-wider mt-1">Clipping</p>
                            </CardContent>
                        </Card>
                        <Card className="glass border-border/30">
                            <CardContent className="py-4 text-center">
                                <p className={`text-2xl font-bold ${qcResult?.pass_fail === "PASS" ? "text-auralis-emerald" : "text-destructive"}`}>
                                    {qcResult?.pass_fail || "—"}
                                </p>
                                <p className="text-[10px] text-muted-foreground uppercase tracking-wider mt-1">QC</p>
                            </CardContent>
                        </Card>
                    </div>

                    {/* Mastering chain stages */}
                    <Card className="glass border-border/30">
                        <CardHeader className="pb-3">
                            <CardTitle className="text-sm flex items-center gap-2">
                                <span>🔗</span> Mastering Chain
                                <Badge variant="outline" className="ml-auto text-[10px]">
                                    {masterResult.stages.length} stages
                                </Badge>
                            </CardTitle>
                        </CardHeader>
                        <CardContent>
                            <div className="flex items-center gap-1.5 flex-wrap">
                                {STAGES_INFO.map((s) => {
                                    const applied = masterResult.stages.includes(s.key);
                                    return (
                                        <div
                                            key={s.key}
                                            className={`rounded-lg px-3 py-2 text-center transition-all ${applied
                                                    ? "bg-primary/10 border border-primary/25"
                                                    : "bg-secondary/20 border border-border/10 opacity-40"
                                                }`}
                                        >
                                            <span className="text-sm">{s.icon}</span>
                                            <p className="text-[9px] font-medium mt-0.5">{s.label}</p>
                                        </div>
                                    );
                                })}
                            </div>
                        </CardContent>
                    </Card>

                    {/* QC Details */}
                    {qcResult && (
                        <Card className="glass border-border/30">
                            <CardHeader className="pb-3">
                                <CardTitle className="text-sm flex items-center gap-2">
                                    <span>📋</span> Quality Analysis
                                    {qcResult.issues.length > 0 && (
                                        <Badge variant="destructive" className="text-[10px]">
                                            {qcResult.issues.length} issues
                                        </Badge>
                                    )}
                                </CardTitle>
                            </CardHeader>
                            <CardContent>
                                <div className="grid grid-cols-3 gap-4">
                                    <div className="space-y-2">
                                        <p className="text-[10px] text-muted-foreground uppercase tracking-wider">Dynamics</p>
                                        <div className="space-y-1 text-xs">
                                            <div className="flex justify-between"><span>Peak</span><span className="font-mono">{qcResult.dynamics.peak_db.toFixed(1)} dB</span></div>
                                            <div className="flex justify-between"><span>RMS</span><span className="font-mono">{qcResult.dynamics.rms_db.toFixed(1)} dB</span></div>
                                            <div className="flex justify-between"><span>Crest</span><span className="font-mono">{qcResult.dynamics.crest_factor_db.toFixed(1)} dB</span></div>
                                            <div className="flex justify-between"><span>DR</span><span className="font-mono">{qcResult.dynamics.dynamic_range_db.toFixed(1)} dB</span></div>
                                        </div>
                                    </div>
                                    {qcResult.stereo && (
                                        <div className="space-y-2">
                                            <p className="text-[10px] text-muted-foreground uppercase tracking-wider">Stereo</p>
                                            <div className="space-y-1 text-xs">
                                                <div className="flex justify-between"><span>Correlation</span><span className="font-mono">{qcResult.stereo.correlation.toFixed(2)}</span></div>
                                                <div className="flex justify-between"><span>Width</span><span className="font-mono">{qcResult.stereo.width.toFixed(2)}</span></div>
                                                <div className="flex justify-between"><span>Mono OK</span><span className={qcResult.stereo.mono_compatible ? "text-auralis-emerald" : "text-destructive"}>{qcResult.stereo.mono_compatible ? "✓" : "✗"}</span></div>
                                            </div>
                                        </div>
                                    )}
                                    {qcResult.loudness && (
                                        <div className="space-y-2">
                                            <p className="text-[10px] text-muted-foreground uppercase tracking-wider">Loudness</p>
                                            <div className="space-y-1 text-xs">
                                                <div className="flex justify-between"><span>LUFS</span><span className="font-mono">{qcResult.loudness.integrated_lufs.toFixed(1)}</span></div>
                                                <div className="flex justify-between"><span>True Peak</span><span className="font-mono">{qcResult.loudness.true_peak_dbtp.toFixed(1)} dBTP</span></div>
                                            </div>
                                        </div>
                                    )}
                                </div>
                                {qcResult.issues.length > 0 && (
                                    <div className="mt-4 space-y-1">
                                        {qcResult.issues.map((issue, i) => (
                                            <p key={i} className="text-xs text-destructive">⚠️ {issue}</p>
                                        ))}
                                    </div>
                                )}
                            </CardContent>
                        </Card>
                    )}

                    {/* Visualizations */}
                    {Object.keys(vizImages).length > 0 && (
                        <Card className="glass border-border/30">
                            <CardHeader className="pb-3">
                                <CardTitle className="text-sm flex items-center gap-2">
                                    <span>📊</span> Visual Comparison
                                    <Badge variant="outline" className="text-[10px]">300dpi</Badge>
                                </CardTitle>
                                <CardDescription className="text-xs">
                                    Original vs Master — high-resolution analysis
                                </CardDescription>
                            </CardHeader>
                            <CardContent className="space-y-4">
                                {vizImages.spectrum && (
                                    <div>
                                        <p className="text-xs text-muted-foreground mb-2 uppercase tracking-wider">7-Band Spectrum</p>
                                        <img src={vizImages.spectrum} alt="Spectrum comparison" className="w-full rounded-lg" />
                                    </div>
                                )}
                                {vizImages.waveform && (
                                    <div>
                                        <p className="text-xs text-muted-foreground mb-2 uppercase tracking-wider">Waveform</p>
                                        <img src={vizImages.waveform} alt="Waveform comparison" className="w-full rounded-lg" />
                                    </div>
                                )}
                                <div className="grid grid-cols-2 gap-4">
                                    {vizImages.radar && (
                                        <div>
                                            <p className="text-xs text-muted-foreground mb-2 uppercase tracking-wider">Quality Radar</p>
                                            <img src={vizImages.radar} alt="QC radar" className="w-full rounded-lg" />
                                        </div>
                                    )}
                                    {vizImages.loudness && (
                                        <div>
                                            <p className="text-xs text-muted-foreground mb-2 uppercase tracking-wider">Loudness Timeline</p>
                                            <img src={vizImages.loudness} alt="Loudness timeline" className="w-full rounded-lg" />
                                        </div>
                                    )}
                                </div>
                            </CardContent>
                        </Card>
                    )}

                    <Button variant="outline" onClick={reset}>
                        Master Another Track
                    </Button>
                </>
            )}

            {/* Pipeline info */}
            <Card className="glass border-border/30">
                <CardHeader className="pb-3">
                    <CardTitle className="text-xs text-muted-foreground uppercase tracking-wider">
                        Mastering Pipeline
                    </CardTitle>
                </CardHeader>
                <CardContent>
                    <div className="grid grid-cols-5 gap-3">
                        {[
                            { label: "M/S EQ", desc: "Surgical frequency shaping", icon: "🎛️" },
                            { label: "Saturation", desc: "3-band harmonic warmth", icon: "🔥" },
                            { label: "Compression", desc: "Linear-phase 3-band", icon: "💪" },
                            { label: "Limiter", desc: "4× oversampled brickwall", icon: "🧱" },
                            { label: "QC", desc: "7-band spectral analysis", icon: "🔍" },
                        ].map((item) => (
                            <div key={item.label} className="rounded-lg bg-secondary/30 p-3 text-center">
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
