"use client";

import { useState, useEffect } from "react";
import { apiFetch } from "@/lib/api";

/* ── Types ── */
interface SpectralCompareProps {
    jobId: string;
    hasOriginal?: boolean;
    hasMaster?: boolean;
}

type ViewMode = "side-by-side" | "overlay" | "original" | "master";

/* ── Component ── */
export default function SpectralCompare({ jobId, hasOriginal = true, hasMaster = true }: SpectralCompareProps) {
    const [mode, setMode] = useState<ViewMode>("side-by-side");
    const [originalUrl, setOriginalUrl] = useState<string>("");
    const [masterUrl, setMasterUrl] = useState<string>("");
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string>("");

    useEffect(() => {
        setLoading(true);
        setError("");

        const loadImages = async () => {
            try {
                const urls: string[] = [];
                if (hasOriginal) {
                    const res = await apiFetch(`/api/reconstruct/spectrogram/${jobId}/original`);
                    if (res.ok) {
                        const blob = await res.blob();
                        urls.push(URL.createObjectURL(blob));
                    }
                }
                if (hasMaster) {
                    const res = await apiFetch(`/api/reconstruct/spectrogram/${jobId}/master`);
                    if (res.ok) {
                        const blob = await res.blob();
                        urls.push(URL.createObjectURL(blob));
                    }
                }

                if (urls.length > 0) setOriginalUrl(urls[0] || "");
                if (urls.length > 1) setMasterUrl(urls[1] || "");
            } catch (e) {
                setError("Failed to load spectrograms");
            } finally {
                setLoading(false);
            }
        };

        loadImages();

        // Cleanup blob URLs
        return () => {
            if (originalUrl) URL.revokeObjectURL(originalUrl);
            if (masterUrl) URL.revokeObjectURL(masterUrl);
        };
    }, [jobId]); // eslint-disable-line

    if (!hasOriginal && !hasMaster) return null;

    const modes: Array<{ key: ViewMode; label: string; icon: string }> = [
        { key: "side-by-side", label: "Side by Side", icon: "⬜⬜" },
        { key: "overlay", label: "Overlay", icon: "🔀" },
        { key: "original", label: "Original", icon: "🎵" },
        { key: "master", label: "Master", icon: "💎" },
    ];

    return (
        <div className="rounded-xl border border-zinc-800 bg-zinc-900/80 backdrop-blur-sm overflow-hidden">
            {/* Header */}
            <div className="flex items-center justify-between px-4 py-3 border-b border-zinc-800/50">
                <div className="flex items-center gap-2">
                    <span className="text-lg">📊</span>
                    <span className="text-sm font-semibold text-zinc-200">Spectral Comparison</span>
                    <span className="text-[10px] text-zinc-500">Mel Spectrogram · 128 bands</span>
                </div>

                {/* Mode toggles */}
                <div className="flex gap-1">
                    {modes.map((m) => (
                        <button
                            key={m.key}
                            onClick={() => setMode(m.key)}
                            className={`text-[10px] px-2 py-1 rounded-md transition-all ${mode === m.key
                                ? "bg-amber-500/20 text-amber-400 border border-amber-500/30"
                                : "text-zinc-500 hover:text-zinc-300 border border-transparent"
                                }`}
                        >
                            {m.icon} {m.label}
                        </button>
                    ))}
                </div>
            </div>

            {/* Content */}
            <div className="p-3">
                {loading ? (
                    <div className="flex items-center justify-center h-40">
                        <div className="flex items-center gap-3 text-zinc-500">
                            <div className="w-5 h-5 rounded-full border-2 border-amber-500/30 border-t-amber-400 animate-spin" />
                            <span className="text-sm">Generating spectrograms...</span>
                        </div>
                    </div>
                ) : error ? (
                    <div className="text-center text-red-400 text-sm py-8">{error}</div>
                ) : (
                    <>
                        {mode === "side-by-side" && (
                            <div className="grid grid-cols-2 gap-3">
                                {originalUrl && (
                                    <div>
                                        <div className="text-[10px] text-amber-400 font-semibold mb-1.5 flex items-center gap-1">
                                            <span className="w-2 h-2 rounded-full bg-amber-500" />
                                            Original
                                        </div>
                                        <div className="rounded-lg overflow-hidden border border-zinc-800">
                                            <img
                                                src={originalUrl}
                                                alt="Original spectrogram"
                                                className="w-full h-auto"
                                                style={{ imageRendering: "auto" }}
                                            />
                                        </div>
                                    </div>
                                )}
                                {masterUrl && (
                                    <div>
                                        <div className="text-[10px] text-purple-400 font-semibold mb-1.5 flex items-center gap-1">
                                            <span className="w-2 h-2 rounded-full bg-purple-500" />
                                            Master
                                        </div>
                                        <div className="rounded-lg overflow-hidden border border-zinc-800">
                                            <img
                                                src={masterUrl}
                                                alt="Master spectrogram"
                                                className="w-full h-auto"
                                                style={{ imageRendering: "auto" }}
                                            />
                                        </div>
                                    </div>
                                )}
                            </div>
                        )}

                        {mode === "overlay" && (
                            <div className="relative">
                                <div className="text-[10px] text-zinc-400 font-semibold mb-1.5 flex items-center gap-2">
                                    <span className="flex items-center gap-1">
                                        <span className="w-2 h-2 rounded-full bg-amber-500" /> Original (base)
                                    </span>
                                    <span className="text-zinc-600">+</span>
                                    <span className="flex items-center gap-1">
                                        <span className="w-2 h-2 rounded-full bg-purple-500" /> Master (overlay 50%)
                                    </span>
                                </div>
                                <div className="relative rounded-lg overflow-hidden border border-zinc-800">
                                    {originalUrl && (
                                        <img
                                            src={originalUrl}
                                            alt="Original spectrogram"
                                            className="w-full h-auto"
                                        />
                                    )}
                                    {masterUrl && (
                                        <img
                                            src={masterUrl}
                                            alt="Master spectrogram overlay"
                                            className="absolute inset-0 w-full h-full opacity-50 mix-blend-screen"
                                        />
                                    )}
                                </div>
                            </div>
                        )}

                        {mode === "original" && originalUrl && (
                            <div>
                                <div className="text-[10px] text-amber-400 font-semibold mb-1.5 flex items-center gap-1">
                                    <span className="w-2 h-2 rounded-full bg-amber-500" />
                                    Original — Full Resolution
                                </div>
                                <div className="rounded-lg overflow-hidden border border-zinc-800">
                                    <img
                                        src={originalUrl}
                                        alt="Original spectrogram"
                                        className="w-full h-auto"
                                    />
                                </div>
                            </div>
                        )}

                        {mode === "master" && masterUrl && (
                            <div>
                                <div className="text-[10px] text-purple-400 font-semibold mb-1.5 flex items-center gap-1">
                                    <span className="w-2 h-2 rounded-full bg-purple-500" />
                                    Master — Full Resolution
                                </div>
                                <div className="rounded-lg overflow-hidden border border-zinc-800">
                                    <img
                                        src={masterUrl}
                                        alt="Master spectrogram"
                                        className="w-full h-auto"
                                    />
                                </div>
                            </div>
                        )}
                    </>
                )}
            </div>

            {/* Footer */}
            <div className="flex items-center gap-4 px-4 py-2 border-t border-zinc-800/50 text-[10px] text-zinc-500">
                <span>128 Mel bands · 22kHz</span>
                <span>Magma colormap · -60 → 0 dB</span>
                <span>Cached after first generation</span>
            </div>
        </div>
    );
}
