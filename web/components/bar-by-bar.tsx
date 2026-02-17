"use client";

import { useMemo } from "react";

/* ── Types ── */
interface Section {
    start: number;
    end: number;
    label: string;
    energy: number;
}

interface Layer {
    name: string;
    type: "point" | "region";
    events: Array<{
        time?: number;
        start?: number;
        end?: number;
        duration?: number;
        label?: string;
        energy?: number;
        type?: string;
    }>;
}

interface XrayAnalysis {
    duration: number;
    layers: Layer[];
    analysis: {
        arrangement?: {
            sections?: Section[];
            tempo_bpm?: number;
            key?: string;
            scale?: string;
        };
        drums?: { total_hits?: number };
        bass?: { type?: string };
        vocals?: { total_regions?: number };
        other?: { total_elements?: number };
    };
}

interface BarByBarProps {
    xrayAnalysis: XrayAnalysis;
    currentTime?: number;
    onSeek?: (time: number) => void;
}

/* ── Constants ── */
const STEM_CONFIG: Record<string, { icon: string; color: string; bgColor: string; borderColor: string }> = {
    Drums: { icon: "🥁", color: "#06b6d4", bgColor: "rgba(6,182,212,0.15)", borderColor: "rgba(6,182,212,0.3)" },
    Bass: { icon: "🎸", color: "#f43f5e", bgColor: "rgba(244,63,94,0.15)", borderColor: "rgba(244,63,94,0.3)" },
    Vocals: { icon: "🎤", color: "#a855f7", bgColor: "rgba(168,85,247,0.15)", borderColor: "rgba(168,85,247,0.3)" },
    Other: { icon: "🎹", color: "#10b981", bgColor: "rgba(16,185,129,0.15)", borderColor: "rgba(16,185,129,0.3)" },
};

const SECTION_COLORS: Record<string, { bg: string; text: string; border: string }> = {
    Intro: { bg: "rgba(99,102,241,0.15)", text: "#818cf8", border: "rgba(99,102,241,0.3)" },
    Verse: { bg: "rgba(34,197,94,0.15)", text: "#4ade80", border: "rgba(34,197,94,0.3)" },
    Chorus: { bg: "rgba(249,115,22,0.15)", text: "#fb923c", border: "rgba(249,115,22,0.3)" },
    Bridge: { bg: "rgba(236,72,153,0.15)", text: "#f472b6", border: "rgba(236,72,153,0.3)" },
    Drop: { bg: "rgba(239,68,68,0.15)", text: "#f87171", border: "rgba(239,68,68,0.3)" },
    Outro: { bg: "rgba(107,114,128,0.15)", text: "#9ca3af", border: "rgba(107,114,128,0.3)" },
    Buildup: { bg: "rgba(251,191,36,0.15)", text: "#fbbf24", border: "rgba(251,191,36,0.3)" },
};

/* ── Component ── */
export default function BarByBar({ xrayAnalysis, currentTime = 0, onSeek }: BarByBarProps) {
    const bpm = xrayAnalysis.analysis?.arrangement?.tempo_bpm || 120;
    const duration = xrayAnalysis.duration || 0;
    const beatsPerBar = 4;
    const barDuration = (60 / bpm) * beatsPerBar; // seconds per bar
    const totalBars = Math.ceil(duration / barDuration);
    const sections = xrayAnalysis.analysis?.arrangement?.sections || [];
    const layers = xrayAnalysis.layers || [];

    // Pre-compute: for each bar, which stems are active and their intensity
    const barData = useMemo(() => {
        const data: Array<{
            barIndex: number;
            barStart: number;
            barEnd: number;
            section: Section | null;
            stems: Record<string, { active: boolean; intensity: number; count: number; labels: string[] }>;
        }> = [];

        for (let b = 0; b < totalBars; b++) {
            const barStart = b * barDuration;
            const barEnd = Math.min((b + 1) * barDuration, duration);

            // Find section for this bar
            const section = sections.find((s) => barStart >= s.start && barStart < s.end) || null;

            // Check each layer
            const stems: Record<string, { active: boolean; intensity: number; count: number; labels: string[] }> = {};

            for (const layer of layers) {
                const stemName = layer.name;
                let count = 0;
                let totalEnergy = 0;
                const labels: string[] = [];

                for (const evt of layer.events) {
                    if (layer.type === "point") {
                        // Point events: check if event time falls in this bar
                        const t = evt.time ?? 0;
                        if (t >= barStart && t < barEnd) {
                            count++;
                            totalEnergy += evt.energy ?? 0.5;
                            if (evt.label && !labels.includes(evt.label)) {
                                labels.push(evt.label);
                            }
                        }
                    } else {
                        // Region events: check overlap
                        const rStart = evt.start ?? evt.time ?? 0;
                        const rEnd = evt.end ?? (rStart + (evt.duration ?? 0));
                        if (rStart < barEnd && rEnd > barStart) {
                            count++;
                            totalEnergy += evt.energy ?? 0.5;
                            const label = evt.label ?? evt.type ?? "";
                            if (label && !labels.includes(label)) {
                                labels.push(label);
                            }
                        }
                    }
                }

                stems[stemName] = {
                    active: count > 0,
                    intensity: count > 0 ? Math.min(totalEnergy / count, 1) : 0,
                    count,
                    labels,
                };
            }

            data.push({ barIndex: b, barStart, barEnd, section, stems });
        }

        return data;
    }, [totalBars, barDuration, duration, sections, layers]);

    // Current bar highlight
    const currentBar = Math.floor(currentTime / barDuration);

    // Group contiguous bars with same section label
    const sectionSpans = useMemo(() => {
        const spans: Array<{ label: string; startBar: number; endBar: number }> = [];

        for (const bar of barData) {
            const label = bar.section?.label || "";
            if (!label) continue;
            if (spans.length > 0 && spans[spans.length - 1].label === label) {
                spans[spans.length - 1].endBar = bar.barIndex;
            } else {
                spans.push({ label, startBar: bar.barIndex, endBar: bar.barIndex });
            }
        }
        return spans;
    }, [barData]);

    if (totalBars === 0 || layers.length === 0) return null;

    const stemNames = layers.map((l) => l.name);
    const barWidth = 36; // px per bar
    const headerHeight = 36;
    const rowHeight = 28;
    const sectionRowHeight = 24;
    const totalWidth = totalBars * barWidth;
    const totalHeight = headerHeight + sectionRowHeight + stemNames.length * rowHeight;

    return (
        <div className="rounded-xl border border-zinc-800 bg-zinc-900/80 backdrop-blur-sm overflow-hidden">
            {/* Header */}
            <div className="flex items-center justify-between px-4 py-3 border-b border-zinc-800/50">
                <div className="flex items-center gap-2">
                    <span className="text-lg">🎼</span>
                    <span className="text-sm font-semibold text-zinc-200">Bar-by-Bar Reconstruction</span>
                    <span className="text-[10px] text-zinc-500 font-mono">
                        {totalBars} bars @ {bpm.toFixed(0)} BPM
                    </span>
                </div>
                <div className="flex gap-2">
                    {stemNames.map((name) => {
                        const cfg = STEM_CONFIG[name] || { icon: "🎵", color: "#888" };
                        return (
                            <span
                                key={name}
                                className="text-[10px] font-medium px-2 py-0.5 rounded-full"
                                style={{ color: cfg.color, backgroundColor: cfg.bgColor, border: `1px solid ${cfg.borderColor}` }}
                            >
                                {cfg.icon} {name}
                            </span>
                        );
                    })}
                </div>
            </div>

            {/* Scrollable Grid */}
            <div className="overflow-x-auto" style={{ scrollbarWidth: "thin", scrollbarColor: "#333 transparent" }}>
                <div style={{ width: `${totalWidth + 60}px`, minHeight: `${totalHeight + 16}px` }} className="relative p-2">
                    {/* Stem labels (fixed left column) */}
                    <div className="sticky left-0 z-10" style={{ width: "56px" }}>
                        <div style={{ height: `${headerHeight}px` }} />
                        <div style={{ height: `${sectionRowHeight}px` }} />
                        {stemNames.map((name) => {
                            const cfg = STEM_CONFIG[name] || { icon: "🎵", color: "#888" };
                            return (
                                <div
                                    key={name}
                                    style={{ height: `${rowHeight}px`, color: cfg.color }}
                                    className="flex items-center gap-1 text-[10px] font-semibold pr-2"
                                >
                                    <span>{cfg.icon}</span>
                                    <span className="truncate">{name}</span>
                                </div>
                            );
                        })}
                    </div>

                    {/* Grid area */}
                    <div className="absolute left-[56px] top-2 right-0">
                        {/* Bar numbers */}
                        <div className="flex" style={{ height: `${headerHeight}px` }}>
                            {barData.map((bar) => (
                                <div
                                    key={bar.barIndex}
                                    style={{ width: `${barWidth}px` }}
                                    className={`flex items-end justify-center pb-1 text-[9px] font-mono ${bar.barIndex === currentBar
                                            ? "text-amber-400 font-bold"
                                            : bar.barIndex % 4 === 0
                                                ? "text-zinc-400"
                                                : "text-zinc-600"
                                        }`}
                                >
                                    {bar.barIndex % 4 === 0 || bar.barIndex === currentBar
                                        ? bar.barIndex + 1
                                        : "·"
                                    }
                                </div>
                            ))}
                        </div>

                        {/* Section labels row */}
                        <div className="flex relative" style={{ height: `${sectionRowHeight}px` }}>
                            {sectionSpans.map((span, i) => {
                                const colors = SECTION_COLORS[span.label] || SECTION_COLORS.Verse;
                                const left = span.startBar * barWidth;
                                const width = (span.endBar - span.startBar + 1) * barWidth - 2;
                                return (
                                    <div
                                        key={i}
                                        className="absolute rounded-md flex items-center justify-center text-[9px] font-bold tracking-wider uppercase"
                                        style={{
                                            left: `${left}px`,
                                            width: `${width}px`,
                                            height: `${sectionRowHeight - 4}px`,
                                            top: "2px",
                                            backgroundColor: colors.bg,
                                            color: colors.text,
                                            border: `1px solid ${colors.border}`,
                                        }}
                                    >
                                        {span.label}
                                    </div>
                                );
                            })}
                        </div>

                        {/* Stem rows */}
                        {stemNames.map((name) => {
                            const cfg = STEM_CONFIG[name] || { icon: "🎵", color: "#888", bgColor: "rgba(136,136,136,0.15)" };
                            return (
                                <div key={name} className="flex" style={{ height: `${rowHeight}px` }}>
                                    {barData.map((bar) => {
                                        const stem = bar.stems[name];
                                        const isActive = stem?.active;
                                        const intensity = stem?.intensity || 0;
                                        const isCurrent = bar.barIndex === currentBar;

                                        return (
                                            <div
                                                key={bar.barIndex}
                                                style={{ width: `${barWidth}px` }}
                                                className={`flex items-center justify-center p-0.5 cursor-pointer transition-all ${isCurrent ? "ring-1 ring-amber-500/50" : ""
                                                    }`}
                                                onClick={() => onSeek?.(bar.barStart)}
                                                title={
                                                    isActive
                                                        ? `${name}: ${stem.count} events${stem.labels.length > 0 ? ` (${stem.labels.slice(0, 3).join(", ")})` : ""}`
                                                        : `${name}: —`
                                                }
                                            >
                                                <div
                                                    className="w-full h-5 rounded-[3px] transition-all"
                                                    style={{
                                                        backgroundColor: isActive
                                                            ? `${cfg.color}${Math.round(0.15 + intensity * 0.55).toString(16).padStart(2, "0")}`
                                                            : "rgba(39,39,42,0.3)",
                                                        border: isActive
                                                            ? `1px solid ${cfg.color}${Math.round(0.2 + intensity * 0.4).toString(16).padStart(2, "0")}`
                                                            : "1px solid rgba(39,39,42,0.15)",
                                                        boxShadow: isActive && intensity > 0.7
                                                            ? `0 0 6px ${cfg.color}30`
                                                            : "none",
                                                    }}
                                                />
                                            </div>
                                        );
                                    })}
                                </div>
                            );
                        })}
                    </div>

                    {/* Playhead line */}
                    {currentTime > 0 && (
                        <div
                            className="absolute top-0 bottom-0 w-px bg-amber-400/80 z-20 pointer-events-none"
                            style={{
                                left: `${56 + (currentTime / barDuration) * barWidth}px`,
                                boxShadow: "0 0 4px rgba(245,158,11,0.5)",
                            }}
                        />
                    )}
                </div>
            </div>

            {/* Footer summary */}
            <div className="flex items-center gap-4 px-4 py-2 border-t border-zinc-800/50 text-[10px] text-zinc-500">
                <span>{sections.length} sections detected</span>
                <span>Click any bar to seek</span>
                {xrayAnalysis.analysis?.drums?.total_hits && (
                    <span>🥁 {xrayAnalysis.analysis.drums.total_hits} hits</span>
                )}
                {xrayAnalysis.analysis?.bass?.type && (
                    <span>🎸 {xrayAnalysis.analysis.bass.type}</span>
                )}
                {xrayAnalysis.analysis?.vocals?.total_regions && (
                    <span>🎤 {xrayAnalysis.analysis.vocals.total_regions} regions</span>
                )}
            </div>
        </div>
    );
}
