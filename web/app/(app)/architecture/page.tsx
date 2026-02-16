"use client";

import { useState } from "react";
import {
    Card,
    CardContent,
    CardHeader,
    CardTitle,
} from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";

/* ── Architecture Data ─────────────────────────────── */

const LAYERS = [
    {
        icon: "👂",
        name: "EAR",
        title: "Deconstrucción Inteligente",
        color: "from-cyan-500 to-blue-600",
        border: "border-cyan-500/30",
        badge: "text-cyan-400",
        tools: [
            { name: "Mel-RoFormer", desc: "Separación de stems — 9.96 dB SDR, mejor calidad disponible", status: "planned" },
            { name: "HTDemucs v4", desc: "Fallback separator — 7.68 dB SDR, ya implementado", status: "ready" },
            { name: "Basic-Pitch", desc: "Spotify — MIDI polifónico, multi-instrumento, pitch bend", status: "planned" },
            { name: "Profiler", desc: "BPM, key, sections, FX detection, energy map", status: "ready" },
        ],
    },
    {
        icon: "🎹",
        name: "HANDS",
        title: "Síntesis Inteligente",
        color: "from-purple-500 to-pink-600",
        border: "border-purple-500/30",
        badge: "text-purple-400",
        tools: [
            { name: "DawDreamer", desc: "VST host en Python — carga Surge XT, Vital, cualquier VST", status: "ready" },
            { name: "TokenSynth", desc: "Zero-shot instrument cloning via codec language model", status: "planned" },
            { name: "RAVE", desc: "Real-time timbre transfer — disponible como VST plugin", status: "planned" },
            { name: "Faust DSP", desc: "DSP custom integrado en DawDreamer", status: "ready" },
            { name: "JAX + Flax", desc: "Optimización ML de parámetros de síntesis", status: "planned" },
        ],
    },
    {
        icon: "🎚️",
        name: "CONSOLE",
        title: "Mezcla y Master",
        color: "from-amber-500 to-orange-600",
        border: "border-amber-500/30",
        badge: "text-amber-400",
        tools: [
            { name: "Pedalboard", desc: "Spotify — EQ, comp, reverb, delay, chorus, limiter", status: "ready" },
            { name: "matchering", desc: "Master automático por referencia — EQ, RMS, peak, stereo", status: "planned" },
            { name: "Sidechain Replicator", desc: "Detecta pump ratio del original y lo replica", status: "planned" },
            { name: "Stereo Width Matcher", desc: "Replica S/M ratio por sección", status: "planned" },
        ],
    },
    {
        icon: "📐",
        name: "GRID",
        title: "Composición y Arreglo",
        color: "from-emerald-500 to-teal-600",
        border: "border-emerald-500/30",
        badge: "text-emerald-400",
        tools: [
            { name: "mido", desc: "Generación/edición MIDI programática", status: "ready" },
            { name: "musicpy", desc: "Teoría musical — escalas, acordes, progresiones", status: "ready" },
            { name: "Auto-Structure", desc: "Detección automática de secciones por energía RMS", status: "planned" },
            { name: "Pattern Matcher", desc: "Detección de repeticiones en MIDI", status: "planned" },
        ],
    },
    {
        icon: "🧠",
        name: "BRAIN",
        title: "Orquestador LLM",
        color: "from-pink-500 to-rose-600",
        border: "border-pink-500/30",
        badge: "text-pink-400",
        tools: [
            { name: "GPT-4o / Claude", desc: "LLM para decisiones de producción", status: "ready" },
            { name: "Analysis Context", desc: "Recibe análisis EAR completo como JSON", status: "ready" },
            { name: "Iteration Loop", desc: "Reconstruir → QC → ajustar → repetir hasta ≥90%", status: "planned" },
        ],
    },
    {
        icon: "🔍",
        name: "QC",
        title: "Quality Control — 12 Dimensiones",
        color: "from-orange-500 to-red-600",
        border: "border-orange-500/30",
        badge: "text-orange-400",
        tools: [
            { name: "Spectral Correlation", desc: "Comparación de espectrogramas Mel", status: "planned" },
            { name: "MERT Embeddings", desc: "Timbre similarity via cosine similarity", status: "planned" },
            { name: "RMS Match", desc: "dB difference por sección detectada", status: "ready" },
            { name: "Energy Curve", desc: "Correlación de curva RMS (≥0.90 target)", status: "planned" },
            { name: "Arrangement Match", desc: "Section boundary alignment", status: "planned" },
            { name: "Stereo/Bass/Kick", desc: "S/M, MIDI, onset detection", status: "planned" },
        ],
    },
];

const PHASES = [
    { num: 1, name: "EAR Upgrade", desc: "Mel-RoFormer + Basic-Pitch", status: "next" },
    { num: 2, name: "Auto-Structure", desc: "Detección automática de secciones", status: "pending" },
    { num: 3, name: "Synthesis Pipeline", desc: "TokenSynth + DawDreamer + RAVE", status: "pending" },
    { num: 4, name: "Console Upgrade", desc: "matchering + auto FX replication", status: "pending" },
    { num: 5, name: "QC 12D", desc: "12 dimensiones + MERT", status: "pending" },
    { num: 6, name: "BRAIN Loop", desc: "Iteración automática ≥90%", status: "pending" },
    { num: 7, name: "UI Pipeline", desc: "Upload → live progress → A/B", status: "pending" },
];

/* ── Component ──────────────────────────────────────── */

export default function ArchitecturePage() {
    const [expandedLayer, setExpandedLayer] = useState<string | null>(null);

    const getStatusBadge = (status: string) => {
        switch (status) {
            case "ready":
                return <Badge variant="outline" className="text-[9px] border-emerald-500/30 text-emerald-400">Ready</Badge>;
            case "planned":
                return <Badge variant="outline" className="text-[9px] border-amber-500/30 text-amber-400">Planned</Badge>;
            default:
                return <Badge variant="outline" className="text-[9px] border-zinc-700 text-zinc-500">—</Badge>;
        }
    };

    return (
        <div className="space-y-6 p-6">
            {/* Header */}
            <div>
                <h1 className="text-3xl font-bold bg-gradient-to-r from-cyan-400 via-purple-500 to-pink-500 bg-clip-text text-transparent">
                    🏗️ Reconstruction Engine
                </h1>
                <p className="text-zinc-400 mt-1">
                    Track-agnostic — upload ANY track → deconstruct → reconstruct → match quality
                </p>
            </div>

            {/* Pipeline Flow */}
            <Card className="bg-zinc-900/50 border-zinc-800">
                <CardHeader>
                    <CardTitle className="text-sm text-zinc-400">Pipeline Flow</CardTitle>
                </CardHeader>
                <CardContent>
                    <div className="flex items-center gap-2 flex-wrap text-sm">
                        {["Upload ANY track", "👂 EAR", "🧠 BRAIN", "📐 GRID", "🎹 HANDS", "🎚️ CONSOLE", "🔍 QC", "♻️ Iterate"].map((step, i) => (
                            <div key={i} className="flex items-center gap-2">
                                <span className={`px-3 py-1.5 rounded-lg ${i === 0 ? "bg-zinc-700 text-white" : i === 7 ? "bg-emerald-500/20 text-emerald-400 border border-emerald-500/30" : "bg-zinc-800 text-zinc-300"}`}>
                                    {step}
                                </span>
                                {i < 7 && <span className="text-zinc-600">→</span>}
                            </div>
                        ))}
                    </div>
                </CardContent>
            </Card>

            {/* 6 Layers */}
            <div className="space-y-3">
                {LAYERS.map((layer) => (
                    <Card
                        key={layer.name}
                        className={`bg-zinc-900/50 border cursor-pointer transition-all ${expandedLayer === layer.name ? layer.border : "border-zinc-800 hover:border-zinc-700"
                            }`}
                        onClick={() => setExpandedLayer(expandedLayer === layer.name ? null : layer.name)}
                    >
                        <CardHeader className="pb-2">
                            <div className="flex items-center justify-between">
                                <div className="flex items-center gap-3">
                                    <span className="text-2xl">{layer.icon}</span>
                                    <div>
                                        <CardTitle className={`text-lg bg-gradient-to-r ${layer.color} bg-clip-text text-transparent`}>
                                            {layer.name}
                                        </CardTitle>
                                        <p className="text-xs text-zinc-500">{layer.title}</p>
                                    </div>
                                </div>
                                <div className="flex gap-1">
                                    <Badge variant="outline" className={`text-[9px] ${layer.badge}`}>
                                        {layer.tools.filter(t => t.status === "ready").length}/{layer.tools.length} ready
                                    </Badge>
                                </div>
                            </div>
                        </CardHeader>
                        {expandedLayer === layer.name && (
                            <CardContent>
                                <div className="space-y-2">
                                    {layer.tools.map((tool) => (
                                        <div
                                            key={tool.name}
                                            className="flex items-center justify-between rounded-lg bg-zinc-800/50 px-3 py-2"
                                        >
                                            <div>
                                                <span className="text-sm font-medium text-zinc-200">{tool.name}</span>
                                                <p className="text-[11px] text-zinc-500">{tool.desc}</p>
                                            </div>
                                            {getStatusBadge(tool.status)}
                                        </div>
                                    ))}
                                </div>
                            </CardContent>
                        )}
                    </Card>
                ))}
            </div>

            {/* Execution Phases */}
            <Card className="bg-zinc-900/50 border-zinc-800">
                <CardHeader>
                    <CardTitle className="text-lg">🚀 Execution Roadmap</CardTitle>
                </CardHeader>
                <CardContent>
                    <div className="space-y-2">
                        {PHASES.map((phase) => (
                            <div
                                key={phase.num}
                                className={`flex items-center gap-4 rounded-lg px-4 py-3 ${phase.status === "next"
                                        ? "bg-amber-500/10 border border-amber-500/30"
                                        : "bg-zinc-800/30"
                                    }`}
                            >
                                <div className={`text-lg font-bold ${phase.status === "next" ? "text-amber-400" : "text-zinc-600"}`}>
                                    {phase.num}
                                </div>
                                <div className="flex-1">
                                    <span className={`text-sm font-medium ${phase.status === "next" ? "text-amber-300" : "text-zinc-400"}`}>
                                        {phase.name}
                                    </span>
                                    <p className="text-[11px] text-zinc-600">{phase.desc}</p>
                                </div>
                                {phase.status === "next" && (
                                    <Badge className="bg-amber-500/20 text-amber-400 border border-amber-500/30 text-[9px]">
                                        NEXT
                                    </Badge>
                                )}
                            </div>
                        ))}
                    </div>
                </CardContent>
            </Card>

            {/* Design Principle */}
            <div className="text-center text-zinc-600 text-xs py-4 border-t border-zinc-800/50">
                100% Track-Agnostic — Zero hardcoded — Upload any track, get exact reconstruction
            </div>
        </div>
    );
}
