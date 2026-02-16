"use client";

import { useCallback, useEffect, useState } from "react";
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

interface SynthPreset {
    name: string;
    description: string;
    oscillators: number;
    unison: number;
    has_filter: boolean;
}

interface EffectChain {
    name: string;
    has_eq: boolean;
    has_compressor: boolean;
    has_distortion: boolean;
    has_reverb: boolean;
    has_delay: boolean;
    has_chorus: boolean;
    has_sidechain: boolean;
}

interface ScaleInfo {
    intervals: number[];
    notes: number;
}

interface GenreInfo {
    sections: number;
    section_types: string[];
    total_bars: number;
}

export default function StudioPage() {
    const [presets, setPresets] = useState<Record<string, SynthPreset>>({});
    const [chains, setChains] = useState<Record<string, EffectChain>>({});
    const [scales, setScales] = useState<Record<string, ScaleInfo>>({});
    const [genres, setGenres] = useState<Record<string, GenreInfo>>({});
    const [loading, setLoading] = useState(true);
    const [synthResult, setSynthResult] = useState<string | null>(null);
    const [selectedPreset, setSelectedPreset] = useState("supersaw");
    const [freq, setFreq] = useState(440);

    useEffect(() => {
        const load = async () => {
            try {
                const [p, c, s, g] = await Promise.all([
                    api<Record<string, SynthPreset>>("/api/hands/presets"),
                    api<Record<string, EffectChain>>("/api/hands/effect-chains"),
                    api<Record<string, ScaleInfo>>("/api/grid/scales"),
                    api<Record<string, GenreInfo>>("/api/grid/genres"),
                ]);
                setPresets(p);
                setChains(c);
                setScales(s);
                setGenres(g);
            } catch {
                // API not available
            } finally {
                setLoading(false);
            }
        };
        load();
    }, []);

    const handleSynth = useCallback(async () => {
        try {
            const res = await api<{ path: string; preset: string; samples: number }>(
                "/api/hands/synth",
                {
                    method: "POST",
                    body: JSON.stringify({
                        freq_hz: freq,
                        duration_s: 2.0,
                        preset: selectedPreset,
                    }),
                }
            );
            setSynthResult(`✅ Generated: ${res.preset} @ ${freq}Hz (${res.samples} samples)`);
        } catch {
            setSynthResult("❌ Failed to synthesize");
        }
    }, [freq, selectedPreset]);

    if (loading) {
        return (
            <div className="flex items-center justify-center h-64">
                <div className="w-8 h-8 border-2 border-emerald-500 border-t-transparent rounded-full animate-spin" />
            </div>
        );
    }

    return (
        <div className="flex flex-col gap-6">
            <div>
                <h1 className="text-3xl font-bold tracking-tight">🎹 Studio</h1>
                <p className="text-muted-foreground mt-1">
                    Synthesizers, effects, and sound design tools
                </p>
            </div>

            {/* Synth Section */}
            <Card>
                <CardHeader>
                    <CardTitle>Synthesizer</CardTitle>
                    <CardDescription>Generate sounds from presets</CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                    <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                        {Object.entries(presets).map(([key, preset]) => (
                            <button
                                key={key}
                                onClick={() => setSelectedPreset(key)}
                                className={`text-left p-3 rounded-lg border transition ${selectedPreset === key
                                        ? "border-emerald-500 bg-emerald-500/10"
                                        : "border-zinc-700 bg-zinc-900 hover:border-zinc-600"
                                    }`}
                            >
                                <p className="font-bold text-sm">{preset.name}</p>
                                <p className="text-xs text-zinc-500 mt-1">{preset.description}</p>
                                <div className="flex gap-1 mt-2">
                                    <Badge variant="secondary" className="text-[10px]">
                                        {preset.oscillators} osc
                                    </Badge>
                                    {preset.unison > 1 && (
                                        <Badge variant="secondary" className="text-[10px]">
                                            {preset.unison}x unison
                                        </Badge>
                                    )}
                                    {preset.has_filter && (
                                        <Badge variant="secondary" className="text-[10px]">filter</Badge>
                                    )}
                                </div>
                            </button>
                        ))}
                    </div>

                    <div className="flex gap-3 items-end">
                        <div>
                            <label className="text-xs text-zinc-500 block mb-1">Frequency (Hz)</label>
                            <input
                                type="number"
                                className="w-32 bg-zinc-900 border border-zinc-700 rounded-lg px-3 py-2 text-white text-sm"
                                value={freq}
                                onChange={(e) => setFreq(Number(e.target.value))}
                                min={20}
                                max={20000}
                            />
                        </div>
                        <Button onClick={handleSynth} className="bg-emerald-600 hover:bg-emerald-700">
                            🔊 Generate
                        </Button>
                    </div>

                    {synthResult && (
                        <p className="text-sm text-zinc-400">{synthResult}</p>
                    )}
                </CardContent>
            </Card>

            {/* Effects Section */}
            <Card>
                <CardHeader>
                    <CardTitle>Effect Chains</CardTitle>
                    <CardDescription>Pre-built processing chains</CardDescription>
                </CardHeader>
                <CardContent>
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                        {Object.entries(chains).map(([key, chain]) => (
                            <div key={key} className="p-3 rounded-lg border border-zinc-700 bg-zinc-900">
                                <p className="font-bold text-sm">{chain.name}</p>
                                <div className="flex flex-wrap gap-1 mt-2">
                                    {chain.has_eq && <Badge className="bg-blue-500/20 text-blue-400 text-[10px]">EQ</Badge>}
                                    {chain.has_compressor && <Badge className="bg-amber-500/20 text-amber-400 text-[10px]">Comp</Badge>}
                                    {chain.has_distortion && <Badge className="bg-red-500/20 text-red-400 text-[10px]">Dist</Badge>}
                                    {chain.has_chorus && <Badge className="bg-purple-500/20 text-purple-400 text-[10px]">Chorus</Badge>}
                                    {chain.has_delay && <Badge className="bg-cyan-500/20 text-cyan-400 text-[10px]">Delay</Badge>}
                                    {chain.has_reverb && <Badge className="bg-indigo-500/20 text-indigo-400 text-[10px]">Reverb</Badge>}
                                    {chain.has_sidechain && <Badge className="bg-emerald-500/20 text-emerald-400 text-[10px]">SC</Badge>}
                                </div>
                            </div>
                        ))}
                    </div>
                </CardContent>
            </Card>

            {/* Music Theory / Grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <Card>
                    <CardHeader>
                        <CardTitle>Scales</CardTitle>
                        <CardDescription>{Object.keys(scales).length} available</CardDescription>
                    </CardHeader>
                    <CardContent>
                        <div className="grid grid-cols-2 gap-2">
                            {Object.entries(scales).map(([name, info]) => (
                                <div key={name} className="flex items-center justify-between bg-zinc-900 rounded-lg px-3 py-2">
                                    <span className="text-sm">{name.replace("_", " ")}</span>
                                    <Badge variant="secondary" className="text-[10px]">{info.notes} notes</Badge>
                                </div>
                            ))}
                        </div>
                    </CardContent>
                </Card>

                <Card>
                    <CardHeader>
                        <CardTitle>Genre Templates</CardTitle>
                        <CardDescription>{Object.keys(genres).length} available</CardDescription>
                    </CardHeader>
                    <CardContent>
                        <div className="space-y-3">
                            {Object.entries(genres).map(([name, info]) => (
                                <div key={name} className="bg-zinc-900 rounded-lg p-3">
                                    <div className="flex items-center justify-between mb-2">
                                        <span className="font-bold text-sm capitalize">{name.replace("_", " ")}</span>
                                        <Badge variant="secondary" className="text-[10px]">{info.total_bars} bars</Badge>
                                    </div>
                                    <div className="flex flex-wrap gap-1">
                                        {info.section_types.map((s, i) => (
                                            <Badge key={i} variant="outline" className="text-[10px]">{s}</Badge>
                                        ))}
                                    </div>
                                </div>
                            ))}
                        </div>
                    </CardContent>
                </Card>
            </div>
        </div>
    );
}
