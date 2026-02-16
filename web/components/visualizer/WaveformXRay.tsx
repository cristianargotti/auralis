'use client';

import { useEffect, useRef, useState, useCallback } from 'react';
import WaveSurfer from 'wavesurfer.js';
import RegionsPlugin from 'wavesurfer.js/dist/plugins/regions.esm.js';
import TimelinePlugin from 'wavesurfer.js/dist/plugins/timeline.esm.js';
import ZoomPlugin from 'wavesurfer.js/dist/plugins/zoom.esm.js';

interface WaveformXRayProps {
    jobId: string;
}

interface Event {
    time?: number;
    start?: number;
    end?: number;
    label: string;
    energy?: number;
}

interface Layer {
    name: string;
    type: 'point' | 'region';
    events: Event[];
}

interface WaveformData {
    duration: number;
    waveform: number[];
    layers: Layer[];
}

export default function WaveformXRay({ jobId }: WaveformXRayProps) {
    const containerRef = useRef<HTMLDivElement>(null);
    const [wavesurfer, setWavesurfer] = useState<WaveSurfer | null>(null);
    const [isPlaying, setIsPlaying] = useState(false);
    const [data, setData] = useState<WaveformData | null>(null);
    const [loading, setLoading] = useState(true);
    const [zoom, setZoom] = useState(10);
    const [audioSource, setAudioSource] = useState<'original' | 'mix'>('original');
    const regionsRef = useRef<any>(null);

    // Fetch X-Ray Data
    useEffect(() => {
        const fetchData = async () => {
            try {
                const res = await fetch(`${process.env.NEXT_PUBLIC_API_URL || ''}/api/reconstruct/waveform/${jobId}`);
                if (res.ok) {
                    const json = await res.json();
                    setData(json);
                }
            } catch (e) {
                console.error("Failed to fetch waveform data", e);
            } finally {
                setLoading(false);
            }
        };
        fetchData();
    }, [jobId]);

    // Initialize WaveSurfer
    useEffect(() => {
        if (!containerRef.current || !data) return;

        const ws = WaveSurfer.create({
            container: containerRef.current,
            waveColor: '#4f46e5', // Indigo-600
            progressColor: '#818cf8', // Indigo-400
            cursorColor: '#c7d2fe', // Indigo-200
            barWidth: 2,
            barGap: 1,
            barRadius: 2,
            height: 128,
            normalize: true,
            minPxPerSec: zoom,
            url: `${process.env.NEXT_PUBLIC_API_URL || ''}/api/reconstruct/audio/${jobId}/${audioSource}`,
            peaks: [data.waveform], // Use pre-computed peaks for instant render
        });

        // Plugins
        const wsRegions = ws.registerPlugin(RegionsPlugin.create());
        regionsRef.current = wsRegions;

        ws.registerPlugin(TimelinePlugin.create({
            container: '#waveform-timeline',
            primaryLabelInterval: 10,
            secondaryLabelInterval: 1,
            style: {
                fontSize: '10px',
                color: '#6b7280',
            },
        }));

        // Event Listeners
        ws.on('play', () => setIsPlaying(true));
        ws.on('pause', () => setIsPlaying(false));
        ws.on('finish', () => setIsPlaying(false));

        setWavesurfer(ws);

        return () => {
            ws.destroy();
        };
    }, [data, jobId, audioSource]); // Re-init on source change/data load

    // Add Regions (Layers)
    useEffect(() => {
        if (!wavesurfer || !data || !regionsRef.current) return;

        regionsRef.current.clearRegions();

        data.layers.forEach((layer) => {
            const color =
                layer.name === 'Drums' ? 'rgba(239, 68, 68, 0.2)' : // Red
                    layer.name === 'Bass' ? 'rgba(245, 158, 11, 0.2)' : // Amber
                        layer.name === 'Vocals' ? 'rgba(16, 185, 129, 0.2)' : // Emerald
                            'rgba(99, 102, 241, 0.2)'; // Indigo

            layer.events.forEach((event) => {
                if (layer.type === 'point' && event.time !== undefined) {
                    // Point event (Kick/Snare) -> create small region
                    regionsRef.current.addRegion({
                        start: event.time,
                        end: event.time + 0.1,
                        content: event.label,
                        color: color,
                        drag: false,
                        resize: false,
                    });
                } else if (layer.type === 'region' && event.start !== undefined && event.end !== undefined) {
                    // Range event (Vocals)
                    regionsRef.current.addRegion({
                        start: event.start,
                        end: event.end,
                        content: event.label,
                        color: color,
                        drag: false,
                        resize: false,
                    });
                }
            });
        });

    }, [wavesurfer, data]);

    // Handle Zoom
    useEffect(() => {
        if (wavesurfer) {
            wavesurfer.zoom(zoom);
        }
    }, [zoom, wavesurfer]);

    const togglePlay = () => wavesurfer?.playPause();

    if (loading) return <div className="p-4 text-center text-gray-400">Loading X-Ray data...</div>;
    if (!data) return <div className="p-4 text-center text-red-400">Failed to load X-Ray data.</div>;

    return (
        <div className="w-full bg-zinc-900/50 border border-zinc-800 rounded-xl p-4 space-y-4">
            <div className="flex items-center justify-between">
                <h3 className="text-lg font-semibold text-zinc-100 flex items-center gap-2">
                    <span className="text-xl">🩻</span> Sonic X-Ray
                </h3>

                <div className="flex items-center gap-2 bg-zinc-800 p-1 rounded-lg">
                    <button
                        onClick={() => setAudioSource('original')}
                        className={`px-3 py-1 rounded text-xs font-medium transition-colors ${audioSource === 'original' ? 'bg-indigo-600 text-white' : 'text-zinc-400 hover:text-white'}`}
                    >
                        Original
                    </button>
                    <button
                        onClick={() => setAudioSource('mix')}
                        className={`px-3 py-1 rounded text-xs font-medium transition-colors ${audioSource === 'mix' ? 'bg-emerald-600 text-white' : 'text-zinc-400 hover:text-white'}`}
                    >
                        Reconstructed
                    </button>
                </div>
            </div>

            <div className="relative group">
                <div id="waveform" ref={containerRef} className="rounded-lg overflow-hidden" />
                <div id="waveform-timeline" className="mt-1" />

                {/* Play Overlay */}
                {!isPlaying && (
                    <div
                        className="absolute inset-0 flex items-center justify-center bg-black/10 cursor-pointer"
                        onClick={togglePlay}
                    >
                        <div className="w-12 h-12 bg-white/90 rounded-full flex items-center justify-center shadow-lg hover:scale-105 transition-transform">
                            <svg className="w-6 h-6 text-black ml-1" fill="currentColor" viewBox="0 0 24 24"><path d="M8 5v14l11-7z" /></svg>
                        </div>
                    </div>
                )}
            </div>

            {/* Controls */}
            <div className="flex items-center gap-4 text-xs text-zinc-400">
                <button onClick={togglePlay} className="hover:text-white">
                    {isPlaying ? 'Pause' : 'Play'}
                </button>
                <span className="text-zinc-600">|</span>
                <div className="flex items-center gap-2 flex-1">
                    <span>Zoom</span>
                    <input
                        type="range"
                        min="10"
                        max="200"
                        value={zoom}
                        onChange={(e) => setZoom(Number(e.target.value))}
                        className="w-32 h-1 bg-zinc-700 rounded-lg appearance-none cursor-pointer"
                    />
                </div>

                <div className="flex items-center gap-2">
                    <div className="flex items-center gap-1">
                        <span className="w-2 h-2 rounded-full bg-red-500/50"></span> Drums
                    </div>
                    <div className="flex items-center gap-1">
                        <span className="w-2 h-2 rounded-full bg-amber-500/50"></span> Bass
                    </div>
                    <div className="flex items-center gap-1">
                        <span className="w-2 h-2 rounded-full bg-emerald-500/50"></span> Vocals
                    </div>
                </div>
            </div>
        </div>
    );
}
