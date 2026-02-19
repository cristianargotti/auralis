'use client';

import { useEffect, useRef, useState, useCallback, Component, ReactNode } from 'react';
import type WaveSurferType from 'wavesurfer.js';
import { api } from '@/lib/api';

/* ── Error Boundary ─────────────────────────────────── */

interface ErrorBoundaryProps {
    children: ReactNode;
    fallback?: ReactNode;
}

interface ErrorBoundaryState {
    hasError: boolean;
    error?: Error;
}

class WaveformErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
    constructor(props: ErrorBoundaryProps) {
        super(props);
        this.state = { hasError: false };
    }

    static getDerivedStateFromError(error: Error): ErrorBoundaryState {
        return { hasError: true, error };
    }

    componentDidCatch(error: Error, info: React.ErrorInfo) {
        console.error('[WaveformXRay] Error boundary caught:', error, info);
    }

    render() {
        if (this.state.hasError) {
            return this.props.fallback || (
                <div className="p-6 text-center bg-gradient-to-br from-red-950/20 to-zinc-900/50 border border-red-800/20 rounded-2xl">
                    <span className="text-2xl">⚠️</span>
                    <p className="text-sm mt-2 text-red-300">Waveform visualization encountered an error.</p>
                    <p className="text-xs text-zinc-600 mt-1">{this.state.error?.message}</p>
                </div>
            );
        }
        return this.props.children;
    }
}

/* ── Types ───────────────────────────────────────────── */

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

/* ── Helpers ─────────────────────────────────────────── */

function formatTime(secs: number): string {
    const m = Math.floor(secs / 60);
    const s = Math.floor(secs % 60);
    return `${m}:${s.toString().padStart(2, '0')}`;
}

/* ── Stem Colors ─────────────────────────────────────── */

const STEM_COLORS: Record<string, { bg: string; dot: string; border: string; region: string }> = {
    Drums: { bg: 'bg-red-500/10', dot: 'bg-red-500', border: 'border-red-500/20', region: 'rgba(239, 68, 68, 0.15)' },
    Bass: { bg: 'bg-amber-500/10', dot: 'bg-amber-500', border: 'border-amber-500/20', region: 'rgba(245, 158, 11, 0.15)' },
    Vocals: { bg: 'bg-emerald-500/10', dot: 'bg-emerald-500', border: 'border-emerald-500/20', region: 'rgba(16, 185, 129, 0.15)' },
    Other: { bg: 'bg-indigo-500/10', dot: 'bg-indigo-500', border: 'border-indigo-500/20', region: 'rgba(99, 102, 241, 0.15)' },
};

/* ── Inner Component (wrapped by Error Boundary) ───── */

function WaveformXRayInner({ jobId }: WaveformXRayProps) {
    const containerRef = useRef<HTMLDivElement>(null);
    const [wavesurfer, setWavesurfer] = useState<WaveSurferType | null>(null);
    const [isPlaying, setIsPlaying] = useState(false);
    const [data, setData] = useState<WaveformData | null>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [zoom, setZoom] = useState(10);
    const [audioSource, setAudioSource] = useState<'original' | 'mix'>('original');
    const [currentTime, setCurrentTime] = useState(0);
    const [duration, setDuration] = useState(0);
    const [volume, setVolume] = useState(0.8);
    const regionsRef = useRef<any>(null);
    const animRef = useRef<number | null>(null);

    // Fetch X-Ray Data
    useEffect(() => {
        const fetchData = async () => {
            try {
                setError(null);
                const json = await api<WaveformData>(`/api/reconstruct/waveform/${jobId}`);
                if (!json || typeof json.duration !== 'number') {
                    setError('Invalid waveform data received');
                    return;
                }
                if (!Array.isArray(json.waveform) || json.waveform.length === 0) {
                    json.waveform = new Array(200).fill(0);
                }
                if (!Array.isArray(json.layers)) {
                    json.layers = [];
                }
                setData(json);
                setDuration(json.duration);
            } catch (e: any) {
                console.error("[WaveformXRay] Failed to fetch waveform data:", e);
                setError(e?.message || 'Failed to load waveform data');
            } finally {
                setLoading(false);
            }
        };
        fetchData();
    }, [jobId]);

    // Initialize WaveSurfer
    useEffect(() => {
        if (!containerRef.current || !data) return;

        let ws: WaveSurferType | null = null;

        const initWaveSurfer = async () => {
            try {
                const WaveSurfer = (await import('wavesurfer.js')).default;
                const RegionsPlugin = (await import('wavesurfer.js/dist/plugins/regions.esm.js')).default;
                const TimelinePlugin = (await import('wavesurfer.js/dist/plugins/timeline.esm.js')).default;

                if (!containerRef.current) return;

                const token = typeof window !== 'undefined' ? localStorage.getItem('auralis_token') : null;
                const authHeaders = token ? { Authorization: `Bearer ${token}` } : {};
                const safePeaks = data.waveform.length > 0 ? data.waveform : new Array(200).fill(0);

                // Create gradient for waveform
                const ctx = document.createElement('canvas').getContext('2d')!;
                const gradient = ctx.createLinearGradient(0, 0, 0, 150);
                gradient.addColorStop(0, '#a78bfa');    // violet-400
                gradient.addColorStop(0.3, '#7c3aed');  // violet-600
                gradient.addColorStop(0.7, '#4f46e5');  // indigo-600
                gradient.addColorStop(1, '#3730a3');    // indigo-800

                const progressGradient = ctx.createLinearGradient(0, 0, 0, 150);
                progressGradient.addColorStop(0, '#c4b5fd');   // violet-300
                progressGradient.addColorStop(0.3, '#a78bfa'); // violet-400
                progressGradient.addColorStop(0.7, '#8b5cf6'); // violet-500
                progressGradient.addColorStop(1, '#7c3aed');   // violet-600

                ws = WaveSurfer.create({
                    container: containerRef.current,
                    waveColor: gradient,
                    progressColor: progressGradient,
                    cursorColor: '#e0e7ff',
                    cursorWidth: 2,
                    barWidth: 2,
                    barGap: 1,
                    barRadius: 3,
                    height: 140,
                    normalize: true,
                    minPxPerSec: 10,
                    url: `${process.env.NEXT_PUBLIC_API_URL || ''}/api/reconstruct/audio/${jobId}/original`,
                    fetchParams: {
                        headers: authHeaders as HeadersInit
                    },
                    peaks: [safePeaks],
                });

                // Plugins
                const wsRegions = ws.registerPlugin(RegionsPlugin.create());
                regionsRef.current = wsRegions;

                ws.registerPlugin(TimelinePlugin.create({
                    container: '#xray-timeline',
                    primaryLabelInterval: 10,
                    secondaryLabelInterval: 5,
                    style: {
                        fontSize: '9px',
                        color: '#52525b',
                    },
                }));

                // Events
                ws.on('error', (err) => {
                    console.warn('[WaveformXRay] WaveSurfer error (non-fatal):', err);
                });
                ws.on('play', () => setIsPlaying(true));
                ws.on('pause', () => setIsPlaying(false));
                ws.on('finish', () => setIsPlaying(false));
                ws.on('ready', () => {
                    setDuration(ws?.getDuration() || data.duration);
                    if (ws) ws.setVolume(volume);
                });
                ws.on('timeupdate', (time) => setCurrentTime(time));

                setWavesurfer(ws);
            } catch (err: any) {
                console.error('[WaveformXRay] WaveSurfer initialization failed:', err);
                setError(`Visualization failed: ${err?.message || 'Unknown error'}`);
            }
        };

        initWaveSurfer();

        return () => {
            if (animRef.current) cancelAnimationFrame(animRef.current);
            try { ws?.destroy(); } catch { }
        };
    }, [data, jobId]);  // Only re-init on job change, NOT zoom/source

    // Add Regions (Layers)
    useEffect(() => {
        if (!wavesurfer || !data || !regionsRef.current) return;
        try {
            regionsRef.current.clearRegions();
            data.layers.forEach((layer) => {
                const colors = STEM_COLORS[layer.name] || STEM_COLORS.Other;
                (layer.events || []).forEach((event) => {
                    if (layer.type === 'point' && event.time !== undefined && isFinite(event.time)) {
                        regionsRef.current.addRegion({
                            start: event.time,
                            end: event.time + 0.1,
                            content: event.label,
                            color: colors.region,
                            drag: false,
                            resize: false,
                        });
                    } else if (layer.type === 'region' && event.start !== undefined && event.end !== undefined && isFinite(event.start) && isFinite(event.end)) {
                        regionsRef.current.addRegion({
                            start: event.start,
                            end: event.end,
                            content: event.label,
                            color: colors.region,
                            drag: false,
                            resize: false,
                        });
                    }
                });
            });
        } catch (err) {
            console.warn('[WaveformXRay] Error adding regions (non-fatal):', err);
        }
    }, [wavesurfer, data]);

    // Handle Zoom — just update zoom level, don't recreate
    useEffect(() => {
        if (wavesurfer) {
            try { wavesurfer.zoom(zoom); } catch { }
        }
    }, [zoom, wavesurfer]);

    // Handle Audio Source change — load new URL without recreating WaveSurfer
    useEffect(() => {
        if (wavesurfer) {
            const token = typeof window !== 'undefined' ? localStorage.getItem('auralis_token') : null;
            const url = `${process.env.NEXT_PUBLIC_API_URL || ''}/api/reconstruct/audio/${jobId}/${audioSource}`;
            try {
                wavesurfer.load(url, data?.waveform ? [data.waveform] : undefined);
            } catch (err) {
                console.warn('[WaveformXRay] Failed to load new source:', err);
            }
        }
    }, [audioSource]);  // eslint-disable-line react-hooks/exhaustive-deps

    // Handle Volume
    useEffect(() => {
        if (wavesurfer) {
            try { wavesurfer.setVolume(volume); } catch { }
        }
    }, [volume, wavesurfer]);

    const togglePlay = useCallback(() => {
        try { wavesurfer?.playPause(); } catch { }
    }, [wavesurfer]);

    const skipForward = useCallback(() => {
        if (wavesurfer) {
            try { wavesurfer.skip(5); } catch { }
        }
    }, [wavesurfer]);

    const skipBackward = useCallback(() => {
        if (wavesurfer) {
            try { wavesurfer.skip(-5); } catch { }
        }
    }, [wavesurfer]);

    // ── Loading State ──
    if (loading) {
        return (
            <div className="w-full bg-gradient-to-br from-zinc-900/80 to-zinc-950/80 border border-zinc-800/50 rounded-2xl p-8">
                <div className="flex flex-col items-center gap-3">
                    <div className="w-10 h-10 rounded-full border-2 border-violet-500/30 border-t-violet-400 animate-spin" />
                    <p className="text-sm text-zinc-500">Loading Sonic X-Ray...</p>
                </div>
            </div>
        );
    }

    // ── Error State ──
    if (error || !data) {
        return (
            <div className="w-full bg-gradient-to-br from-zinc-900/80 to-zinc-950/80 border border-zinc-800/50 rounded-2xl p-8 text-center space-y-3">
                <div className="w-12 h-12 mx-auto rounded-xl bg-amber-500/10 flex items-center justify-center">
                    <span className="text-2xl">🩻</span>
                </div>
                <p className="text-zinc-300 text-sm font-medium">
                    {error || 'X-Ray data not available for this job.'}
                </p>
                <p className="text-zinc-600 text-xs max-w-md mx-auto">
                    This may happen if the server was restarted after reconstruction.
                    Try uploading and reconstructing the track again.
                </p>
            </div>
        );
    }

    // ── Active Stems ──
    const activeStems = data.layers.map(l => l.name);

    return (
        <div className="w-full rounded-2xl overflow-hidden bg-gradient-to-br from-zinc-900/90 via-zinc-900/70 to-violet-950/20 border border-zinc-800/50 backdrop-blur-sm">

            {/* ── Header ── */}
            <div className="flex items-center justify-between px-5 pt-4 pb-2">
                <div className="flex items-center gap-3">
                    <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-violet-500/20 to-indigo-500/20 border border-violet-500/20 flex items-center justify-center">
                        <span className="text-base">🩻</span>
                    </div>
                    <div>
                        <h3 className="text-sm font-semibold text-zinc-100">Sonic X-Ray</h3>
                        <p className="text-[10px] text-zinc-600">Waveform · Stems · Regions</p>
                    </div>
                </div>

                {/* Source Toggle */}
                <div className="flex items-center gap-1 bg-zinc-800/60 p-0.5 rounded-lg border border-zinc-700/30">
                    <button
                        onClick={() => setAudioSource('original')}
                        className={`px-3 py-1.5 rounded-md text-[11px] font-medium transition-all ${audioSource === 'original'
                            ? 'bg-violet-600/80 text-white shadow-lg shadow-violet-500/20'
                            : 'text-zinc-500 hover:text-zinc-300'
                            }`}
                    >
                        Original
                    </button>
                    <button
                        onClick={() => setAudioSource('mix')}
                        className={`px-3 py-1.5 rounded-md text-[11px] font-medium transition-all ${audioSource === 'mix'
                            ? 'bg-emerald-600/80 text-white shadow-lg shadow-emerald-500/20'
                            : 'text-zinc-500 hover:text-zinc-300'
                            }`}
                    >
                        Reconstructed
                    </button>
                </div>
            </div>

            {/* ── Waveform ── */}
            <div className="relative group px-4 pt-2">
                <div
                    id="waveform"
                    ref={containerRef}
                    className="rounded-xl overflow-hidden bg-zinc-950/40 border border-zinc-800/30"
                    style={{ cursor: 'pointer' }}
                />
                <div id="xray-timeline" className="mt-0.5 px-1" />

                {/* Play/Pause Overlay */}
                {!isPlaying && (
                    <div
                        className="absolute inset-0 mx-4 mt-2 flex items-center justify-center bg-black/5 rounded-xl cursor-pointer opacity-0 group-hover:opacity-100 transition-opacity duration-200"
                        onClick={togglePlay}
                    >
                        <div className="w-14 h-14 bg-white/10 backdrop-blur-md rounded-full flex items-center justify-center border border-white/20 shadow-2xl hover:scale-110 hover:bg-white/15 transition-all duration-200">
                            <svg className="w-7 h-7 text-white ml-0.5" fill="currentColor" viewBox="0 0 24 24">
                                <path d="M8 5v14l11-7z" />
                            </svg>
                        </div>
                    </div>
                )}
            </div>

            {/* ── Transport Controls ── */}
            <div className="px-5 py-3">
                <div className="flex items-center gap-3">

                    {/* Time Display */}
                    <div className="flex items-center gap-1.5 min-w-[100px]">
                        <span className="text-xs font-mono text-violet-300 tabular-nums">
                            {formatTime(currentTime)}
                        </span>
                        <span className="text-[10px] text-zinc-700">/</span>
                        <span className="text-xs font-mono text-zinc-600 tabular-nums">
                            {formatTime(duration)}
                        </span>
                    </div>

                    {/* Playback Controls */}
                    <div className="flex items-center gap-1">
                        <button
                            onClick={skipBackward}
                            className="w-8 h-8 rounded-lg flex items-center justify-center text-zinc-500 hover:text-zinc-200 hover:bg-zinc-800/50 transition-all"
                            title="Skip back 5s"
                        >
                            <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                                <path strokeLinecap="round" strokeLinejoin="round" d="M12.066 11.2a1 1 0 000 1.6l5.334 4A1 1 0 0019 16V8a1 1 0 00-1.6-.8l-5.334 4zM4.066 11.2a1 1 0 000 1.6l5.334 4A1 1 0 0011 16V8a1 1 0 00-1.6-.8l-5.334 4z" />
                            </svg>
                        </button>
                        <button
                            onClick={togglePlay}
                            className={`w-10 h-10 rounded-xl flex items-center justify-center transition-all ${isPlaying
                                ? 'bg-violet-600/20 text-violet-300 border border-violet-500/30 hover:bg-violet-600/30'
                                : 'bg-violet-600/80 text-white shadow-lg shadow-violet-500/25 hover:bg-violet-500/80'
                                }`}
                        >
                            {isPlaying ? (
                                <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 24 24">
                                    <path d="M6 4h4v16H6V4zm8 0h4v16h-4V4z" />
                                </svg>
                            ) : (
                                <svg className="w-4 h-4 ml-0.5" fill="currentColor" viewBox="0 0 24 24">
                                    <path d="M8 5v14l11-7z" />
                                </svg>
                            )}
                        </button>
                        <button
                            onClick={skipForward}
                            className="w-8 h-8 rounded-lg flex items-center justify-center text-zinc-500 hover:text-zinc-200 hover:bg-zinc-800/50 transition-all"
                            title="Skip forward 5s"
                        >
                            <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                                <path strokeLinecap="round" strokeLinejoin="round" d="M11.933 12.8a1 1 0 000-1.6L6.6 7.2A1 1 0 005 8v8a1 1 0 001.6.8l5.333-4zM19.933 12.8a1 1 0 000-1.6l-5.333-4A1 1 0 0013 8v8a1 1 0 001.6.8l5.333-4z" />
                            </svg>
                        </button>
                    </div>

                    {/* Zoom */}
                    <div className="flex items-center gap-2 ml-2">
                        <svg className="w-3.5 h-3.5 text-zinc-600" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                            <path strokeLinecap="round" strokeLinejoin="round" d="M21 21l-5.197-5.197m0 0A7.5 7.5 0 105.196 5.196a7.5 7.5 0 0010.607 10.607zM10.5 7.5v6m3-3h-6" />
                        </svg>
                        <input
                            type="range"
                            min="10"
                            max="200"
                            value={zoom}
                            onChange={(e) => setZoom(Number(e.target.value))}
                            className="w-20 h-1 bg-zinc-800 rounded-lg appearance-none cursor-pointer accent-violet-500"
                        />
                    </div>

                    {/* Volume */}
                    <div className="flex items-center gap-2 ml-1">
                        <button
                            onClick={() => setVolume(v => v > 0 ? 0 : 0.8)}
                            className="text-zinc-600 hover:text-zinc-300 transition-colors"
                        >
                            {volume === 0 ? (
                                <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                                    <path strokeLinecap="round" strokeLinejoin="round" d="M17.25 9.75L19.5 12m0 0l2.25 2.25M19.5 12l2.25-2.25M19.5 12l-2.25 2.25m-10.5-6l4.72-3.15a.75.75 0 011.28.53v13.74a.75.75 0 01-1.28.530L6.75 14.25H3.75a.75.75 0 01-.75-.75v-3a.75.75 0 01.75-.75h3z" />
                                </svg>
                            ) : (
                                <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                                    <path strokeLinecap="round" strokeLinejoin="round" d="M19.114 5.636a9 9 0 010 12.728M16.463 8.288a5.25 5.25 0 010 7.424M6.75 8.25l4.72-3.15a.75.75 0 011.28.53v12.74a.75.75 0 01-1.28.53l-4.72-3.15H3.75a.75.75 0 01-.75-.75v-3a.75.75 0 01.75-.75h3z" />
                                </svg>
                            )}
                        </button>
                        <input
                            type="range"
                            min="0"
                            max="1"
                            step="0.05"
                            value={volume}
                            onChange={(e) => setVolume(Number(e.target.value))}
                            className="w-16 h-1 bg-zinc-800 rounded-lg appearance-none cursor-pointer accent-violet-500"
                        />
                    </div>

                    {/* Spacer */}
                    <div className="flex-1" />

                    {/* Stem Legend */}
                    <div className="flex items-center gap-2">
                        {(activeStems.length > 0 ? activeStems : ['Drums', 'Bass', 'Vocals']).map(name => {
                            const c = STEM_COLORS[name] || STEM_COLORS.Other;
                            return (
                                <div
                                    key={name}
                                    className={`flex items-center gap-1.5 px-2 py-1 rounded-md ${c.bg} border ${c.border}`}
                                >
                                    <span className={`w-1.5 h-1.5 rounded-full ${c.dot}`} />
                                    <span className="text-[10px] text-zinc-400">{name}</span>
                                </div>
                            );
                        })}
                    </div>
                </div>
            </div>
        </div>
    );
}

/* ── Exported Component (with Error Boundary) ──────── */

export default function WaveformXRay({ jobId }: WaveformXRayProps) {
    return (
        <WaveformErrorBoundary>
            <WaveformXRayInner jobId={jobId} />
        </WaveformErrorBoundary>
    );
}
