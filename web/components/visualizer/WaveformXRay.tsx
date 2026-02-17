'use client';

import { useEffect, useRef, useState, Component, ReactNode } from 'react';
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
                <div className="p-4 text-center text-amber-400 bg-amber-500/10 border border-amber-500/20 rounded-xl">
                    <span className="text-xl">⚠️</span>
                    <p className="text-sm mt-1">Waveform visualization encountered an error.</p>
                    <p className="text-xs text-zinc-500 mt-1">{this.state.error?.message}</p>
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
    const regionsRef = useRef<any>(null);

    // Fetch X-Ray Data
    useEffect(() => {
        const fetchData = async () => {
            try {
                setError(null);
                const json = await api<WaveformData>(`/api/reconstruct/waveform/${jobId}`);

                // Validate critical fields
                if (!json || typeof json.duration !== 'number') {
                    setError('Invalid waveform data received');
                    return;
                }

                // Ensure waveform is a non-empty numeric array
                if (!Array.isArray(json.waveform) || json.waveform.length === 0) {
                    json.waveform = new Array(200).fill(0);
                }

                // Ensure layers is an array
                if (!Array.isArray(json.layers)) {
                    json.layers = [];
                }

                setData(json);
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
                // Dynamic imports to avoid SSR issues
                const WaveSurfer = (await import('wavesurfer.js')).default;
                const RegionsPlugin = (await import('wavesurfer.js/dist/plugins/regions.esm.js')).default;
                const TimelinePlugin = (await import('wavesurfer.js/dist/plugins/timeline.esm.js')).default;

                if (!containerRef.current) return; // Guard against unmount during async

                const token = typeof window !== 'undefined' ? localStorage.getItem('auralis_token') : null;
                const authHeaders = token ? { Authorization: `Bearer ${token}` } : {};

                // Ensure peaks data is valid (at least one non-zero value or a dummy array)
                const safePeaks = data.waveform.length > 0 ? data.waveform : new Array(200).fill(0);

                ws = WaveSurfer.create({
                    container: containerRef.current,
                    waveColor: '#4f46e5',
                    progressColor: '#818cf8',
                    cursorColor: '#c7d2fe',
                    barWidth: 2,
                    barGap: 1,
                    barRadius: 2,
                    height: 128,
                    normalize: true,
                    minPxPerSec: zoom,
                    url: `${process.env.NEXT_PUBLIC_API_URL || ''}/api/reconstruct/audio/${jobId}/${audioSource}`,
                    fetchParams: {
                        headers: authHeaders as HeadersInit
                    },
                    peaks: [safePeaks],
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

                // Error handler — catch audio load failures gracefully
                ws.on('error', (err) => {
                    console.warn('[WaveformXRay] WaveSurfer error (non-fatal):', err);
                    // Don't crash — just disable playback
                });

                ws.on('play', () => setIsPlaying(true));
                ws.on('pause', () => setIsPlaying(false));
                ws.on('finish', () => setIsPlaying(false));

                setWavesurfer(ws);
            } catch (err: any) {
                console.error('[WaveformXRay] WaveSurfer initialization failed:', err);
                setError(`Visualization failed: ${err?.message || 'Unknown error'}`);
            }
        };

        initWaveSurfer();

        return () => {
            try {
                ws?.destroy();
            } catch {
                // Ignore destroy errors
            }
        };
    }, [data, jobId, audioSource, zoom]);

    // Add Regions (Layers)
    useEffect(() => {
        if (!wavesurfer || !data || !regionsRef.current) return;

        try {
            regionsRef.current.clearRegions();

            data.layers.forEach((layer) => {
                const color =
                    layer.name === 'Drums' ? 'rgba(239, 68, 68, 0.2)' :
                        layer.name === 'Bass' ? 'rgba(245, 158, 11, 0.2)' :
                            layer.name === 'Vocals' ? 'rgba(16, 185, 129, 0.2)' :
                                'rgba(99, 102, 241, 0.2)';

                (layer.events || []).forEach((event) => {
                    if (layer.type === 'point' && event.time !== undefined && isFinite(event.time)) {
                        regionsRef.current.addRegion({
                            start: event.time,
                            end: event.time + 0.1,
                            content: event.label,
                            color: color,
                            drag: false,
                            resize: false,
                        });
                    } else if (layer.type === 'region' && event.start !== undefined && event.end !== undefined && isFinite(event.start) && isFinite(event.end)) {
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
        } catch (err) {
            console.warn('[WaveformXRay] Error adding regions (non-fatal):', err);
        }
    }, [wavesurfer, data]);

    // Handle Zoom
    useEffect(() => {
        if (wavesurfer) {
            try {
                wavesurfer.zoom(zoom);
            } catch {
                // Ignore zoom errors
            }
        }
    }, [zoom, wavesurfer]);

    const togglePlay = () => {
        try {
            wavesurfer?.playPause();
        } catch {
            // Ignore playback errors
        }
    };

    if (loading) return <div className="p-4 text-center text-gray-400">Loading X-Ray data...</div>;

    if (error || !data) {
        return (
            <div className="w-full bg-zinc-900/50 border border-zinc-800 rounded-xl p-6 text-center space-y-2">
                <span className="text-2xl">🩻</span>
                <p className="text-zinc-400 text-sm">
                    {error || 'X-Ray data not available for this job.'}
                </p>
                <p className="text-zinc-600 text-xs">
                    This may happen if the server was restarted after reconstruction.
                    Try uploading and reconstructing the track again.
                </p>
            </div>
        );
    }

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

/* ── Exported Component (with Error Boundary) ──────── */

export default function WaveformXRay({ jobId }: WaveformXRayProps) {
    return (
        <WaveformErrorBoundary>
            <WaveformXRayInner jobId={jobId} />
        </WaveformErrorBoundary>
    );
}
