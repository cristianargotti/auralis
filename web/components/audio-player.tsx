"use client";

import { useState, useRef, useEffect, useCallback } from "react";

/* ── Types ── */
interface Track {
    key: string;
    label: string;
    icon: string;
    color: string;
    group: "main" | "stem";
}

interface AudioPlayerProps {
    jobId: string;
    hasOriginal?: boolean;
    hasMix?: boolean;
    hasMaster?: boolean;
    stems?: string[];
}

const TRACKS: Track[] = [
    { key: "original", label: "Original", icon: "🎵", color: "#f59e0b", group: "main" },
    { key: "master", label: "Master", icon: "💎", color: "#8b5cf6", group: "main" },
    { key: "stem_drums", label: "Drums", icon: "🥁", color: "#06b6d4", group: "stem" },
    { key: "stem_bass", label: "Bass", icon: "🎸", color: "#f43f5e", group: "stem" },
    { key: "stem_vocals", label: "Vocals", icon: "🎤", color: "#a855f7", group: "stem" },
    { key: "stem_other", label: "Other", icon: "🎹", color: "#10b981", group: "stem" },
];

/* ── Helpers ── */
function formatTime(s: number): string {
    const m = Math.floor(s / 60);
    const sec = Math.floor(s % 60);
    return `${m}:${sec.toString().padStart(2, "0")}`;
}

function getAudioUrl(jobId: string, fileKey: string): string {
    const token = typeof window !== "undefined" ? localStorage.getItem("auralis_token") : null;
    return `/api/reconstruct/audio/${jobId}/${fileKey}${token ? `?token=${token}` : ""}`;
}

/* ── Component ── */
export default function AudioPlayer({ jobId, hasOriginal = true, hasMaster = true, stems = [] }: AudioPlayerProps) {
    const audioRef = useRef<HTMLAudioElement>(null);
    const abAudioRef = useRef<HTMLAudioElement>(null);
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const animRef = useRef<number>(0);
    const analyserRef = useRef<AnalyserNode | null>(null);
    const sourceRef = useRef<MediaElementAudioSourceNode | null>(null);
    const ctxRef = useRef<AudioContext | null>(null);

    const [activeTrack, setActiveTrack] = useState<string>("original");
    const [playing, setPlaying] = useState(false);
    const [currentTime, setCurrentTime] = useState(0);
    const [duration, setDuration] = useState(0);
    const [volume, setVolume] = useState(0.8);
    const [abMode, setAbMode] = useState(false);
    const [abTrack, setAbTrack] = useState<"original" | "master">("original");
    const [loading, setLoading] = useState(false);
    const [waveformData, setWaveformData] = useState<number[]>([]);

    // Available tracks based on what files exist
    const availableTracks = TRACKS.filter((t) => {
        if (t.key === "original") return hasOriginal;
        if (t.key === "master") return hasMaster;
        if (t.group === "stem") return stems.length > 0;
        return true;
    });

    // Load track
    const loadTrack = useCallback((trackKey: string) => {
        const audio = audioRef.current;
        if (!audio) return;
        setLoading(true);
        const wasPlaying = playing;
        const pos = audio.currentTime;
        audio.src = getAudioUrl(jobId, trackKey);
        audio.load();
        audio.onloadeddata = () => {
            setLoading(false);
            setDuration(audio.duration);
            if (wasPlaying) {
                audio.currentTime = pos;
                audio.play().catch(() => { });
            }
        };
        audio.onerror = () => setLoading(false);
        setActiveTrack(trackKey);
    }, [jobId, playing]);

    // Switch track
    const switchTrack = useCallback((trackKey: string) => {
        if (trackKey === activeTrack) return;
        loadTrack(trackKey);
    }, [activeTrack, loadTrack]);

    // A/B Toggle — instant switch between original and master
    const toggleAB = useCallback(() => {
        if (!abMode) {
            setAbMode(true);
            setAbTrack("original");
            loadTrack("original");
            return;
        }
        const next = abTrack === "original" ? "master" : "original";
        setAbTrack(next);
        loadTrack(next);
    }, [abMode, abTrack, loadTrack]);

    // Play/Pause
    const togglePlay = useCallback(() => {
        const audio = audioRef.current;
        if (!audio) return;
        if (playing) {
            audio.pause();
        } else {
            // Setup audio context for visualizer on first play
            if (!ctxRef.current) {
                const ctx = new AudioContext();
                const analyser = ctx.createAnalyser();
                analyser.fftSize = 256;
                const source = ctx.createMediaElementSource(audio);
                source.connect(analyser);
                analyser.connect(ctx.destination);
                ctxRef.current = ctx;
                analyserRef.current = analyser;
                sourceRef.current = source;
            }
            audio.play().catch(() => { });
        }
        setPlaying(!playing);
    }, [playing]);

    // Seek
    const handleSeek = useCallback((e: React.MouseEvent<HTMLDivElement>) => {
        const audio = audioRef.current;
        if (!audio || !duration) return;
        const rect = e.currentTarget.getBoundingClientRect();
        const pct = (e.clientX - rect.left) / rect.width;
        audio.currentTime = pct * duration;
    }, [duration]);

    // Time update
    useEffect(() => {
        const audio = audioRef.current;
        if (!audio) return;
        const handler = () => setCurrentTime(audio.currentTime);
        const endHandler = () => setPlaying(false);
        audio.addEventListener("timeupdate", handler);
        audio.addEventListener("ended", endHandler);
        return () => {
            audio.removeEventListener("timeupdate", handler);
            audio.removeEventListener("ended", endHandler);
        };
    }, []);

    // Volume
    useEffect(() => {
        if (audioRef.current) audioRef.current.volume = volume;
    }, [volume]);

    // Load initial track
    useEffect(() => {
        loadTrack("original");
    }, [jobId]); // eslint-disable-line

    // Visualizer animation
    useEffect(() => {
        const canvas = canvasRef.current;
        const analyser = analyserRef.current;
        if (!canvas || !analyser || !playing) return;

        const ctx2d = canvas.getContext("2d");
        if (!ctx2d) return;

        const bufferLength = analyser.frequencyBinCount;
        const dataArray = new Uint8Array(bufferLength);

        const draw = () => {
            animRef.current = requestAnimationFrame(draw);
            analyser.getByteFrequencyData(dataArray);

            const w = canvas.width;
            const h = canvas.height;
            ctx2d.clearRect(0, 0, w, h);

            const barW = (w / bufferLength) * 2.5;
            let x = 0;

            const trackConfig = TRACKS.find((t) => t.key === activeTrack);
            const color = trackConfig?.color || "#f59e0b";

            for (let i = 0; i < bufferLength; i++) {
                const barH = (dataArray[i] / 255) * h;
                const alpha = 0.4 + (dataArray[i] / 255) * 0.6;

                ctx2d.fillStyle = `${color}${Math.round(alpha * 255).toString(16).padStart(2, "0")}`;
                ctx2d.fillRect(x, h - barH, barW - 1, barH);

                // Mirror effect
                ctx2d.fillStyle = `${color}22`;
                ctx2d.fillRect(x, 0, barW - 1, barH * 0.3);

                x += barW;
            }
        };
        draw();

        return () => cancelAnimationFrame(animRef.current);
    }, [playing, activeTrack]);

    const progress = duration > 0 ? (currentTime / duration) * 100 : 0;
    const activeConfig = TRACKS.find((t) => t.key === activeTrack);

    return (
        <div className="rounded-xl border border-zinc-800 bg-zinc-900/80 backdrop-blur-sm overflow-hidden">
            {/* Header */}
            <div className="flex items-center justify-between px-4 py-3 border-b border-zinc-800/50">
                <div className="flex items-center gap-2">
                    <span className="text-lg">🎧</span>
                    <span className="text-sm font-semibold text-zinc-200">Audio Player</span>
                    {activeConfig && (
                        <span className="text-xs px-2 py-0.5 rounded-full border" style={{
                            borderColor: `${activeConfig.color}40`,
                            color: activeConfig.color,
                            backgroundColor: `${activeConfig.color}10`,
                        }}>
                            {activeConfig.icon} {activeConfig.label}
                        </span>
                    )}
                </div>
                {/* A/B Toggle */}
                {hasOriginal && hasMaster && (
                    <button
                        onClick={toggleAB}
                        className={`text-xs font-bold px-3 py-1.5 rounded-lg transition-all ${abMode
                                ? "bg-amber-500/20 text-amber-400 border border-amber-500/30"
                                : "bg-zinc-800 text-zinc-500 hover:text-zinc-300 border border-zinc-700"
                            }`}
                    >
                        {abMode ? `A/B: ${abTrack === "original" ? "A (Original)" : "B (Master)"}` : "A/B Compare"}
                    </button>
                )}
            </div>

            {/* Visualizer */}
            <div className="relative h-20 bg-zinc-950/50">
                <canvas
                    ref={canvasRef}
                    width={800}
                    height={80}
                    className="w-full h-full"
                />
                {loading && (
                    <div className="absolute inset-0 flex items-center justify-center bg-zinc-950/70">
                        <div className="animate-pulse text-zinc-500 text-sm">Loading audio...</div>
                    </div>
                )}
            </div>

            {/* Progress bar + controls */}
            <div className="px-4 py-3 space-y-2">
                {/* Seekbar */}
                <div
                    className="relative h-2 bg-zinc-800 rounded-full cursor-pointer group"
                    onClick={handleSeek}
                >
                    <div
                        className="absolute left-0 top-0 h-full rounded-full transition-all"
                        style={{
                            width: `${progress}%`,
                            background: `linear-gradient(90deg, ${activeConfig?.color || "#f59e0b"}80, ${activeConfig?.color || "#f59e0b"})`,
                        }}
                    />
                    <div
                        className="absolute top-1/2 -translate-y-1/2 w-3 h-3 rounded-full opacity-0 group-hover:opacity-100 transition-opacity shadow-lg"
                        style={{
                            left: `${progress}%`,
                            transform: `translateX(-50%) translateY(-50%)`,
                            backgroundColor: activeConfig?.color || "#f59e0b",
                        }}
                    />
                </div>

                {/* Controls row */}
                <div className="flex items-center justify-between">
                    <div className="flex items-center gap-3">
                        {/* Play/Pause */}
                        <button
                            onClick={togglePlay}
                            className="w-10 h-10 rounded-full bg-gradient-to-br from-amber-500 to-orange-600 flex items-center justify-center text-white shadow-lg shadow-amber-500/20 hover:shadow-amber-500/40 transition-all active:scale-95"
                        >
                            {playing ? (
                                <svg width="14" height="14" viewBox="0 0 14 14" fill="currentColor">
                                    <rect x="2" y="1" width="4" height="12" rx="1" />
                                    <rect x="8" y="1" width="4" height="12" rx="1" />
                                </svg>
                            ) : (
                                <svg width="14" height="14" viewBox="0 0 14 14" fill="currentColor">
                                    <path d="M3 1.5v11l9-5.5L3 1.5z" />
                                </svg>
                            )}
                        </button>

                        {/* Time */}
                        <span className="text-xs font-mono text-zinc-400">
                            {formatTime(currentTime)} / {formatTime(duration)}
                        </span>
                    </div>

                    {/* Volume */}
                    <div className="flex items-center gap-2">
                        <span className="text-zinc-500 text-xs">🔊</span>
                        <input
                            type="range"
                            min="0"
                            max="1"
                            step="0.05"
                            value={volume}
                            onChange={(e) => setVolume(parseFloat(e.target.value))}
                            className="w-20 h-1 bg-zinc-700 rounded-full appearance-none cursor-pointer [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-3 [&::-webkit-slider-thumb]:h-3 [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-amber-500"
                        />
                    </div>
                </div>
            </div>

            {/* Track selector */}
            <div className="px-4 pb-3">
                <div className="flex gap-1.5 flex-wrap">
                    {availableTracks.map((track) => (
                        <button
                            key={track.key}
                            onClick={() => {
                                if (abMode) setAbMode(false);
                                switchTrack(track.key);
                            }}
                            className={`text-xs px-3 py-1.5 rounded-lg transition-all font-medium ${activeTrack === track.key
                                    ? "text-white shadow-lg"
                                    : "bg-zinc-800/50 text-zinc-500 hover:text-zinc-300 hover:bg-zinc-800"
                                }`}
                            style={activeTrack === track.key ? {
                                backgroundColor: `${track.color}20`,
                                color: track.color,
                                boxShadow: `0 2px 8px ${track.color}15`,
                                border: `1px solid ${track.color}30`,
                            } : {}}
                        >
                            {track.icon} {track.label}
                        </button>
                    ))}
                </div>
            </div>

            {/* Hidden audio element */}
            <audio ref={audioRef} preload="auto" crossOrigin="anonymous" />
            <audio ref={abAudioRef} preload="auto" crossOrigin="anonymous" />
        </div>
    );
}
