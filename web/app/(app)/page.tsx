"use client";

import { useEffect, useState } from "react";
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
import Link from "next/link";

interface SystemStatus {
    ear: boolean;
    hands: boolean;
    console: boolean;
    grid: boolean;
    brain: boolean;
}

export default function DashboardPage() {
    const [status, setStatus] = useState<SystemStatus>({
        ear: true,
        hands: true,
        console: true,
        grid: true,
        brain: true,
    });
    const [time, setTime] = useState("");

    useEffect(() => {
        const tick = () =>
            setTime(new Date().toLocaleTimeString("en-US", { hour12: false }));
        tick();
        const interval = setInterval(tick, 1000);
        return () => clearInterval(interval);
    }, []);

    const layers = [
        { key: "ear", label: "EAR", icon: "👂", desc: "Analysis & Deconstruction", ready: status.ear },
        { key: "hands", label: "HANDS", icon: "🎹", desc: "Synthesis & Sound Design", ready: status.hands },
        { key: "console", label: "CONSOLE", icon: "🎚️", desc: "Mixing & Mastering", ready: status.console },
        { key: "grid", label: "GRID", icon: "📐", desc: "Composition & Arrangement", ready: status.grid },
        { key: "brain", label: "BRAIN", icon: "🧠", desc: "AI Intelligence", ready: status.brain },
    ];

    const readyCount = layers.filter((l) => l.ready).length;

    return (
        <div className="space-y-6">
            {/* Hero header */}
            <div className="flex items-end justify-between">
                <div>
                    <h1 className="text-3xl font-bold tracking-tight">
                        <span className="text-gradient">AURALIS</span>{" "}
                        <span className="text-muted-foreground font-normal">Engine</span>
                    </h1>
                    <p className="mt-1 text-sm text-muted-foreground">
                        Hear deeper. Create beyond.
                    </p>
                </div>
                <div className="text-right">
                    <p className="text-2xl font-mono text-muted-foreground/60">{time}</p>
                    <p className="text-xs text-muted-foreground/40">
                        {readyCount}/{layers.length} layers online
                    </p>
                </div>
            </div>

            {/* System status */}
            <Card className="glass border-border/30">
                <CardHeader className="pb-3">
                    <CardTitle className="text-sm font-medium text-muted-foreground uppercase tracking-wider">
                        System Status
                    </CardTitle>
                </CardHeader>
                <CardContent>
                    <div className="grid grid-cols-5 gap-3">
                        {layers.map((layer) => (
                            <div
                                key={layer.key}
                                className={`rounded-xl p-4 text-center transition-all duration-300 ${layer.ready
                                    ? "bg-primary/8 border border-primary/20 glow-cyan"
                                    : "bg-secondary/50 border border-border/20"
                                    }`}
                            >
                                <span className="text-2xl">{layer.icon}</span>
                                <p className="mt-2 text-xs font-bold tracking-wider">
                                    {layer.label}
                                </p>
                                <p className="text-[10px] text-muted-foreground mt-1">
                                    {layer.desc}
                                </p>
                                <Badge
                                    variant={layer.ready ? "default" : "secondary"}
                                    className="mt-2 text-[10px]"
                                >
                                    {layer.ready ? "Online" : "Pending"}
                                </Badge>
                            </div>
                        ))}
                    </div>
                    <div className="mt-4 flex items-center gap-3">
                        <Progress value={(readyCount / layers.length) * 100} className="h-1.5" />
                        <span className="text-xs text-muted-foreground shrink-0">
                            {Math.round((readyCount / layers.length) * 100)}%
                        </span>
                    </div>
                </CardContent>
            </Card>

            {/* Quick actions */}
            <div className="grid grid-cols-3 gap-4">
                <Card className="glass border-border/30 group hover:glow-cyan transition-all duration-300">
                    <CardHeader>
                        <CardTitle className="flex items-center gap-2 text-base">
                            <span className="text-xl">👂</span> Deconstruct
                        </CardTitle>
                        <CardDescription className="text-xs">
                            Upload any track → stems, MIDI, DNA map
                        </CardDescription>
                    </CardHeader>
                    <CardContent>
                        <Link href="/deconstruct">
                            <Button className="w-full" size="sm">
                                Open Deconstructor
                            </Button>
                        </Link>
                    </CardContent>
                </Card>

                <Card className="glass border-border/30 group hover:glow-violet transition-all duration-300">
                    <CardHeader>
                        <CardTitle className="flex items-center gap-2 text-base">
                            <span className="text-xl">🌟</span> Create
                        </CardTitle>
                        <CardDescription className="text-xs">
                            Describe your track → AI produces everything
                        </CardDescription>
                    </CardHeader>
                    <CardContent>
                        <Link href="/chat">
                            <Button variant="secondary" className="w-full" size="sm">
                                Open Creator
                            </Button>
                        </Link>
                    </CardContent>
                </Card>

                <Card className="glass border-border/30 group hover:glow-cyan transition-all duration-300">
                    <CardHeader>
                        <CardTitle className="flex items-center gap-2 text-base">
                            <span className="text-xl">💎</span> Master
                        </CardTitle>
                        <CardDescription className="text-xs">
                            Reference-matched mastering at 100%
                        </CardDescription>
                    </CardHeader>
                    <CardContent>
                        <Link href="/master">
                            <Button variant="secondary" className="w-full" size="sm">
                                Open Master Suite
                            </Button>
                        </Link>
                    </CardContent>
                </Card>
            </div>

            {/* Recent projects */}
            <Card className="glass border-border/30">
                <CardHeader>
                    <CardTitle className="text-sm font-medium text-muted-foreground uppercase tracking-wider">
                        Recent Projects
                    </CardTitle>
                </CardHeader>
                <CardContent>
                    <div className="flex flex-col items-center justify-center py-12 text-center">
                        <div className="text-4xl mb-3 opacity-30">🎵</div>
                        <p className="text-sm text-muted-foreground">
                            No projects yet. Upload a track to get started.
                        </p>
                        <Link href="/deconstruct" className="mt-4">
                            <Button size="sm" variant="outline">
                                Upload Your First Track
                            </Button>
                        </Link>
                    </div>
                </CardContent>
            </Card>

            {/* Tech stack badges */}
            <div className="flex items-center gap-2 flex-wrap">
                <span className="text-[10px] text-muted-foreground/40 uppercase tracking-widest mr-2">
                    Powered by
                </span>
                {[
                    "HTDemucs v4",
                    "basic-pitch",
                    "librosa",
                    "FastAPI",
                    "OpenAI GPT",
                    "TorchFX",
                    "Pedalboard",
                ].map((tech) => (
                    <Badge key={tech} variant="outline" className="text-[10px] text-muted-foreground/50">
                        {tech}
                    </Badge>
                ))}
            </div>
        </div>
    );
}
