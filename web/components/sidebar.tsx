"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useAuth } from "@/lib/auth-context";

const NAV_ITEMS = [
    {
        label: "Dashboard",
        href: "/",
        icon: "⚡",
        description: "Projects & activity",
    },
    {
        label: "Architecture",
        href: "/architecture",
        icon: "🏗️",
        description: "Engine blueprint",
    },
    {
        label: "Deconstructor",
        href: "/deconstruct",
        icon: "👂",
        description: "Upload & analyze",
    },
    {
        label: "Studio",
        href: "/studio",
        icon: "🎹",
        description: "Arrangement & synthesis",
    },
    {
        label: "Mixer",
        href: "/mixer",
        icon: "🎚️",
        description: "Mix & effects",
    },
    {
        label: "Master",
        href: "/master",
        icon: "💎",
        description: "Reference mastering",
    },
    {
        label: "QC",
        href: "/qc",
        icon: "🔍",
        description: "Quality report",
    },
    {
        label: "Reconstruct",
        href: "/reconstruct",
        icon: "🔬",
        description: "Bar-by-bar rebuild",
    },
    {
        label: "Creator",
        href: "/creator",
        icon: "✨",
        description: "AI production",
    },
    {
        label: "AI Chat",
        href: "/chat",
        icon: "🧠",
        description: "AI assistant",
    },
];

export function Sidebar() {
    const pathname = usePathname();
    const { logout } = useAuth();

    return (
        <aside className="glass-strong flex w-[220px] shrink-0 flex-col border-r border-border/50">
            {/* Logo */}
            <div className="flex items-center gap-3 px-5 py-5">
                <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-gradient-auralis glow-cyan">
                    <span className="text-lg font-bold text-gradient">A</span>
                </div>
                <div>
                    <h1 className="text-sm font-bold tracking-wider text-gradient">
                        AURALIS
                    </h1>
                    <p className="text-[10px] tracking-widest text-muted-foreground">
                        v0.1.0
                    </p>
                </div>
            </div>

            <div className="h-px bg-border/30 mx-4" />

            {/* Navigation */}
            <nav className="flex-1 space-y-0.5 px-3 py-3">
                {NAV_ITEMS.map((item) => {
                    const isActive = pathname === item.href;
                    return (
                        <Link
                            key={item.href}
                            href={item.href}
                            className={`group flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm transition-all duration-200 ${isActive
                                ? "bg-primary/10 text-primary glow-cyan"
                                : "text-muted-foreground hover:bg-secondary hover:text-foreground"
                                }`}
                        >
                            <span className="text-base">{item.icon}</span>
                            <div className="flex flex-col">
                                <span className="font-medium leading-tight">{item.label}</span>
                                <span className="text-[10px] opacity-60">
                                    {item.description}
                                </span>
                            </div>
                        </Link>
                    );
                })}
            </nav>

            {/* Status footer */}
            <div className="border-t border-border/30 px-4 py-3">
                <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                        <div className="h-2 w-2 rounded-full bg-auralis-emerald animate-pulse-slow" />
                        <span className="text-[11px] text-muted-foreground">
                            Engine ready
                        </span>
                    </div>
                    <button
                        onClick={logout}
                        className="text-[10px] text-muted-foreground/50 hover:text-destructive transition-colors"
                        title="Sign out"
                    >
                        Sign out
                    </button>
                </div>
                <p className="mt-1 text-[10px] text-muted-foreground/50">
                    Hear deeper. Create beyond.
                </p>
            </div>
        </aside>
    );
}
