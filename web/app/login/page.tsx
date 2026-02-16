"use client";

import { useState, type FormEvent } from "react";
import { useRouter } from "next/navigation";
import { Button } from "@/components/ui/button";
import {
    Card,
    CardContent,
    CardDescription,
    CardHeader,
    CardTitle,
} from "@/components/ui/card";
import { useAuth } from "@/lib/auth-context";

export default function LoginPage() {
    const { login } = useAuth();
    const router = useRouter();
    const [username, setUsername] = useState("");
    const [password, setPassword] = useState("");
    const [error, setError] = useState<string | null>(null);
    const [loading, setLoading] = useState(false);

    const handleSubmit = async (e: FormEvent) => {
        e.preventDefault();
        setError(null);
        setLoading(true);
        try {
            await login(username, password);
            router.push("/");
        } catch (err) {
            setError(err instanceof Error ? err.message : "Login failed");
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="flex h-full w-full items-center justify-center bg-gradient-to-br from-background via-background to-accent/5">
            <Card className="w-full max-w-sm glass border-border/30 glow-cyan">
                <CardHeader className="text-center space-y-3">
                    {/* Logo */}
                    <div className="mx-auto flex h-14 w-14 items-center justify-center rounded-2xl bg-gradient-auralis glow-cyan">
                        <span className="text-2xl font-bold text-gradient">A</span>
                    </div>
                    <div>
                        <CardTitle className="text-xl font-bold tracking-wider text-gradient">
                            AURALIS
                        </CardTitle>
                        <CardDescription className="text-xs tracking-widest text-muted-foreground mt-1">
                            AI MUSIC PRODUCTION ENGINE
                        </CardDescription>
                    </div>
                </CardHeader>

                <CardContent>
                    <form onSubmit={handleSubmit} className="space-y-4">
                        {error && (
                            <div className="rounded-lg bg-destructive/10 border border-destructive/20 px-3 py-2.5 text-sm text-destructive">
                                {error}
                            </div>
                        )}

                        <div className="space-y-2">
                            <label
                                htmlFor="username"
                                className="text-xs font-medium text-muted-foreground uppercase tracking-wider"
                            >
                                Username
                            </label>
                            <input
                                id="username"
                                type="text"
                                autoComplete="username"
                                required
                                value={username}
                                onChange={(e) => setUsername(e.target.value)}
                                className="w-full rounded-lg bg-input px-4 py-2.5 text-sm placeholder:text-muted-foreground/40 focus:outline-none focus:ring-2 focus:ring-primary/50"
                                placeholder="admin"
                            />
                        </div>

                        <div className="space-y-2">
                            <label
                                htmlFor="password"
                                className="text-xs font-medium text-muted-foreground uppercase tracking-wider"
                            >
                                Password
                            </label>
                            <input
                                id="password"
                                type="password"
                                autoComplete="current-password"
                                required
                                value={password}
                                onChange={(e) => setPassword(e.target.value)}
                                className="w-full rounded-lg bg-input px-4 py-2.5 text-sm placeholder:text-muted-foreground/40 focus:outline-none focus:ring-2 focus:ring-primary/50"
                                placeholder="••••••••"
                            />
                        </div>

                        <Button
                            type="submit"
                            className="w-full"
                            disabled={loading || !username || !password}
                        >
                            {loading ? (
                                <span className="flex items-center gap-2">
                                    <span className="h-4 w-4 animate-spin rounded-full border-2 border-current border-t-transparent" />
                                    Authenticating...
                                </span>
                            ) : (
                                "Sign In"
                            )}
                        </Button>

                        <p className="text-center text-[10px] text-muted-foreground/40 mt-3">
                            Hear deeper. Create beyond.
                        </p>
                    </form>
                </CardContent>
            </Card>
        </div>
    );
}
