"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";
import { useAuth } from "@/lib/auth-context";

/**
 * Client-side route guard — redirects to /login if not authenticated.
 * Wraps all (app) pages.
 */
export function AuthGuard({ children }: { children: React.ReactNode }) {
    const { isAuthenticated, isLoading } = useAuth();
    const router = useRouter();

    useEffect(() => {
        if (!isLoading && !isAuthenticated) {
            router.replace("/login");
        }
    }, [isAuthenticated, isLoading, router]);

    if (isLoading) {
        return (
            <div className="flex h-dvh items-center justify-center bg-black">
                <div className="flex items-center gap-3 text-zinc-400">
                    <span className="h-5 w-5 animate-spin rounded-full border-2 border-current border-t-transparent" />
                    <span className="text-sm">Restoring session...</span>
                </div>
            </div>
        );
    }

    if (!isAuthenticated) return null; // Will redirect via effect

    return <>{children}</>;
}
