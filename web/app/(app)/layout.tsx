"use client";

import { Sidebar } from "@/components/sidebar";

export default function AppLayout({ children }: { children: React.ReactNode }) {
    return (
        <>
            <Sidebar />
            <main className="flex-1 overflow-y-auto">
                <div className="mx-auto max-w-7xl px-6 py-6">{children}</div>
            </main>
        </>
    );
}
