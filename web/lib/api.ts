/**
 * AURALIS API client — authenticated fetch with token management.
 *
 * In production, API calls go through Caddy reverse proxy (same origin).
 * In development, set NEXT_PUBLIC_API_URL=http://localhost:8000
 */

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "";
const TOKEN_KEY = "auralis_token";

function getToken(): string | null {
    if (typeof window === "undefined") return null;
    return localStorage.getItem(TOKEN_KEY);
}

/**
 * Authenticated fetch wrapper. Automatically adds JWT Bearer token.
 */
export async function apiFetch(
    path: string,
    options: RequestInit = {}
): Promise<Response> {
    const token = getToken();
    const headers = new Headers(options.headers);

    if (token) {
        headers.set("Authorization", `Bearer ${token}`);
    }

    return fetch(`${API_BASE}${path}`, {
        ...options,
        headers,
    });
}

/**
 * Upload a file to the EAR layer.
 */
export async function uploadTrack(file: File) {
    const formData = new FormData();
    formData.append("file", file);

    const res = await apiFetch("/api/ear/upload", {
        method: "POST",
        body: formData,
    });

    if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        throw new Error(data.detail || "Upload failed");
    }

    return res.json() as Promise<{
        project_id: string;
        file: string;
        size_mb: string;
        message: string;
    }>;
}

/**
 * Start analysis pipeline for an uploaded project.
 */
export async function startAnalysis(projectId: string) {
    const res = await apiFetch(`/api/ear/analyze/${projectId}`, {
        method: "POST",
    });

    if (!res.ok) throw new Error("Analysis start failed");

    return res.json() as Promise<{
        job_id: string;
        status: string;
        progress: number;
        total_steps: number;
        message: string;
    }>;
}

/**
 * Poll job status.
 */
export async function getJobStatus(jobId: string) {
    const res = await apiFetch(`/api/ear/status/${jobId}`);
    if (!res.ok) throw new Error("Status check failed");

    return res.json() as Promise<{
        job_id: string;
        status: string;
        progress: number;
        total_steps: number;
        message: string;
        result: Record<string, unknown> | null;
    }>;
}

/**
 * Get GPU instance status.
 */
export async function getGpuStatus() {
    const res = await apiFetch("/api/gpu/status");
    if (!res.ok) throw new Error("GPU status check failed");
    return res.json();
}

/**
 * Start GPU instance.
 */
export async function startGpu() {
    const res = await apiFetch("/api/gpu/start", { method: "POST" });
    if (!res.ok) throw new Error("GPU start failed");
    return res.json();
}

/**
 * Stop GPU instance.
 */
export async function stopGpu() {
    const res = await apiFetch("/api/gpu/stop", { method: "POST" });
    if (!res.ok) throw new Error("GPU stop failed");
    return res.json();
}
