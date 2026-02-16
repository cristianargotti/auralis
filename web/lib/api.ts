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
 * Typed API helper — wraps apiFetch with JSON parsing.
 */
export async function api<T = unknown>(
    path: string,
    options: RequestInit = {}
): Promise<T> {
    const headers = new Headers(options.headers);
    if (options.body && typeof options.body === "string") {
        headers.set("Content-Type", "application/json");
    }
    const res = await apiFetch(path, { ...options, headers });
    if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        throw new Error(data.detail || `API error: ${res.status}`);
    }
    return res.json() as Promise<T>;
}

// ── EAR Layer ──────────────────────────────────────────

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

// ── GPU Manager ────────────────────────────────────────

export async function getGpuStatus() {
    const res = await apiFetch("/api/gpu/status");
    if (!res.ok) throw new Error("GPU status check failed");
    return res.json();
}

export async function startGpu() {
    const res = await apiFetch("/api/gpu/start", { method: "POST" });
    if (!res.ok) throw new Error("GPU start failed");
    return res.json();
}

export async function stopGpu() {
    const res = await apiFetch("/api/gpu/stop", { method: "POST" });
    if (!res.ok) throw new Error("GPU stop failed");
    return res.json();
}

// ── Console (Mastering + QC) ───────────────────────────

export interface MasterResult {
    status: string;
    output: string;
    peak_dbtp: number;
    rms_db: number;
    est_lufs: number;
    clipping_samples: number;
    stages: string[];
    error?: string;
}

export interface QCResult {
    status: string;
    pass_fail: string;
    issues: string[];
    dynamics: {
        peak_db: number;
        rms_db: number;
        crest_factor_db: number;
        dynamic_range_db: number;
    };
    clipping: {
        is_clipping: boolean;
        clipped_samples: number;
    };
    stereo: {
        correlation: number;
        width: number;
        mono_compatible: boolean;
    } | null;
    loudness: {
        integrated_lufs: number;
        true_peak_dbtp: number;
    } | null;
    spectrum: {
        sub: number;
        bass: number;
        low_mid: number;
        mid: number;
        upper_mid: number;
        presence: number;
        brilliance: number;
    };
    error?: string;
}

export interface PresetConfig {
    target_lufs: number;
    drive: number;
    width: number;
    ceiling_db: number;
}

export async function masterTrack(
    projectId: string,
    preset: string = "streaming",
    overrides?: { target_lufs?: number; drive?: number; width?: number }
): Promise<MasterResult> {
    const params = new URLSearchParams({ preset });
    if (overrides?.target_lufs != null) params.set("target_lufs", String(overrides.target_lufs));
    if (overrides?.drive != null) params.set("drive", String(overrides.drive));
    if (overrides?.width != null) params.set("width", String(overrides.width));

    const res = await apiFetch(`/api/console/master/${projectId}?${params}`, {
        method: "POST",
    });
    if (!res.ok) throw new Error("Mastering failed");
    return res.json();
}

export async function runQC(projectId: string): Promise<QCResult> {
    const res = await apiFetch(`/api/console/qc/${projectId}`);
    if (!res.ok) throw new Error("QC analysis failed");
    return res.json();
}

export async function getPresets(): Promise<Record<string, PresetConfig>> {
    const res = await apiFetch("/api/console/presets");
    if (!res.ok) throw new Error("Failed to load presets");
    return res.json();
}

/**
 * Get visualization image URL (returns authenticated URL).
 */
export function getVizUrl(type: string, projectId: string): string {
    return `${API_BASE}/api/console/viz/${type}/${projectId}`;
}

/**
 * Fetch a visualization image as a blob URL for display.
 */
export async function fetchVizImage(type: string, projectId: string): Promise<string> {
    const res = await apiFetch(`/api/console/viz/${type}/${projectId}`);
    if (!res.ok) throw new Error(`Visualization ${type} failed`);
    const blob = await res.blob();
    return URL.createObjectURL(blob);
}
