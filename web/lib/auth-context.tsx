"use client";

import {
    createContext,
    useCallback,
    useContext,
    useEffect,
    useMemo,
    useState,
} from "react";

interface AuthContextType {
    token: string | null;
    isAuthenticated: boolean;
    isLoading: boolean;
    login: (username: string, password: string) => Promise<void>;
    logout: () => void;
    error: string | null;
}

const AuthContext = createContext<AuthContextType | null>(null);

const TOKEN_KEY = "auralis_token";
const API_BASE = process.env.NEXT_PUBLIC_API_URL || "";

export function AuthProvider({ children }: { children: React.ReactNode }) {
    const [token, setToken] = useState<string | null>(null);
    const [isLoading, setIsLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    // Hydrate token from localStorage on mount
    useEffect(() => {
        const saved = localStorage.getItem(TOKEN_KEY);
        if (saved) setToken(saved);
        setIsLoading(false);
    }, []);

    const login = useCallback(async (username: string, password: string) => {
        setError(null);
        try {
            const form = new URLSearchParams();
            form.append("username", username);
            form.append("password", password);

            const res = await fetch(`${API_BASE}/api/auth/login`, {
                method: "POST",
                headers: { "Content-Type": "application/x-www-form-urlencoded" },
                body: form.toString(),
            });

            if (!res.ok) {
                const data = await res.json().catch(() => ({}));
                throw new Error(data.detail || "Login failed");
            }

            const data = await res.json();
            localStorage.setItem(TOKEN_KEY, data.access_token);
            setToken(data.access_token);
        } catch (err) {
            setError(err instanceof Error ? err.message : "Login failed");
            throw err;
        }
    }, []);

    const logout = useCallback(() => {
        localStorage.removeItem(TOKEN_KEY);
        setToken(null);
    }, []);

    const value = useMemo(
        () => ({
            token,
            isAuthenticated: !!token,
            isLoading,
            login,
            logout,
            error,
        }),
        [token, isLoading, login, logout, error]
    );

    return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth(): AuthContextType {
    const context = useContext(AuthContext);
    if (!context) throw new Error("useAuth must be used within AuthProvider");
    return context;
}
