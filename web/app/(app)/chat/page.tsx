"use client";

import { useCallback, useRef, useState } from "react";
import {
    Card,
    CardContent,
    CardHeader,
    CardTitle,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { api } from "@/lib/api";

interface Message {
    role: "user" | "assistant";
    content: string;
    timestamp: Date;
}

export default function ChatPage() {
    const [messages, setMessages] = useState<Message[]>([]);
    const [input, setInput] = useState("");
    const [loading, setLoading] = useState(false);
    const scrollRef = useRef<HTMLDivElement>(null);

    const sendMessage = useCallback(async () => {
        if (!input.trim() || loading) return;

        const userMessage: Message = {
            role: "user",
            content: input.trim(),
            timestamp: new Date(),
        };

        setMessages((prev) => [...prev, userMessage]);
        setInput("");
        setLoading(true);

        try {
            const history = messages.map((m) => ({
                role: m.role,
                content: m.content,
            }));

            const res = await api<{ response: string }>("/api/brain/chat", {
                method: "POST",
                body: JSON.stringify({
                    message: userMessage.content,
                    history,
                }),
            });

            const assistantMessage: Message = {
                role: "assistant",
                content: res.response,
                timestamp: new Date(),
            };

            setMessages((prev) => [...prev, assistantMessage]);
        } catch {
            const errorMessage: Message = {
                role: "assistant",
                content: "⚠️ Could not reach AURALIS Brain. Make sure OPENAI_API_KEY is configured.",
                timestamp: new Date(),
            };
            setMessages((prev) => [...prev, errorMessage]);
        } finally {
            setLoading(false);
            setTimeout(() => {
                scrollRef.current?.scrollIntoView({ behavior: "smooth" });
            }, 100);
        }
    }, [input, loading, messages]);

    const handleKeyDown = (e: React.KeyboardEvent) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    };

    return (
        <div className="flex flex-col h-[calc(100vh-8rem)]">
            <div className="mb-4">
                <h1 className="text-3xl font-bold tracking-tight">🧠 AI Chat</h1>
                <p className="text-muted-foreground mt-1">
                    Talk to AURALIS about music production, sound design, mixing, and mastering
                </p>
            </div>

            {/* Messages */}
            <div className="flex-1 overflow-y-auto space-y-4 mb-4 pr-2">
                {messages.length === 0 && (
                    <div className="flex items-center justify-center h-full">
                        <div className="text-center space-y-4">
                            <div className="text-6xl">🎛️</div>
                            <p className="text-zinc-500 text-sm max-w-md">
                                Ask AURALIS anything about music production — sound design,
                                mixing techniques, arrangement ideas, or creative direction.
                            </p>
                            <div className="flex flex-wrap gap-2 justify-center">
                                {[
                                    "How do I get a wider stereo image?",
                                    "Best EQ for crispy hi-hats?",
                                    "Sidechain compression settings for house",
                                    "How to layer kicks and bass?",
                                ].map((q) => (
                                    <button
                                        key={q}
                                        className="text-xs bg-zinc-800 hover:bg-zinc-700 text-zinc-400 px-3 py-1.5 rounded-full transition"
                                        onClick={() => { setInput(q); }}
                                    >
                                        {q}
                                    </button>
                                ))}
                            </div>
                        </div>
                    </div>
                )}

                {messages.map((msg, i) => (
                    <div key={i} className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}>
                        <div className={`max-w-[80%] rounded-2xl px-4 py-3 ${msg.role === "user"
                                ? "bg-emerald-600 text-white"
                                : "bg-zinc-800 text-zinc-200"
                            }`}>
                            <p className="text-sm whitespace-pre-wrap">{msg.content}</p>
                            <p className="text-[10px] mt-1 opacity-50">
                                {msg.timestamp.toLocaleTimeString()}
                            </p>
                        </div>
                    </div>
                ))}

                {loading && (
                    <div className="flex justify-start">
                        <div className="bg-zinc-800 rounded-2xl px-4 py-3">
                            <div className="flex gap-1">
                                <div className="w-2 h-2 bg-emerald-500 rounded-full animate-bounce" />
                                <div className="w-2 h-2 bg-emerald-500 rounded-full animate-bounce [animation-delay:150ms]" />
                                <div className="w-2 h-2 bg-emerald-500 rounded-full animate-bounce [animation-delay:300ms]" />
                            </div>
                        </div>
                    </div>
                )}

                <div ref={scrollRef} />
            </div>

            {/* Input */}
            <div className="flex gap-3 items-end">
                <textarea
                    className="flex-1 bg-zinc-900 border border-zinc-700 rounded-xl p-3 text-white placeholder:text-zinc-500 resize-none focus:border-emerald-500 focus:outline-none min-h-[48px] max-h-[120px]"
                    placeholder="Ask AURALIS about music production..."
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    onKeyDown={handleKeyDown}
                    rows={1}
                    disabled={loading}
                />
                <Button
                    onClick={sendMessage}
                    disabled={!input.trim() || loading}
                    className="bg-emerald-600 hover:bg-emerald-700 h-12 px-6"
                >
                    Send
                </Button>
            </div>
        </div>
    );
}
