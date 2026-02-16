"use client";

import { useState } from "react";
import {
    Card,
    CardContent,
    CardHeader,
    CardTitle,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";

interface ChatMessage {
    role: "user" | "assistant" | "system";
    content: string;
    timestamp: string;
}

export default function ChatPage() {
    const [messages, setMessages] = useState<ChatMessage[]>([
        {
            role: "system",
            content:
                "Welcome to AURALIS AI. I can help you with arrangement decisions, sound design, mixing strategies, and creative direction. What would you like to create?",
            timestamp: new Date().toLocaleTimeString("en-US", { hour12: false }),
        },
    ]);
    const [input, setInput] = useState("");
    const [isLoading, setIsLoading] = useState(false);

    const sendMessage = async () => {
        if (!input.trim() || isLoading) return;

        const userMessage: ChatMessage = {
            role: "user",
            content: input.trim(),
            timestamp: new Date().toLocaleTimeString("en-US", { hour12: false }),
        };
        setMessages((prev) => [...prev, userMessage]);
        setInput("");
        setIsLoading(true);

        // Simulated AI response (will connect to OpenAI via backend in Phase 3)
        setTimeout(() => {
            const aiMessage: ChatMessage = {
                role: "assistant",
                content:
                    "AI integration coming in Phase 3. I'll be powered by OpenAI GPT with deep knowledge of your project's Track DNA, arrangement patterns, and mixing strategies. Stay tuned! 🧠",
                timestamp: new Date().toLocaleTimeString("en-US", { hour12: false }),
            };
            setMessages((prev) => [...prev, aiMessage]);
            setIsLoading(false);
        }, 1000);
    };

    return (
        <div className="flex flex-col h-[calc(100dvh-48px)]">
            <div className="mb-4">
                <h1 className="text-2xl font-bold tracking-tight">
                    <span className="text-gradient">AI Chat</span>
                </h1>
                <p className="text-sm text-muted-foreground mt-1">
                    Talk to AURALIS about arrangement, sound design, and production.
                </p>
            </div>

            {/* Messages */}
            <Card className="glass border-border/30 flex-1 flex flex-col min-h-0">
                <ScrollArea className="flex-1 p-4">
                    <div className="space-y-4">
                        {messages.map((msg, i) => (
                            <div
                                key={i}
                                className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"
                                    }`}
                            >
                                <div
                                    className={`max-w-[75%] rounded-xl px-4 py-3 ${msg.role === "user"
                                            ? "bg-primary text-primary-foreground"
                                            : msg.role === "system"
                                                ? "bg-accent/20 border border-accent/20"
                                                : "bg-secondary"
                                        }`}
                                >
                                    <p className="text-sm">{msg.content}</p>
                                    <p className="text-[10px] mt-1 opacity-50">
                                        {msg.timestamp}
                                    </p>
                                </div>
                            </div>
                        ))}
                        {isLoading && (
                            <div className="flex justify-start">
                                <div className="bg-secondary rounded-xl px-4 py-3">
                                    <div className="flex gap-1">
                                        <span className="h-2 w-2 rounded-full bg-primary animate-bounce" />
                                        <span className="h-2 w-2 rounded-full bg-primary animate-bounce [animation-delay:150ms]" />
                                        <span className="h-2 w-2 rounded-full bg-primary animate-bounce [animation-delay:300ms]" />
                                    </div>
                                </div>
                            </div>
                        )}
                    </div>
                </ScrollArea>

                {/* Input */}
                <div className="border-t border-border/30 p-4">
                    <div className="flex gap-3">
                        <input
                            type="text"
                            value={input}
                            onChange={(e) => setInput(e.target.value)}
                            onKeyDown={(e) => e.key === "Enter" && sendMessage()}
                            placeholder="Describe what you want to create..."
                            className="flex-1 rounded-lg bg-input px-4 py-2.5 text-sm placeholder:text-muted-foreground/50 focus:outline-none focus:ring-2 focus:ring-primary/50"
                        />
                        <Button
                            onClick={sendMessage}
                            disabled={!input.trim() || isLoading}
                            size="sm"
                        >
                            Send
                        </Button>
                    </div>
                    <div className="flex items-center gap-2 mt-2">
                        <Badge variant="outline" className="text-[10px]">
                            Phase 3
                        </Badge>
                        <span className="text-[10px] text-muted-foreground/40">
                            OpenAI GPT integration pending
                        </span>
                    </div>
                </div>
            </Card>
        </div>
    );
}
