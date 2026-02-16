import {
    Card,
    CardContent,
    CardDescription,
    CardHeader,
    CardTitle,
} from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";

export default function StudioPage() {
    const features = [
        { icon: "🎹", label: "Piano Roll", desc: "MIDI editing with velocity & CC lanes" },
        { icon: "📐", label: "Arrangement", desc: "Section-based timeline with drag & drop" },
        { icon: "🔊", label: "Synthesis", desc: "DawDreamer VST host + TorchFX GPU DSP" },
        { icon: "🎲", label: "AI Generation", desc: "YuE + Stable Audio for sound creation" },
    ];

    return (
        <div className="space-y-6">
            <div>
                <h1 className="text-2xl font-bold tracking-tight">
                    <span className="text-gradient">Studio</span>
                </h1>
                <p className="text-sm text-muted-foreground mt-1">
                    Arrangement, synthesis, and MIDI editing workspace.
                </p>
            </div>

            <div className="grid grid-cols-2 gap-4">
                {features.map((f) => (
                    <Card key={f.label} className="glass border-border/30 hover:glow-violet transition-all duration-300">
                        <CardHeader>
                            <CardTitle className="flex items-center gap-2 text-base">
                                <span className="text-xl">{f.icon}</span> {f.label}
                            </CardTitle>
                            <CardDescription className="text-xs">{f.desc}</CardDescription>
                        </CardHeader>
                        <CardContent>
                            <Badge variant="secondary" className="text-[10px]">
                                Phase 2
                            </Badge>
                        </CardContent>
                    </Card>
                ))}
            </div>

            <Card className="glass border-border/30">
                <CardContent className="flex flex-col items-center justify-center py-16 text-center">
                    <div className="text-5xl mb-4 opacity-30">🎹</div>
                    <p className="text-lg font-medium text-muted-foreground/60">
                        Coming in Phase 2
                    </p>
                    <p className="text-xs text-muted-foreground/40 mt-1 max-w-md">
                        The Studio will feature a full arrangement timeline, MIDI piano roll,
                        VST synthesis via DawDreamer, GPU-accelerated DSP with TorchFX,
                        and AI-powered sound generation.
                    </p>
                </CardContent>
            </Card>
        </div>
    );
}
