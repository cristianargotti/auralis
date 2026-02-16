import {
    Card,
    CardContent,
    CardDescription,
    CardHeader,
    CardTitle,
} from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";

export default function MixerPage() {
    const channels = [
        { icon: "🎤", label: "Vocals", color: "bg-auralis-magenta/20" },
        { icon: "🥁", label: "Drums", color: "bg-auralis-gold/20" },
        { icon: "🎸", label: "Bass", color: "bg-auralis-emerald/20" },
        { icon: "🎹", label: "Synths", color: "bg-auralis-cyan/20" },
        { icon: "🌊", label: "FX", color: "bg-auralis-violet/20" },
        { icon: "🎼", label: "Master", color: "bg-primary/10" },
    ];

    return (
        <div className="space-y-6">
            <div>
                <h1 className="text-2xl font-bold tracking-tight">
                    <span className="text-gradient">Mixer</span>
                </h1>
                <p className="text-sm text-muted-foreground mt-1">
                    Visual mixer with faders, sends, EQ, and effect chains.
                </p>
            </div>

            {/* Channel strip preview */}
            <div className="grid grid-cols-6 gap-3">
                {channels.map((ch) => (
                    <Card key={ch.label} className={`glass border-border/30 ${ch.color}`}>
                        <CardContent className="flex flex-col items-center justify-center py-8">
                            <span className="text-3xl mb-2">{ch.icon}</span>
                            <p className="text-xs font-medium">{ch.label}</p>
                            <div className="h-32 w-1 bg-border/30 rounded-full mt-3 relative">
                                <div className="absolute bottom-0 w-full h-2/3 bg-primary/40 rounded-full" />
                            </div>
                            <p className="text-[10px] text-muted-foreground mt-2">-3.2 dB</p>
                        </CardContent>
                    </Card>
                ))}
            </div>

            <Card className="glass border-border/30">
                <CardHeader>
                    <CardTitle className="text-sm flex items-center gap-2">
                        <span>🎛️</span> Mix Bus Architecture
                    </CardTitle>
                    <CardDescription className="text-xs">
                        9 buses: Kick, Percussion, Bass, Leads, Pads, FX, Vocals, Room, Master
                    </CardDescription>
                </CardHeader>
                <CardContent className="flex flex-col items-center justify-center py-8 text-center">
                    <Badge variant="secondary" className="text-[10px]">
                        Phase 2
                    </Badge>
                    <p className="text-xs text-muted-foreground/40 mt-3 max-w-md">
                        Full mixing console with Pedalboard FX chains, parametric EQ,
                        compression, sidechain, and a visual send/return matrix.
                    </p>
                </CardContent>
            </Card>
        </div>
    );
}
