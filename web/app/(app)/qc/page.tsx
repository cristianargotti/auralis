import {
    Card,
    CardContent,
    CardDescription,
    CardHeader,
    CardTitle,
} from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";

export default function QCPage() {
    const dimensions = [
        "Spectral Balance",
        "Loudness (LUFS)",
        "Stereo Width",
        "Dynamic Range",
        "Crest Factor",
        "Phase Coherence",
        "Bass Energy",
        "Presence Energy",
        "Air Band",
        "Beat Alignment",
        "MFCC Distance",
        "Key Match",
    ];

    return (
        <div className="space-y-6">
            <div>
                <h1 className="text-2xl font-bold tracking-tight">
                    <span className="text-gradient">QC Report</span>
                </h1>
                <p className="text-sm text-muted-foreground mt-1">
                    12-dimension quality scoring and A/B comparison.
                </p>
            </div>

            <Card className="glass border-border/30">
                <CardHeader>
                    <CardTitle className="text-sm flex items-center gap-2">
                        <span>📋</span> Quality Dimensions
                    </CardTitle>
                    <CardDescription className="text-xs">
                        Each dimension scored 0-100% against your reference
                    </CardDescription>
                </CardHeader>
                <CardContent>
                    <div className="grid grid-cols-4 gap-2">
                        {dimensions.map((dim) => (
                            <div
                                key={dim}
                                className="rounded-lg bg-secondary/30 px-3 py-2 text-center"
                            >
                                <p className="text-[10px] font-medium text-muted-foreground">
                                    {dim}
                                </p>
                            </div>
                        ))}
                    </div>
                </CardContent>
            </Card>

            <Card className="glass border-border/30">
                <CardContent className="flex flex-col items-center justify-center py-12 text-center">
                    <div className="text-5xl mb-4 opacity-30">🔍</div>
                    <p className="text-lg font-medium text-muted-foreground/60">
                        Coming in Phase 2
                    </p>
                    <p className="text-xs text-muted-foreground/40 mt-1 max-w-md">
                        Spectral overlay visualization, A/B comparison player with
                        waveform sync, and automated quality scoring against your
                        reference track.
                    </p>
                </CardContent>
            </Card>
        </div>
    );
}
