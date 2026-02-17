"""AURALIS Gap Analyzer — Compare your track against reference DNA averages.

Generates a human-readable Gap Report that tells the artist EXACTLY what's
missing in their track compared to the professional references in the bank.

Each gap comes with an actionable suggestion that the stem recipes can use
to automatically adjust processing (EQ, compression, volume, sends).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import structlog

from auralis.ear.reference_bank import ReferenceBank, StemFingerprint

logger = structlog.get_logger()


# ── Data Types ───────────────────────────────────────────


@dataclass
class FreqGap:
    """Gap in a single frequency band."""

    band: str          # "low", "mid", "high"
    ref_pct: float     # reference average %
    your_pct: float    # your track %
    gap_pct: float     # difference (negative = you're below ref)
    action: str        # "Boost +4dB at 60Hz"


@dataclass
class StemGap:
    """What's missing in one stem vs the reference average."""

    stem_name: str
    quality_score: float = 100.0         # 0-100
    rms_gap_db: float = 0.0             # negative = quieter than refs
    peak_gap_db: float = 0.0
    dynamic_range_gap_db: float = 0.0
    energy_gap_pct: float = 0.0
    freq_gaps: list[FreqGap] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GapReport:
    """Full gap analysis: your track vs professional references."""

    overall_score: float = 0.0          # 0-100
    reference_count: int = 0
    lufs_gap: float = 0.0              # negative = quieter than refs
    your_lufs: float = -14.0
    ref_lufs: float = -14.0
    stem_gaps: dict[str, StemGap] = field(default_factory=dict)
    top_improvements: list[str] = field(default_factory=list)
    summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        result = {
            "overall_score": round(self.overall_score, 1),
            "reference_count": self.reference_count,
            "lufs_gap": round(self.lufs_gap, 1),
            "your_lufs": round(self.your_lufs, 1),
            "ref_lufs": round(self.ref_lufs, 1),
            "stem_gaps": {k: v.to_dict() for k, v in self.stem_gaps.items()},
            "top_improvements": self.top_improvements,
            "summary": self.summary,
        }
        return result


# ── Frequency Band Helpers ───────────────────────────────

_BAND_LABELS = {
    "low": "Sub/Bass (20-250Hz)",
    "mid": "Mids (250-4kHz)",
    "high": "Highs/Air (4-20kHz)",
}

_BAND_EQ_TARGETS = {
    "low": {"drums": "50Hz", "bass": "80Hz", "vocals": "200Hz", "other": "150Hz"},
    "mid": {"drums": "1kHz", "bass": "500Hz", "vocals": "2kHz", "other": "1.5kHz"},
    "high": {"drums": "8kHz", "bass": "3kHz", "vocals": "10kHz", "other": "8kHz"},
}


# ── Gap Analysis Engine ─────────────────────────────────


def _analyze_stem_gap(
    stem_name: str,
    your_analysis: dict[str, Any],
    ref_avg: StemFingerprint,
) -> StemGap:
    """Compare a single stem against the reference average."""
    your_rms = your_analysis.get("rms_db", -20.0)
    your_peak = your_analysis.get("peak_db", -6.0)
    your_energy = your_analysis.get("energy_pct", 25.0)
    your_bands = your_analysis.get("freq_bands", {"low": 33.3, "mid": 33.3, "high": 33.3})
    your_dr = abs(your_peak - your_rms)

    rms_gap = your_rms - ref_avg.rms_db
    peak_gap = your_peak - ref_avg.peak_db
    dr_gap = your_dr - ref_avg.dynamic_range_db
    energy_gap = your_energy - ref_avg.energy_pct

    # Analyze frequency band gaps
    freq_gaps: list[FreqGap] = []
    for band_key in ("low", "mid", "high"):
        ref_pct = ref_avg.freq_bands.get(band_key, 33.3)
        your_pct = your_bands.get(band_key, 33.3)
        gap = your_pct - ref_pct

        eq_target = _BAND_EQ_TARGETS.get(band_key, {}).get(stem_name, "1kHz")

        if abs(gap) > 3.0:  # Only report significant gaps
            if gap < 0:
                action = f"Boost +{abs(gap):.0f}% at {eq_target}"
            else:
                action = f"Cut -{gap:.0f}% at {eq_target}"

            freq_gaps.append(FreqGap(
                band=band_key,
                ref_pct=round(ref_pct, 1),
                your_pct=round(your_pct, 1),
                gap_pct=round(gap, 1),
                action=action,
            ))

    # Generate suggestions
    suggestions: list[str] = []

    if rms_gap < -3.0:
        suggestions.append(f"Volume: boost {abs(rms_gap):.1f}dB (too quiet vs refs)")
    elif rms_gap > 3.0:
        suggestions.append(f"Volume: reduce {rms_gap:.1f}dB (too loud vs refs)")

    if dr_gap > 4.0:
        suggestions.append(f"Compression: dynamics {dr_gap:.1f}dB wider than refs → compress harder")
    elif dr_gap < -4.0:
        suggestions.append(f"Dynamics: {abs(dr_gap):.1f}dB over-compressed vs refs → ease off")

    for fg in freq_gaps:
        label = _BAND_LABELS.get(fg.band, fg.band)
        if fg.gap_pct < 0:
            suggestions.append(f"{label}: {abs(fg.gap_pct):.1f}% below refs → {fg.action}")
        else:
            suggestions.append(f"{label}: {fg.gap_pct:.1f}% above refs → {fg.action}")

    # Calculate quality score (100 = perfect match)
    penalties = 0.0
    penalties += min(abs(rms_gap) * 3, 20)      # Max 20 point penalty for volume
    penalties += min(abs(dr_gap) * 2, 15)        # Max 15 for dynamics
    for fg in freq_gaps:
        penalties += min(abs(fg.gap_pct) * 1.5, 10)  # Max 10 per band
    penalties += min(abs(energy_gap) * 1, 10)    # Max 10 for energy

    quality = max(0.0, 100.0 - penalties)

    return StemGap(
        stem_name=stem_name,
        quality_score=round(quality, 1),
        rms_gap_db=round(rms_gap, 1),
        peak_gap_db=round(peak_gap, 1),
        dynamic_range_gap_db=round(dr_gap, 1),
        energy_gap_pct=round(energy_gap, 1),
        freq_gaps=freq_gaps,
        suggestions=suggestions,
    )


def analyze_gaps(
    ear_analysis: dict[str, Any],
    stem_analysis: dict[str, dict[str, Any]],
    bank: ReferenceBank,
) -> GapReport:
    """Run full gap analysis: your track vs averaged references.

    Args:
        ear_analysis: Full EAR output (BPM, key, LUFS, sections, etc.)
        stem_analysis: Per-stem analysis from EAR stage
        bank: The reference bank with stored DNA entries

    Returns:
        GapReport with per-stem gaps + top improvements
    """
    ref_count = bank.count()
    if ref_count == 0:
        return GapReport(
            summary="No references in the bank yet. Add professional tracks first!",
        )

    stem_avgs = bank.get_stem_averages()
    master_avgs = bank.get_master_averages()

    # Your master stats
    your_lufs = float(ear_analysis.get("integrated_lufs", -14.0))
    ref_lufs = master_avgs.get("lufs", -14.0)
    lufs_gap = your_lufs - ref_lufs

    # Per-stem gap analysis
    stem_gaps: dict[str, StemGap] = {}
    for stem_name, sa in stem_analysis.items():
        if isinstance(sa, dict) and "error" not in sa:
            ref_fp = stem_avgs.get(stem_name)
            if ref_fp:
                stem_gaps[stem_name] = _analyze_stem_gap(stem_name, sa, ref_fp)

    # Overall quality score (weighted average of stem scores + master penalty)
    if stem_gaps:
        stem_scores = [sg.quality_score for sg in stem_gaps.values()]
        avg_stem_score = sum(stem_scores) / len(stem_scores)
    else:
        avg_stem_score = 50.0

    master_penalty = min(abs(lufs_gap) * 3, 20)
    overall = max(0.0, avg_stem_score - master_penalty)

    # Generate top improvements (sorted by impact)
    all_suggestions: list[tuple[float, str]] = []

    # Master loudness is always high impact
    if abs(lufs_gap) > 1.0:
        impact = abs(lufs_gap) * 5
        direction = "louder" if lufs_gap < 0 else "quieter"
        all_suggestions.append((
            impact,
            f"Master loudness: {abs(lufs_gap):.1f} LUFS {direction} than refs "
            f"(yours: {your_lufs:.1f}, refs: {ref_lufs:.1f})",
        ))

    for sg in stem_gaps.values():
        for suggestion in sg.suggestions:
            # Estimate impact from gap magnitude
            impact = 100.0 - sg.quality_score
            all_suggestions.append((impact, f"[{sg.stem_name}] {suggestion}"))

    all_suggestions.sort(key=lambda x: x[0], reverse=True)
    top_improvements = [s[1] for s in all_suggestions[:5]]

    # Summary
    stem_status = []
    for sn, sg in stem_gaps.items():
        emoji = "✓" if sg.quality_score >= 80 else "✗"
        stem_status.append(f"{emoji} {sn}: {sg.quality_score:.0f}/100")

    summary = (
        f"Overall: {overall:.0f}/100 vs {ref_count} references | "
        + " | ".join(stem_status)
    )

    return GapReport(
        overall_score=round(overall, 1),
        reference_count=ref_count,
        lufs_gap=round(lufs_gap, 1),
        your_lufs=round(your_lufs, 1),
        ref_lufs=round(ref_lufs, 1),
        stem_gaps=stem_gaps,
        top_improvements=top_improvements,
        summary=summary,
    )


# ── Log Formatting ───────────────────────────────────────


def format_gap_report_for_logs(report: GapReport) -> list[str]:
    """Format a GapReport as Engine Log lines.

    Returns a list of log strings to be used with _log() in the pipeline.
    """
    lines: list[str] = []

    if report.reference_count == 0:
        lines.append("📊 No references in bank — add pro tracks for gap analysis")
        return lines

    lines.append(
        f"📊 GAP ANALYSIS vs {report.reference_count} references "
        f"(Score: {report.overall_score:.0f}/100)"
    )
    lines.append("")

    # Per-stem gaps
    stem_emojis = {"drums": "🥁", "bass": "🎸", "vocals": "🎤", "other": "🎹"}
    for stem_name, sg in report.stem_gaps.items():
        emoji = stem_emojis.get(stem_name, "🎵")
        score_bar = "█" * int(sg.quality_score / 10) + "░" * (10 - int(sg.quality_score / 10))
        lines.append(f"  {emoji} {stem_name} [{score_bar}] {sg.quality_score:.0f}/100")

        for suggestion in sg.suggestions:
            lines.append(f"     → {suggestion}")

        if not sg.suggestions:
            lines.append("     ✓ Matches reference quality")

        lines.append("")

    # Master gap
    if abs(report.lufs_gap) > 0.5:
        direction = "↑" if report.lufs_gap < 0 else "↓"
        lines.append(
            f"  💎 Master: {report.your_lufs:.1f} LUFS "
            f"{direction} ref avg {report.ref_lufs:.1f} LUFS "
            f"(gap: {report.lufs_gap:+.1f})"
        )
        lines.append("")

    # Top improvements
    if report.top_improvements:
        lines.append("  🎯 Top improvements:")
        for i, imp in enumerate(report.top_improvements[:3], 1):
            lines.append(f"     {i}. {imp}")

    return lines
