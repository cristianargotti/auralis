"""AURALIS Stem Decision Engine — Intelligent keep/correct/enhance/replace/mute.

Reads a GapReport and makes per-stem decisions:
  - KEEP      (80+)  : ref-targeted recipe only
  - CORRECT   (50-79): aggressive recipe + extra FX
  - ENHANCE   (25-49): layer generated support stem
  - REPLACE   (<25)  : mute original + generate replacement
  - MUTE      (any)  : remove harmful elements

The engine NEVER copies reference loops.  Instead it analyses the
reference DNA profile and generates *original* patterns using the
synth engine + MIDI generators, matched to the user track's BPM & key.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import structlog

from auralis.console.gap_analyzer import GapReport, StemGap
from auralis.ear.reference_bank import ReferenceBank

logger = structlog.get_logger()


# ── Decision Thresholds ────────────────────────────────

THRESHOLD_KEEP = 80.0
THRESHOLD_CORRECT = 50.0
THRESHOLD_ENHANCE = 25.0
# Below ENHANCE = REPLACE

# Stems that can be regenerated via synth
REGENERABLE_STEMS = {"bass", "drums", "other"}

# FX that can be intelligently added during CORRECT
SMART_FX = {"chorus", "sidechain", "reverb_boost", "saturation", "de_mud_eq"}


# ── Data Types ──────────────────────────────────────────


@dataclass
class StemDecision:
    """Decision for a single stem."""

    stem_name: str
    action: Literal["keep", "correct", "enhance", "replace", "mute"]
    quality_score: float
    reason: str
    # For ENHANCE/REPLACE: what to generate
    synth_patch: str | None = None
    pattern_style: str | None = None      # e.g. "syncopated", "four_on_floor"
    # FX intelligence
    extra_fx: list[str] = field(default_factory=list)
    # Volume override (-inf = mute)
    volume_adjust_db: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "stem_name": self.stem_name,
            "action": self.action,
            "quality_score": round(self.quality_score, 1),
            "reason": self.reason,
            "synth_patch": self.synth_patch,
            "pattern_style": self.pattern_style,
            "extra_fx": self.extra_fx,
            "volume_adjust_db": self.volume_adjust_db,
        }


@dataclass
class DecisionReport:
    """All stem decisions + metadata."""

    decisions: dict[str, StemDecision] = field(default_factory=dict)
    reference_count: int = 0
    bpm: float = 120.0
    key: str = "C"
    scale: str = "minor"

    def to_dict(self) -> dict[str, Any]:
        return {
            "decisions": {k: v.to_dict() for k, v in self.decisions.items()},
            "reference_count": self.reference_count,
            "bpm": self.bpm,
            "key": self.key,
            "scale": self.scale,
        }


# ── Smart FX Selection ──────────────────────────────────


def _select_smart_fx(stem_name: str, stem_gap: StemGap) -> list[str]:
    """Decide which extra FX to add based on gap analysis."""
    fx: list[str] = []

    # Check frequency gaps
    for fg in stem_gap.freq_gaps:
        # Muddy mids → cut
        if fg.band == "mid" and fg.gap_pct > 5.0 and stem_name != "bass":
            fx.append("de_mud_eq")
        # Missing highs → add presence/air
        if fg.band == "high" and fg.gap_pct < -5.0 and stem_name in ("vocals", "other"):
            fx.append("air_boost")

    # Stereo width — if "other" stem is narrow vs ref
    if stem_name == "other" and stem_gap.energy_gap_pct < -8.0:
        fx.append("chorus")

    # Dynamics — if too dynamic vs refs, add compression
    if stem_gap.dynamic_range_gap_db > 4.0:
        fx.append("compress_harder")

    # Pumping — drums benefit from sidechain feel
    if stem_name == "bass" and stem_gap.rms_gap_db < -4.0:
        fx.append("sidechain")

    # Thin sound → saturation for harmonic richness
    if stem_gap.rms_gap_db < -6.0 and stem_name in ("bass", "drums"):
        fx.append("saturation")

    # Space — if the reference has more reverb character
    if stem_name == "vocals" and stem_gap.energy_gap_pct < -5.0:
        fx.append("reverb_boost")

    return fx


# ── Pattern Style Selection ─────────────────────────────


def _select_pattern_style(stem_name: str, bpm: float) -> str:
    """Pick a generation style based on stem type and BPM."""
    if stem_name == "bass":
        if bpm >= 135:
            return "syncopated"
        elif bpm >= 115:
            return "walking"
        else:
            return "simple"
    elif stem_name == "drums":
        if bpm >= 135:
            return "four_on_floor"
        elif bpm >= 100:
            return "breakbeat"
        elif bpm >= 70:
            return "trap"
        else:
            return "minimal"
    else:  # "other"
        return "pad"


def _select_synth_patch(stem_name: str, bpm: float) -> str:
    """Pick a synth preset for generation — narrative-aware.

    Thinks about what the track NEEDS at this BPM:
      - Slow grooves need weight (808, warm pads)
      - Mid-tempo needs character (acid, keys, plucks)
      - High-energy needs tension (reese, dark pads, supersaw)
    """
    if stem_name == "bass":
        if bpm > 150:
            return "bass_reese"   # DnB energy — dark, evolving tension
        elif bpm >= 130:
            return "acid_303"     # Techno drive — squelchy, rhythmic
        else:
            return "bass_808"     # Deep groove — gravitational weight
    elif stem_name == "drums":
        return ""  # Drums use organic/AI generation, not synth presets
    elif stem_name == "other":
        if bpm > 140:
            return "pad_dark"     # High-energy = dark atmosphere, tension
        elif bpm >= 125:
            return "supersaw"     # Peak-time = big wide leads
        elif bpm >= 110:
            return "pluck"        # Mid-tempo = melodic identity
        else:
            return "pad_warm"     # Slow = emotional warmth
    return "pluck"


# ── Harmful Element Detection ───────────────────────────


def _should_mute(stem_name: str, stem_gap: StemGap) -> tuple[bool, str]:
    """Detect stems that actively hurt the mix."""
    # If energy is WAY above reference → it's dominating and problematic
    if stem_gap.energy_gap_pct > 20.0 and stem_gap.quality_score < 40.0:
        return True, f"Energy {stem_gap.energy_gap_pct:.0f}% above refs — dominating the mix"

    # Extreme frequency imbalance → probably noise/artifacts
    for fg in stem_gap.freq_gaps:
        if abs(fg.gap_pct) > 25.0 and stem_gap.quality_score < 30.0:
            return True, f"Extreme {fg.band} imbalance ({fg.gap_pct:+.0f}%) — likely artifacts"

    return False, ""


# ── Main Decision Engine ────────────────────────────────


def make_decisions(
    gap_report: GapReport,
    ear_analysis: dict[str, Any],
    bank: ReferenceBank | None = None,
    brain_report: Any | None = None,
) -> DecisionReport:
    """Analyse gap report and produce per-stem decisions.

    Args:
        gap_report: Full gap analysis from gap_analyzer
        ear_analysis: EAR stage output (BPM, key, LUFS, etc.)
        bank: Reference bank (for counting, not for copying stems)
        brain_report: Optional BrainReport from DNABrain.think()

    Returns:
        DecisionReport with one StemDecision per stem.
    """
    bpm = float(ear_analysis.get("bpm", 120.0))
    key = str(ear_analysis.get("key", "C"))
    scale = str(ear_analysis.get("scale", "minor"))

    # Extract brain stem plans if available
    brain_plans: dict[str, Any] = {}
    if brain_report is not None:
        brain_plans = getattr(brain_report, "stem_plans", {})
        if brain_plans:
            logger.info("stem_decisions.using_brain", stems=list(brain_plans.keys()))

    report = DecisionReport(
        reference_count=gap_report.reference_count,
        bpm=bpm,
        key=key,
        scale=scale,
    )

    if gap_report.reference_count == 0:
        logger.info("stem_decisions.no_references")
        return report

    for stem_name, stem_gap in gap_report.stem_gaps.items():
        score = stem_gap.quality_score
        brain_plan = brain_plans.get(stem_name)  # StemPlan or None

        # Phase 1: Check for harmful elements → MUTE
        should_mute, mute_reason = _should_mute(stem_name, stem_gap)
        if should_mute:
            report.decisions[stem_name] = StemDecision(
                stem_name=stem_name,
                action="mute",
                quality_score=score,
                reason=f"MUTE: {mute_reason}",
                volume_adjust_db=float("-inf"),
            )
            logger.info("stem_decision", stem=stem_name, action="mute", reason=mute_reason)
            continue

        # Helper: get patch/style/fx from brain or fallback
        def _get_patch() -> str:
            if brain_plan and getattr(brain_plan, "patch", ""):
                return brain_plan.patch
            return _select_synth_patch(stem_name, bpm)

        def _get_style() -> str:
            if brain_plan and getattr(brain_plan, "style", ""):
                return brain_plan.style
            return _select_pattern_style(stem_name, bpm)

        def _get_fx() -> list[str]:
            if brain_plan and getattr(brain_plan, "fx_chain", []):
                return list(brain_plan.fx_chain)
            return _select_smart_fx(stem_name, stem_gap)

        # Phase 2: Score-based decisions
        if score >= THRESHOLD_KEEP:
            # KEEP — just apply ref-targeted recipe
            report.decisions[stem_name] = StemDecision(
                stem_name=stem_name,
                action="keep",
                quality_score=score,
                reason=f"Quality {score:.0f}/100 — ref-targeted recipe only",
            )
            logger.info("stem_decision", stem=stem_name, action="keep", score=score)

        elif score >= THRESHOLD_CORRECT:
            # CORRECT — aggressive recipe + smart FX (brain-guided)
            extra_fx = _get_fx()
            brain_tag = " [🧠]" if brain_plan else ""
            report.decisions[stem_name] = StemDecision(
                stem_name=stem_name,
                action="correct",
                quality_score=score,
                reason=f"Quality {score:.0f}/100 — aggressive recipe + {', '.join(extra_fx) if extra_fx else 'standard FX'}{brain_tag}",
                extra_fx=extra_fx,
            )
            logger.info("stem_decision", stem=stem_name, action="correct", score=score, fx=extra_fx, brain=bool(brain_plan))

        elif score >= THRESHOLD_ENHANCE and stem_name in REGENERABLE_STEMS:
            # ENHANCE — keep original + layer synth support (brain-guided)
            patch = _get_patch()
            style = _get_style()
            brain_tag = " [🧠]" if brain_plan else ""
            report.decisions[stem_name] = StemDecision(
                stem_name=stem_name,
                action="enhance",
                quality_score=score,
                reason=f"Quality {score:.0f}/100 — layering {patch} ({style}) for support{brain_tag}",
                synth_patch=patch,
                pattern_style=style,
            )
            logger.info("stem_decision", stem=stem_name, action="enhance", patch=patch, style=style, brain=bool(brain_plan))

        elif stem_name in REGENERABLE_STEMS:
            # REPLACE — mute original, generate replacement (brain-guided)
            patch = _get_patch()
            style = _get_style()
            brain_tag = " [🧠]" if brain_plan else ""
            report.decisions[stem_name] = StemDecision(
                stem_name=stem_name,
                action="replace",
                quality_score=score,
                reason=f"Quality {score:.0f}/100 — replacing with generated {patch} ({style}){brain_tag}",
                synth_patch=patch,
                pattern_style=style,
                volume_adjust_db=float("-inf"),  # mute original
            )
            logger.info("stem_decision", stem=stem_name, action="replace", patch=patch, style=style, brain=bool(brain_plan))

        else:
            # Non-regenerable stem (vocals) with low score → heavy CORRECT
            extra_fx = _get_fx()
            brain_tag = " [🧠]" if brain_plan else ""
            report.decisions[stem_name] = StemDecision(
                stem_name=stem_name,
                action="correct",
                quality_score=score,
                reason=f"Quality {score:.0f}/100 — cannot regenerate {stem_name}, applying heavy correction{brain_tag}",
                extra_fx=extra_fx,
            )
            logger.info("stem_decision", stem=stem_name, action="correct_fallback", score=score, brain=bool(brain_plan))

    return report


# ── Log Formatting ──────────────────────────────────────


def format_decisions_for_logs(report: DecisionReport) -> list[str]:
    """Format DecisionReport as Engine Log lines."""
    lines: list[str] = []

    if not report.decisions:
        if report.reference_count == 0:
            lines.append("🧠 No stem decisions (reference bank empty — add reference tracks first)")
        else:
            lines.append(
                f"🧠 No stem decisions (refs={report.reference_count}, but no stem gaps found "
                "— check stem_analysis data)"
            )
        return lines

    lines.append("── STEM DECISIONS ──────────────────────")

    stem_emojis = {"drums": "🥁", "bass": "🎸", "vocals": "🎤", "other": "🎹"}
    action_labels = {
        "keep": "✅ KEEP",
        "correct": "🔧 CORRECT",
        "enhance": "⚡ ENHANCE",
        "replace": "🔄 REPLACE",
        "mute": "🔇 MUTE",
    }

    for stem_name, decision in report.decisions.items():
        emoji = stem_emojis.get(stem_name, "🎵")
        label = action_labels.get(decision.action, decision.action.upper())
        lines.append(
            f"  {emoji} {stem_name:8s} → {label:12s} ({decision.quality_score:.0f}/100)"
        )
        lines.append(f"     {decision.reason}")

        if decision.extra_fx:
            lines.append(f"     FX: {', '.join(decision.extra_fx)}")
        if decision.synth_patch:
            lines.append(
                f"     GEN: {decision.synth_patch} pattern={decision.pattern_style}"
            )
        lines.append("")

    return lines

